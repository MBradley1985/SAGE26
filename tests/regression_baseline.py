#!/usr/bin/env python3
"""
SAGE26 regression baseline: capture and verify bit-identical outputs.

Phase 0 of the pre-release cleanup pass — see docs/developer/REGRESSION_BASELINE.md.

Two modes:
  capture <param_file>   Run sage against the .par file, hash every output, and
                         write tests/baseline/<param_stem>/manifest.json.
  verify  <param_file>   Re-run sage, recompute hashes, diff against the stored
                         manifest. Exits 1 on any drift; prints what changed.

Invariants enforced:
  - Serial sage build (sage binary contains no MPI symbols).
  - OutputDir is wiped before each run so no stale files contaminate the hash set.
  - Every model*.hdf5 in OutputDir is hashed at the file level and recursively at
    the dataset level. Per-dataset hashes localise drift when a file-level hash
    changes.

The manifest also stores build-environment fingerprints (compiler, GSL, HDF5,
platform, git commit). Verification warns loudly on a build-env mismatch — the
bit-identical guarantee only holds within a fixed toolchain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SAGE_BIN = REPO_ROOT / "sage"
BASELINE_DIR = REPO_ROOT / "tests" / "baseline"
MANIFEST_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Parameter-file parsing
# ---------------------------------------------------------------------------

PAR_KV_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\s+(\S+)")


def parse_par(par_path: Path) -> dict[str, str]:
    """Return {key: value} for every assignment line in a SAGE .par file."""
    params: dict[str, str] = {}
    for raw in par_path.read_text().splitlines():
        line = raw.split("%", 1)[0]
        line = line.split(";", 1)[0]
        m = PAR_KV_RE.match(line)
        if m:
            params[m.group(1)] = m.group(2)
    return params


def resolve_output_dir(par_path: Path) -> Path:
    params = parse_par(par_path)
    if "OutputDir" not in params:
        raise SystemExit(f"OutputDir not found in {par_path}")
    out = Path(params["OutputDir"])
    if not out.is_absolute():
        out = REPO_ROOT / out
    return out


# ---------------------------------------------------------------------------
# Build environment fingerprint
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def build_env() -> dict[str, str]:
    git_commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    git_tag = _run(["git", "-C", str(REPO_ROOT), "describe", "--tags", "--exact-match", "HEAD"])
    gsl = _run(["gsl-config", "--version"])
    cc = _run(["cc", "--version"]).splitlines()[0] if _run(["cc", "--version"]) else ""
    hdf5_runtime = f"{h5py.version.hdf5_version} (h5py {h5py.__version__})"
    return {
        "git_commit": git_commit,
        "git_tag": git_tag,
        "compiler": cc,
        "gsl": gsl,
        "hdf5_runtime": hdf5_runtime,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Serial-build assertion
# ---------------------------------------------------------------------------

def assert_serial_sage() -> None:
    if not SAGE_BIN.exists():
        raise SystemExit(f"sage binary not found at {SAGE_BIN}. Build it first with: make USE-MPI=")
    if sys.platform == "darwin":
        ldd = _run(["otool", "-L", str(SAGE_BIN)])
    else:
        ldd = _run(["ldd", str(SAGE_BIN)])
    if "libmpi" in ldd.lower() or "mpi.so" in ldd.lower():
        raise SystemExit(
            f"sage at {SAGE_BIN} is linked against MPI. The regression baseline requires "
            "a serial build. Rebuild with: make clean && make USE-MPI="
        )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_hdf5_datasets(path: Path) -> dict[str, dict]:
    """Return {dataset_path: {sha256, shape, dtype, nbytes}} for every dataset."""
    out: dict[str, dict] = {}

    def visit(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            arr = np.ascontiguousarray(obj[...])
            out[name] = {
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
            }

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


# ---------------------------------------------------------------------------
# Smoke statistics
# ---------------------------------------------------------------------------

SMOKE_FIELDS = ("StellarMass", "BlackHoleMass", "ColdGas", "HotGas", "Type")


def smoke_stats(path: Path) -> dict:
    """Cheap per-snapshot summary that is human-readable when a baseline drifts."""
    stats: dict = {}
    with h5py.File(path, "r") as f:
        snap_groups = sorted(k for k in f.keys() if k.startswith("Snap_"))
        for snap in snap_groups:
            g = f[snap]
            entry: dict = {}
            n = None
            for field in SMOKE_FIELDS:
                if field not in g:
                    continue
                arr = np.asarray(g[field][...])
                if n is None:
                    n = int(arr.shape[0])
                if field == "Type":
                    counts = np.bincount(arr.astype(np.int64), minlength=3)[:3].tolist()
                    entry["type_counts"] = counts
                else:
                    finite = arr[np.isfinite(arr)]
                    nonzero = finite[finite > 0]
                    if nonzero.size:
                        entry[f"{field}_log10_min"] = float(np.log10(nonzero.min()))
                        entry[f"{field}_log10_max"] = float(np.log10(nonzero.max()))
                        entry[f"{field}_log10_mean"] = float(np.log10(nonzero).mean())
                    entry[f"{field}_sum"] = float(finite.sum())
            if n is not None:
                entry["n_galaxies"] = n
            if entry:
                stats[snap] = entry
    return stats


# ---------------------------------------------------------------------------
# Capture + verify
# ---------------------------------------------------------------------------

def run_sage(par_path: Path, output_dir: Path) -> float:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = subprocess.run(
        [str(SAGE_BIN), str(par_path)],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        sys.stderr.write(result.stdout.decode("utf-8", errors="replace"))
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(f"sage failed (rc={result.returncode}) for {par_path}")
    return elapsed


def gather(par_path: Path) -> dict:
    output_dir = resolve_output_dir(par_path)
    elapsed = run_sage(par_path, output_dir)
    h5_files = sorted(p for p in output_dir.glob("model*.hdf5"))
    if not h5_files:
        raise SystemExit(f"No model*.hdf5 produced in {output_dir}")

    files: dict[str, dict] = {}
    for p in h5_files:
        files[p.name] = {
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "datasets": hash_hdf5_datasets(p),
            "smoke": smoke_stats(p),
        }

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "par_file": str(par_path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed, 2),
        "build_env": build_env(),
        "files": files,
    }


def manifest_path_for(par_path: Path) -> Path:
    return BASELINE_DIR / par_path.stem / "manifest.json"


def capture(par_path: Path) -> int:
    assert_serial_sage()
    print(f"[capture] running sage on {par_path}")
    manifest = gather(par_path)
    mpath = manifest_path_for(par_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[capture] wrote {mpath} ({manifest['runtime_seconds']}s sage runtime)")
    for fname, meta in manifest["files"].items():
        print(f"  {fname}: {meta['size_bytes']:>10} bytes  {meta['sha256'][:16]}…  "
              f"{len(meta['datasets'])} datasets")
    return 0


def diff_manifests(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """Return (failures, warnings).

    Failures: missing/extra files, missing/extra datasets, per-dataset hash drift.
    Warnings: file-level hash drift with no dataset drift (HDF5 metadata-only
    differences — internal B-tree layout, chunk allocation order, etc. — are
    cosmetic and don't indicate a scientific output change).
    """
    failures: list[str] = []
    warnings: list[str] = []
    old_files = set(old["files"])
    new_files = set(new["files"])
    for missing in sorted(old_files - new_files):
        failures.append(f"  missing output file: {missing}")
    for extra in sorted(new_files - old_files):
        failures.append(f"  unexpected output file: {extra}")
    for fname in sorted(old_files & new_files):
        ofile = old["files"][fname]
        nfile = new["files"][fname]
        old_ds = ofile["datasets"]
        new_ds = nfile["datasets"]
        dataset_issues: list[str] = []
        for name in sorted(set(old_ds) | set(new_ds)):
            if name not in old_ds:
                dataset_issues.append(f"    + dataset added: {name}")
            elif name not in new_ds:
                dataset_issues.append(f"    - dataset removed: {name}")
            elif old_ds[name]["sha256"] != new_ds[name]["sha256"]:
                dataset_issues.append(
                    f"    ~ dataset drift: {name} "
                    f"(shape {old_ds[name]['shape']} → {new_ds[name]['shape']})"
                )
        if dataset_issues:
            failures.append(f"  {fname}: {len(dataset_issues)} dataset(s) drifted")
            failures.extend(dataset_issues)
        elif ofile["sha256"] != nfile["sha256"]:
            warnings.append(
                f"  {fname}: HDF5 container bytes differ but every dataset is bit-identical "
                f"(file sha {ofile['sha256'][:12]}… → {nfile['sha256'][:12]}…)"
            )
    return failures, warnings


def verify(par_path: Path) -> int:
    assert_serial_sage()
    mpath = manifest_path_for(par_path)
    if not mpath.exists():
        raise SystemExit(f"No baseline manifest at {mpath}. Run: capture {par_path}")
    old = json.loads(mpath.read_text())

    new = gather(par_path)

    env_diff = []
    for key in ("compiler", "gsl", "hdf5_runtime", "platform"):
        if old["build_env"].get(key) != new["build_env"].get(key):
            env_diff.append(f"  {key}: {old['build_env'].get(key)!r} → {new['build_env'].get(key)!r}")
    if env_diff:
        sys.stderr.write("WARNING: build environment differs from baseline:\n")
        sys.stderr.write("\n".join(env_diff) + "\n")
        sys.stderr.write("Bit-identical guarantee only holds within a fixed toolchain.\n\n")

    failures, warnings = diff_manifests(old, new)
    n_files = len(new["files"])
    n_datasets = sum(len(f["datasets"]) for f in new["files"].values())
    if warnings:
        print(f"[verify] NOTE — {par_path}: {len(warnings)} HDF5-metadata-only differences:")
        print("\n".join(warnings))
    if failures:
        print(f"[verify] FAIL — {par_path} drifted from baseline:")
        print("\n".join(failures))
        return 1
    print(f"[verify] PASS — {par_path} ({n_files} files, {n_datasets} datasets bit-identical)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("capture", "verify"))
    p.add_argument("par_file", type=Path, help="path to a SAGE .par file")
    args = p.parse_args(argv)
    par_path = args.par_file.resolve()
    if not par_path.exists():
        raise SystemExit(f"parameter file not found: {par_path}")
    if args.mode == "capture":
        return capture(par_path)
    return verify(par_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

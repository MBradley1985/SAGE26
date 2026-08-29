#!/usr/bin/env python3
"""Build, run, and plot SAGE for a given simulation.

Usage:
    python3 fullsage.py millennium
    python3 fullsage.py mini_millennium
    python3 fullsage.py millennium --skip-build   # skip make clean && make
    python3 fullsage.py millennium --skip-plot    # skip the plotting step
"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Build, run, and plot SAGE.")
    parser.add_argument("sim", help="Simulation name, e.g. millennium or mini_millennium")
    parser.add_argument("--skip-build", action="store_true", help="Skip make clean && make")
    parser.add_argument("--skip-plot", action="store_true", help="Skip the plotting step")
    args = parser.parse_args()

    if not args.skip_build:
        run(["make", "clean"])
        run(["make"])

    run(["./sage", f"input/{args.sim}.par"])

    if not args.skip_plot:
        # Pass the glob pattern as a literal string, same as quoting it in the shell —
        # the plotting script expands it internally.
        run([
            "python3",
            "plotting/allresults-blackholes.py",
            "--i", f"./output/{args.sim}/model_*.hdf5",
        ])

        run([
            "python3",
            "plotting/bh_lrd_analysis.py",
            "-i", f"./output/{args.sim}/model_*.hdf5",
            "-s", "27",
        ])

        run([
                    "python3",
                    "plotting/bh_lrd_analysis_multiz.py",
                    "-i", f"./output/{args.sim}/model_*.hdf5",
                    #"-s", "27",
                ])

        

    print("\nDone.")


if __name__ == "__main__":
    main()
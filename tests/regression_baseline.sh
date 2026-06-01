#!/usr/bin/env bash
# SAGE26 regression baseline driver.
#
# Default mode: verify every golden .par file matches its stored manifest
# bit-for-bit. Pass --capture to re-capture all manifests instead (use after a
# labelled bug fix; the new hashes replace the old ones).
#
# Exit codes:
#   0  every golden config matched its baseline
#   1  one or more configs drifted
#   2  setup / build problem (sage missing, MPI linked, etc.)
#
# See docs/developer/REGRESSION_BASELINE.md for the policy.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="verify"
if [ "${1:-}" = "--capture" ]; then
    MODE="capture"
    shift
elif [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# Golden parameter files for the per-conversation regression sweep.
#
# Kept deliberately small: every cleanup commit re-runs every config here, and
# anything slower than ~30s ends up skipped in practice. Add slow configs as
# release-time spot checks, not as the default sweep.
#
# Currently:
#   - input/millennium.par   ~11s, 1.5 GB output, exercises classical cooling/SF/feedback.
#
# Available but not in the default sweep (too slow for per-commit use):
#   - input/microuchuu.par   ~255s, 17 GB output. Exercises CGM + FFB + FIRE on
#     Uchuu100. Capture/verify manually before tagging a release:
#       python3 tests/regression_baseline.py capture input/microuchuu.par
#       python3 tests/regression_baseline.py verify  input/microuchuu.par
GOLDEN=(
    "input/millennium.par"
)

if [ ! -x ./sage ]; then
    echo "ERROR: ./sage not found or not executable. Build it first with:" >&2
    echo "  make clean && make USE-MPI=" >&2
    exit 2
fi

PASS=0
FAIL=0
FAILED_CONFIGS=()

for par in "${GOLDEN[@]}"; do
    if [ ! -f "$par" ]; then
        echo "ERROR: golden config $par not found" >&2
        FAIL=$((FAIL + 1))
        FAILED_CONFIGS+=("$par (missing)")
        continue
    fi
    echo "─── ${MODE}: $par ───"
    if python3 tests/regression_baseline.py "$MODE" "$par"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_CONFIGS+=("$par")
    fi
    echo
done

echo "═══ regression baseline ${MODE} summary ═══"
echo "  passed: $PASS"
echo "  failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "  failed configs:"
    for c in "${FAILED_CONFIGS[@]}"; do
        echo "    - $c"
    done
    exit 1
fi
exit 0

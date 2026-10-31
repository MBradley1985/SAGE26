#!/bin/bash
# DARKSAGEMODE INTEGRATION CHECKS
#
# Validates, on the Millennium trees:
#   1. Pinned radius: built with -DDARKSAGE_PIN_DISK_RADIUS (DiskScaleRadius kept at SAGE26's
#      value), DarkSAGEMode 1 gives output identical to DarkSAGEMode 0 in every field the two
#      share -- the annuli only resolve SAGE26's bulk evolution.
#   2. Annulus sums: in DarkSAGEMode 1 and 2 (normal build) the annuli sum to the bulk fields
#      (gas, gas metals, disc stars, disc star metals, disc SFR); prints the worst mismatch.
#
# Run from the SAGE26 root: bash tests/darksage_mode_checks.sh   (PYTHON=... to pick python)
set -e
ROOT=$(pwd)
PYTHON=${PYTHON:-python3}
PAR=input/millennium_darkSAGE.par
W=$ROOT/tests/test_output/darksage
rm -rf "$W" && mkdir -p "$W/pinned_build"

run_mode() {   # binary mode outdir
    mkdir -p "$3"
    sed -e "s|^OutputDir .*|OutputDir          $3/|" -e "s|^DarkSAGEMode .*|DarkSAGEMode                $2|" $PAR > "$3.par"
    mpirun -np 8 "$1" "$3.par" > "$3.log" 2>&1
}

echo "--- 1. pinned radius: mode 1 against mode 0"
rsync -a --exclude '*.o' src Makefile "$W/pinned_build/"
(cd "$W/pinned_build" && make -j8 CC="mpicc -DDARKSAGE_PIN_DISK_RADIUS" > build.log 2>&1)
run_mode "$W/pinned_build/sage" 0 "$W/pinned_mode0"
run_mode "$W/pinned_build/sage" 1 "$W/pinned_mode1"
$PYTHON tests/darksage_compare_runs.py "$W/pinned_mode0" "$W/pinned_mode1" 63 50 40 30 20 | tail -1

echo "--- 2. annulus sums in modes 1 and 2"
for m in 1 2; do
    run_mode "$ROOT/sage" $m "$W/mode$m"
    $PYTHON tests/darksage_annuli_sums.py "$W/mode$m" 63 40 20
done

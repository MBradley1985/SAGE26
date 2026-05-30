#!/bin/bash
# Launches one SAGE26 instance per microuchuu .par file in parallel.
# Each process logs to logs/<parname>.log. The script blocks until all finish.
# On macOS the OS scheduler spreads CPU-bound processes across cores automatically.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SAGE_BIN="./sage"
LOG_DIR="logs"

if [[ ! -x "$SAGE_BIN" ]]; then
    echo "Error: SAGE binary not found at $SAGE_BIN" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

PAR_FILES=(input/microuchuu*.par)

if [[ ! -e "${PAR_FILES[0]}" ]]; then
    echo "No microuchuu*.par files found in input/" >&2
    exit 1
fi

echo "Cleaning output directories..."
echo ""
for par in "${PAR_FILES[@]}"; do
    output_dir="$(grep -m1 '^OutputDir' "$par" | awk '{print $2}')"
    if [[ -z "$output_dir" ]]; then
        echo "  Warning: could not parse OutputDir from $par, skipping cleanup" >&2
        continue
    fi
    model_files=("$output_dir"/model*.hdf5)
    if [[ -e "${model_files[0]}" ]]; then
        echo "  Removing ${#model_files[@]} model file(s) from $output_dir/"
        rm -f "${model_files[@]}"
    else
        echo "  Clean: $output_dir/"
    fi
done
echo ""

echo "Launching ${#PAR_FILES[@]} SAGE26 instances..."
echo ""

pids=()
logs=()
for par in "${PAR_FILES[@]}"; do
    name="$(basename "$par" .par)"
    log="$LOG_DIR/${name}.log"
    echo "  Starting: $par  ->  $log"
    "$SAGE_BIN" "$par" > "$log" 2>&1 &
    pids+=("$!")
    logs+=("$log")
done

echo ""
echo "All instances launched (PIDs: ${pids[*]})"
echo "Waiting for completion..."
echo ""

failed=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    par="${PAR_FILES[$i]}"
    if wait "$pid"; then
        echo "  [OK]   $(basename "$par")"
    else
        status=$?
        echo "  [FAIL] $(basename "$par")  (exit $status) -- see ${logs[$i]}"
        failed=$((failed + 1))
    fi
done

echo ""
if [[ $failed -eq 0 ]]; then
    echo "All ${#PAR_FILES[@]} runs completed successfully."
else
    echo "$failed / ${#PAR_FILES[@]} runs failed. Check logs in $LOG_DIR/"
    exit 1
fi

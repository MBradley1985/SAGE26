#!/bin/bash
#
# SAGE26 Regression Benchmark
#
# Runs SAGE on a single tree file and compares output byte-for-byte
# against a committed reference, catching any unintended physics changes
# during refactoring or cleanup.
#
# Usage (run from the project root):
#   bash tests/run_benchmark_test.sh generate   # first time: create reference
#   bash tests/run_benchmark_test.sh verify     # after each change: check output
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
BENCHMARK_DIR="$SCRIPT_DIR/benchmark"
BENCHMARK_PAR="$BENCHMARK_DIR/benchmark.par"
REFERENCE_FILE="$BENCHMARK_DIR/reference.sha256"
OUTPUT_DIR="$BENCHMARK_DIR/output"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODE="${1:-}"

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SAGE26 REGRESSION BENCHMARK${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

if [ -z "$MODE" ] || [ "$MODE" != "generate" ] && [ "$MODE" != "verify" ]; then
    echo "Usage: bash tests/run_benchmark_test.sh <generate|verify>"
    echo ""
    echo "  generate  Run SAGE and save output checksums as the reference."
    echo "            Do this once from the known-good committed state."
    echo ""
    echo "  verify    Run SAGE and compare output against the saved reference."
    echo "            Do this after every refactoring change."
    echo ""
    exit 1
fi

# ---- Preflight checks ----

if [ ! -f "$PROJECT_ROOT/sage" ]; then
    echo -e "${RED}Error: sage executable not found. Run 'make' first.${NC}"
    exit 1
fi

if [ ! -f "$BENCHMARK_PAR" ]; then
    echo -e "${RED}Error: $BENCHMARK_PAR not found.${NC}"
    exit 1
fi

if [ "$MODE" = "verify" ] && [ ! -f "$REFERENCE_FILE" ]; then
    echo -e "${RED}Error: No reference checksums found at:${NC}"
    echo "  $REFERENCE_FILE"
    echo ""
    echo -e "Run '${YELLOW}bash tests/run_benchmark_test.sh generate${NC}' first."
    exit 1
fi

# ---- Run SAGE ----

echo -e "${YELLOW}▸ Running SAGE benchmark (single tree file)...${NC}"
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/benchmark_*

START=$(date +%s)

cd "$PROJECT_ROOT"
mpirun -n 1 ./sage "$BENCHMARK_PAR" > "$BENCHMARK_DIR/sage.log" 2>&1
EXIT_CODE=$?

END=$(date +%s)
ELAPSED=$((END - START))

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "${RED}  ✗ SAGE run failed (exit code $EXIT_CODE)${NC}"
    echo "  See: $BENCHMARK_DIR/sage.log"
    exit 1
fi

# Count output files
NFILES=$(ls "$OUTPUT_DIR"/benchmark_* 2>/dev/null | wc -l | tr -d ' ')
if [ "$NFILES" -eq 0 ]; then
    echo -e "${RED}  ✗ No output files produced${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ SAGE completed in ${ELAPSED}s — ${NFILES} output files${NC}"
echo ""

# ---- Compute checksums ----

echo -e "${YELLOW}▸ Computing SHA256 checksums...${NC}"
cd "$OUTPUT_DIR"
CHECKSUMS=$(shasum -a 256 benchmark_* | sort)
cd "$PROJECT_ROOT"

NCHECK=$(echo "$CHECKSUMS" | wc -l | tr -d ' ')
echo -e "${GREEN}  ✓ ${NCHECK} checksums computed${NC}"
echo ""

# ---- Generate or Verify ----

if [ "$MODE" = "generate" ]; then

    echo -e "${YELLOW}▸ Saving reference checksums...${NC}"
    echo "# SAGE26 benchmark reference checksums" > "$REFERENCE_FILE"
    echo "# Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$REFERENCE_FILE"
    echo "# Parameters: tests/benchmark/benchmark.par" >> "$REFERENCE_FILE"
    echo "# Files: ${NFILES} binary output files (single tree file, all snapshots)" >> "$REFERENCE_FILE"
    echo "$CHECKSUMS" >> "$REFERENCE_FILE"

    echo -e "${GREEN}  ✓ Reference saved to: tests/benchmark/reference.sha256${NC}"
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ BENCHMARK REFERENCE GENERATED${NC}"
    echo ""
    echo "  Commit tests/benchmark/reference.sha256 to lock in this state."
    echo "  Run 'bash tests/run_benchmark_test.sh verify' after each change."
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

else  # verify

    echo -e "${YELLOW}▸ Comparing against reference...${NC}"

    # Build comparable checksum string from reference (strip comment lines, sort)
    REFERENCE=$(grep -v '^#' "$REFERENCE_FILE" | sort)

    if [ "$CHECKSUMS" = "$REFERENCE" ]; then
        echo -e "${GREEN}  ✓ All ${NCHECK} checksums match${NC}"
        echo ""
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ BENCHMARK PASSED — no physics changes detected${NC}"
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        echo ""
        exit 0
    else
        echo -e "${RED}  ✗ Checksum mismatch detected${NC}"
        echo ""

        # Show which files differ
        echo -e "${YELLOW}  Changed files:${NC}"
        while IFS= read -r ref_line; do
            [ -z "$ref_line" ] && continue
            ref_hash=$(echo "$ref_line" | awk '{print $1}')
            ref_file=$(echo "$ref_line" | awk '{print $2}')
            cur_line=$(echo "$CHECKSUMS" | grep " $ref_file$" || true)
            if [ -z "$cur_line" ]; then
                echo -e "  ${RED}  MISSING: $ref_file${NC}"
            elif [ "$cur_line" != "$ref_line" ]; then
                echo -e "  ${RED}  CHANGED: $ref_file${NC}"
            fi
        done <<< "$REFERENCE"

        # Check for new files not in reference
        while IFS= read -r cur_line; do
            [ -z "$cur_line" ] && continue
            cur_file=$(echo "$cur_line" | awk '{print $2}')
            if ! echo "$REFERENCE" | grep -q " $cur_file$"; then
                echo -e "  ${YELLOW}  NEW:     $cur_file${NC}"
            fi
        done <<< "$CHECKSUMS"

        echo ""
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  ✗ BENCHMARK FAILED — physics output has changed${NC}"
        echo ""
        echo "  If this change is intentional, regenerate the reference:"
        echo "    bash tests/run_benchmark_test.sh generate"
        echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
        echo ""
        exit 1
    fi

fi

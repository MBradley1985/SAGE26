#!/bin/bash
#
# Integration Test Runner
#
# This script runs the SAGE model with test parameters and validates outputs
# against expected behaviors.
#

set -e  # Exit on error

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  SAGE26 MODEL INTEGRATION TESTS${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if SAGE executable exists
if [ ! -f "$PROJECT_ROOT/sage" ]; then
    echo -e "${YELLOW}⚠ SAGE executable not found. Compile first with:${NC}"
    echo "   cd $PROJECT_ROOT && make"
    echo ""
fi

# Create test output directory
TEST_OUTPUT_DIR="$SCRIPT_DIR/test_output"
mkdir -p "$TEST_OUTPUT_DIR"

# Change to tests directory
cd "$SCRIPT_DIR"

echo -e "${YELLOW}▸ Test 1: Building all test suites${NC}"
make clean > /dev/null 2>&1
if make all > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ PASS - All test suites compiled successfully${NC}"
else
    echo -e "${RED}  ✗ FAIL - Build failed${NC}"
    exit 1
fi

echo -e "${YELLOW}▸ Test 2: Conservation Laws (37 tests)${NC}"
if ./test_build/test_conservation > "$TEST_OUTPUT_DIR/conservation.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/conservation.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some conservation tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/conservation.log"
fi

echo -e "${YELLOW}▸ Test 3: Regime Determination & CGM Physics (21 tests)${NC}"
if ./test_build/test_regime_cgm > "$TEST_OUTPUT_DIR/regime_cgm.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/regime_cgm.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some regime/CGM tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/regime_cgm.log"
fi

echo -e "${YELLOW}▸ Test 4: Bulge Size Physics (13 tests)${NC}"
if ./test_build/test_bulge_sizes > "$TEST_OUTPUT_DIR/bulge_sizes.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/bulge_sizes.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some bulge size tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/bulge_sizes.log"
fi

echo -e "${YELLOW}▸ Test 5: Physics Validation (31 tests)${NC}"
if ./test_build/test_physics_validation > "$TEST_OUTPUT_DIR/physics_validation.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/physics_validation.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some physics validation tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/physics_validation.log"
fi

echo -e "${YELLOW}▸ Test 6: Galaxy Mergers (10 tests)${NC}"
if ./test_build/test_mergers > "$TEST_OUTPUT_DIR/mergers.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/mergers.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some merger tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/mergers.log"
fi

echo -e "${YELLOW}▸ Test 7: Disk Instability (7 tests)${NC}"
if ./test_build/test_disk_instability > "$TEST_OUTPUT_DIR/disk_instability.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/disk_instability.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some disk instability tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/disk_instability.log"
fi

echo -e "${YELLOW}▸ Test 8: Gas Infall (7 tests)${NC}"
if ./test_build/test_infall > "$TEST_OUTPUT_DIR/infall.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/infall.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some infall tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/infall.log"
fi

echo -e "${YELLOW}▸ Test 9: Numerical Stability (7 tests)${NC}"
if ./test_build/test_numerical_stability > "$TEST_OUTPUT_DIR/numerical_stability.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/numerical_stability.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some numerical stability tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/numerical_stability.log"
fi

# Count total passes and fails from individual test outputs
TOTAL_TESTS=140
PASSED=0
FAILED=0

# Sum up results from each log file
for log in "$TEST_OUTPUT_DIR"/*.log; do
    if [ -f "$log" ]; then
        # Look for the summary line specifically "  Passed:       XX (YY%)"
        # For Passed: take 2nd number (actual count), for Failed: take 1st number
        P=$(grep "  Passed:" "$log" | grep -oE '[0-9]+' | sed -n '2p')
        F=$(grep "  Failed:" "$log" | grep -oE '[0-9]+' | sed -n '1p')
        if [ -n "$P" ]; then
            PASSED=$((PASSED + P))
        fi
        if [ -n "$F" ]; then
            FAILED=$((FAILED + F))
        fi
    fi
done

# Calculate percentages using bc or python
if command -v bc &> /dev/null; then
    PASS_PCT=$(echo "scale=1; ($PASSED * 100) / $TOTAL_TESTS" | bc)
    FAIL_PCT=$(echo "scale=1; ($FAILED * 100) / $TOTAL_TESTS" | bc)
else
    PASS_PCT=$(python3 -c "print(f'{($PASSED/$TOTAL_TESTS)*100:.1f}')")
    FAIL_PCT=$(python3 -c "print(f'{($FAILED/$TOTAL_TESTS)*100:.1f}')")
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  TEST SUMMARY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "  Total tests:  $TOTAL_TESTS"
echo -e "  Passed:       $PASSED (${PASS_PCT}%)"
echo -e "  Failed:       $FAILED (${FAIL_PCT}%)"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}  ✓ ALL INTEGRATION TESTS PASSED${NC}"
    echo ""
    echo -e "Logs saved to: ${TEST_OUTPUT_DIR}"
    exit 0
else
    echo -e "${RED}  ✗ SOME TESTS FAILED${NC}"
    echo ""
    echo -e "Review logs in: ${TEST_OUTPUT_DIR}"
    exit 1
fi

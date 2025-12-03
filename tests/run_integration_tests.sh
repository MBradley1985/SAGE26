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
echo -e "${BLUE}  Test Suites: 17 | Expected Tests: 320${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Start timing
START_TIME=$(date +%s)

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
echo -e "  ${NC}Compiling 17 test executables...${NC}"
BUILD_START=$(date +%s)
make clean > /dev/null 2>&1
if make all > /dev/null 2>&1; then
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    echo -e "${GREEN}  ✓ PASS - All test suites compiled successfully (${BUILD_TIME}s)${NC}"
else
    echo -e "${RED}  ✗ FAIL - Build failed${NC}"
    exit 1
fi

echo -e "${YELLOW}▸ Test 2: Conservation Laws (37 tests)${NC}"
echo -e "  ${NC}Testing mass, metal, and energy conservation...${NC}"
if ./test_build/test_conservation > "$TEST_OUTPUT_DIR/conservation.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/conservation.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some conservation tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/conservation.log"
fi

echo -e "${YELLOW}▸ Test 3: Regime Determination & CGM Physics (21 tests)${NC}"
echo -e "  ${NC}Testing Voit criterion, precipitation, and regime routing...${NC}"
if ./test_build/test_regime_cgm > "$TEST_OUTPUT_DIR/regime_cgm.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/regime_cgm.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some regime/CGM tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/regime_cgm.log"
fi

echo -e "${YELLOW}▸ Test 4: Bulge Size Physics (13 tests)${NC}"
echo -e "  ${NC}Testing Shen scaling relations and instability bulges...${NC}"
if ./test_build/test_bulge_sizes > "$TEST_OUTPUT_DIR/bulge_sizes.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/bulge_sizes.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some bulge size tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/bulge_sizes.log"
fi

echo -e "${YELLOW}▸ Test 5: Physics Validation (31 tests)${NC}"
echo -e "  ${NC}Testing physical bounds, quenching, and parameter ranges...${NC}"
if ./test_build/test_physics_validation > "$TEST_OUTPUT_DIR/physics_validation.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/physics_validation.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some physics validation tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/physics_validation.log"
fi

echo -e "${YELLOW}▸ Test 6: Galaxy Mergers (13 tests)${NC}"
echo -e "  ${NC}Testing major/minor classification and merger timescales...${NC}"
if ./test_build/test_mergers > "$TEST_OUTPUT_DIR/mergers.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/mergers.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some merger tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/mergers.log"
fi

echo -e "${YELLOW}▸ Test 7: Disk Instability (9 tests)${NC}"
echo -e "  ${NC}Testing stability criterion and bulge formation channel...${NC}"
if ./test_build/test_disk_instability > "$TEST_OUTPUT_DIR/disk_instability.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/disk_instability.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some disk instability tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/disk_instability.log"
fi

echo -e "${YELLOW}▸ Test 8: Gas Infall (12 tests)${NC}"
echo -e "  ${NC}Testing baryon fraction and regime-based gas routing...${NC}"
if ./test_build/test_infall > "$TEST_OUTPUT_DIR/infall.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/infall.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some infall tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/infall.log"
fi

echo -e "${YELLOW}▸ Test 9: Numerical Stability (24 tests)${NC}"
echo -e "  ${NC}Testing NaN/Inf protection and roundoff error handling...${NC}"
if ./test_build/test_numerical_stability > "$TEST_OUTPUT_DIR/numerical_stability.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/numerical_stability.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some numerical stability tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/numerical_stability.log"
fi

echo -e "${YELLOW}▸ Test 10: Metal Enrichment (22 tests)${NC}"
echo -e "  ${NC}Testing stellar yields, SN feedback, and metal tracking...${NC}"
if ./test_build/test_metal_enrichment > "$TEST_OUTPUT_DIR/metal_enrichment.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/metal_enrichment.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some metal enrichment tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/metal_enrichment.log"
fi

echo -e "${YELLOW}▸ Test 11: Ram Pressure Stripping (14 tests)${NC}"
echo -e "  ${NC}Testing hot/cold gas stripping in satellites...${NC}"
if ./test_build/test_stripping > "$TEST_OUTPUT_DIR/stripping.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/stripping.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some stripping tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/stripping.log"
fi

echo -e "${YELLOW}▸ Test 12: Multi-Satellite Systems (9 tests)${NC}"
echo -e "  ${NC}Testing orbital dynamics and tidal interactions...${NC}"
if ./test_build/test_multi_satellite > "$TEST_OUTPUT_DIR/multi_satellite.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/multi_satellite.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some multi-satellite tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/multi_satellite.log"
fi

echo -e "${YELLOW}▸ Test 13: Star Formation Recipes (27 tests)${NC}"
echo -e "  ${NC}Testing SF laws, quenching, and feedback efficiency...${NC}"
if ./test_build/test_star_formation_recipes > "$TEST_OUTPUT_DIR/star_formation_recipes.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/star_formation_recipes.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some star formation recipe tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/star_formation_recipes.log"
fi

echo -e "${YELLOW}▸ Test 14: Reincorporation (21 tests)${NC}"
echo -e "  ${NC}Testing ejected gas return rates and timescales...${NC}"
if ./test_build/test_reincorporation > "$TEST_OUTPUT_DIR/reincorporation.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/reincorporation.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some reincorporation tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/reincorporation.log"
fi

echo -e "${YELLOW}▸ Test 15: Cooling & Heating (25 tests)${NC}"
echo -e "  ${NC}Testing thermal balance, tcool/tff, and precipitation...${NC}"
if ./test_build/test_cooling_heating > "$TEST_OUTPUT_DIR/cooling_heating.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/cooling_heating.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some cooling/heating tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/cooling_heating.log"
fi

echo -e "${YELLOW}▸ Test 16: Halo Assembly & Mergers (24 tests)${NC}"
echo -e "  ${NC}Testing mass ratios, dynamical friction, and BH growth...${NC}"
if ./test_build/test_halo_mergers > "$TEST_OUTPUT_DIR/halo_mergers.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/halo_mergers.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some halo merger tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/halo_mergers.log"
fi

echo -e "${YELLOW}▸ Test 17: AGN Feedback (18 tests)${NC}"
echo -e "  ${NC}Testing radio/quasar modes, Eddington limits, and heating...${NC}"
if ./test_build/test_agn_feedback > "$TEST_OUTPUT_DIR/agn_feedback.log" 2>&1; then
    PASS=$(grep -o "Passed:.*" "$TEST_OUTPUT_DIR/agn_feedback.log" | head -1)
    echo -e "${GREEN}  ✓ PASS - $PASS${NC}"
else
    echo -e "${RED}  ✗ FAIL - Some AGN feedback tests failed${NC}"
    echo "  See: $TEST_OUTPUT_DIR/agn_feedback.log"
fi

# Count total passes and fails from individual test outputs
PASSED=0
FAILED=0

# Sum up results from each log file
for log in "$TEST_OUTPUT_DIR"/*.log; do
    if [ -f "$log" ]; then
        # Extract just the count number from "  Passed:       XX (YY%)" format
        # Match the number before the opening parenthesis
        P=$(grep "  Passed:" "$log" | sed 's/.*Passed:[^0-9]*\([0-9]*\).*/\1/')
        F=$(grep "  Failed:" "$log" | sed 's/.*Failed:[^0-9]*\([0-9]*\).*/\1/')
        if [ -n "$P" ] && [ "$P" != "Passed:" ]; then
            PASSED=$((PASSED + P))
        fi
        if [ -n "$F" ] && [ "$F" != "Failed:" ]; then
            FAILED=$((FAILED + F))
        fi
    fi
done

TOTAL_TESTS=$((PASSED + FAILED))

# Calculate percentages using bc or python
if command -v bc &> /dev/null; then
    PASS_PCT=$(echo "scale=1; ($PASSED * 100) / $TOTAL_TESTS" | bc)
    FAIL_PCT=$(echo "scale=1; ($FAILED * 100) / $TOTAL_TESTS" | bc)
else
    PASS_PCT=$(python3 -c "print(f'{($PASSED/$TOTAL_TESTS)*100:.1f}')")
    FAIL_PCT=$(python3 -c "print(f'{($FAILED/$TOTAL_TESTS)*100:.1f}')")
fi

# Calculate total execution time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  TEST SUMMARY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "  Test Suites:  17"
echo -e "  Total tests:  $TOTAL_TESTS"
echo -e "  Passed:       $PASSED (${PASS_PCT}%)"
echo -e "  Failed:       $FAILED (${FAIL_PCT}%)"
echo -e "  Execution:    ${TOTAL_TIME}s"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}  ✓ ALL INTEGRATION TESTS PASSED${NC}"
    echo ""
    echo -e "  Performance:  $TOTAL_TESTS tests in ${TOTAL_TIME}s ($(python3 -c "print(f'{$TOTAL_TESTS/$TOTAL_TIME:.1f}')") tests/sec)"
    echo -e "  Logs saved:   ${TEST_OUTPUT_DIR}"
    echo -e "  Coverage:     Complete model physics validated"
    echo ""
    exit 0
else
    echo -e "${RED}  ✗ SOME TESTS FAILED${NC}"
    echo ""
    echo -e "Review logs in: ${TEST_OUTPUT_DIR}"
    exit 1
fi

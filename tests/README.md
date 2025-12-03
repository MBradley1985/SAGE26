# SAGE26 Test Suite

Comprehensive unit and integration tests for the SAGE26 semi-analytic galaxy formation model with CGM physics extensions.

## Overview

This test suite validates the physical correctness and numerical stability of the SAGE26 model through **320 automated tests** across **17 test suites** covering:
- Conservation laws (mass, metals, energy)
- Regime determination and CGM physics
- Bulge size calculations and morphology
- Physics validation and parameter bounds
- Galaxy mergers and halo assembly
- Disk instability
- Gas infall and reionization
- Numerical stability and edge cases
- Metal enrichment and yields
- Ram pressure stripping
- Multi-satellite systems
- Star formation recipes
- Gas reincorporation
- Cooling and heating physics
- AGN feedback and black hole growth

## Quick Start

### Running All Tests

```bash
# From the tests directory
make test
```

### Running Specific Test Suites

```bash
# Individual test suites
make test_conservation    # Conservation laws (37 tests)
make test_regime         # Regime/CGM physics (21 tests)
make test_bulge          # Bulge sizes (13 tests)
make test_physics        # Physics validation (31 tests)

# Quick test (just conservation - fastest)
make quick
```

### Running Integration Tests

```bash
# Comprehensive integration test with detailed reporting
bash run_integration_tests.sh
```

## Test Suite Details

### 1. Conservation Laws (`test_conservation.c`) - 37 tests

**What it tests:**
- Baryonic mass conservation during star formation
- Metal mass conservation across all processes
- Mass conservation during feedback
- Mass conservation during reincorporation
- Mass conservation during cooling
- No negative masses allowed
- Edge cases with extreme values
- Multi-timestep integration
- Performance benchmarks

**Why it matters:** Conservation laws are the foundation of any SAM. Violations indicate fundamental bugs.

**Key functions tested:**
- `update_from_star_formation()`
- `update_from_feedback()`
- `cooling_recipe()`
- `reincorporate_gas()`

### 2. Regime Determination & CGM Physics (`test_regime_cgm.c`) - 21 tests

**What it tests:**
- Voit (2015) regime boundary calculation (M/Mshock)^(4/3)
- Power law scaling of regime criterion
- CGM precipitation criterion (tcool/tff < 10)
- Gas routing to correct reservoirs (CGM vs HotGas)
- Regime transitions and gas preservation
- Cold stream fraction scaling

**Why it matters:** The two-regime CGM model is a core innovation in SAGE26. Incorrect regime determination leads to wrong cooling/feedback behavior.

**Key functions tested:**
- `determine_and_store_regime()`
- `cooling_recipe_cgm()`
- `cooling_recipe_regime_aware()`
- `update_from_feedback()` (regime-aware routing)

### 3. Bulge Size Physics (`test_bulge_sizes.c`) - 13 tests

**What it tests:**
- Shen et al. (2003) scaling relation (r ∝ M^0.56)
- Two-regime Shen relation (dwarfs vs giants)
- Tonini et al. (2016) separate merger/instability bulges
- Instability bulge radius evolution
- Zero-mass bulge handling
- Disk radius shrinkage after instability

**Why it matters:** Accurate bulge sizes affect observables like half-light radii and size-mass relations.

**Key functions tested:**
- `get_bulge_radius()`
- `update_instability_bulge_radius()`

### 4. Physics Validation (`test_physics_validation.c`) - 31 tests

**What it tests:**
- Star formation efficiency in physical range
- Cooling time scaling with halo mass
- Feedback energy balance
- Metallicities stay in [0, 1] range
- No runaway mass growth
- Massive halo quenching (AGN feedback)
- Fast feedback efficiency bounds

**Why it matters:** These tests catch unphysical parameter values and behaviors that might not violate conservation but produce wrong results.

**Key functions tested:**
- `cooling_recipe()`
- Physics parameter validation
- Multi-step evolution checks

### 5. Galaxy Mergers (`test_mergers.c`) - 13 tests

**What it tests:**
- Major vs minor merger classification (threshold = 0.3)
- Merger timescale calculations (dynamical friction)
- Merger timescale scaling with mass
- Mass conservation during mergers
- Merger remnant bulge radius calculation
- Black hole growth during mergers

**Why it matters:** Mergers drive bulge formation and black hole growth. Incorrect merger physics affects morphology and quenching.

**Key functions tested:**
- `estimate_merging_time()`
- `deal_with_galaxy_merger()`

### 6. Disk Instability (`test_disk_instability.c`) - 9 tests

**What it tests:**
- Critical disk mass (Mcrit) calculation
- Disk stability criterion (Vmax, radius dependence)
- Gas/star fraction during instability
- Bulge growth from disk instability
- Mass conservation during instability
- Zero disk mass edge case
- Gas-only disk handling

**Why it matters:** Disk instability is a key bulge formation channel distinct from mergers.

**Key functions tested:**
- `check_disk_instability()`
- Mass redistribution logic

### 7. Gas Infall (`test_infall.c`) - 12 tests

**What it tests:**
- Baryon fraction in infall calculation
- Positive and negative infall scenarios
- Infall routing by regime (CGM vs HotGas)
- Satellite gas transfer to central
- Mass conservation in infall
- Reionization suppression of infall

**Why it matters:** Infall provides the fuel for star formation. Incorrect infall physics starves or over-feeds galaxies.

**Key functions tested:**
- `add_infall_to_hot()`
- `infall_recipe()`

### 8. Numerical Stability (`test_numerical_stability.c`) - 24 tests

**What it tests:**
- No NaN/Inf values after physics operations
- Extreme mass ratio handling (10^10:1)
- Mass conservation to machine precision
- Roundoff error accumulation over 10k operations
- Division by zero protection
- Timestep convergence
- Protection against negative masses

**Why it matters:** Numerical instabilities can cause crashes or incorrect results in long integrations.

**Key functions tested:**
- All major physics functions under extreme conditions

## Test Framework

### Test Macros

The test framework (`test_framework.h`) provides:

```c
// Assertions
ASSERT_TRUE(condition, message)
ASSERT_FALSE(condition, message)
ASSERT_EQUAL_INT(expected, actual, message)
ASSERT_EQUAL_FLOAT(expected, actual, message)
ASSERT_CLOSE(expected, actual, tolerance, message)
ASSERT_GREATER_THAN(value, threshold, message)
ASSERT_LESS_THAN(value, threshold, message)
ASSERT_IN_RANGE(value, min, max, message)

// Test structure
BEGIN_TEST_SUITE(name)
BEGIN_TEST(name)
END_TEST_SUITE()
PRINT_TEST_SUMMARY()
TEST_EXIT_CODE()
```

### Adding New Tests

1. **Create a new test file:** `test_myfeature.c`

```c
#include "test_framework.h"
#include "../src/core_allvars.h"
#include "../src/model_myfeature.h"

void test_my_physics() {
    BEGIN_TEST("My Physics Test");
    
    // Setup
    struct GALAXY gal;
    memset(&gal, 0, sizeof(struct GALAXY));
    gal.ColdGas = 1.0;
    
    // Test
    double result = my_physics_function(&gal);
    
    // Assert
    ASSERT_GREATER_THAN(result, 0.0, "Result should be positive");
}

int main() {
    BEGIN_TEST_SUITE("My Feature Tests");
    test_my_physics();
    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    return TEST_EXIT_CODE();
}
```

2. **Add to Makefile:**

```makefile
TEST_SOURCES = \
    test_conservation.c \
    test_myfeature.c \
    ...
```

3. **Add to integration test script:**

```bash
echo -e "${YELLOW}▸ Test X: My Feature (N tests)${NC}"
if ./test_build/test_myfeature > "$TEST_OUTPUT_DIR/myfeature.log" 2>&1; then
    ...
fi
```

## Test Output

### Console Output

```
═══════════════════════════════════════════════════════════
  TEST SUITE: Conservation Laws
═══════════════════════════════════════════════════════════

▸ TEST: Star Formation Mass Conservation
  ✓ PASS: ColdGas decreased after SF
  ✓ PASS: StellarMass increased after SF
  ✓ PASS: Total mass conserved
  ✓ PASS: Total metal mass conserved

───────────────────────────────────────────────────────────

═══════════════════════════════════════════════════════════
  TEST SUMMARY
═══════════════════════════════════════════════════════════
  Total tests:  37
  Passed:       37 (100.0%)
  Failed:       0 (0.0%)
═══════════════════════════════════════════════════════════

  ✓ ALL TESTS PASSED!
```

### Log Files

Individual test logs are saved to `test_output/*.log` for detailed debugging.

## Continuous Integration

The test suite is designed for CI/CD integration:

```bash
# Exit code 0 if all tests pass, 1 if any fail
make test
echo $?  # 0 = success, 1 = failure

# Or use integration script
bash run_integration_tests.sh
```

## Performance Benchmarks

Some tests include performance benchmarks:

```
▸ TEST: Performance Regression Test
  ✓ PASS: 10k SF updates in < 10ms
  ℹ Performance: 10k updates completed in 0.123 ms
```

These help catch performance regressions during development.

## Coverage Summary

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| Conservation Laws | 37 | Core physics operations |
| Regime/CGM | 21 | Regime determination, CGM cooling |
| Bulge Sizes | 13 | Size calculations all modes |
| Physics Validation | 31 | Physical bounds, quenching |
| Mergers | 13 | Merger physics, timescales |
| Disk Instability | 9 | Instability criterion |
| Gas Infall | 12 | Infall routing, reionization |
| Numerical Stability | 24 | Edge cases, precision |
| Metal Enrichment | 22 | Stellar yields, SN feedback |
| Ram Pressure Stripping | 14 | Hot/cold gas stripping |
| Multi-Satellite Systems | 9 | Orbital dynamics, tidal effects |
| Star Formation Recipes | 27 | SF laws, quenching mechanisms |
| Reincorporation | 21 | Ejected gas return rates |
| Cooling & Heating | 25 | Thermal balance, precipitation |
| Halo Assembly & Mergers | 24 | Mass ratios, dynamical friction |
| AGN Feedback | 18 | Radio/quasar modes, Eddington limits |
| **Total** | **320** | **Complete model physics** |

## Troubleshooting

### Build Failures

```bash
# Clean and rebuild
make clean
make all
```

### Test Failures

1. Check the detailed log in `test_output/*.log`
2. Run individual test suite: `make test_conservation`
3. Use a debugger: `lldb test_build/test_conservation`

### Missing Source Files

The Makefile requires these source files from `../src/`:
- `core_utils.c`
- `model_misc.c`
- `model_starformation_and_feedback.c`
- `model_cooling_heating.c`
- `model_reincorporation.c`
- `model_infall.c`
- `model_disk_instability.c`
- `model_mergers.c`
- `core_cool_func.c`

If compilation fails, ensure these files exist.

## Best Practices

1. **Run tests before committing:** `make test`
2. **Add tests for new features:** Cover both normal and edge cases
3. **Test conservation first:** Any new physics must conserve mass
4. **Use appropriate tolerances:** 
   - `1e-6` for single precision
   - `1e-12` for double precision
   - `1e-3` for accumulated operations
5. **Document test purpose:** Explain what physics is being tested
6. **Keep tests fast:** Unit tests should run in < 1 second

## References

- Voit (2015) - Regime determination criterion
- Shen et al. (2003) - Bulge size scaling relations
- Tonini et al. (2016) - Separate merger/instability bulges

## Contact

For questions about the test suite, see the main SAGE26 documentation or contact the development team.

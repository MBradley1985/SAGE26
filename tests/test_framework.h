#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// Test statistics
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

// Color codes for output
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_YELLOW  "\x1b[33m"
#define COLOR_BLUE    "\x1b[34m"
#define COLOR_RESET   "\x1b[0m"

// Tolerance for floating point comparisons
#define FLOAT_TOLERANCE 1e-6
#define DOUBLE_TOLERANCE 1e-12

// Test assertion macros
#define ASSERT_TRUE(condition, message) \
    do { \
        tests_run++; \
        if (condition) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Condition failed: %s\n" COLOR_RESET, #condition); \
        } \
    } while(0)

#define ASSERT_FALSE(condition, message) \
    ASSERT_TRUE(!(condition), message)

#define ASSERT_EQUAL_INT(expected, actual, message) \
    do { \
        tests_run++; \
        if ((expected) == (actual)) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected: %d, Got: %d\n" COLOR_RESET, (expected), (actual)); \
        } \
    } while(0)

#define ASSERT_EQUAL_FLOAT(expected, actual, message) \
    do { \
        tests_run++; \
        if (fabs((expected) - (actual)) < FLOAT_TOLERANCE) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected: %.6e, Got: %.6e (diff: %.6e)\n" COLOR_RESET, \
                   (expected), (actual), fabs((expected) - (actual))); \
        } \
    } while(0)

#define ASSERT_EQUAL_DOUBLE(expected, actual, message) \
    do { \
        tests_run++; \
        if (fabs((expected) - (actual)) < DOUBLE_TOLERANCE) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected: %.12e, Got: %.12e (diff: %.12e)\n" COLOR_RESET, \
                   (expected), (actual), fabs((expected) - (actual))); \
        } \
    } while(0)

#define ASSERT_CLOSE(expected, actual, tolerance, message) \
    do { \
        tests_run++; \
        double rel_error = fabs(((expected) - (actual)) / ((expected) + 1e-100)); \
        if (rel_error < (tolerance)) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected: %.6e, Got: %.6e (rel error: %.2e)\n" COLOR_RESET, \
                   (expected), (actual), rel_error); \
        } \
    } while(0)

#define ASSERT_GREATER_THAN(value, threshold, message) \
    do { \
        tests_run++; \
        if ((value) > (threshold)) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected > %.6e, Got: %.6e\n" COLOR_RESET, (threshold), (value)); \
        } \
    } while(0)

#define ASSERT_LESS_THAN(value, threshold, message) \
    do { \
        tests_run++; \
        if ((value) < (threshold)) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected < %.6e, Got: %.6e\n" COLOR_RESET, (threshold), (value)); \
        } \
    } while(0)

#define ASSERT_IN_RANGE(value, min, max, message) \
    do { \
        tests_run++; \
        if ((value) >= (min) && (value) <= (max)) { \
            tests_passed++; \
            printf(COLOR_GREEN "  ✓ PASS: %s\n" COLOR_RESET, message); \
        } else { \
            tests_failed++; \
            printf(COLOR_RED "  ✗ FAIL: %s\n" COLOR_RESET, message); \
            printf(COLOR_RED "    Expected in [%.6e, %.6e], Got: %.6e\n" COLOR_RESET, \
                   (min), (max), (value)); \
        } \
    } while(0)

// Test suite macros
#define BEGIN_TEST_SUITE(name) \
    printf("\n" COLOR_BLUE "═══════════════════════════════════════════════════════════\n"); \
    printf("  TEST SUITE: %s\n", name); \
    printf("═══════════════════════════════════════════════════════════\n" COLOR_RESET);

#define END_TEST_SUITE() \
    printf(COLOR_BLUE "───────────────────────────────────────────────────────────\n" COLOR_RESET);

#define BEGIN_TEST(name) \
    printf(COLOR_YELLOW "\n▸ TEST: %s\n" COLOR_RESET, name);

#define PRINT_TEST_SUMMARY() \
    do { \
        printf("\n" COLOR_BLUE "═══════════════════════════════════════════════════════════\n"); \
        printf("  TEST SUMMARY\n"); \
        printf("═══════════════════════════════════════════════════════════\n" COLOR_RESET); \
        printf("  Total tests:  %d\n", tests_run); \
        printf(COLOR_GREEN "  Passed:       %d (%.1f%%)\n" COLOR_RESET, \
               tests_passed, 100.0 * tests_passed / (tests_run + 1e-10)); \
        if (tests_failed > 0) { \
            printf(COLOR_RED "  Failed:       %d (%.1f%%)\n" COLOR_RESET, \
                   tests_failed, 100.0 * tests_failed / (tests_run + 1e-10)); \
        } else { \
            printf(COLOR_GREEN "  Failed:       0 (0.0%%)\n" COLOR_RESET); \
        } \
        printf(COLOR_BLUE "═══════════════════════════════════════════════════════════\n" COLOR_RESET); \
        if (tests_failed == 0) { \
            printf(COLOR_GREEN "\n  ✓ ALL TESTS PASSED!\n\n" COLOR_RESET); \
        } else { \
            printf(COLOR_RED "\n  ✗ SOME TESTS FAILED\n\n" COLOR_RESET); \
        } \
    } while(0)

#define TEST_EXIT_CODE() (tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS)

#endif // TEST_FRAMEWORK_H

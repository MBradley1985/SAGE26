# Makefile for SAGE Model Unit Tests
#
# This Makefile compiles unit tests for the SAGE semi-analytic model
# with CGM physics extensions.
#
# Usage:
#   make              # Build all tests
#   make test         # Build and run all tests
#   make clean        # Clean build artifacts
#   make test_name    # Build specific test

# Compiler settings
CC = gcc
ROOT_DIR := $(shell cd .. && pwd)
CFLAGS = -std=gnu99 -Wall -Wextra -g -O2 -I../src -DROOT_DIR='"$(ROOT_DIR)"'
LDFLAGS = -lm

# Directories
SRC_DIR = ../src
TEST_DIR = .
OUTPUT_DIR = ./test_build

# Source files from main project (needed for tests)
PROJECT_SOURCES = \
	$(SRC_DIR)/core_utils.c \
	$(SRC_DIR)/model_misc.c \
	$(SRC_DIR)/model_starformation_and_feedback.c \
	$(SRC_DIR)/model_cooling_heating.c \
	$(SRC_DIR)/model_reincorporation.c \
	$(SRC_DIR)/model_infall.c \
	$(SRC_DIR)/model_disk_instability.c \
	$(SRC_DIR)/model_mergers.c \
	$(SRC_DIR)/core_cool_func.c

# Test files
TEST_SOURCES = \
	test_conservation.c \
	test_regime_cgm.c \
	test_bulge_sizes.c \
	test_physics_validation.c \
	test_mergers.c \
	test_disk_instability.c \
	test_infall.c \
	test_numerical_stability.c

# Test executables
TEST_BINS = $(TEST_SOURCES:%.c=$(OUTPUT_DIR)/%)

# Default target
all: $(TEST_BINS)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Pattern rule for building tests
$(OUTPUT_DIR)/test_%: test_%.c $(PROJECT_SOURCES) | $(OUTPUT_DIR)
	@echo "Building $@..."
	$(CC) $(CFLAGS) $< $(PROJECT_SOURCES) $(LDFLAGS) -o $@
	@echo "✓ Built $@"

# Run all tests
test: $(TEST_BINS)
	@echo ""
	@echo "════════════════════════════════════════════════════════════"
	@echo "  RUNNING ALL TESTS"
	@echo "════════════════════════════════════════════════════════════"
	@failed=0; \
	for test in $(TEST_BINS); do \
		echo ""; \
		$$test || failed=$$((failed + 1)); \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "════════════════════════════════════════════════════════════"; \
		echo "  ✓ ALL TEST SUITES PASSED"; \
		echo "════════════════════════════════════════════════════════════"; \
		echo ""; \
	else \
		echo "════════════════════════════════════════════════════════════"; \
		echo "  ✗ $$failed TEST SUITE(S) FAILED"; \
		echo "════════════════════════════════════════════════════════════"; \
		echo ""; \
		exit 1; \
	fi

# Run individual test suites
test_conservation: $(OUTPUT_DIR)/test_conservation
	$(OUTPUT_DIR)/test_conservation

test_regime: $(OUTPUT_DIR)/test_regime_cgm
	$(OUTPUT_DIR)/test_regime_cgm

test_bulge: $(OUTPUT_DIR)/test_bulge_sizes
	$(OUTPUT_DIR)/test_bulge_sizes

test_physics: $(OUTPUT_DIR)/test_physics_validation
	$(OUTPUT_DIR)/test_physics_validation

# Quick test (just conservation laws - fastest)
quick: $(OUTPUT_DIR)/test_conservation
	$(OUTPUT_DIR)/test_conservation

# Clean build artifacts
clean:
	rm -rf $(OUTPUT_DIR)
	@echo "✓ Cleaned build artifacts"

# Help target
help:
	@echo "SAGE Model Test Suite"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build all tests"
	@echo "  make test         - Build and run all tests"
	@echo "  make quick        - Run quick tests (conservation only)"
	@echo "  make test_conservation - Run conservation tests"
	@echo "  make test_conservation - Run conservation tests"
	@echo "  make test_regime  - Run regime/CGM tests"
	@echo "  make test_bulge   - Run bulge size tests"
	@echo "  make test_physics - Run physics validation tests"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make help         - Show this help"
	@echo ""

.PHONY: all test quick clean help test_conservation test_regime test_bulge test_physics
#!/bin/bash
#
# Run H2 Area Test Suite
# ======================
# This script runs SAGE with different H2 prescriptions and disk area options
# to test the impact of surface density normalization on H2 fractions.
#
# H2 prescriptions tested:
#   1 = BR06
#   4 = KD12
#   5 = KMT09
#   6 = K13
#   7 = GD14
#
# H2DiskAreaOption values:
#   0 = π*r_s²
#   1 = π*(3*r_s)² (default)
#   2 = 2π*r_s² (central Σ₀)

set -e  # Exit on error

# Configuration
PAR_FILE="input/millennium.par"
PAR_BACKUP="input/millennium.par.backup"
SAGE_EXEC="./sage"
BASE_OUTPUT_DIR="output"

# H2 prescriptions to test
H2_PRESCRIPTIONS=(1 4 5 6 7)
H2_PRESCRIPTION_NAMES=("BR06" "KD12" "KMT09" "K13" "GD14")

# Disk area options to test
AREA_OPTIONS=(0 1 2)
AREA_OPTION_NAMES=("pi_rs2" "9pi_rs2" "2pi_rs2")

echo "========================================"
echo "SAGE H2 Area Test Suite"
echo "========================================"
echo ""
echo "This will run ${#H2_PRESCRIPTIONS[@]} H2 prescriptions × ${#AREA_OPTIONS[@]} area options = $((${#H2_PRESCRIPTIONS[@]} * ${#AREA_OPTIONS[@]})) total runs"
echo ""

# Check if SAGE executable exists
if [ ! -f "$SAGE_EXEC" ]; then
    echo "Error: SAGE executable not found at $SAGE_EXEC"
    echo "Please compile SAGE first: make clean && make"
    exit 1
fi

# Check if parameter file exists
if [ ! -f "$PAR_FILE" ]; then
    echo "Error: Parameter file not found at $PAR_FILE"
    exit 1
fi

# Backup original parameter file
echo "Backing up parameter file to $PAR_BACKUP"
cp "$PAR_FILE" "$PAR_BACKUP"

# Function to update parameter file
update_parameter() {
    local param_name=$1
    local param_value=$2
    local file=$3
    
    # Use sed to replace the parameter value (works on macOS and Linux)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^\(${param_name}[[:space:]]*\)[0-9]*.*/\1${param_value}/" "$file"
    else
        # Linux
        sed -i "s/^\(${param_name}[[:space:]]*\)[0-9]*.*/\1${param_value}/" "$file"
    fi
}

# Function to update output directory in parameter file
update_output_dir() {
    local output_dir=$1
    local file=$2
    
    # Escape forward slashes for sed
    local escaped_dir=$(echo "$output_dir" | sed 's/\//\\\//g')
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^OutputDir[[:space:]].*/OutputDir   ${escaped_dir}/" "$file"
    else
        # Linux
        sed -i "s/^OutputDir[[:space:]].*/OutputDir   ${escaped_dir}/" "$file"
    fi
}

# Counter for tracking progress
total_runs=$((${#H2_PRESCRIPTIONS[@]} * ${#AREA_OPTIONS[@]}))
current_run=0

# Main loop
for idx in "${!H2_PRESCRIPTIONS[@]}"; do
    prescription=${H2_PRESCRIPTIONS[$idx]}
    prescription_name=${H2_PRESCRIPTION_NAMES[$idx]}
    
    for area_idx in "${!AREA_OPTIONS[@]}"; do
        area_option=${AREA_OPTIONS[$area_idx]}
        area_name=${AREA_OPTION_NAMES[$area_idx]}
        
        current_run=$((current_run + 1))
        
        # Define output directory (convert prescription name to lowercase)
        prescription_lower=$(echo "$prescription_name" | tr '[:upper:]' '[:lower:]')
        output_dir="${BASE_OUTPUT_DIR}/millennium_${prescription_lower}_${area_name}/"
        
        echo ""
        echo "========================================"
        echo "Run $current_run/$total_runs"
        echo "========================================"
        echo "H2 Prescription: $prescription_name (SFprescription=$prescription)"
        echo "Disk Area Option: $area_name (H2DiskAreaOption=$area_option)"
        echo "Output Directory: $output_dir"
        echo ""
        
        # Create output directory
        mkdir -p "$output_dir"
        
        # Update parameter file
        echo "Updating parameter file..."
        update_parameter "SFprescription" "$prescription" "$PAR_FILE"
        update_parameter "H2DiskAreaOption" "$area_option" "$PAR_FILE"
        update_output_dir "$output_dir" "$PAR_FILE"
        
        # Run SAGE
        echo "Running SAGE..."
        start_time=$(date +%s)
        $SAGE_EXEC "$PAR_FILE"
        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        
        echo ""
        echo "✓ Completed in ${runtime}s"
        echo "  Output saved to: $output_dir"
    done
done

# Restore original parameter file
echo ""
echo "========================================"
echo "Restoring original parameter file..."
echo "========================================"
mv "$PAR_BACKUP" "$PAR_FILE"

echo ""
echo "========================================"
echo "All runs complete!"
echo "========================================"
echo ""
echo "Total runs completed: $total_runs"
echo ""
echo "Output directories created:"
for idx in "${!H2_PRESCRIPTIONS[@]}"; do
    prescription_name=${H2_PRESCRIPTION_NAMES[$idx]}
    prescription_lower=$(echo "$prescription_name" | tr '[:upper:]' '[:lower:]')
    for area_name in "${AREA_OPTION_NAMES[@]}"; do
        echo "  - ${BASE_OUTPUT_DIR}/millennium_${prescription_lower}_${area_name}/"
    done
done
echo ""
echo "Next step: Run plotting script to analyze results"
echo "  python3 plotting/h2_area_comparison_plots.py"
echo ""

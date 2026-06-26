#!/bin/bash
# =============================================================================
# Top-K Rice Varieties Classification Pipeline
# =============================================================================
# This script runs the complete pipeline for all top-k variety configurations
# (from top-1 to top-13 varieties) and generates comparison results.
#
# Usage:
#   chmod +x run_topk_pipeline.sh
#   ./run_topk_pipeline.sh
#
# Or run for specific k values:
#   ./run_topk_pipeline.sh 1 5 10 13
# =============================================================================

# Don't use set -e as it can cause issues with the loop
# set -e  # Exit on error

# Configuration
BASE_DIR="/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2/global motif counting based/all_top_k"
GPU_ID=2
MAX_VARIETIES=13

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo "=============================================================================="
echo "           TOP-K RICE VARIETIES CLASSIFICATION PIPELINE"
echo "=============================================================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Base Directory: ${BASE_DIR}"
echo "GPU Device: ${GPU_ID}"
echo ""

# Get k values from arguments or use all
if [ $# -eq 0 ]; then
    # No arguments - run for all k values (1 to 13)
    K_VALUES=($(seq 1 ${MAX_VARIETIES}))
    echo -e "${BLUE}Running for ALL top-k values: 1 to ${MAX_VARIETIES}${NC}"
else
    # Use provided k values
    K_VALUES=("$@")
    echo -e "${BLUE}Running for specified k values: ${K_VALUES[*]}${NC}"
fi

echo ""
echo "=============================================================================="

# Create log directory
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Function to run pipeline for a single k value
run_pipeline_for_k() {
    local k=$1
    local start_time=$(date +%s)
    
    echo ""
    echo -e "${YELLOW}##############################################################################${NC}"
    echo -e "${YELLOW}#  PROCESSING TOP-${k} VARIETIES${NC}"
    echo -e "${YELLOW}##############################################################################${NC}"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Create output directories
    OUTPUT_DIR="${BASE_DIR}/top${k}"
    DATA_DIR="${OUTPUT_DIR}/data"
    RESULTS_DIR="${OUTPUT_DIR}/results"
    mkdir -p "${DATA_DIR}"
    mkdir -p "${RESULTS_DIR}"
    
    # Step 1: Run motif pipeline
    echo -e "${BLUE}Step 1: Running Motif Pipeline for top-${k}...${NC}"
    PIPELINE_LOG="${LOG_DIR}/top${k}_pipeline.log"
    
    cd "${BASE_DIR}"
    python rice_motif_pipeline_topk.py --top_k ${k} > "${PIPELINE_LOG}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Motif pipeline completed successfully${NC}"
    else
        echo -e "${RED}  ✗ Motif pipeline failed! Check ${PIPELINE_LOG}${NC}"
        return 1
    fi
    
    # Step 2: Run classifier
    echo -e "${BLUE}Step 2: Running Hybrid Classifier for top-${k}...${NC}"
    CLASSIFIER_LOG="${LOG_DIR}/top${k}_classifier.log"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python rice_hybrid_classifier_topk.py --top_k ${k} --gpu ${GPU_ID} > "${CLASSIFIER_LOG}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Classifier completed successfully${NC}"
    else
        echo -e "${RED}  ✗ Classifier failed! Check ${CLASSIFIER_LOG}${NC}"
        return 1
    fi
    
    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    
    echo ""
    echo -e "${GREEN}TOP-${k} COMPLETED in ${minutes}m ${seconds}s${NC}"
    echo "Results saved to: ${RESULTS_DIR}"
    
    # Extract and display key results
    if [ -f "${RESULTS_DIR}/results_summary.json" ]; then
        echo ""
        echo "Quick Results (Best Model):"
        python3 -c "
import json
try:
    with open('${RESULTS_DIR}/results_summary.json', 'r') as f:
        data = json.load(f)
    best = data['best_model']
    res = data['results'][best]['test']
    print(f\"  Model: {best}\")
    print(f\"  AUC: {res['auc']:.4f}, F1: {res['f1']:.4f}, MCC: {res['mcc']:.4f}\")
except Exception as e:
    print(f\"  Could not load results: {e}\")
" || true
    fi
    
    return 0
}

# Track overall progress
TOTAL_K=${#K_VALUES[@]}
COMPLETED=0
FAILED=0
FAILED_K=()

PIPELINE_START=$(date +%s)

# Run for each k value
for k in "${K_VALUES[@]}"; do
    run_pipeline_for_k ${k}
    
    if [ $? -eq 0 ]; then
        ((COMPLETED++))
    else
        ((FAILED++))
        FAILED_K+=($k)
    fi
    
    echo ""
    echo "Progress: ${COMPLETED}/${TOTAL_K} completed, ${FAILED} failed"
    echo ""
done

# Run comparison script
echo ""
echo "=============================================================================="
echo -e "${YELLOW}RUNNING COMPARISON ANALYSIS${NC}"
echo "=============================================================================="

cd "${BASE_DIR}"
python compare_topk_results.py

echo ""
echo "=============================================================================="
echo -e "${GREEN}PIPELINE COMPLETE${NC}"
echo "=============================================================================="

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total Elapsed Time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "Summary:"
echo "  - Total configurations: ${TOTAL_K}"
echo "  - Successful: ${COMPLETED}"
echo "  - Failed: ${FAILED}"

if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}  - Failed k values: ${FAILED_K[*]}${NC}"
fi

echo ""
echo "Results location: ${BASE_DIR}"
echo "  - Individual results: ${BASE_DIR}/top{k}/results/"
echo "  - Comparison results: ${BASE_DIR}/comparison_results/"
echo "  - Logs: ${LOG_DIR}/"
echo ""
echo "=============================================================================="

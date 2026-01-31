#!/bin/bash

# Script to run ablation experiments in parallel across multiple GPUs
# Usage: ./run_ablation_parallel.sh

# Configuration
CFG_DIR="cfg_files/ablation/neuman/cloth"
BASE_CFG_DIR="cfg_files/ablation/neuman/cloth"
NUM_GPUS=6  # Number of GPUs available

# List of ablation config files
declare -a configs=(
    "abl_cloth_baseline.yaml"
    "abl_cloth_no_deform.yaml"
    "abl_cloth_sim_only.yaml"
    "abl_cloth_arap_only.yaml"
    "abl_cloth_mask_only.yaml"
    "abl_cloth_sim_arap.yaml"
    "abl_cloth_sim_mask.yaml"
    "abl_cloth_arap_mask.yaml"
    "abl_cloth_full.yaml"
    "abl_cloth_full_no_lbs.yaml"
    "abl_cloth_sim_only_no_lbs.yaml"
    "abl_cloth_full_lbs_weak.yaml"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Parallel Ablation Experiment Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo -e "${YELLOW}Error: main.py not found. Please run from ml-hugs directory.${NC}"
    exit 1
fi

# Count configs
num_configs=${#configs[@]}
echo -e "${GREEN}Found ${num_configs} ablation configs${NC}"
echo -e "${GREEN}Using ${NUM_GPUS} GPUs${NC}"
echo ""

# Create log directory for parallel run logs
mkdir -p logs_parallel
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="logs_parallel/run_${timestamp}"
mkdir -p "$log_dir"

echo -e "${BLUE}Starting experiments...${NC}"
echo -e "${BLUE}Logs will be saved to: ${log_dir}${NC}"
echo ""

# Function to run experiment on a specific GPU
run_experiment() {
    local cfg_file=$1
    local gpu_id=$2
    local log_file="$log_dir/gpu${gpu_id}_${cfg_file%.yaml}.log"
    
    echo -e "${GREEN}[GPU ${gpu_id}]${NC} Starting: ${cfg_file}"
    python main.py \
        --cfg_file "${BASE_CFG_DIR}/${cfg_file}" \
        --gpu ${gpu_id} \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[GPU ${gpu_id}]${NC} Completed: ${cfg_file}"
    else
        echo -e "${YELLOW}[GPU ${gpu_id}]${NC} Failed: ${cfg_file} (exit code: $exit_code)"
    fi
    return $exit_code
}

# Launch experiments in parallel
pids=()
gpu_counter=0

for i in "${!configs[@]}"; do
    cfg_file="${configs[$i]}"
    
    # Assign GPU (round-robin)
    gpu_id=$((gpu_counter % NUM_GPUS))
    
    # Run experiment in background
    run_experiment "$cfg_file" "$gpu_id" &
    pids+=($!)
    
    echo -e "${BLUE}Launched ${cfg_file} on GPU ${gpu_id} (PID: ${pids[-1]})${NC}"
    
    # Increment GPU counter
    ((gpu_counter++))
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All experiments launched!${NC}"
echo -e "${BLUE}Waiting for completion...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Wait for all background processes to complete
failed_count=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    cfg_file="${configs[$i]}"
    
    if wait $pid; then
        echo -e "${GREEN}✓${NC} ${cfg_file} completed successfully"
    else
        echo -e "${YELLOW}✗${NC} ${cfg_file} failed"
        ((failed_count++))
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total experiments: ${num_configs}"
echo -e "${GREEN}Successful: $((num_configs - failed_count))${NC}"
if [ $failed_count -gt 0 ]; then
    echo -e "${YELLOW}Failed: ${failed_count}${NC}"
fi
echo -e "Log directory: ${log_dir}"
echo ""

if [ $failed_count -eq 0 ]; then
    echo -e "${GREEN}All experiments completed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some experiments failed. Check logs in ${log_dir}${NC}"
    exit 1
fi

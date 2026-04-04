#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /home/alfred/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ml-delta

DATASET=${1:-"mmmlu"}

LANGS=("bn" "ja" "sw")
GPU_IDS="4,5,6,7"
LOG_DIR="${PROJECT_ROOT}/outputs-exp_v2/logs"
mkdir -p "$LOG_DIR"

# Sequence of settings to run:
# 1. filter, lora
# 2. filter, full
# 3. no-filter, lora
# 4. no-filter, full
SETTINGS=(
    "filter lora"
    "filter full"
    "no-filter lora"
    "no-filter full"
)

for setting in "${SETTINGS[@]}"; do
    read -r CURRENT_FILTER CURRENT_MODEL_TYPE <<< "$setting"
    
    echo "--------------------------------------------------------------------------------"
    echo "Running DPO Training -> Dataset: $DATASET | Filter: $CURRENT_FILTER | Type: $CURRENT_MODEL_TYPE"
    echo "--------------------------------------------------------------------------------"

    EXTRA_ARGS=""
    if [ "$CURRENT_MODEL_TYPE" == "full" ]; then
        EXTRA_ARGS="--full_finetune"
    fi

    for lang in "${LANGS[@]}"; do
        run_name="dpo_${CURRENT_FILTER}_${CURRENT_MODEL_TYPE}_${DATASET}_${lang}"
        model_dir="/external1/alfred/models/ml-reasoning-exp_v2/${run_name}"

        # Skip if model already exists (check for adapter_config.json for LoRA, config.json for full)
        if [ "$CURRENT_MODEL_TYPE" == "lora" ] && [ -f "${model_dir}/adapter_config.json" ]; then
            echo "  [${lang}] Already exists: ${model_dir}. Skipping."
            continue
        elif [ "$CURRENT_MODEL_TYPE" == "full" ] && [ -f "${model_dir}/config.json" ]; then
            echo "  [${lang}] Already exists: ${model_dir}. Skipping."
            continue
        fi

        log_file="${LOG_DIR}/${run_name}.log"
        echo "  [${lang}] Training on GPUs ${GPU_IDS} -> ${log_file}"

        DS_ARGS=""
        if [ "$CURRENT_MODEL_TYPE" == "full" ]; then
            DS_ARGS="--use_deepspeed --deepspeed_config_file ${PROJECT_ROOT}/configs/deepspeed_zero3.json"
        fi

        CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch --num_processes 4 $DS_ARGS \
            "${PROJECT_ROOT}/training/run_dpo-exp_v2.py" \
            --dataset "$DATASET" --language "$lang" --filter "$CURRENT_FILTER" $EXTRA_ARGS \
            > "$log_file" 2>&1

        if [ $? -ne 0 ]; then
            echo "DPO training failed for ${lang} in setting: ${CURRENT_FILTER} ${CURRENT_MODEL_TYPE}"
            exit 1
        fi
    done

    echo "Completed setting: $CURRENT_FILTER $CURRENT_MODEL_TYPE"
done

echo "All DPO training sequences complete for $DATASET."

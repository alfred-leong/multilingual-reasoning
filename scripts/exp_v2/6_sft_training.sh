#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /home/alfred/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate ml-delta

DATASET=${1:-"mmmlu"}

LANGS=("ja" "bn" "sw")
GPU_IDS="1,2,3"          # all three GPUs (used together for full finetune)
SINGLE_GPUS=(1 2 3)      # one per language (used for LoRA parallel runs)
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
    echo "Running SFT Training -> Dataset: $DATASET | Filter: $CURRENT_FILTER | Type: $CURRENT_MODEL_TYPE"
    echo "--------------------------------------------------------------------------------"

    EXTRA_ARGS=""
    if [ "$CURRENT_MODEL_TYPE" == "full" ]; then
        EXTRA_ARGS="--full_finetune"
    fi

    if [ "$CURRENT_MODEL_TYPE" == "lora" ]; then
        # LoRA: run all 3 languages in parallel, one GPU each
        PIDS=()
        for i in {0..2}; do
            lang="${LANGS[$i]}"
            gpu="${SINGLE_GPUS[$i]}"
            run_name="sft_${CURRENT_FILTER}_${CURRENT_MODEL_TYPE}_${DATASET}_${lang}"
            model_dir="/external1/alfred/models/ml-reasoning-exp_v2/${run_name}"

            if [ -f "${model_dir}/adapter_config.json" ]; then
                echo "  [${lang}] Already exists: ${model_dir}. Skipping."
                continue
            fi

            log_file="${LOG_DIR}/${run_name}.log"
            echo "  [${lang}] Training on GPU ${gpu} -> ${log_file}"

            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/training/run_sft-exp_v2.py" \
                --dataset "$DATASET" --language "$lang" --filter "$CURRENT_FILTER" $EXTRA_ARGS \
                > "$log_file" 2>&1 &

            PIDS+=($!)
        done

        FAIL=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                FAIL=1
            fi
        done

        if [[ $FAIL -eq 1 ]]; then
            echo "SFT training encountered an error in setting: $CURRENT_FILTER $CURRENT_MODEL_TYPE"
            exit 1
        fi

    else
        # Full finetune: run each language sequentially using all 3 GPUs + DeepSpeed
        for lang in "${LANGS[@]}"; do
            run_name="sft_${CURRENT_FILTER}_${CURRENT_MODEL_TYPE}_${DATASET}_${lang}"
            model_dir="/external1/alfred/models/ml-reasoning-exp_v2/${run_name}"

            if [ -f "${model_dir}/config.json" ]; then
                echo "  [${lang}] Already exists: ${model_dir}. Skipping."
                continue
            fi

            log_file="${LOG_DIR}/${run_name}.log"
            echo "  [${lang}] Training on GPUs ${GPU_IDS} -> ${log_file}"

            CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch --num_processes 3 \
                --use_deepspeed --deepspeed_config_file "${PROJECT_ROOT}/configs/deepspeed_zero3.json" \
                "${PROJECT_ROOT}/training/run_sft-exp_v2.py" \
                --dataset "$DATASET" --language "$lang" --filter "$CURRENT_FILTER" $EXTRA_ARGS \
                > "$log_file" 2>&1

            if [ $? -ne 0 ]; then
                echo "SFT training failed for ${lang} in setting: ${CURRENT_FILTER} ${CURRENT_MODEL_TYPE}"
                exit 1
            fi
        done
    fi

    echo "Completed setting: $CURRENT_FILTER $CURRENT_MODEL_TYPE"
done

echo "All SFT training sequences complete for $DATASET."

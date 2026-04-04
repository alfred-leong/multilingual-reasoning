#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_SETUP="${CONDA_SETUP:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SETUP"
conda activate "${CONDA_ENV:-ml-delta}"

DATASET=${1:-"mmmlu"}
SUBSET=${2:-""}

LANGS=("ja" "bn" "sw")
GPUS=(5 6 7)
MODEL_STORE="${MODEL_STORE:-/external1/alfred/models/ml-reasoning-exp_v2}"
LOG_DIR="${PROJECT_ROOT}/outputs-exp_v2/logs"
mkdir -p "$LOG_DIR"

BASE_MODEL="Qwen/Qwen3-1.7B"

SUBSET_ARGS=""
SUBSET_TAG=""
if [ -n "$SUBSET" ]; then
    SUBSET_ARGS="--subset $SUBSET"
    SUBSET_TAG="_${SUBSET}"
fi

echo "Evaluating All Models -> Dataset: $DATASET (subset: ${SUBSET:-full})"

PIDS=()

for i in {0..2}; do
    lang="${LANGS[$i]}"
    gpu="${GPUS[$i]}"

    # Collect all model dirs for this language+dataset
    mapfile -t MODEL_DIRS < <(find "$MODEL_STORE" -maxdepth 1 -type d -name "*${DATASET}*${lang}*" | sort)

    (
        set -e

        # 1. Base Model
        log_file="${LOG_DIR}/eval_base_${DATASET}${SUBSET_TAG}_${lang}.log"
        out_file="${PROJECT_ROOT}/outputs-exp_v2/eval_${DATASET}${SUBSET_TAG}_${lang}_Qwen3-1.7B.json"
        if [ -f "$out_file" ]; then
            echo "  [${lang}] BASE already evaluated. Skipping."
        else
            echo "  [${lang}] Evaluating BASE on GPU ${gpu} -> ${log_file}"
            CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
                --dataset "$DATASET" --language "$lang" --model_dir "$BASE_MODEL" --full_finetune $SUBSET_ARGS \
                > "$log_file" 2>&1
        fi

        # 2. All discovered model dirs — sequential on this GPU
        for model_dir in "${MODEL_DIRS[@]}"; do
            model_name="$(basename "$model_dir")"
            log_file="${LOG_DIR}/eval_${model_name}${SUBSET_TAG}.log"
            out_file="${PROJECT_ROOT}/outputs-exp_v2/eval_${DATASET}${SUBSET_TAG}_${lang}_${model_name}.json"

            if [ -f "$out_file" ]; then
                echo "  [${lang}] ${model_name} already evaluated. Skipping."
                continue
            fi

            # Detect LoRA vs full finetune by presence of adapter_config.json
            EXTRA_ARGS=""
            if [ ! -f "${model_dir}/adapter_config.json" ]; then
                EXTRA_ARGS="--full_finetune"
            fi

            echo "  [${lang}] Evaluating ${model_name} on GPU ${gpu} -> ${log_file}"
            CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
                --dataset "$DATASET" --language "$lang" --model_dir "$model_dir" $EXTRA_ARGS $SUBSET_ARGS \
                > "$log_file" 2>&1
        done
    ) &
    PIDS+=($!)
done

# Wait for all language subshells
echo "Waiting for all evaluation processes to complete..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [[ $FAIL -eq 1 ]]; then
    echo "Evaluation encountered an error. Check logs in $LOG_DIR"
    exit 1
fi

echo "Evaluation complete. Results saved in outputs-exp_v2/."

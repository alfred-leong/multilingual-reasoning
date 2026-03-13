#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /tier1/home/lweilun/miniconda3/etc/profile.d/conda.sh
conda activate ml-delta

DATASET=${1:-"mmmlu"}
TRAIN_METHOD=${2:-"sft"} # "sft" or "dpo"
FILTER=${3:-"no-filter"}
MODEL_TYPE=${4:-"lora"} # "lora" or "full"

LANGS=("ja" "bn" "sw")
GPUS=(5 6 7)
LOG_DIR="${PROJECT_ROOT}/outputs-exp_v2/logs"
mkdir -p "$LOG_DIR"

echo "Evaluating Models -> Dataset: $DATASET | Method: $TRAIN_METHOD | Filter: $FILTER | Type: $MODEL_TYPE"

EXTRA_ARGS="" 
if [ "$MODEL_TYPE" == "full" ]; then
    EXTRA_ARGS="--full_finetune"
fi

PIDS=()
for i in {0..2}; do
    lang="${LANGS[$i]}"
    gpu="${GPUS[$i]}"
    run_name="${TRAIN_METHOD}_${FILTER}_${MODEL_TYPE}_${DATASET}_${lang}"
    model_dir="/external1/alfred/models/ml-reasoning-exp_v2/${run_name}"
    log_file="${LOG_DIR}/eval_${run_name}.log"
    
    if [ ! -d "$model_dir" ]; then
        echo "  [${lang}] Model directory $model_dir does not exist. Skipping."
        continue
    fi
    
    echo "  [${lang}] Evaluating on GPU ${gpu} -> ${log_file}"
    
    CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/evaluation/run_eval-exp_v2.py" \
        --dataset "$DATASET" --language "$lang" --model_dir "$model_dir" $EXTRA_ARGS \
        > "$log_file" 2>&1 &
    
    PIDS+=($!)
done

# Wait
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [[ $FAIL -eq 1 ]]; then
    echo "Evaluation encountered an error. Check logs."
    exit 1
fi

echo "Evaluation complete. Results saved in outputs-exp_v2/."

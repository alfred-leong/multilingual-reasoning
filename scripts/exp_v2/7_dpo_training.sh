#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /tier1/home/lweilun/miniconda3/etc/profile.d/conda.sh
conda activate ml-delta

DATASET=${1:-"mmmlu"}
FILTER=${2:-"no-filter"}
MODEL_TYPE=${3:-"lora"} # "lora" or "full"

LANGS=("ja" "bn" "sw")
GPUS=(5 6 7)
LOG_DIR="${PROJECT_ROOT}/outputs-exp_v2/logs"
mkdir -p "$LOG_DIR"

echo "Running DPO Training -> Dataset: $DATASET | Filter: $FILTER | Type: $MODEL_TYPE"

EXTRA_ARGS=""
if [ "$MODEL_TYPE" == "full" ]; then
    EXTRA_ARGS="--full_finetune"
fi

PIDS=()
for i in {0..2}; do
    lang="${LANGS[$i]}"
    gpu="${GPUS[$i]}"
    log_file="${LOG_DIR}/dpo_${FILTER}_${MODEL_TYPE}_${DATASET}_${lang}.log"
    echo "  [${lang}] Training on GPU ${gpu} -> ${log_file}"
    
    CUDA_VISIBLE_DEVICES="$gpu" python3 "${PROJECT_ROOT}/training/run_dpo-exp_v2.py" \
        --dataset "$DATASET" --language "$lang" --filter "$FILTER" $EXTRA_ARGS \
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
    echo "DPO training encountered an error."
    exit 1
fi

echo "DPO training complete."

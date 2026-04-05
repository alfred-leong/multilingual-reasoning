#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# source /home/alfred/miniconda3/etc/profile.d/conda.sh
# conda activate ml-delta

DATASET="mmmlu"
SUBSET="100"

# Define 3 configurations: GPU, LANG, MODEL_CFG_NAME
declare -a CONFIGS
CONFIGS[0]="GPU:2|LANG:sw|MODEL_CFG:base|SEED:42"
# CONFIGS[1]="GPU:2|LANG:sw|MODEL_CFG:base|SEED:123"
# CONFIGS[2]="GPU:2|LANG:sw|MODEL_CFG:base|SEED:456"
CONFIGS[1]="GPU:1|LANG:sw|MODEL_CFG:sft_filter_full|SEED:42"
CONFIGS[2]="GPU:1|LANG:sw|MODEL_CFG:sft_filter_full|SEED:123"
CONFIGS[3]="GPU:1|LANG:sw|MODEL_CFG:sft_filter_full|SEED:456"
CONFIGS[4]="GPU:1|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:42"
CONFIGS[5]="GPU:1|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:123"
CONFIGS[6]="GPU:1|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:456"

LOG_DIR="${PROJECT_ROOT}/outputs-self_translate_questions/logs"
mkdir -p "$LOG_DIR"

SUBSET_ARGS=""
SUBSET_TAG=""
if [ -n "$SUBSET" ]; then
    SUBSET_ARGS="--subset $SUBSET"
    SUBSET_TAG="_${SUBSET}"
fi

echo "Evaluating dataset: ${DATASET} (subset: ${SUBSET:-full})"

for config in "${CONFIGS[@]}"; do
    GPU=$(echo "$config" | grep -oP 'GPU:\K[^|]+')
    LANG=$(echo "$config" | grep -oP 'LANG:\K[^|]+')
    MODEL_CFG_NAME=$(echo "$config" | grep -oP 'MODEL_CFG:\K[^|]+')
    SEED=$(echo "$config" | grep -oP 'SEED:\K[^|]+')

    log_file="${LOG_DIR}/eval_bleu_${DATASET}${SUBSET_TAG}_${LANG}-en_${MODEL_CFG_NAME}_seed${SEED}.log"
    out_file="${PROJECT_ROOT}/outputs-self_translate_questions/eval_bleu_${DATASET}${SUBSET_TAG}_${LANG}-en_${MODEL_CFG_NAME}_seed${SEED}.json"
    job_name="${LANG}|${MODEL_CFG_NAME}|gpu${GPU}|seed${SEED}"

    # if [ -f "$out_file" ]; then
    #     echo "[${job_name}] already evaluated. Skipping."
    #     continue
    # fi

    echo "[${job_name}] Logging to ${log_file}"
    CUDA_VISIBLE_DEVICES="$GPU" python3 "${PROJECT_ROOT}/evaluation/run_bleu.py" \
        --dataset "$DATASET" \
        --model_cfg_name "${MODEL_CFG_NAME}" \
        --language "$LANG" \
        --seed "$SEED" \
        $EXTRA_ARGS \
        $SUBSET_ARGS \
        > "$log_file" 2>&1

    echo "[${job_name}] completed."
done

echo "Evaluation complete. Results saved in outputs-self_translate_questions/."

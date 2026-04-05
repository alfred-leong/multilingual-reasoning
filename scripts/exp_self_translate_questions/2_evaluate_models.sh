#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# source /home/alfred/miniconda3/etc/profile.d/conda.sh
# conda activate ml-delta

DATASET="mmmlu"
SUBSET="100"
MODEL_VARIANT="qwen3-1_7b"

# Define 3 configurations: GPU, LANG, MODEL_CFG_NAME
declare -a CONFIGS
CONFIGS[0]="GPU:6|LANG:sw|MODEL_CFG:base|SEED:42"
# CONFIGS[1]="GPU:7|LANG:sw|MODEL_CFG:sft_filter_full|SEED:42"
# CONFIGS[2]="GPU:6|LANG:sw|MODEL_CFG:sft_filter_full|SEED:123"
# CONFIGS[3]="GPU:6|LANG:sw|MODEL_CFG:sft_filter_full|SEED:456"
# CONFIGS[4]="GPU:6|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:42"
# CONFIGS[5]="GPU:6|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:123"
# CONFIGS[6]="GPU:6|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:456"

LOG_DIR="${PROJECT_ROOT}/outputs-self_translate_questions/logs"
mkdir -p "$LOG_DIR"

SUBSET_ARGS=""
SUBSET_TAG=""
if [ -n "$SUBSET" ]; then
    SUBSET_ARGS="--subset $SUBSET"
    SUBSET_TAG="_${SUBSET}"
fi

echo "Evaluating dataset: ${DATASET} (subset: ${SUBSET:-full})"

declare -a PIDS
declare -a JOB_NAMES
FAIL=0

for config in "${CONFIGS[@]}"; do
    GPU=$(echo "$config" | grep -oP 'GPU:\K[^|]+')
    LANG=$(echo "$config" | grep -oP 'LANG:\K[^|]+')
    MODEL_CFG_NAME=$(echo "$config" | grep -oP 'MODEL_CFG:\K[^|]+')
    SEED=$(echo "$config" | grep -oP 'SEED:\K[^|]+')

    MODEL_PATH="/external1/alfred/ml-reasoning-shared/exp_v3/ml-reasoning-exp_v3/${MODEL_CFG_NAME}_${DATASET}_${MODEL_VARIANT}_${LANG}_seed${SEED}"
    if [ "${MODEL_CFG_NAME}" == "base" ]; then
        MODEL_PATH=""
    else
        if [ ! -d "$MODEL_PATH" ]; then
            echo "Model path ${MODEL_PATH} does not exist. Skipping ${LANG} with config ${MODEL_CFG_NAME}."
            exit 1
        fi
    fi
    echo "Model path: ${MODEL_PATH}"

    log_file="${LOG_DIR}/eval_${DATASET}${SUBSET_TAG}_${LANG}-en_${MODEL_CFG_NAME}_seed${SEED}.log"
    out_file="${PROJECT_ROOT}/outputs-self_translate_questions/eval_${DATASET}${SUBSET_TAG}_${LANG}-en_${MODEL_CFG_NAME}_seed${SEED}.json"
    job_name="${LANG}|${MODEL_CFG_NAME}|gpu${GPU}|seed${SEED}"

    if [ -f "$out_file" ]; then
        echo "[${job_name}] already evaluated. Skipping."
        continue
    fi

    echo "[${job_name}] Logging to ${log_file}"
    CUDA_VISIBLE_DEVICES="$GPU" python3 "${PROJECT_ROOT}/evaluation/run_eval_translated_questions.py" \
        --dataset "$DATASET" \
        --model_path "${MODEL_PATH}" \
        --model_cfg_name "${MODEL_CFG_NAME}" \
        --language "$LANG" \
        --seed "$SEED" \
        $EXTRA_ARGS \
        $SUBSET_ARGS \
        > "$log_file" 2>&1 &

    PIDS+=("$!")
    JOB_NAMES+=("$job_name")
done

echo "Waiting for all evaluation processes to complete..."
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    job_name="${JOB_NAMES[$i]}"
    if ! wait "$pid"; then
        echo "[${job_name}] failed."
        FAIL=1
    else
        echo "[${job_name}] completed."
    fi
done

if [[ $FAIL -eq 1 ]]; then
    echo "Evaluation encountered an error. Check logs in ${LOG_DIR}"
    exit 1
fi

echo "Evaluation complete. Results saved in outputs-self_translate_questions/."

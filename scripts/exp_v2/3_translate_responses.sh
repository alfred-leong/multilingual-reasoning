#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /home/alfred/miniconda3/etc/profile.d/conda.sh
conda activate ml-delta

# Settings
MODEL="google/translategemma-4b-it"
PORTS=(8401 8402 8403)
GPUS=(5 6 7)
LANGS=("ja" "bn" "sw")
PORT_WAIT=600
MAX_MODEL_LEN=8192
GPU_UTIL=0.85
LOG_DIR="${PROJECT_ROOT}/outputs-exp_v2/logs"
PID_FILE="/tmp/vllm_translate_gen_pids.txt"

mkdir -p "$LOG_DIR"
> "$PID_FILE"

stop_vllm() {
    if [[ -f "$PID_FILE" ]]; then
        echo "Stopping vLLM servers..."
        xargs kill < "$PID_FILE" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "vLLM servers stopped."
    fi
}
trap stop_vllm EXIT

echo "Starting 3 vLLM servers ($MODEL) on GPUs 5, 6, 7"
for i in {0..2}; do
    log_file="${LOG_DIR}/vllm_trans_gen_${LANGS[$i]}.log"
    echo "  Starting for ${LANGS[$i]} on GPU ${GPUS[$i]}, port ${PORTS[$i]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python "${PROJECT_ROOT}/scripts/launch_vllm_patched.py" "$MODEL" \
        --host 127.0.0.1 \
        --port "${PORTS[$i]}" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code \
        > "$log_file" 2>&1 &
    
    echo "$!" >> "$PID_FILE"
done

# Wait for servers
for i in {0..2}; do
    elapsed=0
    echo -n "Waiting for vLLM ${LANGS[$i]} on port ${PORTS[$i]}..."
    while ! curl -sf "http://localhost:${PORTS[$i]}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
        if [[ $elapsed -ge $PORT_WAIT ]]; then
            echo " ERROR: vLLM did not become healthy within ${PORT_WAIT}s. Check logs."
            exit 1
        fi
    done
    echo " ready! (${elapsed}s)"
done

# Run generation for MGSM and MMMLU concurrently across languages
# echo "Translating MGSM & MMMLU responses..."
echo "Translating MGSM responses..."
TRANS_PIDS=()
for i in {0..2}; do
    lang="${LANGS[$i]}"
    port="${PORTS[$i]}"
    
    # mgsm
    python3 "${PROJECT_ROOT}/data/translate_responses-exp_v2.py" --dataset mgsm --language "$lang" --port "$port" --workers 3 \
        > "${LOG_DIR}/trans_mgsm_${lang}.log" 2>&1 &
    TRANS_PIDS+=($!)
    
    # # mmmlu
    # python3 "${PROJECT_ROOT}/data/translate_responses-exp_v2.py" --dataset mmmlu --language "$lang" --port "$port" --workers 3 \
    #     > "${LOG_DIR}/trans_mmmlu_${lang}.log" 2>&1 &
    # TRANS_PIDS+=($!)
done

FAIL=0
for pid in "${TRANS_PIDS[@]}"; do
    if ! wait "$pid"; then
        FAIL=1
    fi
done

if [[ $FAIL -eq 1 ]]; then
    echo "Translation workers failed."
    exit 1
fi

# Check outputs
echo "Checking MGSM translated responses:"
for lang in "${LANGS[@]}"; do
    python3 "${PROJECT_ROOT}/data/check_translated_responses-exp_v2.py" --dataset mgsm --language "$lang"
    # python3 "${PROJECT_ROOT}/data/check_translated_responses-exp_v2.py" --dataset mmmlu --language "$lang"
done

echo "Translation complete. Stopping vLLM..."
# Trap will stop vLLM

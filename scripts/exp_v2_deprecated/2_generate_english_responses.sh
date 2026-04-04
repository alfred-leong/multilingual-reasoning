#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /home/alfred/miniconda3/etc/profile.d/conda.sh
conda activate ml-delta

# Settings
MODEL="Qwen/Qwen3-1.7B"
PORT=8300
PORT_WAIT=300
GPU=5
MAX_MODEL_LEN=32768
GPU_UTIL=0.85
LOG_DIR="${PROJECT_ROOT}/outputs-exp_v3/logs"
PID_FILE="/tmp/vllm_english_gen_pid.txt"

mkdir -p "$LOG_DIR"

stop_vllm() {
    if [[ -f "$PID_FILE" ]]; then
        echo "Stopping vLLM server..."
        xargs kill < "$PID_FILE" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "vLLM server stopped."
    fi
}
trap stop_vllm EXIT

echo "Starting vLLM ($MODEL) on GPU $GPU, port $PORT"
log_file="${LOG_DIR}/vllm_english_gen.log"

CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    > "$log_file" 2>&1 &
echo "$!" > "$PID_FILE"

# Wait for server
elapsed=0
echo -n "Waiting for vLLM..."
while ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
    elapsed=$((elapsed + 5))
    echo -n "."
    if [[ $elapsed -ge $PORT_WAIT ]]; then
        echo " ERROR: vLLM did not become healthy within ${PORT_WAIT}s. Check $log_file"
        exit 1
    fi
done
echo " ready! (${elapsed}s)"

# Run generation for MGSM and MMMLU
echo "Generating English responses for MGSM..."
python3 "${PROJECT_ROOT}/data/generate_english_responses-exp_v2.py" --dataset mgsm --port "$PORT" --workers 4

echo "Generating English responses for MMMLU..."
python3 "${PROJECT_ROOT}/data/generate_english_responses-exp_v2.py" --dataset mmmlu --port "$PORT" --workers 4

# Check outputs
echo "Checking MGSM responses:"
python3 "${PROJECT_ROOT}/data/check_english_responses-exp_v2.py" --dataset mgsm

echo "Checking MMMLU responses:"
python3 "${PROJECT_ROOT}/data/check_english_responses-exp_v2.py" --dataset mmmlu

echo "English response generation complete. Stopping vLLM..."
# Trap will stop vLLM

#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# source /home/alfred/miniconda3/etc/profile.d/conda.sh
# conda activate ml-delta

# Settings
MODEL="Qwen/Qwen3-1.7B"
PORT_WAIT=300
# MAX_MODEL_LEN=32768
MAX_MODEL_LEN=20000
GPU_UTIL=0.85
LOG_DIR="${PROJECT_ROOT}/outputs-self_translate_questions/logs"
NUM_WORKERS=1

DATASET="mmmlu"
SUBSET="100"
MODEL_VARIANT="qwen3-1_7b"

# Define 3 configurations: GPU, PORT, LANG, MODEL_CFG_NAME
declare -a CONFIGS
CONFIGS[0]="GPU:6|PORT:8501|LANG:sw|MODEL_CFG:sft_filter_full|SEED:42"
CONFIGS[1]="GPU:6|PORT:8501|LANG:sw|MODEL_CFG:sft_filter_full|SEED:123"
CONFIGS[2]="GPU:6|PORT:8501|LANG:sw|MODEL_CFG:sft_filter_full|SEED:456"
CONFIGS[3]="GPU:6|PORT:8501|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:42"
CONFIGS[4]="GPU:6|PORT:8501|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:123"
CONFIGS[5]="GPU:6|PORT:8501|LANG:sw|MODEL_CFG:sft_no-filter_full|SEED:456"

mkdir -p "$LOG_DIR"

declare -a PID_FILES
declare -a NATIVE_PIDS

stop_vllm() {
    VLLM_PID=$1
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        # Give it a moment to shut down gracefully
        sleep 2
        # Force kill if still alive
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}

# Process each configuration sequentially
for config in "${CONFIGS[@]}"; do
    # Parse config
    GPU=$(echo "$config" | grep -oP 'GPU:\K[^|]+')
    PORT=$(echo "$config" | grep -oP 'PORT:\K[^|]+')
    LANG=$(echo "$config" | grep -oP 'LANG:\K[^|]+')
    MODEL_CFG_NAME=$(echo "$config" | grep -oP 'MODEL_CFG:\K[^|]+')
    SEED=$(echo "$config" | grep -oP 'SEED:\K[^|]+')

    MODEL_PATH="/external1/alfred/ml-reasoning-shared/exp_v3/ml-reasoning-exp_v3/${MODEL_CFG_NAME}_${DATASET}_${MODEL_VARIANT}_${LANG}_seed${SEED}"
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model path $MODEL_PATH does not exist. Skipping configuration: GPU=$GPU, PORT=$PORT, LANG=$LANG, MODEL_CFG=$MODEL_CFG_NAME"
        exit 1
    fi
    
    echo "=========================================="
    echo "Starting configuration: GPU=$GPU, PORT=$PORT, LANG=$LANG, MODEL_CFG=$MODEL_CFG_NAME, SEED=$SEED"
    echo "=========================================="
    
    # Start vLLM server
    log_file="${LOG_DIR}/vllm_${LANG}_${MODEL_CFG_NAME}_seed${SEED}.log"
    echo "Logging vLLM output to ${log_file}"
    
    if [[ "$MODEL_CFG_NAME" == *"lora"* ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --dtype bfloat16 \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --trust-remote-code \
            --enable-lora \
            --lora-modules lora=${MODEL_PATH} \
            > "$log_file" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL_PATH" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --dtype bfloat16 \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --trust-remote-code \
            > "$log_file" 2>&1 &
    fi
    VLLM_PID=$!
    echo "vLLM server started (PID: $VLLM_PID)"

    # Wait for server to be healthy
    elapsed=0
    echo -n "Waiting for vLLM server on port $PORT..."
    while ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
        if [[ $elapsed -ge $PORT_WAIT ]]; then
            echo " ERROR: vLLM on port $PORT did not become healthy within ${PORT_WAIT}s. Check $log_file"
            stop_vllm "$VLLM_PID"
            exit 1
        fi
    done
    echo " ready! (${elapsed}s)"

    # Run data generation for this config
    echo "Self-generating ${LANG}-en questions for ${DATASET}..."
    LOG_PATH="${LOG_DIR}/self_generate_${LANG}-en_questions_${DATASET}_${MODEL_CFG_NAME}_${MODEL_VARIANT}_seed${SEED}.log"
    echo "Logging to ${LOG_PATH}"
    
    python3 "${PROJECT_ROOT}/data/self_generate_en_questions.py" \
        --dataset "$DATASET" \
        --subset "$SUBSET" \
        --lang "$LANG" \
        --port "$PORT" \
        --model_path "$MODEL_PATH" \
        --model_cfg_name "$MODEL_CFG_NAME" \
        --seed "$SEED" \
        --workers $NUM_WORKERS > "$LOG_PATH" 2>&1
    
    GEN_EXIT_CODE=$?
    
    # Stop vLLM server
    stop_vllm "$VLLM_PID"
    
    if [[ $GEN_EXIT_CODE -ne 0 ]]; then
        echo "ERROR: Data generation failed for configuration: GPU=$GPU, PORT=$PORT, LANG=$LANG, MODEL_CFG=$MODEL_CFG_NAME, SEED=$SEED"
        exit 1
    fi
    
    echo "Configuration completed: GPU=$GPU, PORT=$PORT, LANG=$LANG, MODEL_CFG=$MODEL_CFG_NAME, SEED=$SEED"
    echo ""
done

echo "=========================================="
echo "All configurations completed successfully!"
echo "=========================================="

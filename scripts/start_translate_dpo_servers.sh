#!/usr/bin/env bash
# =============================================================================
# start_translate_dpo_servers.sh
#
# Starts vLLM servers for the Translate-DPO experiment.
#
# Parallel mode (3 GPUs, one per language):
#   3 × Qwen3-1.7B     (generation)   → ports 8301-8303 on GPUs 1-3
#   3 × TranslateGemma  (translation)  → ports 8401-8403 on GPUs 1-3
#
# Usage:
#   bash scripts/start_translate_dpo_servers.sh gen    # start generation servers
#   bash scripts/start_translate_dpo_servers.sh trans  # start translation servers
#
# Environment:
#   VLLM_PID_FILE   Path to write server PIDs
#                   (default: /tmp/vllm_pids_translate_dpo.txt)
#
# Teardown:
#   xargs kill < /tmp/vllm_pids_translate_dpo.txt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/configs/translate_dpo_config.yaml"

MODE="${1:-gen}"  # gen or trans

LANGUAGES=("JA_JP" "BN_BD" "SW_KE")
GPUS=(1 2 3)
GEN_PORTS=(8301 8302 8303)
TRANS_PORTS=(8401 8402 8403)

# Models — must match configs/translate_dpo_config.yaml
GEN_MODEL="Qwen/Qwen3-1.7B"
TRANS_MODEL_HF="google/translategemma-4b-it"

# Ports — based on mode
if [[ "${MODE}" == "gen" ]]; then
    PORTS=("${GEN_PORTS[@]}")
elif [[ "${MODE}" == "trans" ]]; then
    PORTS=("${TRANS_PORTS[@]}")
else
    echo "ERROR: Unknown mode '${MODE}'. Use 'gen' or 'trans'."
    exit 1
fi

# Resource allocation
GPU_UTIL=0.85
MAX_MODEL_LEN=8192

LOG_DIR="${PROJECT_ROOT}/outputs-translate_dpo/logs"
mkdir -p "${LOG_DIR}"

PID_FILE="${VLLM_PID_FILE:-/tmp/vllm_pids_translate_dpo.txt}"
> "${PID_FILE}"

echo "============================================================"
echo "  Translate-DPO vLLM Servers (mode: ${MODE})"
for i in 0 1 2; do
    if [[ "${MODE}" == "gen" ]]; then
        echo "  ${LANGUAGES[$i]}: ${GEN_MODEL} → GPU ${GPUS[$i]}, port ${GEN_PORTS[$i]}"
    else
        echo "  ${LANGUAGES[$i]}: ${TRANS_MODEL_HF} → GPU ${GPUS[$i]}, port ${TRANS_PORTS[$i]}"
    fi
done
echo "  PID file   : ${PID_FILE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# start_vllm <model> <port> <gpu> <label>
# Standard vLLM launcher (for models without rope_parameters issues)
# ---------------------------------------------------------------------------
start_vllm() {
    local model="$1"
    local port="$2"
    local gpu="$3"
    local label="$4"
    local log_file="${LOG_DIR}/vllm_${label}_port${port}.log"

    echo "Starting ${label} on GPU ${gpu}, port ${port} → ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${model}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --dtype bfloat16 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --trust-remote-code \
        > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" >> "${PID_FILE}"
    echo "  PID=${pid}"
}

# ---------------------------------------------------------------------------
# start_vllm_patched <model> <port> <gpu> <label>
# Uses the monkey-patched launcher to fix the rope_parameters bug for
# Gemma3-family models (TranslateGemma) in vLLM 0.16.0.
# ---------------------------------------------------------------------------
start_vllm_patched() {
    local model="$1"
    local port="$2"
    local gpu="$3"
    local label="$4"
    local log_file="${LOG_DIR}/vllm_${label}_port${port}.log"

    echo "Starting ${label} (patched) on GPU ${gpu}, port ${port} → ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT_DIR}/launch_vllm_patched.py" \
        "${model}" \
        --host 127.0.0.1 \
        --port "${port}" \
        --dtype bfloat16 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --trust-remote-code \
        > "${log_file}" 2>&1 &

    local pid=$!
    echo "${pid}" >> "${PID_FILE}"
    echo "  PID=${pid}"
}

# ---------------------------------------------------------------------------
# wait_for_server <port> <label> <timeout_secs>
# ---------------------------------------------------------------------------
wait_for_server() {
    local port="$1"
    local label="$2"
    local timeout="${3:-300}"
    local elapsed=0

    echo -n "Waiting for ${label} (port ${port}) "
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
        if [[ ${elapsed} -ge ${timeout} ]]; then
            echo ""
            echo "ERROR: ${label} did not become healthy within ${timeout}s."
            echo "Check log: ${LOG_DIR}/vllm_${label}_port${port}.log"
            exit 1
        fi
    done
    echo " ready! (${elapsed}s)"
}

# ---------------------------------------------------------------------------
# Launch servers
# ---------------------------------------------------------------------------

for i in 0 1 2; do
    if [[ "${MODE}" == "gen" ]]; then
        start_vllm "${GEN_MODEL}" "${GEN_PORTS[$i]}" "${GPUS[$i]}" \
            "gen_${LANGUAGES[$i]}"
    else
        start_vllm_patched "${TRANS_MODEL_HF}" "${TRANS_PORTS[$i]}" "${GPUS[$i]}" \
            "trans_${LANGUAGES[$i]}"
    fi
done

for i in 0 1 2; do
    wait_for_server "${PORTS[$i]}" "${MODE}_${LANGUAGES[$i]}" 300
done

echo ""
echo "============================================================"
echo "  All ${MODE} servers are ready."
for i in 0 1 2; do
    echo "  ${LANGUAGES[$i]} → http://localhost:${PORTS[$i]}/v1"
done
echo "  To stop:  xargs kill < ${PID_FILE}"
echo "============================================================"

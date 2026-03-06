#!/usr/bin/env bash
# =============================================================================
# run_translate_dpo_experiment.sh
#
# End-to-end pipeline for the Translate-DPO experiment.
#
# Pipeline:
#   Phase 0: Start 3 vLLM generation servers (one per language/GPU)
#   Phase 1: Generate English + Native responses for MMMLU (parallel)
#   Phase 0': Stop generation servers
#   Phase 2a: Start 3 vLLM TranslateGemma servers (1 per GPU on GPUs 1,2,3)
#   Phase 2b: Translate English responses (3 workers per language, same server)
#   Phase 2c: Merge translation shards + stop vLLM servers
#   Phase 3: DPO training (parallel, one per GPU)
#   Phase 4: MGSM evaluation (parallel, one per GPU)
#   Phase 5: Summary report
#
# GPU layout:
#   GPU 1 — JA_JP (generation → translation server → training → eval)
#   GPU 2 — BN_BD (generation → translation server → training → eval)
#   GPU 3 — SW_KE (generation → translation server → training → eval)
#
# Usage:
#   bash scripts/run_translate_dpo_experiment.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/configs/translate_dpo_config.yaml"

LANGUAGES=("JA_JP" "BN_BD" "SW_KE")
GPUS=(1 2 3)
GEN_PORTS=(8301 8302 8303)
TRANS_PORTS=(8401 8402 8403)

# Models — must match configs/translate_dpo_config.yaml
GEN_MODEL="Qwen/Qwen3-1.7B"
TRANS_MODEL_HF="google/translategemma-4b-it"

# Translation parallelism: 1 vLLM server per GPU, N workers per language
# querying the same server.  vLLM batches concurrent requests internally.
TRANS_SHARDS=3
TRANS_GPU_UTIL=0.85    # single server per GPU → can use most of the memory

# Resource allocation (generation / training / eval)
GPU_UTIL=0.85
MAX_MODEL_LEN=8192

VLLM_PID_FILE="/tmp/vllm_pids_translate_dpo.txt"

LOG_DIR="${PROJECT_ROOT}/outputs-translate_dpo/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Translate-DPO Experiment Pipeline (Parallel)"
echo "  Config       : ${CONFIG}"
echo "  GPU 1        : ${LANGUAGES[0]}  (gen :${GEN_PORTS[0]})"
echo "  GPU 2        : ${LANGUAGES[1]}  (gen :${GEN_PORTS[1]})"
echo "  GPU 3        : ${LANGUAGES[2]}  (gen :${GEN_PORTS[2]})"
echo "  Translation  : 1 vLLM server per GPU, ${TRANS_SHARDS} workers per language"
echo "                 ports ${TRANS_PORTS[0]}, ${TRANS_PORTS[1]}, ${TRANS_PORTS[2]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

stop_vllm_servers() {
    if [[ -f "${VLLM_PID_FILE}" ]]; then
        echo "Stopping vLLM servers …"
        xargs kill < "${VLLM_PID_FILE}" 2>/dev/null || true
        rm -f "${VLLM_PID_FILE}"
        echo "vLLM servers stopped."
    fi
}

trap stop_vllm_servers EXIT

# start_vllm <model> <port> <gpu> <label>
start_vllm() {
    local model="$1" port="$2" gpu="$3" label="$4"
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
    echo "$!" >> "${VLLM_PID_FILE}"
    echo "  PID=$!"
}

# start_vllm_patched <model> <port> <gpu> <label>
start_vllm_patched() {
    local model="$1" port="$2" gpu="$3" label="$4"
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
    echo "$!" >> "${VLLM_PID_FILE}"
    echo "  PID=$!"
}

# wait_for_server <port> <label> <timeout_secs>
wait_for_server() {
    local port="$1" label="$2" timeout="${3:-300}" elapsed=0
    echo -n "Waiting for ${label} (port ${port}) "
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5)); echo -n "."
        if [[ ${elapsed} -ge ${timeout} ]]; then
            echo ""; echo "ERROR: ${label} did not become healthy within ${timeout}s."
            echo "Check log: ${LOG_DIR}/vllm_${label}_port${port}.log"
            exit 1
        fi
    done
    echo " ready! (${elapsed}s)"
}

# # ---------------------------------------------------------------------------
# # Phase 0: Start 3 vLLM generation servers (one per language)
# # ---------------------------------------------------------------------------
# echo ""
# echo "===== Phase 0: Starting 3 generation vLLM servers ====="
# > "${VLLM_PID_FILE}"
# for i in 0 1 2; do
#     start_vllm "${GEN_MODEL}" "${GEN_PORTS[$i]}" "${GPUS[$i]}" \
#         "gen_${LANGUAGES[$i]}"
# done
# for i in 0 1 2; do
#     wait_for_server "${GEN_PORTS[$i]}" "gen_${LANGUAGES[$i]}" 300
# done

# # ---------------------------------------------------------------------------
# # Phase 1: Generate responses (all languages in parallel)
# # ---------------------------------------------------------------------------
# echo ""
# echo "===== Phase 1: Generating responses (Setting 1 + Setting 2) — parallel ====="
# GEN_PIDS=()
# for i in 0 1 2; do
#     LANG="${LANGUAGES[$i]}"
#     LOG="${LOG_DIR}/${LANG}_generate.log"
#     echo "  [${LANG}] Generating on port ${GEN_PORTS[$i]} …"
#     python "${PROJECT_ROOT}/data/generate_translate_dpo_responses.py" \
#         --language "${LANG}" \
#         --config "${CONFIG}" \
#         --port "${GEN_PORTS[$i]}" \
#         > "${LOG}" 2>&1 &
#     GEN_PIDS+=($!)
# done
# # Wait for all generation jobs
# FAIL=0
# for i in 0 1 2; do
#     if wait "${GEN_PIDS[$i]}"; then
#         echo "  [${LANGUAGES[$i]}] Done. Log: ${LOG_DIR}/${LANGUAGES[$i]}_generate.log"
#     else
#         echo "  [${LANGUAGES[$i]}] FAILED. Log: ${LOG_DIR}/${LANGUAGES[$i]}_generate.log"
#         FAIL=1
#     fi
# done
# [[ ${FAIL} -eq 1 ]] && { echo "Phase 1 failed."; exit 1; }
# echo "Phase 1 complete."

# # ---------------------------------------------------------------------------
# # Phase 0': Stop generation servers (free GPU memory)
# # ---------------------------------------------------------------------------
# echo ""
# echo "===== Phase 0': Stopping generation servers ====="
# stop_vllm_servers

# ---------------------------------------------------------------------------
# Phase 2a: Start 3 vLLM TranslateGemma servers (1 per GPU)
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 2a: Starting 3 vLLM TranslateGemma servers (1 per GPU) ====="
> "${VLLM_PID_FILE}"
for i in 0 1 2; do
    GPU="${GPUS[$i]}"
    LANG="${LANGUAGES[$i]}"
    PORT="${TRANS_PORTS[$i]}"
    LABEL="trans_${LANG}"
    LOG_FILE="${LOG_DIR}/vllm_${LABEL}_port${PORT}.log"
    echo "  Starting ${LABEL} on GPU ${GPU}, port ${PORT} → ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU}" python "${SCRIPT_DIR}/launch_vllm_patched.py" \
        "${TRANS_MODEL_HF}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --dtype bfloat16 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${TRANS_GPU_UTIL}" \
        --trust-remote-code \
        > "${LOG_FILE}" 2>&1 &
    echo "$!" >> "${VLLM_PID_FILE}"
done

# Wait for all translation servers to become healthy
for i in 0 1 2; do
    LANG="${LANGUAGES[$i]}"
    PORT="${TRANS_PORTS[$i]}"
    wait_for_server "${PORT}" "trans_${LANG}" 600
done
echo "Phase 2a complete.  3 vLLM servers running."

# ---------------------------------------------------------------------------
# Phase 2b: Translate English responses (3 workers per language)
#           All workers for one language query the same vLLM server.
#           Each worker processes a 1/TRANS_SHARDS slice of the dataset.
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 2b: Translating English responses — ${TRANS_SHARDS} workers per language ====="
TRANS_PIDS=()
TRANS_LABELS=()
for i in 0 1 2; do
    LANG="${LANGUAGES[$i]}"
    PORT="${TRANS_PORTS[$i]}"
    for s in $(seq 0 $((TRANS_SHARDS - 1))); do
        LOG="${LOG_DIR}/${LANG}_translate_shard${s}.log"
        echo "  [${LANG}] shard ${s}/${TRANS_SHARDS} → port ${PORT}"
        python "${PROJECT_ROOT}/data/translate_with_gemma.py" \
            --language "${LANG}" \
            --config "${CONFIG}" \
            --port "${PORT}" \
            --shard_id "${s}" \
            --num_shards "${TRANS_SHARDS}" \
            > "${LOG}" 2>&1 &
        TRANS_PIDS+=($!)
        TRANS_LABELS+=("${LANG}_shard${s}")
    done
done

# Wait for all translation workers (3 languages × TRANS_SHARDS)
FAIL=0
for idx in "${!TRANS_PIDS[@]}"; do
    if wait "${TRANS_PIDS[$idx]}"; then
        echo "  [${TRANS_LABELS[$idx]}] Done."
    else
        echo "  [${TRANS_LABELS[$idx]}] FAILED. Check ${LOG_DIR}/"
        FAIL=1
    fi
done
[[ ${FAIL} -eq 1 ]] && { echo "Phase 2b (translation) failed."; exit 1; }
echo "Phase 2b complete."

# ---------------------------------------------------------------------------
# Phase 2c: Merge translation shards + stop vLLM servers
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 2c: Merging shards & stopping vLLM servers ====="
for LANG in "${LANGUAGES[@]}"; do
    python "${PROJECT_ROOT}/data/translate_with_gemma.py" \
        --language "${LANG}" \
        --config "${CONFIG}" \
        --merge_shards \
        --num_shards "${TRANS_SHARDS}"
done
stop_vllm_servers
trap - EXIT
echo "Phase 2 complete."

# ---------------------------------------------------------------------------
# Phase 3: DPO training (all languages in parallel, one GPU each)
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 3: DPO training — parallel ====="
TRAIN_PIDS=()
for i in 0 1 2; do
    LANG="${LANGUAGES[$i]}"
    LOG="${LOG_DIR}/${LANG}_dpo_train.log"
    echo "  [${LANG}] Training DPO on GPU ${GPUS[$i]} …"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        python "${PROJECT_ROOT}/training/run_translate_dpo_training.py" \
            --language "${LANG}" \
            --config "${CONFIG}" \
            > "${LOG}" 2>&1 &
    TRAIN_PIDS+=($!)
done
FAIL=0
for i in 0 1 2; do
    if wait "${TRAIN_PIDS[$i]}"; then
        echo "  [${LANGUAGES[$i]}] Done. Log: ${LOG_DIR}/${LANGUAGES[$i]}_dpo_train.log"
    else
        echo "  [${LANGUAGES[$i]}] FAILED. Log: ${LOG_DIR}/${LANGUAGES[$i]}_dpo_train.log"
        FAIL=1
    fi
done
[[ ${FAIL} -eq 1 ]] && { echo "Phase 3 failed."; exit 1; }
echo "Phase 3 complete."

# ---------------------------------------------------------------------------
# Phase 4: MGSM evaluation (all languages in parallel, one GPU each)
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 4: MGSM evaluation — parallel ====="
EVAL_PIDS=()
for i in 0 1 2; do
    LANG="${LANGUAGES[$i]}"
    (
        for MODEL_TYPE in base dpo; do
            LOG="${LOG_DIR}/${LANG}_eval_${MODEL_TYPE}.log"
            echo "  [${LANG}/${MODEL_TYPE}] Evaluating on GPU ${GPUS[$i]} …"
            CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
                python "${PROJECT_ROOT}/evaluation/run_translate_dpo_eval.py" \
                    --language "${LANG}" \
                    --model_type "${MODEL_TYPE}" \
                    --config "${CONFIG}" \
                    > "${LOG}" 2>&1
            echo "  [${LANG}/${MODEL_TYPE}] Done. Log: ${LOG}"
        done
    ) &
    EVAL_PIDS+=($!)
done
FAIL=0
for i in 0 1 2; do
    if wait "${EVAL_PIDS[$i]}"; then
        echo "  [${LANGUAGES[$i]}] Eval complete."
    else
        echo "  [${LANGUAGES[$i]}] Eval FAILED."
        FAIL=1
    fi
done
[[ ${FAIL} -eq 1 ]] && { echo "Phase 4 failed."; exit 1; }
echo "Phase 4 complete."

# ---------------------------------------------------------------------------
# Phase 5: Summary
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 5: Results Summary ====="

OUTPUT_DIR="${PROJECT_ROOT}/outputs-translate_dpo"

echo ""
printf "%-8s  %-6s  %8s  %12s  %12s  %14s  %14s\n" \
    "Lang" "Model" "Accuracy" "Full Native" "Full English" "Think Native" "Think English"
printf "%s\n" "$(printf '=%.0s' {1..90})"

for LANG in "${LANGUAGES[@]}"; do
    for MODEL_TYPE in base dpo; do
        EVAL_FILE="${OUTPUT_DIR}/eval_${LANG}_${MODEL_TYPE}.json"
        if [[ -f "${EVAL_FILE}" ]]; then
            ACC=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['accuracy']:.2%}\")")
            FT=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_full_target_ratio']:.3f}\")")
            FE=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_full_english_ratio']:.3f}\")")
            TT=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_think_target_ratio']:.3f}\")")
            TE=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_think_english_ratio']:.3f}\")")
            printf "%-8s  %-6s  %8s  %12s  %12s  %14s  %14s\n" \
                "${LANG}" "${MODEL_TYPE}" "${ACC}" "${FT}" "${FE}" "${TT}" "${TE}"
        else
            printf "%-8s  %-6s  %8s\n" "${LANG}" "${MODEL_TYPE}" "(missing)"
        fi
    done
done

echo ""
echo "============================================================"
echo "  Translate-DPO experiment complete."
echo "  Outputs : ${OUTPUT_DIR}/"
echo "  Logs    : ${LOG_DIR}/"
echo "============================================================"

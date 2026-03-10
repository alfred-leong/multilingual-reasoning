#!/usr/bin/env bash
# =============================================================================
# run_variance_eval.sh
#
# Run evaluation 3 times with sampling (temperature=0.6, top_p=0.95)
# for base, dpo, and dpo_removethink+filter to measure variance.
#
# Uses a FIFO-based GPU pool to schedule jobs across 2 GPUs (6, 7).
# At most 2 experiments run concurrently. Each GPU works independently —
# when it finishes a job, it immediately picks up the next queued one
# without waiting for the other GPU.
# Order: SW_KE → JA_JP → BN_BD
# =============================================================================
set -euo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ml-delta

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/translate_dpo_config.yaml"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluation/run_translate_dpo_eval.py"
OUTPUT_DIR="${PROJECT_ROOT}/outputs-translate_dpo"
LOG_DIR="${OUTPUT_DIR}/logs"

# Sampling settings
TEMPERATURE=0.6
TOP_P=0.95

# GPU pool (max 2 concurrent jobs)
GPUS=(6 7)

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# FIFO-based GPU pool: each GPU ID is a token in the pipe.
# A job reads (acquires) a token before running and writes it back (releases)
# when done. This naturally limits concurrency to ${#GPUS[@]}.
# ---------------------------------------------------------------------------
GPU_FIFO=$(mktemp -u /tmp/gpu_pool.XXXXXX)
mkfifo "${GPU_FIFO}"
exec 3<>"${GPU_FIFO}"
rm "${GPU_FIFO}"          # unlink; fd 3 keeps it alive

for gpu in "${GPUS[@]}"; do echo "$gpu" >&3; done

# run_job LABEL LOGFILE CMD [ARGS...]
#   Launches a background subshell that:
#     1. Blocks until a GPU token is available (read from fd 3)
#     2. Runs the command on that GPU
#     3. Returns the GPU token to the pool
#   Sets LAUNCHED_PID to the subshell PID so the caller can wait on it.
run_job() {
    local label="$1" logfile="$2"; shift 2
    (
        local gpu
        read -r gpu <&3            # acquire GPU (blocks if none free)
        echo "  [${label}] Started on GPU ${gpu}"
        if CUDA_VISIBLE_DEVICES=${gpu} "$@" > "${logfile}" 2>&1; then
            echo "  [${label}] ✓ Complete (GPU ${gpu})"
        else
            echo "  [${label}] ✗ FAILED (GPU ${gpu}) — see ${logfile}"
            echo "${gpu}" >&3      # release GPU
            exit 1
        fi
        echo "${gpu}" >&3          # release GPU
    ) &
    LAUNCHED_PID=$!
}

LANGUAGES=("SW_KE" "JA_JP" "BN_BD")

SFT_OUTPUT_DIR="${PROJECT_ROOT}/outputs-translate_sft"

for LANG in "${LANGUAGES[@]}"; do
    ADAPTER_DPO="${OUTPUT_DIR}/dpo_${LANG}"
    ADAPTER_RTF="${OUTPUT_DIR}/dpo_${LANG}_removethink+filter"
    ADAPTER_RTF_FULL="${OUTPUT_DIR}/dpo_${LANG}_removethink+filter_full"
    ADAPTER_SFT_M1="${SFT_OUTPUT_DIR}/sft_${LANG}_mode1"
    ADAPTER_SFT_M2="${SFT_OUTPUT_DIR}/sft_${LANG}_mode2"
    ADAPTER_SFT_M1_FULL="${SFT_OUTPUT_DIR}/sft_${LANG}_mode1_full"
    ADAPTER_SFT_M2_FULL="${SFT_OUTPUT_DIR}/sft_${LANG}_mode2_full"

    echo ""
    echo "============================================================"
    echo "  ${LANG}: base → dpo → dpo_rt+f → sft_m1 → sft_m2 → dpo_rt+f_full → sft_m1_full → sft_m2_full"
    echo "  GPU pool: ${GPUS[*]}  (max ${#GPUS[@]} concurrent, 3 runs per mode)"
    echo "============================================================"

    FAIL=0
    ALL_PIDS=()

    # # ---- base ----
    # echo "  [${LANG}/base] Launching runs 1-3 …"
    # PIDS=()
    # for RUN in 1 2 3; do
    #     run_job "${LANG}/base/run-${RUN}" \
    #         "${LOG_DIR}/${LANG}_eval_base_run-${RUN}.log" \
    #         python "${EVAL_SCRIPT}" \
    #             --language "${LANG}" \
    #             --model_type base \
    #             --config "${CONFIG}" \
    #             --do_sample \
    #             --temperature "${TEMPERATURE}" \
    #             --top_p "${TOP_P}" \
    #             --output_suffix "_run-${RUN}"
    #     PIDS+=("${LAUNCHED_PID}")
    # done
    # for PID in "${PIDS[@]}"; do wait "${PID}" || FAIL=1; done

    # # ---- dpo ----
    # echo "  [${LANG}/dpo] Launching runs 1-3 …"
    # PIDS=()
    # for RUN in 1 2 3; do
    #     run_job "${LANG}/dpo/run-${RUN}" \
    #         "${LOG_DIR}/${LANG}_eval_dpo_run-${RUN}.log" \
    #         python "${EVAL_SCRIPT}" \
    #             --language "${LANG}" \
    #             --model_type dpo \
    #             --config "${CONFIG}" \
    #             --adapter_dir "${ADAPTER_DPO}" \
    #             --do_sample \
    #             --temperature "${TEMPERATURE}" \
    #             --top_p "${TOP_P}" \
    #             --output_suffix "_run-${RUN}"
    #     PIDS+=("${LAUNCHED_PID}")
    # done
    # for PID in "${PIDS[@]}"; do wait "${PID}" || FAIL=1; done

    # # ---- dpo_removethink+filter ----
    # echo "  [${LANG}/dpo_rt+f] Launching runs 1-3 …"
    # PIDS=()
    # for RUN in 1 2 3; do
    #     run_job "${LANG}/dpo_rt+f/run-${RUN}" \
    #         "${LOG_DIR}/${LANG}_eval_dpo_removethink+filter_run-${RUN}.log" \
    #         python "${EVAL_SCRIPT}" \
    #             --language "${LANG}" \
    #             --model_type dpo \
    #             --config "${CONFIG}" \
    #             --adapter_dir "${ADAPTER_RTF}" \
    #             --do_sample \
    #             --temperature "${TEMPERATURE}" \
    #             --top_p "${TOP_P}" \
    #             --output_suffix "_removethink+filter_run-${RUN}"
    #     PIDS+=("${LAUNCHED_PID}")
    # done
    # for PID in "${PIDS[@]}"; do wait "${PID}" || FAIL=1; done

    # # ---- dpo_removethink+filter_full ----
    if [[ "${LANG}" == "SW_KE" ]]; then
        echo "  [${LANG}/dpo_rt+f_full] Skipping (already done)"
    else
    echo "  [${LANG}/dpo_rt+f_full] Queuing runs 1-3 …"
    for RUN in 1 2 3; do
        run_job "${LANG}/dpo_rt+f_full/run-${RUN}" \
            "${LOG_DIR}/${LANG}_eval_dpo_removethink+filter_full_run-${RUN}.log" \
            python "${EVAL_SCRIPT}" \
                --language "${LANG}" \
                --model_type dpo \
                --config "${CONFIG}" \
                --adapter_dir "${ADAPTER_RTF_FULL}" \
                --full_finetune \
                --do_sample \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --output_suffix "_removethink+filter_full_run-${RUN}"
        ALL_PIDS+=("${LAUNCHED_PID}")
    done
    fi

    # ---- sft_mode1 (LoRA) ----
    echo "  [${LANG}/sft_m1] Queuing runs 1-3 …"
    for RUN in 1 2 3; do
        run_job "${LANG}/sft_m1/run-${RUN}" \
            "${LOG_DIR}/${LANG}_eval_sft_mode1_run-${RUN}.log" \
            python "${EVAL_SCRIPT}" \
                --language "${LANG}" \
                --model_type dpo \
                --config "${CONFIG}" \
                --adapter_dir "${ADAPTER_SFT_M1}" \
                --do_sample \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --output_suffix "_sft_mode1_run-${RUN}"
        ALL_PIDS+=("${LAUNCHED_PID}")
    done

    # ---- sft_mode2 (LoRA) ----
    echo "  [${LANG}/sft_m2] Queuing runs 1-3 …"
    for RUN in 1 2 3; do
        run_job "${LANG}/sft_m2/run-${RUN}" \
            "${LOG_DIR}/${LANG}_eval_sft_mode2_run-${RUN}.log" \
            python "${EVAL_SCRIPT}" \
                --language "${LANG}" \
                --model_type dpo \
                --config "${CONFIG}" \
                --adapter_dir "${ADAPTER_SFT_M2}" \
                --do_sample \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --output_suffix "_sft_mode2_run-${RUN}"
        ALL_PIDS+=("${LAUNCHED_PID}")
    done

    # ---- sft_mode1_full ----
    echo "  [${LANG}/sft_m1_full] Queuing runs 1-3 …"
    for RUN in 1 2 3; do
        run_job "${LANG}/sft_m1_full/run-${RUN}" \
            "${LOG_DIR}/${LANG}_eval_sft_mode1_full_run-${RUN}.log" \
            python "${EVAL_SCRIPT}" \
                --language "${LANG}" \
                --model_type dpo \
                --config "${CONFIG}" \
                --adapter_dir "${ADAPTER_SFT_M1_FULL}" \
                --full_finetune \
                --do_sample \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --output_suffix "_sft_mode1_full_run-${RUN}"
        ALL_PIDS+=("${LAUNCHED_PID}")
    done

    # ---- sft_mode2_full ----
    echo "  [${LANG}/sft_m2_full] Queuing runs 1-3 …"
    for RUN in 1 2 3; do
        run_job "${LANG}/sft_m2_full/run-${RUN}" \
            "${LOG_DIR}/${LANG}_eval_sft_mode2_full_run-${RUN}.log" \
            python "${EVAL_SCRIPT}" \
                --language "${LANG}" \
                --model_type dpo \
                --config "${CONFIG}" \
                --adapter_dir "${ADAPTER_SFT_M2_FULL}" \
                --full_finetune \
                --do_sample \
                --temperature "${TEMPERATURE}" \
                --top_p "${TOP_P}" \
                --output_suffix "_sft_mode2_full_run-${RUN}"
        ALL_PIDS+=("${LAUNCHED_PID}")
    done

    # Wait for ALL jobs across all modes to complete
    echo "  [${LANG}] All ${#ALL_PIDS[@]} jobs queued. Waiting for completion …"
    for PID in "${ALL_PIDS[@]}"; do wait "${PID}" || FAIL=1; done

    if [[ ${FAIL} -eq 1 ]]; then
        echo "  ${LANG} had failures. Continuing to next language …"
    fi

    # Per-language summary
    python3 << PYEOF
import json, statistics
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
lang = "${LANG}"

modes = {
    "base": [f"eval_{lang}_base_run-{r}.json" for r in range(1, 4)],
    "dpo": [f"eval_{lang}_dpo_run-{r}.json" for r in range(1, 4)],
    "dpo_rt+f": [f"eval_{lang}_dpo_removethink+filter_run-{r}.json" for r in range(1, 4)],
    "dpo_rt+f_full": [f"eval_{lang}_dpo_removethink+filter_full_run-{r}.json" for r in range(1, 4)],
    "sft_m1": [f"eval_{lang}_dpo_sft_mode1_run-{r}.json" for r in range(1, 4)],
    "sft_m2": [f"eval_{lang}_dpo_sft_mode2_run-{r}.json" for r in range(1, 4)],
    "sft_m1_full": [f"eval_{lang}_dpo_sft_mode1_full_run-{r}.json" for r in range(1, 4)],
    "sft_m2_full": [f"eval_{lang}_dpo_sft_mode2_full_run-{r}.json" for r in range(1, 4)],
}

print()
print(f"  {lang} Results (temperature=${TEMPERATURE}, top_p=${TOP_P})")
print("=" * 90)
print(f'{"Mode":<16} {"Run-1":>8} {"Run-2":>8} {"Run-3":>8} {"   Avg":>8} {"  Std":>8}')
print("-" * 90)

for mode, files in modes.items():
    accs = []
    for fname in files:
        fpath = output_dir / fname
        if fpath.exists():
            d = json.load(open(fpath))
            accs.append(d["accuracy"] * 100)
        else:
            accs.append(None)

    vals = [f"{a:.2f}%" if a is not None else "N/A" for a in accs]
    valid = [a for a in accs if a is not None]
    avg = statistics.mean(valid) if valid else 0
    std = statistics.stdev(valid) if len(valid) > 1 else 0

    print(f"{mode:<16} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {avg:>7.2f}% {std:>7.2f}%")

print("=" * 90)
print()
PYEOF
done

# ---------------------------------------------------------------------------
# Final combined summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All evaluations complete. Combined summary:"
echo "============================================================"

python3 << PYEOF
import json, statistics
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
languages = ["SW_KE", "JA_JP", "BN_BD"]
mode_map = {
    "base":          lambda lang, r: f"eval_{lang}_base_run-{r}.json",
    "dpo":           lambda lang, r: f"eval_{lang}_dpo_run-{r}.json",
    "dpo_rt+f":      lambda lang, r: f"eval_{lang}_dpo_removethink+filter_run-{r}.json",
    "dpo_rt+f_full": lambda lang, r: f"eval_{lang}_dpo_removethink+filter_full_run-{r}.json",
    "sft_m1":        lambda lang, r: f"eval_{lang}_dpo_sft_mode1_run-{r}.json",
    "sft_m2":        lambda lang, r: f"eval_{lang}_dpo_sft_mode2_run-{r}.json",
    "sft_m1_full":   lambda lang, r: f"eval_{lang}_dpo_sft_mode1_full_run-{r}.json",
    "sft_m2_full":   lambda lang, r: f"eval_{lang}_dpo_sft_mode2_full_run-{r}.json",
}

print()
print("  Combined Results (temperature=0.6, top_p=0.95)")
print("=" * 96)
print(f'{"Lang":<8} {"Mode":<16} {"Run-1":>8} {"Run-2":>8} {"Run-3":>8} {"   Avg":>8} {"  Std":>8}')
print("-" * 96)

for lang in languages:
    for mode, fname_fn in mode_map.items():
        accs = []
        for r in range(1, 4):
            fpath = output_dir / fname_fn(lang, r)
            if fpath.exists():
                d = json.load(open(fpath))
                accs.append(d["accuracy"] * 100)
            else:
                accs.append(None)

        vals = [f"{a:.2f}%" if a is not None else "N/A" for a in accs]
        valid = [a for a in accs if a is not None]
        avg = statistics.mean(valid) if valid else 0
        std = statistics.stdev(valid) if len(valid) > 1 else 0

        print(f"{lang:<8} {mode:<16} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {avg:>7.2f}% {std:>7.2f}%")
    print("-" * 96)

print("=" * 96)
print()
PYEOF

echo "Done."

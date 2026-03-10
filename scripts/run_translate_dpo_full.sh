#!/usr/bin/env bash
# =============================================================================
# run_translate_dpo_full.sh
#
# Full fine-tuning (no LoRA) DPO training using removethink+filter data,
# followed by MGSM evaluation.
#
# GPU layout (one GPU per language):
#   GPU 5 — JA_JP
#   GPU 6 — BN_BD
#   GPU 7 — SW_KE
#
# Usage:
#   bash scripts/run_translate_dpo_full.sh
# =============================================================================
set -euo pipefail

# Activate conda environment (matches other scripts in this project)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate ml-delta 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_ROOT}/configs/translate_dpo_config.yaml"

LANGUAGES=("JA_JP" "BN_BD" "SW_KE")
GPUS=(5 6 7)

OUTPUT_DIR="${PROJECT_ROOT}/outputs-translate_dpo"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Translate-DPO Full Fine-Tuning (no LoRA)"
echo "  Config : ${CONFIG}"
echo "  Data   : removethink+filter"
echo "  GPU 5  : ${LANGUAGES[0]}"
echo "  GPU 6  : ${LANGUAGES[1]}"
echo "  GPU 7  : ${LANGUAGES[2]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: DPO full fine-tuning (all languages in parallel, one GPU each)
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 1: DPO Full FT — parallel ====="
TRAIN_PIDS=()
for i in 0 1 2; do
    LANG="${LANGUAGES[$i]}"
    LOG="${LOG_DIR}/${LANG}_dpo_train_removethink+filter_full.log"
    echo "  [${LANG}] Training DPO (full FT) on GPU ${GPUS[$i]} …"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        python "${PROJECT_ROOT}/training/run_translate_dpo_training.py" \
            --language "${LANG}" \
            --config "${CONFIG}" \
            --full_finetune \
            > "${LOG}" 2>&1 &
    TRAIN_PIDS+=($!)
done
FAIL=0
for i in 0 1 2; do
    if wait "${TRAIN_PIDS[$i]}"; then
        echo "  [${LANGUAGES[$i]}] Done. Log: ${LOG_DIR}/${LANGUAGES[$i]}_dpo_train_removethink+filter_full.log"
    else
        echo "  [${LANGUAGES[$i]}] FAILED. Log: ${LOG_DIR}/${LANGUAGES[$i]}_dpo_train_removethink+filter_full.log"
        FAIL=1
    fi
done
[[ ${FAIL} -eq 1 ]] && { echo "Phase 1 failed."; exit 1; }
echo "Phase 1 complete."

# ---------------------------------------------------------------------------
# Phase 2: MGSM evaluation (all languages in parallel, one GPU each)
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 2: MGSM evaluation (dpo_removethink+filter_full) — parallel ====="
EVAL_PIDS=()
for i in 0 1 2; do
    LANG="${LANGUAGES[$i]}"
    LOG="${LOG_DIR}/${LANG}_eval_dpo_removethink+filter_full.log"
    ADAPTER="${OUTPUT_DIR}/dpo_${LANG}_removethink+filter_full"
    echo "  [${LANG}/dpo_full] Evaluating on GPU ${GPUS[$i]} …"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
        python "${PROJECT_ROOT}/evaluation/run_translate_dpo_eval.py" \
            --language "${LANG}" \
            --model_type dpo \
            --config "${CONFIG}" \
            --adapter_dir "${ADAPTER}" \
            --output_suffix "_removethink+filter_full" \
            --full_finetune \
            > "${LOG}" 2>&1 &
    EVAL_PIDS+=($!)
done
FAIL=0
for i in 0 1 2; do
    if wait "${EVAL_PIDS[$i]}"; then
        echo "  [${LANGUAGES[$i]}] Eval complete. Log: ${LOG_DIR}/${LANGUAGES[$i]}_eval_dpo_removethink+filter_full.log"
    else
        echo "  [${LANGUAGES[$i]}] Eval FAILED. Log: ${LOG_DIR}/${LANGUAGES[$i]}_eval_dpo_removethink+filter_full.log"
        FAIL=1
    fi
done
[[ ${FAIL} -eq 1 ]] && { echo "Phase 2 failed."; exit 1; }
echo "Phase 2 complete."

# ---------------------------------------------------------------------------
# Phase 3: Summary
# ---------------------------------------------------------------------------
echo ""
echo "===== Phase 3: Results Summary ====="

echo ""
printf "%-8s  %-22s  %8s  %12s  %12s  %14s  %14s\n" \
    "Lang" "Model" "Accuracy" "Full Native" "Full English" "Think Native" "Think English"
printf "%s\n" "$(printf '=%.0s' {1..104})"

for LANG in "${LANGUAGES[@]}"; do
    for MODEL_LABEL in "dpo_rt+f" "dpo_rt+f_full"; do
        case "${MODEL_LABEL}" in
            dpo_rt+f)       EVAL_FILE="${OUTPUT_DIR}/eval_${LANG}_dpo_removethink+filter.json" ;;
            dpo_rt+f_full)  EVAL_FILE="${OUTPUT_DIR}/eval_${LANG}_dpo_removethink+filter_full.json" ;;
        esac
        if [[ -f "${EVAL_FILE}" ]]; then
            ACC=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['accuracy']:.2%}\")")
            FT=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_full_target_ratio']:.3f}\")")
            FE=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_full_english_ratio']:.3f}\")")
            TT=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_think_target_ratio']:.3f}\")")
            TE=$(python3 -c "import json; d=json.load(open('${EVAL_FILE}')); print(f\"{d['mean_think_english_ratio']:.3f}\")")
            printf "%-8s  %-22s  %8s  %12s  %12s  %14s  %14s\n" \
                "${LANG}" "${MODEL_LABEL}" "${ACC}" "${FT}" "${FE}" "${TT}" "${TE}"
        else
            printf "%-8s  %-22s  %8s\n" "${LANG}" "${MODEL_LABEL}" "(missing)"
        fi
    done
done

echo ""
echo "============================================================"
echo "  Translate-DPO full fine-tuning experiment complete."
echo "  Outputs : ${OUTPUT_DIR}/dpo_*_full/"
echo "  Logs    : ${LOG_DIR}/"
echo "============================================================"

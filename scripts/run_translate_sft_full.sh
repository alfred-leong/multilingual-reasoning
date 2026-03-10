#!/usr/bin/env bash
# =============================================================================
# run_translate_sft_full.sh
#
# Full fine-tuning (no LoRA) SFT on the translated data from
# data/translate_dpo/translated/
#
# Two modes (run separately or together):
#   Mode 1: Remove think tags, SFT on ALL ~14k data samples per language
#   Mode 2: Remove think tags, filter to CORRECT-only responses, then SFT
#
# GPU layout (one GPU per language):
#   GPU 5 — JA_JP
#   GPU 6 — BN_BD
#   GPU 7 — SW_KE
#
# Usage:
#   bash scripts/run_translate_sft_full.sh         # both modes
#   bash scripts/run_translate_sft_full.sh 1       # mode 1 only
#   bash scripts/run_translate_sft_full.sh 2       # mode 2 only
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

OUTPUT_DIR="${PROJECT_ROOT}/outputs-translate_sft"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Determine which modes to run
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    MODES=("$1")
else
    MODES=(1 2)
fi

echo "============================================================"
echo "  Translate-SFT Training (Full Fine-Tuning — no LoRA)"
echo "  Config : ${CONFIG}"
echo "  Modes  : ${MODES[*]}"
echo "  GPU 5  : ${LANGUAGES[0]}"
echo "  GPU 6  : ${LANGUAGES[1]}"
echo "  GPU 7  : ${LANGUAGES[2]}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run full SFT for each mode
# ---------------------------------------------------------------------------
for MODE in "${MODES[@]}"; do
    echo ""
    echo "===== SFT Full FT Mode ${MODE} — Training all languages in parallel ====="

    TRAIN_PIDS=()
    for i in 0 1 2; do
        LANG="${LANGUAGES[$i]}"
        GPU="${GPUS[$i]}"
        LOG="${LOG_DIR}/${LANG}_sft_train_mode${MODE}_full.log"

        echo "  [${LANG}] SFT full FT mode=${MODE} on GPU ${GPU} → ${LOG}"
        CUDA_VISIBLE_DEVICES="${GPU}" \
            python "${PROJECT_ROOT}/training/run_translate_sft_training.py" \
                --language "${LANG}" \
                --mode "${MODE}" \
                --config "${CONFIG}" \
                --full_finetune \
                > "${LOG}" 2>&1 &
        TRAIN_PIDS+=($!)
    done

    # Wait for all three languages
    FAIL=0
    for i in 0 1 2; do
        if wait "${TRAIN_PIDS[$i]}"; then
            echo "  [${LANGUAGES[$i]}] Mode ${MODE} full FT — Done."
            echo "    Log   : ${LOG_DIR}/${LANGUAGES[$i]}_sft_train_mode${MODE}_full.log"
            echo "    Output: ${OUTPUT_DIR}/sft_${LANGUAGES[$i]}_mode${MODE}_full/"
        else
            echo "  [${LANGUAGES[$i]}] Mode ${MODE} full FT — FAILED."
            echo "    Log: ${LOG_DIR}/${LANGUAGES[$i]}_sft_train_mode${MODE}_full.log"
            FAIL=1
        fi
    done
    [[ ${FAIL} -eq 1 ]] && { echo "SFT full FT mode ${MODE} had failures. Aborting."; exit 1; }
    echo "SFT Full FT Mode ${MODE} training complete."

    # -------------------------------------------------------------------
    # Evaluation (greedy decoding) for this mode
    # -------------------------------------------------------------------
    echo ""
    echo "===== SFT Full FT Mode ${MODE} — Evaluation (greedy) — parallel ====="
    EVAL_PIDS=()
    for i in 0 1 2; do
        LANG="${LANGUAGES[$i]}"
        GPU="${GPUS[$i]}"
        ADAPTER="${OUTPUT_DIR}/sft_${LANG}_mode${MODE}_full"
        SUFFIX="_sft_mode${MODE}_full"
        EVAL_LOG="${LOG_DIR}/${LANG}_eval_sft_mode${MODE}_full.log"

        echo "  [${LANG}] Evaluating SFT full FT mode=${MODE} on GPU ${GPU} → ${EVAL_LOG}"
        CUDA_VISIBLE_DEVICES="${GPU}" \
            python "${PROJECT_ROOT}/evaluation/run_translate_dpo_eval.py" \
                --language "${LANG}" \
                --model_type dpo \
                --config "${CONFIG}" \
                --adapter_dir "${ADAPTER}" \
                --output_suffix "${SUFFIX}" \
                --full_finetune \
                > "${EVAL_LOG}" 2>&1 &
        EVAL_PIDS+=($!)
    done

    FAIL=0
    for i in 0 1 2; do
        if wait "${EVAL_PIDS[$i]}"; then
            echo "  [${LANGUAGES[$i]}] Mode ${MODE} full FT eval — Done."
            echo "    Log : ${LOG_DIR}/${LANGUAGES[$i]}_eval_sft_mode${MODE}_full.log"
        else
            echo "  [${LANGUAGES[$i]}] Mode ${MODE} full FT eval — FAILED."
            echo "    Log : ${LOG_DIR}/${LANGUAGES[$i]}_eval_sft_mode${MODE}_full.log"
            FAIL=1
        fi
    done
    [[ ${FAIL} -eq 1 ]] && { echo "SFT full FT mode ${MODE} evaluation had failures. Aborting."; exit 1; }
    echo "SFT Full FT Mode ${MODE} evaluation complete."
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "===== Results Summary (Full Fine-Tuning SFT) ====="
echo ""
printf "%-8s  %-22s  %8s  %12s  %12s  %14s  %14s\n" \
    "Lang" "Model" "Accuracy" "Full Native" "Full English" "Think Native" "Think English"
printf "%s\n" "$(printf '=%.0s' {1..104})"

# Eval files are written by run_translate_dpo_eval.py into outputs-translate_dpo/
EVAL_DIR="${PROJECT_ROOT}/outputs-translate_dpo"
for LANG in "${LANGUAGES[@]}"; do
    for MODE in "${MODES[@]}"; do
        EVAL_FILE="${EVAL_DIR}/eval_${LANG}_dpo_sft_mode${MODE}_full.json"
        MODEL_LABEL="sft_mode${MODE}_full"
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
echo "  All SFT full fine-tuning + evaluation complete."
echo "  Model outputs : ${OUTPUT_DIR}/sft_*_full/"
echo "  SFT data      : ${PROJECT_ROOT}/data/translate_sft/mode_*/"
echo "  Eval results  : ${EVAL_DIR}/eval_*_sft_*_full.json"
echo "  Logs          : ${LOG_DIR}/"
echo "============================================================"

# multilingual-reasoning

**Adaptive Reasoning for Multilingual LLMs**

This research codebase studies how supervised fine-tuning (SFT) with translated English reasoning traces can improve multilingual chain-of-thought reasoning for low-resource languages.

## Motivation

Large language models often default to English when performing chain-of-thought reasoning, even when prompted in another language. This is especially pronounced for low-resource languages (e.g. Bengali, Swahili) where pre-training data is scarce.

The core idea is that a model reasons better in English, so a high-quality translation of the English reasoning trace should outperform a directly-generated native-language trace. We generate English responses, translate them with TranslateGemma-27B, and use the translated traces as SFT training data.

## Languages

| Code | Language | Resource Level |
|------|----------|----------------|
| ja   | Japanese | Medium         |
| bn   | Bengali  | Low            |
| sw   | Swahili  | Low            |

## Project Structure

```
multilingual-reasoning/
├── configs/
│   └── deepspeed_zero3.json              # DeepSpeed ZeRO-3 config for multi-GPU training
├── data/
│   ├── mgsm/                             # Raw MGSM dataset (tsv files)
│   ├── exp_v2/                           # Processed data (train/test splits, responses, SFT data)
│   ├── split_datasets-exp_v2.py          # Split MMMLU into train/test
│   ├── generate_english_responses-exp_v2.py  # Generate English CoT responses for MMMLU via vLLM
│   ├── check_english_responses-exp_v2.py     # Validate English responses
│   ├── translate_responses-exp_v2.py         # Translate responses with TranslateGemma
│   ├── check_translated_responses-exp_v2.py  # Validate translated responses
│   └── prepare_training_data-exp_v3.py       # Build SFT training data from translations
├── prompts/
│   └── translate_dpo_prompts.py          # System prompts per language
├── training/
│   └── run_sft-exp_v2.py                # SFT fine-tuning (LoRA or full, multi-seed)
├── evaluation/
│   └── run_eval-exp_v2.py               # MGSM + MMMLU accuracy evaluation
├── scripts/
│   ├── exp_v3/                           # Pipeline scripts (see below)
│   └── launch_vllm_patched.py            # Patched vLLM launcher for TranslateGemma
├── outputs-exp_v3/                       # Evaluation results and logs
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd multilingual-reasoning

# 2. Create conda environment
conda create -n ml-delta python=3.11
conda activate ml-delta

# 3. Install dependencies
pip install -r requirements.txt

# 4. Log in to Hugging Face
hf auth login
```

### Environment Variables

The pipeline scripts use environment variables with sensible defaults. Override them as needed:

| Variable          | Default                                                     | Description                              |
|-------------------|-------------------------------------------------------------|------------------------------------------|
| `CONDA_SETUP`     | `$HOME/miniconda3/etc/profile.d/conda.sh`                  | Path to conda setup script               |
| `CONDA_ENV`       | `ml-delta`                                                  | Conda environment name                   |
| `MODEL_STORE`     | NA              | Directory for saving/loading trained models |
| `TRANSLATE_MODEL` | NA | Path to local TranslateGemma-27B weights |

Example:
```bash
export MODEL_STORE="/data/models/ml-reasoning-exp_v3"
export TRANSLATE_MODEL="/data/models/translategemma-27b-it"
```

> **Hardware:** Full-scale experiments require multiple GPUs with >= 80 GB VRAM each (e.g. A100). Training uses DeepSpeed ZeRO-3 across 3 GPUs. Evaluation and generation can run on single GPUs.

## Pipeline (scripts/exp_v3/)

The pipeline is run step-by-step. Each script is self-contained and idempotent (skips already-completed work).

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_split_datasets.sh` | Split MMMLU into train/test sets |
| 2 | `2_generate_english_responses.sh [model]` | Start vLLM with the given model (default: Qwen3-8B), generate English CoT responses for MMMLU |
| 3 | `3_translate_responses_gemma27b.sh` | Start 3 TranslateGemma-27B vLLM servers, translate English responses to ja/bn/sw |
| 5 | `5_prepare_training_data.sh` | Build SFT training JSONL from translated responses (filtered + unfiltered) |
| 6 | `6_sft_training_qwen3-1_7b.sh` | Full-finetune Qwen3-1.7B via SFT (3 seeds x 2 filter modes) for all languages, then evaluate |
| 6 | `6_sft_training_qwen3-8b.sh` | Same as above for Qwen3-8B, Swahili only |
| 8 | `8_evaluate_models.sh [dataset] [subset]` | Evaluate all trained models on multilingual MGSM/MMMLU |
| 9 | `9_evaluate_english_perf.sh` | Evaluate all trained models on English MGSM/MMMLU (regression check) |

### Running the pipeline

```bash
# Step 1: Split MMMLU into train/test
bash scripts/exp_v3/1_split_datasets.sh

# Step 2: Generate English responses (starts a vLLM server, generates, then stops)
# Accepts an optional model argument (default: Qwen/Qwen3-8B)
bash scripts/exp_v3/2_generate_english_responses.sh                # uses Qwen3-8B
bash scripts/exp_v3/2_generate_english_responses.sh Qwen/Qwen3-1.7B  # uses Qwen3-1.7B

# Step 3: Translate to target languages (starts 3 vLLM servers)
bash scripts/exp_v3/3_translate_responses_gemma27b.sh

# Step 5: Prepare SFT training data
bash scripts/exp_v3/5_prepare_training_data.sh

# Step 6: Train and evaluate (pick one or both)
bash scripts/exp_v3/6_sft_training_qwen3-1_7b.sh
bash scripts/exp_v3/6_sft_training_qwen3-8b.sh

# Step 8-9: Additional evaluation
bash scripts/exp_v3/8_evaluate_models.sh mmmlu 100
bash scripts/exp_v3/9_evaluate_english_perf.sh
```

## How It Works

1. **Generate** -- For each MMMLU training question, run a model (Qwen3-8B or Qwen3-1.7B) with an English system prompt to produce chain-of-thought reasoning in English.
2. **Translate** -- Translate the English reasoning traces to each target language (ja, bn, sw) using TranslateGemma-27B, preserving the `<think>...</think>` structure.
3. **Prepare SFT data** -- Pair each translated response with its native-language question and system prompt. Two variants: *no-filter* (all samples) and *filter* (only correctly-answered samples).
4. **SFT Train** -- Full-finetune the target model (Qwen3-1.7B or Qwen3-8B) on the translated SFT data using DeepSpeed ZeRO-3 across 3 GPUs. Each configuration is trained with 3 random seeds.
5. **Evaluate** -- Measure accuracy on MGSM (50 samples) and MMMLU (100 samples from test split) in target languages and English.

## Evaluation Metrics

- **Accuracy** -- Fraction of questions answered correctly (numerical match for MGSM, letter match for MMMLU).
- Results are saved as JSON files in `outputs-exp_v3/`.

## License

This is a research codebase. Model weights are subject to their respective licenses on the Hugging Face Hub.

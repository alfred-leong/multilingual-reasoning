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
| bn   | Bengali  | Medium         |
| sw   | Swahili  | Low            |

## Project Structure

```
multilingual-reasoning/
├── configs/
│   └── deepspeed_zero3.json              # DeepSpeed ZeRO-3 config for multi-GPU training
├── data/
│   ├── mgsm/                             # Raw MGSM dataset (tsv files, 50 samples per language)
│   ├── exp_v2/                           # Processed data (train/test splits, responses, SFT data)
│   ├── split_datasets-exp_v2.py          # Split MMMLU into train/test (80/20)
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

### Prerequisites

- Linux with NVIDIA GPUs (tested on 3x H100 80GB)
- CUDA 12.x
- Conda (Miniconda or Anaconda)
- ~200GB disk for model weights and generated data

### Installation

```bash
# 1. Clone the repository
git clone "https://github.com/alfred-leong/multilingual-reasoning"
cd multilingual-reasoning

# 2. Create conda environment
conda create -n ml-delta python=3.11 -y
conda activate ml-delta

# 3. Install dependencies
pip install -r requirements.txt

# 4. Log in to Hugging Face (required for gated models like Qwen3)
huggingface-cli login

# 5. Log in to Weights & Biases (for training logging)
wandb login
```

### Environment Variables

The pipeline scripts use environment variables with sensible defaults. Set these before running:

| Variable          | Default                                    | Description                              |
|-------------------|--------------------------------------------|------------------------------------------|
| `CONDA_SETUP`     | `$HOME/miniconda3/etc/profile.d/conda.sh` | Path to conda setup script               |
| `CONDA_ENV`       | `ml-delta`                                 | Conda environment name                   |
| `MODEL_STORE`     | *(required)*                               | Directory for saving/loading trained models |
| `TRANSLATE_MODEL` | *(required)*                               | Path to local TranslateGemma-27B weights |

```bash
export MODEL_STORE="/path/to/models/ml-reasoning-exp_v3"
export TRANSLATE_MODEL="/path/to/models/translategemma-27b-it"
```

### Downloading TranslateGemma-27B

The translation step requires a local copy of [google/translategemma-27b-it](https://huggingface.co/google/translategemma-27b-it):

```bash
huggingface-cli download google/translategemma-27b-it --local-dir "$TRANSLATE_MODEL"
```

## Reproducing Results

### Hardware Requirements

| Step | GPUs | VRAM per GPU | Time Estimate |
|------|------|--------------|---------------|
| 2. Generate English responses | 1 GPU | ~20 GB (Qwen3-8B) | ~1 hour |
| 3. Translate responses | 3 GPUs | ~60 GB each (TranslateGemma-27B) | ~2 hours |
| 6. SFT training | 3 GPUs | ~80 GB each (DeepSpeed ZeRO-3) | ~3 hours per seed |
| 8-9. Evaluation | 1-3 GPUs | ~20-40 GB | ~30 min per model |

Total: **3x H100 80GB** recommended. Training is the bottleneck.

### Step-by-Step

GPU IDs below are defaults in the scripts — edit them to match your setup.

```bash
# ── Step 1: Split MMMLU into train/test (80/20) ──────────────────────────────
# Creates data/exp_v2/mmmlu/{train,test}_{en,ja,bn,sw}.jsonl
# MGSM test sets (50 samples) are already included in data/mgsm/*.tsv
bash scripts/exp_v3/1_split_datasets.sh

# ── Step 2: Generate English CoT responses for MMMLU ─────────────────────────
# Starts a vLLM server, generates responses, then stops.
# Default model: Qwen/Qwen3-1.7B. Pass a different model as argument.
bash scripts/exp_v3/2_generate_english_responses.sh                  # Qwen3-1.7B
bash scripts/exp_v3/2_generate_english_responses.sh Qwen/Qwen3-8B   # Qwen3-8B

# ── Step 3: Translate English responses to target languages ───────────────────
# Starts 3 TranslateGemma-27B vLLM servers (one per language), translates, stops.
# Requires TRANSLATE_MODEL to be set.
bash scripts/exp_v3/3_translate_responses_gemma27b.sh

# ── Step 5: Prepare SFT training data ────────────────────────────────────────
# Pairs translated responses with native-language questions.
# Creates two variants per language: no-filter (all) and filter (correct only).
bash scripts/exp_v3/5_prepare_training_data.sh

# ── Step 6: SFT training + evaluation ────────────────────────────────────────
# Each script trains 3 seeds x 2 filter modes, then evaluates on MGSM + MMMLU.
# Qwen3-1.7B: all languages (ja, bn, sw)
bash scripts/exp_v3/6_sft_training_qwen3-1_7b.sh

# Qwen3-8B: Swahili only
bash scripts/exp_v3/6_sft_training_qwen3-8b.sh

# ── Step 8-9: Additional evaluation (optional) ───────────────────────────────
bash scripts/exp_v3/8_evaluate_models.sh mmmlu 100
bash scripts/exp_v3/9_evaluate_english_perf.sh
```

### Notes

- All scripts skip already-completed work (trained models, eval results).
- To re-run a step, delete the corresponding output files first.
- GPU IDs are hardcoded in each script (e.g. `GPU=5`, `GPU_IDS="2,3,4"`). Edit them to match your hardware.
- The Qwen3-1.7B SFT data under `data/exp_v2/qwen3-1_7b/` was generated by running steps 2-5 with `Qwen/Qwen3-1.7B`. The Qwen3-8B data under `data/exp_v2/qwen3-8b/` was generated with `Qwen/Qwen3-8B`.

## How It Works

1. **Generate** -- For each MMMLU training question, run a model (Qwen3-8B or Qwen3-1.7B) with an English system prompt to produce chain-of-thought reasoning in English.
2. **Translate** -- Translate the English reasoning traces to each target language (ja, bn, sw) using TranslateGemma-27B, preserving the `<think>...</think>` structure.
3. **Prepare SFT data** -- Pair each translated response with its native-language question and system prompt. Two variants: *no-filter* (all samples) and *filter* (only correctly-answered samples).
4. **SFT Train** -- Full-finetune the target model (Qwen3-1.7B or Qwen3-8B) on the translated SFT data using DeepSpeed ZeRO-3 across 3 GPUs. Each configuration is trained with 3 random seeds.
5. **Evaluate** -- Measure accuracy on MGSM (50 samples) and MMMLU (100 samples from test split) in target languages and English.

## Results

SFT results are averaged over 3 seeds. **Bold** indicates the best result per column within each group. Learning rate is 2e-7 for Bengali and Japanese, and 2e-5 for Swahili.

### Qwen3-1.7B

#### Bengali (BN)

| Setting | MGSM | MMMLU | MGSM (EN) | MMMLU (EN) |
|---------|------|-------|------------|------------|
| Base | **22.00** | **28.00** | 78.00 | 60.00 |
| AutoSFT-NoFilter | 15.33 | 17.00 | 78.00 | **63.67** |
| AutoSFT-Filter | 20.00 | 21.33 | **80.00** | **63.67** |

#### Japanese (JA)

| Setting | MGSM | MMMLU | MGSM (EN) | MMMLU (EN) |
|---------|------|-------|------------|------------|
| Base | **58.00** | **49.00** | 78.00 | 60.00 |
| AutoSFT-NoFilter | 46.67 | 40.67 | 73.33 | 65.00 |
| AutoSFT-Filter | 48.00 | 39.67 | **79.33** | **65.67** |

#### Swahili (SW)

| Setting | MGSM | MMMLU | MGSM (EN) | MMMLU (EN) |
|---------|------|-------|------------|------------|
| Base | 0.00 | 12.00 | **78.00** | **60.00** |
| AutoSFT-NoFilter | **7.33** | 33.33 | 73.33 | 57.33 |
| AutoSFT-Filter | 6.00 | **34.33** | 73.33 | 54.00 |

### Qwen3-8B (Swahili only)

**lr=2e-5:**

| Setting | MGSM | MMMLU | MGSM (EN) | MMMLU (EN) |
|---------|------|-------|------------|------------|
| Base | **40.00** | 23.00 | **92.00** | **78.00** |
| AutoSFT-NoFilter | 0.00 | 23.33 | 60.00 | 57.33 |
| AutoSFT-Filter | 12.00 | **25.67** | 84.00 | 74.00 |

**lr=2e-7:**

| Setting | MGSM | MMMLU |
|---------|------|-------|
| Base | **40.00** | 23.00 |
| AutoSFT-NoFilter | 38.67 | 19.67 |
| AutoSFT-Filter | 32.00 | **21.00** |

## Evaluation Metrics

- **Accuracy** -- Fraction of questions answered correctly (numerical match for MGSM, letter match for MMMLU).
- Results are saved as JSON files in `outputs-exp_v3/`.

## License

This is a research codebase. Model weights are subject to their respective licenses on the Hugging Face Hub.

# multilingual-reasoning

**Improving low-resource language reasoning in LLMs via Translate-DPO.**

This research codebase studies how preference-based fine-tuning (DPO) can improve multilingual chain-of-thought reasoning — with a focus on low-resource languages — while preserving native-language fidelity in a model's thinking traces.

## Motivation

Large language models often default to English when performing chain-of-thought reasoning, even when prompted in another language.  This is especially pronounced for low-resource languages (e.g. Bengali, Swahili) where pre-training data is scarce.

**Translate-DPO** addresses this by generating responses in both English and the native language from the same small model, translating the English reasoning traces to the native language with a dedicated translation model (TranslateGemma), then training via DPO with the translated trace as *chosen* and the raw native trace as *rejected*.  The hypothesis is that English reasoning is stronger, so a high-quality translation should outperform direct native-language generation.

The pipeline evaluates reasoning accuracy on MGSM and measures the language composition of the model's thinking traces.

## Project Structure

```
multilingual-reasoning/
├── configs/
│   └── translate_dpo_config.yaml         # Experiment parameters
├── data/
│   ├── generate_translate_dpo_responses.py  # English + native response generation
│   ├── translate_with_gemma.py           # Translate English traces with TranslateGemma
│   └── language_utils.py                 # Unicode-based language detection
├── training/
│   └── run_translate_dpo_training.py     # DPO fine-tuning with LoRA
├── evaluation/
│   ├── run_translate_dpo_eval.py         # MGSM accuracy + language analysis
│   └── language_ratio.py                 # Aggregate language-ratio reports
├── prompts/
│   └── translate_dpo_prompts.py          # English + native prompts
├── scripts/
│   ├── run_translate_dpo_experiment.sh   # End-to-end pipeline runner
│   └── start_translate_dpo_servers.sh    # Launch vLLM servers
├── outputs-translate_dpo/                # Results and checkpoints
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone or copy the project
cd multilingual-reasoning

# 2. Create a virtual environment (recommended)
conda create -n multilingual-reasoning python=3.11

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Log in to Hugging Face for gated models
huggingface-cli login
```

> **Hardware:** Full-scale experiments require a GPU with ≥ 24 GB VRAM (e.g. A100/A6000).  LoRA is used during DPO training for memory efficiency.  The pipeline runs three languages in parallel across three GPUs.

---

## Translate-DPO

### How it works

The core idea is that a model reasons better in English, so a translated English reasoning trace should be higher quality than a directly-generated native-language trace.

**Pipeline:**

1. **Generate** — For each MMMLU question, run `Qwen3-1.7B` in thinking mode twice:
   - *Setting 1 (English)*: standard English system prompt; model reasons in English.
   - *Setting 2 (Native)*: native-language system prompt + a language-seeding thinking prefix; model reasons in the target language.
2. **Translate** — Translate the Setting-1 English thinking trace (and final answer) to the target language using `TranslateGemma` (`google/translategemma-4b-it`), preserving the `<think>…</think>` structure.
3. **DPO train** — Fine-tune `Qwen3-1.7B` with LoRA:
   - *Chosen* = translated English response (high-quality reasoning, native language)
   - *Rejected* = raw native-language response (native language, weaker reasoning)
4. **Evaluate** — Run MGSM with the native prompt + thinking prefix (Setting 2 conditions) on both the base and DPO-trained model. Metrics: accuracy and native-language ratio.

### Full pipeline

```bash
bash scripts/run_translate_dpo_experiment.sh
```

This script runs all phases automatically across three GPUs (one per language):

| Phase | Description |
|-------|-------------|
| 0 | Start three vLLM generation servers (one per language/GPU) |
| 1 | Generate English + native responses in parallel |
| 2 | Start TranslateGemma vLLM servers; translate English traces; merge shards |
| 3 | DPO training in parallel (one per GPU) |
| 4 | MGSM evaluation in parallel |
| 5 | Summary report |

### Individual scripts

**Generate English + native responses:**

```bash
python data/generate_translate_dpo_responses.py --language JA_JP
python data/generate_translate_dpo_responses.py              # all languages
python data/generate_translate_dpo_responses.py --language JA_JP --max_samples 50
```

Requires a running vLLM server for `Qwen3-1.7B` (default port 8300).

**Translate English traces with TranslateGemma:**

```bash
# Direct GPU inference (recommended — no vLLM server needed)
python data/translate_with_gemma.py --language JA_JP --gpu 1
python data/translate_with_gemma.py --gpu 1              # all languages

# Legacy vLLM mode
python data/translate_with_gemma.py --language JA_JP --port 8401
```

**DPO training:**

```bash
python training/run_translate_dpo_training.py --language JA_JP
python training/run_translate_dpo_training.py              # all languages
```

**MGSM evaluation:**

```bash
python evaluation/run_translate_dpo_eval.py --language JA_JP --model_type base
python evaluation/run_translate_dpo_eval.py --language JA_JP --model_type dpo
python evaluation/run_translate_dpo_eval.py              # all languages, both models
```

### Configuration (`configs/translate_dpo_config.yaml`)

| Section        | Key                        | Description |
|----------------|----------------------------|-------------|
| `languages`    | `code`, `name`, `mgsm_split` | Target languages and MGSM split names |
| `models`       | `generation_model`         | Model used to generate Setting-1/2 responses (`Qwen3-1.7B`) |
|                | `translation_model`        | Translation model (`google/translategemma-4b-it`) |
|                | `base_model`               | Base model for DPO training and evaluation |
| `generation`   | `max_tokens`, `temperature`, `top_p` | Sampling parameters for response generation |
| `translation`  | `max_tokens`, `temperature`, `top_p` | Sampling parameters for translation |
| `dpo`          | `learning_rate`, `beta`, … | DPO training hyperparameters |
| `vllm_ports`   | `generation`, `translation` | Default vLLM server ports |

### Output files

| Path | Description |
|------|-------------|
| `data/translate_dpo/raw/{LANG}_responses.jsonl` | Setting-1 (English) and Setting-2 (native) responses |
| `data/translate_dpo/translated/{LANG}_translated.jsonl` | Translated English traces |
| `data/translate_dpo/dpo_pairs/{LANG}_pairs.jsonl` | Final chosen/rejected preference pairs |
| `outputs-translate_dpo/dpo_{LANG}/` | LoRA checkpoints + training log |
| `outputs-translate_dpo/eval_{LANG}_{MODEL}.json` | Per-example evaluation results |

---

## Evaluation Metrics

- **Accuracy** – fraction of MGSM questions answered correctly (numerical match).
- **Native-language ratio** (`mean_target_ratio`) – average proportion of target-language characters in the thinking trace.
- **English ratio** (`mean_english_ratio`) – average proportion of English/Latin characters in the thinking trace.

## Languages

| Code | Language | MGSM Split | Resource Level |
|------|----------|------------|----------------|
| JA_JP | Japanese | ja | Medium |
| BN_BD | Bengali | bn | Low |
| SW_KE | Swahili | sw | Low |

## License

This is a research codebase.  Model weights are subject to their respective licenses on the Hugging Face Hub.

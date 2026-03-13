---
annotations_creators:
  - found
language_creators:
  - found
  - expert-generated
language:
  - en
  - es
  - fr
  - de
  - ru
  - zh
  - ja
  - th
  - sw
  - bn
license:
  - cc-by-sa-4.0
multilinguality:
  - multilingual
size_categories:
  - 1K<n<10K
source_datasets:
  - extended|gsm8k
task_categories:
  - text-generation
task_ids: []
paperswithcode_id: multi-task-language-understanding-on-mgsm
pretty_name: Multilingual Grade School Math Benchmark (MGSM)
tags:
  - math-word-problems
dataset_info:
  - config_name: en
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: string
      - name: answer_number
        dtype: int32
      - name: equation_solution
        dtype: string
    splits:
      - name: train
        num_bytes: 3963202
        num_examples: 8
      - name: test
        num_bytes: 713732
        num_examples: 250
    download_size: 4915944
    dataset_size: 4676934
  - config_name: es
    features:
      - name: question
        dtype: string
      - name: answer
        dtype: string
      - name: answer_number
        dtype: int32
      - name: equation_solution
        dtype: string
    splits:
      - name: train
        num_bytes: 3963202
        num_examples: 8
      - name: test
        num_bytes: 713732
        num_examples: 250
    download_size: 4915944
    dataset_size: 4676934
---

# Dataset Card for MGSM

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-instances)
  - [Data Splits](#data-instances)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** https://openai.com/blog/grade-school-math/
- **Repository:** https://github.com/openai/grade-school-math
- **Paper:** https://arxiv.org/abs/2110.14168
- **Leaderboard:** [Needs More Information]
- **Point of Contact:** [Needs More Information]

### Dataset Summary

Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems, proposed in the paper [Language models are multilingual chain-of-thought reasoners](http://arxiv.org/abs/2210.03057).

The same 250 problems from [GSM8K](https://arxiv.org/abs/2110.14168) are each translated via human annotators in 10 languages. The 10 languages are:
- Spanish
- French
- German
- Russian
- Chinese
- Japanese
- Thai
- Swahili
- Bengali
- Telugu

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

You can find the input and targets for each of the ten languages (and English) as `.tsv` files.
We also include few-shot exemplars that are also manually translated from each language in `exemplars.py`.

### Supported Tasks and Leaderboards

[Needs More Information]

### Languages

The same 250 problems from [GSM8K](https://arxiv.org/abs/2110.14168) are each translated via human annotators in 10 languages. The 10 languages are:
- Spanish
- French
- German
- Russian
- Chinese
- Japanese
- Thai
- Swahili
- Bengali
- Telugu

## Dataset Structure

### Data Instances

Each instance in the train split contains:
- a string for the grade-school level math question
- a string for the corresponding answer with chain-of-thought steps.
- the numeric solution to the question
- the equation solution to the question

```python
{'question': 'Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?',
 'answer': 'Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.',
 'answer_number': 11,
 'equation_solution': '5 + 6 = 11.'}
```

Each instance in the test split contains:
- a string for the grade-school level math question
- the numeric solution to the question

```python
{'question': "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
 'answer': None,
 'answer_number': 18,
 'equation_solution': None}
```

### Data Fields

The data fields are the same among `train` and `test` splits.

- question: The question string to a grade school math problem.

- answer: The full solution string to the `question`. It contains multiple steps of reasoning with calculator annotations and the final numeric solution.

- answer_number: The numeric solution to the `question`.

- equation_solution: The equation solution to the `question`.

### Data Splits

- The train split includes 8 few-shot exemplars that are also manually translated from each language.
- The test split includes the same 250 problems from GSM8K translated via human annotators in 10 languages.

| name   |train|test      |
|--------|----:|---------:|
|en      |  8  |   250    |
|es      |  8  |   250    |
|fr      |  8  |   250    |
|de      |  8  |   250    |
|ru      |  8  |   250    |
|zh      |  8  |   250    |
|ja      |  8  |   250    |
|th      |  8  |   250    |
|sw      |  8  |   250    |
|bn      |  8  |   250    |
|te      |  8  |   250    |

## Dataset Creation

### Curation Rationale

[Needs More Information]

### Source Data

#### Initial Data Collection and Normalization

From the paper:

> We initially collected a starting set of a thousand problems and natural language solutions by hiring freelance contractors on Upwork (upwork.com). We then worked with Surge AI (surgehq.ai), an NLP data labeling platform, to scale up our data collection. After collecting the full dataset, we asked workers to re-solve all problems, with no workers re-solving problems they originally wrote. We checked whether their final answers agreed with the original solu- tions, and any problems that produced disagreements were either repaired or discarded. We then performed another round of agreement checks on a smaller subset of problems, finding that 1.7% of problems still produce disagreements among contractors. We estimate this to be the fraction of problems that con- tain breaking errors or ambiguities. It is possible that a larger percentage of problems contain subtle errors.

#### Who are the source language producers?

[Needs More Information]

### Annotations

#### Annotation process

[Needs More Information]

#### Who are the annotators?

Surge AI (surgehq.ai)

### Personal and Sensitive Information

[Needs More Information]

## Considerations for Using the Data

### Social Impact of Dataset

[Needs More Information]

### Discussion of Biases

[Needs More Information]

### Other Known Limitations

[Needs More Information]

## Additional Information

### Dataset Curators

[Needs More Information]

### Licensing Information

The GSM8K dataset is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Citation Information

```bibtex
@article{cobbe2021gsm8k,
    title={Training Verifiers to Solve Math Word Problems},
    author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
    journal={arXiv preprint arXiv:2110.14168},
    year={2021}
}
@misc{shi2022language,
    title={Language Models are Multilingual Chain-of-Thought Reasoners}, 
    author={Freda Shi and Mirac Suzgun and Markus Freitag and Xuezhi Wang and Suraj Srivats and Soroush Vosoughi and Hyung Won Chung and Yi Tay and Sebastian Ruder and Denny Zhou and Dipanjan Das and Jason Wei},
    year={2022},
    eprint={2210.03057},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

### Contributions

Thanks to [@juletx](https://github.com/juletx) for adding this dataset.
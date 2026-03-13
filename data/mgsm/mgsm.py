# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multilingual Grade School Math Benchmark (MGSM)"""

import csv
from .exemplars import (
    EXEMPLAR_NUMBER_ANSWERS,
    EXEMPLAR_EQUATION_SOLUTIONS,
    MGSM_EXEMPLARS,
)
import datasets


_CITATION = """\
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
"""

_DESCRIPTION = """\
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

You can find the input and targets for each of the ten languages (and English) as `.tsv` files.
We also include few-shot exemplars that are also manually translated from each language in `exemplars.py`.
"""

_HOMEPAGE = "https://github.com/google-research/url-nlp/tree/main/mgsm"

_LICENSE = "CC BY SA 4.0"

_BASE_URL = "mgsm_{lang}.tsv"

_LANG = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]


class MGSM(datasets.GeneratorBasedBuilder):
    """Multilingual Grade School Math Benchmark (MGSM)"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=lang,
            description="Multilingual Grade School Math Benchmark (MGSM) {lang} set",
            version=datasets.Version("1.0.0"),
        )
        for lang in _LANG
    ]

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "answer_number": datasets.Value("int32"),
                "equation_solution": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        name = self.config.name

        filepaths = dl_manager.download_and_extract(
            {
                datasets.Split.TRAIN: "exemplars.py",
                datasets.Split.TEST: _BASE_URL.format(lang=name),
            }
        )

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": path},
            )
            for split, path in filepaths.items()
        ]

    def _generate_examples(self, filepath):
        if filepath.endswith(".py"):
            name = self.config.name
            examples = MGSM_EXEMPLARS[name]
            number_answers = EXEMPLAR_NUMBER_ANSWERS
            equation_solutions = EXEMPLAR_EQUATION_SOLUTIONS
            for key, data in examples.items():
                yield key, {
                    "question": data["q"],
                    "answer": data["a"],
                    "answer_number": number_answers[int(key) - 1],
                    "equation_solution": equation_solutions[int(key) - 1],
                }
        else:
            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(
                    csv_file,
                    quotechar='"',
                    delimiter="\t",
                )
                for key, row in enumerate(csv_reader):
                    yield key, {
                        "question": row[0],
                        "answer": None,
                        "answer_number": int(row[1].replace(",", "")),
                        "equation_solution": None,
                    }

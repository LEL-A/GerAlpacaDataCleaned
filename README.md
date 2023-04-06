# 🇩🇪 🦙 🛁 Cleaned German Alpaca Dataset

Welcome to the Cleaned German Alpaca Dataset repository!
This repository hosts cleaned, curated and translated versions of the
[Cleaned Alpaca Dataset](https://github.com/gururise/AlpacaDataCleaned).

## Datasets

### Dataset 1

- `translated_german_alpaca.json`: The raw German translated Cleaned Alpaca Dataset.
Translation was done via the `facebook/wmt19-en-de` model from the Hugging Face Model Hub.
- `Translate-Cleaned-Alpaca-Dataset.ipynb`: the code for translation

### Dataset 2

- `translated_german_alpaca_02.json`: The second raw German translated Cleaned Alpaca Dataset.
Translation was done via the `transformer.wmt19.en-de` 4-model ensemble from
[fairseq](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md).
- `Translate-Cleaned-Alpaca-Dataset.ipynb`: the code for translation

JSON attributes:
- `instruction`: the instruction part of the prompt
- `input`: the input part of the prompt
- `output`: the output / answer part of the prompt
- `output_cliped`: Some outputs were too long to translate.
Mostly this was source code. This output was replaced by an empty string.
This attribute marks this with the help of a boolean variable.
So this prompt (with the value of `True`) should not be used any further,
because it is incomplete.

## Contributions

With over 52k entries, several issues still exist. Please help out by submitting a pull-request.

## Goals

The primary goal of this project is to provide a cleaned and curated version of a German Alpaca dataset that will improve the performance of NLP models trained on this data.
By removing errors and inconsistencies, the goal is to improve performance of the fine-tuned models.

## Acknowledgments

We would like to thank the authors of the Cleaned Alpaca dataset for their effort.

We would like to thank the original creators of the Alpaca datasets for making their data available to the public.

## Licensing

The Cleaned German Alpaca Dataset is licensed under [CC BY NC 4.0](https://github.com/LEL-A/GerAlpacaDataCleaned/blob/main/DATA_LICENSE).

The software and tools in this repository is licensed under the **MIT License** (the "License");
you may not use this file except in compliance with the License. You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/LEL-A/GerAlpacaDataCleaned/blob/main/LICENSE) in the repository.

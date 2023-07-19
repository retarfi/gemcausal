# gemcausal
Generator or Encoder Model for Causal tasks

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue">
  <a href="https://github.com/retarfi/gemcausal/actions/workflows/format.yml">
    <img alt="Test" src="https://github.com/retarfi/gemcausal/actions/workflows/format.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/retarfi/gemcausal">
    <img alt="codecov" src="https://codecov.io/gh/retarfi/gemcausal/branch/main/graph/badge.svg?token=JR2JR5L653">
  </a>
</p>


## Quick Start
### Installation
```sh
poetry lock
poetry install
poetry add torch --source torch_cu117
```

### Run Evaluation
```sh
poetry run python main.py \
<openai|hf-encoder> \
--task_type <sequence-classification|span-detection> \
--dataset_type PDTB \
```

## Data Preprocessing
### UniCausal
Please download processed data from [Unicausal data](https://github.com/tanfiona/UniCausal/tree/main/data/splits) (for AltLex) or process with [UniCausal Jupyter files](https://github.com/tanfiona/UniCausal/tree/main/processing) by yourself, and save downloaded or processed csv files into `./data/`.

## Number of Examples
The number of samples of dev set is as same as that of test set .

| Corpus | Split | Sequence Classification | Span Detection |
| ---- | ---- | ---- | ---- |
| PDTB | All | 42850 | 7294 |
|  | Test | 8083 | 1300 |




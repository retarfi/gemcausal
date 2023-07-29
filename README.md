# gemcausal

Generator or Encoder Model for Causal tasks

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue">
  <a href="https://github.com/retarfi/gemcausal/actions/workflows/format.yml">
    <img alt="Test" src="https://github.com/retarfi/gemcausal/actions/workflows/format.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/retarfi/gemcausal">
    <img alt="codecov" src="https://codecov.io/gh/retarfi/gemcausal/branch/main/graph/badge.svg?token=S3C8W7KKDE">
  </a>
</p>

## Available Tasks

|      Task \ Domain      | <div style="text-align: center;">General</div>                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                      Financial                                      | Financial & Multilingual (Japanese) |
| :---------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------: | :---------------------------------: |
| Sequence Classification | 5 tasks from [UniCausal](https://github.com/tanfiona/UniCausal):<br><ul><li>[AltLex](https://github.com/chridey/altlex)</li><li>[CTB (CausalTimeBank)](https://github.com/paramitamirza/Causal-TimeBank)</li><li>[ESL (EventStoryLine V1.0)](https://github.com/tommasoc80/EventStoryLine)</li><li>[PDTB (Penn Discourse Treebank V3.0)](https://catalog.ldc.upenn.edu/LDC2019T05)</li><li>[SemEval (SemEval 2010 Task 8)](https://semeval2.fbk.eu/semeval2.php?location=tasks&taskid=11)</li> | [FinCausal 2020](https://github.com/yseop/YseopLab/tree/develop/FNP_2020_FinCausal) |                 Financial Results & Nikkei News (Not publicly available, [paper in Japanese](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/D11-3.pdf))                 |
|     Span Detection      | 2 tasks from [UniCausal](https://github.com/tanfiona/UniCausal):<br><ul><li>[AltLex](https://github.com/chridey/altlex)</li><li>[PDTB (Penn Discourse Treebank V3.0)](https://catalog.ldc.upenn.edu/LDC2019T05)</li>                                                                                                                                                                                                                                                                           | [FinCausal 2020](https://github.com/yseop/YseopLab/tree/develop/FNP_2020_FinCausal) |                 Financial Results (Not publicly available, [paper in Japanese](https://search.ieice.org/bin/summary.php?id=j98-d_5_811))                 |
|  Chain Classification   | <div style="text-align: center;">[ReCo](https://github.com/waste-wood/reco)</div>                                                                                                                                                                                                                                                                                                                                                                                                              |                                         TBA                                         |                 TBA                 |

## Quick Start

### Installation

```sh
poetry lock
poetry install
poetry add torch --source torch_cu117
```

### Run Evaluation

See the Available Tasks table for available `--task_type` and `--dataset_type` combinations.

When using OpenAI API:

```sh
OPENAI_API_KEY=XXX poetry run python main.py \
openai \
--task_type <sequence_classification|span_detection|chain_classification> \
--dataset_type <altlex|ctb|esl|fincausal|pdtb|reco|semeval> \
--data_dir data/ \
--test_samples 200 \
--output_dir materials/result/ \
--model gpt-3.5-turbo \
--template template/openai_sequence_classification.json \
--shot 2
```

When using a HuggingFace encoder model:

```sh
poetry run python main.py \
hf-encoder \
--task_type <sequence_classification|span_detection|chain_classification> \
--dataset_type <altlex|ctb|esl|fincausal|pdtb|reco|semeval> \
--data_dir data/ \
--test_samples 200 \
--output_dir materials/result/ \
--model_name google/bert_uncased_L-2_H-128_A-2 \
--lr 5e-6 7e-6 1e-5 2e-5 3e-5 5e-5 \
--train_batch_size 32 \
--eval_batch_size 2 \
--max_epochs 10
```

## Data Preprocessing

For download and preprocess information, see [data/README.md](data/README.md)

## Number of Examples

For sequence classification and span detection:

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Corpus</th>
      <th colspan="3" style="text-align: center;">Sequence Classification</th>
      <th colspan="3" style="text-align: center;">Span Detection</th>
    </tr>
    <tr>
      <th style="text-align: center;">Train</th>
      <th style="text-align: center;">Valid</th>
      <th style="text-align: center;">Test</th>
      <th style="text-align: center;">Train</th>
      <th style="text-align: center;">Valid</th>
      <th style="text-align: center;">Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">AltLex</td>
      <td style="text-align: center;">462</td>
      <td style="text-align: center;">115</td>
      <td style="text-align: center;">401</td>
      <td style="text-align: center;">221</td>
      <td style="text-align: center;">55</td>
      <td style="text-align: center;">100</td>
    </tr>
    <tr>
      <td style="text-align: center;">CTB</td>
      <td style="text-align: center;">1569</td>
      <td style="text-align: center;">316</td>
      <td style="text-align: center;">316</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td style="text-align: center;">ESL</td>
      <td style="text-align: center;">1768</td>
      <td style="text-align: center;">232</td>
      <td style="text-align: center;">232</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td style="text-align: center;">FinCausal</td>
      <td style="text-align: center;">17060</td>
      <td style="text-align: center;">2133</td>
      <td style="text-align: center;">2133</td>
      <td style="text-align: center;">1087</td>
      <td style="text-align: center;">136</td>
      <td style="text-align: center;">136</td>
    </tr>
    <tr>
      <td style="text-align: center;">PDTB</td>
      <td style="text-align: center;">26684</td>
      <td style="text-align: center;">8083</td>
      <td style="text-align: center;">8083</td>
      <td style="text-align: center;">4694</td>
      <td style="text-align: center;">1300</td>
      <td style="text-align: center;">1300</td>
    </tr>
    <tr>
      <td style="text-align: center;">SemEval</td>
      <td style="text-align: center;">6380</td>
      <td style="text-align: center;">1595</td>
      <td style="text-align: center;">2715</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
  </tbody>
</table>

For chain classification:

| Corpus | Train | Valid | Test |
| :----: | :---: | :---: | :--: |
|  ReCo  | 3111  |  417  | 672  |

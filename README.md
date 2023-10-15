
<h1 align="center">GemCausal: Generator or Encoder Model for Causal tasks</h1>


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
| Sequence Classification | 5 tasks from [UniCausal](https://github.com/tanfiona/UniCausal):<br><ul><li>[AltLex](https://github.com/chridey/altlex)</li><li>[BECAUSE v2.0](https://github.com/duncanka/BECAUSE/tree/2.0)</li><li>[CTB (CausalTimeBank)](https://github.com/paramitamirza/Causal-TimeBank)</li><li>[ESL (EventStoryLine V1.0)](https://github.com/tommasoc80/EventStoryLine)</li><li>[PDTB (Penn Discourse Treebank V3.0)](https://catalog.ldc.upenn.edu/LDC2019T05)</li><li>[SemEval (SemEval 2010 Task 8)](https://semeval2.fbk.eu/semeval2.php?location=tasks&taskid=11)</li> | [FinCausal 2020](https://github.com/yseop/YseopLab/tree/develop/FNP_2020_FinCausal) |                 Financial Results & Nikkei News (Not publicly available, [paper in Japanese](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/D11-3.pdf))                 |
|     Span Detection      | 2 tasks from [UniCausal](https://github.com/tanfiona/UniCausal):<br><ul><li>[AltLex](https://github.com/chridey/altlex)</li><li>[BECAUSE v2.0](https://github.com/duncanka/BECAUSE/tree/2.0)</li><li>[PDTB (Penn Discourse Treebank V3.0)](https://catalog.ldc.upenn.edu/LDC2019T05)</li>                                                                                                                                                                                                                                                                           | [FinCausal 2020](https://github.com/yseop/YseopLab/tree/develop/FNP_2020_FinCausal) |                 Financial Results (Not publicly available, [paper in Japanese](https://search.ieice.org/bin/summary.php?id=j98-d_5_811))                 |
|  Chain Classification   | <div style="text-align: center;">[ReCo](https://github.com/waste-wood/reco)</div>                                                                                                                                                                                                                                                                                                                                                                                                              |                                         TBA                                         |                 TBA                 |

## Quick Start

### Installation

```sh
poetry lock
poetry install --without torch
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

There are other arguments such as `--filter_num_sent`, `--filter_num_causal`, `--filter_plicit_type`, and `--evaluate_by_word` (for OpenAI models).<br>
Please see details using `--help` flag.

## Data Preprocessing

For download and preprocess information, see [data/README.md](data/README.md)


## Datasets Datails
### AltLex
AltLex ([Hidey and McKeown, 2016](http://dx.doi.org/10.18653/v1/P16-1135)) investigates causal relations in English Wikipedia articles, focusing on alternative lexicalization connectives (AltLex) within a single sentence. 
These AltLex connectives appear in a broader range of linguistic forms compared to explicit markers, exemplified by phrases such as "This may help explain why" and "This activity produced." 
However, one limitation of this dataset is its assumption that all words preceding and succeeding the causal cues are the sole spans corresponding to the cause and effect intended for extraction.

### BECauSE V2.0 (BECauSE)
BECauSE V2.0 (BECauSE) ([Dunietz et al., 2017](http://dx.doi.org/10.18653/v1/W17-0812)) is designed to annotate intra-sentential explicit causal relations, aiming to encapsulate the diverse constructions utilized to convey cause and effect. 
The arguments directed to causal relation instances are phrase units.
The dataset is derived from several sources, namely the New York Times Annotated Corpus (NYT) ([Sandhaus, 2008](https://doi.org/10.35111/77ba-9x74)), Penn Treebank (PTB) ([Marcus et al., 1994](https://doi.org/10.3115/1075812.1075835)), Congressional Hearings of the 2014 NLP Unshared Task in PoliInformatics (CHRG) ([Smith et al., 2014](http://dx.doi.org/10.3115/v1/W14-2505)), and the Manually Annotated Sub-Corpus (MASC) ([Ide et al., 2010](https://doi.org/10.35111/ctg7-5698)).

### Causal-TimeBank (CTB) and EventStoryLine (ESL)
Causal-TimeBank (CTB) ([Mirza et al., 2014](http://dx.doi.org/10.3115/v1/W14-0702);[ Mirza and Tonelli, 2014](https://aclanthology.org/C14-1198)) and EventStoryLine (ESL) ([Caselli and Vossen, 2016](http://dx.doi.org/10.18653/v1/W16-5708),[ 2017](http://dx.doi.org/10.18653/v1/W17-2711)) datasets are widely recognized in the field of Event Causality Identification (ECI), which concentrates on discerning causal links between events within textual data.
For instance, the ECI model identifies a causal link between "earthquake" and  "tsunami" in the sentence "The earthquake generated a tsunami."
The CTB dataset originates from the TimeBank corpus of the TempEval-3 task ([UzZaman et al., 2013](https://aclanthology.org/S13-2001)) and is designed to annotate solely explicit causal relations via a rule-driven algorithm. 
In contrast, ESL derives from an extended version of the EventCorefBank (ECB+) ([Cybulska and Vossen, 2014](https://aclanthology.org/L14-1646/)) and encompasses explicit and implicit causality.
Both datasets address intra-sentential as well as inter-sentential causality.
However, a limitation is evident in that only the initial words of an event are tagged, leading to the omission of the context from the extracted arguments.

### Penn Discourse Treebank V3.0 (PDTB)
Penn Discourse Treebank V3.0 (PDTB) ([Prasad et al., 2019](https://doi.org/10.35111/qebf-gk47)) represents the third installment of the Penn Discourse Treebank project and stands as the most extensive annotated corpus dedicated to discourse relations.
This project primarily focuses on annotating discourse relations present in the Wall Street Journal (WSJ) segment of Treebank-2.
The uniqueness of PDTB lies in its ability to annotate not only overt discourse relations with explicit connectives but also those that are conveyed through varied forms containing inter-sentential and implicit causality.
It is pertinent to note that causal relations within clauses are excluded from annotation.
Although causal relation is not the main focus of the dataset, one of the relations annotated by PDTB is treated as causality in our study.

### SemEval-2010 Task8 (SemEval)
In the SemEval-2010 Task8 (SemEval) ([Hendrickx et al., 2010](https://aclanthology.org/S10-1006)), the primary emphasis is not causal relations but the multi-faceted classification of semantic relations between noun phrase pairs.
The dataset restricts relation instances to those present within a single sentence; however, they are not restricted to merely explicit instances.
In the nine relations defined by SemEval, we treat the "Cause-Effect (CE)" relation as a causal relation.
We select the six datasets mentioned above based on the previous study related to causal text mining ([Tan et al., 2023](https://doi.org/10.1007/978-3-031-39831-5_23)).

### FinCausal 2020 (FinCausal)
FinCausal 2020 (FinCausal) ([Mariko et al., 2020](https://aclanthology.org/2020.fnp-1.3)) dataset, extracted from financial news articles published in 2019 by Quam and additional data from the EDGAR Database of the U.S. Securities and Exchange Commission (SEC), belongs to the financial domain.
This dataset is made for the FinCausal-2020 Shared Task on "Financial Document Causality Detection," which aims to develop the ability to use external information to explain why changes in market and corporate financial conditions occur.
Key features of the FinCausal 2020 dataset are its restriction of effect spans to quantifiable facts and the common occurrence of causal arguments being presented as entire sentences.
The dataset is also annotated for implicit and inter-sentential causality.

### Japanese financial statement summaries (JFS) and Nikkei news articles (Nikkei)
Our study also employs Japanese datasets sourced from Japanese financial statement summaries (JFS) and Nikkei news articles (Nikkei) ([Sakaji et al., 2017](https://doi.org/10.1109/SSCI.2017.8285265);[ Kobayashi et al., 2023](https://www.anlp.jp/proceedings/annual_meeting/2023/pdf_dir/D11-3.pdf)).
JFS are mandated disclosure documents for publicly listed companies, providing details on business performance and financial condition, typically characterized by specialized and standardized phrasing.
Nikkei refers to a financial newspaper published by Nikkei, Inc.
In these datasets, causal relations present within a single sentence or spanning two adjacent sentences are analyzed, employing explicit markers automatically generated through the bootstrapping method.
In addition to datasets labeled for the presence or absence of causality in sentences, another dataset from JFS is annotated for cause and effect spans by an investor with 15 years of experience.
From examining 30 files in the latter dataset, 478 causal relations are identified.

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
      <td style="text-align: center;">BECAUSE</td>
      <td style="text-align: center;">852</td>
      <td style="text-align: center;">51</td>
      <td style="text-align: center;">51</td>
      <td style="text-align: center;">475</td>
      <td style="text-align: center;">33</td>
      <td style="text-align: center;">33</td>
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

# Data Preparation
## AltLex, CTB, ESL, SemEval
Download csv files from [UniCausal/data/splits/](https://github.com/tanfiona/UniCausal/tree/main/data/splits) and save them in `data/`.

## BECauSE
Download files from [BECAUSE v2.0 Data Release](https://github.com/duncanka/BECAUSE/tree/2.0).<br>
The BECAuSE dataset is divided into four sub-corpora. For Penn Treebank (PTB) and the New York Times Annotated Corpus (NYT), download from [LDC](https://catalog.ldc.upenn.edu/LDC99T42) or somewhere (you may pay or use membership).[<sup>1</sup>]<br>
After download, process the data with [UniCausal/processing/Get BECAUSE.ipynb](https://github.com/tanfiona/UniCausal/blob/main/processing/Get%20BECAUSE.ipynb).<br>
The output `because.csv` is placed in `data/`.

<a name="note1"></a>
<sup>1</sup> Note that we were not able to obtain the raw NYT files that require subscriptions. Therefore, we experimented with the other three corpora in this study.

## PDTB
Download from [LDC](https://catalog.ldc.upenn.edu/LDC2019T05) or somewhere (you may pay or use membership).<br>
After download, process the data with [UniCausal/processing/Get PDTB.ipynb](https://github.com/tanfiona/UniCausal/blob/main/processing/Get%20PDTB.ipynb).<br>
The output `pdtb.csv` is placed in `data/`.

## ReCo
Download dev/test/train.json files from [ReCo/data/english](https://github.com/Waste-Wood/ReCo/tree/main/data/english) and save them in `data/reco/`.

## FinCausal (Task1&2)
Download fnp-2020-<fincausal/fincausal2>-task<1/2>.csv files from [YseopLab/FNP_2020_FinCausal/data](https://github.com/yseop/YseopLab/tree/develop/FNP_2020_FinCausal/data).<br>
After download, save these csv files in `./data` (like `./data/fnp2020-fincausal2-task2.csv`).

## Japanese Datasets
Japanese datasets are not publicly available now.

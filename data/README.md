# Data Preparation
## AltLex, CTB, ESL, SemEval
Download csv files from [UniCausal/data/splits/](https://github.com/tanfiona/UniCausal/tree/main/data/splits) and save them in `data/`.


## PDTB
Download from [LDC](https://catalog.ldc.upenn.edu/LDC2019T05) or somewhere (you may pay or use membership).<br>
After download, process the data with [UniCausal/processing/Get PDTB.ipynb](https://github.com/tanfiona/UniCausal/blob/main/processing/Get%20PDTB.ipynb).<br>
The output `pdtb.csv` is placed in `data/`.

## ReCo
Download dev/test/train.json files from [ReCo/data/english](https://github.com/Waste-Wood/ReCo/tree/main/data/english) and save them in `data/reco/`.

## FinCausal (Task1&2)
Download fnp-2020-<fincausal/fincausal2>-task<1/2>.csv files from [YseopLab/FNP_2020_FinCausal/data](https://github.com/yseop/YseopLab/tree/develop/FNP_2020_FinCausal/data).<br>
After download, save these csv files in `./data` (like `./data/fnp2020-fincausal2-task2.csv`).

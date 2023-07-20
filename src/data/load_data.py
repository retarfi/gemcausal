import os
import re
from enum import Enum
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# from .download import download_altlex, download_ctb, download_because2
from .preprocess.unicausal import get_bio_for_datasets
from .. import TaskType, DatasetType, logger


def _load_data_unicausal_sequence_classification(data_path: str) -> Dataset:
    ds = load_dataset("csv", data_files=data_path, split="train")
    ds = ds.filter(lambda x: x["eg_id"] == 0)
    ds = ds.rename_column("seq_label", "labels")
    return ds


def _load_data_unicausal_span_detection(data_path: str) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    df.drop_duplicates(subset=["corpus", "doc_id", "sent_id"], keep=False, inplace=True)
    df = df[df.seq_label == 1]
    ds = Dataset.from_pandas(df)
    ds = ds.map(get_bio_for_datasets)
    return ds


def load_data(
    task_enum: Enum,
    dataset_enum: Enum,
    data_dir: str,
    test_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    # task_data_dir: str = os.path.join(data_dir, task_name)
    # if not os.path.isdir(task_data_dir):
    #     # download
    #     logger.info(f"Download {task_name}")
    #     tmp_dir: str = os.path.join(data_dir, "tmp")
    #     if task_name == "altlex":
    #         download_altlex(tmp_dir)
    #     elif task_name == "ctb":
    #         download_ctb(tmp_dir)
    #     elif task_name == "because2":
    #         download_because2(tmp_dir)
    #     else:
    #         raise NotImplementedError(
    #             f"Downloading {task_name} is not implemented. Please downlaod manually"
    #         )

    #     # preprocess

    # save

    # load
    ds_train_val: Dataset
    ds_train: Dataset
    ds_valid: Dataset
    ds_test: Dataset
    if dataset_enum in (
        DatasetType.AltLex,
        DatasetType.CTB,
        DatasetType.ESL,
        DatasetType.PDTB,
        DatasetType.SemEval,
    ):
        if dataset_enum == DatasetType.PDTB:
            data_path: str = os.path.join(data_dir, "pdtb.csv")
            ds: Dataset
            if task_enum == TaskType.SEQUENCE_CLASSIFICATION:
                ds = _load_data_unicausal_sequence_classification(data_path)
            elif task_enum == TaskType.SPAN_DETECTION:
                ds = _load_data_unicausal_span_detection(data_path)
            else:
                raise NotImplementedError()
            test_ptn: re.Pattern = re.compile(r"wsj_(00|01|22|23|24).+")
            ds_test = ds.filter(lambda x: test_ptn.match(x["doc_id"]))
            ds_train_val = ds.filter(lambda x: not test_ptn.match(x["doc_id"]))
        else:
            dataset_path_prefix: str
            if dataset_enum == DatasetType.AltLex:
                dataset_path_prefix = "altlex"
            elif dataset_enum == DatasetType.CTB:
                dataset_path_prefix = "ctb"
            elif dataset_enum == DatasetType.ESL:
                dataset_path_prefix = "esl2"
            elif dataset_enum == DatasetType.SemEval:
                dataset_path_prefix = "semeval2010t8"
            else:
                raise NotImplementedError()
            train_val_data_path: str = os.path.join(
                data_dir, f"{dataset_path_prefix}_train.csv"
            )
            test_data_path: str = os.path.join(
                data_dir, f"{dataset_path_prefix}_test.csv"
            )
            if task_enum == TaskType.SEQUENCE_CLASSIFICATION:
                ds_train_val = _load_data_unicausal_sequence_classification(
                    train_val_data_path
                )
                ds_test = _load_data_unicausal_sequence_classification(test_data_path)
            elif task_enum == TaskType.SPAN_DETECTION:
                ds_train_val = _load_data_unicausal_span_detection(train_val_data_path)
                ds_test = _load_data_unicausal_span_detection(test_data_path)
            else:
                raise NotImplementedError()
        test_size: int
        if len(ds_test) * 2 < len(ds_train_val):
            test_size = len(ds_test)
        else:
            test_size = int(len(ds_train_val) * 0.2)
        dsd_train_val: DatasetDict = ds_train_val.train_test_split(
            test_size=test_size, shuffle=True, seed=seed
        )
        ds_train = dsd_train_val["train"]
        ds_valid = dsd_train_val["test"]
    else:
        raise NotImplementedError()

    if test_samples is not None:
        if len(ds_test) >= test_samples:
            ds_test = ds_test.select(list(range(test_samples)))
        else:
            logger.warning(
                (
                    "Test sampling is not executed because test_samples > number of "
                    f"test samples ({len(ds_test)})"
                )
            )
    dsd: DatasetDict = DatasetDict(
        {"train": ds_train, "valid": ds_valid, "test": ds_test}
    )
    logger.info(f"# of samples: {dsd.num_rows}")
    # drop and assert columns
    for key, ds_ in dsd.items():
        set_columns: set[str]
        if task_enum == TaskType.SEQUENCE_CLASSIFICATION:
            set_columns = {"text", "labels"}
        elif task_enum == TaskType.SPAN_DETECTION:
            set_columns = {"tokens", "tags"}
        else:  # pragma: no cover
            raise NotImplementedError()
        dsd[key] = ds_.remove_columns(list(set(ds_.column_names) - set_columns))
        assert set(dsd[key].column_names) == set_columns, dsd[key].column_names
    return dsd

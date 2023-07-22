import os
import re
from enum import Enum
from typing import Optional

from datasets import Dataset, DatasetDict

from .unicausal import (
    _load_data_unicausal_sequence_classification,
    _load_data_unicausal_span_detection,
)
from .reco import load_reco_dataset
from .. import TaskType, DatasetType, assert_dataset_task_pair, logger


def load_data(
    task_enum: Enum,
    dataset_enum: Enum,
    data_dir: str,
    test_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    assert_dataset_task_pair(dataset_enum=dataset_enum, task_enum=task_enum)
    ds_train_val: Dataset
    ds_train: Dataset
    ds_valid: Dataset
    ds_test: Dataset
    if dataset_enum in (
        DatasetType.altlex,
        DatasetType.ctb,
        DatasetType.esl,
        DatasetType.pdtb,
        DatasetType.semeval,
    ):
        if dataset_enum == DatasetType.pdtb:
            data_path: str = os.path.join(data_dir, "pdtb.csv")
            ds: Dataset
            if task_enum == TaskType.sequence_classification:
                ds = _load_data_unicausal_sequence_classification(data_path)
            elif task_enum == TaskType.span_detection:
                ds = _load_data_unicausal_span_detection(data_path)
            else:
                raise NotImplementedError()
            test_ptn: re.Pattern = re.compile(r"wsj_(00|01|22|23|24).+")
            ds_test = ds.filter(lambda x: test_ptn.match(x["doc_id"]))
            ds_train_val = ds.filter(lambda x: not test_ptn.match(x["doc_id"]))
        else:
            dataset_path_prefix: str
            if dataset_enum == DatasetType.altlex:
                dataset_path_prefix = "altlex"
            elif dataset_enum == DatasetType.ctb:
                dataset_path_prefix = "ctb"
            elif dataset_enum == DatasetType.esl:
                dataset_path_prefix = "esl2"
            elif dataset_enum == DatasetType.semeval:
                dataset_path_prefix = "semeval2010t8"
            else:
                raise NotImplementedError()
            train_val_data_path: str = os.path.join(
                data_dir, f"{dataset_path_prefix}_train.csv"
            )
            test_data_path: str = os.path.join(
                data_dir, f"{dataset_path_prefix}_test.csv"
            )
            if task_enum == TaskType.sequence_classification:
                ds_train_val = _load_data_unicausal_sequence_classification(
                    train_val_data_path
                )
                ds_test = _load_data_unicausal_sequence_classification(test_data_path)
            elif task_enum == TaskType.span_detection:
                ds_train_val = _load_data_unicausal_span_detection(train_val_data_path)
                ds_test = _load_data_unicausal_span_detection(test_data_path)
            else:
                raise NotImplementedError()
        test_size: int
        if len(ds_test) * 4 < len(ds_train_val):
            test_size = len(ds_test)
        else:
            test_size = int(len(ds_train_val) * 0.2)
        dsd_train_val: DatasetDict = ds_train_val.train_test_split(
            test_size=test_size, shuffle=True, seed=seed
        )
        ds_train = dsd_train_val["train"]
        ds_valid = dsd_train_val["test"]
    elif dataset_enum == DatasetType.reco:
        reco_dir: str = os.path.join(data_dir, "reco")
        assert os.path.isdir(reco_dir), f"{reco_dir} for ReCo data does not exist"
        ds_train = load_reco_dataset(os.path.join(reco_dir, "train.json"))
        ds_valid = load_reco_dataset(os.path.join(reco_dir, "dev.json"))
        ds_test = load_reco_dataset(os.path.join(reco_dir, "test.json"))
    else:  # pragma: no cover
        raise NotImplementedError()

    if test_samples is not None:
        if len(ds_test) >= test_samples:
            ds_test = ds_test.select(list(range(test_samples)))
        else:
            logger.warning(
                (
                    "Test sampling is not executed because test_samples > number of "
                    "test samples (%s)",
                    len(ds_test),
                )
            )
    dsd: DatasetDict = DatasetDict(
        {"train": ds_train, "valid": ds_valid, "test": ds_test}
    )
    logger.info("# of samples: %s", dsd.num_rows)
    # drop and assert columns
    for key, ds_ in dsd.items():
        set_columns: set[str]
        if task_enum == TaskType.sequence_classification:
            set_columns = {"text", "labels"}
        elif task_enum == TaskType.span_detection:
            set_columns = {"tokens", "tags"}
        elif task_enum == TaskType.chain_classification:
            if dataset_enum == DatasetType.reco:
                set_columns = {"events", "short_contexts", "labels"}
        else:  # pragma: no cover
            raise NotImplementedError()
        dsd[key] = ds_.remove_columns(list(set(ds_.column_names) - set_columns))
        assert set(dsd[key].column_names) == set_columns, dsd[key].column_names
    return dsd

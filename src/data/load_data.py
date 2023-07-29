import os
from enum import Enum
from typing import Optional

from datasets import Dataset, DatasetDict

from .fincausal import load_data_fincausal
from .japanese import load_data_jpfin
from .unicausal import load_data_unicausal
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
        ds_train, ds_valid, ds_test = load_data_unicausal(
            dataset_enum=dataset_enum, task_enum=task_enum, data_dir=data_dir, seed=seed
        )
    elif dataset_enum == DatasetType.fincausal:
        ds_train, ds_valid, ds_test = load_data_fincausal(
            task_enum=task_enum, data_dir=data_dir, seed=seed
        )
    elif dataset_enum in (DatasetType.jpfinresults, DatasetType.jpnikkei):
        ds_train, ds_valid, ds_test = load_data_jpfin(
            dataset_enum=dataset_enum, task_enum=task_enum, data_dir=data_dir, seed=seed
        )
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
            set_columns = {"text", "tokens", "tags"}
        elif task_enum == TaskType.chain_classification:
            if dataset_enum == DatasetType.reco:
                set_columns = {"events", "short_contexts", "labels"}
        else:  # pragma: no cover
            raise NotImplementedError()
        dsd[key] = ds_.remove_columns(list(set(ds_.column_names) - set_columns))
        assert set(dsd[key].column_names) == set_columns, dsd[key].column_names
    return dsd

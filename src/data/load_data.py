import os
from enum import Enum
from typing import Optional, List, Set, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets

from .fincausal import load_data_fincausal
from .japanese import load_data_jpfin
from .unicausal import load_data_unicausal
from .reco import load_reco_dataset
from .. import DatasetType, PlicitType, TaskType, assert_dataset_task_pair, logger


def get_columns_for_task(task_enum: Enum, dataset_enum: Enum) -> Set[str]:
    if task_enum == TaskType.sequence_classification:
        return {"example_id", "text", "labels"}
    elif task_enum == TaskType.span_detection:
        return {"example_id", "text", "tokens", "tags", "tagged_text"}
    elif task_enum == TaskType.chain_classification:
        if dataset_enum == DatasetType.reco:
            return {"example_id", "events", "short_contexts", "labels"}
    else:  # pragma: no cover
        raise NotImplementedError()


def _convert_example_ids(dataset: Dataset) -> Dataset:
    """cast to int type to match the type when concatenating dataset"""
    return dataset.map(
        lambda example: {
            "example_id": int(
                "".join([char for char in str(example["example_id"]) if char.isdigit()])
            )
        }
    )


def _remove_unnecessary_columns(
    ds_train: Dataset, ds_valid: Dataset, ds_test: Dataset, set_columns: Set[str]
) -> Tuple[Dataset, Dataset, Dataset]:
    ds_train = ds_train.remove_columns(list(set(ds_train.column_names) - set_columns))
    ds_valid = ds_valid.remove_columns(list(set(ds_valid.column_names) - set_columns))
    ds_test = ds_test.remove_columns(list(set(ds_test.column_names) - set_columns))
    return ds_train, ds_valid, ds_test


def load_dataset_for_corpus(
    task_enum: Enum,
    dataset_enum: Enum,
    sentencetype_enum: Enum,
    numcausal_enum: Enum,
    plicit_enum: Enum,
    data_dir: str,
    seed: int,
    set_columns: Set[str],
) -> Tuple[Dataset, Dataset, Dataset]:
    if dataset_enum in (
        DatasetType.altlex,
        DatasetType.because,
        DatasetType.ctb,
        DatasetType.esl,
        DatasetType.pdtb,
        DatasetType.semeval,
    ):
        ds_train, ds_valid, ds_test = load_data_unicausal(
            dataset_enum,
            task_enum,
            sentencetype_enum,
            numcausal_enum,
            plicit_enum,
            data_dir,
            seed,
        )

    elif dataset_enum == DatasetType.fincausal:
        ds_train, ds_valid, ds_test = load_data_fincausal(
            task_enum, data_dir, sentencetype_enum, plicit_enum, seed
        )

    elif dataset_enum in (DatasetType.jpfinresults, DatasetType.jpnikkei):
        assert plicit_enum == PlicitType.all
        ds_train, ds_valid, ds_test = load_data_jpfin(
            dataset_enum, task_enum, data_dir, numcausal_enum, seed
        )

    elif dataset_enum == DatasetType.reco:
        reco_dir = os.path.join(data_dir, "reco")
        assert os.path.isdir(reco_dir), f"{reco_dir} for ReCo data does not exist"
        ds_train = load_reco_dataset(os.path.join(reco_dir, "train.json"))
        ds_valid = load_reco_dataset(os.path.join(reco_dir, "dev.json"))
        ds_test = load_reco_dataset(os.path.join(reco_dir, "test.json"))

    elif dataset_enum == DatasetType.all:
        if task_enum == TaskType.sequence_classification:
            train_dataset_list = [
                "altlex",
                "because",
                "ctb",
                "esl",
                "pdtb",
                "semeval",
                "fincausal",
            ]
        elif task_enum == TaskType.span_detection:
            train_dataset_list = ["altlex", "because", "pdtb", "fincausal"]
        else:
            raise NotImplementedError()

        train_datasets, valid_datasets, test_datasets = [], [], []
        for dataset in train_dataset_list:
            _train, _valid, _test = load_dataset_for_corpus(
                task_enum,
                DatasetType[dataset],
                sentencetype_enum,
                numcausal_enum,
                plicit_enum,
                data_dir,
                seed,
                set_columns,
            )
            train_datasets.append(_train)
            valid_datasets.append(_valid)
            test_datasets.append(_test)

        ds_train = concatenate_datasets(
            [_convert_example_ids(ds) for ds in train_datasets]
        ).shuffle(seed=seed)
        ds_valid = concatenate_datasets(
            [_convert_example_ids(ds) for ds in valid_datasets]
        ).shuffle(seed=seed)
        ds_test = concatenate_datasets(
            [_convert_example_ids(ds) for ds in test_datasets]
        ).shuffle(seed=seed)

    else:
        raise NotImplementedError()

    return _remove_unnecessary_columns(ds_train, ds_valid, ds_test, set_columns)


def load_data(
    task_enum: Enum,
    dataset_enum: Enum,
    sentencetype_enum: Enum,
    numcausal_enum: Enum,
    plicit_enum: Enum,
    data_dir: str,
    test_dataset_enum: Optional[Enum] = None,
    test_samples: Optional[int] = None,
    seed: int = 42,
) -> DatasetDict:
    if test_dataset_enum is None:
        test_dataset_enum = dataset_enum
    assert_dataset_task_pair(dataset_enum=dataset_enum, task_enum=task_enum)
    assert_dataset_task_pair(dataset_enum=test_dataset_enum, task_enum=task_enum)
    ds_train: Dataset
    ds_valid: Dataset
    ds_test: Dataset
    set_columns: Set[str] = get_columns_for_task(task_enum, dataset_enum)

    ds_train, ds_valid, _ = load_dataset_for_corpus(
        task_enum,
        dataset_enum,
        sentencetype_enum,
        numcausal_enum,
        plicit_enum,
        data_dir,
        seed,
        set_columns,
    )
    _, _, ds_test = load_dataset_for_corpus(
        task_enum,
        test_dataset_enum,
        sentencetype_enum,
        numcausal_enum,
        plicit_enum,
        data_dir,
        seed,
        set_columns,
    )

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
    return dsd

import os
import re
from enum import Enum
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

# from .download import download_altlex, download_ctb, download_because2
from .preprocess.unicausal import get_bio_for_datasets
from .. import TaskType, DatasetType, logger


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
    data_path: str
    processed_external_dir: str = "processed_external"
    ds_train: Dataset
    ds_valid: Dataset
    ds_test: Dataset
    if dataset_enum == DatasetType.PDTB:
        data_path = os.path.join(data_dir, processed_external_dir, "pdtb.csv")
        ds: Dataset
        if task_enum == TaskType.SEQUENCE_CLASSIFICATION:
            ## Sequcence Classification: text -> seq_label
            ## Tips: You will need to de-duplicate the dataset by taking only the first "eg_id" (=0) as the main row. This is equivalent to doing a group by with "corpus, doc_id, sent_id" columns.
            ## つまり、eq_id=0のもののみfilterすればよい
            ds = load_dataset("csv", data_files=data_path, split="train")
            ds = ds.filter(lambda x: x["eg_id"] == 0)
            ds = ds.rename_column("seq_label", "labels")
        elif task_enum == TaskType.SPAN_DETECTION:
            df: pd.DataFrame = pd.read_csv(data_path)
            # drop all duplicates with text
            # It mainly drops multiple causal relations in one sequence
            df.drop_duplicates(
                subset=["corpus", "doc_id", "sent_id"], keep=False, inplace=True
            )
            df = df[df.seq_label == 1]
            ds = Dataset.from_pandas(df)
            ds = ds.map(get_bio_for_datasets)
        else:
            raise NotImplementedError()
        test_ptn: re.Pattern = re.compile(r"wsj_(00|01|22|23|24).+")
        ds_test = ds.filter(lambda x: test_ptn.match(x["doc_id"]))
        ds_train_val: Dataset = ds.filter(lambda x: not test_ptn.match(x["doc_id"]))
        dsd_train_val: DatasetDict = ds_train_val.train_test_split(
            test_size=len(ds_test), shuffle=True, seed=seed
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
                f"Test sampling is not executed because test_samples > number of test samples ({len(ds_test)})"
            )
    dsd: DatasetDict = DatasetDict(
        {"train": ds_train, "valid": ds_valid, "test": ds_test}
    )
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

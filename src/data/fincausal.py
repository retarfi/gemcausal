import os
from enum import Enum
from typing import Any, Optional

import nltk
import pandas as pd
from datasets import Dataset, DatasetDict

from .. import TaskType
from .split_dataset import split_train_valid_test_dataset


def _filter_data_by_num_sent(
    ds: Dataset, filter_num_sent: Optional[str] = None
) -> Dataset:
    if filter_num_sent == "intra":
        ds = ds.filter(lambda x: len(nltk.sent_tokenize(x["Text"])) == 1)
    elif filter_num_sent == "inter":
        ds = ds.filter(lambda x: len(nltk.sent_tokenize(x["Text"])) >= 2)
    elif filter_num_sent is None:
        pass
    else:  # pragma: no cover
        raise NotImplementedError()
    return ds


def load_data_fincausal(
    task_enum: Enum,
    data_dir: str,
    seed: int,
    filter_num_sent: Optional[str] = None,
) -> tuple[Dataset, Dataset, Dataset]:
    csv_prefix: str = "fnp2020-fincausal"
    task_id: int
    if task_enum == TaskType.sequence_classification:
        task_id = 1
    elif task_enum == TaskType.span_detection:
        task_id = 2
    else:  # pragma: no cover
        raise NotImplementedError()
    # load trial (fincausal-) and practice (fincausal2-) data
    df: pd.DataFrame = pd.concat(
        [
            pd.read_csv(
                os.path.join(data_dir, f"{csv_prefix}{x}-task{task_id}.csv"),
                sep="; ",
                engine="python",
            )
            for x in ["", "2"]
        ]
    )
    df.dropna(subset=["Text"], inplace=True)
    df.drop(columns=["Index"], inplace=True)
    ds: Dataset
    if task_enum == TaskType.sequence_classification:
        ds = Dataset.from_pandas(df.drop_duplicates(subset=["Text"]))
        ds = _filter_data_by_num_sent(ds, filter_num_sent)
        ds = ds.rename_columns({"Text": "text", "Gold": "labels"})
    elif task_enum == TaskType.span_detection:
        ds = Dataset.from_pandas(df.drop_duplicates(subset=["Text"], keep=False))
        ds = _filter_data_by_num_sent(ds, filter_num_sent)
        nltk.download("punkt")

        def tokenize_by_word(example: dict[str, Any]) -> dict[str, Any]:
            example["tokens"] = nltk.tokenize.word_tokenize(example["Text"])
            tokens_cause: list[str] = nltk.tokenize.word_tokenize(example["Cause"])
            tokens_effect: list[str] = nltk.tokenize.word_tokenize(example["Effect"])
            len_c: int = len(tokens_cause)
            len_e: int = len(tokens_effect)
            tags: list[str] = []
            idx: int = 0
            # for i in range(len(example["tokens"])):
            while idx < len(example["tokens"]):
                if example["tokens"][idx : idx + len_c] == tokens_cause:
                    tags.append("B-C")
                    for _ in range(len_c - 1):
                        tags.append("I-C")
                    idx += len_c
                elif example["tokens"][idx : idx + len_e] == tokens_effect:
                    tags.append("B-E")
                    for _ in range(len_e - 1):
                        tags.append("I-E")
                    idx += len_e
                else:
                    tags.append("O")
                    idx += 1
            example["tags"] = tags
            return example

        ds = ds.map(tokenize_by_word)
        ds = ds.rename_columns({"Text": "text"})
    else:  # pragma: no cover
        raise NotImplementedError()

    ds_train: Dataset
    ds_valid: Dataset
    ds_test: Dataset
    ds_train, ds_valid, ds_test = split_train_valid_test_dataset(ds, seed)
    return ds_train, ds_valid, ds_test

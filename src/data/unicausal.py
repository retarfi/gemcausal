import os
import re
from typing import Any
from enum import Enum

import pandas as pd
from datasets import Dataset, DatasetDict, Value, load_dataset

from .. import TaskType, DatasetType


def cast_column_to_int(ds: Dataset, column: str) -> Dataset:
    features = ds.features.copy()
    features[column] = Value("int64")
    ds = ds.cast(features)
    return ds


def remove_tag(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub(r"</*[A-Z]+\d*>", "", tok)


def get_bio(text_w_pairs: str) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    ce_tags: list[str] = []
    next_tag: str = "O"
    tag: str = "O"
    for tok in text_w_pairs.split(" "):
        # Replace if special
        if "<ARG0>" in tok:
            tok = re.sub("<ARG0>", "", tok)
            tag = "B-C"
            next_tag = "I-C"
        elif "</ARG0>" in tok:
            tok = re.sub("</ARG0>", "", tok)
            tag = "I-C"
            next_tag = "O"
        elif "<ARG1>" in tok:
            tok = re.sub("<ARG1>", "", tok)
            tag = "B-E"
            next_tag = "I-E"
        elif "</ARG1>" in tok:
            tok = re.sub("</ARG1>", "", tok)
            tag = "I-E"
            next_tag = "O"

        tokens.append(remove_tag(tok))
        ce_tags.append(tag)
        tag = next_tag
    return tokens, ce_tags


def get_bio_for_datasets(example: dict[str, Any]) -> dict[str, Any]:
    tokens: list[str]
    tags: list[str]
    tokens, tags = get_bio(example["text_w_pairs"])
    example["tokens"] = tokens
    example["tags"] = tags
    return example


def _load_data_unicausal_sequence_classification(data_path: str) -> Dataset:
    ds = load_dataset("csv", data_files=data_path, split="train")
    ds = ds.filter(lambda x: x["eg_id"] == 0)
    ds = ds.rename_columns({"seq_label": "labels", "eg_id": "example_id"})
    return ds


def _load_data_unicausal_span_detection(data_path: str) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    df.drop_duplicates(subset=["corpus", "doc_id", "sent_id"], keep=False, inplace=True)
    df = df[df.seq_label == 1]
    ds = Dataset.from_pandas(df)
    ds = ds.map(get_bio_for_datasets)
    ds = ds.rename_column("eg_id", "example_id")
    return ds


def load_data_unicausal(
    dataset_enum: Enum, task_enum: Enum, data_dir: str, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    if dataset_enum == DatasetType.because:
        data_path: str = os.path.join(data_dir, "because.csv")
        ds: Dataset
        if task_enum == TaskType.sequence_classification:
            ds = _load_data_unicausal_sequence_classification(data_path)
        elif task_enum == TaskType.span_detection:
            ds = _load_data_unicausal_span_detection(data_path)
        else:  # pragma: no cover
            raise NotImplementedError()
        test_ptn: re.Pattern = re.compile(r"wsj_(00|01|22|23|24).+")
        ds_test = ds.filter(
            lambda x: test_ptn.match(x["doc_id"]) or x["doc_id"] == "Article247_327.ann"
        )
        ds_train_val = ds.filter(
            lambda x: not test_ptn.match(x["doc_id"])
            and x["doc_id"] != "Article247_327.ann"
        )
    elif dataset_enum == DatasetType.pdtb:
        data_path: str = os.path.join(data_dir, "pdtb.csv")
        ds: Dataset
        if task_enum == TaskType.sequence_classification:
            ds = _load_data_unicausal_sequence_classification(data_path)
        elif task_enum == TaskType.span_detection:
            ds = _load_data_unicausal_span_detection(data_path)
        else:  # pragma: no cover
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
        else:  # pragma: no cover
            raise NotImplementedError()
        train_val_data_path: str = os.path.join(
            data_dir, f"{dataset_path_prefix}_train.csv"
        )
        test_data_path: str = os.path.join(data_dir, f"{dataset_path_prefix}_test.csv")
        ds_train_val: Dataset
        if task_enum == TaskType.sequence_classification:
            ds_train_val = _load_data_unicausal_sequence_classification(
                train_val_data_path
            )
            ds_train_val = cast_column_to_int(ds_train_val, "labels")
            ds_test = _load_data_unicausal_sequence_classification(test_data_path)
            ds_test = cast_column_to_int(ds_test, "labels")
        elif task_enum == TaskType.span_detection:
            ds_train_val = _load_data_unicausal_span_detection(train_val_data_path)
            ds_test = _load_data_unicausal_span_detection(test_data_path)
        else:  # pragma: no cover
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
    return ds_train, ds_valid, ds_test

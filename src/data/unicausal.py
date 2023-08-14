import os
import re
from typing import Any, Optional, Union
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


def custom_agg(group: Any) -> pd.Series:
    result: dict[str, Union[str, int]] = {}
    for col in group.columns:
        if col == "text_w_pairs":
            # TODO: Integrate multiple causality tags. And handle nested tags separately.
            result[col] = group[col].iloc[0]
        else:
            result[col] = group[col].iloc[0]
    return pd.Series(result)


def _filter_data_by_num_sent(
    dataset_enum: Enum, ds: Dataset, filter_num_sent: Optional[str] = None
) -> Dataset:
    if dataset_enum in (
        DatasetType.altlex,
        DatasetType.because,
        DatasetType.semeval,
    ) and filter_num_sent is not None:
        raise ValueError(f"filter_num_sent is not supported for {dataset_enum}")
    if filter_num_sent == "intra":
        ds = ds.filter(lambda x: x["num_sents"] == 1)
    elif filter_num_sent == "inter":
        ds = ds.filter(lambda x: x["num_sents"] >= 2)
    elif filter_num_sent is None:
        pass
    else:  # pragma: no cover
        raise NotImplementedError()
    return ds


def _filter_data_by_num_causal(
    dataset_enum: Enum, df: pd.DataFrame, filter_num_causal: Optional[str] = None
) -> pd.DataFrame:
    if filter_num_causal == "single":
        df = df[~df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
    elif filter_num_causal == "multi":
        df = df[df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
        groups = df.groupby(["corpus", "doc_id", "sent_id"])
        df = groups.apply(custom_agg).reset_index(drop=True)
    elif filter_num_causal is None:
        groups = df.groupby(["corpus", "doc_id", "sent_id"])
        df = groups.apply(custom_agg).reset_index(drop=True)
    else:  # pragma: no cover
        raise NotImplementedError()
    return df


def _load_data_unicausal_sequence_classification(
    dataset_enum: Enum, data_path: str, filter_num_sent: Optional[str] = None
) -> Dataset:
    ds = load_dataset("csv", data_files=data_path, split="train")
    ds = ds.filter(lambda x: x["eg_id"] == 0)
    ds = ds.rename_columns({"seq_label": "labels", "eg_id": "example_id"})
    ds = _filter_data_by_num_sent(dataset_enum, ds, filter_num_sent)
    return ds


def _load_data_unicausal_span_detection(
    dataset_enum: Enum,
    data_path: str,
    filter_num_sent: Optional[str] = None,
    filter_num_causal: Optional[str] = None,
) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    df = df[df.pair_label == 1]
    df = _filter_data_by_num_causal(dataset_enum, df, filter_num_causal)
    ds = Dataset.from_pandas(df)
    ds = ds.map(get_bio_for_datasets)
    ds = ds.rename_column("eg_id", "example_id")
    ds = _filter_data_by_num_sent(dataset_enum, ds, filter_num_sent)
    return ds


def load_data_unicausal(
    dataset_enum: Enum,
    task_enum: Enum,
    data_dir: str,
    seed: int,
    filter_num_sent: Optional[str] = None,
    filter_num_causal: Optional[str] = None,
) -> tuple[Dataset, Dataset, Dataset]:
    if dataset_enum == DatasetType.because:
        data_path: str = os.path.join(data_dir, "because.csv")
        ds: Dataset
        if task_enum == TaskType.sequence_classification:
            ds = _load_data_unicausal_sequence_classification(
                dataset_enum, data_path, filter_num_sent
            )
        elif task_enum == TaskType.span_detection:
            ds = _load_data_unicausal_span_detection(
                dataset_enum, data_path, filter_num_sent, filter_num_causal
            )
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
            ds = _load_data_unicausal_sequence_classification(
                dataset_enum, data_path, filter_num_sent
            )
        elif task_enum == TaskType.span_detection:
            ds = _load_data_unicausal_span_detection(
                dataset_enum, data_path, filter_num_sent, filter_num_causal
            )
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
                dataset_enum, train_val_data_path, filter_num_sent
            )
            ds_train_val = cast_column_to_int(ds_train_val, "labels")
            ds_test = _load_data_unicausal_sequence_classification(
                dataset_enum, test_data_path, filter_num_sent
            )
            ds_test = cast_column_to_int(ds_test, "labels")
        elif task_enum == TaskType.span_detection:
            ds_train_val = _load_data_unicausal_span_detection(
                dataset_enum, train_val_data_path, filter_num_sent, filter_num_causal
            )
            ds_test = _load_data_unicausal_span_detection(
                dataset_enum, test_data_path, filter_num_sent, filter_num_causal
            )
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

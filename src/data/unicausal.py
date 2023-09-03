import os
import re
from typing import Any, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Value, load_dataset

from .. import (
    DatasetType,
    NumCausalType,
    SentenceType,
    SpanTags,
    SpanTagsFormat,
    TaskType,
)


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
    example["tagged_text"] = (
        example["text_w_pairs"]
        .replace("<ARG0>", SpanTags.cause_begin)
        .replace("</ARG0>", SpanTags.cause_end)
        .replace("<ARG1>", SpanTags.effect_begin)
        .replace("</ARG1>", SpanTags.effect_end)
    )
    return example


def custom_agg(group: Any) -> pd.Series:
    reg_cause: re.Pattern = re.compile(
        f"{SpanTags.cause_begin}(.*?){SpanTags.cause_end}"
    )
    reg_effect: re.Pattern = re.compile(
        f"{SpanTags.effect_begin}(.*?){SpanTags.effect_end}"
    )
    result: dict[str, Union[str, int, None]] = {}
    for col in group.columns:
        if col == "tagged_text":
            text: Optional[str] = group["text"].iloc[0]
            for i, idx in enumerate(
                np.argsort(
                    [
                        -(x.find(SpanTags.cause_begin))
                        for x in group["text_w_pairs"].tolist()
                    ]
                )
            ):
                cause: str = re.search(reg_cause, group[col].iloc[idx]).group(1)
                effect: str = re.search(reg_effect, group[col].iloc[idx]).group(1)
                if cause in text:
                    text = text.replace(
                        cause,
                        SpanTagsFormat.cause_begin.format(i + 1)
                        + cause
                        + SpanTagsFormat.cause_end.format(i + 1),
                    )
                else:
                    text = None
                    break
                if effect in text:
                    text = text.replace(
                        effect,
                        SpanTagsFormat.effect_begin.format(i + 1)
                        + effect
                        + SpanTagsFormat.effect_end.format(i + 1),
                    )
                else:
                    text = None
                    break
            result[col] = text
        else:
            result[col] = group[col].iloc[0]
    return pd.Series(result)


def _filter_data_by_num_sent(
    dataset_enum: Enum, ds: Dataset, sentencetype_enum: Enum
) -> Dataset:
    if (
        dataset_enum in (DatasetType.altlex, DatasetType.because, DatasetType.semeval)
        and sentencetype_enum != SentenceType.all
    ):
        raise ValueError(f"filter_num_sent is not supported for {dataset_enum}")
    if sentencetype_enum == SentenceType.intra:
        ds = ds.filter(lambda x: x["num_sents"] == 1)
    elif sentencetype_enum == SentenceType.inter:
        ds = ds.filter(lambda x: x["num_sents"] >= 2)
    return ds


def _filter_data_by_num_causal(
    dataset_enum: Enum, ds: Dataset, numcausal_enum: Enum
) -> Dataset:
    df: pd.DataFrame = ds.to_pandas()
    if numcausal_enum == NumCausalType.single:
        df = df[~df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
    else:
        if numcausal_enum == NumCausalType.multi:
            df = df[df.duplicated(subset=["corpus", "doc_id", "sent_id"], keep=False)]
        else:
            assert numcausal_enum == NumCausalType.all
        groups = df.groupby(["corpus", "doc_id", "sent_id"])
        df = groups.apply(custom_agg).reset_index(drop=True)
        # Drop nested causal with primitive way
        df.dropna(subset=["tagged_text"])
    return Dataset.from_pandas(df, preserve_index=False)


def _load_data_unicausal_sequence_classification(
    dataset_enum: Enum, data_path: str, sentencetype_enum: Enum
) -> Dataset:
    ds: Dataset = load_dataset("csv", data_files=data_path, split="train")
    ds = ds.filter(lambda x: x["eg_id"] == 0)
    ds = ds.rename_columns({"seq_label": "labels", "eg_id": "example_id"})
    ds = _filter_data_by_num_sent(dataset_enum, ds, sentencetype_enum)
    return ds


def _load_data_unicausal_span_detection(
    dataset_enum: Enum, data_path: str, sentencetype_enum: Enum, numcausal_enum: Enum
) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    df = df[df.pair_label == 1]
    # TODO: remove if no problem
    # if dataset_enum == DatasetType.because:
    #     df = df[(df.doc_id != "20020731-nyt.ann") | (df.sent_id != 120)]
    ds: Dataset = Dataset.from_pandas(df)
    ds = ds.map(get_bio_for_datasets)
    ds = ds.rename_column("eg_id", "example_id")
    ds = _filter_data_by_num_causal(dataset_enum, ds, numcausal_enum)
    ds = _filter_data_by_num_sent(dataset_enum, ds, sentencetype_enum)
    return ds


def load_data_unicausal(
    dataset_enum: Enum,
    task_enum: Enum,
    sentencetype_enum: Enum,
    numcausal_enum: Enum,
    data_dir: str,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    if dataset_enum == DatasetType.because:
        data_path: str = os.path.join(data_dir, "because.csv")
        ds: Dataset
        if task_enum == TaskType.sequence_classification:
            ds = _load_data_unicausal_sequence_classification(
                dataset_enum, data_path, sentencetype_enum
            )
        elif task_enum == TaskType.span_detection:
            ds = _load_data_unicausal_span_detection(
                dataset_enum, data_path, sentencetype_enum, numcausal_enum
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
                dataset_enum, data_path, sentencetype_enum
            )
        elif task_enum == TaskType.span_detection:
            ds = _load_data_unicausal_span_detection(
                dataset_enum, data_path, sentencetype_enum, numcausal_enum
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
                dataset_enum, train_val_data_path, sentencetype_enum
            )
            ds_train_val = cast_column_to_int(ds_train_val, "labels")
            ds_test = _load_data_unicausal_sequence_classification(
                dataset_enum, test_data_path, sentencetype_enum
            )
            ds_test = cast_column_to_int(ds_test, "labels")
        elif task_enum == TaskType.span_detection:
            ds_train_val = _load_data_unicausal_span_detection(
                dataset_enum, train_val_data_path, sentencetype_enum, numcausal_enum
            )
            ds_test = _load_data_unicausal_span_detection(
                dataset_enum, test_data_path, sentencetype_enum, numcausal_enum
            )
        else:  # pragma: no cover
            raise NotImplementedError()
    valid_size: int
    if len(ds_test) * 4 < len(ds_train_val):
        valid_size = len(ds_test)
    else:
        valid_size = int(len(ds_train_val) * 0.2)
    dsd_train_val: DatasetDict = ds_train_val.train_test_split(
        test_size=valid_size, shuffle=True, seed=seed
    )
    ds_train = dsd_train_val["train"]
    ds_valid = dsd_train_val["test"]
    return ds_train, ds_valid, ds_test

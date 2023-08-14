import os
import re
from enum import Enum

import pandas as pd
from datasets import Dataset
from transformers.models.bert_japanese import MecabTokenizer

from .. import TaskType, DatasetType
from .split_dataset import split_train_valid_test_dataset


def load_and_process_span_jpfinresults(fileprefix: str, data_dir: str) -> Dataset:
    df: pd.DataFrame = pd.read_json(
        os.path.join(data_dir, f"kessan/{fileprefix}.jsonl"),
        orient="records",
        lines=True,
    )
    df["example_id"] = [f"kessan_{fileprefix}_{i}" for i in range(len(df))]
    ds: Dataset = Dataset.from_pandas(df)
    # TODO: nested
    ds = ds.filter(lambda example: "<c2>" not in example["tagged_sentence"])
    word_tokenizer = MecabTokenizer(mecab_dic="ipadic")

    def get_tokens_and_tags(example: dict[str, str]) -> dict[str, str]:
        example["text"] = re.sub(r"</?(b|c|r)1>", "", example["tagged_sentence"])
        example["tokens"] = word_tokenizer.tokenize(example["text"])
        text: str = re.sub(r"</?c1>", "", example["tagged_sentence"])
        tags: list[str] = []
        tag: str = "O"
        next_tag: str = "O"
        for token in example["tokens"]:
            if text.startswith("</b1>"):
                tag = "O"
                next_tag = "O"
                text = text[5:]
            elif text.startswith("</r1>"):
                tag = "O"
                next_tag = "O"
                text = text[5:]
            if text.startswith("<b1>"):
                tag = "B-C"
                next_tag = "I-C"
                text = text[4:]
            elif text.startswith("<r1>"):
                tag = "B-E"
                next_tag = "I-E"
                text = text[4:]
            tags.append(tag)
            tag = next_tag
            text = text[len(token) :]
        example["tags"] = tags
        return example

    ds = ds.map(get_tokens_and_tags)
    return ds


def load_data_jpfin(
    dataset_enum: Enum, task_enum: Enum, data_dir: str, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    ds_train: Dataset
    ds_valid: Dataset
    ds_test: Dataset
    if task_enum == TaskType.sequence_classification:
        tsv_filename: str
        index_prefix: str
        if dataset_enum == DatasetType.jpfinresults:
            tsv_filename = "kessan_data_for_classify.tsv"
            index_prefix = "kessan_"
        elif dataset_enum == DatasetType.jpnikkei:
            tsv_filename = "nikkei_data.tsv"
            index_prefix = "nikkei_"
        else:  # pragma: no cover
            raise NotImplementedError()
        df: pd.DataFrame = pd.read_csv(
            os.path.join(data_dir, tsv_filename),
            sep="\t",
            header=0,
            names=["labels", "example_id", "text"],
        )
        df["example_id"] = df["example_id"].map(lambda x: index_prefix + x)
        ds: Dataset = Dataset.from_pandas(df)
        ds_train, ds_valid, ds_test = split_train_valid_test_dataset(ds, seed)
    elif task_enum == TaskType.span_detection:
        if dataset_enum == DatasetType.jpfinresults:
            ds_train = load_and_process_span_jpfinresults("train", data_dir)
            ds_valid = load_and_process_span_jpfinresults("dev", data_dir)
            ds_test = load_and_process_span_jpfinresults("test", data_dir)
        else:  # pragma: no cover
            raise NotImplementedError()
    else:  # pragma: no cover
        raise NotImplementedError()

    return ds_train, ds_valid, ds_test

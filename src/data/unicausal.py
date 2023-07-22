import re
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset


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
    ds = ds.rename_column("seq_label", "labels")
    return ds


def _load_data_unicausal_span_detection(data_path: str) -> Dataset:
    df: pd.DataFrame = pd.read_csv(data_path)
    df.drop_duplicates(subset=["corpus", "doc_id", "sent_id"], keep=False, inplace=True)
    df = df[df.seq_label == 1]
    ds = Dataset.from_pandas(df)
    ds = ds.map(get_bio_for_datasets)
    return ds

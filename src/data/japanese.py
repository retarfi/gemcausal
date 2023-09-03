import itertools
import os
import re
from enum import Enum

import pandas as pd
from datasets import Dataset
from transformers.models.bert_japanese import MecabTokenizer

from .. import DatasetType, NumCausalType, SpanTags, SpanTagsFormat, TaskType
from .split_dataset import split_train_valid_test_dataset


def remove_clue_and_replace_tag(example: dict[str, str]) -> dict[str, str]:
    # remove clue
    text: str = re.sub(r"</?c\d>", "", example["tagged_sentence"])
    # replace tag
    if "<b2>" in text:
        i: int = 1
        while f"<b{i}>" in text:
            text = (
                text.replace(f"<b{i}>", SpanTagsFormat.cause_begin.format(i))
                .replace(f"</b{i}>", SpanTagsFormat.cause_end.format(i))
                .replace(f"<r{i}>", SpanTagsFormat.effect_begin.format(i))
                .replace(f"</r{i}>", SpanTagsFormat.effect_end.format(i))
            )
            i += 1
    else:
        # if only 1 cause-effect pair, remove int
        text = (
            text.replace("<b1>", SpanTags.cause_begin)
            .replace("</b1>", SpanTags.cause_end)
            .replace("<r1>", SpanTags.effect_begin)
            .replace("</r1>", SpanTags.effect_end)
        )
    example["tagged_text"] = text
    return example


def _is_crossing(tpl_a: tuple[int, int], tpl_b: tuple[int, int]) -> bool:
    assert tpl_a[0] < tpl_a[1]
    assert tpl_b[0] < tpl_b[1]
    assert len(tpl_a) == 2
    assert len(tpl_b) == 2
    if tpl_a[1] < tpl_b[0] or tpl_b[1] < tpl_a[0]:
        return False
    else:
        return True


def is_nested_causal(text: str) -> bool:
    try:
        lst_idxs: list[tuple[int, int]] = []
        i: int = 1
        while SpanTagsFormat.cause_begin.format(i) in text:
            lst_idxs.append(
                (
                    text.find(SpanTagsFormat.cause_begin.format(i)),
                    text.find(SpanTagsFormat.cause_end.format(i)),
                )
            )
            lst_idxs.append(
                (
                    text.find(SpanTagsFormat.effect_begin.format(i)),
                    text.find(SpanTagsFormat.effect_end.format(i)),
                )
            )
            i += 1
        # intersection detection
        return any(map(lambda x: _is_crossing(*x), itertools.combinations(lst_idxs, 2)))
    except AssertionError as e:  # pragma: no cover
        print(e)
        raise AssertionError(f"Error with text: {text}")


def load_and_process_span_jpfinresults(fileprefix: str, data_dir: str) -> Dataset:
    df: pd.DataFrame = pd.read_json(
        os.path.join(data_dir, f"kessan/{fileprefix}.jsonl"),
        orient="records",
        lines=True,
    )
    df["example_id"] = [f"kessan_{fileprefix}_{i}" for i in range(len(df))]
    ds: Dataset = Dataset.from_pandas(df)
    ds = ds.map(remove_clue_and_replace_tag)
    ds = ds.filter(lambda example: not is_nested_causal(example["tagged_text"]))
    word_tokenizer = MecabTokenizer(mecab_dic="ipadic")

    def get_tokens_and_tags(example: dict[str, str]) -> dict[str, str]:
        example["text"] = re.sub(r"</?(c|e)\d*>", "", example["tagged_text"])
        example["tokens"] = word_tokenizer.tokenize(example["text"])
        text: str = example["tagged_text"]
        tags: list[str] = []
        tag: str = "O"
        next_tag: str = "O"
        # There are no nested causal
        for token in example["tokens"]:
            if re.match(SpanTagsFormat.cause_end.format("\\d*"), text):
                tag = "O"
                next_tag = "O"
                text = text[
                    len(
                        re.match(SpanTagsFormat.cause_end.format("\\d*"), text).group(0)
                    ) :
                ]
            elif re.match(SpanTagsFormat.effect_end.format("\\d*"), text):
                tag = "O"
                next_tag = "O"
                text = text[
                    len(
                        re.match(SpanTagsFormat.effect_end.format("\\d*"), text).group(
                            0
                        )
                    ) :
                ]
            if re.match(SpanTagsFormat.cause_begin.format("\\d*"), text):
                tag = "B-C"
                next_tag = "I-C"
                text = text[
                    len(
                        re.match(SpanTagsFormat.cause_begin.format("\\d*"), text).group(
                            0
                        )
                    ) :
                ]
            elif re.match(SpanTagsFormat.effect_begin.format("\d*"), text):
                tag = "B-E"
                next_tag = "I-E"
                text = text[
                    len(
                        re.match(SpanTagsFormat.effect_begin.format("\d*"), text).group(
                            0
                        )
                    ) :
                ]
            tags.append(tag)
            tag = next_tag
            text = text[len(token) :]
        example["tags"] = tags
        return example

    ds = ds.map(get_tokens_and_tags)
    return ds


def load_data_jpfin(
    dataset_enum: Enum, task_enum: Enum, data_dir: str, numcausal_enum: Enum, seed: int
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
        assert dataset_enum == DatasetType.jpfinresults
        ds_train = load_and_process_span_jpfinresults("train", data_dir)
        ds_valid = load_and_process_span_jpfinresults("dev", data_dir)
        ds_test = load_and_process_span_jpfinresults("test", data_dir)
        # filter
        tag_b1: str = SpanTagsFormat.cause_begin.format(1)
        if numcausal_enum == NumCausalType.single:
            ds_test = ds_test.filter(lambda x: tag_b1 not in x["tagged_text"])
        elif numcausal_enum == NumCausalType.multi:
            ds_test = ds_test.filter(lambda x: tag_b1 in x["tagged_text"])
    else:  # pragma: no cover
        raise NotImplementedError()

    return ds_train, ds_valid, ds_test

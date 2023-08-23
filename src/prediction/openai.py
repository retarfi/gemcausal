import datetime
import itertools
import json
import os
import random
from argparse import Namespace
from enum import Enum
from typing import Any, Dict, Union

import datasets
import numpy as np
import openai
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from .. import (
    DatasetType,
    NumCausalType,
    SentenceType,
    TaskType,
    assert_dataset_task_pair,
    logger,
)
from ..data.load_data import load_data
from ..setting import assert_filter_option


def api_key_validation() -> None:
    api_key: str = os.environ["OPENAI_API_KEY"]
    assert (
        api_key != ""
    ), "Environment variable OPENAI_API_KEY must be set to use OpenAI models"
    openai.api_key = api_key
    _ = openai.Model.list()


def read_template(path: str) -> dict[str, str]:
    with open(path, "r") as f:
        template: dict[str, str] = json.load(f)
    required_keys: set[str] = {
        "task_description",
        "header_example",
        "format_text",
        "format_class",
        "question",
    }
    left_keys: set[str] = required_keys - set(template.keys())
    assert len(left_keys) == 0, f"Following keys are not in template: {left_keys}"
    return template


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def compute_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str], average: str
) -> dict[str, float]:
    # prepare compute metrics because evaluate module cannot deal with labels other than
    # gold labels
    assert average in {
        "macro",
        "micro",
        "binary",
    }, f"average {average} is not implemented"
    if average == "binary":
        assert (
            len(labels) == 1
        ), "In binary classification the number of labels must be 1"
        average = None

    precision: Union[np.ndarray, float]
    recall: Union[np.ndarray, float]
    f1: Union[np.ndarray, float]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels
    )
    if not isinstance(precision, float):
        precision = precision[0]
    if not isinstance(recall, float):
        recall = recall[0]
    if not isinstance(f1, float):
        f1 = f1[0]
    result: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    arr = confusion_matrix(y_true, y_pred, normalize="true")
    for i in range(len(arr)):
        result[f"accuracy_{i}"] = arr[i, i]
    return result


def predict(args: Namespace) -> None:
    api_key_validation()
    task_type: str = args.task_type
    task_enum: Enum = TaskType[task_type]
    dataset_type: str = args.dataset_type
    dataset_enum: Enum = DatasetType[dataset_type]
    model: str = args.model
    shot: int = args.shot
    output_dir: str = args.output_dir
    seed: int = args.seed
    filter_num_sent: str = args.filter_num_sent
    filter_num_causal: str = args.filter_num_causal

    assert (
        task_enum == TaskType.span_detection or not args.evaluate_by_word
    ), "Argument evaluate_by_word only works with span_detection"
    os.makedirs(output_dir, exist_ok=True)
    template: dict[str, str] = read_template(args.template)

    assert_dataset_task_pair(dataset_enum=dataset_enum, task_enum=task_enum)
    assert_filter_option(dataset_enum=dataset_enum, args=args)
    dsd: DatasetDict = load_data(
        task_enum=task_enum,
        dataset_enum=dataset_enum,
        sentencetype_enum=SentenceType[filter_num_sent],
        numcausal_enum=NumCausalType[filter_num_causal],
        data_dir=args.data_dir,
        test_samples=args.test_samples,
        seed=seed,
    )
    random.seed(seed)
    dsd_icl: Dataset = dsd["train"].select(
        random.sample(range(len(dsd["train"])), k=shot)
    )
    annotation: str = template["header_example"]
    if task_enum == TaskType.sequence_classification:
        for i in range(shot):
            annotation += template["format_text"].format(dsd_icl[i]["text"])
            annotation += template["format_class"].format(dsd_icl[i]["labels"])

        def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
            prompt: str = template["task_description"]
            if shot > 0:
                prompt += annotation
            prompt += template["question"] + template["format_text"].format(
                example["text"]
            )
            example["prompt"] = prompt
            return example

    elif task_enum == TaskType.span_detection:
        if args.evaluate_by_word:
            for i in range(shot):
                annotation += template["format_text"].format(
                    dsd_icl[i]["text"], " / ".join(dsd_icl[i]["tokens"])
                )
                annotation += template["format_class"].format(
                    "".join(
                        map(
                            lambda item: template["format_tag"].format(*item),
                            zip(dsd_icl[i]["tokens"], dsd_icl[i]["tags"]),
                        )
                    )
                )

            def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
                prompt: str = template["task_description"]
                if shot > 0:
                    prompt += annotation
                prompt += template["question"] + template["format_text"].format(
                    example["text"], " / ".join(example["tokens"])
                )
                example["prompt"] = prompt
                return example

        else:
            for i in range(shot):
                annotation += template["format_text"].format(dsd_icl[i]["text"])
                annotation += template["format_class"].format(dsd_icl[i]["tagged_text"])

            def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
                prompt: str = template["task_description"]
                if shot > 0:
                    prompt += annotation
                prompt += template["question"] + template["format_text"].format(
                    example["text"]
                )
                example["prompt"] = prompt
                return example

    elif task_enum == TaskType.chain_classification:
        for i in range(shot):
            annotation += template["format_text"].format(
                ", ".join(dsd_icl[i]["events"]), *dsd_icl[i]["short_contexts"]
            )
            annotation += template["format_class"].format(dsd_icl[i]["labels"])

        def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
            prompt: str = template["task_description"]
            if shot > 0:
                prompt += annotation
            prompt += template["question"] + template["format_text"].format(
                ", ".join(example["events"]), *example["short_contexts"]
            )
            example["prompt"] = prompt
            return example

    else:  # pragma: no cover
        raise NotImplementedError()

    ds_test: Dataset = dsd["test"]
    ds_test = ds_test.map(format_prompt)

    # openai api map
    logger.info("Inference starts")
    lst_output: list[str] = []
    for prompt in tqdm(ds_test["prompt"]):
        completion = completion_with_backoff(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        lst_output.append(completion.choices[0].message["content"])
    logger.info("Inference ends")
    ds_test = ds_test.add_column("output", lst_output)

    ds_output: Dataset
    result: dict[str, float]
    if task_enum in (TaskType.sequence_classification, TaskType.chain_classification):
        features: datasets.Features = ds_test.features.copy()
        features["labels"] = datasets.Value("string")
        ds_output = ds_test.cast(features)

        def extract_label(example: dict[str, Any]) -> dict[str, Any]:
            example["pred"] = example["output"].replace(
                template["format_class"].split("{}")[0], ""
            )
            return example

        ds_output = ds_output.map(extract_label)
        result = compute_metrics(
            ds_output["labels"], ds_output["pred"], labels=["1"], average="binary"
        )
        logger.info("Result: %s", result)

        # output prompt result
        ds_output = ds_output.remove_columns(
            list(set(ds_test.column_names) - {"example_id", "labels", "output", "pred"})
        )
    elif task_enum == TaskType.span_detection:
        if args.evaluate_by_word:

            def extract_label(example: dict[str, Any]) -> dict[str, Any]:
                example["pred_asis"] = example["output"].replace(
                    template["format_class"].split("{}")[0], ""
                )
                lst_pred_tags: list[str] = [
                    line.replace(tag + template["format_tag"].split("{}")[1], "")
                    for tag, line in zip(
                        example["tokens"], example["pred_asis"].split("\n")
                    )
                ]
                example["pred"] = lst_pred_tags
                num_pred: int = len(example["pred"])
                num_tags: int = len(example["tags"])
                if num_pred < num_tags:
                    example["pred"] += [""] * (num_tags - num_pred)
                elif num_pred > num_tags:
                    example["pred"] = example["pred"][:num_pred]
                assert len(example["pred"]) == len(
                    example["tags"]
                ), f"Inconsistent numbers: {example}"
                return example

            ds_output = ds_test.map(extract_label)
            result: dict[str, float] = compute_metrics(
                list(itertools.chain.from_iterable(ds_output["tags"])),
                list(itertools.chain.from_iterable(ds_output["pred"])),
                labels=["B-C", "I-C", "B-E", "I-E", "O"],
                average="macro",
            )
            result["exact_match"] = sum(
                [t == p for t, p in zip(ds_output["tags"], ds_output["pred"])]
            ) / len(ds_output)
            # output prompt result
            ds_output = ds_output.remove_columns(
                list(
                    set(ds_test.column_names)
                    - {
                        "example_id",
                        "text",
                        "tagged_text",
                        "tokens",
                        "tags",
                        "output",
                        "pred",
                        "pred_asis",
                    }
                )
            )

            def extract_by_tokens(
                example: dict[str, Union[list[str], str]]
            ) -> dict[str, Union[list[str], str]]:
                lst: list[dict[str, Union[list[str], str]]] = []
                for i in range(len(example["text"])):
                    for j in range(len(example["tokens"][i])):
                        lst.append(
                            {
                                "example_id": example["example_id"][i],
                                "text": example["text"][i],
                                "tagged_text": example["tagged_text"][i],
                                "token": example["tokens"][i][j],
                                "tag": example["tags"][i][j],
                                "output": example["output"][i],
                                "pred_tag": example["pred"][i][j],
                            }
                        )
                return {k: [x[k] for x in lst] for k in lst[0].keys()}

            ds_output = ds_output.map(
                extract_by_tokens,
                batched=True,
                remove_columns=["tokens", "tags", "pred", "pred_asis"],
            )
        else:

            def extract_label(example: dict[str, Any]) -> dict[str, Any]:
                example["pred"] = example["output"].replace(
                    template["format_class"].split("{}")[0], ""
                )
                return example

            ds_output = ds_test.map(extract_label)
            result = {
                "exact_match": sum(
                    [
                        t == p
                        for t, p in zip(ds_output["tagged_text"], ds_output["pred"])
                    ]
                )
                / len(ds_output)
            }
            # output prompt result
            ds_output = ds_output.remove_columns(
                list(
                    set(ds_test.column_names)
                    - {"example_id", "text", "tagged_text", "output", "pred"}
                )
            )
        logger.info("Result: %s", result)
    else:  # pragma: no cover
        raise NotImplementedError()
    result = {
        **result,
        **{
            "task_type": task_type,
            "dataset_type": dataset_type,
            "intra-/inter-sent": filter_num_sent,
            "single-/multi-causal": filter_num_causal,
            "model": model,
            "template": args.template,
            "shot": shot,
            "seed": seed,
        },
    }
    filehead: str = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M_")
        + f"{task_type}_{dataset_type}_{filter_num_sent}_{filter_num_causal}"
    )
    if args.evaluate_by_word:
        filehead += "_ebw"
    filehead += f"_{model}"
    tpl_eval_key: tuple[str]
    if task_enum == TaskType.span_detection and not args.evaluate_by_word:
        tpl_eval_key = ("exact_match",)
    else:
        tpl_eval_key = ("f1", "accuracy", "precision", "recall")
    for key in tpl_eval_key:
        result[key] = round(result[key], 5)
    with open(os.path.join(output_dir, f"{filehead}.json"), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True, separators=(",", ": "))
    ds_output.to_csv(os.path.join(output_dir, f"{filehead}.csv"))

import datetime
import json
import os
import random
from argparse import Namespace
from typing import Any

import openai
from datasets import Dataset, DatasetDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from .. import DatasetType, TaskType, assert_dataset_task_pair, logger
from ..data.load_data import load_data


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


def predict(args: Namespace) -> None:
    api_key_validation()
    task_type: str = args.task_type
    dataset_type: str = args.dataset_type
    model: str = args.model
    shot: int = args.shot
    output_dir: str = args.output_dir
    seed: int = args.seed

    template: dict[str, str] = read_template(args.template)

    assert_dataset_task_pair(
        dataset_enum=DatasetType[dataset_type], task_enum=TaskType[task_type]
    )
    dsd: DatasetDict = load_data(
        task_enum=TaskType[task_type],
        dataset_enum=DatasetType[dataset_type],
        data_dir=args.data_dir,
        test_samples=args.test_samples,
        seed=seed,
    )
    random.seed(seed)
    dsd_icl: Dataset = dsd["train"].select(
        random.sample(range(len(dsd["train"])), k=shot)
    )
    annotation = template["header_example"]
    for i in range(shot):
        annotation += template["format_text"].format(dsd_icl[i]["text"])
        annotation += template["format_class"].format(dsd_icl[i]["labels"])

    # prepare prompt
    def format_prompt(example: dict[str, Any]) -> dict[str, Any]:
        prompt: str = template["task_description"]
        if shot > 0:
            prompt += annotation
        prompt += template["question"] + template["format_text"].format(example["text"])
        example["prompt"] = prompt
        return example

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

    # TODO: Now only implement sequence_classification,
    # TODO: so span_detection should be implemented
    # evaluate automatically
    def extract_label(example: dict[str, Any]) -> dict[str, Any]:
        example["pred"] = example["output"].replace(
            template["format_class"].split("{}")[0], ""
        )[0]
        return example

    ds_test = ds_test.map(extract_label)
    ds_correct: Dataset = ds_test.filter(
        lambda example: str(example["labels"]) == example["pred"]
    )
    # TODO: change accuracy to F1
    acc: float = len(ds_correct) / len(ds_test)
    logger.info(f"Accuracy: {acc:.3f}")

    # output prompt result
    ds_output: Dataset = ds_test.remove_columns(
        list(set(ds_test.column_names) - {"labels", "output", "pred"})
    )
    filename: str = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M_")
        + f"{task_type}_{dataset_type}_{model}.csv"
    )
    ds_output.to_csv(os.path.join(output_dir, filename))

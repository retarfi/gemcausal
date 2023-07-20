import argparse
import os

import pytest

from src.argument import add_argument_common, add_argument_openai
from src.prediction.openai import (
    api_key_validation,
    completion_with_backoff,
    predict,
    read_template,
)

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


def test_api_key_validation() -> None:
    api_key_validation()


def test_read_template() -> None:
    read_template(os.path.join(THIS_DIR, "../../template/openai_sequence.json"))


def test_completion_with_backoff() -> None:
    _ = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0,
    )


@pytest.mark.parametrize(
    "task_type, dataset_type, json_file",
    [("SEQUENCE_CLASSIFICATION", "PDTB", "openai_sequence.json")],
)
def test_predict(task_type: str, dataset_type: str, json_file: str) -> None:
    parser = argparse.ArgumentParser()
    add_argument_common(parser)
    add_argument_openai(parser)

    args = parser.parse_args(
        [
            "--task_type",
            task_type,
            "--dataset_type",
            dataset_type,
            "--data_dir",
            os.path.join(THIS_DIR, "../../data"),
            "--test_samples",
            "5",
            "--model",
            "gpt-3.5-turbo",
            "--template",
            os.path.join(THIS_DIR, "../../template", json_file),
            "--shot",
            "1",
            "--output_dir",
            os.path.join(THIS_DIR, "../materials/results"),
        ]
    )
    predict(args)

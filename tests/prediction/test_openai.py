import argparse
import math
import os

import pytest

from src.argument import add_argument_common, add_argument_openai
from src.prediction.openai import (
    api_key_validation,
    completion_with_backoff,
    compute_metrics,
    predict,
    read_template,
)

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


def test_api_key_validation() -> None:
    api_key_validation()


def test_read_template() -> None:
    read_template(os.path.join(THIS_DIR, "../../template/openai_sequence_classification.json"))


def test_completion_with_backoff() -> None:
    _ = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0,
    )


@pytest.mark.parametrize(
    "average, labels, expected",
    [
        ("macro", ["cat", "pig"], [0.333, 0.55, 0.333, 0.398]),
        ("binary", ["cat"], [0.333, 0.6, 0.5, 0.545]),
        ("binary", ["pig"], [0.333, 0.5, 0.167, 0.25]),
    ],
)
def test_compute_metrics(
    average: str, labels: list[str], expected: list[float]
) -> None:
    y_true: list[str] = ["cat"] * 6 + ["pig"] * 6
    y_pred: list[str] = (
        ["cat"] * 3 + ["pig", "other", "other"] + ["cat"] * 2 + ["pig"] + ["other"] * 3
    )
    result: dict[str, float] = compute_metrics(
        y_true, y_pred, labels=labels, average=average
    )
    for metric, expected in zip(["accuracy", "precision", "recall", "f1"], expected):
        assert math.isclose(result[metric], expected, abs_tol=1e-3)


@pytest.mark.parametrize(
    "task_type, dataset_type, json_file",
    [
        ("sequence_classification", "pdtb", "openai_sequence_classification.json"),
        ("span_detection", "pdtb", "openai_span_detection.json"),
        ("chain_classification", "reco", "openai_chain_classification.json")
    ],
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
            "--output_dir",
            os.path.join(THIS_DIR, "../materials/results"),
            "--model",
            "gpt-3.5-turbo",
            "--template",
            os.path.join(THIS_DIR, "../../template", json_file),
            "--shot",
            "1",
        ]
    )
    predict(args)

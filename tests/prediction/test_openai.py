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
    read_template(
        os.path.join(THIS_DIR, "../../template/openai_sequence_classification.json")
    )


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
    "task_type, dataset_type, num_sent, num_causal, evaluate_by_word",
    [
        ("sequence_classification", "fincausal", "all", "all", False),
        ("sequence_classification", "pdtb", "all", "all", False),
        ("sequence_classification", "pdtb", "intra", "all", False),
        ("sequence_classification", "pdtb", "inter", "all", False),
        ("chain_classification", "reco", "all", "all", False),
        ("span_detection", "fincausal", "all", "all", True),
        ("span_detection", "fincausal", "intra", "all", True),
        ("span_detection", "fincausal", "inter", "all", True),
        ("span_detection", "pdtb", "intra", "all", True),
        ("span_detection", "pdtb", "inter", "all", True),
        ("span_detection", "pdtb", "all", "single", True),
        ("span_detection", "pdtb", "all", "multi", True),
    ],
)
def test_predict(
    task_type: str, dataset_type: str, num_sent: str, num_causal: str, evaluate_by_word
) -> None:
    json_file: str
    if task_type == "sequence_classification":
        json_file = "openai_sequence_classification.json"
    elif task_type == "span_detection":
        if evaluate_by_word:
            json_file = "openai_span_detection_by_word.json"
        else:
            json_file = "openai_span_detection_by_sentence.json"
    elif task_type == "chain_classification":
        json_file = "openai_chain_classification.json"
    else:
        raise NotImplementedError()
    parser = argparse.ArgumentParser()
    add_argument_common(parser)
    add_argument_openai(parser)

    lst_args: list[str] = [
        "--task_type",
        task_type,
        "--dataset_type",
        dataset_type,
        "--data_dir",
        os.path.join(THIS_DIR, "../../data"),
        "--test_samples",
        "2",
        "--output_dir",
        os.path.join(THIS_DIR, "../materials/results"),
        "--filter_num_sent",
        num_sent,
        "--filter_num_causal",
        num_causal,
        "--model",
        "gpt-3.5-turbo",
        "--template",
        os.path.join(THIS_DIR, "../../template", json_file),
        "--shot",
        "1",
    ]
    if evaluate_by_word:
        lst_args.append("--evaluate_by_word")
    args = parser.parse_args(lst_args)
    predict(args)

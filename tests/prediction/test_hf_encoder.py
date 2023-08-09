import argparse
import os

import pytest

from src.argument import add_argument_common, add_argument_hf_encoder
from src.prediction.hf_encoder import predict

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "task_type, dataset_type",
    [
        ("sequence_classification", "because"),
        ("sequence_classification", "fincausal"),
        ("sequence_classification", "pdtb"),
        ("span_detection", "because"),
        ("span_detection", "fincausal"),
        ("span_detection", "pdtb"),
        ("chain_classification", "reco"),
    ],
)
def test_predict(task_type: str, dataset_type: str) -> None:
    parser = argparse.ArgumentParser()
    add_argument_common(parser)
    add_argument_hf_encoder(parser)

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
            "--model_name",
            "google/bert_uncased_L-2_H-128_A-2",
            "--train_batch_size",
            "32",
            "--eval_batch_size",
            "32",
            "--max_epochs",
            "1",
        ]
    )
    predict(args)


@pytest.mark.parametrize(
    "task_type, dataset_type",
    [
        ("sequence_classification", "jpfinresults"),
        ("sequence_classification", "jpnikkei"),
        ("span_detection", "jpfinresults"),
    ],
)
def test_predict_japanese(task_type: str, dataset_type: str) -> None:
    parser = argparse.ArgumentParser()
    add_argument_common(parser)
    add_argument_hf_encoder(parser)

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
            "--model_name",
            "cl-tohoku/bert-base-japanese",
            "--train_batch_size",
            "32",
            "--eval_batch_size",
            "32",
            "--max_epochs",
            "1",
        ]
    )
    predict(args)

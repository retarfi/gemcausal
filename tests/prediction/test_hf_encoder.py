import argparse
import os

import pytest

from src.argument import add_argument_common, add_argument_hf_encoder
from src.prediction.hf_encoder import predict

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "task_type, dataset_type, num_sent, num_causal",
    [
        ("sequence_classification", "because", "all", "all"),
        ("sequence_classification", "ctb", "intra", "all"),
        ("sequence_classification", "ctb", "inter", "all"),
        ("sequence_classification", "fincausal", "all", "all"),
        ("sequence_classification", "fincausal", "inter", "all"),
        ("span_detection", "fincausal", "all", "all"),
        ("span_detection", "fincausal", "intra", "all"),
        ("span_detection", "fincausal", "inter", "all"),
        ("span_detection", "pdtb", "intra", "all"),
        ("span_detection", "pdtb", "inter", "all"),
        ("span_detection", "pdtb", "all", "single"),
        ("span_detection", "pdtb", "all", "multi"),
        ("chain_classification", "reco", "all", "all"),
    ],
)
def test_predict(
    task_type: str, dataset_type: str, num_sent: str, num_causal: str
) -> None:
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
            "--filter_num_sent",
            num_sent,
            "--filter_num_causal",
            num_causal,
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
    "task_type, dataset_type, num_causal",
    [
        ("sequence_classification", "jpfinresults", "all"),
        ("sequence_classification", "jpnikkei", "all"),
        ("span_detection", "jpfinresults", "all"),
        ("span_detection", "jpfinresults", "multi"),
    ],
)
def test_predict_japanese(task_type: str, dataset_type: str, num_causal: str) -> None:
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
            "--filter_num_sent",
            "all",
            "--filter_num_causal",
            num_causal,
            "--model_name",
            "izumi-lab/bert-small-japanese",
            "--train_batch_size",
            "32",
            "--eval_batch_size",
            "32",
            "--max_epochs",
            "1",
        ]
    )
    predict(args)

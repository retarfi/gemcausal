import argparse
import os

import pytest

from src.argument import add_argument_common, add_argument_hf_encoder
from src.prediction.hf_encoder import predict

os.chdir(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize(
    "task_type, dataset_type", [("SEQUENCE_CLASSIFICATION", "PDTB")]
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
            "../../data",
            "--test_samples",
            "5",
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

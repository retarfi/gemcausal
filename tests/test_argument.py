import os
import sys
from unittest.mock import patch

import pytest

from src.argument import main

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "model_type, task_type, dataset_type",
    [
        ("openai", "sequence_classification", "pdtb"),
        ("hf-encoder", "sequence_classification", "pdtb"),
    ],
)
def test_main(model_type: str, task_type: str, dataset_type: str) -> None:
    args: str = (
        f"{model_type} --task_type {task_type} --dataset_type {dataset_type} "
        f"--data_dir {os.path.join(THIS_DIR, '../data/')} --test_samples 2 "
    )
    if model_type == "openai":
        args += (
            "--model gpt-3.5-turbo "
            f"--template {os.path.join(THIS_DIR, '../template/openai_sequence.json')} "
            "--shot 1 "
            f"--output_dir {os.path.join(THIS_DIR, 'materials/results/')}"
        )
    elif model_type == "hf-encoder":
        args += (
            "--model_name google/bert_uncased_L-2_H-128_A-2 "
            "--train_batch_size 32 --eval_batch_size 2 --max_epochs 1"
        )
    with patch.object(sys, "argv", ["test_main.py"] + args.split()):
        print(args)
        main()

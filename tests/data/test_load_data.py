import os
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Optional

import pytest
from datasets import DatasetDict


from src.data.load_data import load_data
from src import TaskType, DatasetType

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "task_name, dataset_name, test_samples, num_train, num_valid, num_test, expectation",
    [
        ("sequence_classification", "altlex", None, 462, 115, 401, does_not_raise()),
        ("span_detection", "altlex", 300, 221, 55, 100, does_not_raise()),
        ("sequence_classification", "because", None, 852, 51, 51, does_not_raise()),
        ("span_detection", "because", None, 475, 33, 33, does_not_raise()),
        ("sequence_classification", "ctb", None, 1569, 316, 316, does_not_raise()),
        ("sequence_classification", "esl", None, 1768, 232, 232, does_not_raise()),
        (
            "sequence_classification",
            "fincausal",
            None,
            17060,
            2132,
            2133,
            does_not_raise(),
        ),
        ("span_detection", "fincausal", None, 1087, 136, 136, does_not_raise()),
        ("sequence_classification", "pdtb", None, 26684, 8083, 8083, does_not_raise()),
        (
            "sequence_classification",
            "pdtb",
            1000,
            26684,
            8083,
            1000,
            does_not_raise(),
        ),  # limit dataset
        ("span_detection", "pdtb", None, 4694, 1300, 1300, does_not_raise()),
        ("chain_classification", "reco", None, 3111, 417, 672, does_not_raise()),
        (
            "sequence_classification",
            "semeval",
            None,
            6380,
            1595,
            2715,
            does_not_raise(),
        ),
        ("chain_classification", "pdtb", None, -1, -1, -1, pytest.raises(ValueError)),
    ],
)
def test_load_data(
    task_name: str,
    dataset_name: str,
    test_samples: Optional[int],
    num_train: int,
    num_valid: int,
    num_test: int,
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        dsd: DatasetDict = load_data(
            task_enum=TaskType[task_name],
            dataset_enum=DatasetType[dataset_name],
            data_dir=os.path.join(THIS_DIR, "../../data"),
            test_samples=test_samples,
        )
        assert isinstance(dsd, DatasetDict)
        assert len(dsd["train"]) == num_train
        assert len(dsd["valid"]) == num_valid
        assert len(dsd["test"]) == num_test

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
    "task_name, dataset_name, test_samples, num_total, num_test, expectation",
    [
        ("SEQUENCE_CLASSIFICATION", "AltLex", None, 978, 401, does_not_raise()),
        ("SPAN_DETECTION", "AltLex", None, 376, 100, does_not_raise()),
        ("SEQUENCE_CLASSIFICATION", "CTB", None, 2201, 316, does_not_raise()),
        ("SEQUENCE_CLASSIFICATION", "ESL", None, 2232, 232, does_not_raise()),
        ("SEQUENCE_CLASSIFICATION", "PDTB", None, 42850, 8083, does_not_raise()),
        (
            "SEQUENCE_CLASSIFICATION",
            "PDTB",
            1000,
            35767,
            1000,
            does_not_raise(),
        ),  # limit dataset
        ("SPAN_DETECTION", "PDTB", None, 7294, 1300, does_not_raise()),
        ("SEQUENCE_CLASSIFICATION", "SemEval", None, 10690, 2715, does_not_raise()),
        (
            "CHAIN_CONSTRUCTION",
            "PDTB",
            None,
            -1,
            -1,
            pytest.raises(NotImplementedError),
        ),
    ],
)
def test_load_data(
    task_name: str,
    dataset_name: str,
    test_samples: Optional[int],
    num_total: int,
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
        assert sum(dsd.num_rows.values()) == num_total
        assert len(dsd["test"]) == num_test

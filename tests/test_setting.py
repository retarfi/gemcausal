from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest

from src import DatasetType, TaskType
from src.setting import assert_dataset_task_pair


@pytest.mark.parametrize(
    "dataset, task, expectation",
    [
        ("altlex", "sequence_classification", does_not_raise()),
        ("reco", "span_detection", pytest.raises(ValueError)),
    ],
)
def test_assert_dataset_task_pair(
    dataset: str, task: str, expectation: AbstractContextManager
) -> None:
    with expectation:
        assert_dataset_task_pair(DatasetType[dataset], TaskType[task])

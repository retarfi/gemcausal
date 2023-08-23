import argparse
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest

from src import DatasetType, NumCausalType, SentenceType, TaskType
from src.setting import assert_dataset_task_pair, assert_filter_option


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


@pytest.mark.parametrize(
    "dataset, filter_num_sent, filter_num_causal, expectation",
    [
        ("pdtb", "all", "all", does_not_raise()),
        ("because", "intra", "all", does_not_raise()),
        ("because", "inter", "all", pytest.raises(ValueError)),
        ("jpfinresults", "intra", "all", pytest.raises(ValueError)),
        ("jpfinresults", "inter", "all", does_not_raise()),
        ("esl", "all", "single", does_not_raise()),
        ("esl", "all", "multi", pytest.raises(ValueError)),
    ],
)
def test_assert_filter_option(
    dataset: str,
    filter_num_sent: str,
    filter_num_causal: str,
    expectation: AbstractContextManager,
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        choices=[x.name for x in DatasetType],
        required=True,
        type=str.lower,
    )
    parser.add_argument(
        "--filter_num_sent",
        choices=[x.name for x in SentenceType],
        default=SentenceType.all.name,
        help=(
            "If specified, split examples according to whether the sequence crosses "
            "over two or more sentences"
        ),
    )
    parser.add_argument(
        "--filter_num_causal",
        choices=[x.name for x in NumCausalType],
        default=NumCausalType.all.name,
        help=(
            "If specified, split examples according to whether the sequence has "
            "multiple causal relations"
        ),
    )
    args = parser.parse_args(
        [
            "--dataset_type",
            dataset,
            "--filter_num_sent",
            filter_num_sent,
            "--filter_num_causal",
            filter_num_causal,
        ]
    )
    with expectation:
        assert_filter_option(DatasetType[dataset], args)

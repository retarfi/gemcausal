import argparse
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import pytest

from src import DatasetType, NumCausalType, PlicitType, SentenceType, TaskType
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
    "dataset, filter_num_sent, filter_num_causal, plicit_type, expectation",
    [
        ("altlex", "all", "all", "implicit", does_not_raise()),
        ("pdtb", "all", "all", "all", does_not_raise()),
        ("because", "intra", "all", "all", does_not_raise()),
        ("because", "inter", "all", "all", pytest.raises(ValueError)),
        ("jpfinresults", "intra", "all", "all", pytest.raises(ValueError)),
        ("jpfinresults", "all", "all", "implicit", pytest.raises(AssertionError)),
        ("jpfinresults", "inter", "all", "all", does_not_raise()),
        ("esl", "all", "all", "explicit", does_not_raise()),
        ("esl", "all", "single", "all", does_not_raise()),
        ("esl", "all", "multi", "all", pytest.raises(ValueError)),
    ],
)
def test_assert_filter_option(
    dataset: str,
    filter_num_sent: str,
    filter_num_causal: str,
    plicit_type: str,
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
    parser.add_argument(
        "--filter_plicit_type",
        choices=[x.name for x in PlicitType],
        default=PlicitType.all.name,
        help=(
            "If specified, filter examples according to whether the sequence has "
            "explicit or implicit causalities"
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
            "--filter_plicit_type",
            plicit_type,
        ]
    )
    with expectation:
        assert_filter_option(DatasetType[dataset], args)

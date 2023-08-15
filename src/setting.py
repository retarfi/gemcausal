from argparse import Namespace
from collections import namedtuple
from enum import Enum, EnumMeta

from . import logger

DatasetType: EnumMeta = Enum(
    "Dataset",
    (
        "altlex",
        "because",
        "ctb",
        "esl",
        "fincausal",
        "jpfinresults",
        "jpnikkei",
        "pdtb",
        "reco",
        "semeval",
    ),
)
TaskType: EnumMeta = Enum(
    "Task", ("sequence_classification", "span_detection", "chain_classification")
)
DatasetTaskPairs: tuple[tuple[Enum, tuple[Enum]]] = (
    (DatasetType.altlex, (TaskType.sequence_classification, TaskType.span_detection)),
    (DatasetType.because, (TaskType.sequence_classification, TaskType.span_detection)),
    (DatasetType.ctb, (TaskType.sequence_classification,)),
    (DatasetType.esl, (TaskType.sequence_classification,)),
    (
        DatasetType.fincausal,
        (TaskType.sequence_classification, TaskType.span_detection),
    ),
    (
        DatasetType.jpfinresults,
        (TaskType.sequence_classification, TaskType.span_detection),
    ),
    (DatasetType.jpnikkei, (TaskType.sequence_classification,)),
    (DatasetType.pdtb, (TaskType.sequence_classification, TaskType.span_detection)),
    (DatasetType.reco, (TaskType.chain_classification,)),
    (DatasetType.semeval, (TaskType.sequence_classification,)),
)


TagsForSpanDetection = namedtuple(
    "SpanTags", ["cause_begin", "cause_end", "effect_begin", "effetct_end"]
)
SpanTags = TagsForSpanDetection(
    cause_begin="<c{}>", cause_end="</c{}>", effect_begin="<e{}>", effetct_end="</e{}>"
)


def assert_dataset_task_pair(dataset_enum: Enum, task_enum: Enum) -> None:
    all_pairs: tuple[Enum, Enum] = (
        (pair[0], task) for pair in DatasetTaskPairs for task in pair[1]
    )
    if not (dataset_enum, task_enum) in all_pairs:
        raise ValueError(
            f"Invalid pairs of dataset: {dataset_enum} and task: {task_enum}"
        )


def assert_filter_option(dataset_enum: Enum, args: Namespace) -> None:
    tpl_task_inter_sent_available: tuple[str] = (
        DatasetType.ctb,
        DatasetType.esl,
        DatasetType.pdtb,
        DatasetType.fincausal,
    )
    tpl_task_multi_causal_available: tuple[str] = (
        DatasetType.altlex,
        DatasetType.because,
        DatasetType.pdtb,
        DatasetType.jpfinresults,
    )
    if (
        args.filter_num_sent is not None
        and dataset_enum not in tpl_task_inter_sent_available
    ):
        if args.filter_num_sent == "intra":
            logger.warning(
                "filter_num_sent='intra' is not available for %s, "
                "so force to change with None",
                dataset_enum.name,
            )
            args.filter_num_sent = None
        else:
            raise ValueError(
                f"filter_num_sent='inter' is not available for {dataset_enum.name}"
            )
    if (
        args.filter_num_causal is not None
        and dataset_enum not in tpl_task_multi_causal_available
    ):
        if args.filter_num_causal == "single":
            logger.warning(
                "filter_num_causal='single' is not available for %s, "
                "so force to change with None",
                dataset_enum.name,
            )
            args.filter_num_causal = None
        else:
            raise ValueError(
                f"filter_num_causal='mult' is not available for {dataset_enum.name}"
            )

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
SentenceType: EnumMeta = Enum("Sentence", ("intra", "inter", "all"))
NumCausalType: EnumMeta = Enum("NumCausal", ("single", "multi", "all"))

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
    "SpanTags", ["cause_begin", "cause_end", "effect_begin", "effect_end"]
)
SpanTagsFormat = TagsForSpanDetection(
    cause_begin="<c{}>", cause_end="</c{}>", effect_begin="<e{}>", effect_end="</e{}>"
)
SpanTags = TagsForSpanDetection(
    cause_begin=SpanTagsFormat.cause_begin.format(""),
    cause_end=SpanTagsFormat.cause_end.format(""),
    effect_begin=SpanTagsFormat.effect_begin.format(""),
    effect_end=SpanTagsFormat.effect_end.format(""),
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
    tpl_task_only_inter_sent: tuple[str] = (
        DatasetType.jpfinresults,
        DatasetType.jpnikkei,
    )
    tpl_task_multi_causal_separate_available: tuple[str] = (
        DatasetType.altlex,
        DatasetType.because,
        DatasetType.pdtb,
        DatasetType.jpfinresults,
    )
    if (
        args.filter_num_sent != SentenceType.all.name
        and dataset_enum not in tpl_task_inter_sent_available
    ):
        if dataset_enum in tpl_task_only_inter_sent:
            if SentenceType[args.filter_num_sent] == SentenceType.inter:
                logger.warning(
                    "filter_num_sent='%s' is not available for %s, "
                    "so force to change to %s",
                    SentenceType.inter.name,
                    dataset_enum.name,
                    SentenceType.all.name,
                )
                args.filter_num_sent = SentenceType.all.name
            else:
                raise ValueError(
                    f"filter_num_sent='{SentenceType.intra.name}' is not available "
                    f"for {dataset_enum.name}"
                )
        else:
            if SentenceType[args.filter_num_sent] == SentenceType.intra:
                logger.warning(
                    "filter_num_sent='%s' is not available for %s, "
                    "so force to change to %s",
                    SentenceType.intra.name,
                    dataset_enum.name,
                    SentenceType.all.name,
                )
                args.filter_num_sent = SentenceType.all.name
            else:
                raise ValueError(
                    f"filter_num_sent='{SentenceType.inter.name}' is not available "
                    f"for {dataset_enum.name}"
                )
    if (
        args.filter_num_causal != NumCausalType.all.name
        and dataset_enum not in tpl_task_multi_causal_separate_available
    ):
        if NumCausalType[args.filter_num_causal] == NumCausalType.single:
            logger.warning(
                "filter_num_causal='%s' is not available for %s, "
                "so force to change to %s",
                NumCausalType.single.name,
                dataset_enum.name,
                NumCausalType.all.name,
            )
            args.filter_num_causal = NumCausalType.all.name
        else:
            raise ValueError(
                f"filter_num_causal='{NumCausalType.multi.name}' is not available "
                f"for {dataset_enum.name}"
            )

from enum import Enum, EnumMeta

DatasetType: EnumMeta = Enum(
    "Dataset",
    (
        "altlex",
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


def assert_dataset_task_pair(dataset_enum: Enum, task_enum: Enum) -> None:
    all_pairs: tuple[Enum, Enum] = (
        (pair[0], task) for pair in DatasetTaskPairs for task in pair[1]
    )
    if not (dataset_enum, task_enum) in all_pairs:
        raise ValueError(
            f"Invalid pairs of dataset: {dataset_enum} and task: {task_enum}"
        )

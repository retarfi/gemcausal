from enum import Enum, EnumMeta

from .logging import logger

DatasetType: EnumMeta = Enum("Dataset", ("ALTLEX", "PDTB"))
TaskType: EnumMeta = Enum(
    "Task", ("SEQUENCE_CLASSIFICATION", "SPAN_DETECTION", "CHAIN_CONSTRUCTION")
)

from enum import Enum, EnumMeta

from .logging import logger

DatasetType: EnumMeta = Enum("Dataset", ("AltLex", "CTB", "ESL", "PDTB", "SemEval"))
TaskType: EnumMeta = Enum(
    "Task", ("SEQUENCE_CLASSIFICATION", "SPAN_DETECTION", "CHAIN_CONSTRUCTION")
)

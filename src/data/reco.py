import json
from typing import Any, Union

from datasets import Dataset, DatasetDict
from transformers import PretrainedConfig, PreTrainedTokenizer

from .. import logger


def load_reco_dataset(json_path: str) -> Dataset:
    with open(json_path, "r") as f:
        raw_data: dict[str, list[dict[str, Union[list[str], int, str]]]] = json.load(f)
    instance: dict[str, Union[list[str], int, str]]
    data: list[dict[str, Union[list[str], int]]] = []
    for instance in raw_data["instances"]:
        instance["events"] = list(
            map(lambda x: x.replace("_", " "), instance["events"])
        )
        for i in range(len(instance["short_contexts"]) - 1):
            assert instance["label"] in {0, 2, 3, 4}
            # 0: reliable
            # 2: not reliable btw sentence idx 0 (1st sentence) and 1 (2nd sentence)
            # ... 4: not reliable btw sentence idx 2 and 3
            label: int
            if instance["label"] == i + 2:
                label = 0  # not reliable
            else:
                label = 1
            data.append(
                {
                    "events": instance["events"][i : i + 3],
                    "short_contexts": instance["short_contexts"][i : i + 2],
                    "labels": label,
                }
            )
    return Dataset.from_list(data)


def preprocess_reco_for_chain_classification(
    dsd: DatasetDict, tokenizer: PreTrainedTokenizer, config: PretrainedConfig
) -> DatasetDict:
    def tokenize_and_concat_with_sep_token_for_reco(
        example: dict[str, Any]
    ) -> dict[str, Any]:
        # input_ids (temporary, no cls/sep token)
        token_ids: list[list[int]] = [
            tokenizer.encode(x) for x in example["events"] + example["short_contexts"]
        ]
        input_ids: list[int] = token_ids[0]
        tokens: list[int]
        for tokens in token_ids[1:]:
            input_ids.extend([tokenizer.sep_token_id, *tokens])
        example["input_ids"] = input_ids
        return example

    dsd = dsd.map(tokenize_and_concat_with_sep_token_for_reco)
    max_length: int = max(
        map(len, dsd["train"]["input_ids"])
    ) + tokenizer.num_special_tokens_to_add(pair=False)
    if max_length > config.max_position_embeddings:
        logger.warning(
            (
                "Although length of train datasets is %s, "
                "that of the model is %s so we set max length as %s"
            ),
            max_length,
            config.max_position_embeddings,
            config.max_position_embeddings,
        )
        max_length = config.max_position_embeddings
    max_num_tokens: int = max_length - tokenizer.num_special_tokens_to_add(pair=False)

    # truncation and padding
    def _adjust_length(example: dict[str, Any]) -> dict[str, Any]:
        input_ids: list[int] = example["input_ids"][:max_num_tokens]
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        pad_length: int = max_length - len(input_ids)
        example["attention_mask"] = [1] * len(input_ids) + [0] * pad_length
        example["input_ids"] = input_ids + [tokenizer.pad_token_id] * pad_length
        return example

    dsd = dsd.map(_adjust_length)
    return dsd

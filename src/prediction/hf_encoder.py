import datetime
import itertools
import json
import os
from argparse import Namespace
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch
import transformers
from datasets import DatasetDict
from evaluate import CombinedEvaluations
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from .metrics import compute_exact_match, load_metrics
from .. import (
    DatasetType,
    NumCausalType,
    PlicitType,
    SentenceType,
    TaskType,
    assert_dataset_task_pair,
    logger,
)
from ..data.load_data import load_data
from ..data.reco import preprocess_reco_for_chain_classification
from ..setting import assert_filter_option


def preprocess_for_sequence_classification(
    dsd: DatasetDict, tokenizer: PreTrainedTokenizer, model_name: str
) -> tuple[DatasetDict, PretrainedConfig]:
    max_length: int = max(map(lambda x: len(tokenizer.encode(x)), dsd["train"]["text"]))
    config: PretrainedConfig = AutoConfig.from_pretrained(model_name)
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
    dsd = dsd.map(
        lambda example: tokenizer(
            example["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        ),
        remove_columns="example_id",
    )
    return dsd, config


def preprocess_for_span_detection(
    dsd: DatasetDict, tokenizer: PreTrainedTokenizer, model_name: str
) -> tuple[DatasetDict, PretrainedConfig]:
    label_list: list[str] = list(
        set(itertools.chain.from_iterable(dsd["train"]["tags"]))
    )
    label_to_id: dict[str, int] = {l: i for i, l in enumerate(label_list)}
    b_to_i_label: list[str] = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_name, num_labels=len(label_list)
    )

    max_length: int = max(
        map(
            lambda x: len(tokenizer.encode(x, is_split_into_words=True)),
            dsd["train"]["tokens"],
        )
    )
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

    def tokenize_and_align_tags(examples: dict[str, Any]) -> dict[str, Any]:
        tokenized_inputs: dict[str, Any] = tokenizer(
            examples["tokens"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words
            # (with a label for each word).
            is_split_into_words=True,
        )
        tags: list[str] = []
        for i, label in enumerate(examples["tags"]):
            word_ids: list[Optional[int]]
            if tokenizer.is_fast:
                word_ids = tokenized_inputs.word_ids(batch_index=i)
            else:
                word_ids = []
                word_idx: int = 0
                subwords: list[int] = tokenizer.encode(
                    examples["tokens"][i][word_idx], add_special_tokens=False
                )
                cand_subword: int = subwords.pop(0)
                for token_idx in tokenized_inputs["input_ids"][i]:
                    if token_idx in (
                        tokenizer.pad_token_id,
                        tokenizer.cls_token_id,
                        tokenizer.sep_token_id,
                    ):
                        word_ids.append(None)
                    elif token_idx == cand_subword:
                        word_ids.append(word_idx)
                        if word_idx + 1 < len(examples["tokens"][i]):
                            if len(subwords) == 0:
                                word_idx += 1
                                subwords = tokenizer.encode(
                                    examples["tokens"][i][word_idx],
                                    add_special_tokens=False,
                                )
                            cand_subword = subwords.pop(0)
                    else:  # pragma: no cover
                        raise ValueError(examples["tokens"][i])
            previous_word_idx: Optional[int] = None
            label_ids: list[int] = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                previous_word_idx = word_idx
            tags.append(label_ids)
        tokenized_inputs["labels"] = tags
        return tokenized_inputs

    dsd = dsd.map(tokenize_and_align_tags, batched=True)
    return dsd, config


def preprocess_for_chain_classification(
    dsd: DatasetDict, tokenizer: PreTrainedTokenizer, model_name: str, dataset_type: str
) -> tuple[DatasetDict, PretrainedConfig]:
    config: PretrainedConfig = AutoConfig.from_pretrained(model_name)
    if DatasetType[dataset_type] == DatasetType.reco:
        dsd = preprocess_reco_for_chain_classification(dsd, tokenizer, config)
    return dsd, config


def predict(args: Namespace) -> None:
    task_type: str = args.task_type
    task_enum: Enum = TaskType[task_type]
    dataset_type: str = args.dataset_type
    dataset_enum: Enum = DatasetType[dataset_type]
    seed: int = args.seed
    model_name: str = args.model_name
    tokenizer_name: str = (
        args.tokenizer_name if args.tokenizer_name is not None else model_name
    )
    lst_lr: list[float] = args.lr
    train_batch_size: int = args.train_batch_size
    eval_batch_size: int = args.eval_batch_size
    max_epochs: int = args.max_epochs
    filter_num_sent: str = args.filter_num_sent
    filter_num_causal: str = args.filter_num_causal
    filter_plicit_type: str = args.filter_plicit_type

    assert_dataset_task_pair(dataset_enum=dataset_enum, task_enum=task_enum)
    assert_filter_option(dataset_enum=dataset_enum, args=args)
    dsd: DatasetDict = load_data(
        task_enum=task_enum,
        dataset_enum=dataset_enum,
        sentencetype_enum=SentenceType[filter_num_sent],
        numcausal_enum=NumCausalType[filter_num_causal],
        plicit_enum=PlicitType[filter_plicit_type],
        data_dir=args.data_dir,
        test_samples=args.test_samples,
        seed=seed,
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info("Tokenize datasets")
    config: PretrainedConfig
    if task_enum == TaskType.sequence_classification:
        dsd, config = preprocess_for_sequence_classification(dsd, tokenizer, model_name)
    elif task_enum == TaskType.span_detection:
        dsd, config = preprocess_for_span_detection(dsd, tokenizer, model_name)
    elif task_enum == TaskType.chain_classification:
        dsd, config = preprocess_for_chain_classification(
            dsd, tokenizer, model_name, dataset_type=dataset_type
        )
    else:  # pragma: no cover
        raise NotImplementedError()

    lst_grid_results: list[dict[str, float]] = []
    lst_metrics: list[str] = ["f1", "precision", "recall", "accuracy"]
    metrics: CombinedEvaluations = load_metrics(lst_metrics)
    if task_enum == TaskType.span_detection:
        lst_metrics.append("exact_match")
    for lr in lst_lr:
        training_args: TrainingArguments = TrainingArguments(
            output_dir="./materials/",
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=train_batch_size // torch.cuda.device_count(),
            per_device_eval_batch_size=eval_batch_size // torch.cuda.device_count(),
            learning_rate=lr,
            num_train_epochs=max_epochs,
            warmup_ratio=0.1,
            logging_strategy="no",
            save_strategy="no",
            seed=seed,
            report_to="none",
            disable_tqdm=True,
            data_seed=seed,
        )
        model: PreTrainedModel
        metric_average: str
        if task_enum in (
            TaskType.sequence_classification,
            TaskType.chain_classification,
        ):
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            metric_average = "binary"
        elif task_enum == TaskType.span_detection:
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                model_name, config=config
            )
            metric_average = "macro"
        else:  # pragma: no cover
            raise NotImplementedError()

        def compute_metrics(p: transformers.EvalPrediction) -> dict[str, float]:
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.argmax(preds, axis=-1)
            if task_enum == TaskType.span_detection:
                # exact match
                result_em: dict[str, float] = compute_exact_match(preds, p.label_ids)
                preds = preds[p.label_ids != -100].ravel()
                label_ids = p.label_ids[p.label_ids != -100].ravel()
            else:
                label_ids = p.label_ids
            result = metrics.compute(
                predictions=preds, references=label_ids, average=metric_average
            )
            if task_enum == TaskType.span_detection:
                result.update(result_em)
            return result

        trainer: Trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dsd["train"],
            eval_dataset={"valid": dsd["valid"], "test": dsd["test"]},
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=transformers.default_data_collator,
        )
        trainer.train()
        lst_epochs_result: list[dict[str, float]] = list(
            filter(
                lambda d: len(
                    {
                        f"eval_{mode}_{met}"
                        for met in lst_metrics
                        for mode in ("valid", "test")
                    }
                    & set(d.keys())
                )
                > 0,
                trainer.state.log_history,
            )
        )
        lst_tmp: list[dict[str, float]] = []
        for i in range(1, max_epochs + 1):
            d_tmp = {}
            for d in filter(lambda item: item["epoch"] == i, lst_epochs_result):
                d_tmp.update(d)
            lst_tmp.append(d_tmp)
        dct_result: dict[str, float] = max(
            enumerate(lst_tmp), key=lambda x: x[1][f"eval_valid_{lst_metrics[0]}"]
        )[1]
        lst_grid_results.append(dct_result)

    best_result: dict[str, float] = sorted(
        lst_grid_results, key=lambda x: x[f"eval_valid_{lst_metrics[0]}"], reverse=True
    )[0]
    best_result = dict(
        [
            (k.replace("eval_valid_", "valid_").replace("eval_test_", "test_"), v)
            for k, v in best_result.items()
            if any(map(lambda x: f"_{x}" in k, lst_metrics)) or k == "epoch"
        ]
    )
    logger.info("Best result: %s", best_result)

    filehead: str = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M_")
        + f"{task_type}_{dataset_type}_{filter_num_sent}_{filter_num_causal}_hf-encoder"
    )
    result: list[str, Union[list[str], str]] = {
        **best_result,
        **{
            "task_type": task_type,
            "dataset_type": dataset_type,
            "intra-/inter-sent": filter_num_sent,
            "single-/multi-causal": filter_num_causal,
            "ex-/im-plicit": filter_plicit_type,
            "model": model_name,
            "tokenizer": tokenizer_name,
            "lr": lst_lr,
            "train_batch_size": train_batch_size,
            "max_epochs": max_epochs,
            "seed": seed,
            "test_samples": args.test_samples,
        },
    }
    for key in lst_metrics:
        for part in ("valid", "test"):
            result[f"{part}_{key}"] = round(result[f"{part}_{key}"], 5)
    with open(os.path.join(args.output_dir, f"{filehead}.json"), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True, separators=(",", ": "))

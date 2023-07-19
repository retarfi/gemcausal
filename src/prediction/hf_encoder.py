from argparse import Namespace

import evaluate
import numpy as np
import torch
import transformers
from datasets import DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from .. import DatasetType, TaskType, logger
from ..data.load_data import load_data


def predict(args: Namespace) -> None:
    task_type: str = args.task_type
    dataset_type: str = args.dataset_type
    seed: int = args.seed
    model_name: str = args.model_name
    tokenizer_name: str = (
        args.tokenizer_name if args.tokenizer_name is not None else model_name
    )
    lst_lr: list[float] = args.lr
    train_batch_size: int = args.train_batch_size
    eval_batch_size: int = args.eval_batch_size
    max_epochs: int = args.max_epochs

    metrics_name: str = "f1"

    dsd: DatasetDict = load_data(
        task_enum=TaskType[task_type],
        dataset_enum=DatasetType[dataset_type],
        data_dir=args.data_dir,
        test_samples=args.test_samples,
        seed=seed,
    )
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info("Tokenize datasets")

    max_length: int = max(map(lambda x: len(tokenizer.encode(x)), dsd["train"]["text"]))
    config: PretrainedConfig = AutoConfig.from_pretrained(model_name)
    if max_length > config.max_position_embeddings:
        logger.warning(
            (
                f"Although length of train datasets is {max_length}, "
                f"that of the model is {config.max_position_embeddings} "
                f"so we set max length as {config.max_position_embeddings}"
            )
        )
        max_length = config.max_position_embeddings
    dsd = dsd.map(
        lambda example: tokenizer(
            example["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
    )

    lst_grid_results: list[dict[str, float]] = []
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
        if TaskType[task_type] == TaskType.SEQUENCE_CLASSIFICATION:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
        elif TaskType[task_type] == TaskType.SPAN_DETECTION:
            # TODO: Number of labels
            model = transformers.AutoModelForTokenClassification.from_pretrained(
                model_name
            )
        else:
            raise NotImplementedError()

        metric = evaluate.load(metrics_name)

        def compute_metrics(p: transformers.EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
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
                    {f"eval_valid_{metrics_name}", f"eval_test_{metrics_name}"}
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
            enumerate(lst_tmp), key=lambda x: x[1][f"eval_valid_{metrics_name}"]
        )[1]
        lst_grid_results.append(dct_result)

    best_result: dict[str, float] = sorted(
        lst_grid_results, key=lambda x: x[f"eval_valid_{metrics_name}"], reverse=True
    )[0]
    best_result = dict(
        [
            (k.replace("eval_valid_", "valid_").replace("eval_test_", "test_"), v)
            for k, v in best_result.items()
            if f"_{metrics_name}" in k or k == "epoch"
        ]
    )
    logger.info(f"Best result: {best_result}")

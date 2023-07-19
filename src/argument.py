from argparse import ArgumentParser

from . import DatasetType, TaskType


def add_argument_common(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--task_type", choices=[x.name for x in TaskType], required=True
    )
    parser.add_argument(
        "--dataset_type", choices=[x.name for x in DatasetType], required=True
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--test_samples",
        type=int,
        help="Limit test samples from the head. If not specified, use all samples",
    )
    parser.add_argument("--seed", type=int, default=42)


def add_argument_hf_encoder(parser: ArgumentParser) -> None:
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--tokenizer_name",
        help="If not specified, use the tokenizer as same as the model",
    )
    parser.add_argument("--lr", nargs="+", type=float, default=[1e-5])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=10)


def add_argument_openai(parser: ArgumentParser) -> None:
    parser.add_argument("--model", choices=["gpt-3.5-turbo", "gpt-4"], required=True)
    parser.add_argument(
        "--template", help="Json path of prompt template", required=True
    )
    parser.add_argument(
        "--shot", help="Number of shots", choices=[1, 2, 3], required=True, type=int
    )
    parser.add_argument("--output_dir", required=True)

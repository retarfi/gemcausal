from argparse import ArgumentParser

from . import DatasetType, TaskType
from .prediction import predict_hf_encoder, predict_openai


def add_argument_common(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--task_type", choices=[x.name for x in TaskType], required=True, type=str.lower
    )
    parser.add_argument(
        "--dataset_type",
        choices=[x.name for x in DatasetType],
        required=True,
        type=str.lower,
    )
    parser.add_argument("--data_dir", required=True, default="data/")
    parser.add_argument(
        "--test_samples",
        type=int,
        help="Limit test samples from the head. If not specified, use all samples",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for json and csv (OpenAI model) ",
    )
    parser.add_argument(
        "--filter_num_sent",
        choices=["intra", "inter"],
        default=None,
        help="If specified, split examples according to whether the sequence crosses over two or more sentences",
    )


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


def main() -> None:
    parent_parser = ArgumentParser()
    subparsers = parent_parser.add_subparsers()

    parser_hf_encoder = subparsers.add_parser(
        "hf-encoder", help="see `hf-encoder --help`"
    )
    add_argument_common(parser_hf_encoder)
    add_argument_hf_encoder(parser_hf_encoder)
    parser_hf_encoder.set_defaults(func=predict_hf_encoder)

    parser_openai = subparsers.add_parser("openai", help="see `openai --help`")
    add_argument_common(parser_openai)
    add_argument_openai(parser_openai)
    parser_openai.set_defaults(func=predict_openai)

    args = parent_parser.parse_args()
    args.func(args)

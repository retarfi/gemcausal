from argparse import ArgumentParser

from . import DatasetType, NumCausalType, PlicitType, SentenceType, TaskType
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
    parser.add_argument(
        "--test_dataset_type",
        choices=[x.name for x in DatasetType],
        type=str.lower,
        help="If not specified, use the same dataset type as the training dataset",
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
        choices=[x.name for x in SentenceType],
        default=SentenceType.all.name,
        help=(
            "If specified, split examples according to whether the sequence crosses "
            "over two or more sentences"
        ),
    )
    parser.add_argument(
        "--filter_num_causal",
        choices=[x.name for x in NumCausalType],
        default=NumCausalType.all.name,
        help=(
            "If specified, split examples according to whether the sequence has "
            "multiple causal relations"
        ),
    )
    parser.add_argument(
        "--filter_plicit_type",
        choices=[x.name for x in PlicitType],
        default=PlicitType.all.name,
        help=(
            "If specified, filter examples according to whether the sequence has "
            "explicit or implicit causalities"
        ),
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
    parser.add_argument(
        "--model",
        choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
        required=True,
    )
    parser.add_argument(
        "--template", help="Json path of prompt template", required=True
    )
    parser.add_argument(
        "--shot",
        help="Number of shots",
        choices=[0, 1, 3, 5, 10, 30, 50],
        required=True,
        type=int,
    )
    parser.add_argument(
        "--evaluate_by_word",
        action="store_true",
        help=(
            "Evaluate by words, not by sentences with exact match "
            "(only for span detection)"
        ),
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

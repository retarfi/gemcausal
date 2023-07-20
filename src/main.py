import argparse

from .argument import add_argument_common, add_argument_hf_encoder, add_argument_openai
from .prediction import predict_hf_encoder, predict_openai


def main() -> None:
    parent_parser = argparse.ArgumentParser(add_help=False)
    subparsers = parent_parser.add_subparsers()

    parser_hf_encoder = subparsers.add_parser(
        "hf-encoder", parents=[parent_parser], help="see `add -h`"
    )
    add_argument_common(parser_hf_encoder)
    add_argument_hf_encoder(parser_hf_encoder)
    parser_hf_encoder.set_defaults(handler=predict_hf_encoder)

    parser_openai = subparsers.add_parser(
        "openai", parents=[parent_parser], help="see `add -h`"
    )
    add_argument_common(parser_openai)
    add_argument_openai(parser_openai)
    parser_openai.set_defaults(handler=predict_openai)

    args = parent_parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parent_parser.print_help()


if __name__ == "__main__":
    main()

import pytest

from src.data.preprocess.unicausal import remove_tag, get_bio, get_bio_for_datasets


@pytest.mark.parametrize(
    "token, cleaned",
    [
        ("<ARG0>", ""),
        ("</ARG1>", ""),
        ("<ARG0>Start", "Start"),
        ("end</ARG0> .", "end ."),
    ],
)
def test_remove_tag(token: str, cleaned: str) -> None:
    assert remove_tag(token) == cleaned


@pytest.mark.parametrize(
    "text, tokens, tags",
    [
        (
            "Hello, <ARG0>this pencil</ARG0> makes me <ARG1>very mad</ARG1>.",
            ["Hello,", "this", "pencil", "makes", "me", "very", "mad."],
            ["O", "B-C", "I-C", "O", "O", "B-E", "I-E"],
        ),
        (
            "<ARG1>This trouble</ARG0> was caused by <ARG0>your carelessness</ARG0>.",
            ["This", "trouble", "was", "caused", "by", "your", "carelessness."],
            ["B-E", "I-C", "O", "O", "O", "B-C", "I-C"],
        ),
    ],
)
def test_get_bio(text: str, tokens: list[str], tags: list[str]) -> None:
    processed_tokens, processed_tags = get_bio(text)
    assert tokens == processed_tokens
    assert tags == processed_tags


def test_get_bio_for_datasets() -> None:
    text: str = "<ARG0>This pencil</ARG0> makes me <ARG1>very mad</ARG1>."
    dct_input: dict[str, str] = {"text_w_pairs": text, "text": "test"}
    dct_expected: dict[str, str] = {
        **dct_input,
        **{
            "tokens": ["This", "pencil", "makes", "me", "very", "mad."],
            "tags": ["B-C", "I-C", "O", "O", "B-E", "I-E"],
        },
    }
    assert get_bio_for_datasets(dct_input) == dct_expected

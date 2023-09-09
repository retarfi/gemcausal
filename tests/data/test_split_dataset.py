import pytest

from src.data.split_dataset import is_explicit, wrapper_is_explicit


@pytest.mark.parametrize(
    "text, cause, effect, gold",
    [
        (
            "Once you work, you cannot exit soon",
            "you work",
            "you cannot exit soon",
            True,
        ),
        ("This brought on good ending", "This", "good ending", True),
        ("This has brought on good ending", "This", "good ending", True),
        ("This had brought on good ending", "This", "good ending", True),
        (
            "An effect is that I cannot drink coffee",
            "",
            "that I cannot drink coffee",
            True,
        ),
        (
            "The effect is that I cannot drink coffee",
            "",
            "that I cannot drink coffee",
            True,
        ),
        (
            "The effect has been that I cannot drink coffee",
            "",
            "that I cannot drink coffee",
            True,
        ),
        ("This is a pen", "This", "a pen", False),
    ],
)
def test_is_explicit(text: str, cause: str, effect: str, gold: bool) -> None:
    assert is_explicit(text, cause=cause, effect=effect) == gold


@pytest.mark.parametrize(
    "example, gold",
    [
        (
            {
                "text": "Once you work, you cannot exit soon",
                "tagged_text": "Once <c>you work</c>, <e>you cannot exit soon</e>",
            },
            True,
        ),
        (
            {"text": "This is a pen", "tagged_text": "<c>This</c> is <e>a pen</e>"},
            False,
        ),
    ],
)
def test_wrapper_is_explicit(example: dict[str, str], gold: bool) -> None:
    assert wrapper_is_explicit(example) == gold

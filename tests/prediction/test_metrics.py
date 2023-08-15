from typing import Union

import numpy as np
import pytest

from src.prediction.metrics import compute_exact_match


@pytest.mark.parametrize(
    "pred, ref, em",
    [
        (np.array([[0, 1], [1, 1]]), np.array([[0, 0], [0, 1]]), 0.0),
        ([[0, 1], [1, 1]], [[0, 1], [0, 1]], 0.5),
        ([[0, 4, 6], [2, 5, 3]], [[0, 1, -100], [2, -100, 3]], 0.5),
    ],
)
def test_compute_exact_match(
    pred: Union[np.ndarray, list[list[int]]],
    ref: Union[np.ndarray, list[list[int]]],
    em: float,
) -> None:
    assert compute_exact_match(pred, ref)["exact_match"] == em

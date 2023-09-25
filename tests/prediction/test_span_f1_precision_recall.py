import os
import math

import evaluate
import pytest


THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "references, predictions, expected",
    [
        (
            [1, 0, 2, 2, 0, 0, 1, 0, 0],
            [1, 0, 0, 2, 0, 1, 1, 0, 0],
            {"f1": 0.733, "precision": 0.83, "recall": 0.75},
        )
    ],
)
def test_span_class(
    references: list[int], predictions: list[int], expected: dict[str, float]
) -> None:
    metric = evaluate.load(
        os.path.join(THIS_DIR, "../../src/prediction/span_f1_precision_recall.py")
    )
    result: dict[str, float] = metric.compute(
        references=references, predictions=predictions
    )
    for k in result.keys():
        assert math.isclose(result[k], expected[k], rel_tol=0.01)

from collections import namedtuple

import datasets
import evaluate


_DESCRIPTION = ""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    score_labels (`tuple` of `int`): Labels for score calculation. Defaults to [1, 2].

Returns:
    f1 (`float`)
    precision (`float`)
    recall (`float`)
"""

_CITATION = ""


class Span(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
        )

    def _compute(self, predictions, references, score_labels: tuple[int] = (1, 2)):
        score = namedtuple("score", ("f1", "recall", "precision"))
        if not isinstance(predictions, list):
            predictions = predictions.tolist()
        if not isinstance(references, list):
            references = references.tolist()
        lst_scores: list = []
        for label in score_labels:
            pred: list[int] = [i for i, x in enumerate(predictions) if x == label]
            ref: list[int] = [i for i, x in enumerate(references) if x == label]
            prec: float = len(set(pred) & set(ref)) / len(pred) if len(pred) != 0 else 0.0
            rec: float = len(set(pred) & set(ref)) / len(ref) if len(ref) != 0 else 0.0
            f1: float = 2 * prec * rec / (prec + rec) if (prec + rec) != 0.0 else 0.0
            lst_scores.append(score(f1=f1, recall=rec, precision=prec))
        return {
            "f1": sum(map(lambda x: x.f1, lst_scores)) / len(lst_scores),
            "recall": sum(map(lambda x: x.recall, lst_scores)) / len(lst_scores),
            "precision": sum(map(lambda x: x.precision, lst_scores)) / len(lst_scores),
        }

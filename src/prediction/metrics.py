from typing import Dict, List, Union

import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix


def compute(self, predictions=None, references=None, **kwargs):
    """Compute each evaluation module.

    Usage of positional arguments is not allowed to prevent mistakes.

    Args:
        predictions (list/array/tensor, optional): Predictions.
        references (list/array/tensor, optional): References.
        **kwargs (optional): Keyword arguments that will be forwarded to the evaluation module :meth:`_compute`
            method (see details in the docstring).

    Return:
        dict or None

        - Dictionary with the results if this evaluation module is run on the main process (``process_id == 0``).
        - None if the evaluation module is not run on the main process (``process_id != 0``).
    """
    results: List[Dict[str, float]] = []

    for module_names, evaluation_module in zip(
        self.evaluation_module_names, self.evaluation_modules
    ):
        batch = {"predictions": predictions, "references": references, **kwargs}
        lst = evaluation_module._feature_names()
        if module_names in ("f1", "precision", "recall"):
            lst += ["average"]
        batch = {input_name: batch[input_name] for input_name in lst}
        result: Dict[str, float] = evaluation_module.compute(**batch)
        if module_names == "accuracy":
            arr = confusion_matrix(
                y_true=references, y_pred=predictions, normalize="true"
            )
            for i in range(len(arr)):
                result[f"accuracy_{i}"] = arr[i, i]
        results.append(result)

    return self._merge_results(results)


evaluate.CombinedEvaluations.compute = compute


def load_metrics(lst_metrics: list[str]) -> evaluate.CombinedEvaluations:
    # for hf model
    metrics = evaluate.combine(lst_metrics)
    return metrics


def compute_exact_match(
    predictions: Union[np.ndarray, list],
    references: Union[np.ndarray, list],
    ignore_index: int = -100,
) -> dict[str, float]:
    assert len(predictions) == len(references), "Number of examples is not same"
    lst_match: list[int] = []
    for p, r in zip(predictions, references):
        assert len(p) == len(r)
        arr_bool: np.ndarray = np.array(r) != ignore_index
        lst_match.append(
            int(np.array_equal(np.array(p)[arr_bool], np.array(r)[arr_bool]))
        )
    return {"exact_match": np.mean(lst_match)}

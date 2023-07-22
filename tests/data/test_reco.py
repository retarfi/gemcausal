import os

import pytest

from src.data.reco import load_reco_dataset

THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("json_file", ("train.json", "test.json"))
def test_load_reco_datasets(json_file: str) -> None:
    _ = load_reco_dataset(os.path.join(THIS_DIR, "../../data/reco/", json_file))

from datasets import Dataset, DatasetDict


def split_train_valid_test_dataset(
    ds: Dataset, seed: int
) -> tuple[Dataset, Dataset, Dataset]:
    # train: 80%, valid: 10%, test: 10%
    dsd_train_valtest: DatasetDict = ds.train_test_split(
        test_size=0.2, shuffle=True, seed=seed
    )
    ds_train = dsd_train_valtest["train"]
    dsd_val_test: DatasetDict = dsd_train_valtest["test"].train_test_split(
        test_size=0.5, shuffle=True, seed=seed
    )
    ds_valid = dsd_val_test["train"]
    ds_test = dsd_val_test["test"]
    return ds_train, ds_valid, ds_test

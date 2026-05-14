from mrvp.data.split import assert_no_leakage, split_root_ids


def test_split_root_ids_no_leakage():
    splits = split_root_ids(["a", "b", "c", "d", "e"], {"train": 0.6, "test": 0.4}, seed=1)
    assert sum(len(v) for v in splits.values()) == 5
    assert_no_leakage(splits)

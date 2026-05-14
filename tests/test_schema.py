from pathlib import Path

from mrvp.sim.dataset_builder import build_synthetic_dataset, merge_datasets
from mrvp.data.dataset import MRVPDataset


def test_schema_build_and_load(tmp_path: Path):
    cfg = {"world": {"bev_size": 16, "steps_O": 3}, "history_steps": 2, "actors": 2, "actor_features": 3, "recovery_horizon": 4}
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    build_synthetic_dataset(raw, 4, ["SC", "LF"], 3, "testsim", cfg)
    merge_datasets([str(raw)], out, "train:0.5,val:0.25,test:0.25", seed=1)
    ds = MRVPDataset(out, split="train")
    row = ds[0]
    assert row["world_reset"]["A"].shape[-1] == 16
    assert len(row["prefix_controls"].shape) == 2

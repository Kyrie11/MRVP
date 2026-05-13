from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mrvp.action_selection import lower_tail_cvar, select_tail_consistent_action
from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.schema import SchemaDims, row_to_numpy, validate_row_no_leakage
from mrvp.data.synthetic import make_synthetic_rows, write_jsonl
from mrvp.models.cmrt import CounterfactualMotionResetTokenizer
from mrvp.models.rpfn import RecoveryProgramFunnelNetwork


def _batch(tmp_path: Path, n_roots: int = 12, batch_size: int = 8, slot_dim: int = 32):
    path = tmp_path / "smoke.jsonl"
    write_jsonl(make_synthetic_rows(n_roots=n_roots, seed=0), path)
    ds = MRVPDataset(path, split="train", dims=SchemaDims(reset_slot_dim=slot_dim))
    batch = next(iter(DataLoader(ds, batch_size=batch_size, collate_fn=mrvp_collate)))
    return path, ds, batch


def test_cmrt_forward_shapes(tmp_path: Path):
    torch.set_num_threads(1)
    _, _, batch = _batch(tmp_path, slot_dim=32)
    cmrt = CounterfactualMotionResetTokenizer(hidden_dim=32, mixture_count=2, reset_slot_dim=32)
    out = cmrt(batch)
    assert out["reset_slots"].shape == (8, 16, 32)
    assert out["reset_state_mix_mean"].shape == (8, 2, 12)
    assert out["recovery_world_vec"].shape[0] == 8
    assert out["degradation"].shape == (8, 6)
    assert torch.isfinite(cmrt.loss(batch)["loss"])


def test_cmrt_sampling_shapes(tmp_path: Path):
    torch.set_num_threads(1)
    _, _, batch = _batch(tmp_path, batch_size=4, slot_dim=32)
    cmrt = CounterfactualMotionResetTokenizer(hidden_dim=32, mixture_count=2, reset_slot_dim=32)
    samples = cmrt.sample_reset_problems(batch, num_samples=8)
    assert samples["reset_state"].shape == (32, 12)
    assert samples["reset_slots"].shape == (32, 16, 32)
    assert samples["x_plus"].shape == samples["reset_state"].shape


def test_rpfn_rollout_shapes(tmp_path: Path):
    torch.set_num_threads(1)
    _, _, batch = _batch(tmp_path, batch_size=4, slot_dim=32)
    batch["reset_slots"] = batch["reset_slots"][:, :, :32]
    rpfn = RecoveryProgramFunnelNetwork(hidden_dim=32, reset_slot_dim=32, recovery_horizon=6)
    batch["teacher_u"] = batch["teacher_u"][:, :6]
    batch["teacher_traj"] = batch["teacher_traj"][:, :7]
    out = rpfn(batch)
    assert out["program_rollouts"].shape == (4, 6, 7, 12)
    assert out["program_controls"].shape == (4, 6, 6, 3)
    assert out["program_certificates"].shape == (4, 6)
    assert torch.isfinite(rpfn.loss(batch)["loss"])


def test_lower_tail_cvar_maximize():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([0.5, 3.0, 3.0, 3.0])
    assert lower_tail_cvar(a, beta=0.25) > lower_tail_cvar(b, beta=0.25)


def test_root_level_grouping(tmp_path: Path):
    path, ds, _ = _batch(tmp_path, n_roots=12, slot_dim=32)
    for root_id, idxs in ds.root_to_indices.items():
        splits = {ds.rows[i]["split"] for i in idxs}
        assert len(splits) == 1


def test_schema_backward_alias():
    row = {
        "root_id": "r0",
        "action_id": 1,
        "x_plus": [1.0] * 12,
        "event_time": 0.3,
        "deg": [1.0] * 6,
        "world_plus": {"affordance": [1.0, 2.0]},
        "event_tokens": [[0.1] * 4],
        "m_star": [0.1] * 5,
    }
    out = row_to_numpy(row, SchemaDims(reset_slot_dim=4, recovery_world_dim=8))
    assert "reset_state" in out and "recovery_world_vec" in out and "degradation" in out and "reset_slots" in out
    assert out["x_plus"].shape == out["reset_state"].shape
    assert out["world_plus"].shape == out["recovery_world_vec"].shape


def test_no_world_label_leakage(tmp_path: Path):
    path = tmp_path / "smoke.jsonl"
    rows = make_synthetic_rows(n_roots=4, seed=3)
    write_jsonl(rows, path)
    warnings = [w for r in rows for w in validate_row_no_leakage(r)]
    assert not [w for w in warnings if w.startswith("recovery_world_contains")]

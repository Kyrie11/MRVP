from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.synthetic import make_synthetic_rows, write_jsonl
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork


def test_forward_smoke(tmp_path: Path):
    path = tmp_path / "smoke.jsonl"
    write_jsonl(make_synthetic_rows(n_roots=8, seed=0), path)
    ds = MRVPDataset(path, split="train")
    batch = next(iter(DataLoader(ds, batch_size=8, collate_fn=mrvp_collate)))
    msrt = MSRT(hidden_dim=32, mixture_count=2)
    rpn = RecoveryProfileNetwork(hidden_dim=32)
    msrt_loss = msrt.loss(batch)["loss"]
    rpn_loss = rpn.loss(batch)["loss"]
    assert torch.isfinite(msrt_loss)
    assert torch.isfinite(rpn_loss)
    samples = msrt.sample(batch, num_samples=3)
    assert samples["x_plus"].shape[0] == 24

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mrvp.common.seed import set_seed
from mrvp.data.collate import collate_rows, collate_roots
from mrvp.data.dataset import MRVPDataset
from mrvp.models.cmrt import CMRT
from mrvp.models.rpfn import RPFN
from .utils import load_state_if_exists, move_batch, prepare_run, save_checkpoint, write_metrics


def ordering_step(cmrt, rpfn, roots, optimizer, device: str, train: bool) -> float:
    total = 0.0
    count = 0
    cmrt.eval()
    rpfn.train(train)
    for root_rows in roots:
        batch = collate_rows(root_rows)
        batch = move_batch(batch, device)
        with torch.no_grad():
            reset = cmrt.sample(batch, num_samples=1)[0]
        with torch.set_grad_enabled(train):
            out = rpfn(reset)
            pred = out["cert_pred"].max(dim=1).values
            truth = batch["score_star"]
            losses = []
            for i in range(len(root_rows)):
                for j in range(i + 1, len(root_rows)):
                    gap = truth[i] - truth[j]
                    if torch.abs(gap) >= 0.2:
                        sgn = torch.sign(gap)
                        losses.append(F.relu(0.1 - sgn * (pred[i] - pred[j])))
            if losses:
                loss = torch.stack(losses).mean()
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(rpfn.parameters(), 5.0)
                    optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--cmrt", required=True)
    parser.add_argument("--rpfn", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmrt_ckpt = torch.load(args.cmrt, map_location=device)
    rpfn_ckpt = torch.load(args.rpfn, map_location=device)
    cmrt = CMRT(cmrt_ckpt.get("cfg", {})).to(device)
    rpfn = RPFN(rpfn_ckpt.get("cfg", {})).to(device)
    load_state_if_exists(cmrt, args.cmrt, device)
    load_state_if_exists(rpfn, args.rpfn, device)
    train_ds = MRVPDataset(args.data, split="train", group_by_root=True)
    val_ds = MRVPDataset(args.data, split="val", group_by_root=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_roots)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_roots)
    optimizer = torch.optim.AdamW(rpfn.parameters(), lr=1e-4)
    out_dir = prepare_run(args.output, {"stage": "ordering"})
    best = float("inf")
    for epoch in range(args.epochs):
        train_loss = ordering_step(cmrt, rpfn, (item[0] for item in train_loader), optimizer, device, train=True)
        val_loss = ordering_step(cmrt, rpfn, (item[0] for item in val_loader), optimizer, device, train=False)
        write_metrics(out_dir, "train", epoch, {"loss": train_loss})
        write_metrics(out_dir, "val", epoch, {"loss": val_loss})
        if val_loss <= best:
            best = val_loss
            save_checkpoint(Path(out_dir) / "checkpoints" / "best.pt", rpfn, rpfn_ckpt.get("cfg", {}), {"val_ordering_loss": best})


if __name__ == "__main__":
    main()

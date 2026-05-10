#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.models.baselines import DirectActionRiskNetwork, UnstructuredLatentRiskNetwork
from mrvp.training.checkpoints import save_checkpoint
from mrvp.training.loops import eval_loss, train_one_epoch


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRVP experiment baseline networks.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="runs/default")
    parser.add_argument("--baseline", choices=["direct_action_to_risk", "unstructured_latent", "scalar_recoverability"], default="direct_action_to_risk")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    train_ds = MRVPDataset(args.data, split="train")
    val_ds = MRVPDataset(args.data, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=mrvp_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mrvp_collate)
    if args.baseline == "unstructured_latent":
        model = UnstructuredLatentRiskNetwork(hidden_dim=args.hidden_dim)
    else:
        model = DirectActionRiskNetwork(hidden_dim=args.hidden_dim, scalar=args.baseline == "scalar_recoverability")
    device = auto_device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_path = Path(args.out_dir) / f"{args.baseline}.pt"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": {"act": 0.5, "bd": 2.0, "sigma_bd": 0.5}}, desc=f"{args.baseline} train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": {"act": 0.5, "bd": 2.0, "sigma_bd": 0.5}}, desc=f"{args.baseline} val {epoch}")
        print(json.dumps({"baseline": args.baseline, "epoch": epoch, "train": tr, "val": va}))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"baseline": args.baseline, "epoch": epoch, "args": vars(args)})
    print(json.dumps({"output": str(out_path)}))


if __name__ == "__main__":
    main()

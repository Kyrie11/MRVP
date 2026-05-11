#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from mrvp.models.context import ContextPretrainingHead, SceneContextEncoder
from mrvp.public_pretraining.dataset import PublicPretrainDataset, pretrain_collate
from mrvp.training.checkpoints import save_checkpoint
from mrvp.training.loops import to_device


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain MRVP scene/context encoder on public driving data.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/default/context_encoder.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    ds = PublicPretrainDataset(args.data)
    n_val = max(1, int(0.1 * len(ds))) if len(ds) > 10 else 0
    if n_val:
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    else:
        train_ds, val_ds = ds, None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pretrain_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pretrain_collate) if val_ds else None
    device = auto_device(args.device)
    encoder = SceneContextEncoder(out_dim=128)
    model = ContextPretrainingHead(encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"context train {epoch}", leave=False):
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        logs = {"train_loss": total / max(n, 1)}
        if val_loader is not None:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = to_device(batch, device)
                    loss = model.loss(batch)
                    vt += float(loss.detach().cpu())
                    vn += 1
            logs["val_loss"] = vt / max(vn, 1)
        print(json.dumps({"epoch": epoch, **logs}))
    save_checkpoint(args.output, model.encoder, {"stage": "context_pretraining", "args": vars(args)})
    print(json.dumps({"output": args.output, "records": len(ds)}))


if __name__ == "__main__":
    main()

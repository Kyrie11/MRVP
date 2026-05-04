#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.pairs import SameRootPairDataset, pair_collate
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork, ordering_loss
from mrvp.training.checkpoints import load_model, save_checkpoint
from mrvp.training.loops import eval_loss, to_device, train_one_epoch


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_msrt(args, device: torch.device) -> Path:
    train_ds = MRVPDataset(args.data, split="train")
    val_ds = MRVPDataset(args.data, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mrvp_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mrvp_collate)
    model = MSRT(mixture_count=args.mixture_count, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    out_path = Path(args.out_dir) / "msrt.pt"
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": {"mech": args.lambda_mech, "phys": args.lambda_phys, "deg": 1.0}}, desc=f"msrt train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": {"mech": args.lambda_mech, "phys": args.lambda_phys, "deg": 1.0}}, desc=f"msrt val {epoch}")
        print(json.dumps({"stage": "msrt", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"stage": "msrt", "epoch": epoch, "args": vars(args)})
    return out_path


def train_rpn(args, device: torch.device) -> Path:
    train_ds = MRVPDataset(args.data, split="train")
    val_ds = MRVPDataset(args.data, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mrvp_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mrvp_collate)
    model = RecoveryProfileNetwork(hidden_dim=args.hidden_dim, scalar=args.scalar_rpn).to(device)
    if args.rpn_init and Path(args.rpn_init).exists():
        load_model(model, args.rpn_init, device, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    out_path = Path(args.out_dir) / ("rpn_scalar.pt" if args.scalar_rpn else "rpn.pt")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": {"act": args.lambda_act, "bd": args.lambda_bd, "sigma_bd": args.sigma_bd, "mono": args.lambda_mono}, "enable_mono": args.lambda_mono > 0}, desc=f"rpn train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": {"act": args.lambda_act, "bd": args.lambda_bd, "sigma_bd": args.sigma_bd}, "enable_mono": False}, desc=f"rpn val {epoch}")
        print(json.dumps({"stage": "rpn", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"stage": "rpn", "epoch": epoch, "args": vars(args)})
    return out_path


def finetune_pairs(args, device: torch.device) -> Path:
    base = MRVPDataset(args.data, split="train")
    pair_ds = SameRootPairDataset(base, eps_s=args.eps_s, max_pairs_per_root=args.max_pairs_per_root)
    loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=pair_collate)
    model = RecoveryProfileNetwork(hidden_dim=args.hidden_dim, scalar=args.scalar_rpn).to(device)
    init = Path(args.rpn_init) if args.rpn_init else Path(args.out_dir) / ("rpn_scalar.pt" if args.scalar_rpn else "rpn.pt")
    if init.exists():
        load_model(model, init, device, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        n = 0
        for batch in tqdm(loader, desc=f"pair finetune {epoch}", leave=False):
            bi = to_device(batch["i"], device)
            bj = to_device(batch["j"], device)
            opt.zero_grad(set_to_none=True)
            oi = model(bi)
            oj = model(bj)
            l_ord = ordering_loss(oi["V_smooth"], oj["V_smooth"], bi["s_star"].float(), bj["s_star"].float(), margin=args.pair_margin)
            li = model.loss(bi, lambdas={"act": args.lambda_act, "bd": args.lambda_bd, "sigma_bd": args.sigma_bd})["loss"]
            lj = model.loss(bj, lambdas={"act": args.lambda_act, "bd": args.lambda_bd, "sigma_bd": args.sigma_bd})["loss"]
            loss = 0.5 * (li + lj) + args.lambda_ord * l_ord
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            vals = {"loss": float(loss.detach().cpu()), "ord": float(l_ord.detach().cpu())}
            for k, v in vals.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        logs = {k: v / max(n, 1) for k, v in totals.items()}
        print(json.dumps({"stage": "finetune", "epoch": epoch, "train": logs, "pairs": len(pair_ds)}, ensure_ascii=False))
    out_path = Path(args.out_dir) / ("rpn_finetuned_scalar.pt" if args.scalar_rpn else "rpn_finetuned.pt")
    save_checkpoint(out_path, model, {"stage": "finetune", "epoch": args.epochs, "args": vars(args)})
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRVP MSRT/RPN models.")
    parser.add_argument("--data", required=True, help="MRVP JSONL dataset.")
    parser.add_argument("--out-dir", default="runs/default")
    parser.add_argument("--stage", choices=["msrt", "rpn", "finetune", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mixture-count", type=int, default=5)
    parser.add_argument("--lambda-mech", type=float, default=1.0)
    parser.add_argument("--lambda-phys", type=float, default=0.2)
    parser.add_argument("--lambda-act", type=float, default=0.5)
    parser.add_argument("--lambda-bd", type=float, default=2.0)
    parser.add_argument("--sigma-bd", type=float, default=0.5)
    parser.add_argument("--lambda-mono", type=float, default=0.0)
    parser.add_argument("--lambda-ord", type=float, default=1.0)
    parser.add_argument("--eps-s", type=float, default=0.25)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-root", type=int, default=64)
    parser.add_argument("--rpn-init", default="")
    parser.add_argument("--scalar-rpn", action="store_true", help="Ablation: one scalar margin broadcast to all bottlenecks.")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads; 1 is usually fastest for small batches.")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = auto_device(args.device)
    print(json.dumps({"device": str(device), "stage": args.stage, "out_dir": args.out_dir}, ensure_ascii=False))
    if args.stage in ("msrt", "all"):
        train_msrt(args, device)
    if args.stage in ("rpn", "all"):
        train_rpn(args, device)
    if args.stage in ("finetune", "all"):
        if not args.rpn_init:
            args.rpn_init = str(Path(args.out_dir) / ("rpn_scalar.pt" if args.scalar_rpn else "rpn.pt"))
        finetune_pairs(args, device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.pairs import SameRootPairDataset, pair_collate
from mrvp.data.schema import RECOVERY_HORIZON, STRATEGY_COUNT, TOKEN_COUNT, TOKEN_DIM, SchemaDims
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork, ordering_loss
from mrvp.training.checkpoints import load_model, save_checkpoint
from mrvp.training.loops import eval_loss, to_device, train_one_epoch


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_msrt(args) -> MSRT:
    return MSRT(
        mixture_count=args.mixture_count,
        hidden_dim=args.hidden_dim,
        token_count=args.token_count,
        token_dim=args.token_dim,
    )


def build_rpn(args) -> RecoveryProfileNetwork:
    return RecoveryProfileNetwork(
        hidden_dim=args.hidden_dim,
        token_count=args.token_count,
        token_dim=args.token_dim,
        strategy_count=args.strategy_count,
        recovery_horizon=args.recovery_horizon,
        scalar=args.scalar_rpn,
    )


def msrt_lambdas(args) -> Dict[str, float]:
    return {
        "nll": args.lambda_nll,
        "event": args.lambda_event,
        "token": args.lambda_token,
        "world": args.lambda_world,
        "deg": args.lambda_deg,
        "probe": args.lambda_probe,
        "ctr": args.lambda_ctr,
        "suf": args.lambda_suf,
        "temperature": args.contrastive_temperature,
    }


def rpn_lambdas(args) -> Dict[str, float]:
    return {
        "act": args.lambda_act,
        "bd": args.lambda_bd,
        "sigma_bd": args.sigma_bd,
        "strat": args.lambda_strat,
        "dyn": args.lambda_dyn,
        "ctrl": args.lambda_ctrl,
        "mono": args.lambda_mono,
        "lambda_xi": args.lambda_xi,
    }


def dataset_dims(args) -> SchemaDims:
    return SchemaDims(
        token_count=args.token_count,
        token_dim=args.token_dim,
        recovery_horizon=args.recovery_horizon,
    )


def train_msrt(args, device: torch.device) -> Path:
    dims = dataset_dims(args)
    train_ds = MRVPDataset(args.data, split="train", dims=dims)
    val_ds = MRVPDataset(args.data, split="val", dims=dims)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mrvp_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mrvp_collate)
    model = build_msrt(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    out_path = Path(args.out_dir) / "msrt.pt"
    lambdas = msrt_lambdas(args)
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": lambdas}, desc=f"msrt train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": lambdas}, desc=f"msrt val {epoch}")
        print(json.dumps({"stage": "msrt", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"stage": "msrt", "epoch": epoch, "args": vars(args)})
    return out_path


def train_rpn(args, device: torch.device) -> Path:
    dims = dataset_dims(args)
    train_ds = MRVPDataset(args.data, split="train", dims=dims)
    val_ds = MRVPDataset(args.data, split="val", dims=dims)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mrvp_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mrvp_collate)
    model = build_rpn(args).to(device)
    if args.rpn_init and Path(args.rpn_init).exists():
        load_model(model, args.rpn_init, device, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    out_path = Path(args.out_dir) / ("rpn_scalar.pt" if args.scalar_rpn else "rpn.pt")
    lambdas = rpn_lambdas(args)
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": lambdas, "enable_mono": args.lambda_mono > 0}, desc=f"rpn train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": lambdas, "enable_mono": False}, desc=f"rpn val {epoch}")
        print(json.dumps({"stage": "rpn", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"stage": "rpn", "epoch": epoch, "args": vars(args)})
    return out_path


def finetune_pairs(args, device: torch.device) -> Path:
    base = MRVPDataset(args.data, split="train", dims=dataset_dims(args))
    pair_ds = SameRootPairDataset(base, eps_s=args.eps_s, max_pairs_per_root=args.max_pairs_per_root)
    loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=pair_collate)
    model = build_rpn(args).to(device)
    init = Path(args.rpn_init) if args.rpn_init else Path(args.out_dir) / ("rpn_scalar.pt" if args.scalar_rpn else "rpn.pt")
    if init.exists():
        load_model(model, init, device, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    lambdas = rpn_lambdas(args)
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
            li = model.loss(bi, lambdas=lambdas)["loss"]
            lj = model.loss(bj, lambdas=lambdas)["loss"]
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
    parser.add_argument("--token-count", type=int, default=TOKEN_COUNT)
    parser.add_argument("--token-dim", type=int, default=TOKEN_DIM)
    parser.add_argument("--strategy-count", type=int, default=STRATEGY_COUNT)
    parser.add_argument("--recovery-horizon", type=int, default=RECOVERY_HORIZON)

    # MSRT losses. lambda-mech/lambda-phys are kept as legacy aliases but no
    # longer drive the revised path directly.
    parser.add_argument("--lambda-nll", type=float, default=1.0)
    parser.add_argument("--lambda-event", type=float, default=1.0)
    parser.add_argument("--lambda-token", type=float, default=0.5)
    parser.add_argument("--lambda-world", type=float, default=0.5)
    parser.add_argument("--lambda-deg", type=float, default=1.0)
    parser.add_argument("--lambda-probe", type=float, default=0.1)
    parser.add_argument("--lambda-ctr", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--lambda-mech", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-phys", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-suf", type=float, default=0.0, help="Optional gradient-reversal action adversary on event-token pool; use only for diagnostics.")

    # RPN losses.
    parser.add_argument("--lambda-act", type=float, default=0.5)
    parser.add_argument("--lambda-bd", type=float, default=2.0)
    parser.add_argument("--sigma-bd", type=float, default=0.5)
    parser.add_argument("--lambda-strat", type=float, default=0.5)
    parser.add_argument("--lambda-dyn", type=float, default=0.05)
    parser.add_argument("--lambda-ctrl", type=float, default=0.05)
    parser.add_argument("--lambda-xi", type=float, default=0.25)
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

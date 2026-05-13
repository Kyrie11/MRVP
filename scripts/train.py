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
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.pairs import SameRootPairDataset, pair_collate
from mrvp.data.schema import RECOVERY_HORIZON, PROGRAM_COUNT, RESET_SLOT_COUNT, RESET_SLOT_DIM, SchemaDims
from mrvp.models.cmrt import CounterfactualMotionResetTokenizer
from mrvp.models.rpfn import RecoveryProgramFunnelNetwork, ordering_loss
from mrvp.training.checkpoints import load_model, save_checkpoint
from mrvp.training.loops import eval_loss, to_device, train_one_epoch


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_cmrt(args) -> CounterfactualMotionResetTokenizer:
    return CounterfactualMotionResetTokenizer(
        mixture_count=args.mixture_count,
        hidden_dim=args.hidden_dim,
        reset_slot_count=args.reset_slot_count,
        reset_slot_dim=args.reset_slot_dim,
        prefix_horizon=args.prefix_horizon,
    )


def build_rpfn(args) -> RecoveryProgramFunnelNetwork:
    return RecoveryProgramFunnelNetwork(
        hidden_dim=args.hidden_dim,
        reset_slot_count=args.reset_slot_count,
        reset_slot_dim=args.reset_slot_dim,
        program_count=args.program_count,
        recovery_horizon=args.recovery_horizon,
        scalar=args.scalar_rpfn,
    )


def cmrt_lambdas(args) -> Dict[str, float]:
    return {
        "reset": args.lambda_reset,
        "world": args.lambda_world,
        "degradation": args.lambda_degradation,
        "counterfactual": args.lambda_counterfactual,
        "uncertainty": args.lambda_uncertainty,
        "audit_event": args.lambda_audit_event,
        "slot_distill": args.lambda_slot_distill,
        "probe": args.lambda_probe,
        "suf": args.lambda_suf,
        "temperature": args.contrastive_temperature,
    }


def rpfn_lambdas(args) -> Dict[str, float]:
    return {
        "execution": args.lambda_execution,
        "certificate": args.lambda_certificate,
        "funnel": args.lambda_funnel,
        "act": args.lambda_act,
        "bd": args.lambda_bd,
        "sigma_bd": args.sigma_bd,
        "dyn": args.lambda_dyn,
        "ctrl": args.lambda_ctrl,
        "mono": args.lambda_mono,
        "lambda_xi": args.lambda_xi,
    }


def dataset_dims(args) -> SchemaDims:
    return SchemaDims(
        reset_slot_count=args.reset_slot_count,
        reset_slot_dim=args.reset_slot_dim,
        program_count=args.program_count,
        recovery_horizon=args.recovery_horizon,
    )


def _make_loaders(args):
    dims = dataset_dims(args)
    train_ds = MRVPDataset(args.data, split="train", dims=dims)
    val_ds = MRVPDataset(args.data, split="val", dims=dims)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=mrvp_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=mrvp_collate)
    return train_ds, val_ds, train_loader, val_loader


def train_cmrt(args, device: torch.device) -> Path:
    _, _, train_loader, val_loader = _make_loaders(args)
    model = build_cmrt(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    out_path = Path(args.out_dir) / "cmrt.pt"
    lambdas = cmrt_lambdas(args)
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": lambdas}, desc=f"cmrt train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": lambdas}, desc=f"cmrt val {epoch}")
        print(json.dumps({"stage": "cmrt", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"stage": "cmrt", "epoch": epoch, "args": vars(args)})
    return out_path


def train_rpfn(args, device: torch.device) -> Path:
    _, _, train_loader, val_loader = _make_loaders(args)
    model = build_rpfn(args).to(device)
    if args.rpfn_init and Path(args.rpfn_init).exists():
        load_model(model, args.rpfn_init, device, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    out_path = Path(args.out_dir) / ("rpfn_scalar.pt" if args.scalar_rpfn else "rpfn.pt")
    lambdas = rpfn_lambdas(args)
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, loss_kwargs={"lambdas": lambdas, "enable_mono": args.lambda_mono > 0}, desc=f"rpfn train {epoch}")
        va = eval_loss(model, val_loader, device, loss_kwargs={"lambdas": lambdas, "enable_mono": False}, desc=f"rpfn val {epoch}")
        print(json.dumps({"stage": "rpfn", "epoch": epoch, "train": tr, "val": va}, ensure_ascii=False))
        if va.get("loss", tr["loss"]) < best:
            best = va.get("loss", tr["loss"])
            save_checkpoint(out_path, model, {"stage": "rpfn", "epoch": epoch, "args": vars(args)})
    return out_path


def finetune_pairs(args, device: torch.device) -> Path:
    base = MRVPDataset(args.data, split="train", dims=dataset_dims(args))
    pair_ds = SameRootPairDataset(base, eps_s=args.eps_s, max_pairs_per_root=args.max_pairs_per_root)
    loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=pair_collate)
    model = build_rpfn(args).to(device)
    init = Path(args.rpfn_init) if args.rpfn_init else Path(args.out_dir) / ("rpfn_scalar.pt" if args.scalar_rpfn else "rpfn.pt")
    if init.exists():
        load_model(model, init, device, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    lambdas = rpfn_lambdas(args)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Dict[str, float] = {}
        n = 0
        for batch in tqdm(loader, desc=f"rpfn pair finetune {epoch}", leave=False):
            bi = to_device(batch["i"], device)
            bj = to_device(batch["j"], device)
            opt.zero_grad(set_to_none=True)
            oi = model(bi)
            oj = model(bj)
            l_ord = ordering_loss(oi["best_certificate"], oj["best_certificate"], bi["s_star"].float(), bj["s_star"].float(), margin=args.pair_margin)
            li = model.loss(bi, lambdas=lambdas)["loss"]
            lj = model.loss(bj, lambdas=lambdas)["loss"]
            loss = 0.5 * (li + lj) + args.lambda_order * l_ord
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            vals = {"loss": float(loss.detach().cpu()), "order": float(l_ord.detach().cpu())}
            for k, v in vals.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        logs = {k: v / max(n, 1) for k, v in totals.items()}
        print(json.dumps({"stage": "finetune", "epoch": epoch, "train": logs, "pairs": len(pair_ds)}, ensure_ascii=False))
    out_path = Path(args.out_dir) / ("rpfn_finetuned_scalar.pt" if args.scalar_rpfn else "rpfn_finetuned.pt")
    save_checkpoint(out_path, model, {"stage": "finetune", "epoch": args.epochs, "args": vars(args)})
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MRVP CMRT/RPFN models.")
    parser.add_argument("--data", required=True, help="MRVP JSONL dataset.")
    parser.add_argument("--out-dir", default="runs/default")
    parser.add_argument("--stage", choices=["cmrt", "rpfn", "finetune", "all", "msrt", "rpn"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mixture-count", type=int, default=5)
    parser.add_argument("--reset-slot-count", type=int, default=RESET_SLOT_COUNT)
    parser.add_argument("--reset-slot-dim", type=int, default=RESET_SLOT_DIM)
    parser.add_argument("--program-count", type=int, default=PROGRAM_COUNT)
    parser.add_argument("--recovery-horizon", type=int, default=RECOVERY_HORIZON)
    parser.add_argument("--prefix-horizon", type=int, default=20)
    # Legacy argument aliases.
    parser.add_argument("--token-count", dest="reset_slot_count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--token-dim", dest="reset_slot_dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--strategy-count", dest="program_count", type=int, help=argparse.SUPPRESS)

    # CMRT losses.
    parser.add_argument("--lambda-reset", type=float, default=1.0)
    parser.add_argument("--lambda-world", type=float, default=0.5)
    parser.add_argument("--lambda-degradation", type=float, default=1.0)
    parser.add_argument("--lambda-counterfactual", type=float, default=0.2)
    parser.add_argument("--lambda-uncertainty", type=float, default=1.0)
    parser.add_argument("--lambda-audit-event", type=float, default=0.0)
    parser.add_argument("--lambda-slot-distill", type=float, default=0.0)
    parser.add_argument("--lambda-probe", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--lambda-suf", type=float, default=0.0, help="Optional diagnostic action adversary on reset slot pool.")
    # Legacy loss aliases.
    parser.add_argument("--lambda-event", dest="lambda_audit_event", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-token", dest="lambda_slot_distill", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-deg", dest="lambda_degradation", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-nll", dest="lambda_reset", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-ctr", dest="lambda_counterfactual", type=float, help=argparse.SUPPRESS)

    # RPFN losses.
    parser.add_argument("--lambda-execution", type=float, default=0.5)
    parser.add_argument("--lambda-certificate", type=float, default=1.0)
    parser.add_argument("--lambda-funnel", type=float, default=0.2)
    parser.add_argument("--lambda-act", type=float, default=0.0)
    parser.add_argument("--lambda-bd", type=float, default=2.0)
    parser.add_argument("--sigma-bd", type=float, default=0.5)
    parser.add_argument("--lambda-dyn", type=float, default=0.02)
    parser.add_argument("--lambda-ctrl", type=float, default=0.05)
    parser.add_argument("--lambda-xi", type=float, default=0.25)
    parser.add_argument("--lambda-mono", type=float, default=0.0)
    parser.add_argument("--lambda-order", type=float, default=1.0)
    parser.add_argument("--eps-s", type=float, default=0.25)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--max-pairs-per-root", type=int, default=64)
    parser.add_argument("--rpfn-init", default="")
    parser.add_argument("--scalar-rpfn", action="store_true", help="Ablation: one scalar certificate broadcast to all margins.")
    # Legacy aliases.
    parser.add_argument("--rpn-init", dest="rpfn_init", default="", help=argparse.SUPPRESS)
    parser.add_argument("--scalar-rpn", dest="scalar_rpfn", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lambda-strat", dest="lambda_execution", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lambda-ord", dest="lambda_order", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads; 1 is usually fastest for small batches.")
    args = parser.parse_args()
    if args.stage == "msrt":
        args.stage = "cmrt"
    if args.stage == "rpn":
        args.stage = "rpfn"
    torch.set_num_threads(max(1, args.torch_threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=True, allow_unicode=True), encoding="utf-8")
    device = auto_device(args.device)
    print(json.dumps({"device": str(device), "stage": args.stage, "out_dir": args.out_dir}, ensure_ascii=False))
    if args.stage in ("cmrt", "all"):
        train_cmrt(args, device)
    if args.stage in ("rpfn", "all"):
        train_rpfn(args, device)
    if args.stage in ("finetune", "all"):
        if not args.rpfn_init:
            args.rpfn_init = str(out_dir / ("rpfn_scalar.pt" if args.scalar_rpfn else "rpfn.pt"))
        finetune_pairs(args, device)


if __name__ == "__main__":
    main()

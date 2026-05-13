#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.schema import NUM_ACTIONS, RESET_SLOT_COUNT, RESET_SLOT_DIM, SchemaDims
from mrvp.models.common import make_mlp
from mrvp.models.cmrt import CounterfactualMotionResetTokenizer
from mrvp.training.checkpoints import load_model
from mrvp.training.loops import to_device


class Probe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, target: str = "recoverable") -> None:
        super().__init__()
        out_dim = 1 if target == "recoverable" else 5
        self.target = target
        self.net = make_mlp(in_dim, [hidden_dim, hidden_dim], out_dim, dropout=0.05, layer_norm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) if self.target == "recoverable" else self.net(x)


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def binary_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = y_true.astype(np.int64).reshape(-1)
    p = prob.astype(np.float64).reshape(-1)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_p = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        if end - start > 1:
            avg = 0.5 * (start + 1 + end)
            ranks[order[start:end]] = avg
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(np.int64).reshape(-1)
    p = y_pred.astype(np.int64).reshape(-1)
    vals = []
    for cls in [0, 1]:
        mask = y == cls
        if mask.sum() == 0:
            continue
        vals.append(float((p[mask] == cls).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(f1s))


@torch.no_grad()
def build_features(
    data_path: str,
    split: str,
    feature_source: str,
    cmrt: CounterfactualMotionResetTokenizer | None,
    device: torch.device,
    batch_size: int,
    reset_slot_count: int = RESET_SLOT_COUNT,
    reset_slot_dim: int = RESET_SLOT_DIM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dims = SchemaDims(reset_slot_count=reset_slot_count, reset_slot_dim=reset_slot_dim)
    ds = MRVPDataset(data_path, split=split, dims=dims)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=mrvp_collate)
    x0_parts = []
    x1_parts = []
    y_rec_parts = []
    y_b_parts = []
    for batch in tqdm(loader, desc=f"features {split}", leave=False):
        if feature_source == "predicted_cmrt":
            if cmrt is None:
                raise ValueError("--feature-source predicted_cmrt requires --cmrt")
            mb = to_device(batch, device)
            pred = cmrt.sample_reset_problems(mb, num_samples=1, deterministic=True)
            reset_state = pred["reset_state"].detach().cpu()
            degradation = pred["degradation"].detach().cpu()
            reset_slots = pred["reset_slots"].detach().cpu()
            recovery_world = pred["recovery_world_vec"].detach().cpu()
        else:
            reset_state = batch["reset_state"].float()
            degradation = batch["degradation"].float()
            reset_slots = batch["reset_slots"].float()
            recovery_world = batch["recovery_world_vec"].float()
        h_ctx = batch["h_ctx"].float()
        aid = batch["action_id"].long().clamp_min(0) % NUM_ACTIONS
        action_onehot = F.one_hot(aid, NUM_ACTIONS).float()
        action_feat = torch.cat([batch["action_vec"].float(), action_onehot], dim=-1)
        x0 = torch.cat([reset_state, degradation, reset_slots.flatten(1), recovery_world, h_ctx], dim=-1)
        x1 = torch.cat([x0, action_feat], dim=-1)
        x0_parts.append(x0.numpy())
        x1_parts.append(x1.numpy())
        y_rec_parts.append((batch["s_star"].float() >= 0.0).long().numpy())
        y_b_parts.append(batch["b_star"].long().numpy())
    return (
        np.concatenate(x0_parts, axis=0).astype(np.float32),
        np.concatenate(x1_parts, axis=0).astype(np.float32),
        np.concatenate(y_rec_parts, axis=0).astype(np.int64),
        np.concatenate(y_b_parts, axis=0).astype(np.int64),
    )


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    stats = {"mean_abs": float(np.abs(mean).mean()), "std_mean": float(std.mean())}
    return ((train - mean) / std).astype(np.float32), ((val - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32), stats


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle)


def loss_for_logits(logits: torch.Tensor, y: torch.Tensor, target: str) -> torch.Tensor:
    if target == "recoverable":
        return F.binary_cross_entropy_with_logits(logits, y.float())
    return F.cross_entropy(logits, y.long())


@torch.no_grad()
def evaluate_probe(model: Probe, loader: DataLoader, device: torch.device, target: str) -> Dict[str, float]:
    model.eval()
    losses = []
    ys = []
    outs = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        losses.append(float(loss_for_logits(logits, y, target).detach().cpu()) * x.shape[0])
        ys.append(y.detach().cpu().numpy())
        outs.append(logits.detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0)
    out_np = np.concatenate(outs, axis=0)
    nll = float(np.sum(losses) / max(1, len(y_np)))
    if target == "recoverable":
        prob = 1.0 / (1.0 + np.exp(-out_np))
        pred = (prob >= 0.5).astype(np.int64)
        return {
            "nll": nll,
            "auc": binary_auc(y_np, prob),
            "balanced_acc": balanced_accuracy(y_np, pred),
        }
    pred = out_np.argmax(axis=-1).astype(np.int64)
    return {
        "nll": nll,
        "macro_f1": macro_f1_np(y_np, pred, num_classes=5),
        "acc": float((pred == y_np).mean()),
    }


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    model = Probe(x_train.shape[1], hidden_dim=hidden_dim, target=target).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)
    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_for_logits(logits, y, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu()) * x.shape[0]
            seen += x.shape[0]
        val_metrics = evaluate_probe(model, val_loader, device, target)
        train_nll = total / max(1, seen)
        history.append({"epoch": epoch, "train_nll": train_nll, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["nll"] < best_val:
            best_val = val_metrics["nll"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"val": evaluate_probe(model, val_loader, device, target), "test": evaluate_probe(model, test_loader, device, target), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Residual-action sufficiency diagnostic. Probe0 sees (reset_state,degradation,reset_slots,recovery_world,h_ctx); "
            "Probe1 additionally sees action_vec and action_id one-hot. A small Probe1-Probe0 gain supports CMRT reset-problem sufficiency."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/default/residual_action_diagnostic.json")
    parser.add_argument("--feature-source", choices=["true_reset_problem", "predicted_cmrt"], default="true_reset_problem")
    parser.add_argument("--cmrt", default="")
    parser.add_argument("--target", choices=["recoverable", "bottleneck"], default="recoverable")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cmrt-hidden-dim", type=int, default=256)
    parser.add_argument("--mixture-count", type=int, default=5)
    parser.add_argument("--reset-slot-count", type=int, default=RESET_SLOT_COUNT)
    parser.add_argument("--reset-slot-dim", type=int, default=RESET_SLOT_DIM)
    parser.add_argument("--token-count", dest="reset_slot_count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--token-dim", dest="reset_slot_dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--msrt", dest="cmrt", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = auto_device(args.device)
    cmrt = None
    if args.feature_source == "predicted_cmrt":
        if not args.cmrt:
            raise SystemExit("--feature-source predicted_cmrt requires --cmrt")
        cmrt = CounterfactualMotionResetTokenizer(
            hidden_dim=args.cmrt_hidden_dim,
            mixture_count=args.mixture_count,
            reset_slot_count=args.reset_slot_count,
            reset_slot_dim=args.reset_slot_dim,
        ).to(device)
        load_model(cmrt, args.cmrt, device, strict=False)
        cmrt.eval()

    x0_train, x1_train, y_rec_train, y_b_train = build_features(args.data, "train", args.feature_source, cmrt, device, args.batch_size, args.reset_slot_count, args.reset_slot_dim)
    x0_val, x1_val, y_rec_val, y_b_val = build_features(args.data, "val", args.feature_source, cmrt, device, args.batch_size, args.reset_slot_count, args.reset_slot_dim)
    x0_test, x1_test, y_rec_test, y_b_test = build_features(args.data, "test", args.feature_source, cmrt, device, args.batch_size, args.reset_slot_count, args.reset_slot_dim)

    y_train, y_val, y_test = (y_rec_train, y_rec_val, y_rec_test) if args.target == "recoverable" else (y_b_train, y_b_val, y_b_test)
    x0_train_s, x0_val_s, x0_test_s, stats0 = standardize(x0_train, x0_val, x0_test)
    x1_train_s, x1_val_s, x1_test_s, stats1 = standardize(x1_train, x1_val, x1_test)

    probe0 = train_probe(x0_train_s, y_train, x0_val_s, y_val, x0_test_s, y_test, args.target, args.hidden_dim, args.epochs, args.batch_size, args.lr, device, args.seed)
    probe1 = train_probe(x1_train_s, y_train, x1_val_s, y_val, x1_test_s, y_test, args.target, args.hidden_dim, args.epochs, args.batch_size, args.lr, device, args.seed + 1)

    test0 = probe0["test"]
    test1 = probe1["test"]
    deltas: Dict[str, float] = {"delta_nll_probe1_minus_probe0": float(test1["nll"] - test0["nll"])}
    if args.target == "recoverable":
        deltas["delta_auc_probe1_minus_probe0"] = float(test1["auc"] - test0["auc"])
        deltas["delta_balanced_acc_probe1_minus_probe0"] = float(test1["balanced_acc"] - test0["balanced_acc"])
    else:
        deltas["delta_macro_f1_probe1_minus_probe0"] = float(test1["macro_f1"] - test0["macro_f1"])
        deltas["delta_acc_probe1_minus_probe0"] = float(test1["acc"] - test0["acc"])

    out = {
        "feature_source": args.feature_source,
        "target": args.target,
        "interpretation": {
            "probe0_input": "reset_state,degradation,reset_slots,recovery_world,h_ctx",
            "probe1_input": "reset_state,degradation,reset_slots,recovery_world,h_ctx,action_vec,action_id_onehot",
            "rule_of_thumb": "Small or negative Probe1-Probe0 gain supports CMRT reset-problem sufficiency; a large positive gain indicates residual action shortcut or missing reset information.",
        },
        "probe0_no_action": {"val": probe0["val"], "test": probe0["test"]},
        "probe1_with_action": {"val": probe1["val"], "test": probe1["test"]},
        "deltas": deltas,
        "feature_stats": {"probe0": stats0, "probe1": stats1},
        "sizes": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(path), "deltas": deltas}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

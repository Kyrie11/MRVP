#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mrvp.calibration import load_calibration_table
from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.selection import select_action_with_models
from mrvp.training.checkpoints import load_model


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MRVP Algorithm 1 for one root scenario from a JSONL dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--root-id", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--msrt", required=True)
    parser.add_argument("--rpn", required=True)
    parser.add_argument("--calibration", default="")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mixture-count", type=int, default=5)
    parser.add_argument("--scalar-rpn", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    device = auto_device(args.device)
    ds = MRVPDataset(args.data, split=args.split)
    idxs = ds.root_to_indices.get(str(args.root_id))
    if not idxs:
        raise SystemExit(f"root_id {args.root_id!r} not found in data/split")
    root_items = [ds[i] for i in idxs]
    root_batch = mrvp_collate(root_items)
    root_rows = [ds.rows[i] for i in idxs]
    msrt = MSRT(mixture_count=args.mixture_count, hidden_dim=args.hidden_dim)
    rpn = RecoveryProfileNetwork(hidden_dim=args.hidden_dim, scalar=args.scalar_rpn)
    load_model(msrt, args.msrt, device, strict=False)
    load_model(rpn, args.rpn, device, strict=False)
    table = load_calibration_table(args.calibration or None)
    sel = select_action_with_models(root_batch, root_rows, msrt, rpn, table, num_samples=args.num_samples, beta=args.beta, device=device)
    print(json.dumps(sel, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import torch

from mrvp.calibration import fit_calibration_table, predict_rpn, save_calibration_table
from mrvp.data.dataset import MRVPDataset
from mrvp.data.schema import TOKEN_COUNT, TOKEN_DIM, STRATEGY_COUNT, RECOVERY_HORIZON, SchemaDims
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.models.baselines import DirectActionRiskNetwork, UnstructuredLatentRiskNetwork
from mrvp.training.checkpoints import load_model


def build_profile_model(model_type: str, hidden_dim: int, scalar_rpn: bool = False, token_count: int = TOKEN_COUNT, token_dim: int = TOKEN_DIM, strategy_count: int = STRATEGY_COUNT, recovery_horizon: int = RECOVERY_HORIZON):
    if model_type == "rpn":
        return RecoveryProfileNetwork(hidden_dim=hidden_dim, token_count=token_count, token_dim=token_dim, strategy_count=strategy_count, recovery_horizon=recovery_horizon, scalar=scalar_rpn)
    if model_type == "direct_action_to_risk":
        return DirectActionRiskNetwork(hidden_dim=hidden_dim, scalar=scalar_rpn)
    if model_type == "unstructured_latent":
        return UnstructuredLatentRiskNetwork(hidden_dim=hidden_dim)
    raise ValueError(f"Unknown model_type: {model_type}")


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit group/scenario-level MRVP calibration quantiles.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--rpn", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="cal")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--delta-b", type=float, default=0.02)
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--token-count", type=int, default=TOKEN_COUNT)
    parser.add_argument("--token-dim", type=int, default=TOKEN_DIM)
    parser.add_argument("--strategy-count", type=int, default=STRATEGY_COUNT)
    parser.add_argument("--recovery-horizon", type=int, default=RECOVERY_HORIZON)
    parser.add_argument("--scalar-rpn", action="store_true")
    parser.add_argument("--model-type", choices=["rpn", "direct_action_to_risk", "unstructured_latent"], default="rpn")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    device = auto_device(args.device)
    dims = SchemaDims(token_count=args.token_count, token_dim=args.token_dim, recovery_horizon=args.recovery_horizon)
    ds = MRVPDataset(args.data, split=args.split, dims=dims)
    model = build_profile_model(args.model_type, args.hidden_dim, args.scalar_rpn, args.token_count, args.token_dim, args.strategy_count, args.recovery_horizon)
    load_model(model, args.rpn, device, strict=False)
    r_hat = predict_rpn(model, ds, batch_size=args.batch_size, device=device)
    table = fit_calibration_table(ds.rows, r_hat, delta_b=args.delta_b, n_min=args.n_min)
    save_calibration_table(table, args.output)
    print(json.dumps({"output": args.output, "split": args.split, "roots": len(ds.root_to_indices), "rows": len(ds), "groups_full": len(table["quantiles"].get("full", {}))}, indent=2))


if __name__ == "__main__":
    main()

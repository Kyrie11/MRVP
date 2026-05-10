#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mrvp.calibration import load_calibration_table, lower_bounds_for_rows, predict_rpn
from mrvp.data.dataset import MRVPDataset
from mrvp.evaluation import baseline_scores, evaluate_selection, lower_bounds_from_scalar_scores
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.models.baselines import DirectActionRiskNetwork, UnstructuredLatentRiskNetwork
from mrvp.training.checkpoints import load_model


def build_profile_model(model_type: str, hidden_dim: int, scalar_rpn: bool = False):
    if model_type == "rpn":
        return RecoveryProfileNetwork(hidden_dim=hidden_dim, scalar=scalar_rpn)
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
    parser = argparse.ArgumentParser(description="Evaluate MRVP and paper baselines on a held-out split.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--rpn", default="")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="runs/default/eval_test.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--scalar-rpn", action="store_true")
    parser.add_argument("--model-type", choices=["rpn", "direct_action_to_risk", "unstructured_latent"], default="rpn")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    device = auto_device(args.device)
    ds = MRVPDataset(args.data, split=args.split)
    results = {}
    if args.rpn:
        model = build_profile_model(args.model_type, args.hidden_dim, args.scalar_rpn)
        load_model(model, args.rpn, device, strict=False)
        r_hat = predict_rpn(model, ds, batch_size=args.batch_size, device=device)
        table = load_calibration_table(args.calibration or None)
        lower = lower_bounds_for_rows(r_hat, ds.rows, table)
        results["Oracle_MRVP_true_transition"] = evaluate_selection(ds.rows, lower, r_hat)
        results["Uncalibrated_Oracle_MRVP"] = evaluate_selection(ds.rows, r_hat, r_hat)
        mean_lower = np.repeat(lower.mean(axis=1, keepdims=True), lower.shape[1], axis=1)
        results["Scalar_recoverability_network_proxy"] = evaluate_selection(ds.rows, mean_lower, mean_lower)
    for name, kind in [("Severity_only", "severity"), ("Weighted_post_impact_cost", "weighted_post_impact"), ("Teacher_oracle", "teacher_oracle")]:
        scores = baseline_scores(ds.rows, kind=kind)
        fake_lower = lower_bounds_from_scalar_scores(ds.rows, scores)
        results[name] = evaluate_selection(ds.rows, fake_lower, fake_lower)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    keys = sorted({k for m in results.values() for k in m.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + keys)
        writer.writeheader()
        for method, vals in results.items():
            writer.writerow({"method": method, **vals})
    print(json.dumps({"output": str(out), "csv": str(csv_path), "methods": list(results.keys())}, indent=2))


if __name__ == "__main__":
    main()

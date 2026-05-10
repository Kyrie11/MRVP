#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from mrvp.calibration import load_calibration_table, lower_bounds_for_rows, predict_rpn, quantile_for_group
from mrvp.data.dataset import MRVPDataset, iter_root_batches
from mrvp.data.schema import BOTTLE_NECKS
from mrvp.evaluation import (
    baseline_scores,
    evaluate_selected_indices,
    evaluate_selection,
    lower_bounds_from_scalar_scores,
)
from mrvp.models.baselines import DirectActionRiskNetwork, UnstructuredLatentRiskNetwork
from mrvp.models.common import empirical_cvar
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.selection import admissible_indices
from mrvp.training.checkpoints import load_model
from mrvp.training.loops import to_device


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_profile_model(model_type: str, hidden_dim: int, scalar: bool = False):
    if model_type == "rpn":
        return RecoveryProfileNetwork(hidden_dim=hidden_dim, scalar=scalar)
    if model_type == "direct_action_to_risk":
        return DirectActionRiskNetwork(hidden_dim=hidden_dim, scalar=scalar)
    if model_type == "unstructured_latent":
        return UnstructuredLatentRiskNetwork(hidden_dim=hidden_dim)
    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def score_root_with_msrt(
    root_batch: Dict[str, Any],
    root_rows: Sequence[Mapping[str, Any]],
    msrt: torch.nn.Module,
    rpn: torch.nn.Module,
    calibration_table: Mapping[str, Any],
    num_samples: int,
    beta: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Score every action in one root by MSRT samples -> RPN -> calibration -> CVaR.

    Returns one summary per local action. Selection should still be restricted
    to the minimum-harm-bin admissible set by the caller.
    """
    batch = to_device(root_batch, device)
    summaries: List[Dict[str, Any]] = []
    for local_idx, row in enumerate(root_rows):
        single: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                single[k] = v[local_idx : local_idx + 1]
            else:
                single[k] = [v[local_idx]] if isinstance(v, list) else v
        samples = msrt.sample(single, num_samples=num_samples, deterministic=False)
        rep = {
            "o_hist": single["o_hist"].repeat_interleave(num_samples, dim=0),
            "h_ctx": single["h_ctx"].repeat_interleave(num_samples, dim=0),
            "x_plus": samples["x_plus"],
            "d_deg": samples["d_deg"],
            "z_mech": samples["z_mech"],
        }
        pred = rpn(rep)["r_hat"]
        q = torch.as_tensor(
            quantile_for_group(row.get("calib_group", {}), calibration_table),
            device=device,
            dtype=pred.dtype,
        )
        lower = pred - q[None, :]
        v_lower = lower.min(dim=-1).values
        losses = torch.clamp(-v_lower, min=0.0)
        cvar = empirical_cvar(losses[None, :], beta=beta, dim=-1).squeeze(0)
        mean_risk = losses.mean()
        mean_lower = lower.mean(dim=0)
        summaries.append(
            {
                "local_index": int(local_idx),
                "action_id": int(row["action_id"]),
                "action_name": row.get("action_name", str(row["action_id"])),
                "harm_bin": int(row["harm_bin"]),
                "risk_cvar": float(cvar.detach().cpu()),
                "risk_mean": float(mean_risk.detach().cpu()),
                "score_cvar": float(-cvar.detach().cpu()),
                "score_mean": float(-mean_risk.detach().cpu()),
                "mean_lower_V": float(v_lower.mean().detach().cpu()),
                "p_violation": float((losses > 0).float().mean().detach().cpu()),
                "mean_lower_bounds": mean_lower.detach().cpu().numpy().astype(float).tolist(),
            }
        )
    return summaries


def select_from_root_summaries(root_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], risk_key: str) -> int:
    adm = admissible_indices(root_rows)
    if not adm:
        raise ValueError("No admissible action for root.")
    best = min(adm, key=lambda i: float(summaries[i][risk_key]))
    return int(best)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Claim-level evaluation: Oracle-MRVP uses true x_plus/d/z; "
            "Predicted-MRVP samples MSRT from pre-impact scene/action and then runs RPN + calibration + CVaR."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--msrt", required=True)
    parser.add_argument("--rpn", required=True)
    parser.add_argument("--calibration", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="runs/default/eval_predicted_mrvp.json")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mixture-count", type=int, default=5)
    parser.add_argument("--scalar-rpn", action="store_true")
    parser.add_argument("--direct-model", default="", help="Optional trained direct_action_to_risk checkpoint.")
    parser.add_argument("--direct-calibration", default="", help="Optional calibration table for the direct baseline.")
    parser.add_argument("--direct-model-type", choices=["direct_action_to_risk", "unstructured_latent"], default="direct_action_to_risk")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = auto_device(args.device)
    ds = MRVPDataset(args.data, split=args.split)

    msrt = MSRT(hidden_dim=args.hidden_dim, mixture_count=args.mixture_count).to(device)
    load_model(msrt, args.msrt, device, strict=False)
    msrt.eval()
    rpn = RecoveryProfileNetwork(hidden_dim=args.hidden_dim, scalar=args.scalar_rpn).to(device)
    load_model(rpn, args.rpn, device, strict=False)
    rpn.eval()
    table = load_calibration_table(args.calibration or None)

    # Oracle upper-bound path: RPN consumes true transition/mechanism labels.
    oracle_r_hat = predict_rpn(rpn, ds, batch_size=args.batch_size, device=device)
    oracle_lower = lower_bounds_for_rows(oracle_r_hat, ds.rows, table)
    results: Dict[str, Dict[str, float]] = {
        "Oracle_MRVP_true_transition": evaluate_selection(ds.rows, oracle_lower, oracle_r_hat),
        "Oracle_MRVP_uncalibrated": evaluate_selection(ds.rows, oracle_r_hat, oracle_r_hat),
    }

    # Predicted path: score all actions in root batches, then select only within the harm gate.
    scores_cvar = np.full((len(ds),), np.nan, dtype=np.float32)
    scores_mean = np.full((len(ds),), np.nan, dtype=np.float32)
    lower_mean = np.zeros((len(ds), len(BOTTLE_NECKS)), dtype=np.float32)
    selected_cvar: List[int] = []
    selected_mean: List[int] = []
    per_root: List[Dict[str, Any]] = []

    for root_id, indices, root_rows, root_batch in tqdm(iter_root_batches(ds), total=len(ds.root_to_indices), desc="predicted_mrvp"):
        summaries = score_root_with_msrt(root_batch, root_rows, msrt, rpn, table, args.num_samples, args.beta, device)
        best_cvar_local = select_from_root_summaries(root_rows, summaries, risk_key="risk_cvar")
        best_mean_local = select_from_root_summaries(root_rows, summaries, risk_key="risk_mean")
        selected_cvar.append(indices[best_cvar_local])
        selected_mean.append(indices[best_mean_local])
        for local_i, global_i in enumerate(indices):
            scores_cvar[global_i] = summaries[local_i]["score_cvar"]
            scores_mean[global_i] = summaries[local_i]["score_mean"]
            lower_mean[global_i] = np.asarray(summaries[local_i]["mean_lower_bounds"], dtype=np.float32)
        adm = admissible_indices(root_rows)
        teacher_local = max(adm, key=lambda i: float(root_rows[i]["s_star"])) if adm else 0
        per_root.append(
            {
                "root_id": root_id,
                "admissible_local_indices": adm,
                "selected_cvar_local": best_cvar_local,
                "selected_cvar_global": indices[best_cvar_local],
                "selected_mean_local": best_mean_local,
                "selected_mean_global": indices[best_mean_local],
                "teacher_best_local": teacher_local,
                "teacher_best_global": indices[teacher_local],
                "selected_cvar_s_star": float(root_rows[best_cvar_local]["s_star"]),
                "selected_mean_s_star": float(root_rows[best_mean_local]["s_star"]),
                "teacher_best_s_star": float(root_rows[teacher_local]["s_star"]),
                "candidate_summaries": summaries,
            }
        )

    results["Predicted_MRVP_MSRT_CVaR"] = evaluate_selected_indices(ds.rows, selected_cvar, scores=scores_cvar, lower_bounds=lower_mean)
    results["Predicted_MRVP_MSRT_mean_risk"] = evaluate_selected_indices(ds.rows, selected_mean, scores=scores_mean, lower_bounds=lower_mean)

    if args.direct_model:
        direct = build_profile_model(args.direct_model_type, args.hidden_dim, scalar=args.scalar_rpn).to(device)
        load_model(direct, args.direct_model, device, strict=False)
        direct_r_hat = predict_rpn(direct, ds, batch_size=args.batch_size, device=device)
        direct_table = load_calibration_table(args.direct_calibration or None)
        direct_lower = lower_bounds_for_rows(direct_r_hat, ds.rows, direct_table)
        results[f"Direct_baseline_{args.direct_model_type}"] = evaluate_selection(ds.rows, direct_lower, direct_r_hat)

    for name, kind in [("Severity_only", "severity"), ("Weighted_post_impact_cost", "weighted_post_impact"), ("Teacher_oracle", "teacher_oracle")]:
        base_scores = baseline_scores(ds.rows, kind=kind)
        fake_lower = lower_bounds_from_scalar_scores(ds.rows, base_scores)
        results[name] = evaluate_selection(ds.rows, fake_lower, fake_lower)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    keys = sorted({k for m in results.values() for k in m.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + keys)
        writer.writeheader()
        for method, vals in results.items():
            writer.writerow({"method": method, **vals})
    detail_path = out.with_name(out.stem + "_per_root.jsonl")
    write_jsonl(detail_path, per_root)
    print(
        json.dumps(
            {
                "output": str(out),
                "csv": str(csv_path),
                "per_root": str(detail_path),
                "methods": list(results.keys()),
                "roots": len(ds.root_to_indices),
                "rows": len(ds),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

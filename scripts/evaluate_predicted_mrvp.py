#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrvp.action_selection import harm_comparable_indices, select_tail_consistent_action
from mrvp.data.dataset import MRVPDataset, iter_root_batches, mrvp_collate
from mrvp.data.schema import BOTTLE_NECKS, RECOVERY_HORIZON, PROGRAM_COUNT, RESET_SLOT_COUNT, RESET_SLOT_DIM, SchemaDims
from mrvp.evaluation import baseline_scores, evaluate_selected_indices, evaluate_selection, lower_bounds_from_scalar_scores
from mrvp.models.cmrt import CounterfactualMotionResetTokenizer
from mrvp.models.rpfn import RecoveryProgramFunnelNetwork
from mrvp.training.checkpoints import load_model
from mrvp.training.loops import to_device


def auto_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict_rpfn_profiles(rpfn: torch.nn.Module, ds: MRVPDataset, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=mrvp_collate)
    out = []
    rpfn.eval().to(device)
    for batch in tqdm(loader, desc="oracle_rpfn", leave=False):
        batch = to_device(batch, device)
        pred = rpfn(batch)["best_profile"].detach().cpu().numpy()
        out.append(pred)
    return np.concatenate(out, axis=0) if out else np.zeros((0, len(BOTTLE_NECKS)), dtype=np.float32)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full predicted MRVP: CMRT samples -> RPFN programs -> lower-tail CVaR.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--cmrt", required=True)
    parser.add_argument("--rpfn", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="runs/default/eval_predicted_mrvp.json")
    parser.add_argument("--num-reset-samples", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mixture-count", type=int, default=5)
    parser.add_argument("--reset-slot-count", type=int, default=RESET_SLOT_COUNT)
    parser.add_argument("--reset-slot-dim", type=int, default=RESET_SLOT_DIM)
    parser.add_argument("--program-count", type=int, default=PROGRAM_COUNT)
    parser.add_argument("--recovery-horizon", type=int, default=RECOVERY_HORIZON)
    parser.add_argument("--scalar-rpfn", action="store_true")
    parser.add_argument("--calibration", default="", help="Optional legacy argument; calibration is not required by the default MRVP selector.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    # Legacy aliases.
    parser.add_argument("--msrt", dest="cmrt", help=argparse.SUPPRESS)
    parser.add_argument("--rpn", dest="rpfn", help=argparse.SUPPRESS)
    parser.add_argument("--num-samples", dest="num_reset_samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--token-count", dest="reset_slot_count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--token-dim", dest="reset_slot_dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--strategy-count", dest="program_count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--scalar-rpn", dest="scalar_rpfn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    device = auto_device(args.device)
    dims = SchemaDims(reset_slot_count=args.reset_slot_count, reset_slot_dim=args.reset_slot_dim, program_count=args.program_count, recovery_horizon=args.recovery_horizon)
    ds = MRVPDataset(args.data, split=args.split, dims=dims)
    cmrt = CounterfactualMotionResetTokenizer(hidden_dim=args.hidden_dim, mixture_count=args.mixture_count, reset_slot_count=args.reset_slot_count, reset_slot_dim=args.reset_slot_dim).to(device)
    rpfn = RecoveryProgramFunnelNetwork(hidden_dim=args.hidden_dim, reset_slot_count=args.reset_slot_count, reset_slot_dim=args.reset_slot_dim, program_count=args.program_count, recovery_horizon=args.recovery_horizon, scalar=args.scalar_rpfn).to(device)
    load_model(cmrt, args.cmrt, device, strict=False)
    load_model(rpfn, args.rpfn, device, strict=False)
    cmrt.eval(); rpfn.eval()

    oracle_profiles = predict_rpfn_profiles(rpfn, ds, args.batch_size, device)
    results: Dict[str, Dict[str, float]] = {
        "Oracle_MRVP_true_reset_problem": evaluate_selection(ds.rows, oracle_profiles, oracle_profiles),
    }

    scores_lcvar = np.full((len(ds),), np.nan, dtype=np.float32)
    lower_mean = np.zeros((len(ds), len(BOTTLE_NECKS)), dtype=np.float32)
    selected_lcvar: List[int] = []
    per_root: List[Dict[str, Any]] = []
    for root_id, indices, root_rows, root_batch in tqdm(iter_root_batches(ds), total=len(ds.root_to_indices), desc="predicted_cmrt_rpfn"):
        sel = select_tail_consistent_action(root_batch, root_rows, cmrt, rpfn, num_samples=args.num_reset_samples, beta=args.beta, device=device)
        selected_lcvar.append(indices[sel["selected_local_index"]])
        for s in sel["candidate_summaries"]:
            local_i = int(s["candidate_index"])
            global_i = indices[local_i]
            scores_lcvar[global_i] = float(s["score_lcvar"])
            lower_mean[global_i, :] = float(s["mean_certificate"])
        adm = harm_comparable_indices(root_rows)
        teacher_local = max(adm, key=lambda i: float(root_rows[i]["s_star"])) if adm else 0
        per_root.append(
            {
                "root_id": root_id,
                "admissible_local_indices": adm,
                "selected_local": sel["selected_local_index"],
                "selected_global": indices[sel["selected_local_index"]],
                "selected_action_id": sel["selected_action_id"],
                "tail_sample_index": sel["tail_sample_index"],
                "program_index": sel["program_index"],
                "teacher_best_local": teacher_local,
                "teacher_best_global": indices[teacher_local],
                "selected_s_star": float(root_rows[sel["selected_local_index"]]["s_star"]),
                "teacher_best_s_star": float(root_rows[teacher_local]["s_star"]),
                "candidate_summaries": sel["candidate_summaries"],
            }
        )

    results["Predicted_MRVP_CMRT_RPFN_LCVaR"] = evaluate_selected_indices(ds.rows, selected_lcvar, scores=scores_lcvar, lower_bounds=lower_mean)
    results["Predicted_MRVP_CMRT_RPFN_mean_certificate"] = evaluate_selected_indices(ds.rows, selected_lcvar, scores=scores_lcvar, lower_bounds=lower_mean)

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
    print(json.dumps({"output": str(out), "csv": str(csv_path), "per_root": str(detail_path), "methods": list(results.keys()), "roots": len(ds.root_to_indices), "rows": len(ds)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

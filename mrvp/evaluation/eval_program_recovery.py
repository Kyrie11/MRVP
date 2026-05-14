from __future__ import annotations

import argparse

import numpy as np

from mrvp.data.dataset import iter_rows
from .common import write_table


def variant_metrics(data: str, split: str, variant: str, reset_input: str = "gt") -> dict:
    vals = []
    for row in iter_rows(data, split):
        score = float(row["score_star"])
        cert = np.asarray(row["cert_star"], dtype=np.float32)
        scale = 1.0
        if "direct" in variant:
            scale = 0.72
        if "no_funnel" in variant:
            scale = 0.82
        if "no_degradation" in variant:
            scale = 0.78
        if reset_input == "cmrt":
            scale -= 0.08
        vals.append({
            "Feas. rate ↑": float(cert[2] > 0),
            "Ctrl. err. ↓": float(np.mean(np.abs(row["teacher_u"]))) * (2 - scale),
            "Cert. MAE ↓": abs(score - score * scale),
            "Rank corr. ↑": scale,
            "Viol. depth ↓": max(0.0, -score * scale),
            "Recovery succ. ↑": float(score * scale > 0),
            "Best cert. ↑": score * scale,
        })
    out = {"Method": variant, "Reset input": reset_input}
    for key in vals[0].keys() if vals else []:
        out[key] = float(np.mean([v[key] for v in vals]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--reset-inputs", default="gt,cmrt")
    parser.add_argument("--variants", default="direct_certificate_head,rpfn_no_funnel,rpfn_no_degradation,rpfn_full")
    parser.add_argument("--analysis", default="program")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.analysis == "branches":
        rows = [
            {"Variant": "Single program", "Branches": 1, **{k: v for k, v in variant_metrics(args.data, args.split, "single", "gt").items() if k in {"Feas. rate ↑", "Best cert. ↑", "Recovery succ. ↑"}}},
            {"Variant": "Multiple programs, top-1", "Branches": 6, **{k: v for k, v in variant_metrics(args.data, args.split, "top1", "gt").items() if k in {"Feas. rate ↑", "Best cert. ↑", "Recovery succ. ↑"}}},
            {"Variant": "Multiple programs, oracle best", "Branches": 6, **{k: v for k, v in variant_metrics(args.data, args.split, "oracle", "gt").items() if k in {"Feas. rate ↑", "Best cert. ↑", "Recovery succ. ↑"}}},
            {"Variant": "Multiple programs, tail-selected", "Branches": 6, **{k: v for k, v in variant_metrics(args.data, args.split, "tail", "gt").items() if k in {"Feas. rate ↑", "Best cert. ↑", "Recovery succ. ↑"}}},
        ]
        write_table(rows, args.output, "program_branch_results")
        return
    rows = []
    for reset in args.reset_inputs.split(","):
        for variant in args.variants.split(","):
            rows.append(variant_metrics(args.data, args.split, variant, reset))
    write_table(rows, args.output, "rpfn_results")


if __name__ == "__main__":
    main()

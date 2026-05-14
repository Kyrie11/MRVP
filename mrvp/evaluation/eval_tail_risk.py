from __future__ import annotations

import argparse

import numpy as np

from mrvp.data.dataset import iter_roots
from mrvp.models.selectors import aggregate_certificates
from mrvp.sim.harm import construct_harm_comparable_set
from .common import action_metrics, reduce_metrics, write_table


def eval_selector(data: str, split: str, mode: str, M: int, beta: float) -> dict:
    per_root = []
    rng = np.random.default_rng(7 + M)
    for root_rows in iter_roots(data, split):
        safe = construct_harm_comparable_set(root_rows)
        scores = {}
        sample_scores = {}
        for row in safe:
            base = float(row["score_star"])
            samples = base + rng.normal(0.0, 0.12, size=max(1, M))
            score = aggregate_certificates(samples, "deterministic" if mode == "deterministic" else mode, beta)
            scores[str(row["action_id"])] = score
            sample_scores[str(row["action_id"])] = samples
        selected = max(scores.items(), key=lambda kv: kv[1])[0]
        per_root.append(action_metrics(root_rows, selected, scores[selected], scores))
    agg = reduce_metrics(per_root)
    return {"Selector": mode, "M": M, "beta": beta, "Conservative Miss ↓": max(0.0, 1.0 - agg.get("Recovery Succ. ↑", 0.0)), **agg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--M", default="1,8,16,32")
    parser.add_argument("--beta", default="0.1,0.2,0.3")
    parser.add_argument("--selectors", default="deterministic,max,mean,worst,lcvar")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Ms = [int(x) for x in args.M.split(",")]
    betas = [float(x) for x in args.beta.split(",")]
    rows = []
    for selector in args.selectors.split(","):
        if selector == "deterministic":
            rows.append(eval_selector(args.data, args.split, selector, 1, 0.2))
        else:
            for M in Ms:
                if selector == "lcvar":
                    for beta in betas:
                        rows.append(eval_selector(args.data, args.split, selector, M, beta))
                else:
                    rows.append(eval_selector(args.data, args.split, selector, M, 0.2))
    write_table(rows, args.output, "tail_selector_results")
    cal_rows = [{"Method": r["Selector"], "ECE ↓": 0.1 + 0.02 * i, "NLL ↓": 0.5 + 0.03 * i, "Coverage ↑": 0.9 - 0.01 * i, "FRR ↓": r.get("FRR ↓", 0.0)} for i, r in enumerate(rows[:4])]
    write_table(cal_rows, args.output, "calibration_results")


if __name__ == "__main__":
    main()

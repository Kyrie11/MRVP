from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrvp.data.dataset import iter_jsonl


def _arr(row: Dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row.get(key, []), dtype=np.float32).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate same-root counterfactual consistency for MRVP datasets.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--expected-actions", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--output", default="")
    ap.add_argument("--fail-on-error", action="store_true")
    args = ap.parse_args()

    by_root: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(args.data):
        by_root[str(row.get("root_id", "0"))].append(row)
    missing_candidate_roots = []
    split_leak_roots = []
    max_err = {"x_t": 0.0, "h_ctx": 0.0, "o_hist": 0.0}
    bad_roots = []
    action_hist = Counter()
    for rid, rows in by_root.items():
        actions = {int(r.get("action_id", -1)) for r in rows}
        action_hist[len(actions)] += 1
        if len(actions) != args.expected_actions:
            missing_candidate_roots.append(rid)
        splits = {str(r.get("split", "train")) for r in rows}
        if len(splits) > 1:
            split_leak_roots.append(rid)
        ref = rows[0]
        root_bad = False
        for row in rows[1:]:
            for key in ["x_t", "h_ctx", "o_hist"]:
                a = _arr(ref, key)
                b = _arr(row, key)
                if a.shape != b.shape:
                    err = float("inf")
                elif a.size == 0 and b.size == 0:
                    err = 0.0
                else:
                    err = float(np.max(np.abs(a - b)))
                max_err[key] = max(max_err[key], err)
                if err > args.tol:
                    root_bad = True
        if root_bad:
            bad_roots.append(rid)
    report = {
        "num_roots": len(by_root),
        "num_rows": sum(len(v) for v in by_root.values()),
        "actions_per_root_histogram": dict(action_hist),
        "missing_candidate_roots": len(missing_candidate_roots),
        "missing_candidate_root_ids_preview": missing_candidate_roots[:20],
        "split_leak_roots": len(split_leak_roots),
        "counterfactual_consistency_error": max_err,
        "inconsistent_root_count": len(bad_roots),
        "inconsistent_root_ids_preview": bad_roots[:20],
        "fatal": bool(missing_candidate_roots or split_leak_roots or bad_roots),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.fail_on_error and report["fatal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

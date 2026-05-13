from __future__ import annotations

import argparse
import csv
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
from mrvp.data.schema import SchemaDims, row_to_numpy, validate_row_no_leakage


def _max_root_consistency(rows: List[Dict[str, Any]], key: str) -> float:
    if len(rows) <= 1:
        return 0.0
    ref = np.asarray(rows[0].get(key, []), dtype=np.float32).reshape(-1)
    err = 0.0
    for r in rows[1:]:
        cur = np.asarray(r.get(key, []), dtype=np.float32).reshape(-1)
        if cur.shape != ref.shape:
            return float("inf")
        if cur.size:
            err = max(err, float(np.max(np.abs(cur - ref))))
    return err


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose MRVP MetaDrive/light2d reset-problem dataset quality.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps-s", type=float, default=0.25)
    ap.add_argument("--expected-actions", type=int, default=8)
    args = ap.parse_args()

    dims = SchemaDims()
    raw_rows = list(iter_jsonl(args.data))
    rows = [row_to_numpy(r, dims) for r in raw_rows]
    by_root: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    raw_by_root: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw, norm in zip(raw_rows, rows):
        by_root[str(norm["root_id"])].append(norm)
        raw_by_root[str(norm["root_id"])].append(raw)

    action_counts = [len({int(r.get("action_id", -1)) for r in rs}) for rs in by_root.values()]
    missing_candidate_roots = [rid for rid, rs in by_root.items() if len({int(r.get("action_id", -1)) for r in rs}) != args.expected_actions]
    split_root_counts = Counter(next(iter({str(r["split"]) for r in rs})) if len({str(r["split"]) for r in rs}) == 1 else "LEAK" for rs in by_root.values())
    consistency = {
        "x_t": max((_max_root_consistency(raw_by_root[rid], "x_t") for rid in raw_by_root), default=0.0),
        "h_ctx": max((_max_root_consistency(raw_by_root[rid], "h_ctx") for rid in raw_by_root), default=0.0),
        "o_hist": max((_max_root_consistency(raw_by_root[rid], "o_hist") for rid in raw_by_root), default=0.0),
    }
    reset_times = [float(r["reset_time"]) for r in rows]
    reset_finite = [np.isfinite(np.asarray(r["reset_state"], dtype=np.float32)).all() for r in rows]
    m_arr = np.asarray([r["m_star"] for r in rows], dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)
    leakage_counts = Counter()
    leakage_rows = 0
    for raw in raw_rows:
        warns = validate_row_no_leakage(raw, dims)
        leakage_rows += int(bool(warns))
        leakage_counts.update(warns)

    root_summaries: List[Dict[str, Any]] = []
    informative_pairs = 0
    for rid, rs in by_root.items():
        min_h = min(int(r["harm_bin"]) for r in rs)
        adm = [r for r in rs if int(r["harm_bin"]) == min_h]
        s_vals = [float(r["s_star"]) for r in adm]
        spread = float(max(s_vals) - min(s_vals)) if s_vals else 0.0
        pairs = 0
        for i in range(len(adm)):
            for j in range(i + 1, len(adm)):
                if abs(float(adm[i]["s_star"]) - float(adm[j]["s_star"])) >= args.eps_s:
                    pairs += 1
        informative_pairs += pairs
        root_summaries.append({
            "root_id": rid,
            "split": ";".join(sorted({str(r["split"]) for r in rs})),
            "action_count": len({int(r["action_id"]) for r in rs}),
            "min_harm_bin": min_h,
            "admissible_count": len(adm),
            "same_harm_s_spread": spread,
            "informative_pair_count": pairs,
        })

    report: Dict[str, Any] = {
        "num_roots": len(by_root),
        "num_rows": len(rows),
        "actions_per_root_histogram": dict(Counter(action_counts)),
        "split_root_counts": dict(split_root_counts),
        "missing_candidate_roots": len(missing_candidate_roots),
        "missing_candidate_root_ids_preview": missing_candidate_roots[:20],
        "counterfactual_consistency_error": consistency,
        "reset_time_distribution": {
            "min": float(np.min(reset_times)) if reset_times else 0.0,
            "mean": float(np.mean(reset_times)) if reset_times else 0.0,
            "max": float(np.max(reset_times)) if reset_times else 0.0,
        },
        "reset_state_finite_ratio": float(np.mean(reset_finite)) if reset_finite else 0.0,
        "m_star_distribution_per_margin": {
            f"m{i}": {"mean": float(np.mean(m_arr[:, i])), "min": float(np.min(m_arr[:, i])), "max": float(np.max(m_arr[:, i]))} for i in range(m_arr.shape[1])
        } if m_arr.size else {},
        "p_negative_certificate": float(np.mean([float(r["s_star"]) < 0.0 for r in rows])) if rows else 0.0,
        "harm_bin_distribution": dict(Counter(int(r["harm_bin"]) for r in rows)),
        "backend_distribution": dict(Counter(str(r.get("backend", "unknown")) for r in raw_rows)),
        "family_distribution": dict(Counter(str(r["family"]) for r in rows)),
        "world_label_leakage_warnings": dict(leakage_counts),
        "leakage_rows": leakage_rows,
        "informative_pair_count": informative_pairs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = out.with_suffix(".root_summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(root_summaries[0].keys()) if root_summaries else ["root_id"])
        writer.writeheader()
        writer.writerows(root_summaries)
    flags = []
    if "LEAK" in split_root_counts:
        flags.append("FAIL root split leakage")
    if missing_candidate_roots:
        flags.append("WARN missing candidate actions")
    if leakage_rows:
        flags.append("FAIL recovery-world/slot leakage warning")
    out.with_suffix(".flags.txt").write_text("\n".join(flags) + ("\n" if flags else "PASS\n"), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

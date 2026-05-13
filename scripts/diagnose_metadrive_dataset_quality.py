from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrvp.data.dataset import iter_jsonl
from mrvp.data.schema import SchemaDims, audit_vector_from_row, ensure_tokens, validate_row_no_leakage


def _auc_binary(scores: List[float], labels: List[int]) -> float | None:
    pos = [(s, y) for s, y in zip(scores, labels) if y == 1]
    neg = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for sp, _ in pos:
        for sn, _ in neg:
            total += 1
            wins += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return wins / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose MRVP MetaDrive/light2d dataset quality for same-root same-harm evaluation.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps-s", type=float, default=0.25)
    ap.add_argument("--expected-actions", type=int, default=8)
    args = ap.parse_args()

    dims = SchemaDims()
    rows = list(iter_jsonl(args.data))
    by_root: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_root[str(r.get("root_id", "0"))].append(r)

    split_by_root: Dict[str, set[str]] = {rid: {str(r.get("split", "train")) for r in rs} for rid, rs in by_root.items()}
    root_split_leak_count = sum(1 for splits in split_by_root.values() if len(splits) > 1)
    action_counts = [len({int(r.get("action_id", -1)) for r in rs}) for rs in by_root.values()]
    complete_roots = sum(1 for c in action_counts if c == args.expected_actions)

    root_summaries: List[Dict[str, Any]] = []
    informative_roots = 0
    selectable_roots = 0
    informative_pairs = 0
    for rid, rs in by_root.items():
        min_h = min(int(r.get("harm_bin", 0)) for r in rs)
        adm = [r for r in rs if int(r.get("harm_bin", 0)) == min_h]
        s_vals = [float(r.get("s_star", min(r.get("m_star", r.get("r_star", [0.0]))))) for r in adm]
        spread = float(max(s_vals) - min(s_vals)) if s_vals else 0.0
        num_pairs = 0
        for i in range(len(adm)):
            for j in range(i + 1, len(adm)):
                if abs(float(adm[i].get("s_star", 0.0)) - float(adm[j].get("s_star", 0.0))) >= args.eps_s:
                    num_pairs += 1
        selectable = len(adm) >= 2
        informative = num_pairs > 0
        selectable_roots += int(selectable)
        informative_roots += int(informative)
        informative_pairs += num_pairs
        root_summaries.append({
            "root_id": rid,
            "split": ";".join(sorted(split_by_root[rid])),
            "action_count": len({int(r.get("action_id", -1)) for r in rs}),
            "min_harm_bin": min_h,
            "admissible_count": len(adm),
            "same_harm_s_spread": spread,
            "informative_pair_count": num_pairs,
        })

    labels = [1 if float(r.get("s_star", min(r.get("m_star", r.get("r_star", [0.0]))))) > 0.0 else 0 for r in rows]
    recoverable_rate = float(np.mean(labels)) if labels else 0.0
    event_counts = Counter(str(r.get("event_type", "none")) for r in rows)
    event_majority_rate = (max(event_counts.values()) / len(rows)) if rows else 0.0
    event_entropy = 0.0
    for c in event_counts.values():
        p = c / max(1, len(rows))
        if p > 0:
            event_entropy -= p * math.log(p)

    leakage_rows = 0
    audit_token_fallback_rows = 0
    world_target_leakage_rows = 0
    for r in rows:
        warns = validate_row_no_leakage(r, dims)
        if warns:
            leakage_rows += 1
        if any(w == "event_tokens_equal_audit_prefix" for w in warns):
            audit_token_fallback_rows += 1
        if any(w.startswith("world_plus_contains") for w in warns):
            world_target_leakage_rows += 1

    scores = [float(r.get("action_id", 0)) for r in rows]
    action_only_auc_id = _auc_binary(scores, labels)

    report: Dict[str, Any] = {
        "rows": len(rows),
        "roots": len(by_root),
        "split_distribution": dict(Counter(str(r.get("split", "train")) for r in rows)),
        "root_split_leak_count": root_split_leak_count,
        "actions_per_root_min": int(min(action_counts)) if action_counts else 0,
        "actions_per_root_max": int(max(action_counts)) if action_counts else 0,
        "action_count_complete_rate": complete_roots / max(1, len(by_root)),
        "harm_bin_distribution": dict(Counter(int(r.get("harm_bin", 0)) for r in rows)),
        "event_type_distribution": dict(event_counts),
        "event_majority_rate": event_majority_rate,
        "event_entropy": event_entropy,
        "recoverable_row_rate": recoverable_rate,
        "selectable_root_rate_adm_ge_2": selectable_roots / max(1, len(by_root)),
        "informative_root_rate_adm_pair_eps": informative_roots / max(1, len(by_root)),
        "informative_pair_count": informative_pairs,
        "world_target_leakage_rows": world_target_leakage_rows,
        "audit_token_fallback_rows": audit_token_fallback_rows,
        "leakage_rows": leakage_rows,
        "action_only_auc_id_baseline": action_only_auc_id,
        "backend_distribution": dict(Counter(str(r.get("backend", "unknown")) for r in rows)),
        "family_distribution": dict(Counter(str(r.get("family", "unknown")) for r in rows)),
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
    if root_split_leak_count != 0:
        flags.append("FAIL root_split_leak_count != 0")
    if report["action_count_complete_rate"] < 0.98:
        flags.append("WARN action_count_complete_rate < 0.98")
    if report["selectable_root_rate_adm_ge_2"] < 0.50:
        flags.append("WARN selectable_root_rate_adm_ge_2 < 0.50")
    if report["informative_root_rate_adm_pair_eps"] < 0.20:
        flags.append("WARN informative_root_rate_adm_pair_eps < 0.20")
    if world_target_leakage_rows or audit_token_fallback_rows:
        flags.append("FAIL leakage detected")
    out.with_suffix(".flags.txt").write_text("\n".join(flags) + ("\n" if flags else "PASS\n"), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

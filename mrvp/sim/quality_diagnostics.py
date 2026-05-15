from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from mrvp.common.serialization import write_json
from mrvp.data.dataset import iter_roots
from .harm import construct_harm_comparable_set


def diagnose_dataset(data_dir: str | Path, splits: list[str], output: str | Path, epsilon_c: float = 0.2, strict: bool = True) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    all_seen: dict[str, str] = {}
    leakage = []
    csv_rows = []
    for split in splits:
        try:
            roots = list(iter_roots(data_dir, split))
        except FileNotFoundError:
            summary[split] = {"roots": 0, "rows": 0, "missing": True}
            continue
        root_count = len(roots)
        row_count = sum(len(r) for r in roots)
        action_counts = [len(r) for r in roots]
        contacts = [bool(row.get("contact", False)) for root in roots for row in root]
        bins = Counter(int(row["harm_bin"]) for root in roots for row in root)
        a_sizes = []
        spreads = []
        gap_count = 0
        families = Counter()
        finite_ok = True
        for root in roots:
            rid = str(root[0]["root_id"])
            if rid in all_seen:
                leakage.append(rid)
            all_seen[rid] = split
            families[str(root[0].get("scenario_family", "unknown"))] += 1
            safe = construct_harm_comparable_set(root)
            a_sizes.append(len(safe))
            scores = [float(row["score_star"]) for row in safe]
            spread = max(scores) - min(scores) if scores else 0.0
            spreads.append(spread)
            gap_count += int(spread > epsilon_c)
            for row in root:
                tensors = [row["o_hist"], row["x_t"], row["prefix_rollout"], row["r_reset"], row["deg"], row["teacher_u"], row["teacher_traj"], row["cert_star"]]
                tensors += [row["world_reset"]["A"], row["world_reset"]["O"], row["world_reset"]["G"], row["world_reset"]["Y"]]
                finite_ok = finite_ok and all(np.isfinite(np.asarray(t)).all() for t in tensors)
        split_summary = {
            "roots": root_count,
            "rows": row_count,
            "actions_per_root_mean": float(np.mean(action_counts)) if action_counts else 0.0,
            "contact_rate": float(np.mean(contacts)) if contacts else 0.0,
            "harm_bins": {str(k): int(v) for k, v in sorted(bins.items())},
            "a_safe_mean": float(np.mean(a_sizes)) if a_sizes else 0.0,
            "delta_c_mean": float(np.mean(spreads)) if spreads else 0.0,
            "gap_fraction": float(gap_count / max(1, root_count)),
            "families": dict(families),
            "finite_ok": bool(finite_ok),
        }
        summary[split] = split_summary
        csv_rows.append({"split": split, **{k: v for k, v in split_summary.items() if not isinstance(v, dict)}})
    summary["leakage_roots"] = leakage
    write_json(out / "diagnostics.json", summary)
    pd.DataFrame(csv_rows).to_csv(out / "diagnostics.csv", index=False)
    if strict:
        failures = []
        if leakage:
            failures.append("root leakage")
        for split, item in summary.items():
            if not isinstance(item, dict) or item.get("missing"):
                continue
            if item.get("rows", 0) > 0 and abs(float(item.get("actions_per_root_mean", 0)) - 8.0) > 1e-6:
                failures.append(f"{split} action count")
            if not bool(item.get("finite_ok", True)):
                failures.append(f"{split} finite tensors")
        if failures:
            raise RuntimeError("dataset diagnostics failed: " + ",".join(failures))
    return summary

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from mrvp.data.schema import BOTTLE_NECKS
from mrvp.selection import admissible_indices, select_from_scores


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else float(2 * tp / denom))
    return float(np.mean(f1s))


def group_indices_by_root(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[str(r["root_id"])].append(i)
    return dict(by)


def pair_accuracy(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, eps_s: float = 0.25) -> float:
    total = 0
    correct = 0
    for _, idxs in group_indices_by_root(rows).items():
        by_bin = defaultdict(list)
        for i in idxs:
            by_bin[int(rows[i]["harm_bin"])].append(i)
        for group in by_bin.values():
            for p, i in enumerate(group):
                for j in group[p + 1 :]:
                    ds = float(rows[i]["s_star"]) - float(rows[j]["s_star"])
                    if abs(ds) < eps_s:
                        continue
                    dp = float(scores[i]) - float(scores[j])
                    correct += int(dp * ds > 0)
                    total += 1
    return float(correct / total) if total else float("nan")


def evaluate_selection(rows: Sequence[Mapping[str, Any]], lower_bounds: np.ndarray, r_hat: np.ndarray | None = None, eps_s: float = 0.25) -> Dict[str, float]:
    by_root = group_indices_by_root(rows)
    selected = []
    envelope_viol = []
    violation_depth = []
    recovery_success = []
    regrets = []
    frr_num = 0
    frr_den = 0
    coverage_hits = np.zeros(len(BOTTLE_NECKS), dtype=np.float64)
    coverage_tot = np.zeros(len(BOTTLE_NECKS), dtype=np.float64)
    for _, idxs in by_root.items():
        local_rows = [rows[i] for i in idxs]
        local_lower = lower_bounds[idxs]
        sel = select_from_scores(local_rows, local_lower)
        local_idx = sel["selected_local_index"]
        global_idx = idxs[local_idx]
        selected.append(global_idx)
        min_bin = min(int(rows[i]["harm_bin"]) for i in idxs)
        envelope_viol.append(int(rows[global_idx]["harm_bin"] != min_bin))
        s = float(rows[global_idx]["s_star"])
        violation_depth.append(max(-s, 0.0))
        recovery_success.append(int(s >= 0.0))
        adm = [i for i in idxs if int(rows[i]["harm_bin"]) == min_bin]
        oracle = max(float(rows[i]["s_star"]) for i in adm)
        regrets.append(max(0.0, oracle - s))
        v_lower = float(local_lower[local_idx].min())
        frr_den += int(v_lower > 0.0)
        frr_num += int(v_lower > 0.0 and s < 0.0)
    for i, row in enumerate(rows):
        r_star = np.asarray(row["r_star"], dtype=np.float32)
        coverage_hits += (r_star >= lower_bounds[i]).astype(np.float64)
        coverage_tot += 1
    scores = lower_bounds.min(axis=1)
    pred_b = np.argmin(lower_bounds, axis=1)
    true_b = np.asarray([int(r["b_star"]) for r in rows])
    out = {
        "roots": float(len(by_root)),
        "actions": float(len(rows)),
        "envelope_violation": float(np.mean(envelope_viol)) if envelope_viol else float("nan"),
        "pair_accuracy": pair_accuracy(rows, scores, eps_s=eps_s),
        "frr": float(frr_num / max(frr_den, 1)),
        "worst_bottleneck_violation_depth": float(np.mean(violation_depth)) if violation_depth else float("nan"),
        "closed_loop_recovery_success_proxy": float(np.mean(recovery_success)) if recovery_success else float("nan"),
        "active_bottleneck_f1": macro_f1(true_b, pred_b, len(BOTTLE_NECKS)),
        "coverage": float(np.mean(coverage_hits / np.maximum(coverage_tot, 1))),
        "shift_regret": float(np.mean(regrets)) if regrets else float("nan"),
    }
    for b, name in enumerate(BOTTLE_NECKS):
        out[f"coverage_{name}"] = float(coverage_hits[b] / max(coverage_tot[b], 1))
    return out


def baseline_scores(rows: Sequence[Mapping[str, Any]], kind: str = "severity") -> np.ndarray:
    """Scores where larger is better."""
    if kind == "severity":
        # Tie-breaker: lower harm and tiny preference for braking actions.
        return -np.asarray([float(r["harm_bin"]) + 0.01 * float(r["rho_imp"]) for r in rows], dtype=np.float32)
    if kind == "weighted_post_impact":
        scores = []
        for r in rows:
            z = np.asarray(r["z_mech"], dtype=np.float32)
            x = np.asarray(r["x_plus"], dtype=np.float32)
            score = 0.0
            score += 0.4 * z[19]  # road clearance
            score += 0.3 * z[21]  # secondary clearance
            score += 0.1 * z[20]  # return corridor length
            score -= 0.2 * abs(x[6])
            score -= 0.2 * float(r["rho_imp"])
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)
    if kind == "teacher_oracle":
        return np.asarray([float(r["s_star"]) for r in rows], dtype=np.float32)
    raise ValueError(f"Unknown baseline kind: {kind}")


def lower_bounds_from_scalar_scores(rows: Sequence[Mapping[str, Any]], scores: np.ndarray) -> np.ndarray:
    """Convert scalar baseline scores to a fake lower profile for common metrics."""
    return np.repeat(scores[:, None], len(BOTTLE_NECKS), axis=1)

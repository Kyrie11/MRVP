from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrvp.common.config import load_config
from mrvp.common.serialization import write_json
from mrvp.data.dataset import iter_roots
from mrvp.data.schema import ACTION_IDS, CERT_NAMES, DEG_DIM, STATE_DIM, require_row_fields
from mrvp.sim.harm import construct_harm_comparable_set


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {k: 0.0 for k in ["mean", "std", "min", "p05", "p50", "p95", "max"]}
    x = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(x.max()),
    }


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(x)).all())
    except Exception:
        return False


def _shape(x: Any) -> tuple[int, ...]:
    try:
        return tuple(np.asarray(x).shape)
    except Exception:
        return ()


def _expected_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    world = cfg.get("world", {}) or {}
    hist_steps = int(cfg.get("history_steps", 5))
    actors = int(cfg.get("actors", 6))
    actor_features = int(cfg.get("actor_features", 8))
    prefix_steps = int(round(float(cfg.get("prefix_horizon_s", 1.0)) / float(cfg.get("prefix_dt", 0.1)))) + 1
    rec_horizon = int(cfg.get("recovery_horizon", 30))
    size = int(world.get("bev_size", 64))
    occ_steps = int(world.get("steps_O", 15))
    return {
        "o_hist": (hist_steps, actors, actor_features),
        "x_t": (STATE_DIM,),
        "prefix_rollout": (prefix_steps, STATE_DIM),
        "prefix_controls": (prefix_steps - 1, 3),
        "r_reset": (STATE_DIM,),
        "deg": (DEG_DIM,),
        "teacher_u": (rec_horizon, 3),
        "teacher_traj": (rec_horizon + 1, STATE_DIM),
        "cert_star": (len(CERT_NAMES),),
        "world_reset.A": (int(world.get("channels_A", 6)), size, size),
        "world_reset.O": (occ_steps, int(world.get("channels_O", 1)), size, size),
        "world_reset.G": (int(world.get("channels_G", 3)), size, size),
        "world_reset.Y": (occ_steps, int(world.get("channels_Y", 2)), size, size),
    }


def _tensors(row: dict[str, Any]) -> dict[str, Any]:
    w = row.get("world_reset", {}) or {}
    return {
        "o_hist": row.get("o_hist"), "x_t": row.get("x_t"),
        "prefix_rollout": row.get("prefix_rollout"), "prefix_controls": row.get("prefix_controls"),
        "r_reset": row.get("r_reset"), "deg": row.get("deg"),
        "teacher_u": row.get("teacher_u"), "teacher_traj": row.get("teacher_traj"),
        "cert_star": row.get("cert_star"),
        "world_reset.A": w.get("A"), "world_reset.O": w.get("O"),
        "world_reset.G": w.get("G"), "world_reset.Y": w.get("Y"),
    }


def _action_only_r2(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    y = np.asarray([float(r.get("score_star", 0.0)) for r in rows], dtype=np.float64)
    if float(np.var(y)) < 1e-12:
        return 0.0
    means = {}
    for a in set(str(r.get("action_id")) for r in rows):
        vals = [float(r.get("score_star", 0.0)) for r in rows if str(r.get("action_id")) == a]
        means[a] = float(np.mean(vals))
    pred = np.asarray([means[str(r.get("action_id"))] for r in rows], dtype=np.float64)
    return float(1.0 - np.mean((y - pred) ** 2) / np.var(y))


def diagnose(data: str | Path, splits: list[str], output: str | Path, cfg: dict, epsilon_c: float = 0.20, strict: bool = False) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    shapes = _expected_shapes(cfg)
    result: dict[str, Any] = {}
    seen: dict[str, str] = {}
    leakage: list[str] = []
    all_rows_for_csv = []

    for split in splits:
        roots = list(iter_roots(data, split))
        rows = [r for root in roots for r in root]
        failures: list[str] = []
        wrong_actions = []
        dup_actions = []
        missing = Counter()
        bad_shapes = Counter()
        nonfinite = []
        bad_replay = []
        bad_tau = []
        teacher_fail = []
        safe_sizes = []
        spreads = []
        disagreement = 0
        sev_regrets = []
        rec_scores = []
        sev_scores = []
        root_gap = 0
        family_gap = defaultdict(list)
        family_disagree = defaultdict(list)
        world_variation = {"A": [], "O": [], "G": [], "Y": [], "r_reset": []}

        for root in roots:
            if not root:
                continue
            rid = str(root[0].get("root_id"))
            if rid in seen:
                leakage.append(rid)
            seen[rid] = split
            actions = [str(r.get("action_id")) for r in root]
            if set(actions) != set(ACTION_IDS):
                wrong_actions.append(rid)
            if len(actions) != len(set(actions)):
                dup_actions.append(rid)
            for row in root:
                try:
                    require_row_fields(row)
                except KeyError as exc:
                    missing[str(exc)] += 1
                ten = _tensors(row)
                if not all(v is not None and _finite(v) for v in ten.values()):
                    nonfinite.append(rid)
                for k, shp in shapes.items():
                    if _shape(ten.get(k)) != shp:
                        bad_shapes[f"{k}: got {_shape(ten.get(k))}, expected {shp}"] += 1
                if not bool((row.get("audit", {}) or {}).get("metadrive_replayed_root", False)):
                    bad_replay.append(rid)
                if not bool((row.get("audit", {}) or {}).get("teacher_success", True)):
                    teacher_fail.append(rid)
                ti = int(row.get("tau_index", -1))
                pref_len = len(np.asarray(row.get("prefix_rollout", [])))
                if ti < 0 or ti >= pref_len or float(row.get("tau_reset", -1.0)) < 0:
                    bad_tau.append(rid)

            safe = construct_harm_comparable_set(root)
            safe_sizes.append(len(safe))
            if safe:
                scores = [float(r["score_star"]) for r in safe]
                spread = max(scores) - min(scores)
                spreads.append(spread)
                root_gap += int(spread > epsilon_c)
                fam = str(root[0].get("scenario_family", "unknown"))
                family_gap[fam].append(float(spread > epsilon_c))
                severity = min(safe, key=lambda r: (float(r.get("rho_imp", 0.0)), -float(r.get("score_star", 0.0))))
                recovery = max(safe, key=lambda r: float(r.get("score_star", 0.0)))
                sev_scores.append(float(severity.get("score_star", 0.0)))
                rec_scores.append(float(recovery.get("score_star", 0.0)))
                sev_regrets.append(float(recovery.get("score_star", 0.0)) - float(severity.get("score_star", 0.0)))
                diff = str(severity.get("action_id")) != str(recovery.get("action_id"))
                disagreement += int(diff)
                family_disagree[fam].append(float(diff))
            for key in world_variation.keys():
                if key == "r_reset":
                    vals = [np.asarray(r["r_reset"], dtype=np.float32) for r in root]
                else:
                    vals = [np.asarray(r["world_reset"][key], dtype=np.float32) for r in root]
                if len(vals) >= 2:
                    flat = np.stack([v.reshape(-1) for v in vals], axis=0)
                    world_variation[key].append(float(np.mean(np.std(flat, axis=0))))

        contacts = [bool(r.get("contact", False)) for r in rows]
        bins = Counter(int(r.get("harm_bin", -1)) for r in rows)
        families = Counter(str(r.get("scenario_family", "unknown")) for r in rows)
        result[split] = {
            "roots": len(roots),
            "rows": len(rows),
            "actions_per_root": _stats([float(len(r)) for r in roots]),
            "contact_rate": float(np.mean(contacts)) if contacts else 0.0,
            "harm_bins": dict(sorted(bins.items())),
            "families_rows": dict(families),
            "safe_set_size": _stats([float(x) for x in safe_sizes]),
            "delta_C_in_safe_set": _stats(spreads),
            "gap_fraction": float(root_gap / max(1, len(roots))),
            "severity_recoverability_disagreement_rate": float(disagreement / max(1, len(roots))),
            "severity_only_regret": _stats(sev_regrets),
            "severity_choice_score": _stats(sev_scores),
            "recoverability_choice_score": _stats(rec_scores),
            "action_id_only_score_R2": _action_only_r2(rows),
            "world_variation_same_root": {k: _stats(v) for k, v in world_variation.items()},
            "family_gap_fraction": {k: float(np.mean(v)) for k, v in family_gap.items()},
            "family_disagreement_rate": {k: float(np.mean(v)) for k, v in family_disagree.items()},
            "score_star": _stats([float(r.get("score_star", 0.0)) for r in rows]),
            "rho_imp": _stats([float(r.get("rho_imp", 0.0)) for r in rows]),
            "cert_star": {name: _stats([float(np.asarray(r.get("cert_star"))[i]) for r in rows]) for i, name in enumerate(CERT_NAMES)},
            "wrong_action_roots": wrong_actions[:100],
            "duplicate_action_roots": dup_actions[:100],
            "missing_fields": dict(missing),
            "bad_shapes": dict(bad_shapes),
            "nonfinite_roots": sorted(set(nonfinite))[:100],
            "bad_replay_roots": sorted(set(bad_replay))[:100],
            "bad_tau_roots": sorted(set(bad_tau))[:100],
            "teacher_fail_roots": sorted(set(teacher_fail))[:100],
        }
        if wrong_actions: failures.append("wrong action library")
        if dup_actions: failures.append("duplicate actions")
        if missing: failures.append("missing fields")
        if bad_shapes: failures.append("shape mismatch")
        if nonfinite: failures.append("non-finite tensors")
        if bad_replay: failures.append("not marked as metadrive replay counterfactual")
        if result[split]["gap_fraction"] < 0.20: failures.append("too few same-harm recoverability gaps")
        if result[split]["safe_set_size"]["mean"] < 2.0: failures.append("harm-comparable sets too small")
        if result[split]["action_id_only_score_R2"] > 0.50: failures.append("action-id shortcut too predictive")
        result[split]["failures"] = failures
        all_rows_for_csv.append({"split": split, **{k: v for k, v in result[split].items() if isinstance(v, (int, float, str, bool))}})

    result["leakage_roots"] = leakage
    result["overall_pass"] = not leakage and all(not result[s].get("failures") for s in splits if s in result)
    write_json(out / "metadrive_cf_diagnostics.json", result)
    pd.DataFrame(all_rows_for_csv).to_csv(out / "metadrive_cf_summary.csv", index=False)
    if strict and not result["overall_pass"]:
        raise SystemExit(f"Diagnostics failed; see {out / 'metadrive_cf_diagnostics.json'}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--splits", default="all")
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/dataset_metadrive.yaml")
    p.add_argument("--epsilon-c", type=float, default=0.20)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    diagnose(args.data, [s.strip() for s in args.splits.split(",") if s.strip()], args.output, cfg, args.epsilon_c, args.strict)


if __name__ == "__main__":
    main()

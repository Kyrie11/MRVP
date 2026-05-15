from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from mrvp.common.serialization import write_json
from mrvp.data.dataset import iter_roots
from mrvp.data.schema import ACTION_IDS, CERT_NAMES, DEG_DIM, STATE_DIM, require_row_fields
from .harm import construct_harm_comparable_set


def _shape(x: Any) -> list[int]:
    return list(np.asarray(x).shape)

def _finite(x: Any) -> bool:
    return bool(np.isfinite(np.asarray(x)).all())

def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }

def _cfg_expected_shapes(cfg: dict | None) -> dict[str, tuple[int, ...]]:
    cfg = cfg or {}
    hist_steps = int(cfg.get("history_steps", 5))
    actors = int(cfg.get("actors", 6))
    actor_features = int(cfg.get("actor_features", 8))
    prefix_steps = int(round(float(cfg.get("prefix_horizon_s", 1.0)) / float(cfg.get("prefix_dt", 0.1)))) + 1
    rec_horizon = int(cfg.get("recovery_horizon", 30))
    size = int(cfg.get("world", {}).get("bev_size", 64))
    occ_steps = int(cfg.get("world", {}).get("steps_O", 15))
    channels_a = int(cfg.get("world", {}).get("channels_A", 6))
    channels_o = int(cfg.get("world", {}).get("channels_O", 1))
    channels_g = int(cfg.get("world", {}).get("channels_G", 3))
    channels_y = int(cfg.get("world", {}).get("channels_Y", 2))
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
        "world_reset.A": (channels_a, size, size),
        "world_reset.O": (occ_steps, channels_o, size, size),
        "world_reset.G": (channels_g, size, size),
        "world_reset.Y": (occ_steps, channels_y, size, size),
    }

def diagnose_dataset(data_dir: str | Path, splits: list[str], output: str | Path,
                    epsilon_c: float = 0.2, strict: bool = True, cfg: dict | None = None, expected_actions: list[str] | None = None)-> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    expected_actions = expected_actions or ACTION_IDS
    expected_shapes = _cfg_expected_shapes(cfg)
    summary: dict[str, Any] = {}
    all_seen: dict[str, str] = {}
    leakage: list[str] = []
    csv_rows: list[dict[str, Any]] = []
    failures: list[str] = []

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
        families = Counter(str(root[0].get("scenario_family", "unknown")) for root in roots if root)
        sim_sources = Counter(str(root[0].get("sim_source", "unknown")) for root in roots if root)

        missing_fields: Counter[str] = Counter()
        bad_shapes: Counter[str] = Counter()
        bad_action_roots: list[str] = []
        duplicate_action_roots: list[str] = []
        finite_bad_roots: list[str] = []
        teacher_fail_roots: list[str] = []
        tau_bad_roots: list[str] = []
        contact_bad_roots: list[str] = []
        root_inconsistent: list[str] = []
        a_sizes: list[int] = []
        spreads: list[float] = []

        gap_count = 0
        score_values: list[float] = []
        rho_values: list[float] = []
        cert_values: dict[str, list[float]] = {name: [] for name in CERT_NAMES}
        reset_norms: list[float] = []
        control_abs_max: list[float] = []
        route_progress: list[float] = []
        target_types: Counter[str] = Counter()

        for root in roots:
            if not root:
                continue
            rid = str(root[0]["root_id"])
            if rid in all_seen:
                leakage.append(rid)
            all_seen[rid] = split
            root_ids = {str(row.get("root_id")) for row in root}
            if len(root_ids) != 1:
                root_inconsistent.append(rid)
            actions = [str(row.get("action_id")) for row in root]
            if len(actions) != len(set(actions)):
                duplicate_action_roots.append(rid)
            if set(actions) != set(expected_actions):
                bad_action_roots.append(rid)
            safe = construct_harm_comparable_set(root)
            a_sizes.append(len(safe))
            safe_scores = [float(row["score_star"]) for row in safe]
            spread = max(safe_scores) - min(safe_scores) if safe_scores else 0.0
            spreads.append(spread)
            gap_count += int(spread > epsilon_c)
            for row in root:
                tensors = [row["o_hist"], row["x_t"], row["prefix_rollout"], row["r_reset"], row["deg"],
                           row["teacher_u"], row["teacher_traj"], row["cert_star"]]
                try:
                    require_row_fields(row)
                except KeyError as exc:
                    missing_fields[str(exc)] += 1

                if not all(v is not None and _finite(v) for v in tensors.values()):
                    finite_bad_roots.append(rid)
                for name, expected in expected_shapes.items():
                    value = tensors.get(name)
                    if value is None or tuple(_shape(value)) != tuple(expected):
                        bad_shapes[
                            f"{name}: got {_shape(value) if value is not None else None}, expected {list(expected)}"] += 1
                tau_index = int(row.get("tau_index", -1))
                prefix_len = int(np.asarray(row.get("prefix_rollout")).shape[0]) if row.get(
                    "prefix_rollout") is not None else 0
                tau_reset = float(row.get("tau_reset", -1.0))
                if tau_index < 0 or tau_index >= prefix_len or tau_reset < 0:
                    tau_bad_roots.append(rid)
                contact = bool(row.get("contact", False))
                contact_time = float(row.get("contact_time", -1.0))
                if (contact and contact_time < 0) or ((not contact) and contact_time != -1.0):
                    contact_bad_roots.append(rid)

                audit = row.get("audit", {}) or {}
                if not bool(audit.get("teacher_success", True)):
                    teacher_fail_roots.append(rid)
                target_types[str(audit.get("target_type", "unknown"))] += 1
                if "route_progress" in audit:
                    route_progress.append(float(audit["route_progress"]))
                score_values.append(float(row.get("score_star", 0.0)))
                rho_values.append(float(row.get("rho_imp", 0.0)))
                reset_norms.append(float(np.linalg.norm(np.asarray(row.get("r_reset"), dtype=np.float32))))
                control_abs_max.append(float(np.max(np.abs(np.asarray(row.get("teacher_u"), dtype=np.float32)))))
                for idx, name in enumerate(CERT_NAMES):
                    cert = np.asarray(row.get("cert_star"), dtype=np.float32)
                    if cert.size > idx:
                        cert_values[name].append(float(cert[idx]))

            split_failures = []
            if action_counts and any(c != len(expected_actions) for c in action_counts):
                split_failures.append("wrong action count")
            if bad_action_roots:
                split_failures.append("wrong action ids")
            if duplicate_action_roots:
                split_failures.append("duplicate action ids")
            if missing_fields:
                split_failures.append("missing required fields")
            if bad_shapes:
                split_failures.append("shape mismatch")
            if finite_bad_roots:
                split_failures.append("non-finite tensors")
            if teacher_fail_roots:
                split_failures.append("teacher failure")
            if tau_bad_roots:
                split_failures.append("bad reset boundary")
            if contact_bad_roots:
                split_failures.append("bad contact fields")
            if root_inconsistent:
                split_failures.append("inconsistent root ids")
        split_summary = {
            "roots": root_count,
            "rows": row_count,
            "actions_per_root": _stats([float(x) for x in action_counts]),
            "expected_actions": expected_actions,
            "bad_action_roots": bad_action_roots[:50],
            "duplicate_action_roots": duplicate_action_roots[:50],
            "contact_rate": float(np.mean(contacts)) if contacts else 0.0,
            "harm_bins": {str(k): int(v) for k, v in sorted(bins.items())},
            "has_no_contact_bin": bool(0 in bins),
            "has_contact_bins": bool(any(k > 0 for k in bins)),
            "a_safe_size": _stats([float(x) for x in a_sizes]),
            "delta_c": _stats(spreads),
            "gap_fraction": float(gap_count / max(1, root_count)),
            "families": dict(families),
            "sim_sources": dict(sim_sources),
            "score_star": _stats(score_values),
            "rho_imp": _stats(rho_values),
            "cert_star_by_margin": {name: _stats(vals) for name, vals in cert_values.items()},
            "reset_state_norm": _stats(reset_norms),
            "teacher_control_abs_max": _stats(control_abs_max),
            "route_progress": _stats(route_progress),
            "target_types": dict(target_types),
            "missing_fields": dict(missing_fields),
            "bad_shapes": dict(bad_shapes),
            "finite_ok": not finite_bad_roots,
            "finite_bad_roots": finite_bad_roots[:50],
            "teacher_success_rate": 1.0 - len(set(teacher_fail_roots)) / max(1, root_count),
            "teacher_fail_roots": sorted(set(teacher_fail_roots))[:50],
            "tau_bad_roots": sorted(set(tau_bad_roots))[:50],
            "contact_bad_roots": sorted(set(contact_bad_roots))[:50],
            "root_inconsistent": sorted(set(root_inconsistent))[:50],
            "failures": split_failures,
        }
        summary[split] = split_summary
        csv_rows.append({
            "split": split,
            "roots": root_count,
            "rows": row_count,
            "actions_per_root_mean": split_summary["actions_per_root"]["mean"],
            "contact_rate": split_summary["contact_rate"],
            "a_safe_mean": split_summary["a_safe_size"]["mean"],
            "delta_c_mean": split_summary["delta_c"]["mean"],
            "gap_fraction": split_summary["gap_fraction"],
            "finite_ok": split_summary["finite_ok"],
            "teacher_success_rate": split_summary["teacher_success_rate"],
            "num_bad_shapes": sum(bad_shapes.values()),
            "num_failures": len(split_failures),
        })
        failures.extend(f"{split}: {f}" for f in split_failures)
    summary["leakage_roots"] = sorted(set(leakage))
    if leakage:
        failures.append("root leakage")
    summary["overall_pass"] = not failures
    summary["failures"] = failures
    write_json(out / "diagnostics.json", summary)
    pd.DataFrame(csv_rows).to_csv(out / "diagnostics.csv", index=False)
    if strict and failures:
        raise RuntimeError("dataset diagnostics failed: " + "; ".join(failures))
    return summary

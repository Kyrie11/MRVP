from __future__ import annotations

import numpy as np

from mrvp.sim.harm import construct_harm_comparable_set


INTERNAL_METHODS = [
    "severity_only",
    "post_reset_scalar_risk",
    "generic_world_model_risk",
    "handcrafted_reset_features_risk",
    "cmrt_direct_certificate",
    "mrvp_mean",
    "mrvp_worst",
    "mrvp_full",
]


def _candidate_rows(rows: list[dict], use_harm_filter: bool = True) -> list[dict]:
    return construct_harm_comparable_set(rows) if use_harm_filter else list(rows)


def severity_only(rows: list[dict]) -> dict:
    candidates = _candidate_rows(rows, use_harm_filter=True)
    selected = min(candidates, key=lambda r: (float(r["rho_imp"]), -float(r.get("audit", {}).get("route_progress", 0.0))))
    return {"selected_action": str(selected["action_id"]), "score": -float(selected["rho_imp"])}


def heuristic_certificate(row: dict, mode: str) -> float:
    r = np.asarray(row["r_reset"], dtype=np.float32)
    deg = np.asarray(row["deg"], dtype=np.float32)
    A = np.asarray(row["world_reset"]["A"], dtype=np.float32)
    O = np.asarray(row["world_reset"]["O"], dtype=np.float32)
    speed_penalty = 0.025 * abs(float(r[3]))
    yaw_penalty = 0.2 * abs(float(r[5]))
    road_score = float(A[0].mean() + A[3].mean())
    occ_penalty = float(O.mean())
    authority = float(np.mean(deg[1:4]))
    if mode == "generic_world_model_risk":
        return road_score - 1.5 * occ_penalty - 0.4 * speed_penalty
    if mode == "handcrafted_reset_features_risk":
        return 0.4 * authority + 0.4 * road_score - speed_penalty - yaw_penalty
    if mode == "cmrt_direct_certificate":
        return 0.45 * road_score + 0.35 * authority - 1.2 * occ_penalty - yaw_penalty
    return 0.35 * authority + 0.35 * road_score - occ_penalty - speed_penalty


def select_by_heuristic(rows: list[dict], method: str, use_harm_filter: bool = True) -> dict:
    if method.startswith("external_"):
        raise RuntimeError("外部算法未内置")
    if method == "severity_only":
        return severity_only(rows)
    candidates = _candidate_rows(rows, use_harm_filter=use_harm_filter)
    scored = [(heuristic_certificate(row, method), row) for row in candidates]
    score, selected = max(scored, key=lambda item: item[0])
    return {"selected_action": str(selected["action_id"]), "score": float(score), "scores": {str(row["action_id"]): float(s) for s, row in scored}}


def method_to_selector_mode(method: str) -> str:
    if method == "mrvp_mean":
        return "mean"
    if method == "mrvp_worst":
        return "worst"
    return "lcvar"

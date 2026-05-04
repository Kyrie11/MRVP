from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from mrvp.data.schema import BOTTLE_NECKS


def control_authority_margin(u: np.ndarray, d: np.ndarray) -> float:
    steer_scale, brake_scale, throttle_scale, delay, friction, damage_norm = d[:6]
    steer_margin = float(steer_scale - abs(u[0]))
    brake_margin = float(brake_scale - max(u[1], 0.0))
    throttle_margin = float(throttle_scale - max(u[2], 0.0))
    return min(steer_margin, brake_margin, throttle_margin) - 0.05 * float(delay)


def stability_margin(state: np.ndarray, d: np.ndarray) -> float:
    speed = max(float(state[5]), 0.1)
    yaw_rate = abs(float(state[6]))
    beta = abs(float(state[7]))
    mu = max(float(d[4]), 0.1)
    yaw_limit = mu * 9.81 / speed
    return min(0.45 - beta, yaw_limit - yaw_rate)


def road_margin(state: np.ndarray, h_ctx: np.ndarray | Mapping[str, Any]) -> float:
    if isinstance(h_ctx, Mapping):
        lane_width = float(h_ctx.get("drivable_width", h_ctx.get("lane_width", 3.5)))
        route_y = float(h_ctx.get("route_y", 0.0))
    else:
        lane_width = float(h_ctx[2]) if len(h_ctx) > 2 else 3.5
        route_y = 0.0
    y = float(state[1])
    ego_half_width = 0.95
    return lane_width / 2.0 - abs(y - route_y) - ego_half_width


def secondary_margin(state: np.ndarray, hazards: Sequence[Mapping[str, float]] | None = None, d_safe: float = 1.0) -> float:
    if not hazards:
        return 50.0
    xy = np.asarray([state[0], state[1]], dtype=np.float32)
    best = 50.0
    for h in hazards:
        hxy = np.asarray([h.get("x", 0.0), h.get("y", 0.0)], dtype=np.float32)
        radius = float(h.get("radius", 2.0))
        best = min(best, float(np.linalg.norm(xy - hxy) - radius - d_safe))
    return best


def return_distance_margin(state: np.ndarray, h_ctx: np.ndarray | Mapping[str, Any]) -> float:
    if isinstance(h_ctx, Mapping):
        target_y = float(h_ctx.get("route_y", 0.0))
        corridor = float(h_ctx.get("return_corridor_length", 30.0))
    else:
        target_y = 0.0
        corridor = float(h_ctx[9]) if len(h_ctx) > 9 else 30.0
    return corridor * 0.05 - abs(float(state[1]) - target_y) - 0.2 * abs(float(state[2]))


def compute_bottleneck_targets(
    states: np.ndarray,
    controls: np.ndarray,
    d_deg: np.ndarray,
    h_ctx: np.ndarray | Mapping[str, Any],
    hazards: Sequence[Mapping[str, float]] | None = None,
) -> Dict[str, Any]:
    """Compute signed teacher margins for the five bottlenecks.

    Positive means safe. The return target follows Appendix Eq. safe-until-return:
    max_tau min(distance-to-target margin, min_{k<=tau} instantaneous safety).
    """
    states = np.asarray(states, dtype=np.float32)
    controls = np.asarray(controls, dtype=np.float32)
    d_deg = np.asarray(d_deg, dtype=np.float32)
    sec = np.asarray([secondary_margin(s, hazards) for s in states], dtype=np.float32)
    road = np.asarray([road_margin(s, h_ctx) for s in states], dtype=np.float32)
    stab = np.asarray([stability_margin(s, d_deg) for s in states], dtype=np.float32)
    ctrl = np.asarray([control_authority_margin(u, d_deg) for u in controls], dtype=np.float32)
    safe_inst = np.minimum.reduce([sec, road, stab, ctrl])
    prefix_safe = np.minimum.accumulate(safe_inst)
    ret_dist = np.asarray([return_distance_margin(s, h_ctx) for s in states], dtype=np.float32)
    ret = np.max(np.minimum(ret_dist, prefix_safe))
    r_star = np.asarray([sec.min(), road.min(), stab.min(), ctrl.min(), ret], dtype=np.float32)
    return {
        "r_star": r_star.tolist(),
        "b_star": int(np.argmin(r_star)),
        "s_star": float(np.min(r_star)),
        "margin_traces": {
            "sec": sec.tolist(),
            "road": road.tolist(),
            "stab": stab.tolist(),
            "ctrl": ctrl.tolist(),
            "return_distance": ret_dist.tolist(),
            "safe_prefix": prefix_safe.tolist(),
        },
    }

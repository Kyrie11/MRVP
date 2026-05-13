from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class ResetBoundaryTarget:
    reset_idx: int
    reset_time: float
    reset_state: np.ndarray
    difficulty: float


def lightweight_difficulty(state: np.ndarray, degradation: Sequence[float] | None = None, route_goal_x: float | None = None) -> float:
    """Approximate recovery difficulty used when teacher rollouts per prefix step are too expensive.

    Larger values mean harder recovery.  The score combines lane clearance,
    stability, control authority and route progress without reading teacher
    margins or outcome labels.
    """
    x = np.asarray(state, dtype=np.float32)
    deg = np.asarray(degradation if degradation is not None else [1, 1, 1, 0, 1, 0], dtype=np.float32)
    lane_half_width = 1.75
    road_margin = lane_half_width - abs(float(x[1])) - 0.95
    yaw_rate = abs(float(x[5])) if x.size > 5 else 0.0
    beta = abs(float(x[8])) if x.size > 8 else 0.0
    friction = float(deg[4]) if deg.size > 4 else 1.0
    stability_margin = 1.4 * friction - 0.55 * yaw_rate - 0.35 * beta
    control_margin = min(float(deg[0]) if deg.size > 0 else 1.0, float(deg[1]) if deg.size > 1 else 1.0) - 0.35
    progress_margin = 0.0 if route_goal_x is None else 0.03 * (float(route_goal_x) - float(x[0]))
    return -float(min(road_margin, stability_margin, control_margin, progress_margin))


def construct_reset_boundary_target(
    prefix_states: Sequence[Sequence[float]],
    prefix_times: Sequence[float] | None = None,
    degradation: Sequence[float] | None = None,
    route_goal_x: float | None = None,
) -> ResetBoundaryTarget:
    """Pick the hardest recovery-reasoning boundary along a prefix trajectory."""
    states = [np.asarray(s, dtype=np.float32) for s in prefix_states]
    if not states:
        raise ValueError("prefix_states must contain at least one state")
    scores = [lightweight_difficulty(s, degradation=degradation, route_goal_x=route_goal_x) for s in states]
    idx = int(np.argmax(scores))
    if prefix_times is None:
        reset_time = float(idx)
    else:
        reset_time = float(list(prefix_times)[idx])
    return ResetBoundaryTarget(reset_idx=idx, reset_time=reset_time, reset_state=states[idx].copy(), difficulty=float(scores[idx]))

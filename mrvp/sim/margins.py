from __future__ import annotations

import numpy as np

from .degradation import degraded_bounds


def sample_grid(field: np.ndarray, x: float, y: float) -> float:
    arr = np.asarray(field, dtype=np.float32)
    h, w = arr.shape[-2], arr.shape[-1]
    ix = int(np.clip(round(x + w / 2), 0, w - 1))
    iy = int(np.clip(round(y + h / 2), 0, h - 1))
    return float(arr[..., iy, ix])


def road_margin_state(state: np.ndarray, world_A: np.ndarray) -> float:
    drivable = sample_grid(world_A[0], float(state[0]), float(state[1]))
    signed = sample_grid(world_A[3], float(state[0]), float(state[1]))
    return float(min(2.0 * drivable - 1.0, signed))


def collision_margin_state(state: np.ndarray, world_O: np.ndarray, t: int, safe_dist: float = 0.1) -> float:
    idx = min(t, world_O.shape[0] - 1)
    occ = sample_grid(world_O[idx, 0], float(state[0]), float(state[1]))
    return float(1.0 - occ - safe_dist)


def control_margin(control: np.ndarray, deg: np.ndarray) -> float:
    bounds = degraded_bounds(deg)
    c = np.asarray(control, dtype=np.float32)
    margins = np.array([bounds[0] - abs(c[0]), bounds[1] - abs(c[1]), bounds[2] - abs(c[2])], dtype=np.float32)
    return float(margins.min())


def stability_margin(state: np.ndarray, deg: np.ndarray) -> float:
    beta_lim = float(deg[5])
    yaw_lim = float(deg[6])
    beta = abs(float(state[8])) if len(state) > 8 else 0.0
    yaw_rate = abs(float(state[5])) if len(state) > 5 else 0.0
    return float(min(beta_lim - beta, yaw_lim - yaw_rate))


def goal_margin_state(state: np.ndarray, world_G: np.ndarray) -> float:
    vals = [sample_grid(ch, float(state[0]), float(state[1])) for ch in world_G]
    return float(max(vals) - 0.25)


def compute_rollout_margins(trajectory: np.ndarray, controls: np.ndarray, world: dict, deg: np.ndarray) -> dict[str, float]:
    traj = np.asarray(trajectory, dtype=np.float32)
    ctrl = np.asarray(controls, dtype=np.float32)
    A = np.asarray(world["A"], dtype=np.float32)
    O = np.asarray(world["O"], dtype=np.float32)
    G = np.asarray(world["G"], dtype=np.float32)
    road = min(road_margin_state(x, A) for x in traj)
    col = min(collision_margin_state(x, O, i) for i, x in enumerate(traj))
    ctrl_margin = min(control_margin(u, deg) for u in ctrl) if len(ctrl) else 0.0
    stab = min(stability_margin(x, deg) for x in traj)
    goal = goal_margin_state(traj[-1], G)
    score = min(road, col, ctrl_margin, stab, goal)
    return {"road": road, "col": col, "ctrl": ctrl_margin, "stab": stab, "goal": goal, "score": score}


def violation_depth(score: float) -> float:
    return float(max(0.0, -float(score)))

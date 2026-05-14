from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .degradation import apply_degradation
from .margins import compute_rollout_margins


@dataclass
class TeacherResult:
    success: bool
    controls: np.ndarray
    trajectory: np.ndarray
    margins: dict[str, float]
    score_star: float
    target_type: str
    target_xy: np.ndarray


def rollout_bicycle(x0: np.ndarray, controls: np.ndarray, dt: float = 0.1, wheelbase: float = 2.8) -> np.ndarray:
    x = np.asarray(x0, dtype=np.float32).copy()
    traj = [x.copy()]
    for u in np.asarray(controls, dtype=np.float32):
        steer, brake, traction = float(u[0]), float(u[1]), float(u[2])
        v = max(0.0, float(x[3]) + (traction - brake * 4.0) * dt)
        yaw_rate = v / wheelbase * np.tan(np.clip(steer, -0.6, 0.6))
        x[2] = x[2] + yaw_rate * dt
        x[0] = x[0] + v * np.cos(x[2]) * dt
        x[1] = x[1] + v * np.sin(x[2]) * dt
        x[3] = v
        x[5] = yaw_rate
        x[8] = np.arctan2(float(x[4]), max(1e-3, v))
        x[9] = steer
        x[10] = brake
        x[11] = traction
        traj.append(x.copy())
    return np.stack(traj, axis=0).astype(np.float32)


def candidate_controls(horizon: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray, np.ndarray]]:
    profiles = []
    for name, steer, brake in [
        ("route_continue", 0.0, 0.25),
        ("safe_stop", 0.0, 0.75),
        ("refuge_left", 0.22, 0.45),
        ("refuge_right", -0.22, 0.45),
        ("gentle_left", 0.10, 0.35),
        ("gentle_right", -0.10, 0.35),
    ]:
        arr = np.zeros((horizon, 3), dtype=np.float32)
        arr[:, 0] = steer
        arr[:, 1] = brake
        arr[:, 2] = 0.0
        target = np.array([12.0, 0.0], dtype=np.float32)
        if "left" in name:
            target[1] = 5.0
        if "right" in name:
            target[1] = -5.0
        if name == "safe_stop":
            target[0] = 5.0
        profiles.append((name, arr, target))
    noise = rng.normal(0, 0.04, size=(horizon, 3)).astype(np.float32)
    arr = np.zeros((horizon, 3), dtype=np.float32)
    arr[:, 0] = np.clip(noise[:, 0], -0.2, 0.2)
    arr[:, 1] = np.clip(0.4 + noise[:, 1], 0.0, 1.0)
    profiles.append(("sampled_stop", arr, np.array([6.0, 0.0], dtype=np.float32)))
    return profiles


def solve_teacher(x0: np.ndarray, world: dict, deg: np.ndarray, horizon: int = 30, dt: float = 0.1, seed: int = 0) -> TeacherResult:
    rng = np.random.default_rng(seed)
    best = None
    for name, controls, target in candidate_controls(horizon, rng):
        degraded = apply_degradation(controls, deg)
        traj = rollout_bicycle(x0, degraded, dt=dt)
        margins = compute_rollout_margins(traj, degraded, world, deg)
        score = float(margins["score"])
        if best is None or score > best[0]:
            best = (score, name, degraded, traj, margins, target)
    assert best is not None
    score, name, controls, traj, margins, target = best
    return TeacherResult(
        success=bool(score > -2.0),
        controls=controls.astype(np.float32),
        trajectory=traj.astype(np.float32),
        margins=margins,
        score_star=float(score),
        target_type=name,
        target_xy=target.astype(np.float32),
    )

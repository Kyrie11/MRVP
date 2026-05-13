from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from mrvp.data.schema import CONTROL_DIM, RECOVERY_HORIZON, STATE_DIM


@dataclass
class TeacherResult:
    """Recovery teacher output used by MRVP dataset builders."""

    u: np.ndarray
    traj: np.ndarray
    margins: np.ndarray
    success: bool
    cost: float


def _world_get(world: Mapping[str, Any] | np.ndarray | None, group: str, idx: int, default: float) -> float:
    if isinstance(world, Mapping):
        value = world.get(group, None)
        if value is None:
            return default
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return float(arr[idx]) if idx < arr.size else default
    return default


def _step(state: np.ndarray, control: np.ndarray, degradation: np.ndarray, dt: float) -> np.ndarray:
    x = state.astype(np.float32).copy()
    u = control.astype(np.float32)
    steer_scale = float(degradation[0]) if degradation.size > 0 else 1.0
    brake_scale = float(degradation[1]) if degradation.size > 1 else 1.0
    throttle_scale = float(degradation[2]) if degradation.size > 2 else 1.0
    friction = float(degradation[4]) if degradation.size > 4 else 0.9
    delta = float(np.clip(u[0], -steer_scale, steer_scale))
    brake = float(np.clip(u[1], 0.0, brake_scale))
    throttle = float(np.clip(u[2], 0.0, throttle_scale))
    speed = float(np.hypot(x[3], x[4]))
    yaw = float(x[2])
    beta = float(np.clip(0.45 * delta / max(0.25, friction), -0.7, 0.7))
    acc = 2.0 * throttle - 3.5 * brake - 0.04 * speed
    next_speed = max(0.0, speed + dt * acc)
    yaw_rate = float(np.clip(next_speed * np.tan(delta) / 2.7, -1.5, 1.5))
    next_yaw = yaw + dt * yaw_rate
    x[0] += dt * next_speed * np.cos(next_yaw + beta)
    x[1] += dt * next_speed * np.sin(next_yaw + beta)
    x[2] = next_yaw
    x[3] = next_speed * np.cos(next_yaw + beta)
    x[4] = next_speed * np.sin(next_yaw + beta)
    x[5] = yaw_rate
    x[6] = acc * np.cos(next_yaw)
    x[7] = acc * np.sin(next_yaw)
    x[8] = beta
    x[9:12] = [delta, brake, throttle]
    return x


def _margins(traj: np.ndarray, controls: np.ndarray, world: Mapping[str, Any] | np.ndarray | None, degradation: np.ndarray) -> np.ndarray:
    lane_width = _world_get(world, "affordance", 0, 3.5)
    clearance_hint = _world_get(world, "affordance", 1, lane_width / 2.0)
    actor_density = _world_get(world, "occupancy", 0, 0.2)
    secondary_clear = _world_get(world, "occupancy", 1, 8.0)
    route_dx = _world_get(world, "goal", 0, 20.0)
    friction = float(degradation[4]) if degradation.size > 4 else 0.9
    lateral = np.abs(traj[:, 1])
    road = np.min(lane_width / 2.0 - lateral - 0.9)
    sec = secondary_clear - 2.0 - 3.0 * actor_density - 0.05 * np.max(np.hypot(traj[:, 3], traj[:, 4]))
    stab = friction * 1.5 - float(np.max(np.abs(traj[:, 5]))) - 0.5 * float(np.max(np.abs(traj[:, 8])))
    steer_scale = float(degradation[0]) if degradation.size > 0 else 1.0
    brake_scale = float(degradation[1]) if degradation.size > 1 else 1.0
    throttle_scale = float(degradation[2]) if degradation.size > 2 else 1.0
    ctrl = min(
        steer_scale - float(np.max(np.abs(controls[:, 0]))),
        brake_scale - float(np.max(np.clip(controls[:, 1], 0.0, None))),
        throttle_scale - float(np.max(np.clip(controls[:, 2], 0.0, None))),
    )
    progress = float(traj[-1, 0] - traj[0, 0])
    goal = min(progress / max(1.0, route_dx), 1.0) + 0.2 * float(clearance_hint > 0.0) - 0.2 * abs(float(traj[-1, 1]))
    return np.asarray([sec, road, stab, ctrl, goal], dtype=np.float32)


def solve_recovery_teacher(
    reset_state: np.ndarray,
    recovery_world: Mapping[str, Any] | np.ndarray | None,
    degradation: np.ndarray,
    horizon: int = RECOVERY_HORIZON,
    dt: float = 0.1,
    num_sequences: int = 128,
    seed: Optional[int] = None,
) -> TeacherResult:
    """Lightweight degraded shooting-MPC teacher.

    This is intentionally dependency-free and deterministic enough for data
    generation smoke tests. It samples control sequences around a lane-centering
    stabilizer, rolls them through degraded dynamics, scores explicit margins and
    returns the best feasible sequence.
    """
    rng = np.random.default_rng(seed)
    reset_state = np.asarray(reset_state, dtype=np.float32).reshape(-1)[:STATE_DIM]
    if reset_state.size < STATE_DIM:
        reset_state = np.pad(reset_state, (0, STATE_DIM - reset_state.size))
    degradation = np.asarray(degradation, dtype=np.float32).reshape(-1)
    if degradation.size < 6:
        degradation = np.pad(degradation, (0, 6 - degradation.size), constant_values=1.0)

    candidates = []
    for n in range(max(1, num_sequences)):
        x = reset_state.copy()
        controls = np.zeros((horizon, CONTROL_DIM), dtype=np.float32)
        traj = np.zeros((horizon + 1, STATE_DIM), dtype=np.float32)
        traj[0] = x
        aggressiveness = rng.uniform(0.6, 1.4)
        target_speed = rng.uniform(4.0, 11.0)
        for k in range(horizon):
            alpha = k / max(1, horizon - 1)
            steer = -0.35 * np.tanh(x[1]) - 0.25 * np.tanh(x[2]) + rng.normal(0, 0.04) * (1.0 - alpha)
            speed = float(np.hypot(x[3], x[4]))
            speed_err = target_speed - speed
            brake = max(0.0, -0.25 * speed_err + rng.normal(0, 0.03))
            throttle = max(0.0, 0.18 * speed_err + rng.normal(0, 0.03)) * aggressiveness
            controls[k] = [steer, brake, throttle]
            x = _step(x, controls[k], degradation, dt)
            traj[k + 1] = x
        margins = _margins(traj, controls, recovery_world, degradation)
        cert = float(np.min(margins))
        smooth = float(np.mean(np.abs(np.diff(controls, axis=0)))) if horizon > 1 else 0.0
        cost = -cert + 0.03 * smooth + 0.02 * abs(float(traj[-1, 1]))
        candidates.append((cost, controls, traj, margins))
    cost, controls, traj, margins = min(candidates, key=lambda item: item[0])
    return TeacherResult(u=controls, traj=traj, margins=margins, success=bool(np.min(margins) >= 0.0), cost=float(cost))

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .targets import compute_bottleneck_targets


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def kinematic_step(state: np.ndarray, control: np.ndarray, d: np.ndarray, dt: float = 0.1, wheelbase: float = 2.8) -> np.ndarray:
    x, y, yaw, vx, vy, speed, yaw_rate, beta, steer_prev, t = state.astype(float).tolist()
    steer_cmd, brake_cmd, throttle_cmd = control.astype(float).tolist()
    steer_scale, brake_scale, throttle_scale, delay, friction, _ = d.astype(float).tolist()
    steer = _clip(steer_cmd, -steer_scale, steer_scale)
    brake = _clip(brake_cmd, 0.0, brake_scale)
    throttle = _clip(throttle_cmd, 0.0, throttle_scale)
    # Very small degraded bicycle model; friction bounds acceleration and yaw response.
    accel = 2.5 * throttle - 5.0 * brake
    accel = _clip(accel, -friction * 9.81, friction * 3.5)
    speed = max(0.0, speed + accel * dt)
    yaw_rate = speed / wheelbase * math.tan(_clip(steer, -0.75, 0.75))
    yaw = yaw + yaw_rate * dt
    beta = math.atan2(0.5 * math.tan(steer), 1.0)
    vx = speed * math.cos(yaw + beta)
    vy = speed * math.sin(yaw + beta)
    x = x + vx * dt
    y = y + vy * dt
    return np.asarray([x, y, yaw, vx, vy, speed, yaw_rate, beta, steer, t + dt], dtype=np.float32)


def degraded_mpc_teacher(
    x_plus: Sequence[float],
    d_deg: Sequence[float],
    h_ctx: Sequence[float] | Mapping[str, Any],
    horizon: float = 5.0,
    dt: float = 0.1,
    hazards: Sequence[Mapping[str, float]] | None = None,
) -> Dict[str, Any]:
    """Lightweight degraded MPC-like teacher.

    The implementation uses receding-horizon grid search over a small control
    set; it is deterministic and available without CasADi. Replace this with a
    full MPC/CBF-QP in CARLA closed-loop experiments if desired.
    """
    state = np.asarray(x_plus, dtype=np.float32).copy()
    d = np.asarray(d_deg, dtype=np.float32).copy()
    n = int(round(horizon / dt))
    states = [state.copy()]
    controls = []
    route_y = 0.0 if not isinstance(h_ctx, Mapping) else float(h_ctx.get("route_y", 0.0))
    target_speed = min(float(state[5]), 8.0)
    candidates = []
    for steer in np.linspace(-float(d[0]), float(d[0]), 7):
        for brake in [0.0, 0.25 * float(d[1]), 0.55 * float(d[1]), float(d[1])]:
            for throttle in [0.0, 0.25 * float(d[2])]:
                candidates.append(np.asarray([steer, brake, throttle], dtype=np.float32))
    for _ in range(n):
        best_u = candidates[0]
        best_cost = float("inf")
        # One-step look-ahead plus stabilizing terms. This acts like a fast surrogate MPC.
        for u in candidates:
            ns = kinematic_step(state, u, d, dt)
            y_err = ns[1] - route_y
            yaw_err = math.atan2(math.sin(ns[2]), math.cos(ns[2]))
            speed_err = ns[5] - target_speed
            cost = 3.0 * y_err**2 + 1.2 * yaw_err**2 + 0.1 * speed_err**2 + 0.2 * u[0] ** 2 + 0.1 * u[1] ** 2
            if hazards:
                xy = np.asarray([ns[0], ns[1]], dtype=np.float32)
                for h in hazards:
                    hxy = np.asarray([h.get("x", 0.0), h.get("y", 0.0)], dtype=np.float32)
                    dist = float(np.linalg.norm(xy - hxy))
                    cost += 5.0 / max(dist - float(h.get("radius", 2.0)), 0.2)
            if cost < best_cost:
                best_cost = cost
                best_u = u
        state = kinematic_step(state, best_u, d, dt)
        controls.append(best_u.copy())
        states.append(state.copy())
    controls_arr = np.asarray(controls, dtype=np.float32)
    if controls_arr.shape[0] < len(states):
        controls_arr = np.vstack([controls_arr, controls_arr[-1:] if len(controls_arr) else np.zeros((1, 3), dtype=np.float32)])
    states_arr = np.asarray(states, dtype=np.float32)
    targets = compute_bottleneck_targets(states_arr, controls_arr[: len(states_arr)], d, h_ctx, hazards=hazards)
    return {"teacher": "degraded_mpc_grid", "states": states_arr.tolist(), "controls": controls_arr[: len(states_arr)].tolist(), **targets}


def heuristic_post_impact_teacher(x_plus, d_deg, h_ctx, horizon: float = 5.0, dt: float = 0.1, hazards=None) -> Dict[str, Any]:
    """Even simpler stabilizing recovery teacher for ablations."""
    state = np.asarray(x_plus, dtype=np.float32).copy()
    d = np.asarray(d_deg, dtype=np.float32).copy()
    route_y = 0.0 if not isinstance(h_ctx, Mapping) else float(h_ctx.get("route_y", 0.0))
    states = [state.copy()]
    controls = []
    for _ in range(int(round(horizon / dt))):
        y_err = float(state[1] - route_y)
        yaw_err = math.atan2(math.sin(float(state[2])), math.cos(float(state[2])))
        steer = _clip(-0.35 * y_err - 0.75 * yaw_err - 0.15 * float(state[7]), -float(d[0]), float(d[0]))
        brake = _clip(0.15 + 0.08 * max(float(state[5]) - 6.0, 0.0), 0.0, float(d[1]))
        throttle = 0.0
        u = np.asarray([steer, brake, throttle], dtype=np.float32)
        state = kinematic_step(state, u, d, dt)
        states.append(state.copy())
        controls.append(u.copy())
    controls_arr = np.asarray(controls, dtype=np.float32)
    if controls_arr.shape[0] < len(states):
        controls_arr = np.vstack([controls_arr, controls_arr[-1:]])
    states_arr = np.asarray(states, dtype=np.float32)
    targets = compute_bottleneck_targets(states_arr, controls_arr[: len(states_arr)], d, h_ctx, hazards=hazards)
    return {"teacher": "heuristic_post_impact", "states": states_arr.tolist(), "controls": controls_arr[: len(states_arr)].tolist(), **targets}

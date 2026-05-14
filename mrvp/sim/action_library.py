from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmergencyPrefix:
    action_id: str
    controls: np.ndarray
    params: dict
    duration: float
    dt: float


def _ramp(value: float, steps: int, ramp_steps: int) -> np.ndarray:
    out = np.full(steps, value, dtype=np.float32)
    if ramp_steps > 0:
        out[:ramp_steps] = np.linspace(0.0, value, ramp_steps, dtype=np.float32)
    return out


def make_prefix(action_id: str, steer: np.ndarray, brake: np.ndarray, traction: np.ndarray, duration: float, dt: float, params: dict) -> EmergencyPrefix:
    controls = np.stack([steer, brake, traction], axis=-1).astype(np.float32)
    return EmergencyPrefix(action_id=action_id, controls=controls, params=params, duration=duration, dt=dt)


def default_action_library(
    max_steer: float = 0.6,
    max_brake: float = 1.0,
    duration: float = 1.0,
    dt: float = 0.1,
    boundary_direction: float = 1.0,
    corridor_direction: float = -1.0,
) -> list[EmergencyPrefix]:
    steps = int(round(duration / dt))
    ramp = max(1, int(round(0.25 / dt)))
    zero = np.zeros(steps, dtype=np.float32)
    lib = [
        make_prefix("hard_brake", zero, np.full(steps, 0.9 * max_brake, dtype=np.float32), zero, duration, dt, {"brake": 0.9}),
        make_prefix("brake_left", _ramp(0.35 * max_steer, steps, ramp), np.full(steps, 0.7 * max_brake, dtype=np.float32), zero, duration, dt, {"direction": 1, "brake": 0.7}),
        make_prefix("brake_right", _ramp(-0.35 * max_steer, steps, ramp), np.full(steps, 0.7 * max_brake, dtype=np.float32), zero, duration, dt, {"direction": -1, "brake": 0.7}),
        make_prefix("maintain", zero, np.zeros(steps, dtype=np.float32), np.zeros(steps, dtype=np.float32), duration, dt, {"maintain": 1}),
        make_prefix("mild_left", _ramp(0.18 * max_steer, steps, ramp), np.full(steps, 0.3 * max_brake, dtype=np.float32), zero, duration, dt, {"direction": 1, "brake": 0.3}),
        make_prefix("mild_right", _ramp(-0.18 * max_steer, steps, ramp), np.full(steps, 0.3 * max_brake, dtype=np.float32), zero, duration, dt, {"direction": -1, "brake": 0.3}),
        make_prefix("boundary_away", _ramp(float(boundary_direction) * 0.30 * max_steer, steps, ramp), np.full(steps, 0.5 * max_brake, dtype=np.float32), zero, duration, dt, {"boundary_direction": float(boundary_direction), "brake": 0.5}),
        make_prefix("corridor_seeking", _ramp(float(corridor_direction) * 0.28 * max_steer, steps, ramp), np.full(steps, 0.5 * max_brake, dtype=np.float32), zero, duration, dt, {"corridor_direction": float(corridor_direction), "brake": 0.5}),
    ]
    return lib

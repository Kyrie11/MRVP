from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEG_NAMES = [
    "friction_mu",
    "steering_scale",
    "brake_scale",
    "throttle_scale",
    "response_delay_s",
    "stability_beta_limit",
    "stability_yaw_rate_limit",
    "damage_class",
]


@dataclass
class ControlLimits:
    max_steer: float = 0.6
    max_brake: float = 1.0
    max_throttle: float = 1.0


def nominal_degradation() -> np.ndarray:
    return np.array([0.95, 1.0, 1.0, 1.0, 0.0, 0.35, 1.2, 0.0], dtype=np.float32)


def sample_degradation(rng: np.random.Generator, family: str) -> np.ndarray:
    deg = nominal_degradation()
    if family == "LF":
        deg[0] = rng.uniform(0.28, 0.55)
        deg[5] = rng.uniform(0.15, 0.25)
        deg[6] = rng.uniform(0.45, 0.8)
    if family == "AD":
        deg[1] = rng.uniform(0.35, 0.75)
        deg[2] = rng.uniform(0.45, 0.85)
        deg[3] = rng.uniform(0.45, 0.85)
        deg[4] = rng.uniform(0.08, 0.22)
        deg[7] = rng.integers(1, 4)
    return deg.astype(np.float32)


def degraded_bounds(deg: np.ndarray, base: ControlLimits | None = None) -> np.ndarray:
    base = base or ControlLimits()
    d = np.asarray(deg, dtype=np.float32)
    return np.array([base.max_steer * d[1], base.max_brake * d[2], base.max_throttle * d[3]], dtype=np.float32)


def apply_degradation(controls: np.ndarray, deg: np.ndarray) -> np.ndarray:
    ctrl = np.asarray(controls, dtype=np.float32).copy()
    bounds = degraded_bounds(deg)
    ctrl[..., 0] = np.clip(ctrl[..., 0], -bounds[0], bounds[0])
    ctrl[..., 1] = np.clip(ctrl[..., 1], 0.0, bounds[1])
    ctrl[..., 2] = np.clip(ctrl[..., 2], -bounds[2], bounds[2])
    return ctrl

from __future__ import annotations

import numpy as np

from mrvp.common.geometry import rasterize_disc


def make_recovery_world(size: int, rng: np.random.Generator, family: str, reset_state: np.ndarray, horizon: int = 15) -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2.0
    road_width = size * (0.55 if family in {"NCR", "BLE"} else 0.72)
    lateral = np.abs(yy - center)
    drivable = (lateral < road_width / 2).astype(np.float32)
    boundary = ((np.abs(lateral - road_width / 2) < 1.5)).astype(np.float32)
    static = np.zeros((size, size), dtype=np.float32)
    if family in {"NCR", "OSH", "BLE"}:
        obs_x = int(center + rng.integers(-8, 14))
        obs_y = int(center + rng.choice([-1, 1]) * rng.integers(8, 20))
        static = np.maximum(static, rasterize_disc(size, (obs_x, obs_y), rng.uniform(3, 6)))
    signed = np.clip((road_width / 2 - lateral) / max(1.0, road_width / 2), -1, 1).astype(np.float32)
    route = np.exp(-((yy - center) ** 2) / (2 * (size * 0.05) ** 2)).astype(np.float32) * drivable
    speed = np.full((size, size), 0.7 if family == "LF" else 1.0, dtype=np.float32) * drivable
    A = np.stack([drivable, boundary, static, signed, route, speed], axis=0).astype(np.float32)
    O = np.zeros((horizon, 1, size, size), dtype=np.float32)
    for t in range(horizon):
        shift = t * (1.2 if family in {"CI", "OSH", "SC"} else 0.45)
        cx = center + shift + rng.normal(0, 0.4)
        cy = center + (12 if family in {"SC", "CI", "OSH"} else 0) + rng.normal(0, 0.4)
        O[t, 0] = rasterize_disc(size, (cx, cy), 2.5, value=1.0)
    goal_route = np.exp(-((xx - (center + size * 0.25)) ** 2 + (yy - center) ** 2) / (2 * (size * 0.08) ** 2)).astype(np.float32) * drivable
    safe_stop = np.exp(-((xx - (center + size * 0.12)) ** 2 + (yy - (center + size * 0.18)) ** 2) / (2 * (size * 0.10) ** 2)).astype(np.float32) * drivable
    refuge = np.exp(-((xx - (center + size * 0.20)) ** 2 + (yy - (center - road_width / 2 + 3)) ** 2) / (2 * (size * 0.08) ** 2)).astype(np.float32) * drivable
    G = np.stack([goal_route, safe_stop, refuge], axis=0).astype(np.float32)
    Y = np.zeros((horizon, 2, size, size), dtype=np.float32)
    Y[:, 0] = 0.2 if family in {"CI", "SC"} else 0.05
    Y[:, 1] = -0.1 if family == "OSH" else 0.0
    return {"A": A, "O": O, "G": G, "Y": Y}

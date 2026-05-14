from __future__ import annotations

import numpy as np

from .geometry import rotation_matrix, wrap_angle


def to_reset_frame(states: np.ndarray, reset_state: np.ndarray) -> np.ndarray:
    states_arr = np.asarray(states, dtype=np.float32).copy()
    ref = np.asarray(reset_state, dtype=np.float32)
    rot = rotation_matrix(-float(ref[2]))
    xy = states_arr[..., :2] - ref[:2]
    states_arr[..., :2] = xy @ rot.T
    states_arr[..., 2] = wrap_angle(states_arr[..., 2] - ref[2])
    return states_arr


def normalize_state(x: np.ndarray) -> np.ndarray:
    scale = np.array([50, 50, 3.14, 30, 10, 2, 8, 8, 1, 1, 1, 1], dtype=np.float32)
    return np.asarray(x, dtype=np.float32) / scale


def denormalize_state(x: np.ndarray) -> np.ndarray:
    scale = np.array([50, 50, 3.14, 30, 10, 2, 8, 8, 1, 1, 1, 1], dtype=np.float32)
    return np.asarray(x, dtype=np.float32) * scale

from __future__ import annotations

import numpy as np


def contact_reset_index(prefix_rollout: np.ndarray, contact_index: int, dt: float, epsilon_v: float = 0.5, epsilon_r: float = 0.15, k_s: int = 3, max_wait_s: float = 0.25) -> int:
    states = np.asarray(prefix_rollout, dtype=np.float32)
    max_extra = max(1, int(round(max_wait_s / dt)))
    upper = min(len(states) - 1, contact_index + max_extra)
    if len(states) < 3:
        return upper
    v = states[:, 3]
    yaw_rate = states[:, 5]
    dv = np.abs(np.gradient(v, dt))
    dr = np.abs(np.gradient(yaw_rate, dt))
    for idx in range(max(contact_index + 1, 1), upper + 1):
        end = min(len(states), idx + k_s)
        if end - idx < k_s:
            break
        if np.all(dv[idx:end] < epsilon_v) and np.all(dr[idx:end] < epsilon_r):
            return idx
    return upper


def non_contact_reset_index(prefix_rollout: np.ndarray, margins: np.ndarray | None = None) -> int:
    states = np.asarray(prefix_rollout, dtype=np.float32)
    if margins is not None:
        return int(np.argmin(np.asarray(margins, dtype=np.float32)))
    lateral = np.abs(states[:, 1])
    yaw = np.abs(states[:, 2])
    speed = np.abs(states[:, 3])
    difficulty = 0.05 * lateral + 0.2 * yaw - 0.01 * speed
    return int(np.argmax(difficulty))


def extract_reset(prefix_rollout: np.ndarray, contact: bool, contact_time: float, dt: float, cfg: dict | None = None) -> tuple[int, float, np.ndarray]:
    cfg = cfg or {}
    if contact:
        contact_idx = max(0, int(round(contact_time / dt)))
        idx = contact_reset_index(
            prefix_rollout,
            contact_idx,
            dt,
            float(cfg.get("epsilon_v", 0.5)),
            float(cfg.get("epsilon_r", 0.15)),
            int(cfg.get("K_s", 3)),
            float(cfg.get("max_post_contact_wait", 0.25)),
        )
    else:
        idx = non_contact_reset_index(prefix_rollout)
    idx = int(np.clip(idx, 0, len(prefix_rollout) - 1))
    return idx, float(idx * dt), np.asarray(prefix_rollout[idx], dtype=np.float32)

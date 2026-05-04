from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mrvp.data.schema import STATE_DIM, harm_bin_from_rho


def state_from_frame(frame: Mapping[str, Any]) -> np.ndarray:
    ego = frame["ego"]
    vx = float(ego.get("vx", 0.0))
    vy = float(ego.get("vy", 0.0))
    speed = float(ego.get("speed", math.hypot(vx, vy)))
    yaw = float(ego.get("yaw", 0.0))
    beta = float(math.atan2(vy, vx + 1e-6) - yaw) if speed > 0.1 else 0.0
    beta = math.atan2(math.sin(beta), math.cos(beta))
    return np.asarray(
        [
            float(ego.get("x", 0.0)),
            float(ego.get("y", 0.0)),
            yaw,
            vx,
            vy,
            speed,
            float(ego.get("yaw_rate", 0.0)),
            beta,
            float(ego.get("steer", 0.0)),
            float(frame.get("t", ego.get("t", 0.0))),
        ],
        dtype=np.float32,
    )


def _frame_index_by_carla_frame(frames: Sequence[Mapping[str, Any]], carla_frame: int) -> int:
    ids = [int(f.get("frame", i)) for i, f in enumerate(frames)]
    idx = int(np.searchsorted(np.asarray(ids), int(carla_frame), side="left"))
    return int(np.clip(idx, 0, len(frames) - 1))


def first_collision_event(rollout: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    events = rollout.get("collision_events", []) or []
    if not events:
        return None
    return sorted(events, key=lambda e: int(e.get("frame", 0)))[0]


def estimate_harm(rollout: Mapping[str, Any], collision_event: Optional[Mapping[str, Any]], contact_idx: int) -> float:
    frames = rollout.get("frames", [])
    if collision_event is not None:
        impulse = np.asarray(collision_event.get("normal_impulse", [0.0, 0.0, 0.0]), dtype=np.float32)
        impulse_norm = float(np.linalg.norm(impulse))
        if impulse_norm > 1e-3:
            # CARLA impulse is in N*s. Divide by a nominal mass to get a delta-v-like binning surrogate.
            return impulse_norm / float(rollout.get("ego_mass", 1500.0))
        ego = frames[contact_idx].get("ego", {})
        ev = np.asarray([ego.get("vx", 0.0), ego.get("vy", 0.0)], dtype=np.float32)
        other_id = int(collision_event.get("other_actor_id", -1))
        for actor in frames[contact_idx].get("actors", []):
            if int(actor.get("id", -2)) == other_id:
                ov = np.asarray([actor.get("vx", 0.0), actor.get("vy", 0.0)], dtype=np.float32)
                return float(np.linalg.norm(ev - ov))
        return float(ego.get("speed", 0.0))
    # Non-contact boundary-critical has no first impact. Keep harm at zero and let boundary transition define recovery.
    return 0.0


def contact_transition(
    rollout: Mapping[str, Any],
    K: int = 4,
    eps_v: float = 0.8,
    eps_r: float = 0.4,
    delta_max: float = 0.8,
) -> Dict[str, Any]:
    frames = rollout.get("frames", [])
    if len(frames) < 2:
        raise ValueError("Rollout must contain at least two frames.")
    event = first_collision_event(rollout)
    if event is None:
        raise ValueError("No collision event available for contact transition.")
    idx_c = _frame_index_by_carla_frame(frames, int(event.get("frame", 0)))
    idx_minus = max(0, idx_c - 1)
    states = np.stack([state_from_frame(f) for f in frames], axis=0)
    times = states[:, 9]
    speed = states[:, 5]
    yaw_rate = states[:, 6]
    dt = np.maximum(np.diff(times, prepend=times[0]), 1e-3)
    dv = np.abs(np.diff(speed, prepend=speed[0]) / dt)
    dr = np.abs(np.diff(yaw_rate, prepend=yaw_rate[0]) / dt)
    max_idx = int(np.searchsorted(times, times[idx_c] + delta_max, side="right"))
    max_idx = min(max_idx, len(frames) - 1)
    idx_plus = max_idx
    for i in range(idx_c + 1, max_idx + 1):
        j = min(i + K, len(frames))
        if j - i >= K and np.all(dv[i:j] < eps_v) and np.all(dr[i:j] < eps_r):
            idx_plus = i
            break
    rho = estimate_harm(rollout, event, idx_c)
    return {
        "transition_type": "contact",
        "event": dict(event),
        "idx_c": idx_c,
        "idx_minus": idx_minus,
        "idx_plus": idx_plus,
        "x_minus": states[idx_minus].tolist(),
        "x_plus": states[idx_plus].tolist(),
        "rho_imp": float(rho),
        "harm_bin": harm_bin_from_rho(float(rho)),
    }


def _lse(x: np.ndarray) -> float:
    m = float(np.max(x))
    return float(m + np.log(np.sum(np.exp(x - m))))


def boundary_transition(
    rollout: Mapping[str, Any],
    eps: Mapping[str, float] | None = None,
    sigma: Mapping[str, float] | None = None,
    delta: float = 0.2,
) -> Dict[str, Any]:
    frames = rollout.get("frames", [])
    if len(frames) < 2:
        raise ValueError("Rollout must contain at least two frames.")
    eps = eps or {"road": 0.25, "stab": 0.15, "ctrl": 0.10, "secondary": 1.0, "return": 2.0}
    sigma = sigma or {"road": 0.25, "stab": 0.25, "ctrl": 0.20, "secondary": 2.0, "return": 4.0}
    scores = []
    states = np.stack([state_from_frame(f) for f in frames], axis=0)
    times = states[:, 9]
    for frame in frames:
        margins = frame.get("margins", {})
        vals = []
        for key in ["road", "stab", "ctrl", "secondary", "return"]:
            vals.append((float(eps.get(key, 0.0)) - float(margins.get(key, 0.0))) / max(float(sigma.get(key, 1.0)), 1e-6))
        scores.append(_lse(np.asarray(vals, dtype=np.float32)))
    idx_star = int(np.argmax(np.asarray(scores)))
    idx_plus = int(np.searchsorted(times, times[idx_star] + delta, side="left"))
    idx_plus = int(np.clip(idx_plus, 0, len(frames) - 1))
    return {
        "transition_type": "boundary",
        "event": None,
        "idx_c": None,
        "idx_minus": idx_star,
        "idx_plus": idx_plus,
        "x_minus": states[idx_star].tolist(),
        "x_plus": states[idx_plus].tolist(),
        "rho_imp": 0.0,
        "harm_bin": 0,
        "risk_score": float(scores[idx_star]),
    }


def extract_transition(rollout: Mapping[str, Any], contact_kwargs: Optional[Dict[str, Any]] = None, boundary_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if first_collision_event(rollout) is not None:
        return contact_transition(rollout, **(contact_kwargs or {}))
    return boundary_transition(rollout, **(boundary_kwargs or {}))

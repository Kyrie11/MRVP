from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from mrvp.data.schema import CTX_DIM, MECH_DIM, MECH_SLICES, SIDE_NAMES, SIDE_TO_ID, STATE_KEYS


def _angle_wrap(x: float) -> float:
    return math.atan2(math.sin(x), math.cos(x))


def _body_frame_vector(world_vec: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return rot @ world_vec.astype(np.float32)


def side_from_body_xy(xy: np.ndarray) -> str:
    x, y = float(xy[0]), float(xy[1])
    ang = math.atan2(y, x)
    if abs(ang) < math.radians(35):
        return "front"
    if abs(abs(ang) - math.pi) < math.radians(35):
        return "rear"
    if math.radians(55) < ang < math.radians(125):
        return "left"
    if -math.radians(125) < ang < -math.radians(55):
        return "right"
    return "oblique"


def _nearest_actor(frame: Mapping[str, Any], point_xy: np.ndarray) -> Optional[Mapping[str, Any]]:
    best = None
    best_d = float("inf")
    for a in frame.get("actors", []):
        xy = np.asarray([a.get("x", 0.0), a.get("y", 0.0)], dtype=np.float32)
        d = float(np.linalg.norm(xy - point_xy))
        if d < best_d:
            best = a
            best_d = d
    return best


def h_ctx_from_rollout(rollout: Mapping[str, Any], transition: Mapping[str, Any]) -> np.ndarray:
    frames = rollout.get("frames", [])
    idx = int(transition.get("idx_minus", 0) or 0)
    frame = frames[min(max(idx, 0), len(frames) - 1)] if frames else {"ego": {}, "actors": [], "margins": {}}
    ego = frame.get("ego", {})
    margins = frame.get("margins", {})
    h = np.zeros(CTX_DIM, dtype=np.float32)
    h[0] = 0.0
    h[1] = float(ego.get("route_heading_error", 0.0))
    h[2] = float(ego.get("lane_width", 0.0))
    lateral = float(ego.get("lateral_offset", 0.0))
    lane_width = float(ego.get("lane_width", 0.0))
    h[3] = lane_width / 2.0 + lateral
    h[4] = lane_width / 2.0 - lateral
    h[5] = float(margins.get("secondary", 50.0))
    h[6] = 8.0
    h[7] = float(margins.get("secondary", 50.0))
    speed = float(ego.get("speed", 0.0))
    h[8] = h[7] / max(speed, 0.1)
    h[9] = float(margins.get("return", 0.0))
    h[10] = float(len(frame.get("actors", [])) / 30.0)
    h[11] = float(rollout.get("speed_limit", 13.9))
    h[12] = float(rollout.get("friction", rollout.get("degradation", {}).get("friction", 0.9)))
    h[13] = float(rollout.get("weather_wetness", 0.0))
    h[14] = 0.0
    h[15] = lane_width
    h[16] = 0.0
    h[17] = -lateral
    h[18] = 30.0
    h[19] = -lateral
    h[20] = 0.0
    h[21] = float(rollout.get("occlusion_score", 0.0))
    h[22] = 1.0
    h[23] = float(np.sign(-lateral))
    h[24] = float(rollout.get("town_id_norm", 0.0))
    h[25] = float(rollout.get("family_id_norm", 0.0))
    h[26] = 1.0 if transition.get("transition_type") == "contact" else 0.0
    h[27] = float(rollout.get("degradation", {}).get("damage_class", 0.0)) / 3.0
    h[28] = float(rollout.get("trigger_time", 0.0)) / 10.0
    return h


def d_deg_from_rollout(rollout: Mapping[str, Any]) -> np.ndarray:
    d = rollout.get("degradation", {}) or {}
    return np.asarray(
        [
            float(d.get("steering_scale", 1.0)),
            float(d.get("brake_scale", 1.0)),
            float(d.get("throttle_scale", 1.0)),
            float(d.get("delay", 0.05)),
            float(d.get("friction", rollout.get("friction", 0.9))),
            float(d.get("damage_class", 0.0)) / 3.0,
        ],
        dtype=np.float32,
    )


def compute_mechanism_labels(rollout: Mapping[str, Any], transition: Mapping[str, Any]) -> Dict[str, Any]:
    frames = rollout.get("frames", [])
    if not frames:
        raise ValueError("Rollout has no frames.")
    idx_minus = int(transition.get("idx_minus", 0) or 0)
    idx_plus = int(transition.get("idx_plus", idx_minus) or idx_minus)
    frame_minus = frames[min(max(idx_minus, 0), len(frames) - 1)]
    frame_plus = frames[min(max(idx_plus, 0), len(frames) - 1)]
    ego = frame_minus.get("ego", {})
    ego_xy = np.asarray([ego.get("x", 0.0), ego.get("y", 0.0)], dtype=np.float32)
    yaw = float(ego.get("yaw", 0.0))
    x_minus = np.asarray(transition.get("x_minus"), dtype=np.float32)
    x_plus = np.asarray(transition.get("x_plus"), dtype=np.float32)
    z = np.zeros(MECH_DIM, dtype=np.float32)
    contact = transition.get("transition_type") == "contact"
    z[MECH_SLICES["contact_flag"]] = 1.0 if contact else 0.0
    normal_world = np.asarray([1.0, 0.0], dtype=np.float32)
    side = "oblique"
    rel_heading = 0.0
    rel_speed = 0.0
    if contact:
        event = transition.get("event") or {}
        impulse = np.asarray(event.get("normal_impulse", [0.0, 0.0, 0.0]), dtype=np.float32)[:2]
        if np.linalg.norm(impulse) > 1e-6:
            normal_world = -impulse / (np.linalg.norm(impulse) + 1e-6)
        contact_xy = np.asarray(event.get("contact_point", ego_xy.tolist())[:2], dtype=np.float32)
        body_contact = _body_frame_vector(contact_xy - ego_xy, yaw)
        if np.linalg.norm(body_contact) < 1e-4:
            body_contact = _body_frame_vector(normal_world, yaw)
        side = side_from_body_xy(body_contact)
        other_id = int(event.get("other_actor_id", -1))
        other = None
        for a in frame_minus.get("actors", []):
            if int(a.get("id", -2)) == other_id:
                other = a
                break
        if other is None:
            other = _nearest_actor(frame_minus, ego_xy)
        if other is not None:
            rel_heading = _angle_wrap(float(ego.get("yaw", 0.0)) - float(other.get("yaw", 0.0)))
            ev = np.asarray([ego.get("vx", 0.0), ego.get("vy", 0.0)], dtype=np.float32)
            ov = np.asarray([other.get("vx", 0.0), other.get("vy", 0.0)], dtype=np.float32)
            rel_speed = float(np.linalg.norm(ev - ov))
    else:
        # Boundary side from lateral offset sign; nearest boundary normal points back to lane center.
        lateral = float(ego.get("lateral_offset", 0.0))
        boundary_side = "left" if lateral > 0 else "right"
        side = boundary_side
        normal_world = _body_frame_vector(np.asarray([0.0, -np.sign(lateral) if lateral != 0 else 1.0], dtype=np.float32), -yaw)
    side_id = SIDE_TO_ID.get(side, SIDE_TO_ID["oblique"])
    z[1 + side_id] = 1.0
    normal_body = _body_frame_vector(normal_world, yaw)
    normal_body = normal_body / (np.linalg.norm(normal_body) + 1e-6)
    z[MECH_SLICES["normal_xy"]] = normal_body[:2]
    if contact:
        # Negative clearance/overlap proxy from impulse/harm.
        z[MECH_SLICES["overlap_clearance"]] = -0.1 * float(transition.get("rho_imp", 0.0))
    else:
        z[MECH_SLICES["overlap_clearance"]] = float(frame_minus.get("margins", {}).get("road", ego.get("road_clearance", 0.0)))
    z[MECH_SLICES["relative_heading"]] = rel_heading
    z[MECH_SLICES["relative_speed"]] = rel_speed
    reset = np.zeros(7, dtype=np.float32)
    if x_minus.size >= 8 and x_plus.size >= 8:
        reset[:] = [
            x_plus[3] - x_minus[3],
            x_plus[4] - x_minus[4],
            x_plus[5] - x_minus[5],
            _angle_wrap(float(x_plus[2] - x_minus[2])),
            x_plus[6] - x_minus[6],
            x_plus[7] - x_minus[7],
            x_plus[1] - x_minus[1],
        ]
    z[MECH_SLICES["reset"]] = reset
    h_ctx = h_ctx_from_rollout(rollout, transition)
    # Affordance: width, road clearance, return length, hazard clearance, escape dir, route availability.
    z[MECH_SLICES["affordance"]] = [h_ctx[2], h_ctx[4] if side == "right" else h_ctx[3], h_ctx[9], h_ctx[7], h_ctx[23], h_ctx[22]]
    d = d_deg_from_rollout(rollout)
    z[MECH_SLICES["degradation"]] = [d[0], d[1], d[3], d[4], d[5]]
    z[MECH_SLICES["uncertainty"]] = [float(rollout.get("transition_variance", 0.05)), 0.2 if contact else 0.05, h_ctx[10]]
    group = {
        "contact_type": "contact" if contact else "boundary",
        "contact_side": side,
        "boundary_side": "left" if float(ego.get("lateral_offset", 0.0)) > 0 else "right",
        "friction_bin": "low" if d[4] < 0.55 else ("mid" if d[4] < 0.85 else "high"),
        "damage_class": str(int(round(float(d[5]) * 3))),
        "density_bin": "dense" if h_ctx[10] > 0.65 else "normal",
        "town": str(rollout.get("town", "unknown")),
        "family": str(rollout.get("family", "unknown")),
    }
    return {"z_mech": z.tolist(), "h_ctx": h_ctx.tolist(), "d_deg": d.tolist(), "calib_group": group}

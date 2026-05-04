"""MRVP code-level schema.

The field names mirror the paper appendix: one JSONL row equals one
candidate-action rollout from one root scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

BottleneckName = str

# Keep these names lowercase in JSON/CSV outputs.
BOTTLE_NECKS = ["sec", "road", "stab", "ctrl", "return"]
BOTTLE_NECK_TO_INDEX = {name: i for i, name in enumerate(BOTTLE_NECKS)}

# Default vector sizes used by the implemented PyTorch modules.
STATE_DIM = 10
DEG_DIM = 6
MECH_DIM = 32
HIST_LEN = 10
MAX_AGENTS = 16
ACTOR_FEAT_DIM = 9
CTX_DIM = 32
ACTION_DIM = 4
NUM_ACTIONS = 8

# Emergency action library from Appendix: hard brake, brake-left, brake-right,
# maintain, mild left/right, boundary-side, corridor-side.
ACTION_NAMES = [
    "hard_brake",
    "brake_left",
    "brake_right",
    "maintain",
    "mild_left",
    "mild_right",
    "boundary_side_steer",
    "corridor_side_steer",
]
ACTION_NAME_TO_ID = {name: i for i, name in enumerate(ACTION_NAMES)}

# steer, throttle, brake, open-loop duration seconds. Steering sign follows
# CARLA VehicleControl: negative left, positive right.
ACTION_VECTORS = np.asarray(
    [
        [0.00, 0.00, 1.00, 0.70],
        [-0.65, 0.00, 0.85, 0.70],
        [0.65, 0.00, 0.85, 0.70],
        [0.00, 0.20, 0.00, 0.70],
        [-0.35, 0.10, 0.10, 0.70],
        [0.35, 0.10, 0.10, 0.70],
        [0.55, 0.00, 0.40, 0.70],
        [-0.55, 0.00, 0.40, 0.70],
    ],
    dtype=np.float32,
)

# State layout used by transition extraction and toy teachers.
STATE_KEYS = [
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "speed",
    "yaw_rate",
    "beta",
    "steer",
    "t",
]

# Mechanism vector layout. Categorical side is one-hot front/rear/left/right/oblique.
MECH_SLICES = {
    "contact_flag": slice(0, 1),
    "side_onehot": slice(1, 6),
    "normal_xy": slice(6, 8),
    "overlap_clearance": slice(8, 9),
    "relative_heading": slice(9, 10),
    "relative_speed": slice(10, 11),
    "reset": slice(11, 18),
    "affordance": slice(18, 24),
    "degradation": slice(24, 29),
    "uncertainty": slice(29, 32),
}
SIDE_NAMES = ["front", "rear", "left", "right", "oblique"]
SIDE_TO_ID = {name: i for i, name in enumerate(SIDE_NAMES)}

CTX_KEYS = [
    "route_curvature",
    "route_heading_error",
    "drivable_width",
    "left_clearance",
    "right_clearance",
    "front_clearance",
    "rear_clearance",
    "secondary_hazard_clearance",
    "secondary_ttc",
    "return_corridor_length",
    "actor_density",
    "speed_limit",
    "friction",
    "weather_wetness",
    "map_curvature",
    "lane_width",
    "ego_to_route_dx",
    "ego_to_route_dy",
    "target_refuge_dx",
    "target_refuge_dy",
    "traffic_light_state",
    "occlusion_score",
    "route_available",
    "boundary_side",  # -1 left, +1 right, 0 unknown
    "town_id_norm",
    "family_id_norm",
    "contact_prior",
    "damage_prior",
    "trigger_time_norm",
    "reserved_0",
    "reserved_1",
    "reserved_2",
]


@dataclass(frozen=True)
class SchemaDims:
    state_dim: int = STATE_DIM
    deg_dim: int = DEG_DIM
    mech_dim: int = MECH_DIM
    hist_len: int = HIST_LEN
    max_agents: int = MAX_AGENTS
    actor_feat_dim: int = ACTOR_FEAT_DIM
    ctx_dim: int = CTX_DIM
    action_dim: int = ACTION_DIM
    num_actions: int = NUM_ACTIONS
    num_bottlenecks: int = len(BOTTLE_NECKS)


def ensure_1d_float(value: Any, dim: int, fill: float = 0.0) -> np.ndarray:
    """Convert nested list/dict/scalar to a fixed-size float32 vector."""
    if value is None:
        arr = np.empty((0,), dtype=np.float32)
    elif isinstance(value, Mapping):
        arr = np.asarray([float(value.get(k, fill)) for k in CTX_KEYS[:dim]], dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.full((dim,), fill, dtype=np.float32)
    n = min(dim, arr.shape[0])
    if n:
        out[:n] = arr[:n]
    return out


def ensure_hist(value: Any, hist_len: int = HIST_LEN, max_agents: int = MAX_AGENTS, feat_dim: int = ACTOR_FEAT_DIM) -> np.ndarray:
    """Return a fixed [T, A, F] history tensor."""
    out = np.zeros((hist_len, max_agents, feat_dim), dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    t = min(hist_len, arr.shape[0])
    a = min(max_agents, arr.shape[1])
    f = min(feat_dim, arr.shape[2])
    out[-t:, :a, :f] = arr[-t:, :a, :f]
    return out


def action_id_from_any(action_id: Any, action_name: Any = None) -> int:
    if action_id is not None:
        if isinstance(action_id, str):
            return int(ACTION_NAME_TO_ID.get(action_id, 0))
        return int(action_id) % NUM_ACTIONS
    if action_name is not None:
        return int(ACTION_NAME_TO_ID.get(str(action_name), 0))
    return 0


def action_vec_from_id(action_id: int) -> np.ndarray:
    return ACTION_VECTORS[int(action_id) % len(ACTION_VECTORS)].copy()


def harm_bin_from_rho(rho_imp: float, thresholds: Sequence[float] = (0.5, 2.0, 4.0, 7.0, 11.0)) -> int:
    """Monotone first-impact harm bin B(rho_imp)."""
    return int(np.searchsorted(np.asarray(thresholds, dtype=np.float32), float(rho_imp), side="right"))


def group_dict_to_key(group: Mapping[str, Any] | None, level: str = "full") -> str:
    """Calibration grouping hierarchy.

    full: contact type + side + friction + damage + density + town
    medium: contact type + side + friction
    coarse: contact/non-contact
    global: global
    """
    if not group:
        return "global"
    contact = str(group.get("contact_type", group.get("contact", "unknown")))
    side = str(group.get("contact_side", group.get("boundary_side", "unknown")))
    friction = str(group.get("friction_bin", "unknown"))
    damage = str(group.get("damage_class", "unknown"))
    density = str(group.get("density_bin", "unknown"))
    town = str(group.get("town", group.get("map", "unknown")))
    if level == "full":
        return "|".join([contact, side, friction, damage, density, town])
    if level == "medium":
        return "|".join([contact, side, friction])
    if level == "coarse":
        return contact
    return "global"


def row_group(row: Mapping[str, Any]) -> Dict[str, Any]:
    group = row.get("calib_group") or row.get("z_group") or row.get("group") or {}
    if not isinstance(group, Mapping):
        return {"contact_type": str(group)}
    return dict(group)


def row_to_numpy(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> Dict[str, np.ndarray | int | float | str | Dict[str, Any]]:
    action_id = action_id_from_any(row.get("action_id"), row.get("action_name"))
    r_star = ensure_1d_float(row.get("r_star"), dims.num_bottlenecks)
    s_star = float(row.get("s_star", np.min(r_star) if r_star.size else 0.0))
    b_star = int(row.get("b_star", int(np.argmin(r_star)) if r_star.size else 0))
    return {
        "root_id": str(row.get("root_id", "0")),
        "split": str(row.get("split", "train")),
        "family": str(row.get("family", "unknown")),
        "action_id": action_id,
        "action_vec": ensure_1d_float(row.get("action_vec", action_vec_from_id(action_id)), dims.action_dim),
        "o_hist": ensure_hist(row.get("o_hist"), dims.hist_len, dims.max_agents, dims.actor_feat_dim),
        "h_ctx": ensure_1d_float(row.get("h_ctx"), dims.ctx_dim),
        "rho_imp": float(row.get("rho_imp", 0.0)),
        "harm_bin": int(row.get("harm_bin", harm_bin_from_rho(float(row.get("rho_imp", 0.0))))),
        "x_minus": ensure_1d_float(row.get("x_minus"), dims.state_dim),
        "x_plus": ensure_1d_float(row.get("x_plus"), dims.state_dim),
        "d_deg": ensure_1d_float(row.get("d_deg"), dims.deg_dim),
        "z_mech": ensure_1d_float(row.get("z_mech"), dims.mech_dim),
        "r_star": r_star,
        "b_star": b_star,
        "s_star": s_star,
        "calib_group": row_group(row),
    }

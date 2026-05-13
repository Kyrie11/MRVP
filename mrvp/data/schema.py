"""MRVP code-level schema.

The current paper version treats one JSONL row as one candidate emergency action
rollout from one root scenario.  The preferred field names follow Appendix A:
``event_type``, ``event_time``, ``deg``, ``world_plus``, ``teacher_u``,
``teacher_traj`` and ``m_star``.  The loader remains backward compatible with
older rows that used ``d_deg``, ``z_mech`` and ``r_star``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

BottleneckName = str

# Keep these names lowercase in JSON/CSV outputs.
BOTTLE_NECKS = ["sec", "road", "stab", "ctrl", "return"]
BOTTLE_NECK_TO_INDEX = {name: i for i, name in enumerate(BOTTLE_NECKS)}

# Recovery-critical event vocabulary from the revised paper.
EVENT_TYPES = ["none", "contact", "boundary", "stability", "control"]
EVENT_TYPE_TO_ID = {name: i for i, name in enumerate(EVENT_TYPES)}
ID_TO_EVENT_TYPE = {i: name for name, i in EVENT_TYPE_TO_ID.items()}

# Default vector sizes used by the compact PyTorch implementation.
# State layout follows the paper appendix:
# [p_x, p_y, psi, v_x, v_y, yaw_rate, a_x, a_y, beta, delta, F_b, F_x]
STATE_DIM = 12
DEG_DIM = 6
MECH_DIM = 32  # optional audit fields; not consumed by RPN in the revised path
HIST_LEN = 10
MAX_AGENTS = 16
ACTOR_FEAT_DIM = 9
CTX_DIM = 32
ACTION_DIM = 4
NUM_ACTIONS = 8
WORLD_DIM = 64
TOKEN_COUNT = 16
TOKEN_DIM = 32
CONTROL_DIM = 3
RECOVERY_HORIZON = 30
STRATEGY_COUNT = 6

# Emergency action library from the revised appendix.  The steering sign is the
# abstract paper convention: positive means left and negative means right.  When
# driving a simulator with a different convention, convert at the adapter layer.
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
ACTION_VECTORS = np.asarray(
    [
        [0.00, 0.00, 0.90, 1.00],
        [0.35, 0.00, 0.70, 1.00],
        [-0.35, 0.00, 0.70, 1.00],
        [0.00, 0.20, 0.00, 1.00],
        [0.18, 0.10, 0.30, 1.00],
        [-0.18, 0.10, 0.30, 1.00],
        [0.45, 0.00, 0.50, 1.00],
        [-0.45, 0.00, 0.50, 1.00],
    ],
    dtype=np.float32,
)

STATE_KEYS = ["p_x", "p_y", "psi", "v_x", "v_y", "yaw_rate", "a_x", "a_y", "beta", "delta", "F_b", "F_x"]
DEG_KEYS = ["steer_scale", "brake_scale", "throttle_scale", "delay", "friction", "damage_class"]

# Optional human-readable audit vector layout retained for probes and debugging.
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

WORLD_KEYS = [
    "drivable_crop",
    "future_occupancy",
    "actor_flow",
    "reachable_mask",
    "goal_mask",
    "static_obstacles",
    "dynamic_actors",
    "route_mask",
    "refuge_mask",
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
    world_dim: int = WORLD_DIM
    token_count: int = TOKEN_COUNT
    token_dim: int = TOKEN_DIM
    control_dim: int = CONTROL_DIM
    recovery_horizon: int = RECOVERY_HORIZON


def _flatten_numeric(value: Any) -> np.ndarray:
    """Flatten numeric content from nested lists/dicts into float32."""
    vals: List[float] = []

    def rec(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, Mapping):
            # Stable order for known world keys, then any extras.
            keys = [k for k in WORLD_KEYS if k in x] + sorted(k for k in x.keys() if k not in WORLD_KEYS)
            for k in keys:
                rec(x[k])
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                rec(item)
            return
        arr = np.asarray(x)
        if arr.ndim == 0:
            try:
                vals.append(float(arr))
            except Exception:
                return
        else:
            for y in arr.reshape(-1):
                try:
                    vals.append(float(y))
                except Exception:
                    pass

    rec(value)
    return np.asarray(vals, dtype=np.float32)


def ensure_vector(value: Any, dim: int, fill: float = 0.0, keys: Sequence[str] | None = None) -> np.ndarray:
    """Convert nested list/dict/scalar to a fixed-size float32 vector."""
    if value is None:
        arr = np.empty((0,), dtype=np.float32)
    elif isinstance(value, Mapping) and keys is not None:
        arr = np.asarray([float(value.get(k, fill)) for k in keys[:dim]], dtype=np.float32)
    else:
        arr = _flatten_numeric(value)
    out = np.full((dim,), fill, dtype=np.float32)
    n = min(dim, arr.shape[0])
    if n:
        out[:n] = arr[:n]
    return out


# Backward-compatible name used by older modules.
def ensure_1d_float(value: Any, dim: int, fill: float = 0.0) -> np.ndarray:
    return ensure_vector(value, dim, fill=fill)


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


def ensure_matrix(value: Any, rows: int, cols: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((rows, cols), fill, dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    r = min(rows, arr.shape[0])
    c = min(cols, arr.shape[1])
    if r and c:
        out[:r, :c] = arr[:r, :c]
    return out


def ensure_tokens(value: Any, token_count: int = TOKEN_COUNT, token_dim: int = TOKEN_DIM) -> np.ndarray:
    out = np.zeros((token_count, token_dim), dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        k = min(token_count, arr.shape[0])
        d = min(token_dim, arr.shape[1])
        out[:k, :d] = arr[:k, :d]
        return out
    flat = _flatten_numeric(value)
    n = min(flat.size, token_count * token_dim)
    if n:
        out.reshape(-1)[:n] = flat[:n]
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


def canonical_event_type(value: Any, fallback: str = "none") -> str:
    if value is None:
        value = fallback
    s = str(value).strip().lower()
    # Older rows used contact/non_contact in calibration groups.
    aliases = {
        "non_contact": "none",
        "no_contact": "none",
        "offroad": "boundary",
        "road": "boundary",
        "secondary": "contact",
    }
    s = aliases.get(s, s)
    return s if s in EVENT_TYPE_TO_ID else "none"


def event_type_id(value: Any, fallback: str = "none") -> int:
    return EVENT_TYPE_TO_ID[canonical_event_type(value, fallback)]


def audit_vector_from_row(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> np.ndarray:
    """Return optional audit/mechanism fields as a fixed vector.

    New rows should put human-readable diagnostics in ``audit_mech``.  Older
    rows can still provide ``z_mech`` directly.  This vector is used for probes,
    not as the RPN input in the revised model path.
    """
    direct = row.get("audit_mech", row.get("z_mech"))
    if direct is None:
        direct = {}
    if not isinstance(direct, Mapping):
        return ensure_vector(direct, dims.mech_dim)
    z = np.zeros((dims.mech_dim,), dtype=np.float32)
    ev = canonical_event_type(row.get("event_type", direct.get("event_type")), fallback=str(direct.get("contact_type", "none")))
    z[0] = 1.0 if ev == "contact" else 0.0
    side = str(direct.get("event_side", direct.get("contact_side", direct.get("boundary_side", row.get("contact_side", "oblique"))))).lower()
    side_id = SIDE_TO_ID.get(side, SIDE_TO_ID["oblique"])
    z[1 + side_id] = 1.0
    z[6:8] = ensure_vector(direct.get("normal_xy", direct.get("normal", direct.get("contact_normal"))), 2)
    z[8] = float(direct.get("overlap_clearance", direct.get("signed_clearance", direct.get("clearance", 0.0))) or 0.0)
    z[9] = float(direct.get("relative_heading", 0.0) or 0.0)
    z[10] = float(direct.get("relative_speed", direct.get("impulse_proxy", row.get("rho_imp", 0.0))) or 0.0)
    z[11:18] = ensure_vector(direct.get("reset", direct.get("body_frame_reset")), 7)
    z[18:24] = ensure_vector(direct.get("affordance", direct.get("corridor", direct.get("recovery_affordance"))), 6)
    z[24:29] = ensure_vector(direct.get("degradation", row.get("deg", row.get("d_deg"))), 5)
    z[29:32] = ensure_vector(direct.get("uncertainty", direct.get("timing_uncertainty", direct.get("token_uncertainty"))), 3)
    return z


def _flat_float_set(value: Any, precision: int = 6) -> set[float]:
    vals = _flatten_numeric(value)
    return {round(float(x), precision) for x in vals.reshape(-1)}


def validate_row_no_leakage(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> List[str]:
    """Return schema-level leakage warnings for one raw JSON row.

    The checks are intentionally conservative.  They catch the failure modes that
    invalidate the MRVP claim: audit vectors masquerading as learned tokens and
    recovery-label values being copied into ``world_plus``.  A warning means the
    row should be inspected; it is not always a proof of leakage.
    """
    warnings: List[str] = []
    token_source = row.get("event_tokens", row.get("tokens", row.get("z_tokens", None)))
    if token_source is not None:
        tok = ensure_tokens(token_source, dims.token_count, dims.token_dim).reshape(-1)
        audit = audit_vector_from_row(row, dims)
        n = min(tok.size, audit.size)
        if n and np.allclose(tok[:n], audit[:n], atol=1e-6, rtol=1e-6):
            warnings.append("event_tokens_equal_audit_prefix")
    world = row.get("world_plus", row.get("w_plus", row.get("bev_world")))
    if world is not None:
        wset = _flat_float_set(world)
        for key in ("m_star", "r_star"):
            if key in row:
                for v in _flatten_numeric(row.get(key)):
                    if round(float(v), 6) in wset:
                        warnings.append(f"world_plus_contains_{key}_value")
                        break
        if "s_star" in row and round(float(row.get("s_star", 0.0)), 6) in wset:
            warnings.append("world_plus_contains_s_star_value")
    return warnings


def group_dict_to_key(group: Mapping[str, Any] | None, level: str = "full") -> str:
    """Calibration grouping hierarchy from the paper appendix."""
    if not group:
        return "global"
    event = canonical_event_type(group.get("event_type", group.get("contact_type", group.get("contact", "none"))))
    side = str(group.get("contact_side", group.get("boundary_side", "unknown")))
    friction = str(group.get("friction_bin", "unknown"))
    damage = str(group.get("damage_class", "unknown"))
    density = str(group.get("density_bin", "unknown"))
    town = str(group.get("town", group.get("map", "unknown")))
    if level == "full":
        return "|".join([event, side, friction, damage, density, town])
    if level == "medium":
        return "|".join([event, side, friction])
    if level == "coarse":
        return event
    return "global"


def row_group(row: Mapping[str, Any]) -> Dict[str, Any]:
    group = row.get("calib_group") or row.get("z_group") or row.get("group") or {}
    if not isinstance(group, Mapping):
        group = {"event_type": str(group)}
    group = dict(group)
    if "event_type" not in group:
        group["event_type"] = canonical_event_type(row.get("event_type"), fallback=str(group.get("contact_type", "none")))
    return group


def row_to_numpy(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> Dict[str, np.ndarray | int | float | str | Dict[str, Any]]:
    action_id = action_id_from_any(row.get("action_id"), row.get("action_name"))
    m_value = row.get("m_star", row.get("r_star"))
    m_star = ensure_vector(m_value, dims.num_bottlenecks)
    s_star = float(row.get("s_star", np.min(m_star) if m_star.size else 0.0))
    b_star = int(row.get("b_star", int(np.argmin(m_star)) if m_star.size else 0))
    deg = ensure_vector(row.get("deg", row.get("d_deg")), dims.deg_dim, keys=DEG_KEYS)
    audit = audit_vector_from_row(row, dims)
    token_source = row.get("event_tokens", row.get("tokens", row.get("z_tokens", None)))
    has_event_tokens = token_source is not None
    event_name = canonical_event_type(row.get("event_type"), fallback=str(row_group(row).get("event_type", "none")))
    x_minus = ensure_vector(row.get("x_minus"), dims.state_dim, keys=STATE_KEYS)
    x_t = ensure_vector(row.get("x_t", row.get("ego_state", x_minus)), dims.state_dim, keys=STATE_KEYS)
    x_plus = ensure_vector(row.get("x_plus"), dims.state_dim, keys=STATE_KEYS)
    world_plus = ensure_vector(row.get("world_plus", row.get("w_plus", row.get("bev_world"))), dims.world_dim)
    teacher_u = ensure_matrix(row.get("teacher_u", row.get("recovery_controls")), dims.recovery_horizon, dims.control_dim)
    teacher_traj = ensure_matrix(row.get("teacher_traj", row.get("recovery_traj")), dims.recovery_horizon + 1, dims.state_dim)
    return {
        "root_id": str(row.get("root_id", "0")),
        "split": str(row.get("split", "train")),
        "family": str(row.get("family", row.get("scenario_family", "unknown"))),
        "action_id": action_id,
        "action_vec": ensure_vector(row.get("action_vec", action_vec_from_id(action_id)), dims.action_dim),
        "o_hist": ensure_hist(row.get("o_hist"), dims.hist_len, dims.max_agents, dims.actor_feat_dim),
        "h_ctx": ensure_vector(row.get("h_ctx"), dims.ctx_dim, keys=CTX_KEYS),
        "x_t": x_t,
        "rho_imp": float(row.get("rho_imp", 0.0)),
        "harm_bin": int(row.get("harm_bin", harm_bin_from_rho(float(row.get("rho_imp", 0.0))))),
        "event_type": event_name,
        "event_type_id": event_type_id(event_name),
        "event_time": float(row.get("event_time", row.get("tau", row.get("tau_star", 0.0)))) ,
        "x_minus": x_minus,
        "x_plus": x_plus,
        "deg": deg,
        "d_deg": deg,  # backward-compatible alias
        "event_tokens": ensure_tokens(token_source if has_event_tokens else None, dims.token_count, dims.token_dim),
        "has_event_tokens": np.asarray(float(has_event_tokens), dtype=np.float32),
        "world_plus": world_plus,
        "audit_mech": audit,
        "z_mech": audit,
        "teacher_u": teacher_u,
        "teacher_traj": teacher_traj,
        "m_star": m_star,
        "r_star": m_star,  # backward-compatible alias
        "b_star": b_star,
        "s_star": s_star,
        "calib_group": row_group(row),
    }

"""MRVP reset-problem centered schema utilities.

A JSONL row represents one candidate emergency prefix under a shared root scene.
The primary fields follow MRVP: Motion-Reset Viability Programming:
``reset_state/reset_time/degradation/recovery_world/reset_slots``.  Older
``x_plus/event_time/deg/world_plus/event_tokens`` rows are normalized through
explicit aliases, but old event labels remain diagnostics rather than core model
inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

BottleneckName = str

# Certificate profile order used by RPFN.  ``sec`` is the dynamic-actor/secondary
# exposure margin; ``goal`` is the route/refuge recovery margin.
BOTTLE_NECKS = ["sec", "road", "stab", "ctrl", "goal"]
BOTTLE_NECK_TO_INDEX = {name: i for i, name in enumerate(BOTTLE_NECKS)}

# Diagnostic-only audit vocabulary.  It may be logged or probed, but selection
# and RPFN never consume event_type as a main-path input.
EVENT_TYPES = ["none", "contact", "boundary", "stability", "control"]
EVENT_TYPE_TO_ID = {name: i for i, name in enumerate(EVENT_TYPES)}
ID_TO_EVENT_TYPE = {i: name for name, i in EVENT_TYPE_TO_ID.items()}

# State layout: [p_x, p_y, psi, v_x, v_y, yaw_rate, a_x, a_y, beta, delta, F_b, F_x]
STATE_DIM = 12
DEG_DIM = 6
MECH_DIM = 32
HIST_LEN = 10
MAX_AGENTS = 16
ACTOR_FEAT_DIM = 9
CTX_DIM = 32
ACTION_DIM = 4
NUM_ACTIONS = 8
RESET_SLOT_COUNT = 16
RESET_SLOT_DIM = 64
RECOVERY_WORLD_DIM = 96
PROGRAM_COUNT = 6
CONTROL_DIM = 3
RECOVERY_HORIZON = 30
CERTIFICATE_DIM = len(BOTTLE_NECKS)
RESET_UNCERTAINTY_DIM = 3

# Backward-compatible constant aliases used by legacy scripts/checkpoints.
TOKEN_COUNT = RESET_SLOT_COUNT
TOKEN_DIM = RESET_SLOT_DIM
WORLD_DIM = RECOVERY_WORLD_DIM
STRATEGY_COUNT = PROGRAM_COUNT

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
SIDE_NAMES = ["front", "rear", "left", "right", "oblique"]
SIDE_TO_ID = {name: i for i, name in enumerate(SIDE_NAMES)}

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
    "boundary_side",
    "town_id_norm",
    "family_id_norm",
    "contact_prior",
    "damage_prior",
    "trigger_time_norm",
    "reserved_0",
    "reserved_1",
    "reserved_2",
]

RECOVERY_WORLD_KEYS = ["affordance", "occupancy", "goal", "actor_response"]
# Legacy flatten order.  Kept after the new keys so old rows remain stable.
WORLD_KEYS = RECOVERY_WORLD_KEYS + [
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
    num_bottlenecks: int = CERTIFICATE_DIM
    reset_slot_count: int = RESET_SLOT_COUNT
    reset_slot_dim: int = RESET_SLOT_DIM
    recovery_world_dim: int = RECOVERY_WORLD_DIM
    program_count: int = PROGRAM_COUNT
    control_dim: int = CONTROL_DIM
    recovery_horizon: int = RECOVERY_HORIZON
    reset_uncertainty_dim: int = RESET_UNCERTAINTY_DIM

    # Legacy dataclass field aliases.  Existing scripts may still pass these.
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        mech_dim: int = MECH_DIM,
        hist_len: int = HIST_LEN,
        max_agents: int = MAX_AGENTS,
        actor_feat_dim: int = ACTOR_FEAT_DIM,
        ctx_dim: int = CTX_DIM,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
        num_bottlenecks: int = CERTIFICATE_DIM,
        reset_slot_count: int = RESET_SLOT_COUNT,
        reset_slot_dim: int = RESET_SLOT_DIM,
        recovery_world_dim: int = RECOVERY_WORLD_DIM,
        program_count: int = PROGRAM_COUNT,
        control_dim: int = CONTROL_DIM,
        recovery_horizon: int = RECOVERY_HORIZON,
        reset_uncertainty_dim: int = RESET_UNCERTAINTY_DIM,
        token_count: int | None = None,
        token_dim: int | None = None,
        world_dim: int | None = None,
        strategy_count: int | None = None,
    ) -> None:
        object.__setattr__(self, "state_dim", state_dim)
        object.__setattr__(self, "deg_dim", deg_dim)
        object.__setattr__(self, "mech_dim", mech_dim)
        object.__setattr__(self, "hist_len", hist_len)
        object.__setattr__(self, "max_agents", max_agents)
        object.__setattr__(self, "actor_feat_dim", actor_feat_dim)
        object.__setattr__(self, "ctx_dim", ctx_dim)
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "num_actions", num_actions)
        object.__setattr__(self, "num_bottlenecks", num_bottlenecks)
        object.__setattr__(self, "reset_slot_count", reset_slot_count if token_count is None else token_count)
        object.__setattr__(self, "reset_slot_dim", reset_slot_dim if token_dim is None else token_dim)
        object.__setattr__(self, "recovery_world_dim", recovery_world_dim if world_dim is None else world_dim)
        object.__setattr__(self, "program_count", program_count if strategy_count is None else strategy_count)
        object.__setattr__(self, "control_dim", control_dim)
        object.__setattr__(self, "recovery_horizon", recovery_horizon)
        object.__setattr__(self, "reset_uncertainty_dim", reset_uncertainty_dim)

    @property
    def token_count(self) -> int:
        return self.reset_slot_count

    @property
    def token_dim(self) -> int:
        return self.reset_slot_dim

    @property
    def world_dim(self) -> int:
        return self.recovery_world_dim

    @property
    def strategy_count(self) -> int:
        return self.program_count


def _flatten_numeric(value: Any) -> np.ndarray:
    vals: List[float] = []

    def rec(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, Mapping):
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


def ensure_1d_float(value: Any, dim: int, fill: float = 0.0) -> np.ndarray:
    return ensure_vector(value, dim, fill=fill)


def ensure_hist(value: Any, hist_len: int = HIST_LEN, max_agents: int = MAX_AGENTS, feat_dim: int = ACTOR_FEAT_DIM) -> np.ndarray:
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


def ensure_slots(value: Any, slot_count: int = RESET_SLOT_COUNT, slot_dim: int = RESET_SLOT_DIM) -> np.ndarray:
    out = np.zeros((slot_count, slot_dim), dtype=np.float32)
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        k = min(slot_count, arr.shape[0])
        d = min(slot_dim, arr.shape[1])
        out[:k, :d] = arr[:k, :d]
        return out
    flat = _flatten_numeric(value)
    n = min(flat.size, slot_count * slot_dim)
    if n:
        out.reshape(-1)[:n] = flat[:n]
    return out


def ensure_tokens(value: Any, token_count: int = TOKEN_COUNT, token_dim: int = TOKEN_DIM) -> np.ndarray:
    return ensure_slots(value, token_count, token_dim)


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
    return int(np.searchsorted(np.asarray(thresholds, dtype=np.float32), float(rho_imp), side="right"))


def canonical_event_type(value: Any, fallback: str = "none") -> str:
    if value is None:
        value = fallback
    s = str(value).strip().lower()
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


def audit_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = row.get("audit")
    if isinstance(audit, Mapping):
        return audit
    audit_mech = row.get("audit_mech")
    if isinstance(audit_mech, Mapping):
        return audit_mech
    return {}


def audit_vector_from_row(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> np.ndarray:
    direct = row.get("audit_mech", row.get("z_mech"))
    if direct is None:
        direct = row.get("audit", {})
    if not isinstance(direct, Mapping):
        return ensure_vector(direct, dims.mech_dim)
    audit = audit_payload(row) or direct
    z = np.zeros((dims.mech_dim,), dtype=np.float32)
    ev = canonical_event_type(audit.get("event_type", row.get("event_type")), fallback=str(audit.get("contact_type", "none")))
    z[0] = 1.0 if ev == "contact" else 0.0
    side = str(audit.get("event_side", audit.get("contact_side", audit.get("boundary_side", row.get("contact_side", "oblique"))))).lower()
    side_id = SIDE_TO_ID.get(side, SIDE_TO_ID["oblique"])
    z[1 + side_id] = 1.0
    z[6:8] = ensure_vector(audit.get("normal_xy", audit.get("normal", audit.get("contact_normal"))), 2)
    z[8] = float(audit.get("overlap_clearance", audit.get("signed_clearance", audit.get("clearance", 0.0))) or 0.0)
    z[9] = float(audit.get("relative_heading", 0.0) or 0.0)
    z[10] = float(audit.get("relative_speed", audit.get("impulse_proxy", row.get("rho_imp", 0.0))) or 0.0)
    z[11:18] = ensure_vector(audit.get("reset", audit.get("body_frame_reset")), 7)
    z[18:24] = ensure_vector(audit.get("affordance", audit.get("corridor", audit.get("recovery_affordance"))), 6)
    z[24:29] = ensure_vector(audit.get("degradation", row.get("degradation", row.get("deg", row.get("d_deg")))), 5)
    z[29:32] = ensure_vector(audit.get("uncertainty", audit.get("timing_uncertainty", audit.get("token_uncertainty"))), 3)
    return z


def _flat_float_set(value: Any, precision: int = 6) -> set[float]:
    vals = _flatten_numeric(value)
    return {round(float(x), precision) for x in vals.reshape(-1)}


def validate_row_no_leakage(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> List[str]:
    warnings: List[str] = []
    legacy_token_source = row.get("event_tokens", row.get("tokens", row.get("z_tokens", None)))
    if legacy_token_source is not None:
        tok = ensure_slots(legacy_token_source, dims.reset_slot_count, dims.reset_slot_dim).reshape(-1)
        audit = audit_vector_from_row(row, dims)
        n = min(tok.size, audit.size)
        if n and np.allclose(tok[:n], audit[:n], atol=1e-6, rtol=1e-6):
            warnings.append("event_tokens_equal_audit_prefix")
    world = row.get("recovery_world", row.get("world_plus", row.get("w_plus", row.get("bev_world"))))
    if world is not None:
        wset = _flat_float_set(world)
        for key in ("m_star", "r_star"):
            if key in row:
                for v in _flatten_numeric(row.get(key)):
                    if round(float(v), 6) in wset:
                        warnings.append(f"recovery_world_contains_{key}_value")
                        break
        if "s_star" in row and round(float(row.get("s_star", 0.0)), 6) in wset:
            warnings.append("recovery_world_contains_s_star_value")
    return warnings


def group_dict_to_key(group: Mapping[str, Any] | None, level: str = "full") -> str:
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
    audit = audit_payload(row)
    if "event_type" not in group:
        group["event_type"] = canonical_event_type(audit.get("event_type", row.get("event_type")), fallback=str(group.get("contact_type", "none")))
    return group


def _get_reset_state(row: Mapping[str, Any]) -> Any:
    return row.get("reset_state", row.get("r_reset", row.get("x_plus")))


def _get_reset_time(row: Mapping[str, Any]) -> float:
    return float(row.get("reset_time", row.get("event_time", row.get("tau", row.get("tau_star", 0.0)))))


def _get_degradation(row: Mapping[str, Any]) -> Any:
    return row.get("degradation", row.get("deg", row.get("d_deg")))


def _get_recovery_world(row: Mapping[str, Any]) -> Any:
    return row.get("recovery_world", row.get("world_plus", row.get("w_plus", row.get("bev_world"))))


def row_to_numpy(row: Mapping[str, Any], dims: SchemaDims = SchemaDims()) -> Dict[str, Any]:
    action_id = action_id_from_any(row.get("action_id"), row.get("action_name"))
    m_value = row.get("m_star", row.get("r_star"))
    m_star = ensure_vector(m_value, dims.num_bottlenecks)
    s_star = float(row.get("s_star", np.min(m_star) if m_star.size else 0.0))
    b_star = int(row.get("b_star", int(np.argmin(m_star)) if m_star.size else 0))
    degradation = ensure_vector(_get_degradation(row), dims.deg_dim, keys=DEG_KEYS)
    audit = audit_vector_from_row(row, dims)
    audit_info = audit_payload(row)
    event_name = canonical_event_type(audit_info.get("event_type", row.get("event_type")), fallback=str(row_group(row).get("event_type", "none")))
    x_minus = ensure_vector(row.get("x_minus"), dims.state_dim, keys=STATE_KEYS)
    x_t = ensure_vector(row.get("x_t", row.get("ego_state", x_minus)), dims.state_dim, keys=STATE_KEYS)
    reset_state = ensure_vector(_get_reset_state(row), dims.state_dim, keys=STATE_KEYS)
    recovery_world_vec = ensure_vector(_get_recovery_world(row), dims.recovery_world_dim)
    reset_time = _get_reset_time(row)

    reset_slots_source = row.get("reset_slots", None)
    reset_slots_target_source = row.get("reset_slots_target", None)
    legacy_slots_source = row.get("event_tokens", row.get("tokens", row.get("z_tokens", None)))
    has_reset_slots_target = reset_slots_target_source is not None or reset_slots_source is not None
    reset_slots = ensure_slots(reset_slots_source if reset_slots_source is not None else legacy_slots_source, dims.reset_slot_count, dims.reset_slot_dim)
    reset_slots_target = ensure_slots(reset_slots_target_source if reset_slots_target_source is not None else reset_slots_source, dims.reset_slot_count, dims.reset_slot_dim)
    reset_slots_legacy = ensure_slots(legacy_slots_source, dims.reset_slot_count, dims.reset_slot_dim)

    teacher_u = ensure_matrix(row.get("teacher_u", row.get("recovery_controls")), dims.recovery_horizon, dims.control_dim)
    teacher_traj = ensure_matrix(row.get("teacher_traj", row.get("recovery_traj")), dims.recovery_horizon + 1, dims.state_dim)
    reset_uncertainty = ensure_vector(row.get("reset_uncertainty_target", row.get("reset_uncertainty")), dims.reset_uncertainty_dim)
    out: Dict[str, Any] = {
        "root_id": str(row.get("root_id", "0")),
        "split": str(row.get("split", "train")),
        "family": str(row.get("family", row.get("scenario_family", "unknown"))),
        "action_id": action_id,
        "action_name": str(row.get("action_name", ACTION_NAMES[action_id % len(ACTION_NAMES)])),
        "action_vec": ensure_vector(row.get("action_vec", action_vec_from_id(action_id)), dims.action_dim),
        "o_hist": ensure_hist(row.get("o_hist"), dims.hist_len, dims.max_agents, dims.actor_feat_dim),
        "h_ctx": ensure_vector(row.get("h_ctx"), dims.ctx_dim, keys=CTX_KEYS),
        "x_t": x_t,
        "rho_imp": float(row.get("rho_imp", 0.0)),
        "harm_bin": int(row.get("harm_bin", harm_bin_from_rho(float(row.get("rho_imp", 0.0))))),
        "reset_time": reset_time,
        "reset_state": reset_state,
        "degradation": degradation,
        "recovery_world_vec": recovery_world_vec,
        "reset_slots": reset_slots,
        "reset_slots_target": reset_slots_target,
        "has_reset_slots_target": np.asarray(float(has_reset_slots_target), dtype=np.float32),
        "reset_slots_legacy": reset_slots_legacy,
        "reset_uncertainty_target": reset_uncertainty,
        "audit_event_type": event_name,
        "audit_event_type_id": event_type_id(event_name),
        "audit_mech": audit,
        "teacher_u": teacher_u,
        "teacher_traj": teacher_traj,
        "m_star": m_star,
        "b_star": b_star,
        "s_star": s_star,
        "calib_group": row_group(row),
    }
    # Backward-compatible aliases.  They intentionally point to the normalized
    # reset-problem tensors so old code sees the same values as the new path.
    out.update(
        {
            "event_type": event_name,
            "event_type_id": event_type_id(event_name),
            "event_time": reset_time,
            "x_minus": x_minus,
            "x_plus": reset_state,
            "deg": degradation,
            "d_deg": degradation,
            "world_plus": recovery_world_vec,
            "event_tokens": reset_slots_legacy,
            "has_event_tokens": np.asarray(float(legacy_slots_source is not None), dtype=np.float32),
            "z_mech": audit,
            "r_star": m_star,
        }
    )
    return out

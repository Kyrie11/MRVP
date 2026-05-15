from __future__ import annotations

"""MetaDrive-backed counterfactual sampler for MRVP-CF.

A root is reproduced by resetting MetaDrive with the same seed and replaying the
same pre-trigger ego controls. Candidate emergency prefixes then branch from
that identical replayed root. This is more robust across MetaDrive versions than
using private in-memory snapshot APIs.
"""

from dataclasses import dataclass
from typing import Any, Iterable
import math
import numpy as np

from mrvp.data.schema import STATE_DIM


@dataclass
class MetaDriveRoot:
    root_id: str
    seed: int
    family: str
    trigger_step: int
    pre_actions: np.ndarray
    env_config: dict[str, Any]
    root_state: np.ndarray
    o_hist: np.ndarray
    h_ctx: dict[str, Any]


@dataclass
class PrefixResult:
    prefix_rollout: np.ndarray
    prefix_controls: np.ndarray
    contact: bool
    contact_time: float
    contact_actor_id: int
    contact_normal: np.ndarray
    delta_v: float
    impulse: float
    audit: dict[str, Any]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if callable(x):
            x = x()
        return float(x)
    except Exception:
        return default


def _safe_vec2(x: Any) -> np.ndarray:
    try:
        if callable(x):
            x = x()
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
    except Exception:
        pass
    return np.zeros(2, dtype=np.float32)


def _heading_of(obj: Any) -> float:
    for name in ("heading_theta", "heading", "heading_dir"):
        if hasattr(obj, name):
            val = getattr(obj, name)
            if name == "heading_dir":
                vec = _safe_vec2(val)
                return float(math.atan2(vec[1], vec[0]))
            return _safe_float(val)
    return 0.0


def _speed_of(obj: Any) -> float:
    for name in ("speed", "last_speed"):
        if hasattr(obj, name):
            return max(0.0, _safe_float(getattr(obj, name)))
    if hasattr(obj, "velocity"):
        return float(np.linalg.norm(_safe_vec2(getattr(obj, "velocity"))))
    return 0.0


def _state_from_vehicle(vehicle: Any) -> np.ndarray:
    state = np.zeros(STATE_DIM, dtype=np.float32)
    pos = _safe_vec2(getattr(vehicle, "position", [0.0, 0.0]))
    heading = _heading_of(vehicle)
    speed = _speed_of(vehicle)
    vel = _safe_vec2(getattr(vehicle, "velocity", [speed * math.cos(heading), speed * math.sin(heading)]))
    state[0:2] = pos
    state[2] = heading
    state[3] = float(speed)
    if vel.size >= 2:
        c, s = math.cos(-heading), math.sin(-heading)
        state[3] = float(c * vel[0] - s * vel[1])
        state[4] = float(s * vel[0] + c * vel[1])
    state[5] = _safe_float(getattr(vehicle, "yaw_rate", 0.0))
    state[8] = math.atan2(float(state[4]), max(1e-3, abs(float(state[3]))))
    state[9] = _safe_float(getattr(vehicle, "steering", 0.0))
    tb = _safe_float(getattr(vehicle, "throttle_brake", 0.0))
    state[10] = max(0.0, -tb)
    state[11] = max(0.0, tb)
    return state.astype(np.float32)


def _local_xy(points_xy: np.ndarray, origin_state: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    dx = pts - origin_state[:2][None, :]
    c, s = math.cos(-float(origin_state[2])), math.sin(-float(origin_state[2]))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return dx @ rot.T


def _extract_crash(info: dict[str, Any], agent: Any) -> bool:
    keys = ["crash", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "out_of_road"]
    if any(bool(info.get(k, False)) for k in keys):
        return True
    attrs = ["crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "crash_human"]
    return any(bool(getattr(agent, a, False)) for a in attrs)


def action_to_metadrive(control: np.ndarray) -> np.ndarray:
    """Convert MRVP [steer, brake, traction] to MetaDrive [steer, throttle_brake]."""
    c = np.asarray(control, dtype=np.float32).reshape(-1)
    steer = float(np.clip(c[0], -1.0, 1.0))
    throttle_brake = float(np.clip(c[2] - c[1], -1.0, 1.0))
    return np.array([steer, throttle_brake], dtype=np.float32)


class MetaDriveCounterfactualAdapter:
    def __init__(self, cfg: dict[str, Any]):
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "MetaDrive is not installed. Install from source: "
                "git clone https://github.com/metadriverse/metadrive.git && cd metadrive && "
                "pip install -e . && python -m metadrive.pull_asset"
            ) from exc
        self.MetaDriveEnv = MetaDriveEnv
        self.cfg = cfg
        self.dt = float(cfg.get("prefix_dt", 0.1))
        self.history_steps = int(cfg.get("history_steps", 5))
        self.actors = int(cfg.get("actors", 6))
        self.actor_features = int(cfg.get("actor_features", 8))
        self.env = None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _family_env_config(self, family: str, seed: int) -> dict[str, Any]:
        base = dict(self.cfg.get("metadrive", {}).get("env", {}))
        base.setdefault("use_render", False)
        base.setdefault("traffic_density", 0.25)
        base.setdefault("num_scenarios", 1)
        base.setdefault("start_seed", int(seed))
        base.setdefault("random_lane_width", True)
        base.setdefault("random_lane_num", True)
        base.setdefault("random_agent_model", True)
        base.setdefault("crash_vehicle_done", False)
        base.setdefault("crash_object_done", False)
        base.setdefault("out_of_road_done", False)
        base.setdefault("horizon", max(200, int(self.cfg.get("metadrive", {}).get("episode_horizon", 220))))
        family_overrides = {
            "SC": {"map": "C", "traffic_density": 0.35},
            "LHB": {"map": "SSS", "traffic_density": 0.45},
            "CI": {"map": "SCS", "traffic_density": 0.50},
            "BLE": {"map": "S", "traffic_density": 0.15, "random_lane_num": False},
            "LF": {"map": "C", "traffic_density": 0.25, "wheel_friction": 0.72},
            "AD": {"map": "SCS", "traffic_density": 0.30},
            "OSH": {"map": "X", "traffic_density": 0.55},
            "NCR": {"map": "S", "traffic_density": 0.70, "random_lane_num": False},
        }
        base.update(family_overrides.get(family, {}))
        base["start_seed"] = int(seed)
        base["num_scenarios"] = 1
        return base

    def _make_env(self, env_config: dict[str, Any]):
        self.close()
        self.env = self.MetaDriveEnv(config=env_config)
        return self.env

    def reset(self, env_config: dict[str, Any], seed: int):
        env = self._make_env(env_config)
        obs, info = env.reset(seed=int(seed))
        return obs, info

    def _nominal_action(self, family: str, step: int, rng: np.random.Generator) -> np.ndarray:
        steer = 0.04 * math.sin(step / 9.0)
        throttle = 0.20
        if family in {"LHB", "BLE", "NCR"} and step > 8:
            throttle = 0.05
        if family in {"SC", "CI", "OSH"} and step > 10:
            steer += 0.10 * math.sin(step / 4.0)
        if family == "LF":
            steer += 0.06 * math.sin(step / 5.0)
        steer += float(rng.normal(0.0, 0.01))
        return np.array([np.clip(steer, -0.4, 0.4), np.clip(throttle, -0.3, 0.5)], dtype=np.float32)

    def _actors(self, root_state: np.ndarray | None = None) -> np.ndarray:
        rows: list[np.ndarray] = []
        objects: Iterable[Any] = []
        try:
            objects = self.env.engine.traffic_manager.spawned_objects.values()  # type: ignore[attr-defined]
        except Exception:
            objects = []
        ego_name = getattr(getattr(self.env, "agent", None), "name", None)
        for idx, obj in enumerate(objects):
            if getattr(obj, "name", None) == ego_name:
                continue
            s = _state_from_vehicle(obj)
            if root_state is not None:
                local = _local_xy(s[:2][None, :], root_state)[0]
                rel_heading = float(s[2] - root_state[2])
                row = np.array([local[0], local[1], rel_heading, s[3], s[4], s[5], 1.0, float(idx)], dtype=np.float32)
            else:
                row = np.array([s[0], s[1], s[2], s[3], s[4], s[5], 1.0, float(idx)], dtype=np.float32)
            rows.append(row)
        if len(rows) < self.actors:
            rows.extend([np.zeros(self.actor_features, dtype=np.float32) for _ in range(self.actors - len(rows))])
        return np.stack(rows[: self.actors], axis=0).astype(np.float32)

    def _h_ctx(self, family: str, deg: np.ndarray) -> dict[str, Any]:
        agent = self.env.agent
        left = _safe_float(getattr(agent, "dist_to_left_side", 3.5), 3.5)
        right = _safe_float(getattr(agent, "dist_to_right_side", 3.5), 3.5)
        return {
            "family": family,
            "dist_to_left_side": left,
            "dist_to_right_side": right,
            "friction": float(deg[0]),
            "speed_limit": _safe_float(getattr(agent, "max_speed", 30.0), 30.0),
            "map_seed": int(getattr(self.env, "current_seed", -1)),
        }

    def build_root(self, root_id: str, seed: int, family: str, deg: np.ndarray) -> MetaDriveRoot | None:
        env_config = self._family_env_config(family, seed)
        rng = np.random.default_rng(seed)
        trigger_step = int(self.cfg.get("metadrive", {}).get("trigger_step", 35))
        self.reset(env_config, seed)
        actors_hist: list[np.ndarray] = []
        pre_actions: list[np.ndarray] = []
        for step in range(trigger_step):
            agent_state = _state_from_vehicle(self.env.agent)
            actors_hist.append(self._actors(agent_state))
            a = self._nominal_action(family, step, rng)
            pre_actions.append(a)
            obs, rew, terminated, truncated, info = self.env.step(a)
            if terminated or truncated:
                return None
        root_state = _state_from_vehicle(self.env.agent)
        actors_hist.append(self._actors(root_state))
        hist = actors_hist[-self.history_steps :]
        if len(hist) < self.history_steps:
            hist = [np.zeros((self.actors, self.actor_features), dtype=np.float32)] * (self.history_steps - len(hist)) + hist
        return MetaDriveRoot(root_id, int(seed), family, trigger_step, np.asarray(pre_actions, dtype=np.float32), env_config, root_state.astype(np.float32), np.stack(hist, axis=0).astype(np.float32), self._h_ctx(family, deg))

    def replay_to_root(self, root: MetaDriveRoot) -> None:
        self.reset(root.env_config, root.seed)
        for a in root.pre_actions:
            self.env.step(np.asarray(a, dtype=np.float32))

    def apply_prefix(self, root: MetaDriveRoot, controls: np.ndarray) -> PrefixResult:
        self.replay_to_root(root)
        traj = [_state_from_vehicle(self.env.agent)]
        speeds = [float(np.linalg.norm(traj[-1][3:5]))]
        contact = False
        contact_time = -1.0
        contact_actor_id = -1
        contact_normal = np.zeros(2, dtype=np.float32)
        infos: list[dict[str, Any]] = []
        for k, u in enumerate(np.asarray(controls, dtype=np.float32)):
            obs, rew, terminated, truncated, info = self.env.step(action_to_metadrive(u))
            infos.append(dict(info))
            s = _state_from_vehicle(self.env.agent)
            traj.append(s)
            speeds.append(float(np.linalg.norm(s[3:5])))
            if not contact and _extract_crash(info, self.env.agent):
                contact = True
                contact_time = float((k + 1) * self.dt)
                contact_actor_id = int(info.get("crash_vehicle_id", -1)) if isinstance(info, dict) else -1
                d = traj[-1][:2] - traj[-2][:2]
                n = d / max(1e-6, float(np.linalg.norm(d)))
                contact_normal = n.astype(np.float32)
        speed_arr = np.asarray(speeds, dtype=np.float32)
        if contact:
            idx = max(1, min(len(speed_arr) - 1, int(round(contact_time / self.dt))))
            delta_v = float(max(0.0, speed_arr[idx - 1] - speed_arr[idx]))
        else:
            delta_v = 0.0
        impulse = float(1200.0 * delta_v)
        return PrefixResult(np.stack(traj, axis=0).astype(np.float32), np.asarray(controls, dtype=np.float32), bool(contact), contact_time, contact_actor_id, contact_normal, delta_v, impulse, {"metadrive_infos": infos[-3:], "root_replay": "seed+pre_actions"})

    def actor_states_at_reset(self, reset_state: np.ndarray) -> np.ndarray:
        return self._actors(reset_state)

    def make_world_from_reset(self, reset_state: np.ndarray, deg: np.ndarray, *, size: int, horizon: int) -> dict[str, np.ndarray]:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = cy = (size - 1) / 2.0
        res = float(self.cfg.get("world", {}).get("res_m", 1.0))
        gx = (xx - cx) * res
        gy = (yy - cy) * res
        hctx = self._h_ctx("unknown", deg)
        left = max(2.0, float(hctx.get("dist_to_left_side", 3.5)))
        right = max(2.0, float(hctx.get("dist_to_right_side", 3.5)))
        road_signed = np.minimum(left - gy, right + gy).astype(np.float32)
        drivable = (road_signed > 0.0).astype(np.float32)
        route = np.exp(-(gy ** 2) / (2 * 1.8 ** 2)).astype(np.float32) * drivable
        A = np.zeros((6, size, size), dtype=np.float32)
        A[0] = drivable
        A[1] = route
        A[2] = np.clip(left - gy, -5.0, 5.0) / 5.0
        A[3] = np.clip(road_signed, -5.0, 5.0) / 5.0
        A[4] = float(deg[0])
        A[5] = float(max(0.0, reset_state[3]) / 30.0)
        actors = self.actor_states_at_reset(reset_state)
        O = np.zeros((horizon, 1, size, size), dtype=np.float32)
        Y = np.zeros((horizon, 2, size, size), dtype=np.float32)
        for t in range(horizon):
            for a in actors:
                if a[6] <= 0:
                    continue
                px = float(a[0] + a[3] * self.dt * t)
                py = float(a[1] + a[4] * self.dt * t)
                blob = np.exp(-(((gx - px) ** 2) / (2 * 2.0 ** 2) + ((gy - py) ** 2) / (2 * 1.2 ** 2))).astype(np.float32)
                O[t, 0] = np.maximum(O[t, 0], blob)
                closing = max(0.0, float(reset_state[3]) - float(a[3])) / 30.0
                Y[t, 0] = np.maximum(Y[t, 0], closing * blob)
                Y[t, 1] = np.maximum(Y[t, 1], max(0.0, 1.0 - abs(py) / 6.0) * blob)
        G = np.zeros((3, size, size), dtype=np.float32)
        G[0] = np.exp(-(((gx - 18.0) ** 2) / (2 * 6.0 ** 2) + (gy ** 2) / (2 * 2.2 ** 2))).astype(np.float32) * drivable
        G[1] = np.exp(-(((gx - 10.0) ** 2) / (2 * 5.0 ** 2) + ((gy - max(2.0, left - 1.0)) ** 2) / (2 * 1.8 ** 2))).astype(np.float32) * drivable
        G[2] = np.exp(-(((gx - 6.0) ** 2) / (2 * 4.0 ** 2) + (gy ** 2) / (2 * 2.5 ** 2))).astype(np.float32) * drivable
        return {"A": A, "O": np.clip(O, 0.0, 1.0), "G": G.astype(np.float32), "Y": np.clip(Y, 0.0, 1.0)}

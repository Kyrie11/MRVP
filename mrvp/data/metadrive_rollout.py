from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import (
    ACTION_NAMES,
    ACTION_VECTORS,
    ACTOR_FEAT_DIM,
    CTX_DIM,
    DEG_DIM,
    HIST_LEN,
    MAX_AGENTS,
    MECH_DIM,
    NUM_ACTIONS,
    SIDE_NAMES,
    STATE_DIM,
    harm_bin_from_rho,
)
from .synthetic import FAMILIES, write_jsonl


@dataclass
class Actor2D:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 2.4


@dataclass
class RootScene:
    root_id: str
    root_index: int
    split: str
    family_id: int
    family: str
    seed: int
    base_x: float
    base_y: float
    base_yaw: float
    base_speed: float
    lane_width: float
    road_half_width: float
    friction: float
    boundary_side: int
    damage_class: int
    actor_density: float
    actors: List[Actor2D]
    town: str
    route_dx: float
    route_dy: float
    weather_wetness: float


@dataclass
class RolloutResult:
    x_minus: np.ndarray
    x_plus: np.ndarray
    transition_traj: List[np.ndarray]
    recovery_traj: List[np.ndarray]
    recovery_controls: List[Tuple[float, float, float]]
    collision: bool
    collision_side: str
    normal_xy: np.ndarray
    rho_imp: float
    overlap_clearance: float
    min_actor_clearance: float
    min_road_clearance: float
    offroad: bool
    done_reason: str


def _split_for_root(i: int, n_roots: int) -> str:
    frac = i / max(1, n_roots)
    if frac < 0.70:
        return "train"
    if frac < 0.82:
        return "val"
    if frac < 0.90:
        return "cal"
    return "test"

def write_rows_streaming(
    output: str | Path,
    n_roots: int,
    actions_per_root: int = NUM_ACTIONS,
    seed: int = 7,
    backend: str = "auto",
    shift_test: bool = True,
    harm_thresholds: Sequence[float] = (0.5, 2.0, 4.0, 7.0, 11.0),
    log_every: int = 10,
) -> None:
    rng = np.random.default_rng(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_backend = backend
    if backend == "auto":
        try:
            _try_import_metadrive_env()
            selected_backend = "metadrive"
        except ImportError:
            selected_backend = "light2d"

    split_counts: Dict[str, int] = {}
    backend_counts: Dict[str, int] = {}
    done_counts: Dict[str, int] = {}

    with output.open("w", encoding="utf-8") as f:
        for i in range(n_roots):
            scene = _make_root_scene(i, n_roots, rng, seed=seed, shift_test=shift_test)
            scene.root_id = f"{selected_backend}_{i:06d}"

            for a in range(actions_per_root):
                action_id = a % len(ACTION_NAMES)
                action_vec = ACTION_VECTORS[action_id].astype(np.float32)

                if selected_backend == "metadrive":
                    result = _rollout_metadrive(scene, action_vec)
                elif selected_backend == "light2d":
                    result = _rollout_light2d(scene, action_vec)
                else:
                    raise ValueError(f"Unknown backend {backend!r}.")

                row = _row_from_rollout(scene, action_id, result, harm_thresholds)
                row["backend"] = selected_backend

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
                backend_counts[row["backend"]] = backend_counts.get(row["backend"], 0) + 1
                reason = row["calib_group"]["done_reason"]
                done_counts[reason] = done_counts.get(reason, 0) + 1

            if (i + 1) % log_every == 0 or i == 0 or i + 1 == n_roots:
                print(
                    json.dumps(
                        {
                            "progress": f"{i + 1}/{n_roots}",
                            "rows_written": (i + 1) * actions_per_root,
                            "backend": selected_backend,
                            "splits": split_counts,
                            "done_reason": done_counts,
                            "output": str(output),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    print(
        json.dumps(
            {
                "finished": True,
                "output": str(output),
                "rows": n_roots * actions_per_root,
                "splits": split_counts,
                "backend": backend_counts,
                "done_reason": done_counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

def _state_vec(x: float, y: float, yaw: float, speed: float, yaw_rate: float, beta: float, steer: float, t: float) -> np.ndarray:
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[0] = float(x)
    out[1] = float(y)
    out[2] = float(yaw)
    out[3] = float(speed * math.cos(yaw + beta))
    out[4] = float(speed * math.sin(yaw + beta))
    out[5] = float(speed)
    out[6] = float(yaw_rate)
    out[7] = float(beta)
    out[8] = float(steer)
    out[9] = float(t)
    return out


def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _side_from_normal(normal_xy: np.ndarray, yaw: float) -> str:
    c, s = math.cos(-yaw), math.sin(-yaw)
    bx = c * normal_xy[0] - s * normal_xy[1]
    by = s * normal_xy[0] + c * normal_xy[1]
    if abs(bx) > abs(by) * 1.4:
        return "front" if bx < 0 else "rear"
    if abs(by) > abs(bx) * 1.4:
        return "left" if by > 0 else "right"
    return "oblique"


def _advance_bicycle(state: np.ndarray, steer_cmd: float, throttle: float, brake: float, friction: float, dt: float) -> np.ndarray:
    x, y, yaw, _, _, speed, _, _, _, t = [float(v) for v in state[:10]]
    steer = float(np.clip(steer_cmd, -1.0, 1.0)) * 0.65
    throttle = float(np.clip(throttle, 0.0, 1.0))
    brake = float(np.clip(brake, 0.0, 1.0))
    acc_cmd = 2.4 * throttle - 6.2 * brake
    # Friction limits both braking and yaw response.
    acc_cmd = float(np.clip(acc_cmd, -7.5 * friction, 2.8))
    speed = max(0.0, speed + acc_cmd * dt)
    wheelbase = 2.8
    raw_yaw_rate = speed / wheelbase * math.tan(steer)
    yaw_rate_limit = max(0.15, 0.78 * friction * 9.81 / max(speed, 2.0))
    yaw_rate = float(np.clip(raw_yaw_rate, -yaw_rate_limit, yaw_rate_limit))
    beta = float(np.clip(math.atan(0.45 * math.tan(steer)), -0.65, 0.65))
    yaw = _wrap_angle(yaw + yaw_rate * dt)
    x = x + speed * math.cos(yaw + beta) * dt
    y = y + speed * math.sin(yaw + beta) * dt
    return _state_vec(x, y, yaw, speed, yaw_rate, beta, steer_cmd, t + dt)


def _actor_positions(scene: RootScene, t: float) -> List[Tuple[float, float, float, float, float]]:
    return [(a.x + a.vx * t, a.y + a.vy * t, a.vx, a.vy, a.radius) for a in scene.actors]


def _min_actor_clearance(scene: RootScene, state: np.ndarray) -> Tuple[float, Optional[Tuple[float, float, float, float, float]]]:
    ego_x, ego_y, t = float(state[0]), float(state[1]), float(state[9])
    best = float("inf")
    best_actor = None
    ego_radius = 2.35
    for actor in _actor_positions(scene, t):
        ax, ay, avx, avy, ar = actor
        clearance = math.hypot(ego_x - ax, ego_y - ay) - (ego_radius + ar)
        if clearance < best:
            best = clearance
            best_actor = actor
    if best_actor is None:
        best = 99.0
    return float(best), best_actor


def _road_clearance(scene: RootScene, state: np.ndarray) -> float:
    vehicle_half_width = 0.95
    return float(scene.road_half_width - abs(float(state[1])) - vehicle_half_width)


def _make_root_scene(i: int, n_roots: int, rng: np.random.Generator, seed: int, shift_test: bool = False) -> RootScene:
    family_id = i % len(FAMILIES)
    family = FAMILIES[family_id]
    split = _split_for_root(i, n_roots)
    is_shift = shift_test and split == "test"
    base_speed = float(rng.uniform(6.0, 18.0))
    lane_width = float(rng.uniform(3.0, 4.2))
    road_half_width = lane_width * float(rng.choice([1.0, 1.5], p=[0.75, 0.25]))
    friction = float(rng.choice([0.95, 0.75, 0.45], p=[0.60, 0.28, 0.12]))
    if family_id == 5:
        friction = float(rng.uniform(0.28, 0.55))
    if is_shift:
        friction = min(friction, float(rng.uniform(0.25, 0.50)))
    actor_density = float(rng.uniform(0.05, 0.55))
    if family_id in (0, 2, 3, 7):
        actor_density = float(rng.uniform(0.45, 1.0))
    if is_shift:
        actor_density = max(actor_density, float(rng.uniform(0.70, 1.0)))
    boundary_side = int(rng.choice([-1, 1]))
    base_y = float(rng.normal(0.0, 0.35))
    if family_id in (1, 4):
        base_y = float(boundary_side * rng.uniform(0.25 * lane_width, 0.45 * lane_width))
    base_yaw = float(rng.normal(0.0, 0.08))
    damage_class = int(rng.integers(0, 3))
    if family_id == 6:
        damage_class = int(rng.integers(2, 5))
    if is_shift:
        damage_class = max(damage_class, int(rng.integers(2, 5)))
    actors: List[Actor2D] = []
    # Family-specific hazards, then density-driven background traffic.
    if family_id == 0:
        actors.append(Actor2D(float(rng.uniform(8, 15)), float(base_y + rng.normal(0, 0.25)), base_speed * rng.uniform(0.05, 0.35), 0.0))
    elif family_id == 1:
        actors.append(Actor2D(float(rng.uniform(5, 15)), float(base_y - boundary_side * rng.uniform(1.8, 3.2)), base_speed * rng.uniform(0.5, 0.9), 0.0))
    elif family_id == 2:
        actors.append(Actor2D(float(rng.uniform(9, 18)), float(-boundary_side * rng.uniform(4, 8)), 0.0, float(boundary_side * rng.uniform(3, 7))))
    elif family_id == 3:
        actors.append(Actor2D(float(rng.uniform(7, 14)), float(-boundary_side * rng.uniform(1.5, 3.5)), base_speed * rng.uniform(0.2, 0.7), float(boundary_side * rng.uniform(0.8, 2.0))))
    elif family_id == 7:
        for _ in range(3):
            actors.append(Actor2D(float(rng.uniform(8, 28)), float(rng.normal(0, road_half_width * 0.7)), base_speed * rng.uniform(0.2, 1.1), float(rng.normal(0, 0.5))))
    n_bg = int(np.clip(round(actor_density * 7), 0, MAX_AGENTS - 2))
    for _ in range(n_bg):
        actors.append(Actor2D(float(rng.uniform(12, 55)), float(rng.normal(0, road_half_width * 0.8)), base_speed * rng.uniform(0.25, 1.25), float(rng.normal(0, 0.35))))
    return RootScene(
        root_id=f"md_{i:06d}",
        root_index=i,
        split=split,
        family_id=family_id,
        family=family,
        seed=int(seed + i * 9973),
        base_x=0.0,
        base_y=base_y,
        base_yaw=base_yaw,
        base_speed=base_speed,
        lane_width=lane_width,
        road_half_width=road_half_width,
        friction=friction,
        boundary_side=boundary_side,
        damage_class=damage_class,
        actor_density=actor_density,
        actors=actors,
        town=f"MetaDrivePG{int(rng.integers(1, 6)):02d}",
        route_dx=float(rng.uniform(35, 80)),
        route_dy=float(rng.normal(0, 4)),
        weather_wetness=float(np.clip(1.0 - friction + rng.normal(0, 0.08), 0.0, 1.0)),
    )


def _rollout_light2d(scene: RootScene, action_vec: np.ndarray, dt: float = 0.05, recovery_horizon: float = 4.0) -> RolloutResult:
    steer, throttle, brake, duration = [float(x) for x in action_vec]
    steering_scale = max(0.20, 1.0 - 0.12 * scene.damage_class)
    brake_scale = max(0.25, 1.0 - 0.10 * scene.damage_class)
    state = _state_vec(scene.base_x, scene.base_y, scene.base_yaw, scene.base_speed, 0.0, 0.0, 0.0, 0.0)
    x_minus = state.copy()
    transition: List[np.ndarray] = [state.copy()]
    collision = False
    collision_side = "oblique"
    normal_xy = np.asarray([1.0, 0.0], dtype=np.float32)
    rho_imp = 0.0
    overlap_clearance = 99.0
    min_actor_clear = 99.0
    min_road_clear = _road_clearance(scene, state)
    offroad = False
    steps = max(1, int(round(duration / dt)))
    for _ in range(steps):
        state = _advance_bicycle(state, steer * steering_scale, throttle, brake * brake_scale, scene.friction, dt)
        clear, actor = _min_actor_clearance(scene, state)
        min_actor_clear = min(min_actor_clear, clear)
        min_road_clear = min(min_road_clear, _road_clearance(scene, state))
        if _road_clearance(scene, state) < 0.0:
            offroad = True
        if (not collision) and clear < 0.0 and actor is not None:
            collision = True
            ax, ay, avx, avy, _ = actor
            rel = np.asarray([state[3] - avx, state[4] - avy], dtype=np.float32)
            n = np.asarray([state[0] - ax, state[1] - ay], dtype=np.float32)
            n_norm = np.linalg.norm(n)
            normal_xy = n / (n_norm + 1e-6) if n_norm > 1e-6 else np.asarray([-1.0, 0.0], dtype=np.float32)
            rel_normal = max(0.0, -float(np.dot(rel, normal_xy)))
            if rel_normal <= 0.05:
                rel_normal = float(np.linalg.norm(rel))
            rho_imp = float(rel_normal * (0.55 + 0.10 * scene.damage_class))
            overlap_clearance = float(clear)
            collision_side = _side_from_normal(normal_xy, float(state[2]))
            # Post-contact reset: damp velocity and inject yaw-rate in the contact normal direction.
            speed = max(0.2, float(state[5]) - 0.45 * rho_imp)
            yaw = _wrap_angle(float(state[2]) + 0.035 * rho_imp * float(np.sign(normal_xy[1] + 1e-6)))
            yaw_rate = float(state[6] + 0.12 * rho_imp * np.sign(normal_xy[1] + 1e-6))
            beta = float(np.clip(state[7] + 0.04 * rho_imp * np.sign(normal_xy[1] + 1e-6), -0.75, 0.75))
            state = _state_vec(float(state[0]), float(state[1]), yaw, speed, yaw_rate, beta, steer, float(state[9]))
        transition.append(state.copy())
    x_plus = state.copy()
    recovery: List[np.ndarray] = [state.copy()]
    controls: List[Tuple[float, float, float]] = []
    delay_steps = int(round((0.02 + 0.035 * scene.damage_class) / dt))
    delayed_steer: List[float] = [0.0] * max(1, delay_steps + 1)
    rec_steps = max(1, int(round(recovery_horizon / dt)))
    for _ in range(rec_steps):
        # Degraded lane-return controller from the transition state.
        y = float(state[1])
        yaw = float(state[2])
        speed = float(state[5])
        steer_des = np.clip(-0.42 * y - 1.15 * yaw - 0.15 * state[7], -1.0, 1.0)
        steer_cmd = float(np.clip(steer_des, -steering_scale, steering_scale))
        delayed_steer.append(steer_cmd)
        steer_apply = delayed_steer.pop(0)
        target_speed = min(9.0, max(3.5, scene.base_speed * 0.60))
        brake_cmd = float(np.clip((speed - target_speed) / 7.0 + 0.15 * abs(yaw), 0.0, brake_scale))
        throttle_cmd = float(np.clip((target_speed - speed) / 8.0, 0.0, 0.35)) if brake_cmd < 0.05 else 0.0
        state = _advance_bicycle(state, steer_apply, throttle_cmd, brake_cmd, scene.friction, dt)
        clear, _ = _min_actor_clearance(scene, state)
        min_actor_clear = min(min_actor_clear, clear)
        min_road_clear = min(min_road_clear, _road_clearance(scene, state))
        offroad = offroad or _road_clearance(scene, state) < 0.0
        controls.append((steer_apply, throttle_cmd, brake_cmd))
        recovery.append(state.copy())
    return RolloutResult(
        x_minus=x_minus,
        x_plus=x_plus,
        transition_traj=transition,
        recovery_traj=recovery,
        recovery_controls=controls,
        collision=collision,
        collision_side=collision_side if collision else ("left" if scene.boundary_side < 0 else "right" if offroad else "oblique"),
        normal_xy=normal_xy.astype(np.float32),
        rho_imp=float(rho_imp if collision else 0.0),
        overlap_clearance=float(overlap_clearance if collision else min_road_clear),
        min_actor_clearance=float(min_actor_clear),
        min_road_clearance=float(min_road_clear),
        offroad=offroad,
        done_reason="collision" if collision else ("offroad" if offroad else "horizon"),
    )


def _try_import_metadrive_env():
    try:
        from metadrive import MetaDriveEnv  # type: ignore

        return MetaDriveEnv
    except Exception:
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv  # type: ignore

            return MetaDriveEnv
        except Exception as exc:  # pragma: no cover - depends on optional simulator
            raise ImportError("MetaDrive is not installed. Run `pip install metadrive-simulator` or use --backend light2d.") from exc


def _md_step(env: Any, action: np.ndarray) -> Tuple[Any, float, bool, Mapping[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def _md_reset(env: Any, seed: int) -> Any:
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _md_agent_state(env: Any, t: float, steer: float = 0.0) -> np.ndarray:
    agent = getattr(env, "agent", None) or getattr(env, "vehicle", None)
    if agent is None:
        try:
            agent = env.engine.agent_manager.active_agents[next(iter(env.engine.agent_manager.active_agents))]
        except Exception:
            agent = None
    if agent is None:
        return _state_vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, steer, t)
    pos = getattr(agent, "position", [0.0, 0.0])
    x, y = float(pos[0]), float(pos[1])
    yaw = float(getattr(agent, "heading_theta", getattr(agent, "heading", 0.0)))
    speed = float(getattr(agent, "speed", 0.0))
    vel = getattr(agent, "velocity", None)
    if vel is not None and len(vel) >= 2:
        vx, vy = float(vel[0]), float(vel[1])
        speed = max(speed, math.hypot(vx, vy))
    yaw_rate = float(getattr(agent, "yaw_rate", 0.0))
    beta = float(getattr(agent, "beta", 0.0))
    return _state_vec(x, y, yaw, speed, yaw_rate, beta, steer, t)


def _rollout_metadrive(scene: RootScene, action_vec: np.ndarray, dt: float = 0.05, recovery_horizon: float = 4.0) -> RolloutResult:  # pragma: no cover - optional simulator
    MetaDriveEnv = _try_import_metadrive_env()
    config = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": float(scene.actor_density),
        "start_seed": int(scene.seed),
        "num_scenarios": 1,
        "log_level": 50,
        "image_observation": False,

        "crash_vehicle_done": False,
        "crash_object_done": False,
        "crash_human_done": False,
        "out_of_road_done": False,

        "vehicle_config": {"show_navi_mark": False},
    }
    env = MetaDriveEnv(config)
    steer, throttle, brake, duration = [float(x) for x in action_vec]
    transition: List[np.ndarray] = []
    recovery: List[np.ndarray] = []
    controls: List[Tuple[float, float, float]] = []
    collision = False
    offroad = False
    done_reason = "horizon"
    try:
        _md_reset(env, scene.seed)
        t = 0.0
        x_minus = _md_agent_state(env, t, steer=0.0)
        transition.append(x_minus.copy())
        steps = max(1, int(round(duration / dt)))
        for _ in range(steps):
            md_action = np.asarray([np.clip(steer, -1.0, 1.0), np.clip(throttle - brake, -1.0, 1.0)], dtype=np.float32)
            _, _, done, info = _md_step(env, md_action)
            t += dt
            state = _md_agent_state(env, t, steer=steer)
            transition.append(state.copy())
            collision = collision or bool(info.get("crash", info.get("crash_vehicle", False)))
            offroad = offroad or bool(info.get("out_of_road", info.get("on_lane", True) is False))
            if done:
                done_reason = "terminated"
                break
        x_plus = transition[-1].copy()
        rec_steps = max(1, int(round(recovery_horizon / dt)))
        recovery.append(x_plus.copy())
        for _ in range(rec_steps):
            state = recovery[-1]
            steer_cmd = float(np.clip(-0.35 * state[1] - 0.9 * state[2], -1.0, 1.0))
            speed = float(state[5])
            brake_cmd = float(np.clip((speed - 8.0) / 8.0, 0.0, 1.0))
            throttle_cmd = 0.0 if brake_cmd > 0.05 else float(np.clip((8.0 - speed) / 10.0, 0.0, 0.25))
            md_action = np.asarray([steer_cmd, throttle_cmd - brake_cmd], dtype=np.float32)
            _, _, done, info = _md_step(env, md_action)
            t += dt
            new_state = _md_agent_state(env, t, steer=steer_cmd)
            recovery.append(new_state.copy())
            controls.append((steer_cmd, throttle_cmd, brake_cmd))
            collision = collision or bool(info.get("crash", info.get("crash_vehicle", False)))
            offroad = offroad or bool(info.get("out_of_road", False))
            if done:
                done_reason = "terminated"
                break
    finally:
        env.close()
    # Native MetaDrive clearance is not stable across versions; use lane-context proxies.
    min_road = min(_road_clearance(scene, s) for s in recovery + transition)
    min_actor = -0.5 if collision else max(2.0, 10.0 - 7.0 * scene.actor_density)
    normal = np.asarray([-1.0, float(scene.boundary_side)], dtype=np.float32)
    normal /= np.linalg.norm(normal) + 1e-6
    rel_speed = max(0.0, float(x_minus[5]) - float(x_plus[5]))
    rho = rel_speed * (0.8 if collision else 0.0)
    return RolloutResult(
        x_minus=x_minus,
        x_plus=x_plus,
        transition_traj=transition,
        recovery_traj=recovery,
        recovery_controls=controls,
        collision=collision,
        collision_side=_side_from_normal(normal, float(x_plus[2])) if collision else ("left" if scene.boundary_side < 0 else "right" if offroad else "oblique"),
        normal_xy=normal,
        rho_imp=float(rho),
        overlap_clearance=float(-0.2 if collision else min_road),
        min_actor_clearance=float(min_actor),
        min_road_clearance=float(min_road),
        offroad=offroad,
        done_reason=done_reason,
    )


def _history_from_scene(scene: RootScene) -> np.ndarray:
    hist = np.zeros((HIST_LEN, MAX_AGENTS, ACTOR_FEAT_DIM), dtype=np.float32)
    for ti in range(HIST_LEN):
        dt = (ti - HIST_LEN + 1) * 0.1
        x = scene.base_x + scene.base_speed * math.cos(scene.base_yaw) * dt
        y = scene.base_y + scene.base_speed * math.sin(scene.base_yaw) * dt
        hist[ti, 0, :] = [x, y, scene.base_yaw, scene.base_speed, 0.0, 4.6, 1.9, 1.0, 1.0]
        for ai, actor in enumerate(scene.actors[: MAX_AGENTS - 1], start=1):
            hist[ti, ai, :] = [actor.x + actor.vx * dt, actor.y + actor.vy * dt, 0.0, math.hypot(actor.vx, actor.vy), 0.0, 4.5, 1.8, 1.0, 1.0]
    return hist


def _context_from_scene(scene: RootScene) -> np.ndarray:
    h = np.zeros(CTX_DIM, dtype=np.float32)
    nearest_clear = 99.0
    ego0 = _state_vec(scene.base_x, scene.base_y, scene.base_yaw, scene.base_speed, 0.0, 0.0, 0.0, 0.0)
    if scene.actors:
        nearest_clear, _ = _min_actor_clearance(scene, ego0)
    h[:16] = [
        0.0,
        -scene.base_yaw,
        scene.road_half_width * 2.0,
        scene.road_half_width + scene.base_y,
        scene.road_half_width - scene.base_y,
        nearest_clear,
        8.0,
        nearest_clear,
        max(0.1, nearest_clear / max(scene.base_speed, 0.1)),
        scene.route_dx,
        scene.actor_density,
        13.9,
        scene.friction,
        scene.weather_wetness,
        0.0,
        scene.lane_width,
    ]
    h[16:24] = [0.0, -scene.base_y, scene.route_dx, scene.route_dy, 0.0, min(1.0, scene.actor_density * 0.2), 1.0, scene.boundary_side]
    h[24:29] = [scene.root_index % 10 / 10.0, scene.family_id / max(1, len(FAMILIES) - 1), 1.0 if scene.family_id != 4 else 0.0, scene.damage_class / 4.0, 0.5]
    return h


def _margins_from_rollout(scene: RootScene, result: RolloutResult) -> np.ndarray:
    traj = result.recovery_traj if result.recovery_traj else [result.x_plus]
    controls = result.recovery_controls
    road_margins = np.asarray([_road_clearance(scene, s) for s in traj], dtype=np.float32)
    sec_margins = []
    stab_margins = []
    ctrl_margins = []
    steering_scale = max(0.20, 1.0 - 0.12 * scene.damage_class)
    brake_scale = max(0.25, 1.0 - 0.10 * scene.damage_class)
    delay = 0.02 + 0.035 * scene.damage_class
    for k, s in enumerate(traj):
        clear, _ = _min_actor_clearance(scene, s)
        sec_margins.append(clear - 0.8)
        stab_margins.append(scene.friction * 1.35 - 0.55 * abs(float(s[6])) - 0.50 * abs(float(s[7])) - 0.025 * float(s[5]))
        if k < len(controls):
            steer_cmd, _, brake_cmd = controls[k]
            ctrl_margins.append(min(steering_scale - abs(steer_cmd), brake_scale - brake_cmd) - 0.20 * delay)
    if not ctrl_margins:
        ctrl_margins = [min(steering_scale, brake_scale) - 0.20 * delay]
    final = traj[-1]
    return_center = 0.35 * scene.lane_width - abs(float(final[1]))
    return_heading = 0.40 - abs(_wrap_angle(float(final[2])))
    progress = max(0.0, float(final[0] - result.x_plus[0]))
    return_progress = progress / max(1.0, scene.route_dx * 0.25) - 0.2
    r_return = min(return_center, return_heading, return_progress, float(np.min(road_margins)) + 0.3, float(np.min(sec_margins)) + 0.2)
    return np.asarray(
        [
            float(np.min(sec_margins)),
            float(np.min(road_margins)),
            float(np.min(stab_margins)),
            float(np.min(ctrl_margins)),
            float(r_return),
        ],
        dtype=np.float32,
    )


def _row_from_rollout(scene: RootScene, action_id: int, result: RolloutResult, harm_thresholds: Sequence[float]) -> Dict[str, Any]:
    action_vec = ACTION_VECTORS[action_id % len(ACTION_VECTORS)].astype(np.float32)
    steering_scale = max(0.20, 1.0 - 0.12 * scene.damage_class)
    brake_scale = max(0.25, 1.0 - 0.10 * scene.damage_class)
    throttle_scale = max(0.30, 1.0 - 0.07 * scene.damage_class)
    delay = 0.02 + 0.035 * scene.damage_class
    d_deg = np.asarray([steering_scale, brake_scale, throttle_scale, delay, scene.friction, scene.damage_class / 4.0], dtype=np.float32)
    z = np.zeros(MECH_DIM, dtype=np.float32)
    z[0] = 1.0 if result.collision else 0.0
    side_id = SIDE_NAMES.index(result.collision_side) if result.collision_side in SIDE_NAMES else SIDE_NAMES.index("oblique")
    z[1 + side_id] = 1.0
    normal = result.normal_xy.astype(np.float32)
    if np.linalg.norm(normal) < 1e-6:
        normal = np.asarray([1.0, 0.0], dtype=np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    z[6:8] = normal
    z[8] = float(result.overlap_clearance)
    z[9] = _wrap_angle(float(result.x_plus[2] - result.x_minus[2]))
    z[10] = float(result.rho_imp)
    reset = np.asarray(
        [
            result.x_plus[3] - result.x_minus[3],
            result.x_plus[4] - result.x_minus[4],
            result.x_plus[5] - result.x_minus[5],
            result.x_plus[2] - result.x_minus[2],
            result.x_plus[6] - result.x_minus[6],
            result.x_plus[7] - result.x_minus[7],
            result.x_plus[1] - result.x_minus[1],
        ],
        dtype=np.float32,
    )
    z[11:18] = reset
    road_clear = _road_clearance(scene, result.x_plus)
    sec_clear, _ = _min_actor_clearance(scene, result.x_plus)
    return_corridor = max(0.0, scene.route_dx - max(0.0, abs(float(result.x_plus[1])) - 0.25 * scene.lane_width) * 3.0 - 3.0 * scene.actor_density)
    z[18:24] = [scene.lane_width, road_clear, return_corridor, sec_clear, np.sign(scene.route_dy), 1.0]
    z[24:29] = [steering_scale, brake_scale, delay, scene.friction, scene.damage_class / 4.0]
    z[29:32] = [0.04 + 0.10 * scene.actor_density, 0.05 + 0.18 * float(result.collision), scene.actor_density]
    r_star = _margins_from_rollout(scene, result)
    rho_imp = float(result.rho_imp)
    harm_bin = harm_bin_from_rho(rho_imp, harm_thresholds)
    return {
        "root_id": scene.root_id,
        "split": scene.split,
        "family": scene.family,
        "backend": "metadrive_or_light2d_rollout",
        "action_id": int(action_id),
        "action_name": ACTION_NAMES[action_id % len(ACTION_NAMES)],
        "action_vec": action_vec.tolist(),
        "o_hist": _history_from_scene(scene).tolist(),
        "h_ctx": _context_from_scene(scene).tolist(),
        "rho_imp": rho_imp,
        "harm_bin": int(harm_bin),
        "x_minus": result.x_minus.astype(float).tolist(),
        "x_plus": result.x_plus.astype(float).tolist(),
        "d_deg": d_deg.tolist(),
        "z_mech": z.tolist(),
        "r_star": r_star.tolist(),
        "b_star": int(np.argmin(r_star)),
        "s_star": float(np.min(r_star)),
        "calib_group": {
            "contact_type": "contact" if result.collision else ("boundary" if result.offroad else "non_contact"),
            "contact_side": result.collision_side,
            "boundary_side": "right" if scene.boundary_side > 0 else "left",
            "friction_bin": "low" if scene.friction < 0.55 else ("mid" if scene.friction < 0.85 else "high"),
            "damage_class": str(scene.damage_class),
            "density_bin": "dense" if scene.actor_density > 0.65 else "normal",
            "town": scene.town,
            "family": scene.family,
            "done_reason": result.done_reason,
        },
    }


def make_metadrive_rows(
    n_roots: int = 240,
    actions_per_root: int = NUM_ACTIONS,
    seed: int = 7,
    backend: str = "auto",
    shift_test: bool = True,
    harm_thresholds: Sequence[float] = (0.5, 2.0, 4.0, 7.0, 11.0),
) -> List[Dict[str, Any]]:
    """Generate trajectory-labeled MRVP rows with MetaDrive or a deterministic 2D fallback.

    The important property is that labels come from an open-loop emergency
    rollout followed by a degraded recovery rollout. ``r_star`` is computed from
    recovery trajectories instead of being written directly from action ids.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    selected_backend = backend
    if backend == "auto":
        try:
            _try_import_metadrive_env()
            selected_backend = "metadrive"
        except ImportError:
            selected_backend = "light2d"
    for i in range(n_roots):
        scene = _make_root_scene(i, n_roots, rng, seed=seed, shift_test=shift_test)
        scene.root_id = f"{selected_backend}_{i:06d}"
        for a in range(actions_per_root):
            action_id = a % len(ACTION_NAMES)
            action_vec = ACTION_VECTORS[action_id].astype(np.float32)
            if selected_backend == "metadrive":
                result = _rollout_metadrive(scene, action_vec)
            elif selected_backend == "light2d":
                result = _rollout_light2d(scene, action_vec)
            else:
                raise ValueError(f"Unknown backend {backend!r}. Use auto, metadrive, or light2d.")
            row = _row_from_rollout(scene, action_id, result, harm_thresholds)
            row["backend"] = selected_backend
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MRVP JSONL with MetaDrive counterfactual rollouts or a light 2D fallback.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-roots", type=int, default=240)
    parser.add_argument("--actions-per-root", type=int, default=NUM_ACTIONS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=["auto", "metadrive", "light2d"], default="auto")
    parser.add_argument("--no-shift-test", action="store_true", help="Disable harder friction/density/damage distribution for test roots.")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()



    write_rows_streaming(
        output=args.output,
        n_roots=args.n_roots,
        actions_per_root=args.actions_per_root,
        seed=args.seed,
        backend=args.backend,
        shift_test=not args.no_shift_test,
        log_every=args.log_every,
    )
    return

if __name__ == "__main__":
    main()

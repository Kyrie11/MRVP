from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from mrvp.data.schema import ACTOR_FEAT_DIM, HIST_LEN, MAX_AGENTS, action_vec_from_id
from .action_library import EmergencyAction, default_action_library
from .logger import CarlaRolloutLogger
from .mechanism_labels import compute_mechanism_labels
from .recovery_teachers import degraded_mpc_teacher
from .scenario_templates import RootScenarioSpec, generate_root_specs
from .transition_extractor import extract_transition


def _import_carla():
    try:
        import carla  # type: ignore
        return carla
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "CARLA Python API is not importable. Add CARLA/PythonAPI/carla/dist/carla-*.egg to PYTHONPATH "
            "or install the matching carla package."
        ) from exc


def _weather_from_id(carla, weather_id: int):
    presets = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.MidRainyNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.CloudySunset,
        carla.WeatherParameters.WetSunset,
        carla.WeatherParameters.WetCloudySunset,
        carla.WeatherParameters.MidRainSunset,
        carla.WeatherParameters.HardRainSunset,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.SoftRainSunset,
    ]
    return presets[int(weather_id) % len(presets)]


def _actor_hist_from_rollout(rollout: Dict[str, Any]) -> List[List[List[float]]]:
    frames = rollout.get("frames", [])[-HIST_LEN:]
    hist = np.zeros((HIST_LEN, MAX_AGENTS, ACTOR_FEAT_DIM), dtype=np.float32)
    start = HIST_LEN - len(frames)
    for ti, frame in enumerate(frames):
        ego = frame.get("ego", {})
        hist[start + ti, 0, :] = [
            ego.get("x", 0.0), ego.get("y", 0.0), ego.get("yaw", 0.0), ego.get("vx", 0.0), ego.get("vy", 0.0), ego.get("length", 4.6), ego.get("width", 1.9), 1.0, 1.0
        ]
        for ai, actor in enumerate(frame.get("actors", [])[: MAX_AGENTS - 1], start=1):
            hist[start + ti, ai, :] = [
                actor.get("x", 0.0), actor.get("y", 0.0), actor.get("yaw", 0.0), actor.get("vx", 0.0), actor.get("vy", 0.0), actor.get("length", 4.6), actor.get("width", 1.9), 1.0, 1.0
            ]
    return hist.tolist()


class CarlaMRVPGenerator:
    """CARLA generator for the paper's CARLA-MRVP dataset.

    It creates root scenarios, replays each candidate emergency action from the
    same root seed, extracts x-/x+/d/z and computes recovery teacher margins.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout: float = 20.0,
        fixed_delta_seconds: float = 0.05,
        traffic_manager_port: int = 8000,
    ) -> None:
        self.carla = _import_carla()
        self.client = self.carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.fixed_delta_seconds = fixed_delta_seconds
        self.tm = self.client.get_trafficmanager(traffic_manager_port)
        self.world = None
        self.map = None
        self.actors: List[Any] = []

    def setup_world(self, town: str, seed: int) -> None:
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = max(1, int(math.ceil(self.fixed_delta_seconds / 0.01)))
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(int(seed))
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

    def cleanup(self) -> None:
        for actor in list(self.actors):
            try:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            except Exception:
                pass
        self.actors.clear()

    def close(self) -> None:
        try:
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            self.tm.set_synchronous_mode(False)
        except Exception:
            pass
        self.cleanup()

    def _spawn_vehicle(self, transform, role: str = "autopilot"):
        assert self.world is not None
        bps = self.world.get_blueprint_library().filter("vehicle.*")
        bp = random.choice([b for b in bps if int(b.get_attribute("number_of_wheels")) == 4])
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role)
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            raise RuntimeError("Failed to spawn vehicle. Try a different spawn index/town.")
        self.actors.append(actor)
        return actor

    def _prepare_root(self, spec: RootScenarioSpec):
        self.cleanup()
        assert self.world is not None and self.map is not None
        self.world.set_weather(_weather_from_id(self.carla, spec.weather_id))
        spawns = self.map.get_spawn_points()
        if not spawns:
            raise RuntimeError("CARLA map has no spawn points.")
        ego_tf = spawns[spec.spawn_index % len(spawns)]
        ego = self._spawn_vehicle(ego_tf, role="ego")
        # Surrounding traffic: deterministic autopilot actors around spawn list.
        other_actors = []
        for k in range(spec.actor_density):
            tf = spawns[(spec.spawn_index + 3 + k * 7) % len(spawns)]
            try:
                actor = self._spawn_vehicle(tf, role="background")
                actor.set_autopilot(True, self.tm.get_port())
                self.tm.ignore_lights_percentage(actor, 25.0)
                self.tm.global_percentage_speed_difference(0.0)
                other_actors.append(actor)
            except Exception:
                continue
        # Give ego a target velocity by applying throttle before trigger.
        ego.apply_control(self.carla.VehicleControl(throttle=0.55, brake=0.0, steer=0.0))
        return ego, other_actors

    def rollout_action(self, spec: RootScenarioSpec, action: EmergencyAction, pre_trigger_seconds: float = 1.0, post_action_seconds: float = 1.2) -> Dict[str, Any]:
        self.setup_world(spec.town, spec.seed)
        ego, other_actors = self._prepare_root(spec)
        logger = CarlaRolloutLogger(self.world, ego, other_actors, self.map)
        logger.attach_collision_sensor()
        action_control = action.to_carla_control()
        try:
            n_pre = int(round((pre_trigger_seconds + spec.trigger_time) / self.fixed_delta_seconds))
            for _ in range(n_pre):
                ctrl = self.carla.VehicleControl(throttle=0.45, steer=0.0, brake=0.0)
                ego.apply_control(ctrl)
                frame = self.world.tick()
                snap = self.world.get_snapshot()
                logger.log_frame(frame, snap.timestamp.elapsed_seconds, ctrl)
            n_action = int(round(action.duration / self.fixed_delta_seconds))
            for _ in range(n_action):
                ego.apply_control(action_control)
                frame = self.world.tick()
                snap = self.world.get_snapshot()
                logger.log_frame(frame, snap.timestamp.elapsed_seconds, action_control)
            n_post = int(round(post_action_seconds / self.fixed_delta_seconds))
            for _ in range(n_post):
                # Continue braking mildly until transition extractor has enough post frames.
                ctrl = self.carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.4)
                ego.apply_control(ctrl)
                frame = self.world.tick()
                snap = self.world.get_snapshot()
                logger.log_frame(frame, snap.timestamp.elapsed_seconds, ctrl)
            rollout = logger.as_rollout()
            rollout.update(
                {
                    "root_id": spec.root_id,
                    "split": spec.split,
                    "family": spec.family,
                    "town": spec.town,
                    "seed": spec.seed,
                    "action_id": action.action_id,
                    "action_name": action.name,
                    "action_vec": action.as_vector(),
                    "trigger_time": spec.trigger_time,
                    "friction": spec.friction,
                    "degradation": spec.degradation,
                }
            )
            return rollout
        finally:
            logger.destroy()
            self.cleanup()

    def rollout_to_row(self, spec: RootScenarioSpec, action: EmergencyAction, rollout: Dict[str, Any], recovery_horizon: float = 5.0) -> Dict[str, Any]:
        trans = extract_transition(rollout)
        mech = compute_mechanism_labels(rollout, trans)
        teacher = degraded_mpc_teacher(trans["x_plus"], mech["d_deg"], mech["h_ctx"], horizon=recovery_horizon)
        row = {
            "root_id": spec.root_id,
            "split": spec.split,
            "family": spec.family,
            "action_id": action.action_id,
            "action_name": action.name,
            "action_vec": action.as_vector(),
            "o_hist": _actor_hist_from_rollout(rollout),
            "h_ctx": mech["h_ctx"],
            "rho_imp": trans["rho_imp"],
            "harm_bin": trans["harm_bin"],
            "x_minus": trans["x_minus"],
            "x_plus": trans["x_plus"],
            "d_deg": mech["d_deg"],
            "z_mech": mech["z_mech"],
            "r_star": teacher["r_star"],
            "b_star": teacher["b_star"],
            "s_star": teacher["s_star"],
            "calib_group": mech["calib_group"],
            "metadata": {
                "town": spec.town,
                "seed": spec.seed,
                "transition_type": trans["transition_type"],
                "idx_minus": trans["idx_minus"],
                "idx_plus": trans["idx_plus"],
                "teacher": teacher["teacher"],
            },
        }
        return row

    def generate(self, specs: Sequence[RootScenarioSpec], output: str | Path, flush_every: int = 1) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("w", encoding="utf-8") as f:
            for spec in specs:
                actions = default_action_library(boundary_side=spec.boundary_side, corridor_side=-spec.boundary_side)
                for action in actions:
                    try:
                        rollout = self.rollout_action(spec, action)
                        row = self.rollout_to_row(spec, action, rollout)
                        f.write(json.dumps(row, separators=(",", ":")) + "\n")
                        count += 1
                        if count % flush_every == 0:
                            f.flush()
                        print(json.dumps({"row": count, "root_id": spec.root_id, "action": action.name, "s_star": row["s_star"], "harm_bin": row["harm_bin"]}))
                    except Exception as exc:
                        print(json.dumps({"error": str(exc), "root_id": spec.root_id, "action": action.name}))
                        continue

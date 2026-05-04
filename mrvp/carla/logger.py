from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def _try_import_carla():
    try:
        import carla  # type: ignore
        return carla
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CARLA Python API is not importable. Set PYTHONPATH to CARLA/PythonAPI or the carla egg.") from exc


def vector3_to_list(v) -> List[float]:
    return [float(v.x), float(v.y), float(v.z)]


def transform_to_dict(t) -> Dict[str, float]:
    return {
        "x": float(t.location.x),
        "y": float(t.location.y),
        "z": float(t.location.z),
        "roll": float(t.rotation.roll),
        "pitch": float(t.rotation.pitch),
        "yaw": math.radians(float(t.rotation.yaw)),
    }


def actor_state(actor, frame_time: float, control=None, road_info: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    tr = actor.get_transform()
    vel = actor.get_velocity()
    acc = actor.get_acceleration()
    ang = actor.get_angular_velocity()
    bbox = actor.bounding_box
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    out: Dict[str, Any] = {
        **transform_to_dict(tr),
        "vx": float(vel.x),
        "vy": float(vel.y),
        "vz": float(vel.z),
        "ax": float(acc.x),
        "ay": float(acc.y),
        "az": float(acc.z),
        "speed": float(speed),
        "yaw_rate": math.radians(float(ang.z)),
        "length": float(2.0 * bbox.extent.x),
        "width": float(2.0 * bbox.extent.y),
        "height": float(2.0 * bbox.extent.z),
        "t": float(frame_time),
    }
    if control is not None:
        out.update({"steer": float(control.steer), "throttle": float(control.throttle), "brake": float(control.brake)})
    else:
        out.update({"steer": 0.0, "throttle": 0.0, "brake": 0.0})
    if road_info:
        out.update(road_info)
    return out


class CarlaRolloutLogger:
    """Frame logger plus collision sensor callback for a CARLA rollout."""

    def __init__(self, world, ego, other_actors: List[Any], map_obj=None) -> None:
        self.world = world
        self.ego = ego
        self.other_actors = other_actors
        self.map = map_obj or world.get_map()
        self.frames: List[Dict[str, Any]] = []
        self.collision_events: List[Dict[str, Any]] = []
        self._collision_sensor = None

    def attach_collision_sensor(self) -> None:
        carla = _try_import_carla()
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        sensor.listen(self._on_collision)
        self._collision_sensor = sensor

    def destroy(self) -> None:
        if self._collision_sensor is not None:
            try:
                self._collision_sensor.stop()
                self._collision_sensor.destroy()
            except Exception:
                pass
            self._collision_sensor = None

    def _on_collision(self, event) -> None:
        other = event.other_actor
        self.collision_events.append(
            {
                "frame": int(event.frame),
                "other_actor_id": int(other.id) if other is not None else -1,
                "other_type_id": str(other.type_id) if other is not None else "unknown",
                "normal_impulse": vector3_to_list(event.normal_impulse),
                # CARLA collision events do not expose exact contact point; ego location is a safe fallback.
                "contact_point": vector3_to_list(self.ego.get_transform().location),
            }
        )

    def road_info(self, actor) -> Dict[str, float]:
        carla = _try_import_carla()
        loc = actor.get_transform().location
        wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return {"lane_width": 0.0, "lateral_offset": 0.0, "road_clearance": -1.0, "route_heading_error": 0.0}
        # Lateral offset from lane center in vehicle/map coordinates.
        center = wp.transform.location
        yaw = math.radians(wp.transform.rotation.yaw)
        right = np.asarray([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)], dtype=np.float32)
        delta = np.asarray([loc.x - center.x, loc.y - center.y], dtype=np.float32)
        lateral = float(delta.dot(right))
        ego_width = 2.0 * actor.bounding_box.extent.y
        clearance = float(wp.lane_width / 2.0 - abs(lateral) - ego_width / 2.0)
        ego_yaw = math.radians(actor.get_transform().rotation.yaw)
        return {
            "lane_width": float(wp.lane_width),
            "lateral_offset": lateral,
            "road_clearance": clearance,
            "route_heading_error": float(math.atan2(math.sin(ego_yaw - yaw), math.cos(ego_yaw - yaw))),
        }

    def log_frame(self, frame: int, elapsed_seconds: float, ego_control=None) -> None:
        ego_state = actor_state(self.ego, elapsed_seconds, ego_control, self.road_info(self.ego))
        actors = []
        for actor in list(self.other_actors):
            if actor is None or not actor.is_alive:
                continue
            try:
                actors.append({"id": int(actor.id), "type_id": str(actor.type_id), **actor_state(actor, elapsed_seconds)})
            except RuntimeError:
                continue
        # Approximate secondary hazard clearance from actor centers minus extents.
        sec_clear = 1000.0
        ex = np.asarray([ego_state["x"], ego_state["y"]], dtype=np.float32)
        for a in actors:
            ax = np.asarray([a["x"], a["y"]], dtype=np.float32)
            dist = float(np.linalg.norm(ex - ax) - 0.5 * ego_state["length"] - 0.5 * a["length"])
            sec_clear = min(sec_clear, dist)
        if not np.isfinite(sec_clear) or sec_clear > 999:
            sec_clear = 50.0
        margins = {
            "road": float(ego_state.get("road_clearance", 0.0)),
            "secondary": float(sec_clear),
            "stab": float(1.0 - abs(ego_state["yaw_rate"]) * 0.5 - abs(ego_state.get("route_heading_error", 0.0)) * 0.25),
            "ctrl": float(1.0 - abs(ego_state.get("steer", 0.0))),
            "return": float(max(0.0, 30.0 - abs(ego_state.get("lateral_offset", 0.0)) * 4.0)),
        }
        self.frames.append({"frame": int(frame), "t": float(elapsed_seconds), "ego": ego_state, "actors": actors, "margins": margins})

    def as_rollout(self) -> Dict[str, Any]:
        return {"frames": self.frames, "collision_events": self.collision_events}

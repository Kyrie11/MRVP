from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .teacher_mpc import rollout_bicycle


@dataclass
class CarlaRootState:
    x_t: np.ndarray
    actors: np.ndarray
    seed: int
    family: str


class CarlaAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = None
        self.world = None
        self.dt = float(cfg.get("fixed_delta_seconds", cfg.get("prefix_dt", 0.1)))
        self.root: CarlaRootState | None = None

    def connect(self) -> None:
        try:
            import carla  # type: ignore
            self.client = carla.Client(self.cfg.get("host", "localhost"), int(self.cfg.get("port", 2000)))
            self.client.set_timeout(float(self.cfg.get("timeout", 10.0)))
        except Exception as exc:
            raise RuntimeError("CARLA Python package or server is not available") from exc

    def load_world(self, town_name: str) -> None:
        if self.client is None:
            self.connect()
        self.world = self.client.load_world(town_name)

    def set_synchronous(self, fixed_delta_seconds: float) -> None:
        self.dt = float(fixed_delta_seconds)
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)
            try:
                tm = self.client.get_trafficmanager(int(self.cfg.get("tm_port", 8000)))
                tm.set_synchronous_mode(True)
            except Exception:
                return

    def reset_root(self, root_spec: dict) -> None:
        rng = np.random.default_rng(int(root_spec.get("seed", 0)))
        x_t = np.asarray(root_spec.get("x_t", [0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=np.float32)
        actors = rng.normal(0, 1, size=(int(root_spec.get("actors", 6)), 8)).astype(np.float32)
        self.root = CarlaRootState(x_t=x_t, actors=actors, seed=int(root_spec.get("seed", 0)), family=str(root_spec.get("family", "SC")))

    def snapshot_root_state(self) -> CarlaRootState:
        if self.root is None:
            raise RuntimeError("root state is not initialized")
        return CarlaRootState(self.root.x_t.copy(), self.root.actors.copy(), self.root.seed, self.root.family)

    def restore_root_state(self, snapshot: CarlaRootState) -> None:
        self.root = CarlaRootState(snapshot.x_t.copy(), snapshot.actors.copy(), snapshot.seed, snapshot.family)

    def apply_prefix(self, action) -> dict[str, Any]:
        if self.root is None:
            raise RuntimeError("root state is not initialized")
        traj = rollout_bicycle(self.root.x_t, action.controls, dt=self.dt)
        return {"trajectory": traj, "controls": action.controls.copy()}

    def tick(self, n: int = 1) -> None:
        if self.world is not None:
            for _ in range(int(n)):
                self.world.tick()

    def query_map_context(self, ego_transform, crop_m: float = 80.0, res_m: float = 0.25) -> dict[str, Any]:
        size = int(round(crop_m / max(res_m, 1e-6)))
        return {"crop_m": crop_m, "res_m": res_m, "size": size}

    def query_actor_states(self) -> list[dict[str, Any]]:
        if self.root is None:
            return []
        return [{"state": row.copy()} for row in self.root.actors]

    def query_contacts(self) -> list[dict[str, Any]]:
        return []

    def destroy_all(self) -> None:
        if self.world is None:
            return
        actors = self.world.get_actors()
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                continue

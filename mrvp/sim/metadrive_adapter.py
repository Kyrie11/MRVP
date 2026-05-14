from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .teacher_mpc import rollout_bicycle


@dataclass
class RootState:
    x_t: np.ndarray
    actors: np.ndarray
    seed: int
    family: str


class MetaDriveAdapter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg.get("prefix_dt", 0.1))
        self.root: RootState | None = None
        self._external = None
        try:
            import metadrive  # type: ignore
            self._external = metadrive
        except Exception:
            self._external = None

    def reset_root(self, root_spec: dict) -> None:
        rng = np.random.default_rng(int(root_spec.get("seed", 0)))
        x_t = np.asarray(root_spec.get("x_t", [0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0]), dtype=np.float32)
        actors = rng.normal(0, 1, size=(int(root_spec.get("actors", 6)), 8)).astype(np.float32)
        self.root = RootState(x_t=x_t, actors=actors, seed=int(root_spec.get("seed", 0)), family=str(root_spec.get("family", "SC")))

    def snapshot_root_state(self) -> RootState:
        if self.root is None:
            raise RuntimeError("root state is not initialized")
        return RootState(self.root.x_t.copy(), self.root.actors.copy(), self.root.seed, self.root.family)

    def restore_root_state(self, snapshot: RootState) -> None:
        self.root = RootState(snapshot.x_t.copy(), snapshot.actors.copy(), snapshot.seed, snapshot.family)

    def apply_prefix(self, action) -> dict[str, Any]:
        if self.root is None:
            raise RuntimeError("root state is not initialized")
        traj = rollout_bicycle(self.root.x_t, action.controls, dt=self.dt)
        return {"trajectory": traj, "controls": action.controls.copy()}

    def step_recovery(self, control) -> dict[str, Any]:
        if self.root is None:
            raise RuntimeError("root state is not initialized")
        traj = rollout_bicycle(self.root.x_t, np.asarray(control, dtype=np.float32)[None, :], dt=self.dt)
        self.root.x_t = traj[-1]
        return {"state": self.root.x_t.copy()}

    def query_map_context(self, ego_pose, crop_m: float = 80.0, res_m: float = 0.25) -> dict[str, Any]:
        size = int(round(crop_m / max(res_m, 1e-6)))
        return {"crop_m": crop_m, "res_m": res_m, "size": size}

    def query_actor_states(self) -> list[dict[str, Any]]:
        if self.root is None:
            return []
        return [{"state": row.copy()} for row in self.root.actors]

    def query_contacts(self) -> list[dict[str, Any]]:
        return []

    def set_friction(self, mu: float) -> None:
        self.cfg["friction_mu"] = float(mu)

    def set_degradation(self, deg) -> None:
        self.cfg["degradation"] = np.asarray(deg, dtype=np.float32).tolist()

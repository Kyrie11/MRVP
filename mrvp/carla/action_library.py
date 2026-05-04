from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from mrvp.data.schema import ACTION_NAMES, ACTION_VECTORS


@dataclass(frozen=True)
class EmergencyAction:
    action_id: int
    name: str
    steer: float
    throttle: float
    brake: float
    duration: float

    def as_vector(self):
        return [self.steer, self.throttle, self.brake, self.duration]

    def to_carla_control(self):
        try:
            import carla  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CARLA Python API is not importable. Set PYTHONPATH to the CARLA egg/PythonAPI.") from exc
        return carla.VehicleControl(
            steer=float(self.steer),
            throttle=float(self.throttle),
            brake=float(self.brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )


def default_action_library(boundary_side: int = 1, corridor_side: int = -1) -> List[EmergencyAction]:
    actions = []
    for i, name in enumerate(ACTION_NAMES):
        steer, throttle, brake, duration = ACTION_VECTORS[i].tolist()
        if name == "boundary_side_steer":
            steer = 0.55 * float(boundary_side)
        elif name == "corridor_side_steer":
            steer = 0.55 * float(corridor_side)
        actions.append(EmergencyAction(i, name, float(steer), float(throttle), float(brake), float(duration)))
    return actions

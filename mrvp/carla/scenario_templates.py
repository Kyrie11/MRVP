from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List

import numpy as np

SCENARIO_FAMILIES = [
    "rear_end_blocked_forward_corridor",
    "side_swipe_near_boundary",
    "oblique_intersection_impact",
    "cut_in_unavoidable_contact",
    "boundary_critical_non_contact",
    "low_friction_recovery",
    "actuator_degradation_after_impact",
    "dense_agent_secondary_exposure",
]


@dataclass
class RootScenarioSpec:
    root_id: str
    family: str
    seed: int
    town: str
    spawn_index: int
    weather_id: int
    friction: float
    trigger_time: float
    ego_speed: float
    actor_density: int
    boundary_side: int
    degradation: Dict[str, float]
    split: str

    def to_dict(self) -> Dict:
        return asdict(self)


def root_split(i: int, n: int) -> str:
    f = i / max(n, 1)
    if f < 0.70:
        return "train"
    if f < 0.82:
        return "val"
    if f < 0.90:
        return "cal"
    return "test"


def generate_root_specs(n_roots: int, seed: int = 13, towns: Iterable[str] = ("Town03", "Town04", "Town05", "Town10HD")) -> List[RootScenarioSpec]:
    rng = np.random.default_rng(seed)
    towns = list(towns)
    specs: List[RootScenarioSpec] = []
    for i in range(n_roots):
        family = SCENARIO_FAMILIES[i % len(SCENARIO_FAMILIES)]
        friction = float(rng.choice([0.95, 0.75, 0.45, 0.30], p=[0.50, 0.25, 0.15, 0.10]))
        if family == "low_friction_recovery":
            friction = float(rng.uniform(0.25, 0.55))
        damage_class = int(rng.integers(0, 4)) if family == "actuator_degradation_after_impact" else int(rng.integers(0, 2))
        degradation = {
            "steering_scale": max(0.25, 1.0 - 0.15 * damage_class),
            "brake_scale": max(0.25, 1.0 - 0.12 * damage_class),
            "throttle_scale": max(0.25, 1.0 - 0.08 * damage_class),
            "delay": 0.04 + 0.04 * damage_class,
            "friction": friction,
            "damage_class": float(damage_class),
        }
        specs.append(
            RootScenarioSpec(
                root_id=f"carla_{i:07d}",
                family=family,
                seed=int(seed * 100000 + i),
                town=towns[i % len(towns)],
                spawn_index=int(rng.integers(0, 250)),
                weather_id=int(rng.integers(0, 14)),
                friction=friction,
                trigger_time=float(rng.uniform(1.0, 3.0)),
                ego_speed=float(rng.uniform(7.0, 18.0)),
                actor_density=int(rng.integers(6, 35)) if family == "dense_agent_secondary_exposure" else int(rng.integers(2, 15)),
                boundary_side=int(rng.choice([-1, 1])),
                degradation=degradation,
                split=root_split(i, n_roots),
            )
        )
    return specs

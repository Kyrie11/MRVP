from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATE_DIM = 12
DEG_DIM = 8
CONTROL_DIM = 3
CERT_NAMES = ["road", "col", "ctrl", "stab", "goal"]
SCENARIO_FAMILIES = ["SC", "LHB", "CI", "BLE", "LF", "AD", "OSH", "NCR"]
ACTION_IDS = [
    "hard_brake",
    "brake_left",
    "brake_right",
    "maintain",
    "mild_left",
    "mild_right",
    "boundary_away",
    "corridor_seeking",
]


@dataclass
class MRVPRow:
    root_id: str
    action_id: str
    sim_source: str
    scenario_family: str
    seed: int
    trigger_time: float
    o_hist: np.ndarray
    h_ctx: dict[str, Any]
    x_t: np.ndarray
    action_params: dict[str, Any]
    prefix_rollout: np.ndarray
    prefix_controls: np.ndarray
    contact: bool
    contact_time: float
    contact_actor_id: int
    contact_normal: np.ndarray
    impulse: float
    delta_v: float
    rho_imp: float
    harm_bin: int
    tau_reset: float
    tau_index: int
    r_reset: np.ndarray
    deg: np.ndarray
    world_reset: dict[str, np.ndarray]
    teacher_u: np.ndarray
    teacher_traj: np.ndarray
    cert_star: np.ndarray
    score_star: float
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "action_id": self.action_id,
            "sim_source": self.sim_source,
            "scenario_family": self.scenario_family,
            "seed": int(self.seed),
            "trigger_time": float(self.trigger_time),
            "o_hist": self.o_hist,
            "h_ctx": self.h_ctx,
            "x_t": self.x_t,
            "action_params": self.action_params,
            "prefix_rollout": self.prefix_rollout,
            "prefix_controls": self.prefix_controls,
            "contact": bool(self.contact),
            "contact_time": float(self.contact_time),
            "contact_actor_id": int(self.contact_actor_id),
            "contact_normal": self.contact_normal,
            "impulse": float(self.impulse),
            "delta_v": float(self.delta_v),
            "rho_imp": float(self.rho_imp),
            "harm_bin": int(self.harm_bin),
            "tau_reset": float(self.tau_reset),
            "tau_index": int(self.tau_index),
            "r_reset": self.r_reset,
            "deg": self.deg,
            "world_reset": self.world_reset,
            "teacher_u": self.teacher_u,
            "teacher_traj": self.teacher_traj,
            "cert_star": self.cert_star,
            "score_star": float(self.score_star),
            "audit": self.audit,
        }


def require_row_fields(row: dict[str, Any]) -> None:
    required = [
        "root_id", "action_id", "sim_source", "scenario_family", "seed", "trigger_time",
        "o_hist", "h_ctx", "x_t", "prefix_rollout", "prefix_controls", "rho_imp",
        "harm_bin", "tau_reset", "r_reset", "deg", "world_reset", "teacher_u",
        "teacher_traj", "cert_star", "score_star", "audit",
    ]
    missing = [k for k in required if k not in row]
    if missing:
        raise KeyError(f"row missing fields: {missing}")


def row_to_numeric_summary(row: dict[str, Any]) -> dict[str, float]:
    return {
        "rho_imp": float(row["rho_imp"]),
        "harm_bin": float(row["harm_bin"]),
        "score_star": float(row["score_star"]),
        "contact": float(bool(row.get("contact", False))),
    }

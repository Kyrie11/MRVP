from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HarmBinner:
    harm_type: str = "delta_v"
    num_contact_bins: int = 5
    no_contact_bin: int = 0
    edges: np.ndarray | None = None

    def fit(self, rho_imp: np.ndarray, contact: np.ndarray) -> "HarmBinner":
        values = np.asarray(rho_imp, dtype=np.float32)[np.asarray(contact).astype(bool)]
        if values.size == 0:
            self.edges = np.linspace(0.1, 1.0, self.num_contact_bins + 1, dtype=np.float32)[1:-1]
            return self
        qs = np.linspace(0.0, 1.0, self.num_contact_bins + 1)[1:-1]
        self.edges = np.quantile(values, qs).astype(np.float32)
        self.edges = np.maximum.accumulate(self.edges)
        return self

    def assign(self, rho_imp: float, contact: bool) -> int:
        if not contact:
            return int(self.no_contact_bin)
        if self.edges is None:
            raise ValueError("harm binner must be fitted before assignment")
        return int(1 + np.searchsorted(self.edges, float(rho_imp), side="right"))

    def assign_many(self, rho_imp: np.ndarray, contact: np.ndarray) -> np.ndarray:
        return np.array([self.assign(r, bool(c)) for r, c in zip(rho_imp, contact)], dtype=np.int64)

    def to_dict(self) -> dict:
        return {"harm_type": self.harm_type, "num_contact_bins": self.num_contact_bins, "no_contact_bin": self.no_contact_bin, "edges": [] if self.edges is None else self.edges.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "HarmBinner":
        obj = cls(data.get("harm_type", "delta_v"), int(data.get("num_contact_bins", 5)), int(data.get("no_contact_bin", 0)))
        edges = data.get("edges", [])
        obj.edges = np.asarray(edges, dtype=np.float32) if edges else None
        return obj


def compute_rho(impact_speed: float, impulse: float, delta_v: float, harm_type: str = "delta_v") -> float:
    if harm_type == "speed":
        return float(abs(impact_speed))
    if harm_type == "impulse":
        return float(abs(impulse))
    return float(abs(delta_v))


def construct_harm_comparable_set(rows_for_root: list[dict]) -> list[dict]:
    if not rows_for_root:
        return []
    min_bin = min(int(row["harm_bin"]) for row in rows_for_root)
    return [row for row in rows_for_root if int(row["harm_bin"]) == min_bin]

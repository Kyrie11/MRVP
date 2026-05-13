from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .degraded_mpc import TeacherResult, solve_recovery_teacher


def solve_cem_recovery_teacher(
    reset_state: np.ndarray,
    recovery_world: Mapping[str, Any] | np.ndarray | None,
    degradation: np.ndarray,
    horizon: int = 30,
    dt: float = 0.1,
    population: int = 256,
    elites: int = 32,
    iterations: int = 3,
    seed: Optional[int] = None,
) -> TeacherResult:
    """CEM-style wrapper around the degraded shooting teacher.

    The current implementation exposes the CEM interface used by dataset code
    and delegates the rollout/scoring to the dependency-free degraded teacher.
    Increasing ``population`` and ``iterations`` improves the sampled search;
    ``elites`` is kept for API compatibility with stronger simulator teachers.
    """
    num_sequences = max(int(population), int(elites), 1) * max(int(iterations), 1)
    return solve_recovery_teacher(
        reset_state=reset_state,
        recovery_world=recovery_world,
        degradation=degradation,
        horizon=horizon,
        dt=dt,
        num_sequences=num_sequences,
        seed=seed,
    )

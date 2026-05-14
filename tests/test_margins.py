import numpy as np

from mrvp.sim.degradation import nominal_degradation
from mrvp.sim.margins import compute_rollout_margins


def test_margins_score_keys():
    world = {
        "A": np.ones((6, 16, 16), dtype=np.float32),
        "O": np.zeros((3, 1, 16, 16), dtype=np.float32),
        "G": np.ones((3, 16, 16), dtype=np.float32),
    }
    traj = np.zeros((4, 12), dtype=np.float32)
    controls = np.zeros((3, 3), dtype=np.float32)
    m = compute_rollout_margins(traj, controls, world, nominal_degradation())
    assert set(["road", "col", "ctrl", "stab", "goal", "score"]).issubset(m.keys())

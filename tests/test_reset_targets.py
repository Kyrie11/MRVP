import numpy as np

from mrvp.sim.reset_targets import extract_reset


def test_extract_reset_contact_and_non_contact():
    traj = np.zeros((11, 12), dtype=np.float32)
    traj[:, 3] = np.linspace(10, 1, 11)
    idx, tau, r = extract_reset(traj, True, 0.2, 0.1)
    assert 0 <= idx < len(traj)
    idx2, tau2, r2 = extract_reset(traj, False, -1, 0.1)
    assert 0 <= idx2 < len(traj)
    assert r.shape[0] == 12 and r2.shape[0] == 12

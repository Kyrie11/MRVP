import numpy as np

from mrvp.sim.harm import HarmBinner, construct_harm_comparable_set


def test_harm_binner_and_safe_set():
    b = HarmBinner(num_contact_bins=2).fit(np.array([0, 1, 2, 3]), np.array([False, True, True, True]))
    bins = b.assign_many(np.array([0, 1, 3]), np.array([False, True, True]))
    assert bins[0] == 0
    rows = [{"harm_bin": int(v), "action_id": str(i)} for i, v in enumerate([2, 1, 1])]
    safe = construct_harm_comparable_set(rows)
    assert [r["action_id"] for r in safe] == ["1", "2"]

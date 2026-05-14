from mrvp.models.selectors import lower_tail_cvar


def test_lcvar_low_tail():
    assert lower_tail_cvar([3, 1, 2, 0], 0.5) == 0.5
    assert lower_tail_cvar([5], 0.1) == 5.0

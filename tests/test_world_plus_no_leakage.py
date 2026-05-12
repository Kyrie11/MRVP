from mrvp.data.synthetic import make_synthetic_rows
from mrvp.data.schema import validate_row_no_leakage

def test_world_plus_no_target_keys():
    rows=make_synthetic_rows(n_roots=2,seed=1)
    fatal=[]
    for r in rows:
        fatal += [w for w in validate_row_no_leakage(r) if w.startswith('world_plus_mentions_') or w=='event_tokens_match_audit_mech']
    assert not fatal

from collections import defaultdict
from mrvp.data.synthetic import make_synthetic_rows

def test_root_split_unique():
    d=defaultdict(set)
    for r in make_synthetic_rows(20,seed=3): d[r['root_id']].add(r['split'])
    assert all(len(v)==1 for v in d.values())

from pathlib import Path
from mrvp.data.synthetic import make_synthetic_rows, write_jsonl
from mrvp.data.dataset import MRVPDataset
from mrvp.data.pairs import SameRootPairDataset

def test_pair_dataset_uses_admissible(tmp_path: Path):
    p=tmp_path/'d.jsonl'; write_jsonl(make_synthetic_rows(16,seed=2),p); ds=MRVPDataset(p,'train'); pairs=SameRootPairDataset(ds,eps_s=0.01)
    i,j=pairs.pairs[0]
    assert ds.rows[i]['is_harm_admissible'] and ds.rows[j]['is_harm_admissible']
    assert ds.rows[i]['harm_bin']==ds.rows[j]['harm_bin']

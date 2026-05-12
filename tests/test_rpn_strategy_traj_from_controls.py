import torch
from torch.utils.data import DataLoader
from mrvp.data.synthetic import make_synthetic_rows,write_jsonl
from mrvp.data.dataset import MRVPDataset,mrvp_collate
from mrvp.models.rpn import RecoveryProfileNetwork

def test_rpn_strategy_traj_from_controls(tmp_path):
    p=tmp_path/'d.jsonl'; write_jsonl(make_synthetic_rows(n_roots=4,seed=3),p); b=next(iter(DataLoader(MRVPDataset(p,'train'),batch_size=4,collate_fn=mrvp_collate))); rpn=RecoveryProfileNetwork(hidden_dim=32); out=rpn(b); rollout=rpn.dynamics(b['x_plus'].float(),out['strategy_u'],b['deg'].float()); assert torch.allclose(out['strategy_traj'],rollout,atol=1e-6); assert torch.allclose(out['strategy_traj'][:,:,0,:],b['x_plus'].float().unsqueeze(1),atol=1e-6)

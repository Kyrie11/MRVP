import torch
from mrvp.models.rpn import RecoveryProfileNetwork

def test_rpn_traj_matches_rollout():
    b=2; m=RecoveryProfileNetwork(hidden_dim=32,strategy_count=2,recovery_horizon=4); batch={'o_hist':torch.zeros(b,10,16,9),'h_ctx':torch.zeros(b,32),'x_plus':torch.zeros(b,12),'deg':torch.ones(b,6),'d_deg':torch.ones(b,6),'event_tokens':torch.zeros(b,16,32),'world_plus':torch.zeros(b,64)}; batch['x_plus'][:,3]=5
    out=m(batch); expected=m.rollout(batch['x_plus'],out['strategy_u'],batch['deg'])
    assert torch.allclose(out['strategy_traj'], expected)

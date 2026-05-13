import torch
from mrvp.models.dynamics import DegradedBicycleRollout

def test_rollout_starts_at_xplus_and_is_differentiable():
    roll=DegradedBicycleRollout(); x=torch.zeros(2,12); x[:,3]=5.0; u=torch.zeros(2,3,5,3,requires_grad=True); d=torch.tensor([[1,1,1,0.05,0.9,0],[1,1,1,0.05,0.9,0]],dtype=torch.float32); traj=roll(x,u,d)
    assert torch.allclose(traj[:,:,0,:], x[:,None,:].expand_as(traj[:,:,0,:]))
    loss=traj[...,0].sum(); loss.backward(); assert u.grad is not None

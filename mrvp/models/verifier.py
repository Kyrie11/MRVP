from __future__ import annotations
import torch
from torch import nn
from mrvp.data.schema import STATE_DIM,WORLD_DIM,DEG_DIM
from .common import make_mlp
class StrategyMarginVerifier(nn.Module):
    def __init__(self,state_dim:int=STATE_DIM,world_dim:int=WORLD_DIM,deg_dim:int=DEG_DIM,hidden_dim:int=256):
        super().__init__(); self.state_dim=state_dim; self.learned=make_mlp(state_dim*4+world_dim+deg_dim,[hidden_dim,hidden_dim],3,dropout=0.05,layer_norm=True)
    def _traj_features(self,traj): return torch.cat([traj[:,:,0,:],traj[:,:,-1,:],traj.mean(2),traj.min(2).values],-1)
    def forward(self,traj:torch.Tensor,controls:torch.Tensor,world_plus:torch.Tensor,deg:torch.Tensor)->torch.Tensor:
        b,l,_,d=traj.shape; feat=self._traj_features(traj); w=world_plus[:,None,:].expand(b,l,world_plus.shape[-1]); dg=deg[:,None,:].expand(b,l,deg.shape[-1]); learned=self.learned(torch.cat([feat,w,dg],-1)); sec,road,ret=learned[...,0],learned[...,1],learned[...,2]
        speed=torch.sqrt(traj[...,3].square()+traj[...,4].square()).clamp_min(0.1); yaw_rate=traj[...,5].abs(); beta=traj[...,8].abs(); friction=deg[:,None,None,4].clamp_min(0.1); yaw_limit=friction*9.81/speed; stab=torch.minimum(0.45-beta,yaw_limit-yaw_rate).min(2).values
        steer_scale=deg[:,None,None,0].clamp_min(0.05); brake_scale=deg[:,None,None,1].clamp_min(0.05); throttle_scale=deg[:,None,None,2].clamp_min(0.05); ctrl=torch.minimum(torch.minimum(steer_scale-controls[...,0].abs(),brake_scale-controls[...,1].clamp_min(0)), throttle_scale-controls[...,2].clamp_min(0)).min(2).values
        return torch.stack([sec,road,stab,ctrl,ret],-1)

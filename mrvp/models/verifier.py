from __future__ import annotations
import torch
from torch import nn
from mrvp.data.schema import BOTTLE_NECKS, CONTROL_DIM, DEG_DIM, STATE_DIM, WORLD_DIM, CTX_DIM
from .common import make_mlp
class StrategyMarginVerifier(nn.Module):
    def __init__(self,state_dim:int=STATE_DIM,world_dim:int=WORLD_DIM,deg_dim:int=DEG_DIM,ctx_dim:int=CTX_DIM,hidden_dim:int=128,bottlenecks:int=len(BOTTLE_NECKS),dropout:float=0.05):
        super().__init__(); self.learned=make_mlp(18+world_dim+deg_dim+ctx_dim,[hidden_dim,hidden_dim],3,dropout=dropout,layer_norm=True); self.bottlenecks=bottlenecks
    def _traj_features(self,traj:torch.Tensor,controls:torch.Tensor,deg:torch.Tensor)->torch.Tensor:
        px,py,psi,vx,vy,yaw,beta=traj[...,0],traj[...,1],traj[...,2],traj[...,3],traj[...,4],traj[...,5],traj[...,8]
        speed=torch.sqrt(vx*vx+vy*vy+1e-6); feat=torch.stack([py.abs().amax(-1),py.abs().mean(-1),psi.abs().amax(-1),yaw.abs().amax(-1),beta.abs().amax(-1),speed.amax(-1),speed.mean(-1),px[...,-1]-px[...,0],controls[...,0].abs().amax(-1),controls[...,1].amax(-1),controls[...,2].amax(-1),traj[...,6].abs().mean(-1),traj[...,7].abs().mean(-1),py[...,-1].abs(),psi[...,-1].abs(),deg[:,None,0].expand_as(px[...,0]),deg[:,None,1].expand_as(px[...,0]),deg[:,None,4].expand_as(px[...,0])],-1)
        return feat
    def forward(self,traj:torch.Tensor,controls:torch.Tensor,world_plus:torch.Tensor,deg:torch.Tensor,h_ctx:torch.Tensor)->torch.Tensor:
        b,l=traj.shape[:2]; feat=self._traj_features(traj,controls,deg); w=world_plus[:,None,:].expand(b,l,-1); d=deg[:,None,:].expand(b,l,-1); h=h_ctx[:,None,:].expand(b,l,-1)
        learned=self.learned(torch.cat([feat,w,d,h],-1))
        speed=torch.sqrt(traj[...,3]**2+traj[...,4]**2+1e-6); yaw_abs=traj[...,5].abs(); beta_abs=traj[...,8].abs(); mu=deg[:,None,None,4].clamp_min(0.1); yaw_lim=mu*9.81/speed.clamp_min(2.0); stab=torch.minimum(0.45-beta_abs, yaw_lim-yaw_abs).amin(-1)
        steer_margin=deg[:,None,None,0]-controls[...,0].abs(); brake_margin=deg[:,None,None,1]-controls[...,1].clamp_min(0); throttle_margin=deg[:,None,None,2]-controls[...,2].clamp_min(0); ctrl=torch.minimum(torch.minimum(steer_margin,brake_margin),throttle_margin).amin(-1)-0.05*deg[:,None,3]
        sec=learned[...,0]; road=learned[...,1]; ret=learned[...,2]
        return torch.stack([sec,road,stab,ctrl,ret],-1)

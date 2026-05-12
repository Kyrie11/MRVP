from __future__ import annotations
import torch
from torch import nn
class DegradedBicycleRollout(nn.Module):
    def __init__(self,dt:float=0.1,wheelbase:float=2.8,state_dim:int=12): super().__init__(); self.dt=dt; self.wheelbase=wheelbase; self.state_dim=state_dim
    def forward(self,x0:torch.Tensor,controls:torch.Tensor,deg:torch.Tensor)->torch.Tensor:
        b,l,h,_=controls.shape; dtype=x0.dtype; device=x0.device; x0b=x0[:,None,:].expand(b,l,self.state_dim)
        px=x0b[...,0]; py=x0b[...,1]; yaw=x0b[...,2]; vx=x0b[...,3]; vy=x0b[...,4]; speed=torch.sqrt(vx.square()+vy.square()).clamp_min(0.0); yaw_rate=x0b[...,5]; beta=x0b[...,8] if self.state_dim>8 else torch.zeros_like(px)
        steer_scale=deg[:,0].view(b,1).clamp_min(0.05); brake_scale=deg[:,1].view(b,1).clamp_min(0.05); throttle_scale=deg[:,2].view(b,1).clamp_min(0.05); friction=deg[:,4].view(b,1).clamp_min(0.1)
        states=[x0b]
        for k in range(h):
            u=controls[:,:,k,:]; delta=(steer_scale*u[...,0].clamp(-1,1)).clamp(-0.75,0.75); brake=(brake_scale*u[...,1].clamp(0,1)); throttle=(throttle_scale*u[...,2].clamp(0,1))
            accel=(2.5*throttle-5.0*brake).clamp(min=-100,max=100); accel=torch.maximum(torch.minimum(accel,friction*3.5),-friction*9.81); speed=(speed+accel*self.dt).clamp_min(0.0)
            raw=speed/self.wheelbase*torch.tan(delta); yaw_lim=(friction*9.81/speed.clamp_min(2.0)).clamp_min(0.15); yaw_rate=raw.clamp(-yaw_lim,yaw_lim); yaw=yaw+yaw_rate*self.dt; beta=torch.atan(0.5*torch.tan(delta)); vx=speed*torch.cos(yaw+beta); vy=speed*torch.sin(yaw+beta); px=px+vx*self.dt; py=py+vy*self.dt; ax=accel*torch.cos(yaw+beta); ay=accel*torch.sin(yaw+beta)
            st=torch.zeros(b,l,self.state_dim,device=device,dtype=dtype); st[...,0]=px; st[...,1]=py; st[...,2]=yaw; st[...,3]=vx; st[...,4]=vy; st[...,5]=yaw_rate; st[...,6]=ax; st[...,7]=ay; st[...,8]=beta; st[...,9]=delta; st[...,10]=brake; st[...,11]=throttle; states.append(st)
        return torch.stack(states,dim=2)

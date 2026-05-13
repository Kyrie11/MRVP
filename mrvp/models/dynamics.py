from __future__ import annotations
import torch
from torch import nn
class DegradedBicycleRollout(nn.Module):
    """Differentiable rollout in paper state layout.
    x=[p_x,p_y,psi,v_x,v_y,yaw_rate,a_x,a_y,beta,delta,F_b,F_x]
    controls=[delta,F_b,F_x].
    """
    def __init__(self,dt:float=0.1,wheelbase:float=2.8): super().__init__(); self.dt=dt; self.wheelbase=wheelbase
    def forward(self,x0:torch.Tensor,controls:torch.Tensor,deg:torch.Tensor)->torch.Tensor:
        b,l,h,_=controls.shape; x=x0[:,None,:].expand(b,l,x0.shape[-1]).clone(); traj=[x]
        steer_scale=deg[:,0].view(b,1).clamp_min(0.05); brake_scale=deg[:,1].view(b,1).clamp_min(0.05); throttle_scale=deg[:,2].view(b,1).clamp_min(0.05); friction=deg[:,4].view(b,1).clamp_min(0.1)
        for k in range(h):
            u=controls[:,:,k,:]; delta=steer_scale*torch.tanh(u[...,0]); fb=brake_scale*torch.sigmoid(u[...,1]); fx=throttle_scale*torch.sigmoid(u[...,2])
            px,py,psi,vx,vy=x[...,0],x[...,1],x[...,2],x[...,3],x[...,4]
            speed=torch.sqrt(vx*vx+vy*vy+1e-6); acc=torch.minimum(torch.maximum(2.8*fx-6.5*fb, -7.5*friction), torch.full_like(fx, 3.0))
            speed2=torch.clamp(speed+acc*self.dt,min=0.0); raw_yaw=speed2/self.wheelbase*torch.tan(delta.clamp(-0.75,0.75)); yaw_lim=torch.clamp(friction*9.81/(speed2.clamp_min(2.0)),min=0.15); yaw_rate=raw_yaw.clamp(-1,1)*yaw_lim
            beta=torch.atan(0.45*torch.tan(delta.clamp(-0.75,0.75))).clamp(-0.65,0.65); psi2=psi+yaw_rate*self.dt; vx2=speed2*torch.cos(psi2+beta); vy2=speed2*torch.sin(psi2+beta); px2=px+vx2*self.dt; py2=py+vy2*self.dt; ax=(vx2-vx)/self.dt; ay=(vy2-vy)/self.dt
            x=torch.stack([px2,py2,psi2,vx2,vy2,yaw_rate,ax,ay,beta,delta,fb,fx],-1); traj.append(x)
        return torch.stack(traj,dim=2)

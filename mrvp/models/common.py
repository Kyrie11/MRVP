from __future__ import annotations
from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale): ctx.scale=float(scale); return x.view_as(x)
    @staticmethod
    def backward(ctx,grad_output): return -ctx.scale*grad_output, None
def grad_reverse(x:torch.Tensor, scale:float=1.0)->torch.Tensor: return _GradientReverseFn.apply(x,float(scale))
def make_mlp(in_dim:int, hidden:Sequence[int], out_dim:int, dropout:float=0.0, layer_norm:bool=False)->nn.Sequential:
    layers=[]; last=in_dim
    for h in hidden:
        layers.append(nn.Linear(last,h));
        if layer_norm: layers.append(nn.LayerNorm(h))
        layers.append(nn.SiLU());
        if dropout>0: layers.append(nn.Dropout(dropout))
        last=h
    layers.append(nn.Linear(last,out_dim)); return nn.Sequential(*layers)
class SmoothMin(nn.Module):
    def __init__(self,tau:float=0.1): super().__init__(); self.tau=tau
    def forward(self,x:torch.Tensor,dim:int=-1)->torch.Tensor: return -max(float(self.tau),1e-6)*torch.logsumexp(-x/max(float(self.tau),1e-6),dim=dim)
def diag_gaussian_nll(x,mean,log_scale):
    ls=log_scale.clamp(-7,5); inv=torch.exp(-2*ls); return 0.5*((x-mean)**2*inv+2*ls).sum(dim=-1)
def mixture_diag_gaussian_nll(x,logits,mean,log_scale):
    comp=diag_gaussian_nll(x[:,None,:].expand_as(mean),mean,log_scale); return -torch.logsumexp(F.log_softmax(logits,dim=-1)-comp,dim=-1)
def empirical_cvar(losses:torch.Tensor,beta:float=0.9,dim:int=-1)->torch.Tensor:
    n=losses.shape[dim]; k=max(1,int(torch.ceil(torch.tensor((1.0-beta)*n)).item())); return torch.topk(losses,k=k,dim=dim).values.mean(dim=dim)

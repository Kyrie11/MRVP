from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from mrvp.data.schema import BOTTLE_NECKS,CONTROL_DIM,DEG_DIM,RECOVERY_HORIZON,STATE_DIM,STRATEGY_COUNT,TOKEN_COUNT,TOKEN_DIM,WORLD_DIM
from .common import SmoothMin, make_mlp
from .context import SceneContextEncoder
from .dynamics import DegradedBicycleRollout
from .verifier import StrategyMarginVerifier
class RecoveryProfileNetwork(nn.Module):
    def __init__(self,state_dim:int=STATE_DIM,deg_dim:int=DEG_DIM,token_count:int=TOKEN_COUNT,token_dim:int=TOKEN_DIM,world_dim:int=WORLD_DIM,control_dim:int=CONTROL_DIM,strategy_count:int=STRATEGY_COUNT,recovery_horizon:int=RECOVERY_HORIZON,ctx_emb_dim:int=128,hidden_dim:int=256,bottlenecks:int=len(BOTTLE_NECKS),dropout:float=0.05,scalar:bool=False):
        super().__init__(); self.state_dim=state_dim; self.deg_dim=deg_dim; self.token_count=token_count; self.token_dim=token_dim; self.world_dim=world_dim; self.control_dim=control_dim; self.strategy_count=strategy_count; self.recovery_horizon=recovery_horizon; self.bottlenecks=bottlenecks; self.scalar=scalar
        self.context_encoder=SceneContextEncoder(out_dim=ctx_emb_dim); self.x_encoder=make_mlp(state_dim,[128],128,dropout=dropout,layer_norm=True); self.d_encoder=make_mlp(deg_dim,[64],64,dropout=dropout,layer_norm=True); self.token_encoder=make_mlp(token_dim*2,[128],128,dropout=dropout,layer_norm=True); self.world_encoder=make_mlp(world_dim,[128],128,dropout=dropout,layer_norm=True); self.trunk=make_mlp(ctx_emb_dim+128+64+128+128,[hidden_dim,hidden_dim],hidden_dim,dropout=dropout,layer_norm=True)
        self.strategy_u_head=make_mlp(hidden_dim,[hidden_dim],strategy_count*recovery_horizon*control_dim,dropout=dropout); self.dynamics=DegradedBicycleRollout(state_dim=state_dim); self.verifier=StrategyMarginVerifier(state_dim=state_dim,world_dim=world_dim,deg_dim=deg_dim,hidden_dim=hidden_dim); self.branch_feature=make_mlp(hidden_dim+recovery_horizon*control_dim,[hidden_dim],hidden_dim,dropout=dropout,layer_norm=True); self.active_head=make_mlp(hidden_dim,[max(16,hidden_dim//2)],bottlenecks,dropout=dropout); self.smooth_min=SmoothMin(0.1)
    def _tokens_from_batch(self,batch):
        if 'event_tokens' in batch:
            tok=batch['event_tokens'].float(); return tok.view(tok.shape[0],self.token_count,self.token_dim) if tok.dim()==2 else tok
        b=batch['x_plus'].shape[0]; return torch.zeros(b,self.token_count,self.token_dim,device=batch['x_plus'].device,dtype=batch['x_plus'].dtype)
    def encode(self,batch):
        ctx=self.context_encoder(batch['o_hist'],batch['h_ctx']); x=self.x_encoder(batch['x_plus'].float()); deg=batch.get('deg',batch['d_deg']).float(); d=self.d_encoder(deg); tok0=self._tokens_from_batch(batch); tok=self.token_encoder(torch.cat([tok0.mean(1),tok0.max(1).values],-1)); world=batch.get('world_plus',torch.zeros(x.shape[0],self.world_dim,device=x.device,dtype=x.dtype)); w=self.world_encoder(world.float()); return self.trunk(torch.cat([ctx,x,d,tok,w],-1))
    def forward(self,batch):
        h=self.encode(batch); b=h.shape[0]; deg=batch.get('deg',batch['d_deg']).float(); raw=self.strategy_u_head(h).view(b,self.strategy_count,self.recovery_horizon,self.control_dim); controls=torch.cat([torch.tanh(raw[...,:1]), torch.sigmoid(raw[...,1:3]), raw[...,3:]],-1) if self.control_dim>=3 else torch.tanh(raw); controls=torch.nan_to_num(controls, nan=0.0, posinf=1.0, neginf=-1.0); traj=self.dynamics(batch['x_plus'].float(),controls,deg); traj=torch.nan_to_num(traj, nan=0.0, posinf=1e3, neginf=-1e3); world=batch.get('world_plus',torch.zeros(b,self.world_dim,device=h.device,dtype=h.dtype)).float(); margins=self.verifier(traj,controls,world,deg); margins=torch.nan_to_num(margins, nan=-10.0, posinf=10.0, neginf=-10.0).clamp(-50.0, 50.0)
        if self.scalar: margins=margins.mean(-1,keepdim=True).expand(-1,-1,self.bottlenecks)
        branch_v=margins.min(-1).values; branch_vs=self.smooth_min(margins,-1); best=branch_v.argmax(-1); gm=best.view(b,1,1).expand(-1,1,self.bottlenecks); r_hat=torch.gather(margins,1,gm).squeeze(1)
        gu=best.view(b,1,1,1).expand(-1,1,self.recovery_horizon,self.control_dim); gx=best.view(b,1,1,1).expand(-1,1,self.recovery_horizon+1,self.state_dim); best_u=torch.gather(controls,1,gu).squeeze(1); best_traj=torch.gather(traj,1,gx).squeeze(1)
        bh=self.branch_feature(torch.cat([h[:,None,:].expand(b,self.strategy_count,h.shape[-1]),controls.reshape(b,self.strategy_count,-1)],-1)); branch_active=torch.nan_to_num(self.active_head(bh), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0,50.0); ga=best.view(b,1,1).expand(-1,1,self.bottlenecks); active=torch.gather(branch_active,1,ga).squeeze(1); v_smooth=0.1*torch.logsumexp(branch_vs/0.1,dim=-1)
        return {'r_hat':r_hat,'active_logits':active,'branch_active_logits':branch_active,'V_smooth':v_smooth,'V':branch_v.max(-1).values,'branch_margins':margins,'branch_values':branch_v,'strategy_u':controls,'strategy_traj':traj,'best_strategy_u':best_u,'best_strategy_traj':best_traj,'best_branch':best}
    def _matched_branch(self,out,batch,lambda_xi:float=0.25):
        if 'teacher_u' not in batch or 'teacher_traj' not in batch: return out['best_branch'].detach()
        u_cost=(out['strategy_u']-batch['teacher_u'].float().unsqueeze(1)).abs().mean((2,3)); x_cost=(out['strategy_traj']-batch['teacher_traj'].float().unsqueeze(1)).abs().mean((2,3)); return (u_cost+lambda_xi*x_cost).argmin(-1).detach()
    def ctrl_violation_loss(self,controls,deg):
        d=deg.float(); ss=d[:,0].view(-1,1,1).clamp_min(0.05); bs=d[:,1].view(-1,1,1).clamp_min(0.05); ts=d[:,2].view(-1,1,1).clamp_min(0.05); return (F.relu(controls[...,0].abs()-ss)+F.relu(controls[...,1]-bs)+F.relu(controls[...,2]-ts)).mean()
    def dynamics_regularizer(self,out,batch):
        traj=out['strategy_traj']; start=F.smooth_l1_loss(traj[:,:,0,:],batch['x_plus'].float().unsqueeze(1).expand_as(traj[:,:,0,:])); smooth=(traj[:,:,2:,:]-2*traj[:,:,1:-1,:]+traj[:,:,:-2,:]).abs().mean(); return start+0.05*smooth
    def monotonicity_penalty(self,batch,margins): return torch.tensor(0.0,device=margins.device)
    def loss(self,batch,lambdas:Optional[Dict[str,float]]=None,enable_mono:bool=False):
        lambdas=lambdas or {}; out=self.forward(batch); r_star=batch.get('m_star',batch['r_star']).float(); s_star=batch['s_star'].float(); b=r_star.shape[0]; match=self._matched_branch(out,batch,float(lambdas.get('lambda_xi',0.25))); gm=match.view(b,1,1).expand(-1,1,self.bottlenecks); matched=torch.gather(out['branch_margins'],1,gm).squeeze(1)
        weight=1.0+lambdas.get('bd',2.0)*torch.exp(-torch.abs(s_star)/max(lambdas.get('sigma_bd',0.5),1e-6)); profile=(weight*F.smooth_l1_loss(matched,r_star,reduction='none').sum(-1)).mean(); ga=match.view(b,1,1).expand(-1,1,self.bottlenecks); active=F.cross_entropy(torch.nan_to_num(torch.gather(out['branch_active_logits'],1,ga).squeeze(1), nan=0.0, posinf=50.0, neginf=-50.0),batch['b_star'].long())
        gu=match.view(b,1,1,1).expand(-1,1,self.recovery_horizon,self.control_dim); gx=match.view(b,1,1,1).expand(-1,1,self.recovery_horizon+1,self.state_dim); strat=F.smooth_l1_loss(torch.gather(out['strategy_u'],1,gu).squeeze(1),batch['teacher_u'].float())+float(lambdas.get('lambda_xi',0.25))*F.smooth_l1_loss(torch.gather(out['strategy_traj'],1,gx).squeeze(1),batch['teacher_traj'].float())
        dyn=self.dynamics_regularizer(out,batch); ctrl=self.ctrl_violation_loss(out['strategy_u'],batch.get('deg',batch['d_deg'])); onehot=F.one_hot(match,num_classes=self.strategy_count).bool(); teacher_v=matched.min(-1).values; optimism=F.relu(out['branch_values']-teacher_v[:,None]-0.05).masked_fill(onehot,0.0).mean(); flat=out['strategy_u'].reshape(b,self.strategy_count,-1); dist=(flat[:, :, None, :]-flat[:, None, :, :]).abs().mean(-1); diversity=F.relu(0.05-dist).mean()
        mono=self.monotonicity_penalty(batch,matched) if enable_mono else torch.tensor(0.0,device=r_star.device); total=profile+lambdas.get('act',0.5)*active+lambdas.get('strat',0.5)*strat+lambdas.get('dyn',0.05)*dyn+lambdas.get('ctrl',0.05)*ctrl+lambdas.get('optimism',0.2)*optimism+lambdas.get('diversity',0.05)*diversity+lambdas.get('mono',0.0)*mono; total=torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
        return {'loss':total,'profile_loss':profile.detach(),'active_loss':active.detach(),'strategy_loss':strat.detach(),'dyn_loss':dyn.detach(),'ctrl_loss':ctrl.detach(),'optimism_loss':optimism.detach(),'diversity_loss':diversity.detach(),'mono_loss':mono.detach()}
def ordering_loss(v_i,v_j,s_i,s_j,margin:float=0.05):
    sign=torch.sign(s_i-s_j).clamp(min=-1,max=1); keep=sign.abs()>0
    if keep.sum()==0: return torch.tensor(0.0,device=v_i.device)
    return F.relu(margin-sign[keep]*(v_i[keep]-v_j[keep])).mean()

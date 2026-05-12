from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F
from mrvp.data.schema import ACTION_DIM,DEG_DIM,EVENT_TYPES,MECH_DIM,NUM_ACTIONS,STATE_DIM,TOKEN_COUNT,TOKEN_DIM,WORLD_DIM
from .common import grad_reverse, make_mlp, mixture_diag_gaussian_nll
from .context import SceneContextEncoder
class MSRT(nn.Module):
    def __init__(self,state_dim:int=STATE_DIM,deg_dim:int=DEG_DIM,audit_dim:int=MECH_DIM,action_dim:int=ACTION_DIM,num_actions:int=NUM_ACTIONS,world_dim:int=WORLD_DIM,token_count:int=TOKEN_COUNT,token_dim:int=TOKEN_DIM,ctx_emb_dim:int=128,hidden_dim:int=256,mixture_count:int=5,dropout:float=0.05,event_emb_dim:int=32):
        super().__init__(); self.state_dim=state_dim; self.deg_dim=deg_dim; self.audit_dim=audit_dim; self.action_dim=action_dim; self.num_actions=num_actions; self.world_dim=world_dim; self.token_count=token_count; self.token_dim=token_dim; self.mixture_count=mixture_count
        self.context_encoder=SceneContextEncoder(out_dim=ctx_emb_dim); self.action_embedding=nn.Embedding(num_actions,32); self.action_mlp=make_mlp(action_dim+32,[64],64,dropout=dropout,layer_norm=True); self.xt_mlp=make_mlp(state_dim,[128],128,dropout=dropout,layer_norm=True); self.trunk=make_mlp(ctx_emb_dim+64+128,[hidden_dim,hidden_dim],hidden_dim,dropout=dropout,layer_norm=True)
        self.event_head=make_mlp(hidden_dim,[hidden_dim//2],len(EVENT_TYPES),dropout=dropout); self.event_time_head=make_mlp(hidden_dim,[hidden_dim//2],1,dropout=dropout); self.event_embedding=nn.Embedding(len(EVENT_TYPES),event_emb_dim)
        self.token_head=make_mlp(hidden_dim,[hidden_dim],token_count*token_dim,dropout=dropout); self.world_head=make_mlp(hidden_dim+token_count*token_dim,[hidden_dim],world_dim,dropout=dropout); self.audit_head=make_mlp(hidden_dim+token_count*token_dim,[hidden_dim],audit_dim,dropout=dropout); self.deg_head=make_mlp(hidden_dim+token_count*token_dim,[hidden_dim],deg_dim,dropout=dropout)
        md_in=hidden_dim+token_count*token_dim+deg_dim+event_emb_dim; self.mix_logits=make_mlp(md_in,[hidden_dim],mixture_count,dropout=dropout); self.mix_mean=make_mlp(md_in,[hidden_dim],mixture_count*state_dim,dropout=dropout); self.mix_log_scale=make_mlp(md_in,[hidden_dim],mixture_count*state_dim,dropout=dropout)
        self.target_proj=make_mlp(state_dim+deg_dim+world_dim,[hidden_dim],token_dim,dropout=dropout,layer_norm=True); self.token_pool_proj=make_mlp(token_dim,[hidden_dim//2],token_dim,dropout=dropout,layer_norm=True); self.action_adversary=make_mlp(token_dim,[128],num_actions,dropout=dropout,layer_norm=True)
    def encode(self,batch):
        ctx=self.context_encoder(batch['o_hist'],batch['h_ctx']); aid=batch['action_id'].long().clamp_min(0)%self.num_actions; avec=batch.get('action_vec',torch.zeros(aid.shape[0],self.action_dim,device=aid.device)); act=self.action_mlp(torch.cat([avec.float(),self.action_embedding(aid)],-1)); xt=self.xt_mlp(batch.get('x_t',batch.get('x_minus')).float()); return self.trunk(torch.cat([ctx,act,xt],-1))
    def _constrain_deg(self,d_raw):
        d=d_raw.clone();
        if d.shape[-1]>=3: d[:,:3]=1.2*torch.sigmoid(d_raw[:,:3])
        if d.shape[-1]>=4: d[:,3]=F.softplus(d_raw[:,3])
        if d.shape[-1]>=5: d[:,4]=1.2*torch.sigmoid(d_raw[:,4])
        if d.shape[-1]>=6: d[:,5]=torch.sigmoid(d_raw[:,5])
        return d
    def forward(self,batch:Dict[str,torch.Tensor], teacher_force_event:bool=False)->Dict[str,torch.Tensor]:
        h=self.encode(batch); token_flat=self.token_head(h); tokens=token_flat.view(-1,self.token_count,self.token_dim); token_pool=tokens.mean(1); event_logits=self.event_head(h); event_time=F.softplus(self.event_time_head(h)).squeeze(-1); world=self.world_head(torch.cat([h,token_flat],-1)); audit=self.audit_head(torch.cat([h,token_flat],-1)); deg=self._constrain_deg(self.deg_head(torch.cat([h,token_flat],-1)))
        if teacher_force_event and 'event_type_id' in batch: event_emb=self.event_embedding(batch['event_type_id'].long().clamp(0,len(EVENT_TYPES)-1))
        else: event_emb=F.softmax(event_logits,-1) @ self.event_embedding.weight
        md_in=torch.cat([h,token_flat,deg,event_emb],-1); logits=self.mix_logits(md_in); mean=self.mix_mean(md_in).view(-1,self.mixture_count,self.state_dim); log_scale=self.mix_log_scale(md_in).view(-1,self.mixture_count,self.state_dim).clamp(-6,3); anchor=batch.get('x_minus',batch.get('x_t')).float(); mean=mean+anchor[:,None,:]
        return {'event_logits':event_logits,'event_time':event_time,'event_tokens':tokens,'token_pool':token_pool,'world_plus':world,'z_mech':audit,'deg':deg,'d_deg':deg,'mix_logits':logits,'mix_mean':mean,'mix_log_scale':log_scale}
    def sample(self,batch,num_samples:int=16,deterministic:bool=False):
        out=self.forward(batch); probs=F.softmax(out['mix_logits'],-1); b=probs.shape[0]; comp=(probs.argmax(-1).repeat_interleave(num_samples) if deterministic else torch.multinomial(probs,num_samples,replacement=True).reshape(-1)); mean=out['mix_mean'].repeat_interleave(num_samples,0); log_scale=out['mix_log_scale'].repeat_interleave(num_samples,0); gather=comp.view(-1,1,1).expand(-1,1,self.state_dim); cm=torch.gather(mean,1,gather).squeeze(1); cs=torch.gather(log_scale,1,gather).squeeze(1); x_plus=cm if deterministic else cm+torch.randn_like(cm)*torch.exp(cs).clamp_max(5.0); ep=F.softmax(out['event_logits'],-1); event_type_id=(ep.argmax(-1).repeat_interleave(num_samples) if deterministic else torch.multinomial(ep,num_samples,replacement=True).reshape(-1)); return {'x_plus':x_plus,'event_type_id':event_type_id,'event_time':out['event_time'].repeat_interleave(num_samples,0),'event_tokens':out['event_tokens'].repeat_interleave(num_samples,0),'world_plus':out['world_plus'].repeat_interleave(num_samples,0),'z_mech':out['z_mech'].repeat_interleave(num_samples,0),'deg':out['deg'].repeat_interleave(num_samples,0),'d_deg':out['d_deg'].repeat_interleave(num_samples,0),'sample_root':torch.arange(b,device=x_plus.device).repeat_interleave(num_samples)}
    def same_root_contrastive_loss(self,batch,out,temperature:float=0.1,eps_s:float=0.25):
        roots=batch.get('root_id');
        if not isinstance(roots,list): return torch.tensor(0.0,device=out['token_pool'].device)
        ids=[]; m={}
        for r in roots: m.setdefault(str(r),len(m)); ids.append(m[str(r)])
        device=out['token_pool'].device; root=torch.tensor(ids,device=device); same=root[:,None]==root[None,:]; harm=batch['harm_bin'].long(); same_harm=harm[:,None]==harm[None,:]; adm=batch.get('is_harm_admissible',torch.ones_like(batch['s_star'])).float()>0.5; sgap=(batch['s_star'].float()[:,None]-batch['s_star'].float()[None,:]).abs()>=eps_s; mask=same & same_harm & adm[:,None] & adm[None,:] & sgap
        if mask.sum()==0: return torch.tensor(0.0,device=device)
        z=F.normalize(self.token_pool_proj(out['token_pool']),dim=-1); y=F.normalize(self.target_proj(torch.cat([batch['x_plus'].float(),batch['d_deg'].float(),batch['world_plus'].float()],-1)),dim=-1); sim=z@y.T/max(temperature,1e-6); sim=sim.masked_fill(~mask,-1e4); keep=mask.sum(1)>0; labels=torch.arange(sim.shape[0],device=device); return F.cross_entropy(sim[keep],labels[keep])
    def loss(self,batch,lambdas:Optional[Dict[str,float]]=None):
        lambdas=lambdas or {}; out=self.forward(batch,teacher_force_event=True); trans=mixture_diag_gaussian_nll(batch['x_plus'].float(),out['mix_logits'],out['mix_mean'],out['mix_log_scale']).mean(); event=F.cross_entropy(out['event_logits'],batch['event_type_id'].long().clamp(0,len(EVENT_TYPES)-1)); et=F.smooth_l1_loss(out['event_time'],batch['event_time'].float())
        token_weight=float(lambdas.get('token',0.0)); token_loss=torch.tensor(0.0,device=trans.device)
        if token_weight>0 and 'has_event_tokens' in batch and batch['has_event_tokens'].float().sum()>0:
            mask=batch['has_event_tokens'].float().view(-1,1,1); token_loss=(F.smooth_l1_loss(out['event_tokens'],batch['event_tokens'].float(),reduction='none')*mask).sum()/mask.sum().clamp_min(1.0)
        world=F.smooth_l1_loss(out['world_plus'],batch['world_plus'].float()); deg=F.smooth_l1_loss(out['d_deg'],batch['d_deg'].float()); audit=F.smooth_l1_loss(out['z_mech'],batch['z_mech'].float()); ctr=self.same_root_contrastive_loss(batch,out,float(lambdas.get('temperature',0.1)))
        suf=float(lambdas.get('suf',0.0)); adv=F.cross_entropy(self.action_adversary(grad_reverse(out['token_pool'])),batch['action_id'].long().clamp_min(0)%self.num_actions) if suf>0 else torch.tensor(0.0,device=trans.device)
        total=lambdas.get('nll',1.0)*trans+lambdas.get('event',1.0)*(event+0.2*et)+token_weight*token_loss+lambdas.get('world',0.5)*world+lambdas.get('deg',1.0)*deg+lambdas.get('probe',0.1)*audit+lambdas.get('ctr',0.1)*ctr+suf*adv
        return {'loss':total,'trans_nll':trans.detach(),'event_loss':event.detach(),'event_time_loss':et.detach(),'token_loss':token_loss.detach(),'world_loss':world.detach(),'deg_loss':deg.detach(),'audit_loss':audit.detach(),'contrastive_loss':ctr.detach(),'action_adv_loss':adv.detach()}

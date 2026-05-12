#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
from mrvp.data.dataset import MRVPDataset,mrvp_collate
from mrvp.data.schema import NUM_ACTIONS,TOKEN_COUNT,TOKEN_DIM,SchemaDims
from mrvp.models.common import make_mlp
from mrvp.models.msrt import MSRT
from mrvp.training.checkpoints import load_model
from mrvp.training.loops import to_device
class Probe(nn.Module):
    def __init__(self,in_dim,hidden_dim=128): super().__init__(); self.net=make_mlp(in_dim,[hidden_dim,hidden_dim],1,dropout=0.05,layer_norm=True)
    def forward(self,x): return self.net(x).squeeze(-1)
def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def auc(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p); pos=y==1; neg=y==0
    if pos.sum()==0 or neg.sum()==0: return float('nan')
    order=np.argsort(p); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(p)+1); return float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
def bacc(y,p): return float(np.mean([((p[y==c]==c).mean() if (y==c).sum() else np.nan) for c in [0,1]]))
@torch.no_grad()
def build_features(path,split,feature_source,msrt,device,batch_size,token_count,token_dim,admissible_only=False):
    ds=MRVPDataset(path,split,SchemaDims(token_count=token_count,token_dim=token_dim)); loader=DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=mrvp_collate); x0=[]; x1=[]; y=[]
    for batch in tqdm(loader,desc=f'features {split}',leave=False):
        mask=torch.ones(batch['s_star'].shape[0],dtype=torch.bool)
        if admissible_only and 'is_harm_admissible' in batch: mask=batch['is_harm_admissible'].float()>0.5
        if mask.sum()==0: continue
        if feature_source=='predicted_msrt':
            if msrt is None: raise ValueError('--feature-source predicted_msrt requires --msrt')
            mb=to_device(batch,device); pred=msrt.sample(mb,num_samples=1,deterministic=True); x_plus=pred['x_plus'].cpu(); d=pred['d_deg'].cpu(); tok=pred['event_tokens'].cpu(); world=pred['world_plus'].cpu()
        elif feature_source=='leaky_current': x_plus=batch['x_plus'].float(); d=batch['d_deg'].float(); tok=batch['z_mech'].float().view(batch['z_mech'].shape[0],1,-1).repeat(1,token_count,1)[...,:token_dim]; world=batch['world_plus'].float()
        else: x_plus=batch['x_plus'].float(); d=batch['d_deg'].float(); tok=batch['event_tokens'].float(); world=batch['world_plus'].float()
        h=batch['h_ctx'].float(); parts=[]
        if feature_source in ['true_clean','predicted_msrt','leaky_current','xplus_only']: parts += [x_plus]
        if feature_source in ['true_clean','predicted_msrt','leaky_current']: parts += [d,tok.flatten(1),world,h]
        elif feature_source=='world_only': parts += [world]
        elif feature_source=='tokens_only': parts += [tok.flatten(1)]
        base=torch.cat(parts,-1); aid=batch['action_id'].long()%NUM_ACTIONS; action=torch.cat([batch['action_vec'].float(),F.one_hot(aid,NUM_ACTIONS).float()],-1); x0.append(base[mask].numpy()); x1.append(torch.cat([base,action],-1)[mask].numpy()); y.append((batch['s_star'].float()[mask]>=0).long().numpy())
    return np.concatenate(x0).astype(np.float32),np.concatenate(x1).astype(np.float32),np.concatenate(y).astype(np.int64)
def standardize(a,b,c):
    m=a.mean(0,keepdims=True); s=a.std(0,keepdims=True); s=np.where(s<1e-6,1,s); return ((a-m)/s).astype(np.float32),((b-m)/s).astype(np.float32),((c-m)/s).astype(np.float32)
def train_eval(xtr,ytr,xv,yv,xt,yt,epochs,batch,lr,hidden,device,seed):
    torch.manual_seed(seed); model=Probe(xtr.shape[1],hidden).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); tr=DataLoader(TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr)),batch_size=batch,shuffle=True); va=DataLoader(TensorDataset(torch.from_numpy(xv),torch.from_numpy(yv)),batch_size=batch); te=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(yt)),batch_size=batch)
    best=None; bestn=1e99
    for _ in range(epochs):
        model.train()
        for x,y in tr:
            x=x.to(device); y=y.to(device).float(); opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(model(x),y); loss.backward(); opt.step()
        m=evaluate(model,va,device)
        if m['nll']<bestn: bestn=m['nll']; best={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best: model.load_state_dict(best)
    return {'val':evaluate(model,va,device),'test':evaluate(model,te,device)}
@torch.no_grad()
def evaluate(model,loader,device):
    ys=[]; ps=[]; losses=[]; model.eval()
    for x,y in loader:
        x=x.to(device); logit=model(x).cpu(); losses.append(F.binary_cross_entropy_with_logits(logit,y.float(),reduction='sum').item()); ys.append(y.numpy()); ps.append(torch.sigmoid(logit).numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); pred=(p>=0.5).astype(int); return {'nll':float(sum(losses)/max(1,len(y))),'auc':auc(y,p),'balanced_acc':bacc(y,pred)}
def main():
    p=argparse.ArgumentParser(description='Residual-action sufficiency diagnostic.'); p.add_argument('--data',required=True); p.add_argument('--output',default='runs/default/residual_action.json'); p.add_argument('--feature-source',choices=['true_clean','predicted_msrt','leaky_current','xplus_only','world_only','tokens_only','true'],default='true_clean'); p.add_argument('--msrt',default=''); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--hidden-dim',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--device',default='auto'); p.add_argument('--torch-threads',type=int,default=1); p.add_argument('--seed',type=int,default=7); p.add_argument('--msrt-hidden-dim',type=int,default=256); p.add_argument('--mixture-count',type=int,default=5); p.add_argument('--token-count',type=int,default=TOKEN_COUNT); p.add_argument('--token-dim',type=int,default=TOKEN_DIM); p.add_argument('--admissible-only',action='store_true'); p.add_argument('--target',choices=['recoverable'],default='recoverable'); args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); device=auto_device(args.device); fs='true_clean' if args.feature_source=='true' else args.feature_source; msrt=None
    if fs=='predicted_msrt': msrt=MSRT(hidden_dim=args.msrt_hidden_dim,mixture_count=args.mixture_count,token_count=args.token_count,token_dim=args.token_dim).to(device); load_model(msrt,args.msrt,device,strict=False); msrt.eval()
    x0tr,x1tr,ytr=build_features(args.data,'train',fs,msrt,device,args.batch_size,args.token_count,args.token_dim,args.admissible_only); x0v,x1v,yv=build_features(args.data,'val',fs,msrt,device,args.batch_size,args.token_count,args.token_dim,args.admissible_only); x0t,x1t,yt=build_features(args.data,'test',fs,msrt,device,args.batch_size,args.token_count,args.token_dim,args.admissible_only); x0tr,x0v,x0t=standardize(x0tr,x0v,x0t); x1tr,x1v,x1t=standardize(x1tr,x1v,x1t); p0=train_eval(x0tr,ytr,x0v,yv,x0t,yt,args.epochs,args.batch_size,args.lr,args.hidden_dim,device,args.seed); p1=train_eval(x1tr,ytr,x1v,yv,x1t,yt,args.epochs,args.batch_size,args.lr,args.hidden_dim,device,args.seed+1); t0=p0['test']; t1=p1['test']; out={'feature_source':fs,'target':'recoverable','overall':{'probe0_auc':t0['auc'],'probe1_auc':t1['auc'],'delta_auc':t1['auc']-t0['auc'],'probe0_balanced_acc':t0['balanced_acc'],'probe1_balanced_acc':t1['balanced_acc'],'delta_balanced_acc':t1['balanced_acc']-t0['balanced_acc'],'probe0_nll':t0['nll'],'probe1_nll':t1['nll'],'delta_nll':t1['nll']-t0['nll']},'probe0_no_action':p0,'probe1_with_action':p1,'sizes':{'train':len(ytr),'val':len(yv),'test':len(yt)}}; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps({'output':args.output,'overall':out['overall']},indent=2))
if __name__=='__main__': main()

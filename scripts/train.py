#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse, json
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mrvp.data.dataset import MRVPDataset,mrvp_collate
from mrvp.data.pairs import SameRootPairDataset,pair_collate
from mrvp.data.schema import RECOVERY_HORIZON,STRATEGY_COUNT,TOKEN_COUNT,TOKEN_DIM,SchemaDims
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork,ordering_loss
from mrvp.training.checkpoints import load_model,save_checkpoint
from mrvp.training.loops import eval_loss,to_device,train_one_epoch,scalar_logs
from mrvp.training.rpn_batches import build_rpn_batch_from_msrt

def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def dims(args): return SchemaDims(token_count=args.token_count,token_dim=args.token_dim,recovery_horizon=args.recovery_horizon)
def build_msrt(args): return MSRT(mixture_count=args.mixture_count,hidden_dim=args.hidden_dim,token_count=args.token_count,token_dim=args.token_dim)
def build_rpn(args): return RecoveryProfileNetwork(hidden_dim=args.hidden_dim,token_count=args.token_count,token_dim=args.token_dim,strategy_count=args.strategy_count,recovery_horizon=args.recovery_horizon,scalar=args.scalar_rpn)
def msrt_lambdas(args): return {'nll':args.lambda_nll,'event':args.lambda_event,'token':args.lambda_token,'world':args.lambda_world,'deg':args.lambda_deg,'probe':args.lambda_probe,'ctr':args.lambda_ctr,'suf':args.lambda_suf,'temperature':args.contrastive_temperature}
def rpn_lambdas(args): return {'act':args.lambda_act,'bd':args.lambda_bd,'sigma_bd':args.sigma_bd,'strat':args.lambda_strat,'dyn':args.lambda_dyn,'ctrl':args.lambda_ctrl,'mono':args.lambda_mono,'lambda_xi':args.lambda_xi,'optimism':args.lambda_optimism,'diversity':args.lambda_diversity}
def train_msrt(args,device):
    train_ds=MRVPDataset(args.data,'train',dims(args)); val_ds=MRVPDataset(args.data,'val',dims(args)); tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=mrvp_collate); vl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=mrvp_collate); model=build_msrt(args).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); out=Path(args.out_dir)/'msrt.pt'; best=1e99; lamb=msrt_lambdas(args)
    for e in range(1,args.epochs+1):
        tr=train_one_epoch(model,tl,opt,device,{'lambdas':lamb},f'msrt train {e}'); va=eval_loss(model,vl,device,{'lambdas':lamb},f'msrt val {e}'); print(json.dumps({'stage':'msrt','epoch':e,'train':tr,'val':va},ensure_ascii=False));
        if va.get('loss',tr['loss'])<best: best=va.get('loss',tr['loss']); save_checkpoint(out,model,{'stage':'msrt','epoch':e,'args':vars(args)})
    return out

def _rpn_epoch(rpn,msrt,loader,opt,device,lamb,train=True,desc='rpn'):
    rpn.train(train); msrt.eval(); totals={}; n=0
    for batch in tqdm(loader,desc=desc,leave=False):
        batch=to_device(batch,device); rb=build_rpn_batch_from_msrt(msrt,batch,'oracle_clean',detach_msrt=True)
        if train: opt.zero_grad(set_to_none=True)
        logs=rpn.loss(rb,lambdas=lamb,enable_mono=False)
        if train: logs['loss'].backward(); torch.nn.utils.clip_grad_norm_(rpn.parameters(),5.0); opt.step()
        s=scalar_logs(logs); totals={k:totals.get(k,0)+v for k,v in s.items()}; n+=1
    return {k:v/max(n,1) for k,v in totals.items()}
def train_rpn_oracle_clean(args,device):
    msrt_path=Path(args.msrt or Path(args.out_dir)/'msrt.pt')
    if not msrt_path.exists(): raise SystemExit('--stage rpn_oracle_clean requires --msrt or out_dir/msrt.pt')
    msrt=build_msrt(args).to(device); load_model(msrt,msrt_path,device,strict=False); train_ds=MRVPDataset(args.data,'train',dims(args)); val_ds=MRVPDataset(args.data,'val',dims(args)); tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=mrvp_collate); vl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=mrvp_collate); rpn=build_rpn(args).to(device)
    if args.rpn_init and Path(args.rpn_init).exists(): load_model(rpn,args.rpn_init,device,strict=False)
    opt=torch.optim.AdamW(rpn.parameters(),lr=args.lr,weight_decay=args.weight_decay); out=Path(args.out_dir)/('rpn_scalar.pt' if args.scalar_rpn else 'rpn.pt'); best=1e99; lamb=rpn_lambdas(args)
    for e in range(1,args.epochs+1):
        tr=_rpn_epoch(rpn,msrt,tl,opt,device,lamb,True,f'rpn oracle-clean train {e}'); va=_rpn_epoch(rpn,msrt,vl,opt,device,lamb,False,f'rpn oracle-clean val {e}'); print(json.dumps({'stage':'rpn_oracle_clean','epoch':e,'train':tr,'val':va},ensure_ascii=False));
        if va.get('loss',tr['loss'])<best: best=va.get('loss',tr['loss']); save_checkpoint(out,rpn,{'stage':'rpn_oracle_clean','epoch':e,'args':vars(args)})
    return out

def finetune_pairs(args,device):
    msrt_path=Path(args.msrt or Path(args.out_dir)/'msrt.pt')
    if not msrt_path.exists(): raise SystemExit('--stage pair requires --msrt or out_dir/msrt.pt')
    msrt=build_msrt(args).to(device); load_model(msrt,msrt_path,device,strict=False); base=MRVPDataset(args.data,'train',dims(args)); pair_ds=SameRootPairDataset(base,args.eps_s,args.max_pairs_per_root); loader=DataLoader(pair_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=pair_collate); rpn=build_rpn(args).to(device); init=Path(args.rpn or args.rpn_init or Path(args.out_dir)/('rpn_scalar.pt' if args.scalar_rpn else 'rpn.pt'))
    if init.exists(): load_model(rpn,init,device,strict=False)
    opt=torch.optim.AdamW(rpn.parameters(),lr=args.lr*0.5,weight_decay=args.weight_decay); lamb=rpn_lambdas(args)
    for e in range(1,args.epochs+1):
        rpn.train(); totals={}; n=0
        for batch in tqdm(loader,desc=f'pair finetune {e}',leave=False):
            bi=to_device(batch['i'],device); bj=to_device(batch['j'],device); bi=build_rpn_batch_from_msrt(msrt,bi,'oracle_clean'); bj=build_rpn_batch_from_msrt(msrt,bj,'oracle_clean'); opt.zero_grad(set_to_none=True); oi=rpn(bi); oj=rpn(bj); l_ord=ordering_loss(oi['V_smooth'],oj['V_smooth'],bi['s_star'].float(),bj['s_star'].float(),args.pair_margin); loss=0.5*(rpn.loss(bi,lambdas=lamb)['loss']+rpn.loss(bj,lambdas=lamb)['loss'])+args.lambda_ord*l_ord; loss.backward(); torch.nn.utils.clip_grad_norm_(rpn.parameters(),5.0); opt.step(); totals['loss']=totals.get('loss',0)+float(loss.detach().cpu()); totals['ord']=totals.get('ord',0)+float(l_ord.detach().cpu()); n+=1
        print(json.dumps({'stage':'pair','epoch':e,'train':{k:v/max(n,1) for k,v in totals.items()},'pair_diagnostics':pair_ds.diagnostics},ensure_ascii=False))
    out=Path(args.out_dir)/('rpn_finetuned_scalar.pt' if args.scalar_rpn else 'rpn_finetuned.pt'); save_checkpoint(out,rpn,{'stage':'pair','args':vars(args)}); return out

def main():
    p=argparse.ArgumentParser(description='Train MRVP MSRT/RPN models with clean token flow.'); p.add_argument('--data',required=True); p.add_argument('--out-dir',default='runs/default'); p.add_argument('--stage',choices=['msrt','rpn','rpn_oracle_clean','pair','finetune','all'],default='all'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--device',default='auto'); p.add_argument('--num-workers',type=int,default=0); p.add_argument('--hidden-dim',type=int,default=256); p.add_argument('--mixture-count',type=int,default=5); p.add_argument('--token-count',type=int,default=TOKEN_COUNT); p.add_argument('--token-dim',type=int,default=TOKEN_DIM); p.add_argument('--strategy-count',type=int,default=STRATEGY_COUNT); p.add_argument('--recovery-horizon',type=int,default=RECOVERY_HORIZON); p.add_argument('--msrt',default=''); p.add_argument('--rpn',default=''); p.add_argument('--rpn-init',default=''); p.add_argument('--scalar-rpn',action='store_true')
    p.add_argument('--lambda-nll',type=float,default=1.0); p.add_argument('--lambda-event',type=float,default=1.0); p.add_argument('--lambda-token',type=float,default=0.0); p.add_argument('--lambda-world',type=float,default=0.5); p.add_argument('--lambda-deg',type=float,default=1.0); p.add_argument('--lambda-probe',type=float,default=0.1); p.add_argument('--lambda-ctr',type=float,default=0.1); p.add_argument('--contrastive-temperature',type=float,default=0.1); p.add_argument('--lambda-suf',type=float,default=0.0)
    p.add_argument('--lambda-act',type=float,default=0.5); p.add_argument('--lambda-bd',type=float,default=2.0); p.add_argument('--sigma-bd',type=float,default=0.5); p.add_argument('--lambda-strat',type=float,default=0.5); p.add_argument('--lambda-dyn',type=float,default=0.05); p.add_argument('--lambda-ctrl',type=float,default=0.05); p.add_argument('--lambda-xi',type=float,default=0.25); p.add_argument('--lambda-optimism',type=float,default=0.2); p.add_argument('--lambda-diversity',type=float,default=0.05); p.add_argument('--lambda-mono',type=float,default=0.0); p.add_argument('--lambda-ord',type=float,default=1.0); p.add_argument('--eps-s',type=float,default=0.25); p.add_argument('--pair-margin',type=float,default=0.05); p.add_argument('--max-pairs-per-root',type=int,default=64); p.add_argument('--torch-threads',type=int,default=1)
    args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); Path(args.out_dir).mkdir(parents=True,exist_ok=True); device=auto_device(args.device); print(json.dumps({'device':str(device),'stage':args.stage,'out_dir':args.out_dir},ensure_ascii=False))
    if args.stage in ('msrt','all'): train_msrt(args,device)
    if args.stage in ('rpn','rpn_oracle_clean','all'): train_rpn_oracle_clean(args,device)
    if args.stage in ('pair','finetune','all'): finetune_pairs(args,device)
if __name__=='__main__': main()

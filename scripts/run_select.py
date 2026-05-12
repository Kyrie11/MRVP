#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,torch
from mrvp.calibration import load_calibration_table
from mrvp.data.dataset import MRVPDataset,mrvp_collate
from mrvp.data.schema import TOKEN_COUNT,TOKEN_DIM,STRATEGY_COUNT,RECOVERY_HORIZON,SchemaDims
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.selection import select_action_with_models
from mrvp.training.checkpoints import load_model
def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def main():
    p=argparse.ArgumentParser(description='Run MRVP Algorithm 1 for one root.'); p.add_argument('--data',required=True); p.add_argument('--root-id',required=True); p.add_argument('--split',default=None); p.add_argument('--msrt',required=True); p.add_argument('--rpn',required=True); p.add_argument('--calibration',default=''); p.add_argument('--num-samples',type=int,default=32); p.add_argument('--beta',type=float,default=0.2); p.add_argument('--device',default='auto'); p.add_argument('--hidden-dim',type=int,default=256); p.add_argument('--mixture-count',type=int,default=5); p.add_argument('--token-count',type=int,default=TOKEN_COUNT); p.add_argument('--token-dim',type=int,default=TOKEN_DIM); p.add_argument('--strategy-count',type=int,default=STRATEGY_COUNT); p.add_argument('--recovery-horizon',type=int,default=RECOVERY_HORIZON); p.add_argument('--scalar-rpn',action='store_true'); p.add_argument('--torch-threads',type=int,default=1); args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); device=auto_device(args.device); ds=MRVPDataset(args.data,args.split,SchemaDims(token_count=args.token_count,token_dim=args.token_dim,recovery_horizon=args.recovery_horizon)); idxs=ds.root_to_indices.get(str(args.root_id));
    if not idxs: raise SystemExit(f'root_id {args.root_id!r} not found')
    root_batch=mrvp_collate([ds[i] for i in idxs]); root_rows=[ds.rows[i] for i in idxs]; msrt=MSRT(hidden_dim=args.hidden_dim,mixture_count=args.mixture_count,token_count=args.token_count,token_dim=args.token_dim); rpn=RecoveryProfileNetwork(hidden_dim=args.hidden_dim,token_count=args.token_count,token_dim=args.token_dim,strategy_count=args.strategy_count,recovery_horizon=args.recovery_horizon,scalar=args.scalar_rpn); load_model(msrt,args.msrt,device,strict=False); load_model(rpn,args.rpn,device,strict=False); table=load_calibration_table(args.calibration or None); print(json.dumps(select_action_with_models(root_batch,root_rows,msrt,rpn,table,args.num_samples,args.beta,device),indent=2,ensure_ascii=False))
if __name__=='__main__': main()

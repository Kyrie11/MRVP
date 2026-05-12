#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,torch
from pathlib import Path
from mrvp.calibration import fit_calibration_table,predict_rpn,predict_rpn_with_msrt,save_calibration_table
from mrvp.data.dataset import MRVPDataset
from mrvp.data.schema import TOKEN_COUNT,TOKEN_DIM,STRATEGY_COUNT,RECOVERY_HORIZON,SchemaDims
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.models.msrt import MSRT
from mrvp.models.baselines import DirectActionRiskNetwork,UnstructuredLatentRiskNetwork
from mrvp.training.checkpoints import load_model
def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def build_profile_model(t,h,scalar=False,token_count=TOKEN_COUNT,token_dim=TOKEN_DIM,strategy_count=STRATEGY_COUNT,recovery_horizon=RECOVERY_HORIZON):
    if t=='rpn': return RecoveryProfileNetwork(hidden_dim=h,token_count=token_count,token_dim=token_dim,strategy_count=strategy_count,recovery_horizon=recovery_horizon,scalar=scalar)
    if t=='direct_action_to_risk': return DirectActionRiskNetwork(hidden_dim=h,scalar=scalar)
    if t=='unstructured_latent': return UnstructuredLatentRiskNetwork(hidden_dim=h)
    raise ValueError(t)
def main():
    p=argparse.ArgumentParser(description='Fit group/scenario calibration quantiles.'); p.add_argument('--data',required=True); p.add_argument('--rpn',required=True); p.add_argument('--msrt',default=''); p.add_argument('--output',required=True); p.add_argument('--split',default='cal'); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--device',default='auto'); p.add_argument('--delta-b',type=float,default=0.02); p.add_argument('--n-min',type=int,default=20); p.add_argument('--hidden-dim',type=int,default=256); p.add_argument('--msrt-hidden-dim',type=int,default=256); p.add_argument('--mixture-count',type=int,default=5); p.add_argument('--token-count',type=int,default=TOKEN_COUNT); p.add_argument('--token-dim',type=int,default=TOKEN_DIM); p.add_argument('--strategy-count',type=int,default=STRATEGY_COUNT); p.add_argument('--recovery-horizon',type=int,default=RECOVERY_HORIZON); p.add_argument('--scalar-rpn',action='store_true'); p.add_argument('--model-type',choices=['rpn','direct_action_to_risk','unstructured_latent'],default='rpn'); p.add_argument('--calibration-mode',choices=['oracle_clean','legacy_direct'],default='oracle_clean'); p.add_argument('--torch-threads',type=int,default=1); args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); device=auto_device(args.device); ds=MRVPDataset(args.data,args.split,SchemaDims(token_count=args.token_count,token_dim=args.token_dim,recovery_horizon=args.recovery_horizon)); model=build_profile_model(args.model_type,args.hidden_dim,args.scalar_rpn,args.token_count,args.token_dim,args.strategy_count,args.recovery_horizon); load_model(model,args.rpn,device,strict=False)
    if args.model_type=='rpn' and args.msrt and args.calibration_mode=='oracle_clean':
        msrt=MSRT(hidden_dim=args.msrt_hidden_dim,mixture_count=args.mixture_count,token_count=args.token_count,token_dim=args.token_dim); load_model(msrt,args.msrt,device,strict=False); r_hat=predict_rpn_with_msrt(model,msrt,ds,'oracle_clean',args.batch_size,device)
    else: r_hat=predict_rpn(model,ds,args.batch_size,device)
    table=fit_calibration_table(ds.rows,r_hat,args.delta_b,args.n_min); save_calibration_table(table,args.output); print(json.dumps({'output':args.output,'split':args.split,'roots':len(ds.root_to_indices),'rows':len(ds),'calibration_mode':args.calibration_mode,'used_msrt':bool(args.msrt)},indent=2))
if __name__=='__main__': main()

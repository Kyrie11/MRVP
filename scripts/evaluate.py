#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,csv,json,torch
from pathlib import Path
import numpy as np
from mrvp.calibration import load_calibration_table,lower_bounds_for_rows,predict_rpn,predict_rpn_with_msrt
from mrvp.data.dataset import MRVPDataset
from mrvp.data.schema import TOKEN_COUNT,TOKEN_DIM,STRATEGY_COUNT,RECOVERY_HORIZON,SchemaDims
from mrvp.evaluation import baseline_scores,evaluate_selection,lower_bounds_from_scalar_scores
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.models.msrt import MSRT
from mrvp.models.baselines import DirectActionRiskNetwork,UnstructuredLatentRiskNetwork
from mrvp.training.checkpoints import load_model
def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def build_model(t,h,scalar=False,token_count=TOKEN_COUNT,token_dim=TOKEN_DIM,strategy_count=STRATEGY_COUNT,recovery_horizon=RECOVERY_HORIZON):
    if t=='rpn': return RecoveryProfileNetwork(hidden_dim=h,token_count=token_count,token_dim=token_dim,strategy_count=strategy_count,recovery_horizon=recovery_horizon,scalar=scalar)
    if t=='direct_action_to_risk': return DirectActionRiskNetwork(hidden_dim=h,scalar=scalar)
    if t=='unstructured_latent': return UnstructuredLatentRiskNetwork(hidden_dim=h)
    raise ValueError(t)
def main():
    p=argparse.ArgumentParser(description='Evaluate MRVP clean oracle and baselines.'); p.add_argument('--data',required=True); p.add_argument('--rpn',default=''); p.add_argument('--msrt',default=''); p.add_argument('--calibration',default=''); p.add_argument('--split',default='test'); p.add_argument('--output',default='runs/default/eval_test.json'); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--device',default='auto'); p.add_argument('--hidden-dim',type=int,default=256); p.add_argument('--msrt-hidden-dim',type=int,default=256); p.add_argument('--mixture-count',type=int,default=5); p.add_argument('--token-count',type=int,default=TOKEN_COUNT); p.add_argument('--token-dim',type=int,default=TOKEN_DIM); p.add_argument('--strategy-count',type=int,default=STRATEGY_COUNT); p.add_argument('--recovery-horizon',type=int,default=RECOVERY_HORIZON); p.add_argument('--scalar-rpn',action='store_true'); p.add_argument('--model-type',choices=['rpn','direct_action_to_risk','unstructured_latent'],default='rpn'); p.add_argument('--include-leaky-oracle-debug',action='store_true'); p.add_argument('--torch-threads',type=int,default=1); args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); device=auto_device(args.device); ds=MRVPDataset(args.data,args.split,SchemaDims(token_count=args.token_count,token_dim=args.token_dim,recovery_horizon=args.recovery_horizon)); results={}; table=load_calibration_table(args.calibration or None)
    if args.rpn:
        model=build_model(args.model_type,args.hidden_dim,args.scalar_rpn,args.token_count,args.token_dim,args.strategy_count,args.recovery_horizon); load_model(model,args.rpn,device,strict=False)
        if args.model_type=='rpn' and args.msrt:
            msrt=MSRT(hidden_dim=args.msrt_hidden_dim,mixture_count=args.mixture_count,token_count=args.token_count,token_dim=args.token_dim); load_model(msrt,args.msrt,device,strict=False); r_hat=predict_rpn_with_msrt(model,msrt,ds,'oracle_clean',args.batch_size,device); lower=lower_bounds_for_rows(r_hat,ds.rows,table); results['Oracle_clean_true_transition']=evaluate_selection(ds.rows,lower,r_hat); results['Oracle_clean_true_transition_uncalibrated']=evaluate_selection(ds.rows,r_hat,r_hat)
        else:
            r_hat=predict_rpn(model,ds,args.batch_size,device); lower=lower_bounds_for_rows(r_hat,ds.rows,table); name='Oracle_leaky_do_not_report' if args.model_type=='rpn' else f'Model_{args.model_type}'; results[name]=evaluate_selection(ds.rows,lower,r_hat); results[name+'_uncalibrated']=evaluate_selection(ds.rows,r_hat,r_hat)
        if args.include_leaky_oracle_debug and args.model_type=='rpn' and args.msrt:
            leaky=predict_rpn(model,ds,args.batch_size,device); results['Oracle_leaky_do_not_report']=evaluate_selection(ds.rows,lower_bounds_for_rows(leaky,ds.rows,table),leaky)
    for name,kind in [('Severity_only','severity'),('Weighted_post_impact_audit_debug','weighted_post_impact_audit_debug'),('Teacher_oracle','teacher_oracle')]:
        s=baseline_scores(ds.rows,kind); fake=lower_bounds_from_scalar_scores(ds.rows,s); results[name]=evaluate_selection(ds.rows,fake,fake)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(results,indent=2),encoding='utf-8'); csv_path=out.with_suffix('.csv'); keys=sorted({k for m in results.values() for k in m})
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['method']+keys); wr.writeheader(); [wr.writerow({'method':m,**v}) for m,v in results.items()]
    print(json.dumps({'output':str(out),'csv':str(csv_path),'methods':list(results)},indent=2))
if __name__=='__main__': main()

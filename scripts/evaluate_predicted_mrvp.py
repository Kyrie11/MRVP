#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,csv,json,torch
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from tqdm import tqdm
from mrvp.calibration import load_calibration_table,lower_bounds_for_rows,predict_rpn,predict_rpn_with_msrt,quantile_for_group
from mrvp.data.dataset import MRVPDataset,iter_root_batches
from mrvp.data.schema import BOTTLE_NECKS,TOKEN_COUNT,TOKEN_DIM,STRATEGY_COUNT,RECOVERY_HORIZON,SchemaDims
from mrvp.evaluation import baseline_scores,evaluate_selected_indices,evaluate_selection,lower_bounds_from_scalar_scores
from mrvp.models.baselines import DirectActionRiskNetwork,UnstructuredLatentRiskNetwork
from mrvp.models.common import empirical_cvar
from mrvp.models.msrt import MSRT
from mrvp.models.rpn import RecoveryProfileNetwork
from mrvp.selection import admissible_indices
from mrvp.training.checkpoints import load_model
from mrvp.training.loops import to_device
def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def build_profile_model(t,h,scalar=False): return DirectActionRiskNetwork(hidden_dim=h,scalar=scalar) if t=='direct_action_to_risk' else UnstructuredLatentRiskNetwork(hidden_dim=h)
@torch.no_grad()
def score_root_with_msrt(root_batch,root_rows,msrt,rpn,table,num_samples,beta,device):
    batch=to_device(root_batch,device); summaries=[]
    for li,row in enumerate(root_rows):
        single={k:(v[li:li+1] if torch.is_tensor(v) else ([v[li]] if isinstance(v,list) else v)) for k,v in batch.items()}; samples=msrt.sample(single,num_samples=num_samples,deterministic=False); rep={'o_hist':single['o_hist'].repeat_interleave(num_samples,0),'h_ctx':single['h_ctx'].repeat_interleave(num_samples,0),'x_plus':samples['x_plus'],'deg':samples['deg'],'d_deg':samples['d_deg'],'event_tokens':samples['event_tokens'],'has_event_tokens':torch.ones(num_samples,device=device),'world_plus':samples['world_plus']}
        for key in ['action_id','action_vec','rho_imp','harm_bin']:
            if key in single and torch.is_tensor(single[key]): rep[key]=single[key].repeat_interleave(num_samples,0)
        pred=rpn(rep)['r_hat']; q=torch.as_tensor(quantile_for_group(row.get('calib_group',{}),table),device=device,dtype=pred.dtype); lower=pred-q[None,:]; v=lower.min(-1).values; losses=torch.clamp(-v,min=0); cvar=empirical_cvar(losses[None,:],beta=beta,dim=-1).squeeze(0); mean=losses.mean(); summaries.append({'local_index':li,'action_id':int(row['action_id']),'action_name':row.get('action_name',str(row['action_id'])),'harm_bin':int(row['harm_bin']),'risk_cvar':float(cvar.cpu()),'risk_mean':float(mean.cpu()),'score_cvar':float(-cvar.cpu()),'score_mean':float(-mean.cpu()),'mean_lower_V':float(v.mean().cpu()),'p_violation':float((losses>0).float().mean().cpu()),'mean_lower_bounds':lower.mean(0).cpu().numpy().astype(float).tolist()})
    return summaries
def select_from_root_summaries(root_rows,summaries,risk_key):
    adm=admissible_indices(root_rows); return int(min(adm,key=lambda i:float(summaries[i][risk_key])))
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def main():
    p=argparse.ArgumentParser(description='Predicted MRVP: MSRT samples -> RPN -> calibration -> CVaR.'); p.add_argument('--data',required=True); p.add_argument('--msrt',required=True); p.add_argument('--rpn',required=True); p.add_argument('--calibration',default=''); p.add_argument('--split',default='test'); p.add_argument('--output',default='runs/default/eval_predicted_mrvp.json'); p.add_argument('--num-samples',type=int,default=32); p.add_argument('--beta',type=float,default=0.2); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--hidden-dim',type=int,default=256); p.add_argument('--mixture-count',type=int,default=5); p.add_argument('--token-count',type=int,default=TOKEN_COUNT); p.add_argument('--token-dim',type=int,default=TOKEN_DIM); p.add_argument('--strategy-count',type=int,default=STRATEGY_COUNT); p.add_argument('--recovery-horizon',type=int,default=RECOVERY_HORIZON); p.add_argument('--scalar-rpn',action='store_true'); p.add_argument('--direct-model',default=''); p.add_argument('--direct-calibration',default=''); p.add_argument('--direct-model-type',choices=['direct_action_to_risk','unstructured_latent'],default='direct_action_to_risk'); p.add_argument('--include-leaky-oracle-debug',action='store_true'); p.add_argument('--device',default='auto'); p.add_argument('--torch-threads',type=int,default=1); args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); device=auto_device(args.device); ds=MRVPDataset(args.data,args.split,SchemaDims(token_count=args.token_count,token_dim=args.token_dim,recovery_horizon=args.recovery_horizon)); msrt=MSRT(hidden_dim=args.hidden_dim,mixture_count=args.mixture_count,token_count=args.token_count,token_dim=args.token_dim).to(device); load_model(msrt,args.msrt,device,strict=False); rpn=RecoveryProfileNetwork(hidden_dim=args.hidden_dim,token_count=args.token_count,token_dim=args.token_dim,strategy_count=args.strategy_count,recovery_horizon=args.recovery_horizon,scalar=args.scalar_rpn).to(device); load_model(rpn,args.rpn,device,strict=False); table=load_calibration_table(args.calibration or None)
    oracle=predict_rpn_with_msrt(rpn,msrt,ds,'oracle_clean',args.batch_size,device); results={'Oracle_clean_true_transition':evaluate_selection(ds.rows,lower_bounds_for_rows(oracle,ds.rows,table),oracle),'Oracle_clean_true_transition_uncalibrated':evaluate_selection(ds.rows,oracle,oracle)}
    if args.include_leaky_oracle_debug:
        leaky=predict_rpn(rpn,ds,args.batch_size,device); results['Oracle_leaky_do_not_report']=evaluate_selection(ds.rows,lower_bounds_for_rows(leaky,ds.rows,table),leaky)
    scores_cvar=np.full((len(ds),),np.nan,dtype=np.float32); scores_mean=np.full((len(ds),),np.nan,dtype=np.float32); lower_mean=np.zeros((len(ds),len(BOTTLE_NECKS)),dtype=np.float32); selected_cvar=[]; selected_mean=[]; per_root=[]
    for root_id,indices,root_rows,root_batch in tqdm(iter_root_batches(ds),total=len(ds.root_to_indices),desc='predicted_mrvp'):
        summaries=score_root_with_msrt(root_batch,root_rows,msrt,rpn,table,args.num_samples,args.beta,device); bc=select_from_root_summaries(root_rows,summaries,'risk_cvar'); bm=select_from_root_summaries(root_rows,summaries,'risk_mean'); selected_cvar.append(indices[bc]); selected_mean.append(indices[bm])
        for li,gi in enumerate(indices): scores_cvar[gi]=summaries[li]['score_cvar']; scores_mean[gi]=summaries[li]['score_mean']; lower_mean[gi]=np.asarray(summaries[li]['mean_lower_bounds'])
        adm=admissible_indices(root_rows); teacher=max(adm,key=lambda i:float(root_rows[i]['s_star'])) if adm else 0; per_root.append({'root_id':root_id,'admissible_local_indices':adm,'selected_cvar_local':bc,'selected_cvar_global':indices[bc],'selected_mean_local':bm,'selected_mean_global':indices[bm],'teacher_best_local':teacher,'teacher_best_global':indices[teacher],'selected_cvar_s_star':float(root_rows[bc]['s_star']),'selected_mean_s_star':float(root_rows[bm]['s_star']),'teacher_best_s_star':float(root_rows[teacher]['s_star']),'candidate_summaries':summaries})
    results['Predicted_MRVP_MSRT_CVaR']=evaluate_selected_indices(ds.rows,selected_cvar,scores=scores_cvar,lower_bounds=lower_mean); results['Predicted_MRVP_MSRT_mean_risk']=evaluate_selected_indices(ds.rows,selected_mean,scores=scores_mean,lower_bounds=lower_mean)
    if args.direct_model:
        direct=build_profile_model(args.direct_model_type,args.hidden_dim,args.scalar_rpn).to(device); load_model(direct,args.direct_model,device,strict=False); dr=predict_rpn(direct,ds,args.batch_size,device); results[f'Direct_action_baseline_{args.direct_model_type}']=evaluate_selection(ds.rows,lower_bounds_for_rows(dr,ds.rows,load_calibration_table(args.direct_calibration or None)),dr)
    for name,kind in [('Severity_only','severity'),('Weighted_post_impact_audit_debug','weighted_post_impact_audit_debug'),('Teacher_oracle','teacher_oracle')]:
        s=baseline_scores(ds.rows,kind); fake=lower_bounds_from_scalar_scores(ds.rows,s); results[name]=evaluate_selection(ds.rows,fake,fake)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8'); csv_path=out.with_suffix('.csv'); keys=sorted({k for m in results.values() for k in m})
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['method']+keys); wr.writeheader(); [wr.writerow({'method':m,**v}) for m,v in results.items()]
    detail=out.with_name(out.stem+'_per_root.jsonl'); write_jsonl(detail,per_root); print(json.dumps({'output':str(out),'csv':str(csv_path),'per_root':str(detail),'methods':list(results),'roots':len(ds.root_to_indices),'rows':len(ds)},indent=2,ensure_ascii=False))
if __name__=='__main__': main()

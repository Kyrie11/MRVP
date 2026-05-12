from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence, List
import numpy as np
import torch
from mrvp.calibration import quantile_for_group
from mrvp.models.common import empirical_cvar

def admissible_indices(rows:Sequence[Mapping[str,Any]])->List[int]:
    if not rows: return []
    min_bin=min(int(r['harm_bin']) for r in rows); return [i for i,r in enumerate(rows) if int(r['harm_bin'])==min_bin]
def select_from_scores(rows:Sequence[Mapping[str,Any]],lower_bounds:np.ndarray,beta:float=0.2)->Dict[str,Any]:
    adm=admissible_indices(rows)
    if not adm: raise ValueError('No candidate actions.')
    v=lower_bounds[adm].min(1); losses=np.maximum(-v,0.0); best_local=int(np.argmin(losses)); best_idx=adm[best_local]
    return {'selected_local_index':best_idx,'selected_action_id':int(rows[best_idx]['action_id']),'tail_risk':float(losses[best_local]),'admissible_indices':adm,'lower_V':float(v[best_local]),'lower_bounds':lower_bounds[best_idx].tolist()}
@torch.no_grad()
def select_action_with_models(root_batch,root_rows,msrt,rpn,calibration_table,num_samples:int=32,beta:float=0.2,device:str|torch.device='cpu'):
    device=torch.device(device); msrt.to(device).eval(); rpn.to(device).eval(); batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in root_batch.items()}; adm=admissible_indices(root_rows)
    risks=[]; summaries=[]
    for idx in adm:
        single={k:(v[idx:idx+1] if torch.is_tensor(v) else ([v[idx]] if isinstance(v,list) else v)) for k,v in batch.items()}; samples=msrt.sample(single,num_samples=num_samples,deterministic=False)
        rep={'o_hist':single['o_hist'].repeat_interleave(num_samples,0),'h_ctx':single['h_ctx'].repeat_interleave(num_samples,0),'x_plus':samples['x_plus'],'deg':samples['deg'],'d_deg':samples['d_deg'],'event_tokens':samples['event_tokens'],'has_event_tokens':torch.ones(num_samples,device=device),'world_plus':samples['world_plus']}
        for key in ['action_id','action_vec','rho_imp','harm_bin']:
            if key in single and torch.is_tensor(single[key]): rep[key]=single[key].repeat_interleave(num_samples,0)
        pred=rpn(rep)['r_hat']; q=torch.as_tensor(quantile_for_group(root_rows[idx].get('calib_group',{}),calibration_table),device=device,dtype=pred.dtype); lower=pred-q[None,:]; v=lower.min(-1).values; losses=torch.clamp(-v,min=0.0); risk=empirical_cvar(losses[None,:],beta=beta,dim=-1).squeeze(0); risks.append(risk); summaries.append({'candidate_index':idx,'action_id':int(root_rows[idx]['action_id']),'risk':float(risk.cpu()),'mean_lower_V':float(v.mean().cpu()),'p_violation':float((losses>0).float().mean().cpu())})
    rt=torch.stack(risks); best_local=int(torch.argmin(rt).cpu()); best_idx=adm[best_local]
    return {'selected_local_index':best_idx,'selected_action_id':int(root_rows[best_idx]['action_id']),'tail_risk':float(rt[best_local].cpu()),'admissible_indices':adm,'candidate_summaries':summaries}

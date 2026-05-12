from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from mrvp.data.schema import BOTTLE_NECKS
from mrvp.selection import select_from_scores

def macro_f1(y_true,y_pred,num_classes:int)->float:
    y_true=np.asarray(y_true,dtype=int); y_pred=np.asarray(y_pred,dtype=int); out=[]
    for c in range(num_classes):
        tp=np.sum((y_true==c)&(y_pred==c)); fp=np.sum((y_true!=c)&(y_pred==c)); fn=np.sum((y_true==c)&(y_pred!=c)); den=2*tp+fp+fn; out.append(0.0 if den==0 else float(2*tp/den))
    return float(np.mean(out))
def group_indices_by_root(rows):
    by=defaultdict(list)
    for i,r in enumerate(rows): by[str(r['root_id'])].append(i)
    return dict(by)
def pair_accuracy(rows,scores,eps_s:float=0.25)->float:
    total=correct=0; scores=np.asarray(scores).reshape(-1)
    for _,idxs in group_indices_by_root(rows).items():
        min_bin=min(int(rows[i]['harm_bin']) for i in idxs); group=[i for i in idxs if int(rows[i]['harm_bin'])==min_bin and float(rows[i].get('is_harm_admissible',1.0))>=0.5]
        for p,i in enumerate(group):
            for j in group[p+1:]:
                if not(np.isfinite(scores[i]) and np.isfinite(scores[j])): continue
                ds=float(rows[i]['s_star'])-float(rows[j]['s_star'])
                if abs(ds)<eps_s: continue
                dp=float(scores[i])-float(scores[j]); correct+=int(dp*ds>0); total+=1
    return float(correct/total) if total else float('nan')
def evaluate_selection(rows,lower_bounds,r_hat=None,eps_s:float=0.25)->Dict[str,float]:
    by=group_indices_by_root(rows); selected=[]; envelope=[]; violation=[]; success=[]; regrets=[]; svals=[]; frr_num=frr_den=0; coverage_hits=np.zeros(len(BOTTLE_NECKS)); coverage_tot=np.zeros(len(BOTTLE_NECKS))
    for _,idxs in by.items():
        local=[rows[i] for i in idxs]; lower=lower_bounds[idxs]; sel=select_from_scores(local,lower); li=sel['selected_local_index']; gi=idxs[li]; selected.append(gi); min_bin=min(int(rows[i]['harm_bin']) for i in idxs); envelope.append(int(rows[gi]['harm_bin']!=min_bin)); s=float(rows[gi]['s_star']); svals.append(s); violation.append(max(-s,0.0)); success.append(int(s>=0)); adm=[i for i in idxs if int(rows[i]['harm_bin'])==min_bin]; oracle=max(float(rows[i]['s_star']) for i in adm); regrets.append(max(0.0,oracle-s)); v_lower=float(lower[li].min()); frr_den+=int(v_lower>0); frr_num+=int(v_lower>0 and s<0)
    for i,r in enumerate(rows): coverage_hits+=(np.asarray(r['r_star'])>=lower_bounds[i]); coverage_tot+=1
    scores=lower_bounds.min(1); pred_b=np.argmin(lower_bounds,1); true_b=np.asarray([int(r['b_star']) for r in rows]); env=float(np.mean(envelope)) if envelope else float('nan'); pair=pair_accuracy(rows,scores,eps_s)
    out={'roots':float(len(by)),'actions':float(len(rows)),'envelope_violation':env,'envelope_violation_rate':env,'first_impact_harm_violation_rate':env,'pair_accuracy':pair,'same_harm_pair_accuracy':pair,'frr':float(frr_num/max(frr_den,1)),'worst_bottleneck_violation_depth':float(np.mean(violation)) if violation else float('nan'),'mean_violation_depth':float(np.mean(violation)) if violation else float('nan'),'closed_loop_recovery_success_proxy':float(np.mean(success)) if success else float('nan'),'selected_recoverable_rate':float(np.mean(success)) if success else float('nan'),'selected_s_star_mean':float(np.mean(svals)) if svals else float('nan'),'active_bottleneck_f1':macro_f1(true_b,pred_b,len(BOTTLE_NECKS)),'coverage':float(np.mean(coverage_hits/np.maximum(coverage_tot,1))),'shift_regret':float(np.mean(regrets)) if regrets else float('nan')}
    for b,n in enumerate(BOTTLE_NECKS): out[f'coverage_{n}']=float(coverage_hits[b]/max(coverage_tot[b],1))
    return out
def baseline_scores(rows,kind:str='severity')->np.ndarray:
    if kind=='severity': return -np.asarray([float(r['harm_bin'])+0.01*float(r['rho_imp']) for r in rows],dtype=np.float32)
    if kind in {'weighted_post_impact','weighted_post_impact_audit_debug'}:
        vals=[]
        for r in rows:
            z=np.asarray(r.get('z_mech',np.zeros(32)),dtype=np.float32); x=np.asarray(r['x_plus'],dtype=np.float32); vals.append(0.4*z[19]+0.3*z[21]+0.1*z[20]-0.2*abs(x[5])-0.2*float(r['rho_imp']))
        return np.asarray(vals,dtype=np.float32)
    if kind=='teacher_oracle': return np.asarray([float(r['s_star']) for r in rows],dtype=np.float32)
    raise ValueError(f'Unknown baseline kind: {kind}')
def lower_bounds_from_scalar_scores(rows,scores): return np.repeat(np.asarray(scores)[:,None],len(BOTTLE_NECKS),axis=1)
def evaluate_selected_indices(rows,selected_indices,scores=None,lower_bounds=None,eps_s:float=0.25):
    by=group_indices_by_root(rows); selset=set(map(int,selected_indices)); chosen=[]; env=[]; viol=[]; succ=[]; regret=[]; svals=[]; frr_num=frr_den=0
    for _,idxs in by.items():
        c=[i for i in idxs if i in selset]
        if not c: continue
        gi=c[0]; chosen.append(gi); min_bin=min(int(rows[i]['harm_bin']) for i in idxs); env.append(int(rows[gi]['harm_bin']!=min_bin)); s=float(rows[gi]['s_star']); svals.append(s); viol.append(max(-s,0)); succ.append(int(s>=0)); adm=[i for i in idxs if int(rows[i]['harm_bin'])==min_bin]; oracle=max(float(rows[i]['s_star']) for i in adm); regret.append(max(0,oracle-s))
        if lower_bounds is not None: vl=float(np.asarray(lower_bounds)[gi].min()); frr_den+=int(vl>0); frr_num+=int(vl>0 and s<0)
    envr=float(np.mean(env)) if env else float('nan'); out={'roots':float(len(chosen)),'actions':float(len(rows)),'envelope_violation':envr,'envelope_violation_rate':envr,'first_impact_harm_violation_rate':envr,'frr':float(frr_num/max(frr_den,1)) if lower_bounds is not None else float('nan'),'worst_bottleneck_violation_depth':float(np.mean(viol)) if viol else float('nan'),'mean_violation_depth':float(np.mean(viol)) if viol else float('nan'),'closed_loop_recovery_success_proxy':float(np.mean(succ)) if succ else float('nan'),'selected_recoverable_rate':float(np.mean(succ)) if succ else float('nan'),'selected_s_star_mean':float(np.mean(svals)) if svals else float('nan'),'shift_regret':float(np.mean(regret)) if regret else float('nan'),'pair_accuracy':pair_accuracy(rows,scores,eps_s) if scores is not None else float('nan'),'same_harm_pair_accuracy':pair_accuracy(rows,scores,eps_s) if scores is not None else float('nan')}
    if lower_bounds is not None:
        arr=np.asarray(lower_bounds,dtype=np.float32); hits=np.zeros(len(BOTTLE_NECKS)); tot=np.zeros(len(BOTTLE_NECKS))
        for i,r in enumerate(rows): hits+=(np.asarray(r['r_star'])>=arr[i]); tot+=1
        out['coverage']=float(np.mean(hits/np.maximum(tot,1))); pred_b=np.argmin(arr,1); true_b=np.asarray([int(r['b_star']) for r in rows]); out['active_bottleneck_f1']=macro_f1(true_b,pred_b,len(BOTTLE_NECKS));
        for b,n in enumerate(BOTTLE_NECKS): out[f'coverage_{n}']=float(hits[b]/max(tot[b],1))
    else:
        out['coverage']=out['active_bottleneck_f1']=float('nan')
        for n in BOTTLE_NECKS: out[f'coverage_{n}']=float('nan')
    return out

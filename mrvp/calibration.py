from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from mrvp.data.dataset import MRVPDataset,mrvp_collate
from mrvp.data.schema import BOTTLE_NECKS,group_dict_to_key
from mrvp.training.loops import to_device

def conformal_quantile(values:Sequence[float],delta:float)->float:
    vals=np.sort(np.asarray(list(values),dtype=np.float32));
    if vals.size==0: return 0.0
    k=int(np.ceil((vals.size+1)*(1.0-delta)))-1; return float(vals[int(np.clip(k,0,vals.size-1))])
@torch.no_grad()
def predict_rpn(model,dataset:MRVPDataset,batch_size:int=256,device:str|torch.device='cpu')->np.ndarray:
    device=torch.device(device); model.to(device).eval(); loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=mrvp_collate); preds=[]
    for batch in tqdm(loader,desc='predict_rpn',leave=False):
        batch=to_device(batch,device); preds.append(model(batch)['r_hat'].detach().cpu().numpy())
    return np.concatenate(preds,0)
@torch.no_grad()
def predict_rpn_with_msrt(rpn,msrt,dataset:MRVPDataset,mode:str='oracle_clean',batch_size:int=256,device:str|torch.device='cpu')->np.ndarray:
    from mrvp.training.rpn_batches import build_rpn_batch_from_msrt
    device=torch.device(device); rpn.to(device).eval(); msrt.to(device).eval(); loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=mrvp_collate); preds=[]
    for batch in tqdm(loader,desc=f'predict_rpn_{mode}',leave=False):
        batch=to_device(batch,device); rb=build_rpn_batch_from_msrt(msrt,batch,mode=mode,detach_msrt=True); preds.append(rpn(rb)['r_hat'].detach().cpu().numpy())
    return np.concatenate(preds,0)
def _group_rows_by_root(rows):
    by={}
    for i,r in enumerate(rows): by.setdefault(str(r['root_id']),[]).append(i)
    return by
def fit_calibration_table(rows,r_hat,delta_b:float|Sequence[float]=0.02,n_min:int=20)->Dict[str,Any]:
    deltas=[float(delta_b)]*len(BOTTLE_NECKS) if isinstance(delta_b,(float,int)) else list(map(float,delta_b)); residuals={lvl:{} for lvl in ['full','medium','coarse','global']}
    for root,idxs in _group_rows_by_root(rows).items():
        min_bin=min(int(rows[i]['harm_bin']) for i in idxs); adm=[i for i in idxs if int(rows[i]['harm_bin'])==min_bin]
        if not adm: continue
        vals=[max(0.0,float(r_hat[i,b])-float(rows[i]['r_star'][b])) for b in range(len(BOTTLE_NECKS)) for i in adm]
        byb=[max([max(0.0,float(r_hat[i,b])-float(rows[i]['r_star'][b])) for i in adm] or [0.0]) for b in range(len(BOTTLE_NECKS))]
        group=rows[adm[0]].get('calib_group',{})
        for lvl in residuals:
            key=group_dict_to_key(group,lvl); residuals[lvl].setdefault(key,{b:[] for b in range(len(BOTTLE_NECKS))})
            for b,val in enumerate(byb): residuals[lvl][key][b].append(val)
    quant={lvl:{} for lvl in residuals}; counts={lvl:{} for lvl in residuals}
    for lvl,groups in residuals.items():
        for key,by_b in groups.items(): counts[lvl][key]=min(len(v) for v in by_b.values()) if by_b else 0; quant[lvl][key]=[conformal_quantile(by_b[b],deltas[b]) for b in range(len(BOTTLE_NECKS))]
    return {'version':2,'bottlenecks':BOTTLE_NECKS,'delta_b':deltas,'n_min':int(n_min),'quantiles':quant,'counts':counts,'fallback_order':['full','medium','coarse','global']}
def save_calibration_table(table,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(table,indent=2),encoding='utf-8')
def load_calibration_table(path):
    if path is None: return {'quantiles':{'global':{'global':[0.0]*len(BOTTLE_NECKS)}},'counts':{'global':{'global':999999}},'n_min':1,'fallback_order':['global']}
    return json.loads(Path(path).read_text(encoding='utf-8'))
def quantile_for_group(group,table):
    n_min=int(table.get('n_min',1))
    for lvl in table.get('fallback_order',['full','medium','coarse','global']):
        key=group_dict_to_key(group,lvl); q=table.get('quantiles',{}).get(lvl,{}).get(key); c=int(table.get('counts',{}).get(lvl,{}).get(key,0))
        if q is not None and c>=n_min: return np.asarray(q,dtype=np.float32)
    return np.asarray(table.get('quantiles',{}).get('global',{}).get('global',[0.0]*len(BOTTLE_NECKS)),dtype=np.float32)
def lower_bounds_for_rows(r_hat,rows,table):
    lowers=np.zeros_like(r_hat,dtype=np.float32)
    for i,row in enumerate(rows): lowers[i]=r_hat[i]-quantile_for_group(row.get('calib_group',{}),table)
    return lowers

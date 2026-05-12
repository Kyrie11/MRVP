from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .schema import SchemaDims, row_to_numpy

def iter_jsonl(paths: str | Path | Sequence[str | Path]) -> Iterable[Dict[str, Any]]:
    files=[Path(paths)] if isinstance(paths,(str,Path)) else [Path(x) for x in paths]
    expanded=[]
    for p in files:
        expanded += sorted(list(p.glob('*.jsonl'))+list(p.glob('*.json'))) if p.is_dir() else [p]
    for file in expanded:
        if file.suffix=='.json':
            obj=json.loads(file.read_text(encoding='utf-8'))
            if isinstance(obj,list):
                for row in obj: yield row
            elif isinstance(obj,Mapping) and 'rows' in obj:
                for row in obj['rows']: yield row
            else: yield obj
        else:
            with file.open('r',encoding='utf-8') as f:
                for line in f:
                    line=line.strip()
                    if line: yield json.loads(line)

class MRVPDataset(Dataset):
    def __init__(self, path: str|Path|Sequence[str|Path], split: Optional[str]=None, dims: SchemaDims=SchemaDims(), keep_rows: bool=True)->None:
        self.path=path; self.split=split; self.dims=dims; self.rows=[]; self.raw_rows=[] if keep_rows else None
        for row in iter_jsonl(path):
            if split is not None and str(row.get('split','train'))!=split: continue
            self.rows.append(row_to_numpy(row,dims))
            if keep_rows: self.raw_rows.append(row)
        if not self.rows: raise ValueError(f'No MRVP rows found in {path!r} for split={split!r}')
        self.root_to_indices: Dict[str,List[int]]={}
        for i,row in enumerate(self.rows): self.root_to_indices.setdefault(str(row['root_id']),[]).append(i)
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx:int)->Dict[str,Any]:
        r=self.rows[idx]
        tensor_float=['action_vec','o_hist','h_ctx','x_t','rho_imp','event_time','x_minus','x_plus','deg','d_deg','event_tokens','has_event_tokens','world_plus','z_mech','audit_mech','teacher_u','teacher_traj','m_star','r_star','s_star','root_action_complete','is_harm_admissible','root_same_harm_s_spread']
        item={'idx':torch.tensor(idx,dtype=torch.long),'action_id':torch.tensor(r['action_id'],dtype=torch.long),'harm_bin':torch.tensor(r['harm_bin'],dtype=torch.long),'event_type_id':torch.tensor(r['event_type_id'],dtype=torch.long),'b_star':torch.tensor(r['b_star'],dtype=torch.long),'root_action_count':torch.tensor(r['root_action_count'],dtype=torch.long),'root_expected_action_count':torch.tensor(r['root_expected_action_count'],dtype=torch.long),'root_min_harm_bin':torch.tensor(r['root_min_harm_bin'],dtype=torch.long),'root_admissible_count':torch.tensor(r['root_admissible_count'],dtype=torch.long)}
        for k in tensor_float:
            if k in r: item[k]=torch.as_tensor(r[k],dtype=torch.float32)
        for k in ['root_id','split','family','event_type','calib_group']:
            item[k]=r[k]
        return item
    def raw(self,idx:int): return None if self.raw_rows is None else self.raw_rows[idx]

def mrvp_collate(batch: Sequence[Dict[str,Any]])->Dict[str,Any]:
    out={}; tensor_keys=['idx','action_id','action_vec','o_hist','h_ctx','x_t','rho_imp','harm_bin','event_type_id','event_time','x_minus','x_plus','deg','d_deg','event_tokens','has_event_tokens','world_plus','z_mech','audit_mech','teacher_u','teacher_traj','m_star','r_star','b_star','s_star','root_action_count','root_expected_action_count','root_action_complete','root_min_harm_bin','is_harm_admissible','root_admissible_count','root_same_harm_s_spread']
    for k in tensor_keys:
        if k in batch[0]: out[k]=torch.stack([x[k] for x in batch],0)
    for k in ['root_id','split','family','event_type','calib_group']:
        out[k]=[x[k] for x in batch]
    return out

def iter_root_batches(dataset: MRVPDataset, shuffle: bool=False, seed:int=0):
    roots=list(dataset.root_to_indices.keys())
    if shuffle: np.random.default_rng(seed).shuffle(roots)
    for root_id in roots:
        idxs=list(dataset.root_to_indices[root_id]); rows=[dataset.rows[i] for i in idxs]; batch=mrvp_collate([dataset[i] for i in idxs]); yield root_id,idxs,rows,batch

def rows_by_root(dataset: MRVPDataset)->Dict[str,List[Dict[str,Any]]]:
    out={}
    for r in dataset.rows: out.setdefault(str(r['root_id']),[]).append(r)
    return out

def root_split_summary(path: str|Path)->Dict[str,int]:
    counts={}; roots={}
    for r in iter_jsonl(path):
        s=str(r.get('split','train')); counts[s]=counts.get(s,0)+1; roots.setdefault(s,set()).add(str(r.get('root_id','0')))
    return {f'rows_{k}':v for k,v in counts.items()} | {f'roots_{k}':len(v) for k,v in roots.items()}

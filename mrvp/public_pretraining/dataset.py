from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import torch
from torch.utils.data import Dataset
from mrvp.data.schema import CTX_DIM,ensure_1d_float,ensure_hist
class PublicPretrainDataset(Dataset):
    def __init__(self,path):
        self.rows=[]
        with Path(path).open('r',encoding='utf-8') as f:
            for line in f:
                if line.strip(): self.rows.append(json.loads(line))
        if not self.rows: raise ValueError(f'No pretrain records found in {path}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx:int)->Dict[str,Any]:
        r=self.rows[idx]; return {'o_hist':torch.as_tensor(ensure_hist(r.get('o_hist')),dtype=torch.float32),'h_ctx':torch.as_tensor(ensure_1d_float(r.get('h_ctx'),CTX_DIM),dtype=torch.float32),'future_delta':torch.as_tensor(ensure_1d_float(r.get('future_delta'),2),dtype=torch.float32),'affordance':torch.as_tensor(ensure_1d_float(r.get('affordance'),4),dtype=torch.float32),'dataset':r.get('dataset','unknown'),'scenario_id':r.get('scenario_id',str(idx))}
def pretrain_collate(batch):
    out={k:torch.stack([b[k] for b in batch],0) for k in ['o_hist','h_ctx','future_delta','affordance']}; out['dataset']=[b['dataset'] for b in batch]; out['scenario_id']=[b['scenario_id'] for b in batch]; return out

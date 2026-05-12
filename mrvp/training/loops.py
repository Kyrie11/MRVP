from __future__ import annotations
from typing import Dict, Mapping
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
def to_device(batch,device:torch.device):
    if isinstance(batch,dict): return {k:to_device(v,device) for k,v in batch.items()}
    if torch.is_tensor(batch): return batch.to(device)
    if isinstance(batch,(list,tuple)): return batch
    return batch
def scalar_logs(logs:Mapping[str,torch.Tensor|float])->Dict[str,float]: return {k:float(v.detach().cpu().mean()) if torch.is_tensor(v) else float(v) for k,v in logs.items()}
def train_one_epoch(model,loader:DataLoader,optimizer,device:torch.device,loss_kwargs:dict|None=None,desc:str='train')->Dict[str,float]:
    model.train(); totals={}; n=0
    for batch in tqdm(loader,desc=desc,leave=False):
        batch=to_device(batch,device); optimizer.zero_grad(set_to_none=True); logs=model.loss(batch,**(loss_kwargs or {})); logs['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step(); s=scalar_logs(logs); totals={k:totals.get(k,0)+v for k,v in s.items()}; n+=1
    return {k:v/max(n,1) for k,v in totals.items()}
@torch.no_grad()
def eval_loss(model,loader:DataLoader,device:torch.device,loss_kwargs:dict|None=None,desc:str='val')->Dict[str,float]:
    model.eval(); totals={}; n=0
    for batch in tqdm(loader,desc=desc,leave=False):
        batch=to_device(batch,device); s=scalar_logs(model.loss(batch,**(loss_kwargs or {}))); totals={k:totals.get(k,0)+v for k,v in s.items()}; n+=1
    return {k:v/max(n,1) for k,v in totals.items()}

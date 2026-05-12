from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from mrvp.data.schema import ACTOR_FEAT_DIM,CTX_DIM,HIST_LEN,MAX_AGENTS
def write_pretrain_jsonl(records:Iterable[Mapping[str,Any]],output:str|Path)->int:
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); n=0
    with output.open('w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(dict(r),separators=(',',':'))+'\n'); n+=1
    return n
def make_o_hist(agent_histories:Sequence[np.ndarray],ego_index:int=0,hist_len:int=HIST_LEN,max_agents:int=MAX_AGENTS)->np.ndarray:
    out=np.zeros((hist_len,max_agents,ACTOR_FEAT_DIM),dtype=np.float32); order=[ego_index]+[i for i in range(len(agent_histories)) if i!=ego_index]
    for ai,src in enumerate(order[:max_agents]):
        arr=np.asarray(agent_histories[src],dtype=np.float32)
        if arr.ndim!=2 or arr.size==0: continue
        t=min(hist_len,arr.shape[0]); f=min(ACTOR_FEAT_DIM,arr.shape[1]); out[-t:,ai,:f]=arr[-t:,:f]
        if f<8: out[-t:,ai,5:8]=[4.6,1.9,1.0]
        out[-t:,ai,8]=1.0
    return out
def basic_context(o_hist:np.ndarray,lane_width:float=3.5,speed_limit:float=13.9,dataset_id:float=0.0)->np.ndarray:
    h=np.zeros(CTX_DIM,dtype=np.float32); ego=o_hist[-1,0]; agents=o_hist[-1]; valid=agents[:,8]>0; d=[float(np.linalg.norm(a[:2]-ego[:2])) for a in agents[1:][valid[1:]]]; md=min(d) if d else 50.0; density=float(np.sum(valid)-1)/max(1,o_hist.shape[1]-1); h[2]=lane_width; h[3]=lane_width/2; h[4]=lane_width/2; h[5]=md; h[7]=md; h[8]=md/max(float(np.linalg.norm(ego[3:5])),0.1); h[9]=30; h[10]=density; h[11]=speed_limit; h[12]=0.9; h[15]=lane_width; h[22]=1; h[24]=dataset_id; return h
def make_record(dataset,scenario_id,o_hist,future_xy=None,h_ctx=None):
    o_hist=np.asarray(o_hist,dtype=np.float32); h_ctx=basic_context(o_hist) if h_ctx is None else np.asarray(h_ctx,dtype=np.float32); future_delta=np.zeros(2,dtype=np.float32) if future_xy is None else np.asarray(future_xy,dtype=np.float32).reshape(-1,2)[-1]-o_hist[-1,0,:2]; affordance=np.asarray([h_ctx[2],h_ctx[7],h_ctx[9],h_ctx[10]],dtype=np.float32); return {'dataset':dataset,'scenario_id':scenario_id,'o_hist':o_hist.tolist(),'h_ctx':h_ctx.tolist(),'future_delta':future_delta.tolist(),'affordance':affordance.tolist()}

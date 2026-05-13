#!/usr/bin/env python3
from __future__ import annotations
import sys, argparse, json
from pathlib import Path as _Path
sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from collections import defaultdict
import numpy as np, torch
from torch.utils.data import DataLoader
from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.schema import EVENT_TYPES, TOKEN_COUNT, TOKEN_DIM, SchemaDims
from mrvp.models.msrt import MSRT
from mrvp.training.checkpoints import load_model
from mrvp.training.loops import to_device

def macro_f1(y,p,n):
    fs=[]
    for c in range(n):
        tp=np.sum((y==c)&(p==c)); fp=np.sum((y!=c)&(p==c)); fn=np.sum((y==c)&(p!=c)); d=2*tp+fp+fn; fs.append(0.0 if d==0 else 2*tp/d)
    return float(np.mean(fs))
@torch.no_grad()
def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--data",required=True); pa.add_argument("--msrt",required=True); pa.add_argument("--split",default="test"); pa.add_argument("--output",default="msrt_transition.json"); pa.add_argument("--batch-size",type=int,default=256); pa.add_argument("--hidden-dim",type=int,default=256); pa.add_argument("--mixture-count",type=int,default=5); pa.add_argument("--token-count",type=int,default=TOKEN_COUNT); pa.add_argument("--token-dim",type=int,default=TOKEN_DIM); pa.add_argument("--device",default="auto"); args=pa.parse_args(); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    ds=MRVPDataset(args.data,args.split,SchemaDims(token_count=args.token_count,token_dim=args.token_dim)); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=mrvp_collate); model=MSRT(hidden_dim=args.hidden_dim,mixture_count=args.mixture_count,token_count=args.token_count,token_dim=args.token_dim).to(device); load_model(model,args.msrt,device,strict=False); model.eval(); y=[]; p=[]; time=[]; pos=[]; yaw=[]; vel=[]; wmse=[]; dm=[]; groups=defaultdict(lambda:{"n":0,"pos":[]})
    for batch in loader:
        mb=to_device(batch,device); out=model(mb); pred=out["event_logits"].argmax(-1).cpu().numpy(); true=batch["event_type_id"].numpy(); y.append(true); p.append(pred); time.extend((out["event_time"].cpu()-batch["event_time"]).abs().numpy().tolist()); mean=out["mix_mean"][:,0].cpu(); xp=batch["x_plus"]; pos.extend((mean[:,0:2]-xp[:,0:2]).abs().mean(1).numpy().tolist()); yaw.extend((mean[:,2]-xp[:,2]).abs().numpy().tolist()); vel.extend((mean[:,3:5]-xp[:,3:5]).abs().mean(1).numpy().tolist()); wmse.extend(((out["world_plus"].cpu()-batch["world_plus"])**2).mean(1).numpy().tolist()); dm.extend((out["deg"].cpu()-batch["deg"]).abs().mean(1).numpy().tolist())
        for i,e in enumerate(batch["event_type"]): groups[f"event_type={e}"]["n"]+=1; groups[f"event_type={e}"]["pos"].append(pos[-len(batch['event_type'])+i])
    y=np.concatenate(y); p=np.concatenate(p); maj=np.full_like(y,np.bincount(y,minlength=len(EVENT_TYPES)).argmax())
    out={"event_macro_f1":macro_f1(y,p,len(EVENT_TYPES)),"event_majority_f1":macro_f1(y,maj,len(EVENT_TYPES)),"event_time_mae":float(np.mean(time)),"x_plus_pos_mae":float(np.mean(pos)),"x_plus_yaw_mae":float(np.mean(yaw)),"x_plus_vel_mae":float(np.mean(vel)),"world_mse":float(np.mean(wmse)),"world_r2":float(1.0-np.mean(wmse)/(np.var(wmse)+1e-6)),"deg_mae":float(np.mean(dm)),"same_root_action_discrimination":float('nan'),"same_harm_pair_representation_auc":float('nan'),"groups":{k:{"n":v["n"],"x_plus_pos_mae":float(np.mean(v["pos"])) if v["pos"] else None} for k,v in groups.items()}}
    _Path(args.output).parent.mkdir(parents=True,exist_ok=True); _Path(args.output).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8"); print(json.dumps({"output":args.output,**{k:out[k] for k in ["event_macro_f1","x_plus_pos_mae","world_mse"]}},indent=2))
if __name__=="__main__": main()

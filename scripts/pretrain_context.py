#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json,torch
from pathlib import Path
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from mrvp.models.context import ContextPretrainingHead,SceneContextEncoder
from mrvp.public_pretraining.dataset import PublicPretrainDataset,pretrain_collate
from mrvp.training.checkpoints import save_checkpoint
from mrvp.training.loops import to_device
def auto_device(name): return torch.device('cuda' if name=='auto' and torch.cuda.is_available() else ('cpu' if name=='auto' else name))
def main():
    p=argparse.ArgumentParser(description='Pretrain MRVP context encoder.'); p.add_argument('--data',required=True); p.add_argument('--output',default='runs/default/context_encoder.pt'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--device',default='auto'); p.add_argument('--torch-threads',type=int,default=1); args=p.parse_args(); torch.set_num_threads(max(1,args.torch_threads)); ds=PublicPretrainDataset(args.data); nv=max(1,int(0.1*len(ds))) if len(ds)>10 else 0; train_ds,val_ds=random_split(ds,[len(ds)-nv,nv],generator=torch.Generator().manual_seed(0)) if nv else (ds,None); tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=pretrain_collate); vl=DataLoader(val_ds,batch_size=args.batch_size,collate_fn=pretrain_collate) if val_ds else None; device=auto_device(args.device); model=ContextPretrainingHead(SceneContextEncoder(out_dim=128)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    for e in range(1,args.epochs+1):
        model.train(); total=n=0
        for batch in tqdm(tl,desc=f'context train {e}',leave=False): batch=to_device(batch,device); opt.zero_grad(); loss=model.loss(batch); loss.backward(); opt.step(); total+=float(loss.cpu()); n+=1
        logs={'train_loss':total/max(n,1)}
        if vl:
            model.eval(); vt=vn=0
            with torch.no_grad():
                for batch in vl: batch=to_device(batch,device); loss=model.loss(batch); vt+=float(loss.cpu()); vn+=1
            logs['val_loss']=vt/max(vn,1)
        print(json.dumps({'epoch':e,**logs}))
    save_checkpoint(args.output,model.encoder,{'stage':'context_pretraining','args':vars(args)}); print(json.dumps({'output':args.output,'records':len(ds)}))
if __name__=='__main__': main()

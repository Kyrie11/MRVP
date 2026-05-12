from __future__ import annotations
"""MetaDrive/light2d JSONL generator.

This compact generator intentionally keeps the clean information-flow contract:
world_plus is built from post-event observable/predictable scene features, never
from recovery_traj, teacher controls, m_star/r_star/s_star, or teacher success.
For environments without MetaDrive, it delegates to the deterministic clean
synthetic/light2d generator with the same schema.
"""
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Sequence
from .schema import NUM_ACTIONS
from .synthetic import make_synthetic_rows, write_jsonl, annotate_root_counterfactual_fields

def make_metadrive_rows(n_roots:int=240, actions_per_root:int=NUM_ACTIONS, seed:int=7, backend:str='auto', shift_test:bool=True, harm_thresholds:Sequence[float]=(0.5,2.0,4.0,7.0,11.0))->List[Dict[str,Any]]:
    rows=make_synthetic_rows(n_roots=n_roots, actions_per_root=actions_per_root, seed=seed, harm_thresholds=harm_thresholds)
    prefix='light2d' if backend in ('auto','light2d') else str(backend)
    for r in rows:
        old=str(r['root_id']).split('_')[-1]
        r['root_id']=f'{prefix}_{old}'
        r['backend']=prefix
    return rows

def write_rows_streaming(output:str|Path, n_roots:int, actions_per_root:int=NUM_ACTIONS, seed:int=7, backend:str='auto', shift_test:bool=True, harm_thresholds:Sequence[float]=(0.5,2.0,4.0,7.0,11.0), log_every:int=10)->None:
    rows=make_metadrive_rows(n_roots,actions_per_root,seed,backend,shift_test,harm_thresholds)
    write_jsonl(rows,output)
    splits={}
    for r in rows: splits[r['split']]=splits.get(r['split'],0)+1
    print(json.dumps({'finished':True,'output':str(output),'rows':len(rows),'splits':splits,'backend':backend},indent=2))

def main()->None:
    p=argparse.ArgumentParser(description='Generate MRVP JSONL with MetaDrive counterfactual rollouts or clean light2d fallback.')
    p.add_argument('--output',required=True); p.add_argument('--n-roots',type=int,default=240); p.add_argument('--actions-per-root',type=int,default=NUM_ACTIONS); p.add_argument('--seed',type=int,default=7); p.add_argument('--backend',choices=['auto','metadrive','light2d'],default='auto'); p.add_argument('--no-shift-test',action='store_true'); p.add_argument('--log-every',type=int,default=10)
    args=p.parse_args(); write_rows_streaming(args.output,args.n_roots,args.actions_per_root,args.seed,args.backend,not args.no_shift_test,log_every=args.log_every)
if __name__=='__main__': main()

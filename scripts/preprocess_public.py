#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json
from mrvp.public_pretraining.preprocessors import PREPROCESSORS
def main():
    p=argparse.ArgumentParser(description='Preprocess public driving datasets into MRVP pretraining JSONL.'); p.add_argument('--dataset',required=True,choices=sorted(PREPROCESSORS)); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--max-records',type=int,default=None); p.add_argument('--version',default='v1.0-mini'); args=p.parse_args(); fn=PREPROCESSORS[args.dataset]; n=fn(args.input,args.output,version=args.version,max_records=args.max_records) if args.dataset=='nuscenes' else fn(args.input,args.output,max_records=args.max_records); print(json.dumps({'dataset':args.dataset,'input':args.input,'output':args.output,'records':n},indent=2))
if __name__=='__main__': main()

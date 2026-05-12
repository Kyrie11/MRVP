#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json,argparse
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument('--rollout',required=True); p.add_argument('--output',required=True); args=p.parse_args(); out={'transition_type':'none','event_type':'none','event_time':0.0,'x_minus':[0]*12,'x_plus':[0]*12,'rho_imp':0.0,'harm_bin':0}; Path(args.output).write_text(json.dumps(out,indent=2)); print(json.dumps({'output':args.output,'transition_type':'none','harm_bin':0}))
if __name__=='__main__': main()

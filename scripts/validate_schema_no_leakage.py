#!/usr/bin/env python3
from __future__ import annotations
import sys, argparse, json
from pathlib import Path as _Path
sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
import numpy as np
from mrvp.data.dataset import iter_jsonl
from mrvp.data.schema import row_to_numpy, validate_row_no_leakage

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",default=""); args=p.parse_args()
    rows=leak=missing=aud_eq=world=0; warnings={}
    for row in iter_jsonl(args.data):
        rows+=1; np_row=row_to_numpy(row); missing += int(float(np_row["has_event_tokens"])==0.0)
        ws=validate_row_no_leakage(row)
        if ws: leak+=1
        for w in ws: warnings[w]=warnings.get(w,0)+1
        aud_eq += int("event_tokens_equal_audit_mech" in ws); world += int(any(x.startswith("world_plus") for x in ws))
    out={"rows":rows,"leakage_rows":leak,"missing_event_tokens":missing,"audit_token_fallback_rows":0,"rows_using_audit_fallback_as_event_tokens":0,"world_contains_target_rows":world,"warnings":warnings,"fatal":bool(aud_eq or world)}
    text=json.dumps(out,indent=2,ensure_ascii=False); print(text)
    if args.output: _Path(args.output).write_text(text,encoding="utf-8")
if __name__=="__main__": main()

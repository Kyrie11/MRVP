#!/usr/bin/env python3
from __future__ import annotations
import sys, argparse, json, csv
from pathlib import Path as _Path
sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from collections import Counter, defaultdict
import numpy as np
from mrvp.data.dataset import iter_jsonl
from mrvp.data.schema import validate_row_no_leakage, BOTTLE_NECKS

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--out",default="dataset_quality_report.json"); p.add_argument("--eps-s",type=float,default=0.25); p.add_argument("--expected-actions",type=int,default=8); args=p.parse_args()
    rows=list(iter_jsonl(args.data)); by=defaultdict(list)
    for r in rows: by[str(r.get("root_id"))].append(r)
    split_by_root=defaultdict(set); [split_by_root[str(r.get("root_id"))].add(str(r.get("split","train"))) for r in rows]
    leaks=sum(bool(validate_row_no_leakage(r)) for r in rows); world=sum(any(w.startswith("world_plus") for w in validate_row_no_leakage(r)) for r in rows); aud=0
    complete=sum(len(v)==args.expected_actions for v in by.values()); selectable=0; informative=0; root_summary=[]
    for rid,rs in by.items():
        min_bin=min(int(r.get("harm_bin",0)) for r in rs); adm=[r for r in rs if int(r.get("harm_bin",0))==min_bin]; selectable+=int(len(adm)>=2); spread=(max(float(r.get("s_star",0)) for r in adm)-min(float(r.get("s_star",0)) for r in adm)) if adm else 0; informative+=int(len(adm)>=2 and spread>=args.eps_s); root_summary.append({"root_id":rid,"split":"|".join(split_by_root[rid]),"actions":len(rs),"min_harm_bin":min_bin,"admissible":len(adm),"s_spread":spread})
    event=Counter(str(r.get("event_type","none")) for r in rows); split=Counter(str(r.get("split","train")) for r in rows); harm=Counter(int(r.get("harm_bin",0)) for r in rows); b=Counter(int(r.get("b_star",0)) for r in rows); backend=Counter(str(r.get("backend","unknown")) for r in rows); family=Counter(str(r.get("family","unknown")) for r in rows)
    recover=np.mean([float(r.get("s_star",0))>=0 for r in rows]) if rows else 0; event_majority=max(event.values())/max(len(rows),1) if event else 0
    out={"rows":len(rows),"roots":len(by),"split_distribution":dict(split),"root_split_leak_count":sum(len(s)>1 for s in split_by_root.values()),"action_count_complete_rate":complete/max(len(by),1),"missing_actions_roots":sum(len(v)!=args.expected_actions for v in by.values()),"harm_bin_distribution":dict(harm),"event_type_distribution":dict(event),"event_majority_rate":event_majority,"recoverable_row_rate":float(recover),"b_star_distribution":dict(b),"selectable_root_rate_adm_ge_2":selectable/max(len(by),1),"informative_root_rate_adm_pair_eps":informative/max(len(by),1),"world_target_leakage_rows":world,"audit_token_fallback_rows":aud,"leakage_rows":leaks,"backend_distribution":dict(backend),"family_distribution":dict(family)}
    path=_Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    with path.with_suffix(".root_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["root_id","split","actions","min_harm_bin","admissible","s_spread"]); w.writeheader(); w.writerows(root_summary)
    print(json.dumps({"output":str(path),"flags":{"root_split_leak_count":out["root_split_leak_count"],"world_target_leakage_rows":world,"audit_token_fallback_rows":aud}},indent=2,ensure_ascii=False))
if __name__=="__main__": main()

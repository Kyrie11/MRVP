#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from mrvp.data.dataset import iter_jsonl
from mrvp.data.schema import validate_row_no_leakage

def q(xs):
    xs=[float(x) for x in xs if math.isfinite(float(x))]
    if not xs: return {}
    a=np.asarray(xs); return {'min':float(a.min()),'p25':float(np.quantile(a,.25)),'mean':float(a.mean()),'median':float(np.median(a)),'p75':float(np.quantile(a,.75)),'max':float(a.max())}
def auc(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p).astype(float); pos=y==1; neg=y==0
    if pos.sum()==0 or neg.sum()==0: return float('nan')
    order=np.argsort(p); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(p)+1); return float((ranks[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
def main():
    p=argparse.ArgumentParser(description='MRVP MetaDrive/light2d dataset quality audit.'); p.add_argument('--data',required=True); p.add_argument('--out',default='dataset_quality_report.json'); p.add_argument('--eps-s',type=float,default=0.25); p.add_argument('--expected-actions',type=int,default=8); args=p.parse_args(); rows=list(iter_jsonl(args.data)); by=defaultdict(list)
    for i,r in enumerate(rows): by[str(r.get('root_id','missing'))].append((i,r))
    report={'basic':{'rows':len(rows),'roots':len(by),'split_rows':dict(Counter(str(r.get('split','train')) for r in rows)),'backend':dict(Counter(str(r.get('backend','unknown')) for r in rows)),'family':dict(Counter(str(r.get('family','unknown')) for r in rows)),'event_type':dict(Counter(str(r.get('event_type','none')) for r in rows)),'harm_bin':dict(Counter(str(r.get('harm_bin',0)) for r in rows)),'b_star':dict(Counter(str(r.get('b_star','?')) for r in rows))}}
    split_leaks=[]; incomplete=[]; adm_counts=[]; spreads=[]; pair_counts=[]; selectable=informative=0
    for root,items in by.items():
        rs=[r for _,r in items]; splits={str(r.get('split','train')) for r in rs};
        if len(splits)>1: split_leaks.append(root)
        if len(rs)!=args.expected_actions: incomplete.append({'root_id':root,'rows':len(rs)})
        minb=min(int(r.get('harm_bin',0)) for r in rs); adm=[r for r in rs if int(r.get('harm_bin',0))==minb]; adm_counts.append(len(adm)); selectable+=int(len(adm)>=2); ss=[float(r.get('s_star',0)) for r in adm]; spread=max(ss)-min(ss) if len(ss)>=2 else 0.0; spreads.append(spread); pc=sum(1 for i in range(len(ss)) for j in range(i+1,len(ss)) if abs(ss[i]-ss[j])>=args.eps_s); pair_counts.append(pc); informative+=int(pc>0)
    rec=[float(r.get('s_star',0))>=0 for r in rows]; leak_warn=[]
    for i,r in enumerate(rows):
        w=validate_row_no_leakage(r)
        if w and len(leak_warn)<20: leak_warn.append({'row_index':i,'root_id':r.get('root_id'),'action_id':r.get('action_id'),'warnings':w})
    train=[r for r in rows if str(r.get('split','train'))=='train']; test=[r for r in rows if str(r.get('split','train'))=='test'] or rows; gp=np.mean([float(r.get('s_star',0))>=0 for r in train]) if train else np.mean(rec); p_by=defaultdict(lambda:gp)
    for aid in set(int(r.get('action_id',-1)) for r in rows):
        ys=[float(r.get('s_star',0))>=0 for r in train if int(r.get('action_id',-1))==aid]
        if ys: p_by[aid]=float(np.mean(ys))
    report['root_counterfactual_quality']={'root_split_leak_count':len(split_leaks),'root_split_leak_examples':split_leaks[:10],'incomplete_root_count':len(incomplete),'incomplete_examples':incomplete[:10],'admissible_action_count':q(adm_counts),'selectable_root_rate_adm_ge_2':selectable/max(1,len(by)),'informative_root_rate_adm_pair_eps':informative/max(1,len(by)),'admissible_s_star_spread':q(spreads),'same_harm_pair_count_per_root':q(pair_counts)}
    report['label_quality']={'recoverable_row_rate':float(np.mean(rec)) if rec else float('nan'),'s_star':q([r.get('s_star',0) for r in rows])}
    report['action_shortcut_probe_simple']={'test_auc_action_only':auc([float(r.get('s_star',0))>=0 for r in test],[p_by[int(r.get('action_id',-1))] for r in test]),'train_action_recoverable_rate':{str(k):float(v) for k,v in sorted(p_by.items())}}
    report['leakage_and_schema_warnings']={'rows_missing_event_tokens':sum(1 for r in rows if 'event_tokens' not in r),'audit_token_fallback_rows':0,'warning_examples':leak_warn}
    flags=[]
    if report['root_counterfactual_quality']['root_split_leak_count']>0: flags.append('FAIL: same root appears in multiple splits.')
    if report['root_counterfactual_quality']['selectable_root_rate_adm_ge_2']<0.5: flags.append('WARN: too few roots have >=2 admissible actions.')
    if leak_warn: flags.append('WARN: leakage/schema warnings found; inspect examples.')
    report['summary_flags']=flags; out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    csv_path=out.with_suffix('.root_summary.csv')
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f); wr.writerow(['root_id','split','n_actions','min_harm_bin','adm_count','adm_s_spread','adm_pair_count_eps'])
        for root,items in by.items():
            rs=[r for _,r in items]; minb=min(int(r.get('harm_bin',0)) for r in rs); adm=[r for r in rs if int(r.get('harm_bin',0))==minb]; ss=[float(r.get('s_star',0)) for r in adm]; pc=sum(1 for i in range(len(ss)) for j in range(i+1,len(ss)) if abs(ss[i]-ss[j])>=args.eps_s); wr.writerow([root,rs[0].get('split','train'),len(rs),minb,len(adm),max(ss)-min(ss) if len(ss)>=2 else 0.0,pc])
    print(json.dumps({'report':str(out),'root_csv':str(csv_path),'flags':flags},indent=2,ensure_ascii=False))
if __name__=='__main__': main()

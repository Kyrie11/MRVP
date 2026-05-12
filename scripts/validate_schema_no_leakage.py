#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse,json
from pathlib import Path
from mrvp.data.dataset import iter_jsonl
from mrvp.data.schema import validate_row_no_leakage

def main():
    p=argparse.ArgumentParser(description='Validate MRVP JSONL schema for token/world leakage.'); p.add_argument('--data',required=True); p.add_argument('--output',default=''); args=p.parse_args(); rows=0; leakage=0; missing=0; audit=0; world=0; examples=[]
    for i,row in enumerate(iter_jsonl(args.data)):
        rows+=1; missing+=int('event_tokens' not in row); warns=validate_row_no_leakage(row); audit+=int('event_tokens_match_audit_mech' in warns); world+=int(any(w.startswith('world_plus_mentions_') for w in warns)); leakage+=int(bool(warns))
        if warns and len(examples)<20: examples.append({'row_index':i,'root_id':row.get('root_id'),'action_id':row.get('action_id'),'warnings':warns})
    out={'rows':rows,'leakage_rows':leakage,'missing_event_tokens':missing,'audit_token_fallback_rows':0,'event_tokens_match_audit_rows':audit,'world_contains_target_rows':world,'fatal':audit>0 or world>0,'examples':examples}
    text=json.dumps(out,indent=2,ensure_ascii=False); print(text)
    if args.output: Path(args.output).write_text(text,encoding='utf-8')
if __name__=='__main__': main()

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from torch.utils.data import Dataset
from .dataset import MRVPDataset

class SameRootPairDataset(Dataset):
    def __init__(self, base: MRVPDataset, eps_s: float=0.25, max_pairs_per_root: int=64)->None:
        self.base=base; self.pairs: List[Tuple[int,int]]=[]; gaps=[]; roots=set(); event_pairs={}
        for root, indices in base.root_to_indices.items():
            group=[]
            for i in indices:
                r=base.rows[i]
                if 'is_harm_admissible' in r and float(r['is_harm_admissible'])<0.5: continue
                if int(r['harm_bin'])!=int(r.get('root_min_harm_bin',r['harm_bin'])): continue
                group.append(i)
            if len(group)<2: continue
            candidates=[]
            for p,i in enumerate(group):
                for j in group[p+1:]:
                    gap=abs(float(base.rows[i]['s_star'])-float(base.rows[j]['s_star']))
                    if gap>=eps_s:
                        candidates.append((i,j)); gaps.append(gap); roots.add(root)
                        key=f"{base.rows[i].get('event_type','?')}|{base.rows[j].get('event_type','?')}"; event_pairs[key]=event_pairs.get(key,0)+1
            if len(candidates)>max_pairs_per_root:
                step=max(1,len(candidates)//max_pairs_per_root); candidates=candidates[::step][:max_pairs_per_root]
            self.pairs.extend(candidates)
        self.diagnostics={'num_pairs':len(self.pairs),'num_roots_with_pairs':len(roots),'mean_s_gap':float(np.mean(gaps)) if gaps else 0.0,'median_s_gap':float(np.median(gaps)) if gaps else 0.0,'event_pair_distribution':event_pairs}
        if not self.pairs: raise ValueError('No admissible same-root same-harm informative pairs found. Lower eps_s or generate more roots.')
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx:int):
        i,j=self.pairs[idx]; return self.base[i], self.base[j]

def pair_collate(batch):
    from .dataset import mrvp_collate
    return {'i':mrvp_collate([b[0] for b in batch]), 'j':mrvp_collate([b[1] for b in batch])}

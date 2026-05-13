import numpy as np
from mrvp.selection import select_from_scores

def test_harm_gate_blocks_higher_harm_better_score():
    rows=[{'action_id':0,'harm_bin':0},{'action_id':1,'harm_bin':1}]
    lower=np.asarray([[-10,-10,-10,-10,-10],[100,100,100,100,100]],dtype=np.float32)
    sel=select_from_scores(rows,lower)
    assert sel['selected_action_id']==0

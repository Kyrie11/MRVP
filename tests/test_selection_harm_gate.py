import numpy as np
from mrvp.selection import select_from_scores

def test_selection_harm_gate():
    rows=[{'action_id':0,'harm_bin':0},{'action_id':1,'harm_bin':1}]
    lower=np.asarray([[-1,-1,-1,-1,-1],[100,100,100,100,100]],dtype=float)
    sel=select_from_scores(rows,lower)
    assert sel['selected_action_id']==0

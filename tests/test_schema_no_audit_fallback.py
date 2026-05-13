import numpy as np
from mrvp.data.schema import row_to_numpy, MECH_DIM

def test_missing_event_tokens_are_zero_not_audit():
    row={'root_id':'r','audit_mech':[1.0]*MECH_DIM,'z_mech':[1.0]*MECH_DIM,'m_star':[0,0,0,0,0],'x_plus':[0]*12,'x_minus':[0]*12,'deg':[1,1,1,0,1,0]}
    out=row_to_numpy(row)
    assert float(out['has_event_tokens'])==0.0
    assert np.all(out['event_tokens']==0.0)
    assert out['z_mech'].sum()>0

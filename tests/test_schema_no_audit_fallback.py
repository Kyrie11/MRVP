import numpy as np
from mrvp.data.schema import row_to_numpy, SchemaDims

def test_schema_no_audit_fallback():
    row={'root_id':'r','action_id':0,'audit_mech':[1.0]*32,'z_mech':[1.0]*32,'x_minus':[0]*12,'x_plus':[0]*12,'m_star':[0]*5}
    out=row_to_numpy(row,SchemaDims())
    assert np.all(out['event_tokens']==0)
    assert float(out['has_event_tokens'])==0.0

from mrvp.data.metadrive_rollout import _make_root_scene, _rollout_light2d, _event_type_from_transition
from mrvp.data.schema import ACTION_VECTORS
import numpy as np

def test_event_type_no_margin_arg():
    rng=np.random.default_rng(1); scene=_make_root_scene(0,4,rng,7); res=_rollout_light2d(scene,ACTION_VECTORS[1]); a=_event_type_from_transition(scene,res); b=_event_type_from_transition(scene,res)
    assert a==b

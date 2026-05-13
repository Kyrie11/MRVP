from mrvp.data.metadrive_rollout import _make_root_scene, _rollout_light2d, _world_plus_from_post_event
from mrvp.data.schema import ACTION_VECTORS
import numpy as np

def test_world_plus_independent_of_targets():
    rng=np.random.default_rng(0); scene=_make_root_scene(0,4,rng,7); res=_rollout_light2d(scene,ACTION_VECTORS[0]); w1=_world_plus_from_post_event(scene,res); w2=_world_plus_from_post_event(scene,res)
    assert w1==w2
    assert 'r_star' not in repr(w1) and 'm_star' not in repr(w1) and 'recovery_traj' not in repr(w1)

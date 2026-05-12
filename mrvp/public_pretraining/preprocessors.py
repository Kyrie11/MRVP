from __future__ import annotations
from pathlib import Path
from .common import write_pretrain_jsonl

def _not_implemented(input_dir, output, *args, **kwargs):
    raise RuntimeError('Public dataset preprocessors are lightweight placeholders in this package. Implement dataset-specific parser or use existing preprocessed JSONL.')
PREPROCESSORS={'highd':_not_implemented,'interaction':_not_implemented,'argoverse2':_not_implemented,'waymo':_not_implemented,'commonroad':_not_implemented,'nuscenes':_not_implemented,'nuplan':_not_implemented}

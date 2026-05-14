from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .geometry import iou_binary


def mean_abs(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    return float(np.mean(np.abs(arr))) if arr.size else 0.0


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    return float(np.mean(arr)) if arr.size else 0.0


def lower_tail_cvar(values: Iterable[float], beta: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    k = max(1, int(math.ceil(float(beta) * len(vals))))
    return float(np.mean(vals[:k]))


def binary_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    return iou_binary(pred, target, threshold)


def calibration_error(pred: np.ndarray, truth: np.ndarray, bins: int = 10) -> float:
    p = np.asarray(pred, dtype=np.float32).reshape(-1)
    y = (np.asarray(truth, dtype=np.float32).reshape(-1) > 0).astype(np.float32)
    conf = 1.0 / (1.0 + np.exp(-p))
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(conf)
    if total == 0:
        return 0.0
    acc = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            acc += float(mask.mean()) * abs(float(y[mask].mean()) - float(conf[mask].mean()))
    return acc

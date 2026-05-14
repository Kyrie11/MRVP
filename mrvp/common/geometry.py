from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi


def rotation_matrix(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_points(points: np.ndarray, origin_xy: Iterable[float], yaw: float, inverse: bool = False) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    origin = np.asarray(list(origin_xy), dtype=np.float32)
    rot = rotation_matrix(yaw)
    if inverse:
        return (pts - origin) @ rot
    return pts @ rot.T + origin


def ego_box_corners(x: np.ndarray, length: float = 4.7, width: float = 2.0) -> np.ndarray:
    px, py, yaw = float(x[0]), float(x[1]), float(x[2])
    local = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float32,
    )
    return transform_points(local, (px, py), yaw, inverse=False)


def point_to_aabb_signed_distance(point: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    h = np.asarray(half_size, dtype=np.float32)
    q = np.abs(p) - h
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(float(q[0]), float(q[1])), 0.0)
    return float(outside + inside)


def min_distance_to_points(point: np.ndarray, cloud: np.ndarray) -> float:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.size == 0:
        return 1e3
    return float(np.linalg.norm(arr - np.asarray(point, dtype=np.float32)[None, :], axis=1).min())


def rasterize_disc(size: int, center_xy: tuple[float, float], radius: float, value: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = center_xy
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    out = np.zeros((size, size), dtype=np.float32)
    out[mask] = value
    return out


def iou_binary(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    p = np.asarray(pred) > threshold
    t = np.asarray(target) > threshold
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)

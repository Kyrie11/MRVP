from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _stack(rows: list[dict[str, Any]], key: str, dtype=torch.float32) -> torch.Tensor:
    arr = np.stack([np.asarray(r[key]) for r in rows], axis=0)
    return torch.as_tensor(arr, dtype=dtype)


def collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "root_id": [str(r["root_id"]) for r in rows],
        "action_id": [str(r["action_id"]) for r in rows],
        "scenario_family": [str(r.get("scenario_family", "unknown")) for r in rows],
        "o_hist": _stack(rows, "o_hist"),
        "x_t": _stack(rows, "x_t"),
        "prefix_rollout": _stack(rows, "prefix_rollout"),
        "prefix_controls": _stack(rows, "prefix_controls"),
        "tau_index": torch.as_tensor([int(r.get("tau_index", 0)) for r in rows], dtype=torch.long),
        "tau_reset": torch.as_tensor([float(r.get("tau_reset", 0.0)) for r in rows], dtype=torch.float32),
        "r_reset": _stack(rows, "r_reset"),
        "deg": _stack(rows, "deg"),
        "teacher_u": _stack(rows, "teacher_u"),
        "teacher_traj": _stack(rows, "teacher_traj"),
        "cert_star": _stack(rows, "cert_star"),
        "score_star": torch.as_tensor([float(r["score_star"]) for r in rows], dtype=torch.float32),
        "rho_imp": torch.as_tensor([float(r["rho_imp"]) for r in rows], dtype=torch.float32),
        "harm_bin": torch.as_tensor([int(r["harm_bin"]) for r in rows], dtype=torch.long),
    }
    world = [r["world_reset"] for r in rows]
    batch["world_A"] = torch.as_tensor(np.stack([np.asarray(w["A"]) for w in world], axis=0), dtype=torch.float32)
    batch["world_O"] = torch.as_tensor(np.stack([np.asarray(w["O"]) for w in world], axis=0), dtype=torch.float32)
    batch["world_G"] = torch.as_tensor(np.stack([np.asarray(w["G"]) for w in world], axis=0), dtype=torch.float32)
    batch["world_Y"] = torch.as_tensor(np.stack([np.asarray(w["Y"]) for w in world], axis=0), dtype=torch.float32)
    return batch


def collate_roots(roots: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return roots

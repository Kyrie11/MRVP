from __future__ import annotations

import glob
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import basic_context, make_o_hist, make_record, write_pretrain_jsonl
from mrvp.data.schema import HIST_LEN


def _yaw_from_vel(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.arctan2(vy, vx + 1e-6)


def _agent_arr(df: pd.DataFrame, x="x", y="y", vx="vx", vy="vy", yaw: str | None = None, length="length", width="width") -> np.ndarray:
    arr = np.zeros((len(df), 9), dtype=np.float32)
    arr[:, 0] = df[x].to_numpy(np.float32)
    arr[:, 1] = df[y].to_numpy(np.float32)
    if yaw and yaw in df.columns:
        arr[:, 2] = df[yaw].to_numpy(np.float32)
    else:
        arr[:, 2] = _yaw_from_vel(df[vx].to_numpy(np.float32), df[vy].to_numpy(np.float32))
    arr[:, 3] = df[vx].to_numpy(np.float32)
    arr[:, 4] = df[vy].to_numpy(np.float32)
    arr[:, 5] = df[length].to_numpy(np.float32) if length in df.columns else 4.6
    arr[:, 6] = df[width].to_numpy(np.float32) if width in df.columns else 1.9
    arr[:, 7] = 1.0
    arr[:, 8] = 1.0
    return arr


def preprocess_highd(input_dir: str | Path, output: str | Path, stride: int = 25, history: int = 25, future: int = 25, max_records: int | None = None) -> int:
    """highD: reads XX_tracks.csv plus metadata if available.

    Expected columns include frame,id,x,y,xVelocity,yVelocity,width,height.
    """
    input_dir = Path(input_dir)
    records = []
    dataset_id = 1.0
    for tracks_file in sorted(input_dir.glob("*_tracks.csv")):
        prefix = tracks_file.name.replace("_tracks.csv", "")
        rec_meta = input_dir / f"{prefix}_recordingMeta.csv"
        lane_width = 3.5
        speed_limit = 33.0
        if rec_meta.exists():
            meta = pd.read_csv(rec_meta)
            if "speedLimit" in meta.columns:
                speed_limit = float(meta["speedLimit"].iloc[0])
            # highD lane markings are y positions separated by ';'.
            for col in ["upperLaneMarkings", "lowerLaneMarkings"]:
                if col in meta.columns and isinstance(meta[col].iloc[0], str):
                    vals = [float(x) for x in meta[col].iloc[0].split(";") if x]
                    if len(vals) >= 2:
                        lane_width = float(np.median(np.diff(vals)))
        df = pd.read_csv(tracks_file)
        rename = {"xVelocity": "vx", "yVelocity": "vy", "height": "length"}
        df = df.rename(columns=rename)
        if "width" not in df.columns:
            df["width"] = 1.9
        if "length" not in df.columns:
            df["length"] = 4.6
        for ego_id, g in df.groupby("id"):
            g = g.sort_values("frame")
            if len(g) < history + future + 1:
                continue
            for start in range(0, len(g) - history - future, stride):
                hist_g = g.iloc[start : start + history]
                future_g = g.iloc[start + history : start + history + future]
                frame0 = int(hist_g["frame"].iloc[-1])
                agents = [_agent_arr(hist_g, x="x", y="y", vx="vx", vy="vy", length="length", width="width")]
                same_frame_ids = df[(df["frame"] >= hist_g["frame"].iloc[0]) & (df["frame"] <= hist_g["frame"].iloc[-1]) & (df["id"] != ego_id)]
                for _, ag in same_frame_ids.groupby("id"):
                    ag = ag.sort_values("frame").tail(history)
                    if len(ag) >= 3:
                        agents.append(_agent_arr(ag, x="x", y="y", vx="vx", vy="vy", length="length", width="width"))
                    if len(agents) >= 16:
                        break
                o_hist = make_o_hist(agents, hist_len=HIST_LEN)
                h_ctx = basic_context(o_hist, lane_width=lane_width, speed_limit=speed_limit, dataset_id=dataset_id)
                records.append(make_record("highD", f"{prefix}_{int(ego_id)}_{frame0}", o_hist, future_g[["x", "y"]].to_numpy(), h_ctx))
                if max_records and len(records) >= max_records:
                    return write_pretrain_jsonl(records, output)
    return write_pretrain_jsonl(records, output)


def preprocess_interaction(input_dir: str | Path, output: str | Path, stride: int = 10, history: int = 10, future: int = 30, max_records: int | None = None) -> int:
    """INTERACTION: reads vehicle_tracks_*.csv files.

    Expected columns: case_id,track_id,frame_id,timestamp_ms,agent_type,x,y,vx,vy,psi_rad,length,width.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("vehicle_tracks_*.csv")) + sorted(input_dir.rglob("*tracks*.csv"))
    records = []
    for file in files:
        df = pd.read_csv(file)
        if "psi_rad" in df.columns:
            yaw_col = "psi_rad"
        elif "heading" in df.columns:
            yaw_col = "heading"
        else:
            yaw_col = None
        if "case_id" not in df.columns:
            df["case_id"] = file.stem
        for case_id, case in df.groupby("case_id"):
            frame_col = "frame_id" if "frame_id" in case.columns else "timestamp_ms"
            for ego_id, g in case.groupby("track_id"):
                g = g.sort_values(frame_col)
                if len(g) < history + future + 1:
                    continue
                for start in range(0, len(g) - history - future, stride):
                    hist_g = g.iloc[start : start + history]
                    fut = g.iloc[start + history : start + history + future]
                    agents = [_agent_arr(hist_g, yaw=yaw_col)]
                    fmin, fmax = hist_g[frame_col].min(), hist_g[frame_col].max()
                    others = case[(case[frame_col] >= fmin) & (case[frame_col] <= fmax) & (case["track_id"] != ego_id)]
                    for _, ag in others.groupby("track_id"):
                        ag = ag.sort_values(frame_col).tail(history)
                        if len(ag) >= 3:
                            agents.append(_agent_arr(ag, yaw=yaw_col))
                        if len(agents) >= 16:
                            break
                    o_hist = make_o_hist(agents)
                    h_ctx = basic_context(o_hist, dataset_id=2.0)
                    records.append(make_record("INTERACTION", f"{case_id}_{ego_id}_{int(hist_g[frame_col].iloc[-1])}", o_hist, fut[["x", "y"]].to_numpy(), h_ctx))
                    if max_records and len(records) >= max_records:
                        return write_pretrain_jsonl(records, output)
    return write_pretrain_jsonl(records, output)


def preprocess_argoverse2(input_dir: str | Path, output: str | Path, max_records: int | None = None) -> int:
    """Argoverse 2 motion forecasting: reads scenario parquet files.

    Works either with raw pandas parquet columns or AV2 SDK-produced parquet.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.rglob("*.parquet"))
    records = []
    for file in files:
        df = pd.read_parquet(file)
        # Common AV2 columns: track_id,timestep,position_x,position_y,velocity_x,velocity_y,heading,object_type.
        cols = set(df.columns)
        x = "position_x" if "position_x" in cols else "x"
        y = "position_y" if "position_y" in cols else "y"
        vx = "velocity_x" if "velocity_x" in cols else ("vx" if "vx" in cols else None)
        vy = "velocity_y" if "velocity_y" in cols else ("vy" if "vy" in cols else None)
        if vx is None or vy is None or x not in cols or y not in cols:
            continue
        time_col = "timestep" if "timestep" in cols else ("timestamp_ns" if "timestamp_ns" in cols else df.columns[0])
        track_col = "track_id" if "track_id" in cols else "id"
        scenario_id = str(df["scenario_id"].iloc[0]) if "scenario_id" in cols else file.parent.name
        focal = str(df["focal_track_id"].iloc[0]) if "focal_track_id" in cols else str(df[track_col].iloc[0])
        # Prefer focal/AV track as ego-like context.
        for ego_id in [focal]:
            g = df[df[track_col].astype(str) == str(ego_id)].sort_values(time_col)
            if len(g) < 30:
                continue
            hist_g = g.iloc[:10]
            fut = g.iloc[10:30]
            tmp = hist_g.rename(columns={x: "x", y: "y", vx: "vx", vy: "vy", "heading": "heading"})
            agents = [_agent_arr(tmp, yaw="heading" if "heading" in tmp.columns else None)]
            last_t = hist_g[time_col].iloc[-1]
            others = df[df[time_col].isin(hist_g[time_col].values) & (df[track_col].astype(str) != str(ego_id))]
            for _, ag in others.groupby(track_col):
                ag = ag.sort_values(time_col).tail(10).rename(columns={x: "x", y: "y", vx: "vx", vy: "vy", "heading": "heading"})
                if len(ag) >= 3:
                    agents.append(_agent_arr(ag, yaw="heading" if "heading" in ag.columns else None))
                if len(agents) >= 16:
                    break
            o_hist = make_o_hist(agents)
            h_ctx = basic_context(o_hist, dataset_id=3.0)
            records.append(make_record("Argoverse2", f"{scenario_id}_{ego_id}", o_hist, fut[[x, y]].to_numpy(), h_ctx))
            if max_records and len(records) >= max_records:
                return write_pretrain_jsonl(records, output)
    return write_pretrain_jsonl(records, output)


def preprocess_waymo(input_dir: str | Path, output: str | Path, max_records: int | None = None) -> int:
    """Waymo Open Motion: reads Scenario TFRecords via official proto package."""
    try:
        import tensorflow as tf  # type: ignore
        from waymo_open_dataset.protos import scenario_pb2  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install tensorflow and waymo-open-dataset to preprocess Waymo Open Motion TFRecords.") from exc
    records = []
    files = sorted(Path(input_dir).rglob("*.tfrecord*"))
    for file in files:
        for raw in tf.data.TFRecordDataset(str(file)):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(raw.numpy()))
            cur = int(scenario.current_time_index)
            hist_idx = list(range(max(0, cur - HIST_LEN + 1), cur + 1))
            agents = []
            for tr in scenario.tracks:
                arr = []
                for k in hist_idx:
                    st = tr.states[k]
                    valid = bool(st.valid)
                    arr.append([st.center_x, st.center_y, st.heading, st.velocity_x, st.velocity_y, st.length, st.width, float(tr.object_type), float(valid)])
                agents.append(np.asarray(arr, dtype=np.float32))
            if not agents:
                continue
            ego_idx = int(scenario.sdc_track_index)
            o_hist = make_o_hist(agents, ego_index=min(ego_idx, len(agents) - 1))
            fut = []
            ego_track = scenario.tracks[ego_idx]
            for k in range(cur + 1, min(len(ego_track.states), cur + 31)):
                st = ego_track.states[k]
                if st.valid:
                    fut.append([st.center_x, st.center_y])
            h_ctx = basic_context(o_hist, dataset_id=4.0)
            records.append(make_record("WaymoOpenMotion", scenario.scenario_id, o_hist, np.asarray(fut) if fut else None, h_ctx))
            if max_records and len(records) >= max_records:
                return write_pretrain_jsonl(records, output)
    return write_pretrain_jsonl(records, output)


def preprocess_commonroad(input_dir: str | Path, output: str | Path, max_records: int | None = None) -> int:
    try:
        from commonroad.common.file_reader import CommonRoadFileReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install commonroad-io to preprocess CommonRoad XML scenarios.") from exc
    records = []
    for file in sorted(Path(input_dir).rglob("*.xml")):
        scenario, planning_problem_set = CommonRoadFileReader(str(file)).open()
        obstacles = list(scenario.dynamic_obstacles)
        for obs in obstacles:
            states = obs.prediction.trajectory.state_list if obs.prediction is not None else []
            if len(states) < 20:
                continue
            arr = []
            for st in states[:HIST_LEN]:
                v = float(getattr(st, "velocity", 0.0))
                yaw = float(getattr(st, "orientation", 0.0))
                arr.append([st.position[0], st.position[1], yaw, v * math.cos(yaw), v * math.sin(yaw), 4.6, 1.9, 1.0, 1.0])
            agents = [np.asarray(arr, dtype=np.float32)]
            for other in obstacles[:15]:
                if other.obstacle_id == obs.obstacle_id or other.prediction is None:
                    continue
                arr2 = []
                for st in other.prediction.trajectory.state_list[:HIST_LEN]:
                    v = float(getattr(st, "velocity", 0.0))
                    yaw = float(getattr(st, "orientation", 0.0))
                    arr2.append([st.position[0], st.position[1], yaw, v * math.cos(yaw), v * math.sin(yaw), 4.6, 1.9, 1.0, 1.0])
                if len(arr2) >= 3:
                    agents.append(np.asarray(arr2, dtype=np.float32))
            o_hist = make_o_hist(agents)
            future_xy = np.asarray([[s.position[0], s.position[1]] for s in states[HIST_LEN : HIST_LEN + 30]], dtype=np.float32)
            h_ctx = basic_context(o_hist, dataset_id=5.0)
            records.append(make_record("CommonRoad", f"{scenario.scenario_id}_{obs.obstacle_id}", o_hist, future_xy, h_ctx))
            if max_records and len(records) >= max_records:
                return write_pretrain_jsonl(records, output)
    return write_pretrain_jsonl(records, output)


def preprocess_nuscenes(input_dir: str | Path, output: str | Path, version: str = "v1.0-mini", max_records: int | None = None) -> int:
    try:
        from nuscenes.nuscenes import NuScenes  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install nuscenes-devkit to preprocess nuScenes.") from exc
    nusc = NuScenes(version=version, dataroot=str(input_dir), verbose=True)
    records = []
    for scene in nusc.scene:
        sample_token = scene["first_sample_token"]
        xy_seq = []
        count = 0
        while sample_token and count < HIST_LEN + 30:
            sample = nusc.get("sample", sample_token)
            ego_pose = nusc.get("ego_pose", nusc.get("sample_data", sample["data"]["LIDAR_TOP"])["ego_pose_token"])
            xy_seq.append([ego_pose["translation"][0], ego_pose["translation"][1], ego_pose["rotation"]])
            sample_token = sample["next"]
            count += 1
        if len(xy_seq) < HIST_LEN + 1:
            continue
        xy = np.asarray([[p[0], p[1]] for p in xy_seq], dtype=np.float32)
        v = np.diff(xy, axis=0, prepend=xy[:1]) / 0.5
        yaw = _yaw_from_vel(v[:, 0], v[:, 1])
        ego_arr = np.stack([xy[:, 0], xy[:, 1], yaw, v[:, 0], v[:, 1], np.full(len(xy), 4.6), np.full(len(xy), 1.9), np.ones(len(xy)), np.ones(len(xy))], axis=1)
        o_hist = make_o_hist([ego_arr[:HIST_LEN]])
        h_ctx = basic_context(o_hist, dataset_id=6.0)
        records.append(make_record("nuScenes", scene["token"], o_hist, xy[HIST_LEN:], h_ctx))
        if max_records and len(records) >= max_records:
            return write_pretrain_jsonl(records, output)
    return write_pretrain_jsonl(records, output)


def preprocess_nuplan(input_dir: str | Path, output: str | Path, max_records: int | None = None) -> int:
    """nuPlan lightweight SQLite fallback.

    The preferred path is the official nuplan-devkit. This fallback extracts ego
    poses from .db files when ego_pose-like tables are present, sufficient for
    context encoder warm-up but not for official nuPlan metrics.
    """
    records = []
    for db_file in sorted(Path(input_dir).rglob("*.db")):
        con = sqlite3.connect(str(db_file))
        try:
            tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'").fetchall()}
            if "ego_pose" not in tables:
                continue
            df = pd.read_sql_query("select * from ego_pose", con)
            # Try common columns.
            xcol = "x" if "x" in df.columns else "trans_x"
            ycol = "y" if "y" in df.columns else "trans_y"
            if xcol not in df.columns or ycol not in df.columns:
                continue
            df = df.sort_values(df.columns[0])
            xy = df[[xcol, ycol]].to_numpy(np.float32)
            if len(xy) < HIST_LEN + 30:
                continue
            for start in range(0, len(xy) - HIST_LEN - 30, 20):
                seg = xy[start : start + HIST_LEN + 30]
                v = np.diff(seg, axis=0, prepend=seg[:1]) / 0.1
                yaw = _yaw_from_vel(v[:, 0], v[:, 1])
                ego_arr = np.stack([seg[:, 0], seg[:, 1], yaw, v[:, 0], v[:, 1], np.full(len(seg), 4.6), np.full(len(seg), 1.9), np.ones(len(seg)), np.ones(len(seg))], axis=1)
                o_hist = make_o_hist([ego_arr[:HIST_LEN]])
                h_ctx = basic_context(o_hist, dataset_id=7.0)
                records.append(make_record("nuPlan", f"{db_file.stem}_{start}", o_hist, seg[HIST_LEN:], h_ctx))
                if max_records and len(records) >= max_records:
                    return write_pretrain_jsonl(records, output)
        finally:
            con.close()
    return write_pretrain_jsonl(records, output)


PREPROCESSORS = {
    "highd": preprocess_highd,
    "interaction": preprocess_interaction,
    "argoverse2": preprocess_argoverse2,
    "waymo": preprocess_waymo,
    "commonroad": preprocess_commonroad,
    "nuscenes": preprocess_nuscenes,
    "nuplan": preprocess_nuplan,
}

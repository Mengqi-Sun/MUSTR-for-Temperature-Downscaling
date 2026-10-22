"""Shared station/window enumeration; observation split semantics are preserved."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import cfg_get as _cfg_get


def _resolve_station_id_col(df: pd.DataFrame) -> str:
    for col in ["Station_Id_C", "Station_ID", "station_id", "station"]:
        if col in df.columns:
            return col
    raise ValueError("station_csv 缺少站点ID列，支持: Station_Id_C/Station_ID/station_id/station")


def _resolve_split_col(df: pd.DataFrame) -> str:
    for col in ["Split", "split", "Set", "set"]:
        if col in df.columns:
            return col
    return ""


def build_station_df_from_config(config: Any, split: str) -> pd.DataFrame:
    station_csv = _cfg_get(config, "data.station_csv")
    station_df = pd.read_csv(station_csv)
    station_id_col = _resolve_station_id_col(station_df)
    split_col = _resolve_split_col(station_df)
    virtual_file_prefix = str(_cfg_get(config, "data.virtual_file_prefix", "VC"))
    station_df[station_id_col] = station_df[station_id_col].astype(str)
    if split_col and split in ["train", "val"]:
        station_df = station_df[station_df[split_col].astype(str).str.lower() == split].copy()
    station_df["is_real"] = ~station_df[station_id_col].str.startswith(virtual_file_prefix)
    return station_df


def split_spec(config):
    return {key: _cfg_get(config, f"data.{key}", default) for key, default in {
        "train_years": [2018, 2019, 2020, 2021, 2022], "test_year": 2023,
        "val_months": {}, "val_use_full_year": False}.items()}


def eligible_days(config, split, day_index):
    spec = split_spec(config)
    months = {int(y): set(ms) for y, ms in spec["val_months"].items()}
    result = []
    for day in day_index:
        date = datetime(2018, 1, 1) + timedelta(days=int(day))
        in_val = date.month in months.get(date.year, set())
        if split == "train":
            ok = date.year in spec["train_years"] and not in_val
        elif split == "val":
            ok = date.year in spec["train_years"] and (spec["val_use_full_year"] or in_val)
        elif split == "test":
            ok = date.year == spec["test_year"]
        else:
            raise ValueError(f"Unknown physics split: {split}")
        result.append(ok)
    return np.asarray(result, dtype=bool)


def window_days(config):
    value = _cfg_get(config, "data.window_days", 7)
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("data.window_days must be a positive integer")
    from ..config import uses_physics
    if uses_physics(config) and value < 2:
        raise ValueError("Daily storage physics requires window_days >= 2")
    if _cfg_get(config, "data.window_hours") is not None:
        raise ValueError("Retired data.window_hours: use data.window_days")
    return int(value)


def build_index_map_from_config(config, split, *, station_ids=None, max_windows=0, max_days=0):
    stations = build_station_df_from_config(config, split)
    ids = stations[_resolve_station_id_col(stations)].astype(str).tolist()
    if station_ids is not None:
        selected = set(map(str, station_ids))
        ids = [sid for sid in ids if sid in selected]
    length = window_days(config)
    stride_key = "data.stride_days" if split == "train" else "data.val_stride_days"
    stride = _cfg_get(config, stride_key, _cfg_get(config, "data.stride_days", 1))
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError(f"{stride_key} must be a positive integer")
    indexes = []
    offsets = {}  # Reuse day-offset integers across stations.
    # Local import avoids a cycle with repacked.py.
    from .repacked import resolve_layout, station_days
    from .scan_cache import directory_signature, load_index_map, load_or_build, save_index_map
    h5_dir = Path(_cfg_get(config, "data.h5_dir"))
    layout = resolve_layout(config, h5_dir, ids)

    def build():
        for sid in ids:
            days = station_days(h5_dir/f"{sid}.h5", layout)
            if not np.all(np.diff(days) == 1):
                raise ValueError(f"{sid}: nonconsecutive days")
            configured = _cfg_get(config, "data.total_daily_steps")
            if configured is not None and int(configured) != len(days):
                raise ValueError(f"{sid}: total_daily_steps does not match HDF5")
            if length > len(days):
                raise ValueError("window_days exceeds the HDF5 daily axis")
            stop = min(len(days), max_days) if max_days else len(days)
            allowed = eligible_days(config, split, days)
            # Observation membership follows the window start date.
            if stop not in offsets:
                offsets[stop] = [(start, start+length) for start in range(0, stop-length+1, int(stride))]
            for start, end in offsets[stop]:
                if allowed[start]:
                    indexes.append((sid, start, end))
                    if max_windows and len(indexes) >= max_windows:
                        return indexes
        return indexes

    return load_or_build(
        h5_dir, "index",
        {"signature": directory_signature(h5_dir), "layout": layout, "split": split,
         "length": length, "stride": int(stride), "stations": ids,
         "max_windows": max_windows, "max_days": max_days,
         "split_spec": split_spec(config), "total": _cfg_get(config, "data.total_daily_steps")},
        build, save_index_map, load_index_map)

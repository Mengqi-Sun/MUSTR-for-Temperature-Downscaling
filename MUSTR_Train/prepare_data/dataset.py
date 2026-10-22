"""Read complete configurable daily windows directly from station HDF5 files."""
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

from DataCube.schema import INPUT_SIZE, MET, validate_header
from ..config import cfg_get, uses_physics
from ..contracts import fill_daily_input
from .features import read_daily_features
from .indexing import window_days, _resolve_station_id_col
from .normalization import load_daily_normalizer
from .physics import build_window_physics
from .repacked import (
    LAYOUT_REPACKED, RepackedStation, cached_station, encode_class_array, resolve_layout,
)
from .supervision import SupervisionStats


class MUPNetDataset(Dataset):
    def __init__(self, config, split, station_df, index_map):
        self.config, self.split = config, split
        self.station_df, self.index_map = station_df, list(index_map)
        self.h5_dir = Path(cfg_get(config, "data.h5_dir"))
        self.window_days = window_days(config)
        self.spatial_size = int(cfg_get(config, "data.spatial_size", INPUT_SIZE))
        if self.spatial_size != INPUT_SIZE:
            raise ValueError(f"Daily cubes require spatial_size={INPUT_SIZE}")
        self.virtual_file_prefix = str(cfg_get(config, "data.virtual_file_prefix", "VC"))
        col = _resolve_station_id_col(station_df)
        self.station_meta = {str(row[col]): row.to_dict() for _, row in station_df.iterrows()}
        self.daily_input_normalizer = load_daily_normalizer(config)
        self.physics_enabled = uses_physics(config)
        self.layout = resolve_layout(config, self.h5_dir, [sid for sid, _, _ in self.index_map])
        # Repacked physics must match the configured priors.
        if self.layout == LAYOUT_REPACKED and self.index_map:
            # Check in the parent process before starting DataLoader workers.
            probe = RepackedStation(self.h5_dir/f"{self.index_map[0][0]}.h5", config, split,
                                    check_physics=self.physics_enabled)
            probe.close()
        LOGGER.info("%s split reads %d %s cubes from %s", split, len({s for s, _, _ in self.index_map}),
                    self.layout, self.h5_dir)
        self.window_supervision_stats = []
        if split == "train" and cfg_get(config, "training.sampling_mode", "") == "semi_constrained":
            self.window_supervision_stats = SupervisionStats(self.h5_dir, self.index_map,
                                                             layout=self.layout)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        sid, start, end = self.index_map[index]
        if end-start != self.window_days:
            raise ValueError("Index map and configured daily context disagree")
        return self._read_sample(str(sid), start, end)

    def get_by_station_and_day(self, station_id, start_day):
        return self._read_sample(str(station_id), int(start_day), int(start_day)+self.window_days)

    def _is_real(self, sid):
        return bool(self.station_meta.get(sid, {}).get(
            "is_real", not sid.startswith(self.virtual_file_prefix)))

    def _read_sample(self, sid, start, end):
        if self.layout == LAYOUT_REPACKED:
            station = cached_station(self.h5_dir/f"{sid}.h5", self.config, self.split,
                                     check_physics=self.physics_enabled)
            return station.read(start, end, is_real=self._is_real(sid))
        with h5py.File(self.h5_dir/f"{sid}.h5", "r") as h5:
            validate_header(h5)
            if start < 0 or end > len(h5["time/day_index"]):
                raise ValueError("Incomplete temporal context; no padding beyond the daily axis")
            features, raw, clcz, days = read_daily_features(h5, start, end)
            normalizer = self.daily_input_normalizer
            # Fill in physical units; preserve unfilled features for validity masks.
            filled = fill_daily_input(features)
            network = normalizer.transform(filled) if normalizer else filled
            labels = {key: torch.from_numpy(h5[f"target/met/daily/{key}"][start:end]) for key in MET}
            lst, hour = h5["target/lst_obs"][start:end], h5["target/lst_hour"][start:end]
            # Observations split by window start; physics eligibility is checked per day.
            sample = {
                "daily_input_features": torch.from_numpy(network),
                "daily_input_valid": torch.from_numpy(np.isfinite(features)),
                "daily_clcz_class": torch.from_numpy(encode_class_array(clcz, 18)),
                "kg_class": torch.from_numpy(encode_class_array(h5["static/kg"][:], 29)),
                "daily_labels": {k: v for k, v in labels.items() if k.startswith("tem")},
                "daily_gst_labels": {k: v for k, v in labels.items() if k.startswith("gst")},
                "lst_obs": torch.from_numpy(lst), "lst_hour": torch.from_numpy(hour),
                "lst_valid": torch.from_numpy(np.isfinite(hour) & np.isfinite(lst).any(axis=(1, 2))),
                "timing_daily": {key: torch.from_numpy(h5[f"daily/derived/{key}"][start:end])
                                 for key in ("era5_skt_min_hour", "era5_skt_max_hour")},
                "is_real": self._is_real(sid),
                "phys_daily": {},
            }
            center = self.spatial_size//2
            for prefix, var in (("ta", "t2m"), ("ts", "skt")):
                reference = np.stack([raw[f"{var}_{stat}"] for stat in ("min", "mean", "max")], axis=1)
                valid = np.isfinite(reference).all(axis=1)
                sample[f"{prefix}_reference_daily"] = torch.from_numpy(np.nan_to_num(reference, nan=0., posinf=0., neginf=0.))
                sample[f"{prefix}_reference_valid"] = torch.from_numpy(valid)
                anchor = raw[f"{var}_mean"][:, center, center]
                # Network fallback only; raw physics and validity remain unchanged.
                sample["phys_daily"][f"{var}_mean"] = torch.from_numpy(np.where(np.isfinite(anchor), anchor, 273.15).astype(np.float32))
            if self.physics_enabled:
                sample["physics"] = build_window_physics(h5, raw, clcz, days, start, end, self.config, self.split)
                sample["sample_key"] = (sid, start, end)
            return sample

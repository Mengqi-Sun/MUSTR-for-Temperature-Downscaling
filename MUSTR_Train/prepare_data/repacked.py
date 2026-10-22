"""Read contiguous DataCubes written by repack_cubes.py; supports repack_v1 and v2."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch

from DataCube.schema import INPUT_SIZE, MET
from ..config import cfg_get, uses_physics
from ..contracts import (
    DAILY_CHANNELS_WITH_VALID, DAILY_INPUT_CHANNELS, fill_daily_input,
)
from .indexing import eligible_days
from .normalization import load_daily_normalizer

# v2 packs spatial data in x and keeps vec/phys_pair resident per station.
REPACK_VERSION = "repack_v2"
# v1 remains readable; conversion writes v2.
READABLE_VERSIONS = ("repack_v1", REPACK_VERSION)
LAYOUT_PRODUCER = "producer"
LAYOUT_REPACKED = "repacked"
LAYOUTS = (LAYOUT_PRODUCER, LAYOUT_REPACKED, "auto")

SPATIAL_EXTRA = ("clcz", "lst_obs")

# Hour anchors and day indices are exactly representable as float32.
VEC_BASE = (("temporal_sin_d", "temporal_cos_d", "temporal_weekend")
            + tuple(MET) + ("lst_hour", "era5_skt_min_hour", "era5_skt_max_hour", "day_index"))

# Keep enough per-worker handles to cover a batch of stations.
_MAX_HANDLES = 96
_HANDLES: "OrderedDict[str, RepackedStation]" = OrderedDict()


def resolve_layout(config, h5_dir, station_ids=()):
    """Resolve the configured layout, probing one cube when set to auto."""
    requested = str(cfg_get(config, "data.h5_layout", "auto")).strip().lower()
    if requested not in LAYOUTS:
        raise ValueError(f"data.h5_layout must be one of {LAYOUTS}, got {requested!r}")
    if requested != "auto":
        return requested
    for sid in list(station_ids)[:1] or [p.stem for p in sorted(Path(h5_dir).glob("*.h5"))[:1]]:
        path = Path(h5_dir) / f"{sid}.h5"
        if not path.exists():
            break
        with h5py.File(path, "r") as probe:
            return LAYOUT_REPACKED if probe.attrs.get("Repack_Version") else LAYOUT_PRODUCER
    return LAYOUT_PRODUCER


_FINGERPRINTS: dict = {}


def physics_fingerprint(config, channels=None):
    """Fingerprint stored physics using the file channel list and current forcing settings."""
    from ..config import physics_priors
    from .provenance import fingerprint
    # Stored physics also depends on the Richardson geometry constants.
    from ..train.losses.physical_loss import (
        RICHARDSON_RANGE, STABILITY_FACTOR_RANGE, STABILITY_HEIGHT_M)
    channels = tuple(pack_channels() if channels is None else channels)
    key = (id(config), channels)
    if key not in _FINGERPRINTS:
        _FINGERPRINTS[key] = fingerprint({
            "priors": physics_priors(config),
            "mode": cfg_get(config, "physics.mode"),
            "windows": cfg_get(config, "physics.windows"),
            "channels": list(channels),
            "input_channels": list(DAILY_INPUT_CHANNELS),
            "stability_geometry": {
                "height_m": float(STABILITY_HEIGHT_M),
                "richardson_range": [float(v) for v in RICHARDSON_RANGE],
                "factor_range": [float(v) for v in STABILITY_FACTOR_RANGE],
            },
        })
    return _FINGERPRINTS[key]


def pack_channels():
    """The channels of `x`, in file order: sources first, then the extras."""
    from DataCube.schema import TEXTURES
    from ..contracts import DAILY_ERA5_INPUT_CHANNELS
    return tuple(DAILY_ERA5_INPUT_CHANNELS) + tuple(TEXTURES) + SPATIAL_EXTRA


def open_repacked(path):
    """Open and header-check one repacked cube."""
    handle = h5py.File(path, "r")
    found = handle.attrs.get("Repack_Version")
    if found not in READABLE_VERSIONS:
        handle.close()
        raise ValueError(
            f"{path} is in layout {found or 'unknown (no Repack_Version attribute)'}; this code reads "
            f"{' and '.join(READABLE_VERSIONS)}. Run "
            f"`python <package>/prepare_data/repack_cubes.py convert --dst {Path(path).parent}` "
            f"and wait for it to finish before training."
        )
    return handle


def station_days(path, layout):
    """The day-index vector, whichever layout the cube uses."""
    if layout == LAYOUT_REPACKED:
        with open_repacked(path) as handle:
            return handle["day_index"][:]
    from DataCube.schema import validate_header
    with h5py.File(path, "r") as handle:
        validate_header(handle)
        return handle["time/day_index"][:]


def supervision_masks(path):
    """Return per-day TEM/GST/Landsat masks, using the 11x11 sampler footprint."""
    with open_repacked(path) as handle:
        if handle.attrs["Repack_Version"] == REPACK_VERSION:
            masks = handle["supervision"][:].astype(bool)      # [4, D]
        else:
            # v1 has no precomputed supervision masks.
            masks = supervision_from_arrays(handle["met"][:], handle["lst_hour"][:],
                                            handle["lst_obs"][:]).astype(bool)
    return masks[0], masks[1], masks[2], masks[3]


def supervision_from_arrays(met, lst_hour, lst_obs):
    """Compute repacked supervision masks with the same 11x11 support as the original reader."""
    index = {name: i for i, name in enumerate(MET)}
    tem, gst = (np.logical_or.reduce([np.isfinite(met[index[f"{prefix}_{stat}"]])
                                      for stat in ("min", "mean", "max")])
                for prefix in ("tem", "gst"))
    raw = np.isfinite(lst_hour)
    centre = lst_obs.shape[-1] // 2
    support = slice(centre - 5, centre + 6)
    lst = raw & np.isfinite(lst_obs[:, support, support]).any(axis=(1, 2))
    return np.stack((tem, gst, lst, raw)).astype(np.uint8)


class RepackedStation:
    """One open repacked cube.  Hold one per station per worker, not per sample."""

    def __init__(self, path, config, split, check_physics=False):
        self.path = str(path)
        self.h5 = open_repacked(path)
        self.version = str(self.h5.attrs["Repack_Version"])
        self.channels = json.loads(self.h5.attrs["Pack_Channels"])
        if check_physics and self.h5.attrs.get("Physics_Fingerprint") != physics_fingerprint(config, self.channels):
            self.h5.close()
            raise ValueError(
                f"{path}: the stored physics coefficients do not match this config's priors "
                "or mode. Re-run repack_cubes.py convert; training on them would use stale physics."
            )
        self.config, self.split = config, split
        self.index = {name: i for i, name in enumerate(self.channels)}
        self.spatial_keys = json.loads(self.h5.attrs["Phys_Spatial_Keys"])
        self.vector_keys = json.loads(self.h5.attrs["Phys_Vector_Keys"])
        self.normalizer = load_daily_normalizer(config)
        self.physics_enabled = uses_physics(config)
        self.n_days = int(self.h5.attrs["Daily_Steps"])
        self.size = int(self.h5.attrs["Input_Size"])
        if self.size != INPUT_SIZE:
            raise ValueError(f"{path}: daily cubes require spatial_size={INPUT_SIZE}")
        self.station_id = str(self.h5.attrs["Station_Id_C"])
        self.static = self.h5["static"][:]
        self.kg = encode_class_array(self.h5["kg"][:], 29)
        if self.version == REPACK_VERSION:
            self.vec = self.h5["vec"][:]
            self.vec_index = {name: i for i, name in enumerate(json.loads(self.h5.attrs["Vec_Names"]))}
            self.phys_pair = self.h5["phys_pair"][:] if self.physics_enabled else None
        else:
            self.vec, self.vec_index, self.phys_pair = self._resident_arrays_from_v1()
        self.day_index = self.vec[:, self.vec_index["day_index"]].astype(np.int64)
        # v2 stores CLCZ and Landsat inside x; v1 uses separate datasets.
        self.clcz_channel = self.index.get("clcz")
        self.lst_channel = self.index.get("lst_obs")

    def _resident_arrays_from_v1(self):
        """Load v1 daily vectors into the resident array layout used by v2."""
        h5 = self.h5
        met = h5["met"][:]                          # [6, D] in MET order
        hours = h5["skt_hours"][:]                  # [2, D]: minimum and maximum hours
        temporal = h5["temporal"][:]                # [D, 3]
        columns = {
            "temporal_sin_d": temporal[:, 0], "temporal_cos_d": temporal[:, 1],
            "temporal_weekend": temporal[:, 2],
            **{name: met[i] for i, name in enumerate(MET)},
            "lst_hour": h5["lst_hour"][:],
            "era5_skt_min_hour": hours[0], "era5_skt_max_hour": hours[1],
            "day_index": h5["day_index"][:],
        }
        names = list(VEC_BASE)
        phys_pair = None
        if self.physics_enabled:
            scalar = h5["phys_scalar"][:]           # [D, 8] in Phys_Vector_Keys order
            columns.update({key: scalar[:, i] for i, key in enumerate(self.vector_keys)})
            names += list(self.vector_keys)

            def pad(values):                       # Pair arrays are one day shorter.
                return np.concatenate([values, np.zeros_like(values[:1])], axis=0)
            phys_pair = np.stack([h5["phys_dry"][:].astype(np.float32),
                                  pad(h5["pair_base"][:]).astype(np.float32),
                                  pad(h5["pair_confidence"][:]).astype(np.float32)], axis=1)
        vec = np.stack([np.asarray(columns[name], dtype=np.float32) for name in names], axis=1)
        return vec, {name: i for i, name in enumerate(names)}, phys_pair

    def close(self):
        self.h5.close()

    def read(self, start, end, is_real=True):
        if start < 0 or end > self.n_days:
            raise ValueError("Incomplete temporal context; no padding beyond the daily axis")

        x = self.h5["x"][start:end]
        v = self.vec[start:end]
        days = self.day_index[start:end]
        shape = (end - start, self.size, self.size)

        if self.clcz_channel is None:
            clcz = self.h5["clcz"][start:end].astype(np.int64)
            lst = self.h5["lst_obs"][start:end]
        else:
            clcz = x[:, self.clcz_channel].astype(np.int64)
            lst = np.ascontiguousarray(x[:, self.lst_channel])

        channels = {name: x[:, i] for name, i in self.index.items()}
        for name in DAILY_CHANNELS_WITH_VALID:
            channels[f"{name}_valid"] = np.isfinite(channels[name]).astype(np.float32)
        for offset, name in enumerate(("dem", "cmtbh", "svf")):
            channels[name] = np.broadcast_to(self.static[offset], shape)
        for name in ("temporal_sin_d", "temporal_cos_d", "temporal_weekend"):
            channels[name] = np.broadcast_to(v[:, self.vec_index[name]][:, None, None], shape)
        features = np.stack([channels[name] for name in DAILY_INPUT_CHANNELS], axis=1).astype(np.float32)

        # Fill in physical units to match the original reader.
        filled = fill_daily_input(features)
        network = self.normalizer.transform(filled) if self.normalizer else filled

        labels = {key: torch.from_numpy(np.ascontiguousarray(v[:, self.vec_index[key]]))
                  for key in MET}
        hour = np.ascontiguousarray(v[:, self.vec_index["lst_hour"]])
        skt_hours = np.stack([v[:, self.vec_index[f"era5_skt_{stat}_hour"]]
                              for stat in ("min", "max")]).astype(np.uint8)

        sample = {
            "daily_input_features": torch.from_numpy(network),
            "daily_input_valid": torch.from_numpy(np.isfinite(features)),
            "daily_clcz_class": torch.from_numpy(encode_class_array(clcz, 18)),
            "kg_class": torch.from_numpy(self.kg),
            "daily_labels": {k: v for k, v in labels.items() if k.startswith("tem")},
            "daily_gst_labels": {k: v for k, v in labels.items() if k.startswith("gst")},
            "lst_obs": torch.from_numpy(lst), "lst_hour": torch.from_numpy(hour),
            "lst_valid": torch.from_numpy(np.isfinite(hour) & np.isfinite(lst).any(axis=(1, 2))),
            "timing_daily": {key: torch.from_numpy(np.ascontiguousarray(skt_hours[i]))
                             for i, key in enumerate(("era5_skt_min_hour", "era5_skt_max_hour"))},
            "is_real": bool(is_real),
            "phys_daily": {},
        }
        centre = self.size // 2
        for prefix, var in (("ta", "t2m"), ("ts", "skt")):
            reference = np.stack([channels[f"{var}_{stat}"] for stat in ("min", "mean", "max")], axis=1)
            valid = np.isfinite(reference).all(axis=1)
            sample[f"{prefix}_reference_daily"] = torch.from_numpy(
                np.nan_to_num(reference, nan=0., posinf=0., neginf=0.))
            sample[f"{prefix}_reference_valid"] = torch.from_numpy(valid)
            anchor = channels[f"{var}_mean"][:, centre, centre]
            # Network fallback only; raw physics and validity remain unchanged.
            sample["phys_daily"][f"{var}_mean"] = torch.from_numpy(
                np.where(np.isfinite(anchor), anchor, 273.15).astype(np.float32))
        if self.physics_enabled:
            sample["physics"] = self._physics(start, end, days)
            sample["sample_key"] = (self.station_id, start, end)
        return sample

    def _physics(self, start, end, days):
        phys = self.h5["phys"][start:end]
        v = self.vec[start:end]
        pair = self.phys_pair[start:end]
        out = {key: torch.from_numpy(np.ascontiguousarray(phys[:, i]))
               for i, key in enumerate(self.spatial_keys)}
        out.update({key: torch.from_numpy(np.ascontiguousarray(v[:, self.vec_index[key]]))
                    for key in self.vector_keys})
        out["dry_paved"] = torch.from_numpy(np.ascontiguousarray(pair[:, 0] > 0.5))
        # Apply split eligibility at read time and discard the padded pair tail.
        allowed = eligible_days(self.config, self.split, days)
        base = pair[:end - 1 - start, 1] > 0.5
        out["pair_valid"] = torch.from_numpy(base & (allowed[1:] & allowed[:-1])[:, None, None])
        out["pair_confidence"] = torch.from_numpy(
            np.ascontiguousarray(pair[:end - 1 - start, 2]))
        out["day_index"] = torch.from_numpy(days.astype(np.int64))
        return out


def cached_station(path, config, split, check_physics=False):
    """Cache open cubes per worker; create handles lazily after worker startup."""
    key = f"{split}:{path}"
    station = _HANDLES.pop(key, None)
    if station is None:
        station = RepackedStation(path, config, split, check_physics)
        while len(_HANDLES) >= _MAX_HANDLES:
            _, stale = _HANDLES.popitem(last=False)
            stale.close()
    _HANDLES[key] = station
    return station


def encode_class_array(values, upper):
    """Class codes 1..upper pass through; anything else becomes 0 (unknown)."""
    values = np.asarray(values)
    valid = np.isfinite(values) & (values == np.floor(values)) & (values >= 1) & (values <= upper)
    return np.where(valid, values, 0).astype(np.int64)

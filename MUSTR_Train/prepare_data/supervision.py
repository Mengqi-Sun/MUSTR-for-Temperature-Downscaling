"""Compact in-memory sampler counts, read directly from daily HDF5 targets."""
import hashlib
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from DataCube.schema import validate_header


class SupervisionStats:
    """Store per-window observation counts for supervision-aware sampling."""
    def __init__(self, h5_dir, index_map, layout=None):
        from .repacked import LAYOUT_REPACKED, supervision_masks
        from .scan_cache import directory_signature, load_or_build

        def build():
            counts = np.zeros((len(index_map), 5), dtype=np.uint16)
            grouped = defaultdict(list)
            for i, (sid, start, end) in enumerate(index_map):
                grouped[sid].append((i, start, end))
            for sid, entries in grouped.items():
                path = Path(h5_dir)/f"{sid}.h5"
                if layout == LAYOUT_REPACKED:
                    tem, gst, lst, raw = supervision_masks(path)
                else:
                    tem, gst, lst, raw = self._producer_masks(path)
                raw = raw & lst  # Restrict sampling counts to the established Landsat footprint.
                arrays = np.stack((tem, gst, lst, raw, gst & lst), axis=1)
                prefix = np.vstack((np.zeros((1, 5), dtype=np.int32), np.cumsum(arrays, axis=0)))
                for i, start, end in entries:
                    counts[i] = prefix[end]-prefix[start]
            return counts

        # Counts depend on the exact window ordering.
        self.counts = load_or_build(
            h5_dir, "supervision",
            {"signature": directory_signature(h5_dir), "layout": layout,
             "windows": len(index_map),
             "map_sha": hashlib.sha256(
                 "\n".join(f"{s}:{a}:{b}" for s, a, b in index_map).encode("utf-8")).hexdigest()},
            build, lambda counts: {"counts": counts}, lambda stored: stored["counts"])

    @staticmethod
    def _producer_masks(path):
        with h5py.File(path, "r") as h5:
            validate_header(h5)
            tem, gst = [np.logical_or.reduce([np.isfinite(h5[f"target/met/daily/{prefix}_{stat}"][:])
                                              for stat in ("min", "mean", "max")]) for prefix in ("tem", "gst")]
            raw = np.isfinite(h5["target/lst_hour"][:])
            lst = np.zeros(len(raw), dtype=bool)
            center = h5["target/lst_obs"].shape[-1]//2
            support = slice(center-5, center+6)  # 11x11 sampler footprint within the 19x19 tile.
            for start in range(0, len(raw), 32):
                lst[start:start+32] = raw[start:start+32] & np.isfinite(
                    h5["target/lst_obs"][start:start+32, support, support]).any(axis=(1, 2))
        return tem, gst, lst, raw

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        tem, gst, lst, raw, overlap = map(int, self.counts[index])
        return {"n_tem_daily_days": tem, "n_gst_daily_days": gst, "n_lst_frames": lst,
                "n_lst_frames_raw": raw, "has_tem_daily": tem > 0, "has_gst_daily": gst > 0,
                "has_landsat": lst > 0, "has_gst_landsat_overlap": overlap > 0,
                "has_overlap": int(tem > 0)+int(gst > 0)+int(lst > 0) >= 2}

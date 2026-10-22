#!/usr/bin/env python
"""Repack DataCubes into contiguous arrays and verify reader equivalence.

Use convert or verify with --src, --dst, and --config. Existing complete outputs
are skipped unless --force is set.
"""

from __future__ import annotations

# Repository-relative defaults; override with --src, --dst, and --config.
SRC_DIR = "DataCube/samples/cubes"
DST_DIR = "DataCube/repacked"
CONFIG_PATH = "MUSTR_Train/configs/sample.yaml"

# None derives the repository and package from this file.
REPO_ROOT = None
PACKAGE = None

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

# Limit BLAS threads before importing torch; stations run in separate processes.
for _threads in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")
# Disable HDF5 locking for Lustre.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

NIGHT_BOUNDARY_KEYS = (
    "richardson_night", "stability_night", "ra_night",
    "G_H_night", "G_E_night", "G_C_night",
)


def _bootstrap():
    """Put the repository root on sys.path and return the package name."""
    here = Path(__file__).resolve()
    root = Path(REPO_ROOT).resolve() if REPO_ROOT else here.parents[2]
    package = PACKAGE or here.parents[1].name
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not (root / package / "prepare_data" / "dataset.py").exists():
        raise SystemExit(
            f"Cannot locate the training package: expected {root/package}/prepare_data/dataset.py\n"
            f"Set REPO_ROOT and PACKAGE at the top of this script."
        )
    return root, package


REPO, PKG = _bootstrap()

import importlib

_config = importlib.import_module(f"{PKG}.config")
_contracts = importlib.import_module(f"{PKG}.contracts")
_features = importlib.import_module(f"{PKG}.prepare_data.features")
_physics = importlib.import_module(f"{PKG}.prepare_data.physics")
_indexing = importlib.import_module(f"{PKG}.prepare_data.indexing")
_dataset = importlib.import_module(f"{PKG}.prepare_data.dataset")
_repacked = importlib.import_module(f"{PKG}.prepare_data.repacked")

from DataCube.schema import INPUT_SIZE, MET, validate_header

load_config = _config.load_config
cfg_get = _config.cfg_get
DAILY_INPUT_CHANNELS = _contracts.DAILY_INPUT_CHANNELS
read_daily_features = _features.read_daily_features
build_window_physics = _physics.build_window_physics
TIMING_FIELDS = _physics.TIMING_FIELDS

RepackedStation = _repacked.RepackedStation
REPACK_VERSION = _repacked.REPACK_VERSION
_physics_fingerprint = _repacked.physics_fingerprint

PACK_CHANNELS = _repacked.pack_channels()

import torch
torch.set_num_threads(1)



def _split_independent_config(config, day_index):
    """Enable all days when storing physics; readers apply their own split masks."""
    from datetime import datetime, timedelta
    years = sorted({(datetime(2018, 1, 1) + timedelta(days=int(d))).year for d in day_index})
    relaxed = deepcopy(config)
    relaxed.setdefault("data", {})["train_years"] = years
    relaxed["data"]["val_months"] = {}
    return relaxed


def _classify_physics(physics, n_days):
    """Group physics tensors into the stored arrays."""
    spatial, vectors = {}, {}
    pairs, dry = {}, None
    for key, tensor in physics.items():
        value = tensor.numpy()
        if key == "day_index":
            continue
        if key == "dry_paved":
            dry = value
        elif key in ("pair_valid", "pair_confidence"):
            pairs[key] = value
        elif value.ndim == 1:
            vectors[key] = value.astype(np.float32)
        elif value.shape[0] == n_days:
            spatial[key] = value.astype(np.float32)
        else:
            raise ValueError(f"Unclassifiable physics key {key} with shape {value.shape}")
    missing = set(TIMING_FIELDS) - set(vectors)
    if missing:
        raise ValueError(f"Physics vectors missing the timing scalars: {sorted(missing)}")
    if dry is None or set(pairs) != {"pair_valid", "pair_confidence"}:
        raise ValueError("Physics output is missing dry_paved or the pair arrays")
    return spatial, vectors, pairs, dry


def repack_station(sid, src_dir, dst_dir, config, force=False):
    """Read one producer cube and write its repacked form.  Returns a stats dict."""
    source = Path(src_dir) / f"{sid}.h5"
    target = Path(dst_dir) / f"{sid}.h5"
    stat = source.stat()
    if target.exists() and not force:
        try:
            with h5py.File(target, "r") as done:
                if (done.attrs.get("Repack_Version") == REPACK_VERSION
                        and int(done.attrs.get("Source_Size", -1)) == stat.st_size
                        and done.attrs.get("Physics_Fingerprint") == _physics_fingerprint(config)):
                    return {"station": sid, "skipped": True, "bytes": target.stat().st_size}
        except Exception:
            pass  # Rewrite unreadable or truncated output.

    started = time.perf_counter()
    with h5py.File(source, "r") as h5:
        validate_header(h5)
        n_days = len(h5["time/day_index"])
        features, raw, clcz, days = read_daily_features(h5, 0, n_days)
        relaxed = _split_independent_config(config, days)
        physics = build_window_physics(h5, raw, clcz, days, 0, n_days, relaxed, "train")

        static = np.stack([h5[f"static/{n}"][:] for n in ("dem", "cmtbh", "svf")]).astype(np.float32)
        kg = h5["static/kg"][:].astype(np.uint8)
        lst_obs = h5["target/lst_obs"][:].astype(np.float32)
        lst_hour = h5["target/lst_hour"][:].astype(np.float32)
        met = np.stack([h5[f"target/met/daily/{k}"][:] for k in MET]).astype(np.float32)
        hours = np.stack([h5[f"daily/derived/era5_skt_{s}_hour"][:] for s in ("min", "max")])

        sources = [raw[name] for name in PACK_CHANNELS if name not in _repacked.SPATIAL_EXTRA]
        stack = np.stack(sources + [clcz.astype(np.float32), lst_obs], axis=1).astype(np.float32)

    spatial, vectors, pairs, dry = _classify_physics(physics, n_days)
    spatial_keys = sorted(spatial)
    vector_keys = sorted(vectors)
    phys = np.stack([spatial[k] for k in spatial_keys], axis=1).astype(np.float32)

    vec_names = list(_repacked.VEC_BASE) + vector_keys
    columns = {
        **{name: features[:, DAILY_INPUT_CHANNELS.index(name)][:, 0, 0]
           for name in ("temporal_sin_d", "temporal_cos_d", "temporal_weekend")},
        **{name: met[i] for i, name in enumerate(MET)},
        "lst_hour": lst_hour,
        "era5_skt_min_hour": hours[0], "era5_skt_max_hour": hours[1],
        "day_index": days,
        **{key: vectors[key] for key in vector_keys},
    }
    vec = np.stack([np.asarray(columns[name], dtype=np.float32) for name in vec_names], axis=1)

    # Pair arrays are one day shorter and padded for storage.
    def _pad(values):
        return np.concatenate([values, np.zeros_like(values[:1])], axis=0)
    phys_pair = np.stack([dry.astype(np.float32),
                          _pad(pairs["pair_valid"]).astype(np.float32),
                          _pad(pairs["pair_confidence"]).astype(np.float32)], axis=1)

    temporary = target.with_suffix(".h5.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Contiguous, uncompressed arrays support whole-window reads.
    with h5py.File(temporary, "w") as out:
        out.attrs["Repack_Version"] = REPACK_VERSION
        out.attrs["Station_Id_C"] = str(sid)
        out.attrs["Daily_Steps"] = n_days
        out.attrs["Input_Size"] = INPUT_SIZE
        out.attrs["Pack_Channels"] = json.dumps(list(PACK_CHANNELS))
        out.attrs["Vec_Names"] = json.dumps(vec_names)
        out.attrs["Phys_Spatial_Keys"] = json.dumps(spatial_keys)
        out.attrs["Phys_Vector_Keys"] = json.dumps(vector_keys)
        out.attrs["Physics_Fingerprint"] = _physics_fingerprint(config)
        out.attrs["Source_Size"] = stat.st_size
        out.attrs["Source_MTime_NS"] = stat.st_mtime_ns
        for name, value in (
            ("x", stack),
            ("vec", vec),
            ("phys", phys),
            ("phys_pair", phys_pair),
            ("static", static),
            ("kg", kg),
            ("day_index", days.astype(np.int32)),
            # Precompute supervision masks for the sampler scan.
            ("supervision", _repacked.supervision_from_arrays(met, lst_hour, lst_obs)),
        ):
            out.create_dataset(name, data=value)
    temporary.replace(target)
    return {"station": sid, "skipped": False, "bytes": target.stat().st_size,
            "seconds": time.perf_counter() - started}



def _original_reader(config, split, sid, window):
    """A one-station MUPNetDataset, so the comparison uses the real code path."""
    import pandas as pd
    quiet = deepcopy(config)
    quiet.setdefault("training", {})["sampling_mode"] = ""
    station_df = pd.DataFrame([{"Station_ID": str(sid), "is_real": True}])
    index_map = [(str(sid), 0, window)]
    return _dataset.MUPNetDataset(quiet, split, station_df, index_map)


def _compare(a, b, path=""):
    """Recursively diff two samples.  Returns a list of (key, description)."""
    import torch
    problems = []
    if isinstance(a, dict):
        if set(a) != set(b):
            problems.append((path, f"key sets differ: only-original={sorted(set(a)-set(b))} "
                                   f"only-repacked={sorted(set(b)-set(a))}"))
            return problems
        for key in sorted(a):
            problems += _compare(a[key], b[key], f"{path}/{key}" if path else key)
        return problems
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return [(path, f"shape {tuple(a.shape)} vs {tuple(b.shape)}")]
        if a.dtype != b.dtype:
            return [(path, f"dtype {a.dtype} vs {b.dtype}")]
        x, y = a.numpy(), b.numpy()
        if x.dtype == bool or np.issubdtype(x.dtype, np.integer):
            if not np.array_equal(x, y):
                return [(path, f"{int((x != y).sum())} of {x.size} entries differ")]
            return []
        if not np.allclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True):
            bad = ~np.isclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True)
            worst = np.abs(np.nan_to_num(x - y))[bad].max() if bad.any() else 0.0
            return [(path, f"{int(bad.sum())} of {x.size} entries differ, max |delta| {worst:.3e}")]
        return []
    if a != b:
        return [(path, f"{a!r} vs {b!r}")]
    return []


def verify_station(sid, src_dir, dst_dir, config, window, n_windows, splits=("train", "val"), seed=0):
    """Compare repacked reads against the live dataset for random windows."""
    result = {"station": sid, "checked": 0, "real": [], "boundary": 0}
    rng = np.random.default_rng(abs(hash(str(sid))) % (2**32) + seed)
    with h5py.File(Path(src_dir) / f"{sid}.h5", "r") as probe:
        n_days = len(probe["time/day_index"])
    starts = rng.choice(np.arange(0, n_days - window + 1), size=n_windows, replace=False)
    starts = [0, n_days - window] + sorted(int(s) for s in starts)

    for split in splits:
        original = _original_reader(config, split, sid, window)
        packed = RepackedStation(Path(dst_dir) / f"{sid}.h5", config, split, check_physics=True)
        try:
            for start in starts:
                want = original.get_by_station_and_day(sid, start)
                got = packed.read(start, start + window, is_real=True)
                for key, detail in _compare(want, got):
                    leaf = key.split("/")[-1]
                    if leaf in NIGHT_BOUNDARY_KEYS:
                        result["boundary"] += 1
                    else:
                        result["real"].append(f"[{split} start={start}] {key}: {detail}")
                result["checked"] += 1
        finally:
            packed.close()
    return result



def _station_ids(config, src_dir, limit, explicit):
    if explicit:
        return [str(s) for s in explicit]
    import pandas as pd
    csv = cfg_get(config, "data.station_csv")
    ids = []
    if csv and Path(csv).exists():
        frame = pd.read_csv(csv)
        column = _indexing._resolve_station_id_col(frame)
        split_column = _indexing._resolve_split_col(frame)
        if split_column:
            frame = frame[frame[split_column].astype(str).str.lower().isin(("train", "val"))]
        ids = [str(s) for s in frame[column].astype(str)]
    present = {p.stem for p in Path(src_dir).glob("*.h5")}
    ids = [s for s in ids if s in present] or sorted(present)
    return ids[:limit] if limit else ids


def _worker(args):
    sid, src, dst, config, force = args
    try:
        return repack_station(sid, src, dst, config, force=force)
    except Exception as error:                                  # noqa: BLE001
        return {"station": sid, "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc(limit=6)}


def _verify_worker(args):
    sid, src, dst, config, window, n_windows = args
    try:
        return verify_station(sid, src, dst, config, window, n_windows)
    except Exception as error:                                  # noqa: BLE001
        return {"station": sid, "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc(limit=6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("convert", "verify"))
    parser.add_argument("--src", default=SRC_DIR)
    parser.add_argument("--dst", default=DST_DIR)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=0, help="First N stations only")
    parser.add_argument("--stations", nargs="*", default=None, help="Explicit station ids")
    parser.add_argument("--workers", type=int, default=4,
                        help="Processes; each peaks around 0.5 GB")
    parser.add_argument("--force", action="store_true", help="Rewrite complete outputs")
    parser.add_argument("--windows", type=int, default=6,
                        help="Random windows per station per split, for verify")
    args = parser.parse_args()

    args.src = str(_config.resolve_project_path(args.src))
    args.dst = str(_config.resolve_project_path(args.dst))
    config = load_config(args.config)
    config.setdefault("data", {})["h5_dir"] = str(Path(args.src))
    window = int(cfg_get(config, "data.window_days", 7))
    ids = _station_ids(config, args.src, args.limit, args.stations)
    if not ids:
        raise SystemExit(f"No station cubes found under {args.src}")

    print(f"{args.action}: {len(ids)} stations  src={args.src}")
    print(f"            dst={args.dst}  config={args.config}  workers={args.workers}")
    if args.action == "convert":
        Path(args.dst).mkdir(parents=True, exist_ok=True)
        print(f"            physics fingerprint {_physics_fingerprint(config)[:16]}")
    print()

    started = time.perf_counter()
    if args.action == "convert":
        jobs = [(sid, args.src, args.dst, config, args.force) for sid in ids]
        done = written = skipped = failed = 0
        total_bytes = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in as_completed([pool.submit(_worker, job) for job in jobs]):
                item = result.result()
                done += 1
                if "error" in item:
                    failed += 1
                    print(f"  FAIL {item['station']}: {item['error']}")
                    print("       " + item["trace"].replace("\n", "\n       "))
                else:
                    total_bytes += item["bytes"]
                    skipped += bool(item["skipped"])
                    written += not item["skipped"]
                if done % 25 == 0 or done == len(jobs):
                    rate = done / max(time.perf_counter() - started, 1e-9)
                    left = (len(jobs) - done) / max(rate, 1e-9)
                    print(f"  {done}/{len(jobs)}  written={written} skipped={skipped} "
                          f"failed={failed}  {rate:.1f} st/s  eta {left/60:.0f} min")
        print()
        print(f"wrote {total_bytes/1e9:.1f} GB in {(time.perf_counter()-started)/60:.1f} min")
        if failed:
            print(f"FAILED on {failed} station(s); rerun to retry only those.")
        raise SystemExit(1 if failed else 0)

    jobs = [(sid, args.src, args.dst, config, window, args.windows) for sid in ids]
    checked = boundary = 0
    problems, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in as_completed([pool.submit(_verify_worker, job) for job in jobs]):
            item = result.result()
            if "error" in item:
                errors.append(f"{item['station']}: {item['error']}\n{item['trace']}")
                continue
            checked += item["checked"]
            boundary += item["boundary"]
            problems += [f"{item['station']} {p}" for p in item["real"]]

    print(f"compared {checked} samples across {len(ids)} stations, "
          f"{args.windows + 2} windows x 2 splits each")
    print(f"  known night-window boundary differences : {boundary}")
    print(f"  unexplained differences                 : {len(problems)}")
    for line in problems[:25]:
        print(f"    {line}")
    if len(problems) > 25:
        print(f"    ... and {len(problems)-25} more")
    for line in errors[:5]:
        print(f"  ERROR {line}")
    print()
    ok = not problems and not errors
    print("PASS - the repacked files reproduce the live dataset." if ok
          else "FAIL - do not switch the training config over.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

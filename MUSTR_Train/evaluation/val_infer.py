"""Save full-period station predictions and diagnostics from trained checkpoints.

Outputs are float32 [day, 14, 5, 5] arrays in Kelvin, ordered by CHANNELS.
Day 0 is 2018-01-01 in UTC+8; the station pixel is [2, 2]. Undefined values
are NaN. Mask acquisition-time comparisons with finite observed Landsat LST.

"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import cfg_get, load_config, public_config, resolve_project_path
from ..contracts import (
    CENTER_SIZE, DAILY_INPUT_FEATURE_COUNT, HALO_PIXELS, MODEL_CONTRACT_VERSION,
    TILE_SIZE, validate_daily_batch,
)
from ..main_models import build_model
from ..prepare_data.dataset import MUPNetDataset
from ..prepare_data.indexing import (
    _resolve_station_id_col, build_station_df_from_config, window_days,
)
from ..prepare_data.repacked import station_days
from ..train.losses.joint_loss import JointLoss
from ..train.losses.physical_loss import (
    solve_closure_temperature, validate_daynight_physics_batch,
)

LOGGER = logging.getLogger("val_infer")

CHANNELS = (
    "Ta_min", "Ta_mean", "Ta_max", "Ta_day", "Ta_night",
    "Ts_min", "Ts_mean", "Ts_max", "Ts_day", "Ts_night",
    "Ts_closure_day", "Ts_closure_night",
    "Ts_landsat_time", "Landsat_LST",
)
CH = {name: index for index, name in enumerate(CHANNELS)}
FIRST_DATE = "2018-01-01"
DEFAULT_OUT = str(resolve_project_path("outputs/predictions"))
DEFAULT_CHECKPOINT = "best_observation.pt"
# Positive weight enables physics reads; evaluation does not optimize the loss.
READER_PHYSICS_WEIGHT = 1.0
_PATCH = slice(HALO_PIXELS, HALO_PIXELS + CENTER_SIZE)
_TILE_CENTRE = TILE_SIZE // 2
# The all pool includes train and val rows only.
SPLITS = ("val", "train", "all")
# The validation reader is extended to all years by reader_config.
READER_SPLIT = "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories (globs allowed); each holds resolved_config.yaml "
                             "and a checkpoint. The directory name names the output folder")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint file relative to each run (default %(default)s)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output root (default %(default)s)")
    parser.add_argument("--split", choices=SPLITS, default="val",
                        help="Station pool: val, train (real and virtual) or all = train + val "
                             "(default %(default)s)")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="Only these station IDs of the pool (default: all of them)")
    parser.add_argument("--stations-file", default=None,
                        help="Text file of station IDs, one per line ('#' starts a comment); "
                             "added to --stations")
    parser.add_argument("--limit", type=int, default=0, help="First N stations only; 0 = all")
    parser.add_argument("--shard", default="0/1", metavar="K/N",
                        help="Process stations K, K+N, K+2N, ... (0-based); for SLURM arrays")
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    parser.add_argument("--batch-windows", type=int, default=128,
                        help="Windows per forward pass (default %(default)s)")
    parser.add_argument("--amp", action="store_true",
                        help="Autocast the forward with the run's training.amp_dtype, as training "
                             "validation did. Default is float32")
    parser.add_argument("--overwrite", action="store_true", help="Redo stations already written")
    parser.add_argument("--h5-dir", default=None, help="Cube directory; layout is auto-detected")
    parser.add_argument("--station-csv", default=None)
    return parser.parse_args()




@dataclass
class Run:
    name: str
    directory: Path
    checkpoint: Path
    config: dict
    model: torch.nn.Module
    epoch: int
    sha256: str
    physics_loss: Any
    landsat_operator: Any


def expand_runs(patterns) -> list[Path]:
    runs = []
    for pattern in patterns:
        pattern = str(resolve_project_path(pattern))
        matches = sorted(glob.glob(pattern)) or [pattern]
        runs.extend(Path(m) for m in matches if Path(m).is_dir())
    if not runs:
        raise FileNotFoundError(f"No run directories match {patterns}")
    names = [run.name for run in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"Run directory names must be unique; they name the outputs: {names}")
    return runs


def reader_config(config: dict, h5_dir=None, station_csv=None) -> dict:
    """Enable physics reads and extend day eligibility to the test year for evaluation."""
    config = deepcopy(config)
    data = config.setdefault("data", {})
    if h5_dir:
        data["h5_dir"] = str(resolve_project_path(h5_dir))
    if station_csv:
        data["station_csv"] = str(resolve_project_path(station_csv))
    for key in ("h5_dir", "station_csv"):
        if data.get(key):
            data[key] = str(resolve_project_path(data[key]))
    years = set(cfg_get(config, "data.train_years", [2018, 2019, 2020, 2021, 2022]))
    years.add(int(cfg_get(config, "data.test_year", 2023)))
    data["train_years"] = sorted(years)
    data["val_use_full_year"] = True
    physics = config.setdefault("physics", {})
    physics["enabled"] = True
    physics["weight"] = READER_PHYSICS_WEIGHT
    physics["phase_overrides"] = {f"phase{i}": {"enabled": True, "weight": READER_PHYSICS_WEIGHT}
                                  for i in range(1, 5)}
    return config


def reader_signature(config: dict) -> str:
    """Return reader settings that must agree across jointly evaluated runs."""
    return yaml.safe_dump({key: config.get(key) for key in ("data", "normalization", "physics")},
                          sort_keys=True, default_flow_style=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(directory: Path, checkpoint_name: str, device: torch.device, args) -> Run:
    path = directory / checkpoint_name
    if not path.is_file():
        path = directory / "checkpoints" / checkpoint_name
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"{path}: not a PyMUPNet trainer checkpoint")
    found = payload.get("model_contract_version")
    if found != MODEL_CONTRACT_VERSION:
        raise RuntimeError(f"{path} was trained under model contract {found}; this package "
                           f"implements {MODEL_CONTRACT_VERSION}")
    # Prefer the config stored in the checkpoint; use resolved_config.yaml as fallback.
    if isinstance(payload.get("config"), dict):
        trained = public_config(payload["config"])
    else:
        trained = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    resolved_path = directory / "resolved_config.yaml"
    if resolved_path.is_file():
        # Keep checkpoint science settings; relocate only data paths.
        portable = load_config(resolved_path)
        for key in ("h5_dir", "station_csv"):
            if cfg_get(portable, f"data.{key}"):
                trained.setdefault("data", {})[key] = portable["data"][key]
    config = reader_config(trained, args.h5_dir, args.station_csv)

    model = build_model(config)
    # Materialize the lazy backbone before loading checkpoint tensors.
    c_in = (DAILY_INPUT_FEATURE_COUNT + int(cfg_get(config, "data.kg_embed_dim", 2))
            + int(cfg_get(config, "data.clcz_embed_dim", 16)))
    model._ensure_backbone(c_in=c_in, device=torch.device("cpu"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    losses = JointLoss.from_config(config)
    return Run(name=directory.name, directory=directory, checkpoint=path, config=config,
               model=model.to(device).eval(), epoch=int(payload.get("epoch", -1)),
               sha256=_sha256(path), physics_loss=losses.physics_loss.to(device),
               landsat_operator=losses.satellite_loss.time_operator.to(device))




def station_pool(config: dict, split: str) -> pd.DataFrame:
    """Rows of the station CSV in the pool; ``is_real`` set by the virtual prefix."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    parts = ("train", "val") if split == "all" else (split,)
    return pd.concat([build_station_df_from_config(config, part) for part in parts], ignore_index=True)


def read_station_ids(path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip()]


def select_stations(config: dict, wanted, limit: int, shard: str, split: str = "val") -> list[str]:
    table = station_pool(config, split)
    ids = table[_resolve_station_id_col(table)].astype(str).tolist()
    if wanted:
        unknown = sorted(set(map(str, wanted)) - set(ids))
        if unknown:
            raise KeyError(f"Not {split} stations in {cfg_get(config, 'data.station_csv')}: {unknown}")
        ids = [sid for sid in ids if sid in set(map(str, wanted))]
    h5_dir = Path(cfg_get(config, "data.h5_dir"))
    present = set(os.listdir(h5_dir))
    missing = [sid for sid in ids if f"{sid}.h5" not in present]
    if missing:
        LOGGER.warning("%d %s stations have no cube in %s (first: %s)",
                       len(missing), split, h5_dir, missing[:5])
    ids = [sid for sid in ids if f"{sid}.h5" in present]
    if limit:
        ids = ids[:limit]
    k, n = (int(part) for part in shard.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard must be K/N with 0 <= K < N, got {shard}")
    return ids[k::n]


def read_station(dataset: MUPNetDataset, sid: str) -> dict:
    """One whole-axis read through the training reader, on the host."""
    days = station_days(dataset.h5_dir / f"{sid}.h5", dataset.layout)
    if not np.all(np.diff(days) == 1):
        raise ValueError(f"{sid}: nonconsecutive day axis")
    # Read the full station time axis rather than a single dataset window.
    sample = dataset._read_sample(sid, 0, len(days))
    sample["day_index"] = np.asarray(days, dtype=np.int64)
    return sample


def _to_device(value, device):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True) if torch.is_tensor(value) else value


def window_plan(n_days: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign days to non-overlapping windows; cover a short tail with a final full window."""
    if n_days < length:
        raise ValueError(f"{n_days} days cannot hold one {length}-day window")
    starts = list(range(0, n_days - length + 1, length))
    if starts[-1] + length < n_days:
        starts.append(n_days - length)
    source = np.empty((n_days, 2), dtype=np.int64)
    for window, start in enumerate(starts):
        source[start:start + length] = np.stack(
            [np.full(length, window), np.arange(length)], axis=1)
    return np.asarray(starts, dtype=np.int64), source




def predict(run: Run, day: dict, starts: np.ndarray, source: np.ndarray, length: int,
            batch_windows: int, amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Ta, Ts as [D, 3, 5, 5] from non-overlapping windows cut out of the day batch."""
    device = day["daily_input_features"].device
    offsets = torch.arange(length, device=device)
    ta_windows, ts_windows = [], []
    amp_dtype = (torch.bfloat16 if str(cfg_get(run.config, "training.amp_dtype", "bfloat16")).lower()
                 == "bfloat16" else torch.float16)
    for first in range(0, len(starts), batch_windows):
        chunk = torch.from_numpy(starts[first:first + batch_windows]).to(device)
        index = chunk[:, None] + offsets[None, :]                                    # [B, L]
        batch = {key: day[key][index] for key in (
            "daily_input_features", "daily_clcz_class", "ta_reference_daily",
            "ta_reference_valid", "ts_reference_daily", "ts_reference_valid")}
        batch["kg_class"] = day["kg_class"].expand(len(chunk), *day["kg_class"].shape)
        batch["phys_daily"] = {key: value[index] for key, value in day["phys_daily"].items()}
        validate_daily_batch(batch)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp and device.type == "cuda"):
            outputs = run.model(batch)
        ta_windows.append(outputs["Ta_daily"].float())
        ts_windows.append(outputs["Ts_daily"].float())
    window = torch.from_numpy(source[:, 0]).to(device)
    position = torch.from_numpy(source[:, 1]).to(device)
    return torch.cat(ta_windows)[window, position], torch.cat(ts_windows)[window, position]


def day_night_and_closure(run: Run, ta: torch.Tensor, ts: torch.Tensor,
                          physics: dict) -> dict[str, torch.Tensor]:
    """Compute stitched-series day/night and closure fields [D, 5, 5], NaN where undefined."""
    loss = run.physics_loss
    validate_daynight_physics_batch(ta, ts, physics)
    dtype = torch.float32
    hours = {key: physics[key].to(dtype) for key in (
        "era5_skt_min_hour", "era5_skt_max_hour", "era5_t2m_min_hour", "era5_t2m_max_hour",
        "sunrise_hour", "sunset_hour")}
    ts_day, ts_night, ts_day_ok, ts_night_ok = loss.time_operator.window_means(
        ts, hours["era5_skt_min_hour"], hours["era5_skt_max_hour"],
        hours["sunrise_hour"], hours["sunset_hour"], n_samples=loss.curve_samples)
    ta_day, ta_night, ta_day_ok, ta_night_ok = loss.time_operator.window_means(
        ta, hours["era5_t2m_min_hour"], hours["era5_t2m_max_hour"],
        hours["sunrise_hour"], hours["sunset_hour"], n_samples=loss.curve_samples)
    if loss.air_source == "era5":
        air = torch.stack([physics[f"t2m_{stat}"] for stat in ("min", "mean", "max")], dim=2).to(dtype)
        air_day, air_night, _, _ = loss.time_operator.window_means(
            air, hours["era5_t2m_min_hour"], hours["era5_t2m_max_hour"],
            hours["sunrise_hour"], hours["sunset_hour"], n_samples=loss.curve_samples)
    else:
        air_day, air_night = ta_day, ta_night

    def masked(values, ok):
        return torch.where(ok[..., None, None], values, torch.full_like(values, float("nan")))

    out = {"Ta_day": masked(ta_day, ta_day_ok), "Ta_night": masked(ta_night, ta_night_ok),
           "Ts_day": masked(ts_day, ts_day_ok), "Ts_night": masked(ts_night, ts_night_ok)}

    # Use the loss support; day d requires its predecessor for night and storage.
    support = physics["pair_valid"] & ts_night_ok[:, 1:, None, None]
    for window, air_window, previous in (
        ("night", air_night[:, 1:], ts_day[:, :-1]),
        ("day", air_day[:, 1:], ts_night[:, 1:]),
    ):
        closure = torch.full_like(ts_day, float("nan"))
        if bool(support.any()):
            c = loss._gather_window(physics, window, support, dtype)
            target, converged = solve_closure_temperature(
                air_window[support], previous[support], c,
                steps=loss.newton_steps, bracket=loss.bracket_k)
            tail = closure[:, 1:]
            tail[support] = torch.where(converged, target, torch.full_like(target, float("nan")))
        out[f"Ts_closure_{window}"] = closure
    return {key: value[0] for key, value in out.items()}


def landsat_channels(run: Run, ts: torch.Tensor, host: dict,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Model Ts at the overpass hour, and the observed LST patch; NaN off scene days."""
    hour = host["lst_hour"].to(device=device, dtype=torch.float32)
    scene = torch.isfinite(hour)
    prediction, diagnostics = run.landsat_operator(
        ts, hour[None],
        host["timing_daily"]["era5_skt_min_hour"].to(device).float()[None],
        host["timing_daily"]["era5_skt_max_hour"].to(device).float()[None])
    ok = (diagnostics["timing_valid"][0] & scene)[:, None, None]
    at_overpass = torch.where(ok, prediction[0].float(), torch.full_like(prediction[0], float("nan")))
    observed = host["lst_obs"][:, _PATCH, _PATCH].to(device=device, dtype=torch.float32)
    observed = torch.where(scene[:, None, None], observed, torch.full_like(observed, float("nan")))
    return at_overpass, observed


def reference_mask(host: dict, prefix: str, device: torch.device) -> torch.Tensor:
    """[D, 5, 5]: ERA5 reference finite at the cell and at the tile centre."""
    valid = host[f"{prefix}_reference_valid"].to(device).bool()
    return valid[:, _PATCH, _PATCH] & valid[:, _TILE_CENTRE, _TILE_CENTRE][:, None, None]


def station_outputs(run: Run, host: dict, day: dict, physics: dict, plan, length: int,
                    args) -> tuple[np.ndarray, str | None]:
    """[D, 14, 5, 5] float32 for one run and one station, plus a closure error if any."""
    starts, source = plan
    device = day["daily_input_features"].device
    ta, ts = predict(run, day, starts, source, length, args.batch_windows, args.amp)
    fields = {"Ta_min": ta[:, 0], "Ta_mean": ta[:, 1], "Ta_max": ta[:, 2],
              "Ts_min": ts[:, 0], "Ts_mean": ts[:, 1], "Ts_max": ts[:, 2]}
    closure_error = None
    try:
        fields.update(day_night_and_closure(run, ta[None], ts[None], physics))
    except ValueError as error:
        # Keep temperature outputs if physics diagnostics fail; record the error.
        closure_error = f"{type(error).__name__}: {error}"
        LOGGER.warning("%s: day/night and closure channels left NaN (%s)", run.name, closure_error)
    fields["Ts_landsat_time"], fields["Landsat_LST"] = landsat_channels(run, ts[None], host, device)

    nan = torch.full_like(fields["Ta_min"], float("nan"))
    air_ok, surface_ok = reference_mask(host, "ta", device), reference_mask(host, "ts", device)
    for name in CHANNELS:
        if name == "Landsat_LST":
            continue
        ok = (air_ok if name.startswith("Ta_") else
              air_ok & surface_ok if name.startswith("Ts_closure") else surface_ok)
        fields[name] = torch.where(ok, fields.get(name, nan), nan)
    stack = torch.stack([fields.get(name, nan) for name in CHANNELS], dim=1)       # [D, 14, 5, 5]
    return stack.cpu().numpy().astype(np.float32), closure_error


def save_npy(path: Path, array: np.ndarray) -> None:
    """Write through a temporary name so an interrupted job leaves no partial file."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:          # Use a handle so np.save does not append a suffix.
        np.save(stream, array)
    os.replace(temporary, path)


def write_manifest(folder: Path, run: Run, n_days: int, length: int, args) -> None:
    manifest = {
        "run": run.name,
        "run_directory": str(run.directory),
        "checkpoint": str(run.checkpoint),
        "checkpoint_epoch": run.epoch,
        "checkpoint_sha256": run.sha256,
        "array": "float32 [day, channel, row, col]",
        "shape": [n_days, len(CHANNELS), CENTER_SIZE, CENTER_SIZE],
        "channels": list(CHANNELS),
        "unit": "K",
        "nodata": "NaN",
        "day_axis": f"day index 0 = {FIRST_DATE}, consecutive local (UTC+8) days",
        "station_pixel": [CENTER_SIZE // 2, CENTER_SIZE // 2],
        "patch": f"centre {CENTER_SIZE}x{CENTER_SIZE} of the {TILE_SIZE}x{TILE_SIZE} cube",
        "windows": f"non-overlapping {length}-day windows (stride {length})",
        "precision": "autocast " + str(cfg_get(run.config, "training.amp_dtype", "bfloat16"))
                     if args.amp else "float32",
        "night_window": "sunset(d-1) .. sunrise(d)",
        "closure": {"air_source": run.physics_loss.air_source,
                    "newton_steps": run.physics_loss.newton_steps,
                    "bracket_k": run.physics_loss.bracket_k,
                    "curve_samples": run.physics_loss.curve_samples,
                    "priors_version": cfg_get(run.config, "physics.priors.version"),
                    "support": "pair_valid of the physics loss, all years, Newton converged"},
        "landsat": "scene day = finite lst_hour; Landsat_LST keeps cloud NaN, "
                   "Ts_landsat_time fills all pixels of a scene day",
        "stations": ("Split == val" if args.split == "val" else
                     "Split in (train, val)" if args.split == "all" else "Split == train")
                    + " in " + str(cfg_get(run.config, "data.station_csv"))
                    + (f", restricted to {args.stations_file}" if args.stations_file else ""),
        "h5_dir": str(cfg_get(run.config, "data.h5_dir")),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = folder / "_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, folder / "_manifest.json")




def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_grad_enabled(False)
    # Disable TF32 so float32 predictions are stable across batch sizes.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    runs = [load_run(directory, args.checkpoint, device, args) for directory in expand_runs(args.runs)]
    signatures = {run.name: reader_signature(run.config) for run in runs}
    if len(set(signatures.values())) != 1:
        raise ValueError("Runs disagree on data, normalization or physics settings, so one read "
                         f"cannot serve them all; run them separately: {sorted(signatures)}")
    config = runs[0].config
    length = window_days(config)
    n_days = int(cfg_get(config, "data.total_daily_steps", 2191))
    wanted = list(args.stations or []) + (read_station_ids(args.stations_file) if args.stations_file else [])
    stations = select_stations(config, wanted, args.limit, args.shard, args.split)
    LOGGER.info("%d runs x %d %s stations (shard %s) on %s, output %s",
                len(runs), len(stations), args.split, args.shard, device, args.out)

    out_root = resolve_project_path(args.out)
    folders = {}
    for run in runs:
        folders[run.name] = out_root / run.name
        folders[run.name].mkdir(parents=True, exist_ok=True)
        write_manifest(folders[run.name], run, n_days, length, args)
        LOGGER.info("%s: %s (epoch %d)", run.name, run.checkpoint, run.epoch)

    existing = {run.name: set(os.listdir(folders[run.name])) for run in runs}
    work = []
    for sid in stations:
        todo = [run for run in runs if args.overwrite or f"{sid}.npy" not in existing[run.name]]
        if todo:
            work.append((sid, todo))
    LOGGER.info("%d stations to run, %d already complete", len(work), len(stations) - len(work))
    if not work:
        return

    dataset = MUPNetDataset(config, READER_SPLIT, station_pool(config, args.split),
                            [(work[0][0], 0, length)])
    failed: dict[str, str] = {}
    closure_failed: dict[str, dict[str, str]] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=1) as reader:
        pending = reader.submit(read_station, dataset, work[0][0])
        for number, (sid, todo) in enumerate(work, 1):
            current = pending
            pending = reader.submit(read_station, dataset, work[number][0]) if number < len(work) else None
            try:
                host = current.result()
                days = host["day_index"]
                if days.min() < 0 or days.max() >= n_days:
                    raise ValueError(f"day_index {days.min()}..{days.max()} outside 0..{n_days - 1}")
                day = _to_device({key: host[key] for key in (
                    "daily_input_features", "daily_clcz_class", "kg_class", "ta_reference_daily",
                    "ta_reference_valid", "ts_reference_daily", "ts_reference_valid", "phys_daily")},
                    device)
                physics = {key: value[None] for key, value in _to_device(host["physics"], device).items()}
                plan = window_plan(len(days), length)
                for run in todo:
                    stack, closure_error = station_outputs(run, host, day, physics, plan, length, args)
                    full = np.full((n_days, len(CHANNELS), CENTER_SIZE, CENTER_SIZE), np.nan, np.float32)
                    full[days] = stack
                    save_npy(folders[run.name] / f"{sid}.npy", full)
                    if closure_error:
                        closure_failed.setdefault(run.name, {})[sid] = closure_error
            except Exception as error:
                LOGGER.exception("%s failed", sid)
                failed[sid] = f"{type(error).__name__}: {error}"
            if number % 10 == 0 or number == len(work):
                rate = (time.time() - started) / number
                LOGGER.info("%d/%d stations (%d failed), %.1f s/station, ETA %.0f min",
                            number, len(work), len(failed), rate, rate * (len(work) - number) / 60)

    if failed or closure_failed:
        k, n = args.shard.split("/")
        report = out_root / f"_failures_shard{k}of{n}.json"
        report.write_text(json.dumps({"failed_stations": failed, "closure_failed": closure_failed},
                                     indent=2), encoding="utf-8")
        LOGGER.warning("%d stations failed, closure failed for %d station-runs; see %s",
                       len(failed), sum(map(len, closure_failed.values())), report)
    LOGGER.info("done in %.1f min", (time.time() - started) / 60)


if __name__ == "__main__":
    main()

"""Aggregate station, Landsat, and physics metrics from saved predictions.

Read [D, 14, 5, 5] station arrays with their DataCubes. Write monthly, annual,
training-period, and test-period metrics to CSV and Excel, including ERA5 and
observed-temperature references. Observation metrics use common finite support
across runs; physics uses each source's converged support.

"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import cfg_get, load_config, resolve_project_path
from ..contracts import CENTER_SIZE, HALO_PIXELS
from ..prepare_data.dataset import MUPNetDataset
from ..prepare_data.indexing import (
    _resolve_station_id_col, build_station_df_from_config, window_days,
)
from ..train.losses.joint_loss import JointLoss
from ..train.losses.physical_loss import solve_closure_temperature, window_residual
from .val_infer import CH, CHANNELS, read_station, read_station_ids, reader_config

LOGGER = logging.getLogger("collate_metrics")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = str(resolve_project_path("outputs/predictions"))
# Use a resolved config with the same reader settings and physics priors.
DEFAULT_CONFIG = PACKAGE_ROOT / "experiments" / "abl_final" / "05_E_physics_no_absolute" / "resolved_config.yaml"
DEFAULT_OUT = resolve_project_path("outputs/metrics")

KELVIN = 273.15
FIRST_DAY = date(2018, 1, 1)
EARTH_KM = 6371.0088
DISTANCE_EDGES_KM = (0.0, 2.0, 5.0, 10.0, 25.0, 50.0)
STATION_VARIABLES = ("Ta_min", "Ta_mean", "Ta_max", "Ts_min", "Ts_mean", "Ts_max")
STATION_AVERAGES = {"Ta_output_mean": STATION_VARIABLES[:3],
                    "Ts_output_mean": STATION_VARIABLES[3:],
                    "output_mean": STATION_VARIABLES}
OBSERVATION_KEYS = ("tem_min", "tem_mean", "tem_max", "gst_min", "gst_mean", "gst_max")
CENTRE = CENTER_SIZE // 2
_PATCH = slice(HALO_PIXELS, HALO_PIXELS + CENTER_SIZE)
_CENTRE = slice(CENTRE, CENTRE + 1)
# Match the satellite loss minimum patch support.
MIN_PATCH_PIXELS = 4
# Flat observed patches have no spatial pattern.
FLAT_NORM_K = 1.0e-6
WINDOWS = ("day", "night")
SCOPES = ("patch_5x5", "station_matched")
ERA5 = "ERA5"
OBSERVED = "Observed"
TIMING_KEYS = ("era5_skt_min_hour", "era5_skt_max_hour", "era5_t2m_min_hour",
               "era5_t2m_max_hour", "sunrise_hour", "sunset_hour")
COEFFICIENT_KEYS = tuple(f"{key}_{window}" for window in WINDOWS
                         for key in ("A", "E", "G_H", "G_E", "G_C")) + ("pressure", "q_air")

# Store additive monthly sums for aggregation at all time scales.
POINT_STATS = ("n", "e", "abs_e", "sq_e", "o", "sq_o")
PAIR_STATS = ("n", "x", "y", "xx", "yy", "xy", "sq_e", "abs_e")
PATCH_STATS = ("patches", "pae", "pae_flat", "pd", "pixels", "neighbours")
PHYSICS_STATS = ("days", "abs_dT", "dT", "abs_res", "res", "pixels", "attempted", "converged")

UNITS = {
    "RMSE": "K", "MAE": "K", "bias": "K", "R2": "1", "r": "1", "slope": "1",
    "sd_obs": "K", "sd_pred": "K", "sd_ratio": "1",
    "PAE": "K", "PAE_flat": "K", "PAE_skill": "1", "PD": "1",
    "MAD": "K", "mean_dT": "K", "abs_residual": "W m-2", "residual": "W m-2",
    "effective_slope": "W m-2 K-1", "newton_convergence": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pred-dir", nargs="+", default=[DEFAULT_PRED_DIR],
                        help="val_infer output root(s) holding one folder per run; a run may be split "
                             "over several roots, e.g. train and val outputs (default %(default)s)")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Only these run folder names (default: every run found in --pred-dir)")
    parser.add_argument("--cube-dir", default=None,
                        help="Station cubes, repacked or original (default: data.h5_dir of --config)")
    parser.add_argument("--station-csv", default=None,
                        help="Station table with Lat/Lon (default: data.station_csv of --config)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="A run's resolved_config.yaml: priors and reader settings (default %(default)s)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output folder (default %(default)s)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Station-reading processes; 1 runs in this process (default %(default)s)")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="Only these stations of --pred-dir (default: every station all runs have)")
    parser.add_argument("--stations-file", default=None,
                        help="Text file of station IDs, one per line ('#' starts a comment); "
                             "added to --stations")
    parser.add_argument("--limit", type=int, default=0, help="First N stations only; 0 = all")
    return parser.parse_args()




def month_layout(n_days: int) -> tuple[list[str], list[int], np.ndarray]:
    """Month labels, their years, and the first day index of each month."""
    dates = [FIRST_DAY + timedelta(days=d) for d in range(n_days)]
    ids = np.array([(d.year - FIRST_DAY.year) * 12 + d.month - 1 for d in dates])
    starts = np.flatnonzero(np.r_[True, np.diff(ids) != 0])
    return ([f"{dates[s].year}-{dates[s].month:02d}" for s in starts],
            [dates[s].year for s in starts], starts)


def periods(labels, years, test_year: int) -> list[tuple[str, str, list[int]]]:
    """(scale, period label, month indices) for every reported period."""
    out = [("monthly", label, [k]) for k, label in enumerate(labels)]
    for year in sorted(set(years)):
        out.append(("annual", str(year), [k for k, y in enumerate(years) if y == year]))
    seen = [k for k, y in enumerate(years) if y < test_year]
    if seen:
        label = f"{years[seen[0]]}-{years[seen[-1]]}"
        out.append((label, label, seen))
    test = [k for k, y in enumerate(years) if y == test_year]
    if test:
        out.append((str(test_year), str(test_year), test))
    return out


def monthly(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Sum the last (day) axis into calendar months."""
    return np.add.reduceat(values, starts, axis=-1)



_CTX: dict = {}


def init_worker(config: dict, paths: dict[str, dict[str, str]], experiments: list[str], starts: np.ndarray,
                n_days: int, threads: int = 1) -> None:
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)
    losses = JointLoss.from_config(config)
    _CTX.update(config=config, paths=paths, experiments=list(experiments),
                starts=starts, n_days=n_days, window=window_days(config), dataset=None,
                station_df=build_station_df_from_config(config, "val"),
                physics_loss=losses.physics_loss, landsat=losses.satellite_loss.time_operator)


def process_station(sid: str) -> dict:
    try:
        return _process_station(sid, _CTX)
    except Exception as error:
        return {"sid": sid, "error": f"{type(error).__name__}: {error}"}


def _process_station(sid: str, ctx: dict) -> dict:
    if ctx["dataset"] is None:
        ctx["dataset"] = MUPNetDataset(ctx["config"], "val", ctx["station_df"],
                                       [(sid, 0, ctx["window"])])
    sample = read_station(ctx["dataset"], sid)
    n_days, starts = ctx["n_days"], ctx["starts"]
    if len(sample["day_index"]) != n_days or int(sample["day_index"][0]) != 0:
        raise ValueError(f"cube day axis {sample['day_index'][0]}..{sample['day_index'][-1]} "
                         f"is not 0..{n_days - 1}")
    preds = np.stack([np.load(ctx["paths"][name][sid]) for name in ctx["experiments"]])
    if preds.shape[1:] != (n_days, len(CHANNELS), CENTER_SIZE, CENTER_SIZE):
        raise ValueError(f"prediction shape {preds.shape[1:]}")

    era5 = {}
    for prefix in ("ta", "ts"):
        reference = sample[f"{prefix}_reference_daily"].numpy()[:, :, _PATCH, _PATCH]
        valid = sample[f"{prefix}_reference_valid"].numpy()[:, _PATCH, _PATCH]
        era5[prefix] = np.where(valid[:, None], reference, np.nan).astype(np.float32)   # [D, 3, 5, 5]
    observed = np.stack([(sample["daily_labels"] if key.startswith("tem") else sample["daily_gst_labels"])
                         [key].numpy() for key in OBSERVATION_KEYS], axis=1).astype(np.float32) + KELVIN

    centre = np.concatenate([
        preds[:, :, [CH[name] for name in STATION_VARIABLES], CENTRE, CENTRE],
        np.concatenate([era5["ta"][:, :, CENTRE, CENTRE], era5["ts"][:, :, CENTRE, CENTRE]], axis=1)[None],
    ]).astype(np.float32)                                                             # [S, D, 6]

    hour = sample["lst_hour"].float()
    timing = sample["timing_daily"]
    prediction, diagnostics = ctx["landsat"](
        torch.from_numpy(era5["ts"])[None], hour[None],
        timing["era5_skt_min_hour"].float()[None], timing["era5_skt_max_hour"].float()[None])
    scene = (diagnostics["timing_valid"][0] & torch.isfinite(hour))[:, None, None]
    era5_overpass = torch.where(scene, prediction[0].float(), torch.tensor(float("nan"))).numpy()
    landsat_point, landsat_patch, landsat_days = landsat_sums(
        preds[0, :, CH["Landsat_LST"]],
        np.concatenate([preds[:, :, CH["Ts_landsat_time"]], era5_overpass[None]]), starts)

    physics, physics_days, guard = physics_sums(sample, preds, era5, observed, ctx)
    return {"sid": sid, "centre": centre, "observed": observed,
            "landsat_point": landsat_point, "landsat_patch": landsat_patch, "landsat_days": landsat_days,
            "physics": physics, "physics_days": physics_days, "guard": guard}


def landsat_sums(observed: np.ndarray, predicted: np.ndarray, starts: np.ndarray):
    """Aggregate [D, 5, 5] observations and [S, D, 5, 5] predictions on common finite support.

    Return point sums [S, 6, M], patch sums [S, 6, M], and patch presence [M].
    """
    valid = np.isfinite(observed) & np.isfinite(predicted).all(axis=0)                 # [D, 5, 5]
    o = np.where(valid, observed, 0.0).astype(np.float64)
    p = np.where(valid[None], predicted, 0.0).astype(np.float64)
    error = p - o[None]
    n = valid.sum(axis=(1, 2)).astype(np.float64)
    shape = error.shape[:2]
    point = np.stack([np.broadcast_to(n, shape), error.sum(axis=(2, 3)),
                      np.abs(error).sum(axis=(2, 3)), np.square(error).sum(axis=(2, 3)),
                      np.broadcast_to(o.sum(axis=(1, 2)), shape),
                      np.broadcast_to(np.square(o).sum(axis=(1, 2)), shape)], axis=1)  # [S, 6, D]

    count = np.maximum(n, 1.0)
    o_anomaly = (o - (o.sum(axis=(1, 2)) / count)[:, None, None]) * valid
    p_anomaly = (p - (p.sum(axis=(2, 3)) / count)[:, :, None, None]) * valid[None]
    o_norm = np.sqrt(np.square(o_anomaly).sum(axis=(1, 2)))
    p_norm = np.sqrt(np.square(p_anomaly).sum(axis=(2, 3)))
    usable = (n >= MIN_PATCH_PIXELS) & (o_norm > FLAT_NORM_K)
    pae = np.abs(p_anomaly - o_anomaly[None]).sum(axis=(2, 3)) / count
    pae_flat = np.abs(o_anomaly).sum(axis=(1, 2)) / count
    denominator = p_norm * o_norm
    # A flat prediction has cosine similarity 0 and PD 1.
    cosine = np.where(denominator > 1.0e-12,
                      (p_anomaly * o_anomaly[None]).sum(axis=(2, 3)) / np.maximum(denominator, 1.0e-12), 0.0)
    pd = 1.0 - np.clip(cosine, -1.0, 1.0)
    neighbours = ((valid[:, :, 1:] & valid[:, :, :-1]).sum(axis=(1, 2))
                  + (valid[:, 1:, :] & valid[:, :-1, :]).sum(axis=(1, 2)))
    keep = usable.astype(np.float64)
    patch = np.stack([np.broadcast_to(keep, shape), pae * keep, np.broadcast_to(pae_flat * keep, shape),
                      pd * keep, np.broadcast_to(n * keep, shape),
                      np.broadcast_to(neighbours * keep, shape)], axis=1)
    return monthly(point, starts), monthly(patch, starts), monthly(usable.astype(np.int64), starts) > 0


def closure_fields(loss, air_day, air_night, surface_day, surface_night, coefficients, support):
    """Solve closure on [D-1, H, W] support; temperatures have shape [D, H, W].

    Night(d) follows day(d-1); day(d) follows night(d). Day 0 is excluded.
    """
    out = {}
    for window, air, surface, previous in (
        ("night", air_night[1:], surface_night[1:], surface_day[:-1]),
        ("day", air_day[1:], surface_day[1:], surface_night[1:]),
    ):
        attempted = support & torch.isfinite(air) & torch.isfinite(surface) & torch.isfinite(previous)
        tstar = torch.full_like(surface, float("nan"))
        residual = torch.full_like(surface, float("nan"))
        converged = torch.zeros_like(attempted)
        if bool(attempted.any()):
            c = loss._gather_window(coefficients, window, attempted[None], torch.float32)
            target, ok = solve_closure_temperature(air[attempted], previous[attempted], c,
                                                   steps=loss.newton_steps, bracket=loss.bracket_k)
            balance = window_residual(surface[attempted], air[attempted], previous[attempted], c)
            nan = torch.full_like(target, float("nan"))
            tstar[attempted] = torch.where(ok, target, nan)
            residual[attempted] = torch.where(ok, balance, nan)
            converged[attempted] = ok

        def with_day0(values, fill):
            head = torch.full_like(values[:1], fill) if values.dtype.is_floating_point else torch.zeros_like(values[:1])
            return torch.cat([head, values]).numpy()

        out[window] = {"tstar": with_day0(tstar, float("nan")),
                       "dT": with_day0(surface - tstar, float("nan")),
                       "residual": with_day0(residual, float("nan")),
                       "attempted": with_day0(attempted, 0), "converged": with_day0(converged, 0)}
    return out


def physics_day_stats(fields: dict, valid: np.ndarray, attempted: np.ndarray,
                      converged: np.ndarray, min_pixels: int):
    """Return daily physics sums and support; require min_pixels after convergence."""
    n = valid.sum(axis=(1, 2)).astype(np.float64)
    usable = n >= min_pixels
    count = np.maximum(n, 1.0)

    def day_mean(values):
        return np.where(valid, values, 0.0).astype(np.float64).sum(axis=(1, 2)) / count

    keep = usable.astype(np.float64)
    stats = np.stack([keep, day_mean(np.abs(fields["dT"])) * keep, day_mean(fields["dT"]) * keep,
                      day_mean(np.abs(fields["residual"])) * keep, day_mean(fields["residual"]) * keep,
                      n * keep, attempted.sum(axis=(1, 2)).astype(np.float64),
                      converged.sum(axis=(1, 2)).astype(np.float64)])
    return stats, usable


def physics_sums(sample, preds, era5, observed, ctx):
    """Monthly physics sums [P, window, scope, stat, M], presence, and the T* guard."""
    loss, starts = ctx["physics_loss"], ctx["starts"]
    physics = sample["physics"]
    hours = {key: physics[key].float()[None] for key in TIMING_KEYS}
    coefficients = {key: physics[key].float()[None] for key in COEFFICIENT_KEYS}

    def windows(triplet, var):
        t = torch.from_numpy(np.ascontiguousarray(triplet, dtype=np.float32))[None]
        day, night, day_ok, night_ok = loss.time_operator.window_means(
            t, hours[f"era5_{var}_min_hour"], hours[f"era5_{var}_max_hour"],
            hours["sunrise_hour"], hours["sunset_hour"], n_samples=loss.curve_samples)
        nan = torch.tensor(float("nan"))
        return (torch.where(day_ok[..., None, None], day, nan)[0],
                torch.where(night_ok[..., None, None], night, nan)[0], night_ok[0])

    era5_ta_day, era5_ta_night, _ = windows(era5["ta"], "t2m")
    era5_ts_day, era5_ts_night, night_ok = windows(era5["ts"], "skt")
    support = physics["pair_valid"].bool() & night_ok[1:, None, None]

    sources, guard = [], {"max_tstar_diff_K": 0.0, "validity_mismatch": 0, "compared": 0}
    for index in range(preds.shape[0]):
        p = torch.from_numpy(np.ascontiguousarray(preds[index]))
        fields = closure_fields(loss, p[:, CH["Ta_day"]], p[:, CH["Ta_night"]],
                                p[:, CH["Ts_day"]], p[:, CH["Ts_night"]], coefficients, support)
        for window in WINDOWS:
            saved = preds[index, :, CH[f"Ts_closure_{window}"]]
            mine = fields[window]["tstar"]
            both = np.isfinite(saved) & np.isfinite(mine)
            if both.any():
                guard["max_tstar_diff_K"] = max(guard["max_tstar_diff_K"],
                                                float(np.abs(saved[both] - mine[both]).max()))
            guard["validity_mismatch"] += int((np.isfinite(saved) != np.isfinite(mine)).sum())
            guard["compared"] += int(both.sum())
        sources.append(fields)
    sources.append(closure_fields(loss, era5_ta_day, era5_ta_night, era5_ts_day, era5_ts_night,
                                  coefficients, support))
    obs_ta_day, obs_ta_night, _ = windows(observed[:, :3, None, None], "t2m")
    obs_ts_day, obs_ts_night, _ = windows(observed[:, 3:, None, None], "skt")
    reference = closure_fields(loss, obs_ta_day, obs_ta_night, obs_ts_day, obs_ts_night,
                               {key: value[..., _CENTRE, _CENTRE] for key, value in coefficients.items()},
                               support[:, _CENTRE, _CENTRE])

    n_sources = len(sources) + 1
    months = len(starts)
    sums = np.zeros((n_sources, len(WINDOWS), len(SCOPES), len(PHYSICS_STATS), months))
    present = np.zeros((n_sources, len(WINDOWS), len(SCOPES), months), dtype=bool)
    for w, window in enumerate(WINDOWS):
        for s, fields in enumerate(sources):
            f = fields[window]
            valid = f["converged"] & np.isfinite(f["dT"])
            stats, usable = physics_day_stats(f, valid, f["attempted"], f["converged"], loss.min_pixels)
            sums[s, w, 0] = monthly(stats, starts)
            present[s, w, 0] = monthly(usable.astype(np.int64), starts) > 0
        # Compare station pixels only where observed and modeled closure both converge.
        centre = [{key: value[:, _CENTRE, _CENTRE] for key, value in fields[window].items()}
                  for fields in sources] + [reference[window]]
        common = np.logical_and.reduce([c["converged"] & np.isfinite(c["dT"]) for c in centre])
        tried = reference[window]["attempted"]
        for s, c in enumerate(centre):
            stats, usable = physics_day_stats(c, common, c["attempted"] & tried,
                                              c["converged"] & tried, 1)
            sums[s, w, 1] = monthly(stats, starts)
            present[s, w, 1] = monthly(usable.astype(np.int64), starts) > 0
    return sums, present, guard




def station_sums(centre, observed, starts):
    """Station sums [S, var, stat, M] and presence [var, station, M] on one common mask."""
    mask = np.isfinite(observed) & np.isfinite(centre).all(axis=0)                     # [St, D, V]
    n_sources, _, _, n_vars = centre.shape
    sums = np.zeros((n_sources, n_vars, len(POINT_STATS), len(starts)))
    presence = np.zeros((n_vars, centre.shape[1], len(starts)), dtype=bool)
    for v in range(n_vars):
        m = mask[:, :, v]
        o = np.where(m, observed[:, :, v], 0.0).astype(np.float64)
        e = np.where(m[None], centre[:, :, :, v].astype(np.float64) - o[None], 0.0)    # [S, St, D]
        per_month = lambda values: monthly(values, starts).sum(axis=-2)
        n = per_month(m.astype(np.float64))
        sums[:, v] = np.stack([np.broadcast_to(n, (n_sources, len(starts))), per_month(e),
                               per_month(np.abs(e)), per_month(np.square(e)),
                               np.broadcast_to(per_month(o), (n_sources, len(starts))),
                               np.broadcast_to(per_month(np.square(o)), (n_sources, len(starts)))], axis=1)
        presence[v] = monthly(m.astype(np.int64), starts) > 0
    return sums, presence


def haversine_km(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    h = (np.sin((phi[:, None] - phi[None]) / 2) ** 2
         + np.cos(phi[:, None]) * np.cos(phi[None]) * np.sin((lam[:, None] - lam[None]) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def pair_sums(centre, observed, distance, starts, chunk=256):
    """Pair sums [S, var, class, stat, M], pair presence per class, and pairs per class."""
    n_sources, n_stations, _, n_vars = centre.shape
    classes = len(DISTANCE_EDGES_KM) - 1
    upper = np.triu_indices(n_stations, 1)
    separation = distance[upper]
    bins = np.searchsorted(DISTANCE_EDGES_KM, separation, side="right") - 1
    inside = (separation >= DISTANCE_EDGES_KM[0]) & (separation < DISTANCE_EDGES_KM[-1])
    mask = np.isfinite(observed) & np.isfinite(centre).all(axis=0)
    sums = np.zeros((n_sources, n_vars, classes, len(PAIR_STATS), len(starts)))
    presence = {}
    for k in range(classes):
        chosen = inside & (bins == k)
        i_all, j_all = upper[0][chosen], upper[1][chosen]
        for v in range(n_vars):
            present = []
            for first in range(0, len(i_all), chunk):
                i, j = i_all[first:first + chunk], j_all[first:first + chunk]
                m = mask[i, :, v] & mask[j, :, v]                                       # [P, D]
                x = np.where(m, observed[i, :, v].astype(np.float64) - observed[j, :, v], 0.0)
                # Orient each pair by its mean observed contrast, independent of the model.
                sign = np.where(x.sum(axis=1) < 0, -1.0, 1.0)
                x *= sign[:, None]
                # Select the variable first to keep advanced indexing from reordering axes.
                source = centre[..., v]
                y = np.where(m[None], (source[:, i].astype(np.float64) - source[:, j])
                             * sign[None, :, None], 0.0)
                e = y - x[None]

                def per_month(values):
                    return monthly(values, starts).sum(axis=-2)

                n = per_month(m.astype(np.float64))
                shape = (n_sources, len(starts))
                sums[:, v, k] += np.stack([np.broadcast_to(n, shape), np.broadcast_to(per_month(x), shape),
                                           per_month(y), np.broadcast_to(per_month(x * x), shape),
                                           per_month(y * y), per_month(y * x[None]),
                                           per_month(e * e), per_month(np.abs(e))], axis=1)
                present.append(monthly(m.astype(np.int64), starts) > 0)
            presence[(v, k)] = (np.concatenate(present) if present
                                else np.zeros((0, len(starts)), dtype=bool))
    counts = [int((inside & (bins == k)).sum()) for k in range(classes)]
    return sums, presence, counts




def point_metrics(s):
    n, e, abs_e, sq_e, o, sq_o = s
    if n <= 0:
        return {"RMSE": np.nan, "MAE": np.nan, "bias": np.nan, "R2": np.nan, "N": 0}
    total = sq_o - o * o / n
    return {"RMSE": math.sqrt(sq_e / n), "MAE": abs_e / n, "bias": e / n,
            "R2": 1.0 - sq_e / total if n > 1 and total > 0 else np.nan, "N": int(n)}


def pair_metrics(s):
    """Compute pair R2 against the constant mean observed difference."""
    n, x, y, xx, yy, xy, sq_e, abs_e = s
    names = ("RMSE", "MAE", "bias", "r", "R2", "slope", "sd_obs", "sd_pred", "sd_ratio")
    if n < 2:
        return {**{name: np.nan for name in names}, "N": int(n)}
    total = xx - x * x / n
    variance = yy - y * y / n
    covariance = xy - x * y / n
    sd_obs, sd_pred = math.sqrt(max(total, 0.0) / n), math.sqrt(max(variance, 0.0) / n)
    return {"RMSE": math.sqrt(sq_e / n), "MAE": abs_e / n, "bias": (y - x) / n,
            "r": covariance / math.sqrt(total * variance) if total > 0 and variance > 0 else np.nan,
            "R2": 1.0 - sq_e / total if total > 0 else np.nan,
            "slope": covariance / total if total > 0 else np.nan,
            "sd_obs": sd_obs, "sd_pred": sd_pred,
            "sd_ratio": sd_pred / sd_obs if sd_obs > 0 else np.nan, "N": int(n)}


def patch_metrics(s):
    patches, pae, pae_flat, pd, pixels, neighbours = s
    if patches <= 0:
        return {"PAE": np.nan, "PAE_flat": np.nan, "PAE_skill": np.nan, "PD": np.nan,
                "N_patches": 0, "N_patch_pixels": 0, "N_neighbours": 0}
    return {"PAE": pae / patches, "PAE_flat": pae_flat / patches,
            "PAE_skill": 1.0 - pae / pae_flat if pae_flat > 0 else np.nan,
            "PD": pd / patches, "N_patches": int(patches), "N_patch_pixels": int(pixels),
            "N_neighbours": int(neighbours)}


def physics_metrics(s):
    days, abs_dT, dT, abs_res, res, pixels, attempted, converged = s
    convergence = converged / attempted if attempted > 0 else np.nan
    if days <= 0:
        return {"MAD": np.nan, "mean_dT": np.nan, "abs_residual": np.nan, "residual": np.nan,
                "effective_slope": np.nan, "newton_convergence": convergence,
                "N_days": 0, "N_pixels": 0, "N_attempted": int(attempted)}
    return {"MAD": abs_dT / days, "mean_dT": dT / days, "abs_residual": abs_res / days,
            "residual": res / days, "effective_slope": abs_res / abs_dT if abs_dT > 0 else np.nan,
            "newton_convergence": convergence, "N_days": int(days), "N_pixels": int(pixels),
            "N_attempted": int(attempted)}




def source_id(name: str) -> str:
    match = re.match(r"(\d+)", name)
    return f"E{int(match.group(1)):02d}" if match else name


# Only per-call manifest fields may differ when pooling one run across roots.
PER_CALL_FIELDS = ("created_utc", "stations")


def discover(pred_dirs: list[Path], config: dict, runs=None):
    """Discover runs and common stations across prediction roots.

    Duplicate stations use the first root; manifests must match outside per-call fields.
    """
    folders: dict[str, list[Path]] = {}
    for root in pred_dirs:
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / "_manifest.json").exists():
                folders.setdefault(p.name, []).append(p)
    if runs:
        unknown = sorted(set(runs) - set(folders))
        if unknown:
            raise FileNotFoundError(f"Runs {unknown} have no folder with _manifest.json in {pred_dirs}")
        folders = {name: folders[name] for name in runs}
    experiments = sorted(folders)
    if not experiments:
        raise FileNotFoundError(f"No run folders with _manifest.json in {pred_dirs}")
    priors = cfg_get(config, "physics.priors.version")
    manifests, paths = {}, {}
    for name in experiments:
        for folder in folders[name]:
            manifest = json.loads((folder / "_manifest.json").read_text(encoding="utf-8"))
            if list(manifest["channels"]) != list(CHANNELS):
                raise ValueError(f"{folder}: channels {manifest['channels']} differ from val_infer.CHANNELS")
            if manifest["closure"]["priors_version"] != priors:
                raise ValueError(f"{folder} was closed with {manifest['closure']['priors_version']}, "
                                 f"--config carries {priors}")
            if name not in manifests:
                manifests[name] = manifest
                continue
            first = manifests[name]
            differ = sorted(key for key in set(first) | set(manifest)
                            if key not in PER_CALL_FIELDS and first.get(key) != manifest.get(key))
            if differ:
                raise ValueError(f"{name}: the manifests in {folders[name][0]} and {folder} differ in "
                                 f"{differ}, so their outputs cannot be pooled")
        found, repeated = {}, 0
        for folder in folders[name]:
            for p in sorted(folder.glob("*.npy")):
                if p.stem in found:
                    repeated += 1
                else:
                    found[p.stem] = str(p)
        if repeated:
            LOGGER.warning("%s: %d stations are in more than one root; the first root's file is used",
                           name, repeated)
        paths[name] = found
    stations = set.intersection(*(set(found) for found in paths.values()))
    uneven = {name: len(set(paths[name]) - stations) for name in experiments}
    if any(uneven.values()):
        LOGGER.warning("Stations missing from some runs are skipped: %s", uneven)
    return experiments, manifests, sorted(stations), paths


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.time()
    trained = load_config(args.config)
    config = reader_config(trained, args.cube_dir, args.station_csv)
    pred_dirs = [resolve_project_path(p) for p in args.pred_dir]
    out = resolve_project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    experiments, manifests, stations, paths = discover(pred_dirs, config, args.runs)
    wanted = list(args.stations or []) + (read_station_ids(args.stations_file) if args.stations_file else [])
    if wanted:
        absent = sorted(set(map(str, wanted)) - set(stations))
        if absent:
            LOGGER.warning("%d requested stations have no output in every run and are skipped: %s",
                           len(absent), absent[:10])
        stations = [sid for sid in stations if sid in set(map(str, wanted))]
    if args.limit:
        stations = stations[:args.limit]
    n_days = int(manifests[experiments[0]]["shape"][0])
    labels, years, starts = month_layout(n_days)
    report_periods = periods(labels, years, int(cfg_get(trained, "data.test_year", 2023)))

    table = pd.read_csv(cfg_get(config, "data.station_csv"))
    id_col = _resolve_station_id_col(table)
    table[id_col] = table[id_col].astype(str)
    table = table.drop_duplicates(id_col).set_index(id_col)
    missing = [sid for sid in stations if sid not in table.index]
    if missing:
        raise KeyError(f"No coordinates in {cfg_get(config, 'data.station_csv')} for {missing[:5]}")
    LOGGER.info("%d runs x %d stations, cubes %s, %d workers", len(experiments), len(stations),
                cfg_get(config, "data.h5_dir"), args.workers)
    MUPNetDataset(config, "val", build_station_df_from_config(config, "val"),
                  [(stations[0], 0, window_days(config))])

    paths = {name: {sid: paths[name][sid] for sid in stations} for name in experiments}
    init_args = (config, paths, experiments, starts, n_days)
    results, failed = {}, {}

    def collect(result, done):
        if "error" in result:
            failed[result["sid"]] = result["error"]
            LOGGER.error("%s failed: %s", result["sid"], result["error"])
        else:
            results[result["sid"]] = result
        if done % 20 == 0 or done == len(stations):
            rate = (time.time() - started) / done
            LOGGER.info("%d/%d stations (%d failed), ETA %.1f min", done, len(stations), len(failed),
                        rate * (len(stations) - done) / 60)

    if args.workers <= 1:
        init_worker(*init_args, threads=max(1, torch.get_num_threads()))
        for done, sid in enumerate(stations, 1):
            collect(process_station(sid), done)
    else:
        # Use spawn to avoid inheriting torch thread pools and open HDF5 state.
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=init_args,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(process_station, sid) for sid in stations]
            for done, future in enumerate(as_completed(futures), 1):
                collect(future.result(), done)
    kept = [sid for sid in stations if sid in results]
    if not kept:
        raise RuntimeError(f"Every station failed; first error: {next(iter(failed.values()), '')}")

    guard = {"max_tstar_diff_K": max(results[s]["guard"]["max_tstar_diff_K"] for s in kept),
             "validity_mismatch": sum(results[s]["guard"]["validity_mismatch"] for s in kept),
             "compared": sum(results[s]["guard"]["compared"] for s in kept)}
    LOGGER.info("T* guard: max |recomputed - saved| %.2e K over %d pixels, %d validity mismatches",
                guard["max_tstar_diff_K"], guard["compared"], guard["validity_mismatch"])
    if guard["max_tstar_diff_K"] > 0.05:
        raise RuntimeError(
            f"Re-solved T* differs from the saved closure by {guard['max_tstar_diff_K']:.3f} K: the cubes "
            "or the priors in --config are not the ones val_infer used, so every physics number would be "
            "measured against a different energy balance. Nothing was written.")

    sources = [source_id(name) for name in experiments] + [ERA5]
    physics_sources = sources + [OBSERVED]
    centre = np.stack([results[s]["centre"] for s in kept], axis=1)                   # [S, St, D, 6]
    observed = np.stack([results[s]["observed"] for s in kept])                        # [St, D, 6]
    station_total, station_presence = station_sums(centre, observed, starts)
    lat = table.loc[kept, "Lat"].to_numpy(float)
    lon = table.loc[kept, "Lon"].to_numpy(float)
    pair_total, pair_presence, pair_counts = pair_sums(centre, observed, haversine_km(lat, lon), starts)
    landsat_point = sum(results[s]["landsat_point"] for s in kept)
    landsat_patch = sum(results[s]["landsat_patch"] for s in kept)
    landsat_presence = np.stack([results[s]["landsat_days"] for s in kept])
    physics_total = sum(results[s]["physics"] for s in kept)
    physics_presence = np.stack([results[s]["physics_days"] for s in kept])           # [St, P, W, Sc, M]

    rows = []

    def emit(group, scale, period, source, variable, metrics, distance="", scope="", units=None):
        for metric, value in metrics.items():
            rows.append({"group": group, "scale": scale, "period": period, "source": source,
                         "variable": variable, "distance_km": distance, "scope": scope,
                         "metric": metric, "value": value,
                         "unit": UNITS.get(metric, "count") if units is None else units})

    classes = [f"{DISTANCE_EDGES_KM[k]:g}-{DISTANCE_EDGES_KM[k + 1]:g}" for k in range(len(pair_counts))]
    for scale, period, months in report_periods:
        channel = {}
        for v, variable in enumerate(STATION_VARIABLES):
            n_stations = int(station_presence[v][:, months].any(axis=1).sum())
            for s, source in enumerate(sources):
                metrics = point_metrics(station_total[s, v][:, months].sum(axis=-1))
                channel[source, variable] = metrics
                emit("station", scale, period, source, variable, {**metrics, "N_stations": n_stations})
            for k, distance in enumerate(classes):
                n_pairs = int(pair_presence[(v, k)][:, months].any(axis=1).sum())
                for s, source in enumerate(sources):
                    metrics = pair_metrics(pair_total[s, v, k][:, months].sum(axis=-1))
                    emit("pair", scale, period, source, variable, {**metrics, "N_pairs": n_pairs},
                         distance=distance)
        # Require every channel before reporting a channel-mean metric.
        for name, members in STATION_AVERAGES.items():
            for source in sources:
                emit("station", scale, period, source, name,
                     {metric: float(np.mean([channel[source, member][metric] for member in members]))
                      for metric in ("RMSE", "MAE")})
        n_stations = int(landsat_presence[:, months].any(axis=1).sum())
        for s, source in enumerate(sources):
            metrics = point_metrics(landsat_point[s][:, months].sum(axis=-1))
            n_pixels = metrics.pop("N")
            metrics = {**metrics, "N_pixels": n_pixels,
                       **patch_metrics(landsat_patch[s][:, months].sum(axis=-1)), "N_stations": n_stations}
            emit("landsat", scale, period, source, "Ts_landsat_time", metrics)
        for w, window in enumerate(WINDOWS):
            for c, scope in enumerate(SCOPES):
                for s, source in enumerate(physics_sources):
                    if source == OBSERVED and scope != "station_matched":
                        continue
                    metrics = physics_metrics(physics_total[s, w, c][:, months].sum(axis=-1))
                    metrics["N_stations"] = int(physics_presence[:, s, w, c][:, months].any(axis=1).sum())
                    emit("physics", scale, period, source, window, metrics, scope=scope)

    long = pd.DataFrame(rows)
    long.to_csv(out / "metrics_long.csv", index=False)
    source_rows = [{"source": source_id(name), "run": name,
                    "checkpoint_epoch": manifests[name].get("checkpoint_epoch"),
                    "checkpoint": manifests[name].get("checkpoint"),
                    "precision": manifests[name].get("precision")} for name in experiments]
    source_rows += [{"source": ERA5, "run": "ERA5-Land daily statistics in the station cubes"},
                    {"source": OBSERVED, "run": "station TEM/GST triplets (physics, station pixel only)"}]
    pd.DataFrame(source_rows).to_csv(out / "sources.csv", index=False)
    write_workbook(long, out / "accuracy_metrics.xlsx", physics_sources, source_rows)

    checks = {"guard": guard, "stations": len(kept), "failed": failed, "runs": experiments,
              "pairs_per_class": dict(zip(classes, pair_counts)),
              "cube_dir": str(cfg_get(config, "data.h5_dir")), "pred_dir": [str(p) for p in pred_dirs],
              "config": str(args.config), "priors_version": cfg_get(config, "physics.priors.version"),
              "minutes": round((time.time() - started) / 60, 1)}
    (out / "_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    LOGGER.info("%d rows for %d stations written to %s in %.1f min", len(long), len(kept), out,
                checks["minutes"])


def write_workbook(long: pd.DataFrame, path: Path, order: list[str], source_rows) -> None:
    """One sheet per group and scale, the sources side by side."""
    keys = ["period", "variable", "distance_km", "scope", "metric"]
    sheets = [("sources", pd.DataFrame(source_rows))]
    for (group, scale), block in long.groupby(["group", "scale"], sort=False):
        wide = block.pivot_table(index=keys, columns="source", values="value",
                                 aggfunc="first", sort=False, dropna=False)
        wide = wide.reindex(columns=[s for s in order if s in wide.columns])
        wide = wide.reset_index()
        unit = block.drop_duplicates("metric").set_index("metric")["unit"]
        wide.insert(wide.columns.get_loc("metric") + 1, "unit", wide["metric"].map(unit))
        wide = wide.loc[:, [c for c in wide.columns if c not in keys or wide[c].astype(str).ne("").any()]]
        sheets.append((f"{group}_{scale}"[:31], wide))
    write_xlsx(path, sheets)


def write_xlsx(path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    """Write worksheets with openpyxl and atomically replace the destination."""
    from itertools import chain
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    temporary = path.with_name(path.name + ".tmp")
    workbook = Workbook(write_only=True)
    try:
        for name, frame in sheets:
            worksheet = workbook.create_sheet(name)
            worksheet.freeze_panes = "A2"
            for values in chain([frame.columns], frame.itertuples(index=False, name=None)):
                cells = []
                for value in values:
                    if value is None or value is pd.NA:
                        value = None
                    elif isinstance(value, (bool, np.bool_)):
                        value = int(value)
                    elif isinstance(value, (int, float, np.integer, np.floating)):
                        value = float(value)
                        if not math.isfinite(value):
                            value = None
                    else:
                        value = str(value)
                    cell = WriteOnlyCell(worksheet, value=value)
                    if isinstance(value, str):
                        cell.data_type = "s"  # Keep labels literal, including strings starting with '='.
                    cells.append(cell)
                worksheet.append(cells)
        workbook.save(temporary)
        os.replace(temporary, path)
    finally:
        workbook.close()
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

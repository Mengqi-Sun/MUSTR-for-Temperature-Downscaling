"""Per-window spatial forcing and fixed prior maps; no persistent caches."""
import numpy as np
import torch
from ..config import physics_priors, PRIOR_WINDOW_KEYS
from .indexing import eligible_days
from ..train.losses.physical_loss import (
    bulk_richardson, coefficients, stability_factor)
from ..main_models.surface_time_operator import SurfaceTimeOperator
from DataCube.schema import center_slice

_TIME_OPERATOR = SurfaceTimeOperator()

SECONDS_PER_HOUR = 3600.0
# Hour anchors use uint8 sentinel 255 and require range checks.
HOUR_SENTINEL = 24.0
# Late daily minima do not reliably phase the diurnal curve.
SKT_MIN_HOUR_MAX = 9.0

DAILY_FIELDS = {
    "sw_down": "ssrd_mean", "lw_down": "strd_mean", "pressure": "sp_mean",
    "wind10": "ws10_mean", "precip": "tp_sum", "snow_fraction_max": "snowc_max",
    "t2m_min": "t2m_min", "skt_min": "skt_min", "albedo": "albedo",
    "t2m_mean": "t2m_mean", "t2m_max": "t2m_max",
    "skt_mean": "skt_mean", "skt_max": "skt_max",
}
# Nighttime shortwave is zero; no night shortwave field is stored.
WINDOW_FIELDS = {
    "sw_down_day": "ssrd_day_mean",
    "lw_down_day": "strd_day_mean", "lw_down_night": "strd_prev_night_mean",
    "wind10_day": "ws10_day_mean", "wind10_night": "ws10_prev_night_mean",
    "api": "api_mean",
}
# Solar geometry is treated as uniform across each tile.
TIMING_FIELDS = (
    "daylight_duration_hours", "prev_night_duration_hours",
    "sunrise_hour", "sunset_hour",
    "era5_skt_min_hour", "era5_skt_max_hour",
    "era5_t2m_min_hour", "era5_t2m_max_hour",
)


def build_window_physics(h5, raw, clcz, days, start, end, config, split):
    """Crop raw spatial daily forcing; compute only this window's prior maps."""
    priors = physics_priors(config)
    if h5.attrs.get("ERA5_Coverage") != "complete_contributing_hours_per_pixel":
        raise ValueError("Physics requires complete-coverage daily ERA5 statistics")
    crop = center_slice()
    classes = clcz[:, crop, crop]
    fields = {target: raw[source][:, crop, crop] for target, source in DAILY_FIELDS.items()}
    windows = {target: raw[source][:, crop, crop] for target, source in WINDOW_FIELDS.items()}
    for key, source in (("q_air", "q_air_mean"), ("rho_air", "rho_air_mean")):
        fields[key] = h5[f"daily/derived/{source}"][start:end, crop, crop]
    timing = {name: np.asarray(h5[f"daily/derived/{name}"][start:end], dtype=np.float32)
              for name in TIMING_FIELDS}

    prior_keys = ("epsilon", "api_half", "stability") + PRIOR_WINDOW_KEYS
    parameters = {key: np.ones(classes.shape, dtype=np.float32) for key in prior_keys}
    dry = np.zeros(classes.shape, dtype=bool)
    for group in priors["groups"].values():
        mask = np.isin(classes, group["classes"])
        for key in prior_keys:
            parameters[key][mask] = group[key]
        dry[mask] = group["dry_paved"]
    reference, floor = priors["wind"]["reference"], priors["wind"]["floor"]
    parameters["ra"] = parameters["ra_ref_day"] * reference / np.maximum(fields["wind10"], floor)
    qf = priors["qf"]
    lookup = np.zeros(19, dtype=np.float32)
    lookup[1:11] = np.asarray(qf["urban_fraction"]) * np.asarray(qf["ah"]) * qf["profile_mean"]
    result = {**fields, **windows, **parameters, "Q_F": lookup[classes], "dry_paved": dry}
    result.update(coefficients_from_primitives(result))
    result.update(window_coefficients(result, timing, reference, floor))
    result.update(timing)

    quality = priors["quality"]
    valid = np.logical_and.reduce([np.isfinite(value) for value in fields.values()])
    valid &= np.logical_and.reduce([np.isfinite(value) for value in windows.values()])
    # Exact-zero albedo is a missing-data sentinel.
    valid &= ((classes > 0) & (fields["albedo"] > 0) & (fields["albedo"] <= 1)
              & (fields["rho_air"] > 0) & (fields["pressure"] > 0) & (fields["q_air"] >= 0) & (fields["q_air"] < 1)
              & (fields["snow_fraction_max"] <= quality["snow_fraction_max"])
              & (fields["t2m_min"] >= quality["freezing_k"]) & (fields["skt_min"] >= quality["freezing_k"]))
    valid &= timing_valid(timing)[:, None, None]
    allowed = eligible_days(config, split, days)
    pairs = valid[1:] & valid[:-1] & (classes[1:] == classes[:-1])
    pairs &= (allowed[1:] & allowed[:-1])[:, None, None]
    wet = fields["precip"][1:]+fields["precip"][:-1] >= quality["wet_pair_mm"]
    pairs &= ~((classes[1:] == 15) & wet)
    result["pair_valid"] = pairs
    result["pair_confidence"] = np.where(wet | np.isin(classes[1:], [17, 18]), quality["low_confidence"], 1.).astype(np.float32)
    result["day_index"] = days.astype(np.int64)
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in result.items()}


def timing_valid(timing):
    """Validate uint8 hour anchors by range and floating-point durations by finiteness."""
    hours = ("era5_skt_min_hour", "era5_skt_max_hour",
             "era5_t2m_min_hour", "era5_t2m_max_hour")
    ok = np.logical_and.reduce([(timing[name] >= 0) & (timing[name] < HOUR_SENTINEL)
                                for name in hours])
    ok &= timing["era5_skt_min_hour"] <= SKT_MIN_HOUR_MAX
    for name in ("daylight_duration_hours", "prev_night_duration_hours"):
        ok &= np.isfinite(timing[name]) & (timing[name] > 1.0)
    ok &= np.isfinite(timing["sunrise_hour"]) & np.isfinite(timing["sunset_hour"])
    return ok


def window_seconds(timing):
    """Return window-centroid separation in seconds for the storage term.

    The interval is the mean of adjacent half-day durations; day 0 lacks a predecessor.
    """
    day = timing["daylight_duration_hours"] * SECONDS_PER_HOUR
    night = timing["prev_night_duration_hours"] * SECONDS_PER_HOUR
    previous_day = np.concatenate(([day[0]], day[:-1]))
    return 0.5 * (night + day), 0.5 * (previous_day + night)


def era5_window_means(triplet, min_hour, max_hour, timing):
    """Compute ERA5 day/night means with the same curve used for predictions."""
    field = torch.from_numpy(np.ascontiguousarray(triplet)).to(torch.float32)[None]
    scalars = [torch.from_numpy(np.asarray(v, dtype=np.float32))[None] for v in
               (min_hour, max_hour, timing["sunrise_hour"], timing["sunset_hour"])]
    day, night, _, _ = _TIME_OPERATOR.window_means(field, *scalars)
    return day[0].numpy(), night[0].numpy()


def window_coefficients(p, timing, reference, floor):
    """Day and night coefficient maps from the same primitives as the 24-hour set."""
    mean_day_seconds, mean_night_seconds = window_seconds(timing)
    # Use ERA5 temperatures for stability so the prediction cannot move its target.
    skt = np.stack([p["skt_min"], p["skt_mean"], p["skt_max"]], axis=1)
    t2m = np.stack([p["t2m_min"], p["t2m_mean"], p["t2m_max"]], axis=1)
    era5_ts = dict(zip(("day", "night"), era5_window_means(
        skt, timing["era5_skt_min_hour"], timing["era5_skt_max_hour"], timing)))
    era5_ta = dict(zip(("day", "night"), era5_window_means(
        t2m, timing["era5_t2m_min_hour"], timing["era5_t2m_max_hour"], timing)))
    out = {}
    for window, sw, lw, wind, ra_ref, rs, capacity, seconds in (
        ("day", p["sw_down_day"], p["lw_down_day"], p["wind10_day"],
         p["ra_ref_day"], p["rs_day"], p["capacity_day"], mean_day_seconds),
        ("night", np.zeros_like(p["sw_down_day"]), p["lw_down_night"], p["wind10_night"],
         p["ra_ref_night"], p["rs_night"], p["capacity_night"], mean_night_seconds),
    ):
        richardson = bulk_richardson(era5_ta[window], era5_ts[window], wind, floor)
        factor = stability_factor(richardson, p["stability"])
        ra = ra_ref * reference / np.maximum(wind, floor) * factor
        # api_half = 0 leaves potential evaporation unchanged.
        beta = np.where(p["api_half"] > 0,
                        p["api"] / np.maximum(p["api"] + p["api_half"], 1.0e-6), 1.0)
        # Use daily mean anthropogenic heat in both windows; no hourly profile is stored.
        window_coeffs = coefficients(
            sw, lw, p["albedo"], p["rho_air"], p["epsilon"], ra, rs,
            capacity, p["Q_F"], p["dry_paved"],
            seconds=seconds[:, None, None].astype(np.float32), beta=beta.astype(np.float32),
        )
        out[f"ra_{window}"] = ra.astype(np.float32)
        out[f"beta_{window}"] = np.broadcast_to(beta, ra.shape).astype(np.float32)
        out[f"richardson_{window}"] = np.broadcast_to(richardson, ra.shape).astype(np.float32)
        out[f"stability_{window}"] = np.broadcast_to(factor, ra.shape).astype(np.float32)
        for key, value in window_coeffs.items():
            out[f"{key}_{window}"] = np.broadcast_to(value, ra.shape).astype(np.float32)
    return out


def coefficients_from_primitives(p):
    """Build legacy 24-hour coefficients from daytime priors without stability correction."""
    return coefficients(p["sw_down"], p["lw_down"], p["albedo"], p["rho_air"],
                        p["epsilon"], p["ra"], p["rs_day"], p["capacity_day"],
                        p["Q_F"], p["dry_paved"])

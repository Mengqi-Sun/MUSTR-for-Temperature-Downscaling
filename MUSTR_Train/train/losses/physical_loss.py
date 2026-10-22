"""Energy-balance equations, validation, and daily physics penalties."""

import math

import numpy as np
import torch
from torch import nn

from ...main_models.surface_time_operator import SurfaceTimeOperator

# Closure requires a specified air temperature source.
AIR_SOURCES = ("predicted_detached", "predicted_attached", "era5")

# Selected windows contribute gradients; both windows retain diagnostics.
PHYSICS_WINDOWS = ("day", "night")

SIGMA = 5.670374419e-8
CP = 1004.0
LV = 2.45e6
RD = 287.05
SECONDS_PER_DAY = 86400.0

GRAVITY = 9.80665
# Match the 10 m wind level and the geometry used to calibrate the priors.
STABILITY_HEIGHT_M = 10.0
# Bulk Richardson bounds differ from the gradient critical value.
RICHARDSON_RANGE = (-2.0, 1.0)
# Bound the resistance multiplier to keep closure temperatures bracketed.
STABILITY_FACTOR_RANGE = (0.25, 6.0)


def _clip(value, low, high):
    """Bounds for either backend; this module is used from numpy and torch."""
    if isinstance(value, torch.Tensor):
        return value.clamp(low, high)
    return np.clip(value, low, high)


def bulk_richardson(t_air, t_surface, wind, floor=0.5):
    """Compute bulk Richardson number from forcing temperatures; positive values are stable."""
    speed = _clip(wind, floor, float("inf"))
    reference = _clip(t_air, 100.0, float("inf"))
    return (GRAVITY * STABILITY_HEIGHT_M * (t_air - t_surface)
            / (reference * speed * speed))


def stability_factor(richardson, strength):
    """Scale neutral aerodynamic resistance using stratification and fitted strength.

    Both branches must remain finite across the clamped Richardson range.
    """
    low, high = RICHARDSON_RANGE
    ri = _clip(richardson, low, high)
    zero = ri - ri
    stable = (1.0 + strength * _clip(ri, 0.0, high)) ** 2
    unstable = (1.0 - strength * _clip(ri, low, 0.0)) ** -0.75
    if isinstance(ri, torch.Tensor):
        factor = torch.where(ri >= zero, stable, unstable)
    else:
        factor = np.where(ri >= zero, stable, unstable)
    return _clip(factor, *STABILITY_FACTOR_RANGE)


def saturation_humidity(temperature: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
    """FAO liquid-water approximation; invalid predictions are errors, not masks."""
    if not bool(torch.isfinite(temperature).all() and torch.isfinite(pressure).all()):
        raise ValueError("Nonfinite temperature/pressure in physics")
    if bool(((temperature <= 35.85) | (pressure <= 0)).any()):
        raise ValueError("Physics requires Kelvin temperature and positive pressure")
    vapor = 610.8 * torch.exp(17.27 * (temperature - 273.15) / (temperature - 35.85))
    if not bool(torch.isfinite(vapor).all()) or bool(((vapor <= 0) | (vapor >= pressure)).any()):
        raise ValueError("Impossible saturation vapor pressure; check predicted Ts and Pa units")
    return 0.622 * vapor / (pressure - 0.378 * vapor)


def saturation_humidity_and_slope(temperature, pressure):
    """Return saturation humidity and its temperature derivative, clamping Newton iterates."""
    safe = temperature.clamp_min(200.0)
    denominator = (safe - 35.85).clamp_min(1.0)
    vapor = 610.8 * torch.exp(17.27 * (safe - 273.15) / denominator)
    vapor = torch.minimum(vapor, 0.9 * pressure)
    gap = (pressure - 0.378 * vapor).clamp_min(1.0)
    q_sat = 0.622 * vapor / gap
    # Differentiate the Tetens expression, then apply the quotient rule.
    d_vapor = vapor * 17.27 * (273.15 - 35.85) / denominator.square()
    d_q = 0.622 * pressure / gap.square()
    return q_sat, d_q * d_vapor


def window_residual(t_surface, t_air, t_previous, c):
    """Half-day energy-balance residual, W/m2.  Zero closes the budget."""
    q_sat, _ = saturation_humidity_and_slope(t_surface, c["pressure"])
    deficit = torch.relu(q_sat - c["q_air"])
    return (c["A"] - c["E"] * t_surface.pow(4)
            - c["G_H"] * (t_surface - t_air)
            - c["G_E"] * deficit
            - c["G_C"] * (t_surface - t_previous))


def residual_slope(t_surface, c):
    """``-dr/dT`` -- strictly positive, so ``r`` has exactly one root in T."""
    q_sat, slope = saturation_humidity_and_slope(t_surface, c["pressure"])
    evaporating = (q_sat > c["q_air"]).to(t_surface.dtype)
    return (4.0 * c["E"] * t_surface.pow(3) + c["G_H"]
            + c["G_E"] * slope * evaporating + c["G_C"])


def solve_closure_temperature(t_air, t_previous, c, steps=4, bracket=25.0):
    """Solve window_residual(T) = 0 with bracketed Newton iterations.

    Returns (T_star, converged); unconverged roots must be excluded from valid support.
    """
    low, high = t_air - bracket, t_air + bracket
    t = t_air.clone()
    for _ in range(steps):
        step = window_residual(t, t_air, t_previous, c) / residual_slope(t, c).clamp_min(1.0e-6)
        t = torch.clamp(t + step, low, high)
    residual = window_residual(t, t_air, t_previous, c)
    converged = torch.isfinite(t) & (residual.abs() < 1.0) & (t > low) & (t < high)
    return t, converged


def coefficients(sw, lw, albedo, rho, epsilon, ra, rs, capacity, qf, dry_paved,
                 seconds=SECONDS_PER_DAY, beta=1.0):
    """Build NumPy or torch coefficients. seconds is the storage interval; beta limits evaporation."""
    ge = beta * rho * LV / (ra + rs)
    ge = ge * (~dry_paved)
    return {"A": (1 - albedo) * sw + epsilon * lw + qf,
            "E": epsilon * SIGMA, "G_H": rho * CP / ra, "G_E": ge,
            "G_C": capacity / seconds, "Q_F": qf}


def pixel_fluxes(ta, ts, previous_ts, c):
    """Inputs must already be gathered onto valid support."""
    sensible = c["G_H"] * (ts - ta)
    evaporating = c["G_E"] > 0
    deficit = torch.relu(saturation_humidity(ts[evaporating], c["pressure"][evaporating])
                         - c["q_air"][evaporating])
    latent = torch.zeros_like(ts).masked_scatter(evaporating, c["G_E"][evaporating] * deficit)
    storage = c["G_C"] * (ts - previous_ts)
    net = c["A"] - c["Q_F"] - c["E"] * ts.pow(4)
    residual = net + c["Q_F"] - sensible - latent - storage
    return {"residual": residual, "Rn": net, "QF": c["Q_F"],
            "H": sensible, "LE": latent, "S": storage}


def window_flux_terms(ts, ta, previous, c, qf):
    """Return fluxes in W/m2 on gathered support, with Rn + QF - H - LE - S = residual."""
    q_sat, _slope = saturation_humidity_and_slope(ts, c["pressure"])
    deficit = torch.relu(q_sat - c["q_air"])
    sensible = c["G_H"] * (ts - ta)
    latent = c["G_E"] * deficit
    storage = c["G_C"] * (ts - previous)
    # A includes Q_F; subtract it to report net radiation separately.
    net = c["A"] - qf - c["E"] * ts.pow(4)
    return {"Rn": net, "QF": qf, "H": sensible, "LE": latent, "S": storage,
            "residual": net + qf - sensible - latent - storage}


def window_fluxes(ta, ts, physics, min_pixels=20):
    """Reduce [B, D, 5, 5] fields to [B, D-1] flux means and confidence.

    Gather valid support before nonlinear arithmetic to avoid masked NaNs.
    """
    if not bool(torch.isfinite(ta).all() and torch.isfinite(ts).all()):
        raise ValueError("Nonfinite predicted temperature in physics")
    mask = physics["pair_valid"]
    counts = mask.sum(dim=(-2, -1))
    valid = counts >= min_pixels
    selected = {}
    for key in ("A", "E", "G_H", "G_E", "G_C", "Q_F", "pressure", "q_air"):
        values = physics[key].to(dtype=ts.dtype).expand_as(ts)[:, 1:][mask]
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"Nonfinite physics coefficient on valid support: {key}")
        if key in ("E", "G_H", "G_C", "pressure") and bool((values <= 0).any()):
            raise ValueError(f"Nonpositive physics coefficient: {key}")
        if key in ("G_E", "Q_F", "q_air") and bool((values < 0).any()):
            raise ValueError(f"Negative physics coefficient: {key}")
        selected[key] = values
    fluxes = pixel_fluxes(ta[:, 1:][mask], ts[:, 1:][mask], ts[:, :-1][mask], selected)
    reduced = {}
    for key, value in fluxes.items():
        pixels = torch.zeros_like(ts[:, 1:]).masked_scatter(mask, value)
        reduced[key] = pixels.sum(dim=(-2, -1)) / counts.clamp_min(1)
    confidence = physics["pair_confidence"].to(dtype=ts.dtype)
    if not bool(torch.isfinite(confidence[mask]).all()) or bool(
        ((confidence[mask] < 0) | (confidence[mask] > 1)).any()
    ):
        raise ValueError("Invalid physics confidence")
    confidence = torch.where(mask, confidence, 0).sum(dim=(-2, -1)) / counts.clamp_min(1)
    return reduced, valid, confidence, counts


COEFFICIENT_KEYS = ("A", "E", "G_H", "G_E", "G_C", "Q_F")
WINDOW_COEFFICIENT_KEYS = tuple(f"{key}_{window}" for window in ("day", "night")
                                for key in ("A", "E", "G_H", "G_E", "G_C"))
# The operator range-checks uint8 hour sentinels.
TIMING_KEYS = ("era5_skt_min_hour", "era5_skt_max_hour",
               "era5_t2m_min_hour", "era5_t2m_max_hour",
               "sunrise_hour", "sunset_hour")


def _check_physics_entries(ta, physics, shapes):
    for key, shape in shapes.items():
        value = physics.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"physics.{key} must have shape {shape}")
        if value.device != ta.device or value.requires_grad:
            raise ValueError(f"physics.{key} must be fixed and on the prediction device")
        expected = torch.bool if key == "pair_valid" else (
            torch.int64 if key == "day_index" else None)
        if expected is not None and value.dtype != expected:
            raise ValueError(f"physics.{key} must have dtype {expected}")
        if expected is None and not value.is_floating_point():
            raise ValueError(f"physics.{key} must be floating point")


def validate_daynight_physics_batch(ta, ts, physics):
    """Validate day/night coefficient shapes, devices, and detached forcing."""
    if ta.shape != ts.shape or ta.ndim != 5 or tuple(ta.shape[2:]) != (3, 5, 5):
        raise ValueError("Physics requires Ta/Ts [B,D,3,5,5] in min/mean/max order")
    b, d = ta.shape[:2]
    if d < 2:
        raise ValueError("Day/night storage physics needs at least two predicted days")
    shapes = {key: (b, d, 5, 5) for key in WINDOW_COEFFICIENT_KEYS}
    shapes.update({key: (b, d) for key in TIMING_KEYS})
    shapes.update(pressure=(b, d, 5, 5), q_air=(b, d, 5, 5),
                  pair_valid=(b, d - 1, 5, 5), pair_confidence=(b, d - 1, 5, 5),
                  day_index=(b, d))
    _check_physics_entries(ta, physics, shapes)
    if not bool((physics["day_index"].diff(dim=1) == 1).all()):
        raise ValueError("Physics day indexes must be consecutive")


def validate_physics_batch(ta, ts, physics):
    if ta.shape != ts.shape or ta.ndim != 5 or tuple(ta.shape[2:]) != (3, 5, 5):
        raise ValueError("Physics requires Ta/Ts [B,D,3,5,5] in min/mean/max order")
    b, d = ta.shape[:2]
    if d < 2:
        raise ValueError("Daily storage physics needs at least two predicted days")
    shapes = {key: (b, d, 5, 5) for key in COEFFICIENT_KEYS}
    shapes.update(pressure=(b, d, 5, 5), q_air=(b, d, 5, 5),
                  pair_valid=(b, d - 1, 5, 5), pair_confidence=(b, d - 1, 5, 5),
                  day_index=(b, d))
    _check_physics_entries(ta, physics, shapes)
    if not bool((physics["day_index"].diff(dim=1) == 1).all()):
        raise ValueError("Physics day indexes must be consecutive")


class PhysicalLoss(nn.Module):
    """Penalize 24-hour flux residuals in W/m2 or day/night closure-temperature errors in Kelvin."""

    DAYNIGHT_MODE = "daynight_temperature_v1"
    RETIRED_24H_MODE = "simple_daily_fixed_band_v1"

    def __init__(self, flux_scale=100.0, residual_tolerance=25.0,
                 huber_transition=1.0, min_pixels=20, mode=DAYNIGHT_MODE,
                 sigma_forcing=30.0, tolerance_k=0.43, huber_delta_sigma=1.0,
                 air_source="predicted_detached",
                 newton_steps=4, bracket_k=25.0, curve_samples=24,
                 windows=PHYSICS_WINDOWS):
        super().__init__()
        if not (math.isfinite(flux_scale) and math.isfinite(residual_tolerance)
                and math.isfinite(huber_transition) and flux_scale > 0
                and residual_tolerance >= 0 and huber_transition > 0 and 1 <= min_pixels <= 25
                and int(min_pixels) == min_pixels):
            raise ValueError("Invalid physics reduction settings")
        if not (math.isfinite(sigma_forcing) and sigma_forcing > 0
                and math.isfinite(tolerance_k) and tolerance_k >= 0
                and math.isfinite(huber_delta_sigma) and huber_delta_sigma > 0
                and math.isfinite(bracket_k) and bracket_k > 0
                and 1 <= int(newton_steps) <= 32 and 4 <= int(curve_samples) <= 96):
            raise ValueError("Invalid day/night physics settings")
        if air_source not in AIR_SOURCES:
            raise ValueError(f"physics.air_source must be one of {sorted(AIR_SOURCES)}")
        windows = tuple(windows)
        if (not windows or len(set(windows)) != len(windows)
                or any(window not in PHYSICS_WINDOWS for window in windows)):
            raise ValueError(
                f"physics.windows must be a non-empty subset of {list(PHYSICS_WINDOWS)} "
                f"without repeats; got {list(windows)}")
        self.flux_scale = float(flux_scale)
        self.residual_tolerance = float(residual_tolerance)
        self.huber_transition = float(huber_transition)
        self.min_pixels = int(min_pixels)
        self.mode = str(mode)
        self.sigma_forcing = float(sigma_forcing)
        # Apply a Kelvin dead band before standardization; the Huber threshold is in sigma units.
        self.tolerance_k = float(tolerance_k)
        self.huber_delta_sigma = float(huber_delta_sigma)
        self.air_source = str(air_source)
        self.newton_steps = int(newton_steps)
        self.bracket_k = float(bracket_k)
        self.curve_samples = int(curve_samples)
        self.windows = tuple(w for w in PHYSICS_WINDOWS if w in windows)
        self.time_operator = SurfaceTimeOperator()

    def forward(self, ta_daily, ts_daily, physics):
        if self.mode == self.DAYNIGHT_MODE:
            return self._daynight_forward(ta_daily, ts_daily, physics)
        validate_physics_batch(ta_daily, ts_daily, physics)
        dtype = torch.float64 if ts_daily.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=ts_daily.device.type, enabled=False):
            ta, ts = ta_daily[:, :, 1].to(dtype), ts_daily[:, :, 1].to(dtype)
            flux, valid, confidence, counts = window_fluxes(ta, ts, physics, self.min_pixels)
            r = flux["residual"][valid]
            z = torch.relu(r.abs() - self.residual_tolerance) / self.flux_scale
            beta = self.huber_transition
            penalty = torch.where(z <= beta, 0.5 * z.square(), beta * (z - 0.5 * beta))
            weighted = confidence[valid] * penalty
            loss = weighted.mean() if bool(valid.any()) else (ta.sum() + ts.sum()) * 0
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite simplified physics loss")
            # Pool sums and counts across the epoch.
            stats = {"physics_window_count": float(valid.sum()),
                     "physics_pair_count": float(valid.numel()),
                     "physics_pixel_count": float(counts[valid].sum()),
                     "physics_penalty_sum": float(weighted.detach().sum()),
                     "physics_tolerance_sum": self.residual_tolerance * float(valid.sum()),
                     "physics_confidence_sum": float(confidence[valid].sum()),
                     "physics_active_count": float((z > 0).sum()),
                     "physics_abs_residual_sum": float(r.detach().abs().sum())}
            stats.update({f"physics_{key}_sum": float(value[valid].detach().sum())
                          for key, value in flux.items()})
        return loss, stats

    def _window_coefficients(self, physics, window, index):
        """Gather the day- or night-window coefficient maps for day ``index``."""
        keys = ("A", "E", "G_H", "G_E", "G_C")
        gathered = {key: physics[f"{key}_{window}"][:, index] for key in keys}
        # Pressure and specific humidity use the daily values in both windows.
        gathered["pressure"] = physics["pressure"][:, index]
        gathered["q_air"] = physics["q_air"][:, index]
        return gathered

    def _gather_qf(self, physics, window, support, dtype):
        """Gather daily anthropogenic heat for diagnostics; return zero if unavailable."""
        for key in (f"Q_F_{window}", "Q_F"):
            value = physics.get(key)
            if value is None:
                continue
            # Pair i corresponds to day i+1; omit day 0.
            values = value.to(dtype)[:, 1:].expand_as(support)[support]
            if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
                raise ValueError(f"Nonfinite or negative physics coefficient: {key}")
            return values
        return torch.zeros_like(support, dtype=dtype)[support]

    def _gather_window(self, physics, window, support, dtype):
        """Gather and validate coefficients on valid support before arithmetic."""
        gathered = {}
        for key, value in self._window_coefficients(physics, window, slice(1, None)).items():
            values = value.to(dtype).expand_as(support)[support]
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"Nonfinite physics coefficient on valid support: {key}_{window}")
            if key in ("E", "G_H", "G_C", "pressure") and bool((values <= 0).any()):
                raise ValueError(f"Nonpositive physics coefficient: {key}_{window}")
            if key in ("G_E", "q_air") and bool((values < 0).any()):
                raise ValueError(f"Negative physics coefficient: {key}_{window}")
            gathered[key] = values
        return gathered

    def _daynight_forward(self, ta_daily, ts_daily, physics):
        validate_daynight_physics_batch(ta_daily, ts_daily, physics)
        if not bool(torch.isfinite(ta_daily).all() and torch.isfinite(ts_daily).all()):
            raise ValueError("Nonfinite predicted temperature in physics")
        dtype = torch.float64 if ts_daily.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=ts_daily.device.type, enabled=False):
            if self.air_source == "era5":
                # Use ERA5 air temperature when the predicted air head is not used.
                missing = [key for key in ("t2m_min", "t2m_mean", "t2m_max") if key not in physics]
                if missing:
                    raise ValueError(f"physics.air_source=era5 requires batch physics {missing}")
                air = torch.stack([physics[f"t2m_{stat}"] for stat in ("min", "mean", "max")],
                                  dim=2).to(dtype)
            else:
                air = ta_daily.to(dtype)
            if self.air_source != "predicted_attached":
                # Detach Ta to hold the closure target fixed.
                air = air.detach()
            surface = ts_daily.to(dtype)
            hours = {key: physics[key].to(dtype) for key in
                     ("era5_skt_min_hour", "era5_skt_max_hour",
                      "era5_t2m_min_hour", "era5_t2m_max_hour",
                      "sunrise_hour", "sunset_hour")}
            ts_day, ts_night, _, night_ok = self.time_operator.window_means(
                surface, hours["era5_skt_min_hour"], hours["era5_skt_max_hour"],
                hours["sunrise_hour"], hours["sunset_hour"], n_samples=self.curve_samples)
            ta_day, ta_night, _, _ = self.time_operator.window_means(
                air, hours["era5_t2m_min_hour"], hours["era5_t2m_max_hour"],
                hours["sunrise_hour"], hours["sunset_hour"], n_samples=self.curve_samples)

            mask = physics["pair_valid"]
            confidence = physics["pair_confidence"].to(dtype)
            support = mask & night_ok[:, 1:, None, None]

            terms, diagnostics = [], {}
            for window, predicted, air_window, previous in (
                # Night(d) follows day(d-1); day(d) follows night(d). Detach the storage reference.
                ("night", ts_night[:, 1:], ta_night[:, 1:], ts_day[:, :-1].detach()),
                ("day", ts_day[:, 1:], ta_day[:, 1:], ts_night[:, 1:].detach()),
            ):
                # Gather valid support before solving; zero masks do not remove NaNs.
                c = self._gather_window(physics, window, support, dtype)
                air_support = air_window[support]
                previous_support = previous[support]
                predicted_support = predicted[support]
                target, converged_support = solve_closure_temperature(
                    air_support, previous_support, c,
                    steps=self.newton_steps, bracket=self.bracket_k)
                if self.air_source != "predicted_attached":
                    # Attached mode also propagates the closure gradient to Ta.
                    target = target.detach()
                slope = residual_slope(target, c).clamp_min(1.0e-6)
                sigma = (self.sigma_forcing / slope).clamp(0.05, 20.0)
                # Kelvin dead band, sigma standardization, then Huber penalty.
                excess = torch.relu((predicted_support - target).abs() - self.tolerance_k)
                z = excess / sigma
                delta = self.huber_delta_sigma
                penalty_support = torch.where(z <= delta, 0.5 * z.square(),
                                              delta * (z - 0.5 * delta))
                qf_support = self._gather_qf(physics, window, support, dtype)
                flux_support = window_flux_terms(
                    predicted_support, air_support, previous_support, c, qf_support)
                residual_support = flux_support["residual"]

                # Scatter to [B, D-1, 5, 5] with exact zeros outside support.
                def restore(values):
                    return torch.zeros_like(predicted).masked_scatter(support, values)

                converged = torch.zeros_like(support).masked_scatter(support, converged_support)
                penalty = restore(penalty_support)
                valid = support & converged
                pixels = valid.to(dtype)
                per_window = ((confidence * penalty * pixels).sum(dim=(-2, -1))
                              / pixels.sum(dim=(-2, -1)).clamp_min(1.0))
                # Count supported pixels after checking Newton convergence.
                usable = pixels.sum(dim=(-2, -1)) >= self.min_pixels
                # Excluded windows retain diagnostics but contribute no loss.
                if window in self.windows:
                    terms.append((per_window, usable))
                diagnostics[window] = {
                    "target": restore(target), "predicted": predicted,
                    "sigma": restore(sigma), "residual": restore(residual_support),
                    "valid": valid, "usable": usable, "converged": converged,
                    "support": support, "confidence": confidence,
                    "fluxes": {key: restore(value) for key, value in flux_support.items()},
                }

            windows = torch.cat([term[1] for term in terms], dim=1)
            values = torch.cat([term[0] for term in terms], dim=1)
            total = int(windows.sum())
            loss = (values[windows].mean() if total
                    else (surface.sum() + air.sum()) * 0)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite day/night physics loss")
            stats = {"physics_window_count": float(total),
                     "physics_pair_count": float(windows.numel()),
                     "physics_penalty_sum": float(values[windows].detach().sum()),
                     "physics_tolerance_sum": self.tolerance_k * total}
            # Sum pixel means per valid window; day/night keys use their own window counts.
            active = residual_total = pixel_total = 0.0
            solved = attempted = 0.0
            flux_totals = {key: 0.0 for key in ("Rn", "QF", "H", "LE", "S", "residual")}
            confidence_total = 0.0
            for window, entry in diagnostics.items():
                target, predicted = entry["target"], entry["predicted"]
                sigma, residual = entry["sigma"], entry["residual"]
                valid, usable = entry["valid"], entry["usable"]
                converged, base = entry["converged"], entry["support"]
                weight = valid.to(dtype)
                per_pixel = weight.sum(dim=(-2, -1)).clamp_min(1.0)
                gap = predicted.detach() - target
                # Measure convergence only on otherwise supported windows.
                window_attempted = float(base.sum())
                window_solved = float((base & converged).sum())
                attempted += window_attempted
                solved += window_solved
                stats[f"physics_{window}_newton_attempted_count"] = window_attempted
                stats[f"physics_{window}_newton_solved_count"] = window_solved

                def window_mean(value):
                    # Diagnostics do not need autograd history.
                    value = value.detach()
                    return float(((value * weight).sum(dim=(-2, -1)) / per_pixel)[usable].sum())

                residual_mean = window_mean(residual.abs())
                stats.update({
                    f"physics_{window}_window_count": float(usable.sum()),
                    f"physics_{window}_pixel_count": float((valid & usable[..., None, None]).sum()),
                    f"physics_{window}_dT_sum": window_mean(gap),
                    f"physics_{window}_abs_dT_sum": window_mean(gap.abs()),
                    f"physics_{window}_Tstar_sum": window_mean(target),
                    f"physics_{window}_sigma_sum": window_mean(sigma),
                    f"physics_{window}_abs_residual_sum": residual_mean,
                })
                window_fluxes_mean = {key: window_mean(value)
                                      for key, value in entry["fluxes"].items()}
                stats.update({f"physics_{window}_{key}_sum": value
                              for key, value in window_fluxes_mean.items()})
                if window not in self.windows:
                    # Pooled loss statistics include only selected windows.
                    continue
                residual_total += residual_mean
                pixel_total += stats[f"physics_{window}_pixel_count"]
                for key, value in window_fluxes_mean.items():
                    flux_totals[key] += value
                # Report the confidence actually used to weight each penalty.
                confidence_total += window_mean(confidence)
                active += float((weight * (gap.abs() > self.tolerance_k)).sum(dim=(-2, -1))
                                .div(per_pixel)[usable].sum())
            # Keep the established residual diagnostic in W/m2.
            stats["physics_abs_residual_sum"] = residual_total
            stats["physics_pixel_count"] = pixel_total
            stats["physics_active_count"] = active
            stats["physics_confidence_sum"] = confidence_total
            stats.update({f"physics_{key}_sum": value for key, value in flux_totals.items()})
            stats["physics_newton_converged_fraction"] = solved / max(attempted, 1.0)
        return loss, stats

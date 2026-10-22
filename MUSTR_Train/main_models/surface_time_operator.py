"""Continuous Landsat observation operator for canonical daily surface temperature."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..contracts import TEMP_MAX_INDEX, TEMP_MEAN_INDEX, TEMP_MIN_INDEX


class SurfaceTimeOperator(nn.Module):
    """Evaluate a raised-cosine curve from daily min/mean/max and ERA5 phase anchors.

    The curve power is solved from the daily mean, without learned overpass parameters.
    """

    def __init__(
        self,
        min_power: float = 0.05,
        max_power: float = 64.0,
        newton_steps: int = 10,
        min_temperature_range_k: float = 0.05,
    ) -> None:
        super().__init__()
        self.min_power = float(max(min_power, 1.0e-3))
        self.max_power = float(max(max_power, self.min_power + 1.0e-3))
        self.newton_steps = int(max(newton_steps, 1))
        self.min_temperature_range_k = float(max(min_temperature_range_k, 1.0e-6))

    @staticmethod
    def _normalized_curve_mean(power: torch.Tensor) -> torch.Tensor:
        # Integral from 0 to 1 of sin(pi*u/2) ** (2*p).
        return torch.exp(
            torch.lgamma(power + 0.5)
            - torch.lgamma(power + 1.0)
            - 0.5 * math.log(math.pi)
        )

    def _power_from_mean_ratio(self, ratio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        min_power = torch.full_like(ratio, self.min_power)
        max_power = torch.full_like(ratio, self.max_power)
        feasible_high = self._normalized_curve_mean(min_power)
        feasible_low = self._normalized_curve_mean(max_power)
        target = ratio.clamp(min=feasible_low, max=feasible_high)

        power = torch.ones_like(target).clamp(self.min_power, self.max_power)
        for _ in range(self.newton_steps):
            current = self._normalized_curve_mean(power)
            derivative = current * (torch.digamma(power + 0.5) - torch.digamma(power + 1.0))
            safe_derivative = torch.where(
                derivative.abs() > 1.0e-8,
                derivative,
                torch.full_like(derivative, -1.0e-8),
            )
            power = (power - (current - target) / safe_derivative).clamp(
                self.min_power,
                self.max_power,
            )
        return power, target

    @staticmethod
    def _base_level(hour, min_anchor, max_anchor):
        """Return the normalized diurnal level; phase anchors must broadcast with hour."""
        rise_duration = (max_anchor - min_anchor).remainder(24.0)
        rise_duration = torch.where(rise_duration < 0.5, torch.full_like(rise_duration, 12.0), rise_duration)
        cooling_duration = (24.0 - rise_duration).clamp_min(0.5)
        phase = (hour - min_anchor).remainder(24.0)
        rising = phase <= rise_duration
        rise_fraction = (phase / rise_duration).clamp(0.0, 1.0)
        cooling_fraction = ((phase - rise_duration) / cooling_duration).clamp(0.0, 1.0)
        rise_level = 0.5 * (1.0 - torch.cos(torch.pi * rise_fraction))
        cooling_level = 0.5 * (1.0 + torch.cos(torch.pi * cooling_fraction))
        return torch.where(rising, rise_level, cooling_level)

    @staticmethod
    def _check_shapes(
        ts_daily: torch.Tensor,
        observation_hour: torch.Tensor,
        min_hour: torch.Tensor,
        max_hour: torch.Tensor,
    ) -> None:
        if ts_daily.ndim != 5 or ts_daily.shape[2] != 3:
            raise ValueError(
                "ts_daily must have shape [B, D, 3, H, W] in [min, mean, max] order"
            )
        expected = tuple(ts_daily.shape[:2])
        for name, value in (
            ("observation_hour", observation_hour),
            ("min_hour", min_hour),
            ("max_hour", max_hour),
        ):
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

    def forward(
        self,
        ts_daily: torch.Tensor,
        observation_hour: torch.Tensor,
        min_hour: torch.Tensor,
        max_hour: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._check_shapes(ts_daily, observation_hour, min_hour, max_hour)
        output_dtype = ts_daily.dtype
        work = ts_daily.float()
        t_min = work[:, :, TEMP_MIN_INDEX]
        t_mean = work[:, :, TEMP_MEAN_INDEX]
        t_max = work[:, :, TEMP_MAX_INDEX]
        temperature_range = (t_max - t_min).clamp_min(self.min_temperature_range_k)
        mean_ratio = ((t_mean - t_min) / temperature_range).clamp(0.0, 1.0)
        power, fitted_ratio = self._power_from_mean_ratio(mean_ratio)

        obs_valid = torch.isfinite(observation_hour) & (observation_hour >= 0) & (observation_hour < 24)
        anchor_valid = (torch.isfinite(min_hour) & torch.isfinite(max_hour)
                        & (min_hour >= 0) & (min_hour < 24) & (max_hour >= 0) & (max_hour < 24))
        obs_hour = torch.nan_to_num(observation_hour.float(), nan=0.0).remainder(24.0)
        min_anchor = torch.nan_to_num(min_hour.float(), nan=0.0).remainder(24.0)
        max_anchor = torch.nan_to_num(max_hour.float(), nan=12.0).remainder(24.0)

        base_level = self._base_level(obs_hour, min_anchor, max_anchor)

        base_level = base_level.unsqueeze(-1).unsqueeze(-1)
        powered_level = base_level.clamp_min(1.0e-7).pow(power)
        powered_level = torch.where(base_level <= 1.0e-7, torch.zeros_like(powered_level), powered_level)
        prediction = t_min + temperature_range * powered_level

        fitted_curve_mean = t_min + temperature_range * self._normalized_curve_mean(power)
        diagnostics = {
            "curve_mean": fitted_curve_mean.to(dtype=output_dtype),
            "curve_mean_target": t_mean.to(dtype=output_dtype),
            "curve_power": power.to(dtype=output_dtype),
            "mean_ratio": mean_ratio.to(dtype=output_dtype),
            "fitted_mean_ratio": fitted_ratio.to(dtype=output_dtype),
            "timing_valid": (obs_valid & anchor_valid),
        }
        return prediction.to(dtype=output_dtype), diagnostics

    def window_means(
        self,
        t_daily: torch.Tensor,
        min_hour: torch.Tensor,
        max_hour: torch.Tensor,
        sunrise: torch.Tensor,
        sunset: torch.Tensor,
        n_samples: int = 24,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return day/night means [B, D, H, W] and validity masks [B, D].

        Input temperatures are [B, D, 3, H, W] in min/mean/max order; hours are [B, D].
        Night(d) spans sunset(d-1) to sunrise(d), so day 0 has no valid night.
        """
        if t_daily.ndim != 5 or t_daily.shape[2] != 3:
            raise ValueError("t_daily must have shape [B, D, 3, H, W] in [min, mean, max] order")
        expected = tuple(t_daily.shape[:2])
        for name, value in (("min_hour", min_hour), ("max_hour", max_hour),
                            ("sunrise", sunrise), ("sunset", sunset)):
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

        output_dtype = t_daily.dtype
        work = t_daily.float()
        t_min = work[:, :, TEMP_MIN_INDEX]
        t_mean = work[:, :, TEMP_MEAN_INDEX]
        t_max = work[:, :, TEMP_MAX_INDEX]
        temperature_range = (t_max - t_min).clamp_min(self.min_temperature_range_k)
        mean_ratio = ((t_mean - t_min) / temperature_range).clamp(0.0, 1.0)
        power, _ = self._power_from_mean_ratio(mean_ratio)

        anchor_valid = (self._hour_in_range(min_hour) & self._hour_in_range(max_hour)
                        & self._hour_in_range(sunrise) & self._hour_in_range(sunset))
        min_anchor = torch.nan_to_num(min_hour.float(), nan=0.0).remainder(24.0)
        max_anchor = torch.nan_to_num(max_hour.float(), nan=12.0).remainder(24.0)
        rise = torch.nan_to_num(sunrise.float(), nan=6.0).remainder(24.0)
        set_ = torch.nan_to_num(sunset.float(), nan=18.0).remainder(24.0)

        step = 24.0 / float(n_samples)
        hours = (torch.arange(n_samples, device=work.device, dtype=work.dtype) + 0.5) * step
        # Broadcast [B, D, S] levels against [B, D, H, W] powers.
        level = self._base_level(hours.view(1, 1, -1), min_anchor[..., None], max_anchor[..., None])
        level = level[..., None, None]
        powered = level.clamp_min(1.0e-7).pow(power[:, :, None])
        powered = torch.where(level <= 1.0e-7, torch.zeros_like(powered), powered)
        curve = t_min[:, :, None] + temperature_range[:, :, None] * powered

        hours_bd = hours.view(1, 1, -1)
        daylight = (hours_bd >= rise[..., None]) & (hours_bd <= set_[..., None])
        pre_dawn = hours_bd < rise[..., None]
        evening = hours_bd > set_[..., None]

        day_sum, day_count = self._masked_sum(curve, daylight)
        head_sum, head_count = self._masked_sum(curve, pre_dawn)
        tail_sum, tail_count = self._masked_sum(curve, evening)

        t_day = day_sum / day_count.clamp_min(1.0)[..., None, None]
        day_valid = anchor_valid & (day_count > 0)

        # Night(d) combines the evening of d-1 and the pre-dawn of d.
        night_sum = torch.zeros_like(head_sum)
        night_count = torch.zeros_like(head_count)
        night_sum[:, 1:] = tail_sum[:, :-1] + head_sum[:, 1:]
        night_count[:, 1:] = tail_count[:, :-1] + head_count[:, 1:]
        t_night = night_sum / night_count.clamp_min(1.0)[..., None, None]
        night_valid = torch.zeros_like(day_valid)
        night_valid[:, 1:] = (anchor_valid[:, :-1] & anchor_valid[:, 1:]
                              & (night_count[:, 1:] > 0))

        return (t_day.to(dtype=output_dtype), t_night.to(dtype=output_dtype),
                day_valid, night_valid)

    @staticmethod
    def _hour_in_range(hour: torch.Tensor) -> torch.Tensor:
        """Range-check uint8 hour anchors, whose missing sentinel is 255."""
        value = hour.float()
        return torch.isfinite(value) & (value >= 0) & (value < 24)

    @staticmethod
    def _masked_sum(curve: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum ``[B, D, S, H, W]`` over the sample axis under a ``[B, D, S]`` mask."""
        weight = mask.to(curve.dtype)
        return (curve * weight[..., None, None]).sum(dim=2), weight.sum(dim=2)


__all__ = ["SurfaceTimeOperator"]

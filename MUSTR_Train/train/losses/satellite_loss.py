"""Timestamped Landsat supervision for canonical daily surface temperature."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...contracts import CENTER_SIZE, TILE_SIZE
from ...main_models.surface_time_operator import SurfaceTimeOperator
from ..metrics import error_sums


class SatelliteLoss(nn.Module):
    """Score acquisition-time Landsat observations over central 5x5 patches.

    Each observed sample-day contributes one patch; anomaly and pattern terms
    require at least four valid pixels.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        huber_beta: float = 1.0,
        absolute_weight: float = 1.0,
        anomaly_weight: float = 0.0,
        pattern_weight: float = 0.0,
        gradient_weight: float = 0.05,
        curve_mean_weight: float = 0.05,
        curve_power_min: float = 0.05,
        curve_power_max: float = 64.0,
        curve_newton_steps: int = 10,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type).lower()
        self.huber_beta = float(max(huber_beta, 1.0e-6))
        self.absolute_weight = float(max(0.0, absolute_weight))
        self.anomaly_weight = float(max(0.0, anomaly_weight))
        self.pattern_weight = float(max(0.0, pattern_weight))
        self.gradient_weight = float(max(0.0, gradient_weight))
        self.curve_mean_weight = float(max(0.0, curve_mean_weight))
        self.time_operator = SurfaceTimeOperator(
            min_power=curve_power_min,
            max_power=curve_power_max,
            newton_steps=curve_newton_steps,
        )
        self.last_stats: dict[str, float] = {}

    def _reduce_loss(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, obs)
        return F.smooth_l1_loss(pred, obs, beta=self.huber_beta)

    def _pointwise(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, obs, reduction="none")
        return F.smooth_l1_loss(pred, obs, reduction="none", beta=self.huber_beta)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Mean over the masked pixels of each row; ``count`` must be at least 1."""
        return (values * mask).flatten(1).sum(dim=1) / count

    @staticmethod
    def _validate_inputs(
        ts_daily: torch.Tensor,
        lst_obs: torch.Tensor,
        lst_hour: torch.Tensor,
        lst_valid: torch.Tensor,
    ) -> None:
        if ts_daily.ndim != 5 or ts_daily.shape[2] != 3:
            raise ValueError("Ts_daily must have shape [B, D, 3, 5, 5]")
        if tuple(ts_daily.shape[-2:]) != (CENTER_SIZE, CENTER_SIZE):
            raise ValueError(f"Ts_daily must use a {CENTER_SIZE}x{CENTER_SIZE} patch")
        if (lst_obs.ndim != 4 or tuple(lst_obs.shape[:2]) != tuple(ts_daily.shape[:2])
                or tuple(lst_obs.shape[-2:]) != (TILE_SIZE, TILE_SIZE)):
            raise ValueError(f"lst_obs must have shape [B,L,{TILE_SIZE},{TILE_SIZE}] aligned with Ts_daily")
        expected = tuple(ts_daily.shape[:2])
        if tuple(lst_hour.shape) != expected or tuple(lst_valid.shape) != expected:
            raise ValueError(f"lst_hour and lst_valid must have shape {expected}")

    def forward(
        self,
        ts_daily: torch.Tensor,
        lst_obs: torch.Tensor,
        lst_hour: torch.Tensor,
        lst_valid: torch.Tensor,
        timing_daily: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        self._validate_inputs(ts_daily, lst_obs, lst_hour, lst_valid)
        min_hour = timing_daily.get("era5_skt_min_hour")
        max_hour = timing_daily.get("era5_skt_max_hour")
        if not isinstance(min_hour, torch.Tensor) or not isinstance(max_hour, torch.Tensor):
            raise KeyError("timing_daily requires era5_skt_min_hour and era5_skt_max_hour")

        prediction, diagnostics = self.time_operator(
            ts_daily,
            lst_hour.to(device=ts_daily.device),
            min_hour.to(device=ts_daily.device),
            max_hour.to(device=ts_daily.device),
        )
        day_valid = lst_valid.to(device=ts_daily.device).bool() & diagnostics["timing_valid"]
        crop_start = (int(lst_obs.shape[-1]) - CENTER_SIZE) // 2
        crop_stop = crop_start + CENTER_SIZE
        observations = lst_obs[:, :, crop_start:crop_stop, crop_start:crop_stop].to(
            device=ts_daily.device,
            dtype=ts_daily.dtype,
        )

        # One float32 row per observed sample-day.
        pred = prediction[day_valid].float()
        obs = observations[day_valid].float()
        total_pairs = int(pred.shape[0])

        obs_finite = torch.isfinite(obs)
        pred_finite = torch.isfinite(pred)
        valid = obs_finite & pred_finite
        mask = valid.float()
        valid_count = valid.flatten(1).sum(dim=1)
        has_valid = valid_count > 0
        count = valid_count.clamp_min(1).float()
        # Zero invalid pixels before arithmetic to keep NaNs out of gradients.
        p = torch.where(valid, pred, torch.zeros_like(pred))
        o = torch.where(valid, obs, torch.zeros_like(obs))

        absolute = self._masked_mean(self._pointwise(p, o), mask, count)

        # Anomaly and pattern terms require at least four valid pixels.
        enough = valid_count >= 4
        p_centered = torch.where(valid, p - (p.flatten(1).sum(dim=1) / count)[:, None, None], torch.zeros_like(p))
        o_centered = torch.where(valid, o - (o.flatten(1).sum(dim=1) / count)[:, None, None], torch.zeros_like(o))
        anomaly = self._masked_mean(self._pointwise(p_centered, o_centered), mask, count)
        anomaly_ok = enough & torch.isfinite(anomaly)

        denominator = (torch.linalg.vector_norm(p_centered.flatten(1), dim=1)
                       * torch.linalg.vector_norm(o_centered.flatten(1), dim=1))
        pattern_defined = enough & torch.isfinite(denominator) & (denominator > 1.0e-8)
        # Flat patches need a finite denominator even when masked out.
        safe_denominator = torch.where(pattern_defined, denominator, torch.ones_like(denominator))
        cosine = torch.clamp((p_centered * o_centered).flatten(1).sum(dim=1) / safe_denominator, -1.0, 1.0)
        pattern = 1.0 - cosine
        pattern_ok = pattern_defined & torch.isfinite(pattern)

        # Average over neighbor directions with valid pairs.
        valid_x = valid[:, :, 1:] & valid[:, :, :-1]
        valid_y = valid[:, 1:, :] & valid[:, :-1, :]
        count_x = valid_x.flatten(1).sum(dim=1)
        count_y = valid_y.flatten(1).sum(dim=1)
        loss_x = self._masked_mean(self._pointwise(p[:, :, 1:] - p[:, :, :-1], o[:, :, 1:] - o[:, :, :-1]),
                                   valid_x.float(), count_x.clamp_min(1).float())
        loss_y = self._masked_mean(self._pointwise(p[:, 1:, :] - p[:, :-1, :], o[:, 1:, :] - o[:, :-1, :]),
                                   valid_y.float(), count_y.clamp_min(1).float())
        has_x, has_y = count_x > 0, count_y > 0
        directions = has_x.float() + has_y.float()
        gradient = ((loss_x * has_x.float() + loss_y * has_y.float())
                    / torch.where(directions > 0, directions, torch.ones_like(directions)))
        gradient_ok = (has_x | has_y) & torch.isfinite(gradient)

        # Normalize by the sum of weights of included terms.
        components = (
            ("absolute", self.absolute_weight, absolute, has_valid),
            ("anomaly", self.anomaly_weight, anomaly, anomaly_ok),
            ("pattern", self.pattern_weight, pattern, pattern_ok),
            ("gradient", self.gradient_weight, gradient, gradient_ok),
        )
        numerator = torch.zeros_like(absolute)
        weight_sum = torch.zeros_like(absolute)
        included: dict[str, torch.Tensor] = {}
        for name, weight, value, ok in components:
            if weight <= 0.0:
                included[name] = torch.zeros_like(has_valid)
                continue
            use = ok & has_valid
            included[name] = use
            numerator = numerator + weight * torch.where(use, value, torch.zeros_like(value))
            weight_sum = weight_sum + weight * use.float()
        used = weight_sum > 0
        pair_loss = torch.where(used, numerator / torch.where(used, weight_sum, torch.ones_like(weight_sum)),
                                torch.zeros_like(numerator))
        used_count = used.sum()
        observation_loss = pair_loss.sum() / used_count.clamp_min(1).float()

        curve_mean = diagnostics["curve_mean"]
        curve_target = diagnostics["curve_mean_target"]
        curve_valid = torch.isfinite(curve_mean) & torch.isfinite(curve_target)
        curve_loss = (
            self._reduce_loss(curve_mean[curve_valid], curve_target[curve_valid])
            if bool(curve_valid.any().item())
            else ts_daily.new_zeros(())
        )
        total = observation_loss + self.curve_mean_weight * curve_loss

        difference = (p - o).detach()
        observed = o.detach()
        component_values = {name: value for name, _, value, _ in components}
        summary = torch.stack([
            day_valid.any(dim=1).sum().double(),
            used_count.double(),
            obs_finite.float().flatten(1).mean(dim=1).sum().double(),
            pred_finite.float().flatten(1).mean(dim=1).sum().double(),
            (mask.flatten(1).mean(dim=1) * has_valid.float()).sum().double(),
            valid_count.sum().double(),
            difference.sum().double(), difference.abs().sum().double(), difference.square().sum().double(),
            observed.sum().double(), observed.square().sum().double(),
            curve_loss.detach().double(),
            diagnostics["curve_power"].detach().float().mean().double(),
            *[included[name].sum().double() for name, *_ in components],
            *[torch.where(included[name], component_values[name].detach(),
                          torch.zeros_like(component_values[name])).sum().double()
              for name, *_ in components],
        ]).tolist()
        (samples_with_lst, used_pairs, obs_finite_ratio, pred_finite_ratio, valid_pixel_ratio,
         accuracy_pixels, error, absolute_error, square_error, observed_sum, observed_square,
         curve_loss_value, curve_power_mean) = summary[:13]
        counts = dict(zip(("absolute", "anomaly", "pattern", "gradient"), summary[13:17]))
        sums = dict(zip(("absolute", "anomaly", "pattern", "gradient"), summary[17:21]))

        self.last_stats = {
            "sat_samples_with_lst_ratio": float(samples_with_lst / max(ts_daily.shape[0], 1)),
            "sat_pairs_total": float(total_pairs),
            "sat_pairs_used": float(used_pairs),
            "sat_pairs_used_ratio": float(used_pairs / max(total_pairs, 1)),
            "sat_obs_finite_ratio": float(obs_finite_ratio / max(total_pairs, 1)),
            "sat_pred_finite_ratio": float(pred_finite_ratio / max(total_pairs, 1)),
            "sat_valid_pixel_ratio": float(valid_pixel_ratio / max(used_pairs, 1)),
            "sat_zero_pair_flag": float(total_pairs == 0),
            "sat_curve_mean_loss_raw": float(curve_loss_value),
            "sat_curve_mean_weight": float(self.curve_mean_weight),
            "sat_curve_power_mean": float(curve_power_mean),
            "sat_loss_type": 0.0 if self.loss_type == "mse" else 1.0,
        }
        for name, weight in (
            ("absolute", self.absolute_weight),
            ("anomaly", self.anomaly_weight),
            ("pattern", self.pattern_weight),
            ("gradient", self.gradient_weight),
        ):
            self.last_stats[f"sat_{name}_terms"] = float(counts[name])
            self.last_stats[f"sat_{name}_loss_raw"] = float(sums[name] / max(counts[name], 1))
            self.last_stats[f"sat_{name}_weight"] = float(weight)
        self.last_stats.update(error_sums("sat", "obs", int(accuracy_pixels), error, absolute_error,
                                          square_error, observed_sum, observed_square))
        return total


__all__ = ["SatelliteLoss"]

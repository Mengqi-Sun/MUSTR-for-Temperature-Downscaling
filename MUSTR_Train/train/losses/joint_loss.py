"""Joint loss for canonical daily Ta/Ts and their observation operators."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import (cfg_get, physics_priors, uses_physics, physics_policy,
                       PHYSICS_WINDOWS)
from ...contracts import CENTER_SIZE
from .physical_loss import PhysicalLoss
from .air_loss import AirLoss
from .gst_loss import GSTLoss
from .satellite_loss import SatelliteLoss


class JointLoss(nn.Module):
    WEIGHT_NAMES = ("lambda_air", "lambda_physics", "lambda_sat", "lambda_gst")

    def __init__(
        self,
        lambda_air: float = 1.0,
        lambda_sat: float = 0.0,
        lambda_gst: float = 0.0,
        air_loss_type: str = "mse",
        air_huber_beta: float = 1.0,
        sat_loss_type: str = "smooth_l1",
        sat_huber_beta: float = 1.0,
        sat_absolute_weight: float = 1.0,
        sat_anomaly_weight: float = 0.0,
        sat_pattern_weight: float = 0.0,
        sat_gradient_weight: float = 0.05,
        sat_curve_mean_weight: float = 0.05,
        sat_curve_power_min: float = 0.05,
        sat_curve_power_max: float = 64.0,
        sat_curve_newton_steps: int = 10,
        gst_loss_type: str = "mse",
        gst_huber_beta: float = 1.0,
        physics_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.air_loss = AirLoss(
            loss_type=air_loss_type,
            huber_beta=air_huber_beta,
        )
        self.satellite_loss = SatelliteLoss(
            loss_type=sat_loss_type,
            huber_beta=sat_huber_beta,
            absolute_weight=sat_absolute_weight,
            anomaly_weight=sat_anomaly_weight,
            pattern_weight=sat_pattern_weight,
            gradient_weight=sat_gradient_weight,
            curve_mean_weight=sat_curve_mean_weight,
            curve_power_min=sat_curve_power_min,
            curve_power_max=sat_curve_power_max,
            curve_newton_steps=sat_curve_newton_steps,
        )
        self.gst_loss = GSTLoss(
            loss_type=gst_loss_type,
            huber_beta=gst_huber_beta,
        )
        self.physics_config = physics_config or {}
        policy = physics_policy(self.physics_config)
        min_pixels = (physics_priors(self.physics_config)["quality"]["min_pixels"]
                      if uses_physics(self.physics_config) else 20)

        def setting(key, default):
            # Explicit null clears an inherited setting and uses the default.
            value = cfg_get(self.physics_config, f"physics.{key}", default)
            return default if value is None else value

        self.physics_loss = PhysicalLoss(
            flux_scale=float(setting("flux_scale_w_m2", 100.0)),
            residual_tolerance=float(setting("residual_tolerance_w_m2", 25.0)),
            huber_transition=float(setting("huber_transition", 1.0)),
            min_pixels=min_pixels,
            mode=policy.mode,
            sigma_forcing=float(setting("sigma_forcing_w_m2", 30.0)),
            tolerance_k=float(setting("tolerance_k", 0.43)),
            huber_delta_sigma=float(setting("huber_delta_sigma", 1.0)),
            air_source=str(setting("air_source", "predicted_detached")),
            newton_steps=int(setting("newton_steps", 4)),
            bracket_k=float(setting("bracket_k", 25.0)),
            curve_samples=int(setting("curve_samples", 24)),
            windows=tuple(setting("windows", PHYSICS_WINDOWS)),
        )
        self.default_weights = {
            "lambda_air": float(lambda_air),
            "lambda_physics": policy.weight if policy.enabled else 0.0,
            "lambda_sat": float(lambda_sat),
            "lambda_gst": float(lambda_gst),
        }
        self.weights = dict(self.default_weights)

    @classmethod
    def from_config(cls, config: dict) -> "JointLoss":
        return cls(
            lambda_air=float(cfg_get(config, "loss.lambda_air", 1.0)),
            lambda_sat=float(cfg_get(config, "loss.lambda_sat", 0.0)),
            lambda_gst=float(cfg_get(config, "loss.lambda_gst", 0.0)),
            air_loss_type=str(cfg_get(config, "loss.air_loss_type", "mse")),
            air_huber_beta=float(cfg_get(config, "loss.air_huber_beta", 1.0)),
            sat_loss_type=str(cfg_get(config, "loss.sat_loss_type", "smooth_l1")),
            sat_huber_beta=float(cfg_get(config, "loss.sat_huber_beta", 1.0)),
            sat_absolute_weight=float(cfg_get(config, "loss.sat_absolute_weight", 1.0)),
            sat_anomaly_weight=float(cfg_get(config, "loss.sat_anomaly_weight", 0.0)),
            sat_pattern_weight=float(cfg_get(config, "loss.sat_pattern_weight", 0.0)),
            sat_gradient_weight=float(cfg_get(config, "loss.sat_gradient_weight", 0.05)),
            sat_curve_mean_weight=float(cfg_get(config, "loss.sat_curve_mean_weight", 0.05)),
            sat_curve_power_min=float(cfg_get(config, "loss.sat_curve_power_min", 0.05)),
            sat_curve_power_max=float(cfg_get(config, "loss.sat_curve_power_max", 64.0)),
            sat_curve_newton_steps=int(cfg_get(config, "loss.sat_curve_newton_steps", 10)),
            gst_loss_type=str(cfg_get(config, "loss.gst_loss_type", "mse")),
            gst_huber_beta=float(cfg_get(config, "loss.gst_huber_beta", 1.0)),
            physics_config=config,
        )

    @staticmethod
    def _sanitize(value: torch.Tensor, cap: float = 1.0e6) -> torch.Tensor:
        return torch.clamp(torch.nan_to_num(value, nan=0.0, posinf=cap, neginf=cap), 0.0, cap)

    @staticmethod
    def _as_float(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().mean().item())
        return float(value)

    def set_phase_loss_weights(self, overrides: dict[str, float] | None = None) -> None:
        self.weights = dict(self.default_weights)
        if overrides:
            for name in self.WEIGHT_NAMES:
                if name == "lambda_physics":
                    continue  # Physics weights are controlled by physics.weight and phase overrides.
                if name in overrides:
                    self.weights[name] = float(overrides[name])

    def get_active_lambda_dict(self) -> dict[str, float]:
        return dict(self.weights)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        phase: int = 1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        center = CENTER_SIZE // 2
        ta_center = outputs["Ta_daily"][..., center, center]
        ts_center = outputs["Ts_daily"][..., center, center]

        air_raw = self.air_loss(
            ta_center,
            batch["daily_labels"],
            batch["is_real"],
        )
        sat_raw = self.satellite_loss(
            outputs["Ts_daily"],
            batch["lst_obs"],
            batch["lst_hour"],
            batch["lst_valid"],
            batch["timing_daily"],
        )
        gst_raw, gst_stats = self.gst_loss(
            ts_center,
            batch["daily_gst_labels"],
            batch["is_real"],
        )
        policy = physics_policy(self.physics_config, phase)
        enabled = policy.enabled
        self.weights["lambda_physics"] = policy.weight if enabled else 0.0
        physics_stats = {"physics_enabled": float(enabled),
                         "physics_weight": self.weights["lambda_physics"]}
        if not enabled:
            physics_raw = ta_center.new_zeros(())
        else:
            if "physics" not in batch:
                raise ValueError("Enabled simplified physics requires batch['physics']")
            physics_raw, more_stats = self.physics_loss(outputs["Ta_daily"], outputs["Ts_daily"], batch["physics"])
            physics_stats.update(more_stats)

        raw = {"air": air_raw, "physics": physics_raw, "sat": sat_raw, "gst": gst_raw}
        weighted_inputs: dict[str, torch.Tensor] = {}
        for name, value in raw.items():
            if name == "physics":
                if not bool(torch.isfinite(value)):
                    raise ValueError("Nonfinite simplified physics cannot be sanitized to zero")
                weighted_inputs[name] = value
            else:
                weighted_inputs[name] = self._sanitize(value)
        total = sum(
            self.weights[f"lambda_{name}"] * weighted_inputs[name]
            for name in ("air", "physics", "sat", "gst")
        )
        stats = {
            "loss_total": self._as_float(total),
            "loss_observation": sum(float(cfg_get(self.physics_config or {}, f"training.selection_weights.{name}", 1.0))
                                    * self._as_float(raw[name]) for name in ("air", "gst", "sat")),
            "L_air_raw": self._as_float(air_raw),
            "L_physics_raw": self._as_float(physics_raw),
            "L_sat_raw": self._as_float(sat_raw),
            "L_gst_raw": self._as_float(gst_raw),
            "L_air_weighted": self._as_float(self.weights["lambda_air"] * weighted_inputs["air"]),
            "L_physics_weighted": self._as_float(
                self.weights["lambda_physics"] * weighted_inputs["physics"]
            ),
            "L_sat_weighted": self._as_float(self.weights["lambda_sat"] * weighted_inputs["sat"]),
            "L_gst_weighted": self._as_float(self.weights["lambda_gst"] * weighted_inputs["gst"]),
            **self.air_loss.last_stats,
            **self.satellite_loss.last_stats,
            **gst_stats,
            **physics_stats,
        }
        return total, stats


__all__ = ["JointLoss"]

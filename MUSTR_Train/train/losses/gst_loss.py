"""Direct daily 0 cm GST supervision of the canonical surface-field center."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import error_terms


class GSTLoss(nn.Module):
    """Component-balanced GST loss in canonical ``[min, mean, max]`` order."""

    DAILY_KEYS = ("gst_min", "gst_mean", "gst_max")

    def __init__(self, loss_type: str = "mse", huber_beta: float = 1.0) -> None:
        super().__init__()
        self.loss_type = str(loss_type).lower()
        self.huber_beta = max(float(huber_beta), 1.0e-6)

    def _elementwise_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target, reduction="none")
        return F.smooth_l1_loss(
            prediction,
            target,
            beta=self.huber_beta,
            reduction="none",
        )

    def forward(
        self,
        ts_center: torch.Tensor,
        gst_labels: dict[str, torch.Tensor],
        is_real: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if ts_center.ndim != 3 or int(ts_center.shape[-1]) != len(self.DAILY_KEYS):
            raise ValueError("ts_center must have shape [B, D, 3] in [min, mean, max] order")

        real_mask = is_real.to(device=ts_center.device).bool()[:, None]
        component_losses: list[torch.Tensor] = []
        stats: dict[str, float] = {}
        total_count = 0

        for channel_index, key in enumerate(self.DAILY_KEYS):
            component = key.removeprefix("gst_")
            prediction = ts_center[:, :, channel_index]
            observation = gst_labels[key].to(
                device=prediction.device,
                dtype=prediction.dtype,
            ) + 273.15
            valid = real_mask & torch.isfinite(prediction) & torch.isfinite(observation)
            count = int(valid.sum().item())
            total_count += count
            if count:
                component_loss = self._elementwise_loss(
                    prediction[valid], observation[valid]
                ).mean()
                component_losses.append(component_loss)
                stats.update(error_terms("gst", component, prediction[valid], observation[valid]))
            else:
                component_loss = prediction.sum() * 0.0
            stats[f"gst_{component}_raw"] = float(component_loss.detach().item())
            stats[f"gst_{component}_terms"] = float(count)

        total = (
            torch.stack(component_losses).mean()
            if component_losses
            else ts_center.sum() * 0.0
        )
        stats["gst_components"] = float(len(component_losses))
        stats["gst_terms"] = float(total_count)
        return total, stats


__all__ = ["GSTLoss"]

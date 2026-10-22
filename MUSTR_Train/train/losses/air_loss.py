"""Direct daily TEM supervision of the canonical air-temperature field center."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import error_terms


class AirLoss(nn.Module):
    """Component-balanced TEM loss in canonical ``[min, mean, max]`` order."""

    DAILY_KEYS = ("tem_min", "tem_mean", "tem_max")

    def __init__(self, loss_type: str = "mse", huber_beta: float = 1.0) -> None:
        super().__init__()
        self.loss_type = str(loss_type).lower()
        self.huber_beta = max(float(huber_beta), 1.0e-6)
        self.last_stats: dict[str, float] = {}

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
        ta_center: torch.Tensor,
        daily_labels: dict[str, torch.Tensor],
        is_real: torch.Tensor,
    ) -> torch.Tensor:
        if ta_center.ndim != 3 or int(ta_center.shape[-1]) != len(self.DAILY_KEYS):
            raise ValueError("ta_center must have shape [B, D, 3] in [min, mean, max] order")

        real_mask = is_real.to(device=ta_center.device).bool()[:, None]
        component_losses: list[torch.Tensor] = []
        stats: dict[str, float] = {}
        total_count = 0

        for channel_index, key in enumerate(self.DAILY_KEYS):
            component = key.removeprefix("tem_")
            prediction = ta_center[:, :, channel_index]
            observation = daily_labels[key].to(
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
                stats.update(error_terms("air", component, prediction[valid], observation[valid]))
            else:
                component_loss = prediction.sum() * 0.0
            stats[f"air_{component}_raw"] = float(component_loss.detach().item())
            stats[f"air_{component}_terms"] = float(count)

        total = (
            torch.stack(component_losses).mean()
            if component_losses
            else ta_center.sum() * 0.0
        )
        stats["air_components"] = float(len(component_losses))
        stats["air_terms"] = float(total_count)
        self.last_stats = stats
        return total


__all__ = ["AirLoss"]

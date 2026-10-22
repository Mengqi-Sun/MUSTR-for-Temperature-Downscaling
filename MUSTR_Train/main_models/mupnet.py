"""Daily Ta/Ts prediction in Kelvin, shaped [B, D, 3, 5, 5] in min/mean/max order."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import cfg_get as _cfg_get
from ..contracts import CENTER_SIZE, HALO_PIXELS, TEMP_CHANNELS, TILE_SIZE, DAILY_INPUT_FEATURE_COUNT
from .backbone import CLCZEmbedding, KGEmbedding, STTCN


def _zero_init_conv(module: nn.Conv2d) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class DailyMUPNet(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.hidden_dim = int(_cfg_get(config, "model.hidden_dim", 128))
        self.n_temporal_blocks = int(_cfg_get(config, "model.n_temporal_blocks", 6))
        self.clcz_embed_dim = int(_cfg_get(config, "data.clcz_embed_dim", 16))
        self.kg_embed_dim = int(_cfg_get(config, "data.kg_embed_dim", 2))
        self.air_spatial_anomaly_scale = float(
            _cfg_get(config, "model.daily.air_spatial_anomaly_scale_k", 1.0)
        )
        self.surface_spatial_anomaly_scale = float(
            _cfg_get(
                config,
                "model.daily.surface_spatial_anomaly_scale_k",
                _cfg_get(config, "model.daily.spatial_anomaly_scale_k", 2.0),
            )
        )
        self.gap_log_scale = float(_cfg_get(config, "model.daily.gap_log_scale", 0.25))
        self.reference_gap_ratio_min = float(
            _cfg_get(config, "model.daily.reference_gap_ratio_min", 0.25)
        )
        self.reference_gap_ratio_max = float(
            _cfg_get(config, "model.daily.reference_gap_ratio_max", 4.0)
        )

        self.backbone = None  # type: Optional[STTCN]
        self.backbone_c_in = None  # type: Optional[int]
        self.clcz_embed = CLCZEmbedding(embed_dim=self.clcz_embed_dim)
        self.kg_embed = KGEmbedding(embed_dim=self.kg_embed_dim)

        # Rows 6:12 retain the tensor shape required by version-4 checkpoints.
        self.state_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 12),
        )
        self.air_spatial_head = nn.Conv2d(self.hidden_dim, 3, kernel_size=1)
        self.surface_spatial_head = nn.Conv2d(self.hidden_dim, 3, kernel_size=1)
        _zero_init_conv(self.air_spatial_head)
        _zero_init_conv(self.surface_spatial_head)

    def _ensure_backbone(self, c_in: int, device: torch.device) -> None:
        if self.backbone is not None:
            if self.backbone_c_in != c_in:
                raise ValueError(
                    f"DailyMUPNet input channels changed: initialized={self.backbone_c_in}, current={c_in}"
                )
            return
        self.backbone = STTCN(
            c_in=c_in,
            hidden=self.hidden_dim,
            n_blocks=self.n_temporal_blocks,
            out_channels=self.hidden_dim,
        )
        self.backbone.to(device)
        self.backbone_c_in = c_in

    def _build_ordered_field(
        self,
        mean: torch.Tensor,
        low_gap: torch.Tensor,
        high_gap: torch.Tensor,
        spatial_raw: torch.Tensor,
        reference: torch.Tensor,
        reference_valid: torch.Tensor,
        anomaly_scale: float,
    ) -> torch.Tensor:
        """Build ordered fields; center station losses also train the spatial corrections."""
        height, width = spatial_raw.shape[-2:]
        expected_reference_shape = (*spatial_raw.shape[:2], 3, height, width)
        if tuple(reference.shape) != expected_reference_shape:
            raise ValueError(
                f"temperature reference must have shape {expected_reference_shape}, "
                f"got {tuple(reference.shape)}"
            )
        expected_valid_shape = (*spatial_raw.shape[:2], height, width)
        if tuple(reference_valid.shape) != expected_valid_shape:
            raise ValueError(
                f"temperature reference validity must have shape {expected_valid_shape}, "
                f"got {tuple(reference_valid.shape)}"
            )

        start = HALO_PIXELS
        stop = start + CENTER_SIZE
        correction_patch = spatial_raw[:, :, :, start:stop, start:stop]
        reference_patch = reference[:, :, :, start:stop, start:stop]
        valid_patch = reference_valid[:, :, start:stop, start:stop].bool()

        center_y, center_x = height // 2, width // 2
        reference_center = reference[:, :, :, center_y, center_x]
        center_valid = reference_valid[:, :, center_y, center_x].bool()
        pair_valid = valid_patch & center_valid[..., None, None]

        reference_min = reference_patch[:, :, 0]
        reference_mean = reference_patch[:, :, 1]
        reference_max = reference_patch[:, :, 2]
        center_min, center_mean, center_max = reference_center.unbind(dim=-1)

        mean_offset = torch.where(
            pair_valid,
            reference_mean - center_mean[..., None, None],
            torch.zeros_like(reference_mean),
        )
        reference_low_gap = (reference_mean - reference_min).clamp_min(0.05)
        reference_high_gap = (reference_max - reference_mean).clamp_min(0.05)
        center_low_gap = (center_mean - center_min).clamp_min(0.05)
        center_high_gap = (center_max - center_mean).clamp_min(0.05)
        low_ratio = torch.where(
            pair_valid,
            reference_low_gap / center_low_gap[..., None, None],
            torch.ones_like(reference_low_gap),
        ).clamp(self.reference_gap_ratio_min, self.reference_gap_ratio_max)
        high_ratio = torch.where(
            pair_valid,
            reference_high_gap / center_high_gap[..., None, None],
            torch.ones_like(reference_high_gap),
        ).clamp(self.reference_gap_ratio_min, self.reference_gap_ratio_max)

        mean_patch = (
            mean[..., None, None]
            + mean_offset
            + anomaly_scale * torch.tanh(correction_patch[:, :, 0] / 2.0)
        )
        low_patch = low_gap[..., None, None] * low_ratio * torch.exp(
            self.gap_log_scale * torch.tanh(correction_patch[:, :, 1])
        )
        high_patch = high_gap[..., None, None] * high_ratio * torch.exp(
            self.gap_log_scale * torch.tanh(correction_patch[:, :, 2])
        )
        return torch.stack(
            [mean_patch - low_patch, mean_patch, mean_patch + high_patch],
            dim=2,
        )

    def forward(self, batch: Dict[str, Any], phase: int = 1) -> Dict[str, Any]:
        del phase
        x = batch["daily_input_features"].float()
        if x.ndim != 5 or tuple(x.shape[2:]) != (DAILY_INPUT_FEATURE_COUNT, TILE_SIZE, TILE_SIZE):
            raise ValueError(f"DailyMUPNet requires [B,L,{DAILY_INPUT_FEATURE_COUNT},{TILE_SIZE},{TILE_SIZE}]")
        batch_size, days, _, height, width = x.shape
        device, dtype = x.device, x.dtype

        kg_class = batch["kg_class"].to(device=device)
        kg_emb = self.kg_embed(kg_class).to(dtype=dtype)
        if kg_emb.ndim == 4:
            kg_emb = kg_emb.unsqueeze(1).expand(-1, days, -1, -1, -1)
        elif int(kg_emb.shape[1]) != days:
            raise ValueError(f"Daily KG length must be {days}, got {kg_emb.shape[1]}")

        clcz_daily = batch["daily_clcz_class"].to(device=device)
        clcz_emb = self.clcz_embed(clcz_daily).to(dtype=dtype)
        if tuple(clcz_emb.shape[:2]) != (batch_size, days):
            raise ValueError(
                f"Daily CLCZ shape must start with [B={batch_size}, D={days}], got {tuple(clcz_emb.shape)}"
            )

        x_backbone = torch.cat([x, kg_emb, clcz_emb], dim=2)
        self._ensure_backbone(c_in=int(x_backbone.shape[2]), device=device)
        assert self.backbone is not None
        latent = self.backbone(x_backbone)
        center_latent = latent[:, :, :, height // 2, width // 2]
        state_raw = self.state_head(center_latent)

        phys_daily = {
            key: value.to(device=device, dtype=dtype)
            for key, value in batch.get("phys_daily", {}).items()
        }
        air_reference = phys_daily.get(
            "t2m_mean",
            torch.full((batch_size, days), 273.15, device=device, dtype=dtype),
        )
        surface_reference = phys_daily.get("skt_mean", air_reference)

        ta_mean = air_reference + 8.0 * torch.tanh(state_raw[..., 0] / 4.0)
        ta_low_gap = F.softplus(state_raw[..., 1]) + 0.1
        ta_high_gap = F.softplus(state_raw[..., 2]) + 0.1
        ts_mean = surface_reference + 12.0 * torch.tanh(state_raw[..., 3] / 4.0)
        ts_low_gap = F.softplus(state_raw[..., 4]) + 0.1
        ts_high_gap = F.softplus(state_raw[..., 5]) + 0.1

        latent_flat = latent.reshape(batch_size * days, self.hidden_dim, height, width)
        air_spatial_raw = self.air_spatial_head(latent_flat).reshape(batch_size, days, 3, height, width)
        surface_spatial_raw = self.surface_spatial_head(latent_flat).reshape(
            batch_size, days, 3, height, width
        )
        ta_daily = self._build_ordered_field(
            ta_mean,
            ta_low_gap,
            ta_high_gap,
            air_spatial_raw,
            batch["ta_reference_daily"].to(device=device, dtype=dtype),
            batch["ta_reference_valid"].to(device=device),
            self.air_spatial_anomaly_scale,
        )
        ts_daily = self._build_ordered_field(
            ts_mean,
            ts_low_gap,
            ts_high_gap,
            surface_spatial_raw,
            batch["ts_reference_daily"].to(device=device, dtype=dtype),
            batch["ts_reference_valid"].to(device=device),
            self.surface_spatial_anomaly_scale,
        )

        return {
            "Ta_daily": ta_daily,
            "Ts_daily": ts_daily,
            "daily_state": {
                "Ta_mean": ta_mean,
                "Ta_amp_day": ta_high_gap,
                "Ta_amp_night": ta_low_gap,
                "Ts_mean": ts_mean,
                "Ts_amp_day": ts_high_gap,
                "Ts_amp_night": ts_low_gap,
            },
            "temperature_channel_order": TEMP_CHANNELS,
        }


__all__ = ["DailyMUPNet"]

"""Spatiotemporal convolutional encoder with categorical embeddings."""

import torch
import torch.nn as nn

from ..contracts import N_CLCZ_CLASSES, N_KG_CLASSES


class CLCZEmbedding(nn.Module):
    """Per-pixel CLCZ embedding with a small spatial context encoder."""

    def __init__(self, num_classes: int = N_CLCZ_CLASSES, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim, padding_idx=0)
        self.norm = nn.LayerNorm(embed_dim)
        self.pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.context = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("CLCZ classes must have shape [B, D, H, W]")
        idx = x.long()
        idx = torch.where(
            (idx >= 0) & (idx < self.embed.num_embeddings),
            idx,
            torch.zeros_like(idx),
        )
        embedded = self.norm(self.embed(idx)).permute(0, 1, 4, 2, 3)
        B, D, E, H, W = embedded.shape
        embedded = embedded.reshape(B * D, E, H, W)
        encoded = self.act(self.pointwise(embedded))
        encoded = encoded + self.context(encoded)
        encoded = encoded * (idx != 0).reshape(B * D, 1, H, W).to(encoded.dtype)
        return encoded.reshape(B, D, E, H, W)


class KGEmbedding(nn.Module):
    """Learnable embedding for the static Köppen-Geiger class map."""

    def __init__(self, num_classes: int = N_KG_CLASSES, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (3, 4):
            raise ValueError("KG classes must have shape [B, H, W] or [B, D, H, W]")
        idx = x.long()
        idx = torch.where(
            (idx >= 0) & (idx < self.embed.num_embeddings),
            idx,
            torch.zeros_like(idx),
        )
        embedded = self.embed(idx)
        if x.ndim == 3:
            return embedded.permute(0, 3, 1, 2)
        return embedded.permute(0, 1, 4, 2, 3)


class STBlock(nn.Module):
    """Residual block with non-causal temporal and per-day spatial convolutions."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # Symmetric padding gives non-causal temporal context.
        self.temporal_conv = nn.Conv1d(
            channels, channels, kernel_size=3,
            dilation=dilation, padding=dilation,
        )
        self.temporal_bn = nn.BatchNorm1d(channels)
        self.spatial_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
        )
        self.spatial_bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B, D, C, H, W] inputs to the same shape."""
        residual = x
        B, W, C, H, Wd = x.shape

        x = x.permute(0, 3, 4, 2, 1).reshape(B * H * Wd, C, W)
        x = self.act(self.temporal_bn(self.temporal_conv(x)))
        x = x.reshape(B, H, Wd, C, W).permute(0, 4, 3, 1, 2)

        x = x.reshape(B * W, C, H, Wd)
        x = self.act(self.spatial_bn(self.spatial_conv(x)))
        x = x.reshape(B, W, C, H, Wd)

        return x + residual


class STTCN(nn.Module):
    """Stack temporal-spatial blocks and project to the requested output channels."""

    def __init__(
        self,
        c_in: int,
        hidden: int = 128,
        n_blocks: int = 10,
        out_channels: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(c_in)
        self.input_proj = nn.Conv2d(c_in, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            STBlock(hidden, dilation=2 ** i) for i in range(n_blocks)
        ])
        self.out_channels = int(max(1, out_channels))
        self.output_head = nn.Conv2d(hidden, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [B, D, C_in, H, W] to [B, D, out_channels, H, W]."""
        B, W, C, H, Wd = x.shape

        x = self.input_norm(x.reshape(B * W, C, H, Wd))
        x = self.input_proj(x)
        hidden = x.shape[1]
        x = x.reshape(B, W, hidden, H, Wd)

        for block in self.blocks:
            x = block(x)

        x = self.output_head(x.reshape(B * W, hidden, H, Wd))
        return x.reshape(B, W, self.out_channels, H, Wd)

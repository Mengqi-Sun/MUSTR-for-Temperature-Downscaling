"""Load and apply persisted input statistics; paper experiments disable normalization."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from DataCube.schema import SCHEMA_VERSION
from ..config import cfg_get
from ..contracts import (
    DAILY_INPUT_CHANNELS,
    DAILY_INPUT_FEATURE_COUNT,
    DAILY_INPUT_NORMALIZED_INDICES,
)

NORMALIZATION_SCHEMA_VERSION = 4


class DailyInputNormalizer:
    """Apply persisted statistics to the continuous daily input channels."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        normalized_indices: Sequence[int],
        feature_names: Sequence[str],
    ):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.normalized_indices = np.asarray(normalized_indices, dtype=np.int64)
        self.feature_names = tuple(feature_names)

    @classmethod
    def load(cls, path: str | Path) -> "DailyInputNormalizer":
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        schema_version = int(payload.get("schema_version", 0)) if isinstance(payload, dict) else 0
        if schema_version != NORMALIZATION_SCHEMA_VERSION:
            raise RuntimeError(
                f"Incompatible daily normalization schema in {path}: "
                f"found {schema_version}, required {NORMALIZATION_SCHEMA_VERSION}. "
                "Refit normalization statistics for the canonical daily input contract."
            )
        feature_names = tuple(payload.get("feature_names", ()))
        normalized_indices = tuple(int(index) for index in payload.get("normalized_indices", ()))
        if feature_names != tuple(DAILY_INPUT_CHANNELS):
            raise RuntimeError(
                f"Daily normalization feature order in {path} does not match MUSTR_Train/contracts.py"
            )
        if normalized_indices != tuple(DAILY_INPUT_NORMALIZED_INDICES):
            raise RuntimeError(
                f"Daily normalization indices in {path} do not match the canonical contract"
            )
        mean, std = np.asarray(payload["mean"]), np.asarray(payload["std"])
        if (mean.shape != (DAILY_INPUT_FEATURE_COUNT,) or std.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()
                or payload.get("h5_schema") != SCHEMA_VERSION):
            raise ValueError("Invalid daily normalization values/schema")
        return cls(
            mean=payload["mean"],
            std=payload["std"],
            normalized_indices=normalized_indices,
            feature_names=feature_names,
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 4:
            raise ValueError(f"daily_input_features expects [D,C,H,W], got {features.shape}")
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"daily feature count mismatch: got {features.shape[1]}, expected {len(self.feature_names)}"
            )
        output = np.asarray(features, dtype=np.float32).copy()
        indices = self.normalized_indices
        mean = self.mean[indices].reshape(1, -1, 1, 1)
        std = self.std[indices].reshape(1, -1, 1, 1)
        output[:, indices] = (output[:, indices] - mean) / (std + 1.0e-8)
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def load_daily_normalizer(config: dict) -> DailyInputNormalizer | None:
    if not bool(cfg_get(config, "normalization.enabled", cfg_get(config, "normalization.enable_daily_input_standardize", False))):
        return None
    stats_path = cfg_get(config, "normalization.stats_file", cfg_get(config, "normalization.daily_input_stats_file"))
    if not stats_path:
        raise ValueError("normalization is enabled but no stats_file is configured")
    return DailyInputNormalizer.load(stats_path)

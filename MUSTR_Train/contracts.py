"""Shared daily data/model contracts used by training and regional inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from DataCube.schema import ERA5 as ERA5_DAILY_CHANNELS, INPUT_SIZE, OUTPUT_SIZE
WINDOW_DAYS = 7
TILE_SIZE = INPUT_SIZE
CENTER_SIZE = OUTPUT_SIZE
HALO_PIXELS = (TILE_SIZE - CENTER_SIZE) // 2
MODEL_CONTRACT_VERSION = 7

TEMP_CHANNELS = ("min", "mean", "max")
TEMP_MIN_INDEX = 0
TEMP_MEAN_INDEX = 1
TEMP_MAX_INDEX = 2

# Class 0 is unknown; CLCZ codes 1-18 retain their source IDs.
N_CLCZ_CLASSES = 19
N_KG_CLASSES = 30

DAILY_CHANNELS = (
    "albedo", "ndvi", "ndbi", "sar_vv", "sar_vh", "ntl", "optical_age", "sar_age",
)
DAILY_CHANNELS_WITH_VALID = tuple(
    name for name in DAILY_CHANNELS if name not in ("optical_age", "sar_age")
)
STATIC_CHANNELS = ("dem", "cmtbh", "svf", "kg")
CONTINUOUS_STATIC_CHANNELS = ("dem", "cmtbh", "svf")

DAILY_TEMPORAL_CHANNELS = (
    "temporal_sin_d",
    "temporal_cos_d",
    "temporal_weekend",
)

# Snow albedo is sparse and excluded from network inputs, but retained in raw data.
EXCLUDED_ERA5_INPUT_CHANNELS = ("snow_albedo_mean",)
DAILY_ERA5_INPUT_CHANNELS = tuple(
    name for name in ERA5_DAILY_CHANNELS if name not in EXCLUDED_ERA5_INPUT_CHANNELS
)

DAILY_INPUT_CHANNELS = (
    DAILY_ERA5_INPUT_CHANNELS
    + tuple(
        channel
        for name in DAILY_CHANNELS
        for channel in (
            (name, f"{name}_valid") if name in DAILY_CHANNELS_WITH_VALID else (name,)
        )
    )
    + CONTINUOUS_STATIC_CHANNELS
    + DAILY_TEMPORAL_CHANNELS
)
DAILY_INPUT_FEATURE_COUNT = len(DAILY_INPUT_CHANNELS)

_DAILY_INPUT_NORMALIZED_NAMES = (
    DAILY_ERA5_INPUT_CHANNELS
    + DAILY_CHANNELS
    + CONTINUOUS_STATIC_CHANNELS
)
DAILY_INPUT_NORMALIZED_INDICES = tuple(
    index
    for index, name in enumerate(DAILY_INPUT_CHANNELS)
    if name in _DAILY_INPUT_NORMALIZED_NAMES
)

# Fill network inputs only; validity masks retain the original missingness.
DAILY_INPUT_FILL_VALUES = {
    "ntl": 148.0,
    # Finite non-sentinel medians; exact-zero sentinels are masked before filling.
    "albedo": 0.132,
    "ndvi": 0.318,
    "ndbi": -0.083,
    "sar_vv": -7.46,
    "sar_vh": -14.88,
}
DAILY_INPUT_FILL = np.array(
    [DAILY_INPUT_FILL_VALUES.get(name, 0.0) for name in DAILY_INPUT_CHANNELS],
    dtype=np.float32,
)


def fill_daily_input(features: np.ndarray) -> np.ndarray:
    """Fill non-finite values in [D, C, H, W] inputs without modifying the original array."""
    if features.ndim != 4 or features.shape[1] != DAILY_INPUT_FEATURE_COUNT:
        raise ValueError(
            f"fill_daily_input expects [D, {DAILY_INPUT_FEATURE_COUNT}, H, W], "
            f"got {tuple(features.shape)}"
        )
    return np.where(
        np.isfinite(features), features, DAILY_INPUT_FILL[None, :, None, None]
    ).astype(np.float32)


# These source channels use exact zero as NoData. NTL uses NaN.
DAILY_ZERO_SENTINEL_CHANNELS = ("albedo", "ndvi", "ndbi", "sar_vv", "sar_vh")

# Physical ranges; positive SAR backscatter is valid, so no upper bound is applied.
DAILY_PHYSICAL_RANGES = {
    "albedo": (0.0, 1.0),
    "ndvi": (-1.0, 1.0),
    "ndbi": (-1.0, 1.0),
}


def mask_daily_sentinels(raw: dict) -> dict:
    """Convert source NoData sentinels to NaN before computing validity masks."""
    out = dict(raw)
    for name in DAILY_ZERO_SENTINEL_CHANNELS:
        if name not in out:
            continue
        values = np.asarray(out[name], dtype=np.float32).copy()
        bad = values == 0.0
        low, high = DAILY_PHYSICAL_RANGES.get(name, (-np.inf, np.inf))
        bad |= (values < low) | (values > high)
        values[bad] = np.nan
        out[name] = values
    return out

REQUIRED_BATCH_KEYS = (
    "daily_input_features",
    "daily_clcz_class",
    "kg_class",
    "phys_daily",
    "ta_reference_daily",
    "ta_reference_valid",
    "ts_reference_daily",
    "ts_reference_valid",
)

REQUIRED_MODEL_OUTPUT_KEYS = (
    "Ta_daily",
    "Ts_daily",
    "daily_state",
)


def validate_daily_batch(batch: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(f"Missing required daily batch keys: {missing}")
    features = batch["daily_input_features"]
    if getattr(features, "ndim", None) != 5:
        raise ValueError("daily_input_features must have shape [B, D, C, H, W]")
    if int(features.shape[2]) != DAILY_INPUT_FEATURE_COUNT:
        raise ValueError(
            f"daily_input_features expects {DAILY_INPUT_FEATURE_COUNT} channels, got {features.shape[2]}"
        )

    if tuple(features.shape[-2:]) != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"Daily input must use {TILE_SIZE}x{TILE_SIZE}")
    valid = batch.get("daily_input_valid")
    if valid is not None and tuple(valid.shape) != tuple(features.shape):
        raise ValueError("daily_input_valid must match the numerical feature tensor")
    clcz = batch["daily_clcz_class"]
    expected_clcz_shape = (features.shape[0], features.shape[1], features.shape[3], features.shape[4])
    if getattr(clcz, "ndim", None) != 4 or tuple(clcz.shape) != tuple(expected_clcz_shape):
        raise ValueError(
            "daily_clcz_class must have shape [B, D, H, W] matching daily_input_features; "
            f"expected {tuple(expected_clcz_shape)}, got {getattr(clcz, 'shape', None)}"
        )
    if clcz.numel() and (int(clcz.min()) < 0 or int(clcz.max()) >= N_CLCZ_CLASSES):
        raise ValueError(f"daily_clcz_class indices must be in [0, {N_CLCZ_CLASSES - 1}]")

    kg = batch["kg_class"]
    static_kg_shape = (features.shape[0], features.shape[3], features.shape[4])
    daily_kg_shape = (features.shape[0], features.shape[1], features.shape[3], features.shape[4])
    if getattr(kg, "ndim", None) not in (3, 4) or tuple(kg.shape) not in (
        tuple(static_kg_shape),
        tuple(daily_kg_shape),
    ):
        raise ValueError(
            "kg_class must have shape [B, H, W] or [B, D, H, W] matching daily_input_features; "
            f"got {getattr(kg, 'shape', None)}"
        )
    if kg.numel() and (int(kg.min()) < 0 or int(kg.max()) >= N_KG_CLASSES):
        raise ValueError(f"kg_class indices must be in [0, {N_KG_CLASSES - 1}]")

    reference_shape = (
        features.shape[0],
        features.shape[1],
        len(TEMP_CHANNELS),
        features.shape[3],
        features.shape[4],
    )
    valid_shape = (
        features.shape[0],
        features.shape[1],
        features.shape[3],
        features.shape[4],
    )
    for prefix in ("ta", "ts"):
        reference = batch[f"{prefix}_reference_daily"]
        valid = batch[f"{prefix}_reference_valid"]
        if getattr(reference, "ndim", None) != 5 or tuple(reference.shape) != tuple(reference_shape):
            raise ValueError(
                f"{prefix}_reference_daily must have shape {tuple(reference_shape)} in "
                f"{TEMP_CHANNELS} order; got {getattr(reference, 'shape', None)}"
            )
        if getattr(valid, "ndim", None) != 4 or tuple(valid.shape) != tuple(valid_shape):
            raise ValueError(
                f"{prefix}_reference_valid must have shape {tuple(valid_shape)}; "
                f"got {getattr(valid, 'shape', None)}"
            )


def validate_model_outputs(outputs: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_MODEL_OUTPUT_KEYS if key not in outputs]
    if missing:
        raise KeyError(f"Missing required DailyMUPNet outputs: {missing}")
    for key in ("Ta_daily", "Ts_daily"):
        value = outputs[key]
        if getattr(value, "ndim", None) != 5 or int(value.shape[2]) != len(TEMP_CHANNELS):
            raise ValueError(
                f"{key} must have shape [B, D, 3, H, W] in {TEMP_CHANNELS} order; "
                f"got {getattr(value, 'shape', None)}"
            )
        if tuple(value.shape[-2:]) != (CENTER_SIZE, CENTER_SIZE):
            raise ValueError(
                f"{key} must use the canonical {CENTER_SIZE}x{CENTER_SIZE} output patch; "
                f"got {tuple(value.shape[-2:])}"
            )

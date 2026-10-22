"""One daily feature assembly path for dataset reads and normalization fitting."""
import numpy as np
import pandas as pd

from DataCube.schema import ERA5, TEXTURES, TIME_ORIGIN
from ..contracts import (
    DAILY_INPUT_CHANNELS, DAILY_CHANNELS_WITH_VALID, mask_daily_sentinels,
)


def read_daily_features(h5, start, end):
    days = h5["time/day_index"][start:end].astype(np.int64)
    if len(days) != end-start or not np.all(np.diff(days) == 1):
        raise ValueError("The requested complete daily window is unavailable")
    dates = pd.DatetimeIndex(pd.Timestamp(TIME_ORIGIN)+pd.to_timedelta(days, unit="D"))
    raw = {name: h5[f"daily/era5/{name}"][start:end] for name in ERA5}
    raw.update({name: h5[f"daily/{name}"][start:end].astype(np.float32) for name in TEXTURES})
    # Mask source zero sentinels before constructing validity channels.
    raw = mask_daily_sentinels(raw)
    channels = dict(raw)
    for name in DAILY_CHANNELS_WITH_VALID:
        channels[f"{name}_valid"] = np.isfinite(raw[name]).astype(np.float32)
    spatial_shape = raw[ERA5[0]].shape
    for name in ("dem", "cmtbh", "svf"):
        channels[name] = np.broadcast_to(h5[f"static/{name}"][:], spatial_shape)
    phase = 2*np.pi*(dates.dayofyear.to_numpy()-1)/np.where(dates.is_leap_year, 366, 365)
    for name, values in (("temporal_sin_d", np.sin(phase)), ("temporal_cos_d", np.cos(phase)),
                         ("temporal_weekend", (dates.dayofweek >= 5).astype(float))):
        channels[name] = np.broadcast_to(np.asarray(values)[:, None, None], spatial_shape)
    features = np.stack([channels[name] for name in DAILY_INPUT_CHANNELS], axis=1).astype(np.float32)
    year_map = {int(year): i for i, year in enumerate(h5["time/year"][:])}
    try:
        indices = [year_map[int(year)] for year in dates.year]
    except KeyError as exc:
        raise ValueError("CLCZ year axis does not cover requested days") from exc
    clcz = h5["annual/clcz"][:][indices].astype(np.int64)
    return features, raw, clcz, days

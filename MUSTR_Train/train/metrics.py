"""Pool epoch accuracy sums in Kelvin; bias is prediction minus observation."""

from __future__ import annotations

import math

import torch


ACCURACY_SOURCES = {
    "air": ("min", "mean", "max"),
    "gst": ("min", "mean", "max"),
    "sat": ("obs",),
}


def error_sums(source, channel, count, error, absolute, square, obs=0.0, obs_square=0.0):
    """Return named error and observation sums for pooled metrics."""
    prefix = f"acc_{source}_{channel}"
    return {
        f"{prefix}_count": float(count),
        f"{prefix}_err_sum": float(error),
        f"{prefix}_ae_sum": float(absolute),
        f"{prefix}_se_sum": float(square),
        f"{prefix}_obs_sum": float(obs),
        f"{prefix}_obs_sq_sum": float(obs_square),
    }


def error_terms(source, channel, prediction, observation):
    """Accumulate masked errors in float32 to avoid reduced-precision squared sums."""
    observation = observation.detach().float().flatten()
    difference = prediction.detach().float().flatten() - observation
    error, absolute, square, obs, obs_square = torch.stack(
        (difference.sum(), difference.abs().sum(), difference.square().sum(),
         observation.sum(), observation.square().sum())
    ).tolist()
    return error_sums(source, channel, difference.numel(), error, absolute, square, obs, obs_square)


def reduce_accuracy(totals):
    """Reduce epoch sums once: rmse = sqrt(sum(e^2)/N) over all valid terms."""
    result = {}
    for source, channels in ACCURACY_SOURCES.items():
        for channel in channels:
            prefix = f"acc_{source}_{channel}"
            count = float(totals.get(f"{prefix}_count", 0.0))
            if count <= 0.0:
                continue
            square_error = totals[f"{prefix}_se_sum"]
            result[f"rmse_{source}_{channel}_K"] = math.sqrt(square_error / count)
            result[f"mae_{source}_{channel}_K"] = totals[f"{prefix}_ae_sum"] / count
            result[f"bias_{source}_{channel}_K"] = totals[f"{prefix}_err_sum"] / count
            result[f"n_{source}_{channel}"] = count
            # R2 is undefined for fewer than two samples or zero observation variance.
            obs_sum = float(totals.get(f"{prefix}_obs_sum", 0.0))
            total_square = float(totals.get(f"{prefix}_obs_sq_sum", 0.0)) - obs_sum * obs_sum / count
            if total_square > 0.0:
                result[f"r2_{source}_{channel}"] = 1.0 - square_error / total_square
    return result


def station_rmse(reduced, source):
    """Average available min/mean/max channel RMSEs in Kelvin; return NaN without data."""
    values = [reduced[f"rmse_{source}_{channel}_K"]
              for channel in ACCURACY_SOURCES[source]
              if f"rmse_{source}_{channel}_K" in reduced]
    return sum(values) / len(values) if values else float("nan")


__all__ = ["ACCURACY_SOURCES", "error_sums", "error_terms", "reduce_accuracy", "station_rmse"]

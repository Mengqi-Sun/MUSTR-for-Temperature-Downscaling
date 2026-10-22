"""Plot temperature accuracy and active objectives from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Set the headless backend before importing pyplot.
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)

PANELS = (("air", "min"), ("air", "mean"), ("air", "max"),
          ("gst", "min"), ("gst", "mean"), ("gst", "max"))
ROW_LABELS = ("Air temperature (TEM)", "Ground surface temperature (GST)",
              "Pooled error and raw objectives")
SPLITS = (("train", "--", 0.55), ("val", "-", 1.0))
COLORS = {"rmse": "tab:blue", "mae": "tab:orange", "r2": "tab:green",
          "physics": "tab:purple", "sat": "tab:red"}
R2_FLOOR = -1.0
FIGURE_NAME = "accuracy_trends.png"


def read_epochs(metrics_path: Path) -> list[dict]:
    """One record per epoch, ordered; after a resume the last line for an epoch wins."""
    records: dict[int, dict] = {}
    with metrics_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # Ignore a truncated final JSONL record.
            if "epoch" in record:
                records[int(record["epoch"])] = record
    return [records[key] for key in sorted(records)]


def _number(record: dict, key: str) -> float | None:
    """The finite float at ``key``, or None; bools are not numbers here."""
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Epochs and values where ``key`` is present and finite; gaps are dropped."""
    epochs: list[int] = []
    values: list[float] = []
    for record in records:
        value = _number(record, key)
        if value is not None:
            epochs.append(int(record["epoch"]))
            values.append(value)
    return epochs, values


def phase_boundaries(records: list[dict]) -> list[tuple[int, int]]:
    """First epoch of each phase after the first, as ``(epoch, phase)``."""
    boundaries: list[tuple[int, int]] = []
    previous = None
    for record in records:
        phase = record.get("phase")
        if phase is None:
            continue
        phase = int(phase)
        if previous is not None and phase != previous:
            boundaries.append((int(record["epoch"]), phase))
        previous = phase
    return boundaries


def pooled_error(record: dict, split: str) -> tuple[float, float] | None:
    """Compute count-weighted RMSE and MAE across all six temperature channels."""
    square_error = absolute_error = count = 0.0
    for source, channel in PANELS:
        prefix = f"{split}/acc_{source}_{channel}"
        channel_count = _number(record, f"{prefix}_count")
        channel_square = _number(record, f"{prefix}_se_sum")
        channel_absolute = _number(record, f"{prefix}_ae_sum")
        if channel_count is None or channel_square is None or channel_absolute is None:
            # Reconstruct pooled sums for older metric records.
            channel_count = _number(record, f"{split}/n_{source}_{channel}")
            rmse = _number(record, f"{split}/rmse_{source}_{channel}_K")
            mae = _number(record, f"{split}/mae_{source}_{channel}_K")
            if channel_count is None or rmse is None or mae is None:
                continue
            channel_square, channel_absolute = rmse * rmse * channel_count, mae * channel_count
        if channel_count <= 0.0:
            continue
        square_error += channel_square
        absolute_error += channel_absolute
        count += channel_count
    if count <= 0.0:
        return None
    return math.sqrt(square_error / count), absolute_error / count


def active_series(records: list[dict], split: str, term: str) -> tuple[list[int], list[float]]:
    """Return raw losses only for epochs where the objective has positive weight."""
    epochs: list[int] = []
    values: list[float] = []
    started = False
    for record in records:
        raw = _number(record, f"{split}/L_{term}_raw")
        if raw is None:
            continue
        weighted = _number(record, f"{split}/L_{term}_weighted")
        active = weighted > 0.0 if weighted is not None else raw != 0.0
        if not active:
            continue
        started = True
        epochs.append(int(record["epoch"]))
        values.append(raw)
    return (epochs, values) if started else ([], [])


def _decorate(axis, boundaries: list[tuple[int, int]], ylabel: str) -> None:
    for epoch, phase in boundaries:
        axis.axvline(epoch, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
        axis.annotate(f"P{phase}", xy=(epoch, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(2, -10), textcoords="offset points", fontsize=7, color="0.45")
    axis.set_ylabel(ylabel, fontsize=9)
    axis.set_xlabel("epoch", fontsize=9)
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.tick_params(labelsize=8)


def _empty(axis, title: str, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes,
              ha="center", va="center", color="0.5", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=10, loc="left")


def _draw_channel_panel(axis, records: list[dict], source: str, channel: str,
                        boundaries: list[tuple[int, int]]) -> tuple[bool, bool]:
    """Draw channel errors and available R2 values; return which curves were drawn."""
    right = axis.twinx()
    drawn = False
    has_r2 = False
    clamped = False

    for split, style, alpha in SPLITS:
        for name, key in (("rmse", f"{split}/rmse_{source}_{channel}_K"),
                          ("mae", f"{split}/mae_{source}_{channel}_K")):
            epochs, values = series(records, key)
            if not epochs:
                continue
            axis.plot(epochs, values, style, color=COLORS[name], alpha=alpha, linewidth=1.6)
            drawn = True

        epochs, values = series(records, f"{split}/r2_{source}_{channel}")
        if epochs:
            floored = [max(value, R2_FLOOR) for value in values]
            clamped = clamped or floored != values
            right.plot(epochs, floored, style, color=COLORS["r2"], alpha=alpha, linewidth=1.6)
            drawn = has_r2 = True

    if not has_r2:
        right.axis("off")
    if not drawn:
        _empty(axis, f"{source.upper()} {channel}", "no supervision")
        return False, False

    _decorate(axis, boundaries, "RMSE / MAE (K)")
    if has_r2:
        right.set_ylabel("R$^2$" + (f" (clipped at {R2_FLOOR:g})" if clamped else ""),
                         fontsize=9, color=COLORS["r2"])
        right.tick_params(axis="y", labelsize=8, labelcolor=COLORS["r2"])
        right.axhline(0.0, color=COLORS["r2"], linewidth=0.6, alpha=0.35, zorder=0)

    best_epochs, best_values = series(records, f"val/rmse_{source}_{channel}_K")
    title = f"{source.upper()} {channel}"
    if best_values:
        index = min(range(len(best_values)), key=best_values.__getitem__)
        title += f"   best val RMSE {best_values[index]:.3f} K @ epoch {best_epochs[index]}"
    axis.set_title(title, fontsize=10, loc="left")
    return True, has_r2


def _draw_total_panel(axis, records: list[dict], boundaries: list[tuple[int, int]]) -> bool:
    """Pooled RMSE and MAE over all six channels, on one Kelvin axis."""
    best: tuple[float, int] | None = None
    drawn = False
    for split, style, alpha in SPLITS:
        epochs: list[int] = []
        rmse_values: list[float] = []
        mae_values: list[float] = []
        for record in records:
            pooled = pooled_error(record, split)
            if pooled is None:
                continue
            epochs.append(int(record["epoch"]))
            rmse_values.append(pooled[0])
            mae_values.append(pooled[1])
        if not epochs:
            continue
        axis.plot(epochs, rmse_values, style, color=COLORS["rmse"], alpha=alpha, linewidth=1.6)
        axis.plot(epochs, mae_values, style, color=COLORS["mae"], alpha=alpha, linewidth=1.6)
        drawn = True
        if split == "val":
            index = min(range(len(rmse_values)), key=rmse_values.__getitem__)
            best = (rmse_values[index], epochs[index])

    title = "ALL 6 channels pooled"
    if not drawn:
        _empty(axis, title, "no supervision")
        return False
    _decorate(axis, boundaries, "pooled RMSE / MAE (K)")
    if best is not None:
        title += f"   best val RMSE {best[0]:.3f} K @ epoch {best[1]}"
    axis.set_title(title, fontsize=10, loc="left")
    return True


def _draw_objective_panel(axis, records: list[dict], term: str, label: str,
                          boundaries: list[tuple[int, int]]) -> bool:
    """Raw loss for one curriculum objective, over the epochs where it is active."""
    start = end = None
    drawn = False
    for split, style, alpha in SPLITS:
        epochs, values = active_series(records, split, term)
        if not epochs:
            continue
        axis.plot(epochs, values, style, color=COLORS[term], alpha=alpha, linewidth=1.6)
        start = epochs[0] if start is None else min(start, epochs[0])
        end = epochs[-1] if end is None else max(end, epochs[-1])
        drawn = True

    title = f"{label} loss (raw)"
    if not drawn:
        _empty(axis, title, f"{label} not active in the logged epochs")
        return False
    # Restrict phase markers to epochs where this objective is active.
    _decorate(axis, [entry for entry in boundaries if start <= entry[0] <= end], "raw loss")
    if start > int(records[0]["epoch"]):
        title += f"   active from epoch {start}"
        if start not in {epoch for epoch, _ in boundaries}:
            axis.axvline(start, color=COLORS[term], linewidth=1.0, alpha=0.5, zorder=0)
    axis.set_title(title, fontsize=10, loc="left")
    return True


def plot_accuracy_trends(log_dir: str | Path, metrics_name: str = "metrics.jsonl") -> Path | None:
    """Save accuracy_trends.png beside the log; return None when no records are available."""
    log_dir = Path(log_dir)
    metrics_path = log_dir / metrics_name
    if not metrics_path.exists():
        LOGGER.warning("No %s under %s; skipping accuracy trend plot", metrics_name, log_dir)
        return None

    records = read_epochs(metrics_path)
    if not records:
        LOGGER.warning("%s holds no epoch records; skipping accuracy trend plot", metrics_path)
        return None

    boundaries = phase_boundaries(records)
    figure, axes = plt.subplots(3, 3, figsize=(18, 13.5), constrained_layout=True)
    populated = 0
    any_r2 = False
    for axis, (source, channel) in zip(axes[:2].ravel(), PANELS):
        drawn, has_r2 = _draw_channel_panel(axis, records, source, channel, boundaries)
        populated += int(drawn)
        any_r2 = any_r2 or has_r2
    populated += int(_draw_total_panel(axes[2, 0], records, boundaries))
    _draw_objective_panel(axes[2, 1], records, "physics", "Physics", boundaries)
    _draw_objective_panel(axes[2, 2], records, "sat", "Landsat", boundaries)

    if populated == 0:
        plt.close(figure)
        LOGGER.warning("No air/gst accuracy metrics in %s; skipping plot", metrics_path)
        return None

    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].text(-0.17, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, ha="center", va="center", fontsize=11)

    entries = [("rmse", "RMSE (K)"), ("mae", "MAE (K)")] + ([("r2", "R$^2$")] if any_r2 else [])
    handles = [plt.Line2D([], [], color=COLORS[name], linewidth=1.6, label=label)
               for name, label in entries]
    handles += [plt.Line2D([], [], color="0.35", linestyle=style, linewidth=1.6,
                           alpha=alpha, label=split)
                for split, style, alpha in SPLITS]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                  frameon=False, fontsize=9)
    figure.suptitle(f"Accuracy trends - {log_dir.parent.name}   ({len(records)} epochs)",
                    fontsize=13)

    output = log_dir / FIGURE_NAME
    figure.savefig(output, dpi=150)
    plt.close(figure)
    LOGGER.info("Accuracy trend plot written to %s", output)
    return output


def resolve_log_dir(run: Path) -> Path:
    """Accept either a run directory or its ``tensorboard`` subdirectory."""
    candidate = run / "tensorboard"
    return candidate if candidate.is_dir() else run


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RMSE/MAE/R2 accuracy trends for a run")
    parser.add_argument("--run", required=True,
                        help="Run directory, or the tensorboard directory holding metrics.jsonl")
    parser.add_argument("--metrics-name", default="metrics.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = plot_accuracy_trends(resolve_log_dir(Path(args.run)), args.metrics_name)
    if output is None:
        raise SystemExit(1)
    print(output)


if __name__ == "__main__":
    main()


__all__ = ["plot_accuracy_trends", "pooled_error", "active_series", "read_epochs", "resolve_log_dir"]

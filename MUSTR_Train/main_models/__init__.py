"""Daily MUPNet model components."""

from .mupnet import DailyMUPNet
from .surface_time_operator import SurfaceTimeOperator


def build_model(config):
    if "task_mode" in config.get("model", {}):
        raise ValueError("Retired model.task_mode: PyMUPNet is daily-only.")
    return DailyMUPNet(config)


__all__ = ["DailyMUPNet", "SurfaceTimeOperator", "build_model"]

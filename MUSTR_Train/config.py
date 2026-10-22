"""Configuration loading, path resolution, and physics validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

FORMULA_VERSION = "simple_daily_fixed_band_v1"
DAYNIGHT_FORMULA_VERSION = "daynight_temperature_v1"
PHYSICS_MODES = (FORMULA_VERSION, DAYNIGHT_FORMULA_VERSION)
# Default to day/night physics; retain the 24-hour mode for compatibility.
DEFAULT_PHYSICS_MODE = DAYNIGHT_FORMULA_VERSION

PHYSICS_COMMON_KEYS = {"enabled", "mode", "weight", "phase_overrides",
                       "priors", "priors_file"}
# Reject settings from a different physics formulation.
PHYSICS_MODE_KEYS = {
    FORMULA_VERSION: {"flux_scale_w_m2", "residual_tolerance_w_m2", "huber_transition"},
    DAYNIGHT_FORMULA_VERSION: {"air_source", "sigma_forcing_w_m2", "tolerance_k",
                               "huber_delta_sigma", "newton_steps", "bracket_k",
                               "curve_samples", "windows"},
}

PHYSICS_WINDOWS = ("day", "night")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PROJECT_PATH_FIELDS = (
    ("data", "h5_dir"),
    ("data", "station_csv"),
    ("normalization", "stats_file"),
    ("normalization", "daily_input_stats_file"),
    ("physics", "priors_file"),
    ("experiment", "output_root"),
    ("training", "init_from"),
    ("training", "resume"),
)


def cfg_get(config: Any, path: str, default: Any = None) -> Any:
    """Read a dotted key from mappings or simple configuration objects."""
    value = config
    for key in path.split("."):
        if isinstance(value, Mapping):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path using the repository root as the only relative-path base."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_known_paths(config: dict[str, Any]) -> None:
    for section, key in _PROJECT_PATH_FIELDS:
        container = config.get(section)
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if isinstance(value, str) and value.strip():
            container[key] = str(resolve_project_path(value))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(config_path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Cyclic config inheritance: {chain}")
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    base_value = payload.pop("base", None)
    if not base_value:
        return payload
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_path).resolve()
    return _deep_merge(_load_yaml(base_path, (*stack, config_path)), payload)


def _inline_physics_priors(config: dict[str, Any]) -> None:
    """Inline external priors so resolved configurations retain calibrated values."""
    section = config.get("physics")
    if not isinstance(section, dict):
        return
    source = section.pop("priors_file", None)
    if not source:
        return
    # An external priors file replaces inherited inline priors.
    path = resolve_project_path(source)
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"physics.priors_file must contain a mapping: {path}")
    section["priors"] = loaded


def load_config(path: str | Path, *, priors_file: str | Path | None = None) -> dict[str, Any]:
    """Load inherited YAML settings, optionally replacing the priors file."""
    config_path = resolve_project_path(path)

    config = _load_yaml(config_path)
    _resolve_known_paths(config)
    if priors_file is not None:
        config.setdefault("physics", {})["priors_file"] = str(priors_file)
    _inline_physics_priors(config)
    config["__config_path__"] = str(config_path)
    config["__project_root__"] = str(PROJECT_ROOT)
    return config


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serializable configuration without private runtime metadata."""
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("__")}


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(public_config(config), stream, allow_unicode=True, sort_keys=False)


@dataclass(frozen=True)
class PhysicsPolicy:
    enabled: bool
    mode: str
    weight: float


def physics_policy(config: dict, phase: int = 1) -> PhysicsPolicy:
    """One formulation, global defaults and explicit per-phase overrides."""
    loss_section = cfg_get(config, "loss", {}) or {}
    retired_names = {"dynamic_balance", "ema_momentum", "ema_eps"}
    retired_balance = retired_names & set(loss_section)
    for value in loss_section.values():
        if isinstance(value, Mapping):
            retired_balance.update(retired_names & set(value))
    if retired_balance:
        raise ValueError(
            f"Retired loss settings: {sorted(retired_balance)}; observation losses use fixed weights"
        )
    section = config.get("physics")
    if "physics" in config and not isinstance(section, dict):
        raise ValueError("physics must be a mapping; omit it for observation-only configuration")
    if section is None:
        section = {"enabled": False}
    legacy_weights = cfg_get(config, "loss", {}) or {}
    if ("lambda_physics" in legacy_weights
            or any(isinstance(v, dict) and "lambda_physics" in v for v in legacy_weights.values())
            or cfg_get(config, "training.physics_start_phase") is not None):
        raise ValueError("Retired physics settings: use physics.enabled/weight/phase_overrides")
    # Explicit null clears an inherited setting.
    section = {key: value for key, value in section.items() if value is not None}
    mode = section.get("mode", DEFAULT_PHYSICS_MODE)
    if mode not in PHYSICS_MODES:
        raise ValueError(f"Unknown physics.mode: {mode}")
    unknown = set(section) - PHYSICS_COMMON_KEYS - set().union(*PHYSICS_MODE_KEYS.values())
    if unknown:
        raise ValueError(f"Unknown physics settings: {sorted(unknown)}")
    foreign = sorted(set(section) & set().union(
        *(keys for other, keys in PHYSICS_MODE_KEYS.items() if other != mode)))
    if foreign:
        raise ValueError(
            f"physics.mode={mode} does not use {foreign}; they belong to the other "
            f"formulation and would be ignored. Remove them, or set them to null "
            f"to clear values inherited from a base config.")
    if "windows" in section:
        selected = section["windows"]
        if (not isinstance(selected, (list, tuple)) or not selected
                or len(set(selected)) != len(selected)
                or any(window not in PHYSICS_WINDOWS for window in selected)):
            raise ValueError(
                f"physics.windows must be a non-empty list of distinct values from "
                f"{list(PHYSICS_WINDOWS)}; got {selected!r}")
    enabled = section.get("enabled", True)
    weight = section.get("weight", 0.001)
    if not isinstance(enabled, bool) or not math.isfinite(float(weight)) or float(weight) < 0:
        raise ValueError("Physics requires boolean enabled and finite nonnegative weight")
    overrides = section.get("phase_overrides", {}) or {}
    for key, override in overrides.items():
        if key not in {f"phase{i}" for i in range(1, 5)}:
            raise ValueError(f"Unknown physics phase override: {key}")
        if not isinstance(override, dict) or set(override) - {"enabled", "weight"}:
            raise ValueError("Physics phase overrides support only enabled and weight")
        if "enabled" in override and not isinstance(override["enabled"], bool):
            raise ValueError("physics enabled settings must be booleans")
        if "weight" in override:
            candidate = float(override["weight"])
            if not math.isfinite(candidate) or candidate < 0:
                raise ValueError("physics weights must be finite and nonnegative")
    override = overrides.get(f"phase{phase}", {})
    enabled = override.get("enabled", enabled)
    weight = float(override.get("weight", weight))
    if not isinstance(enabled, bool) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Physics requires boolean enabled and finite nonnegative weight")
    # A zero weight disables physics reads and loss computation.
    return PhysicsPolicy(enabled and weight > 0, mode, weight)


def selection_weights(config: dict) -> dict[str, float]:
    """Validate that each source used for selection is supervised in an active phase."""
    raw = cfg_get(config, "training.selection_weights", {}) or {}
    unknown = set(raw) - {"air", "gst", "sat"}
    if unknown:
        raise ValueError(f"Unknown training.selection_weights keys: {sorted(unknown)}")
    resolved = {name: float(raw.get(name, 1.0)) for name in ("air", "gst", "sat")}
    if (any(not math.isfinite(value) or value < 0 for value in resolved.values())
            or sum(resolved.values()) <= 0):
        raise ValueError("training.selection_weights must be finite, nonnegative and not all zero")
    phases = [phase for phase in range(1, 5)
              if int(cfg_get(config, f"training.curriculum.phase{phase}_max_epochs", 0)) > 0]
    for name, weight in resolved.items():
        if weight <= 0:
            continue
        trained = max(
            (float((cfg_get(config, f"loss.phase{phase}", {}) or {}).get(
                f"lambda_{name}", cfg_get(config, f"loss.lambda_{name}", 0.0)))
             for phase in phases),
            default=0.0,
        )
        if trained <= 0:
            raise ValueError(
                f"training.selection_weights.{name} > 0 but no enabled phase sets lambda_{name} > 0"
            )
    return resolved


def uses_physics(config: dict) -> bool:
    return any(p.enabled
               for p in (physics_policy(config, phase) for phase in range(1, 5)))


PRIOR_GROUP_KEYS = {
    "simple_priors_v1": {"classes", "epsilon", "ra_ref", "rs", "capacity", "dry_paved"},
    "simple_priors_v2": {"classes", "epsilon", "ra_ref_day", "ra_ref_night", "rs",
                         "capacity", "api_half", "dry_paved"},
    # Emissivity and moisture half-saturation are shared between day and night.
    "simple_priors_v3": {"classes", "epsilon", "ra_ref_day", "ra_ref_night",
                         "rs_day", "rs_night", "capacity_day", "capacity_night",
                         "api_half", "stability", "dry_paved"},
    # v3-v6 share a schema; their versions identify different calibrations.
    "simple_priors_v4": {"classes", "epsilon", "ra_ref_day", "ra_ref_night",
                         "rs_day", "rs_night", "capacity_day", "capacity_night",
                         "api_half", "stability", "dry_paved"},
    # v5/v6 priors use 10 m Richardson geometry; earlier fits use 2 m.
    "simple_priors_v5": {"classes", "epsilon", "ra_ref_day", "ra_ref_night",
                         "rs_day", "rs_night", "capacity_day", "capacity_night",
                         "api_half", "stability", "dry_paved"},
    "simple_priors_v6": {"classes", "epsilon", "ra_ref_day", "ra_ref_night",
                         "rs_day", "rs_night", "capacity_day", "capacity_night",
                         "api_half", "stability", "dry_paved"},
}
# These schemas already contain all per-window parameters.
PRIOR_PER_WINDOW_VERSIONS = ("simple_priors_v3", "simple_priors_v4",
                             "simple_priors_v5", "simple_priors_v6")
PRIOR_WINDOW_KEYS = ("ra_ref_day", "ra_ref_night", "rs_day", "rs_night",
                     "capacity_day", "capacity_night")
PRIOR_UNITS = {
    "simple_priors_v1": {"epsilon": "1", "ra_ref": "s/m", "rs": "s/m",
                         "capacity": "J/m2/K", "qf": "W/m2"},
    "simple_priors_v2": {"epsilon": "1", "ra_ref_day": "s/m", "ra_ref_night": "s/m",
                         "rs": "s/m", "capacity": "J/m2/K", "api_half": "mm",
                         "qf": "W/m2"},
    "simple_priors_v3": {"epsilon": "1", "ra_ref_day": "s/m", "ra_ref_night": "s/m",
                         "rs_day": "s/m", "rs_night": "s/m",
                         "capacity_day": "J/m2/K", "capacity_night": "J/m2/K",
                         "api_half": "mm", "stability": "1", "qf": "W/m2"},
    "simple_priors_v4": {"epsilon": "1", "ra_ref_day": "s/m", "ra_ref_night": "s/m",
                         "rs_day": "s/m", "rs_night": "s/m",
                         "capacity_day": "J/m2/K", "capacity_night": "J/m2/K",
                         "api_half": "mm", "stability": "1", "qf": "W/m2"},
    "simple_priors_v5": {"epsilon": "1", "ra_ref_day": "s/m", "ra_ref_night": "s/m",
                         "rs_day": "s/m", "rs_night": "s/m",
                         "capacity_day": "J/m2/K", "capacity_night": "J/m2/K",
                         "api_half": "mm", "stability": "1", "qf": "W/m2"},
    "simple_priors_v6": {"epsilon": "1", "ra_ref_day": "s/m", "ra_ref_night": "s/m",
                         "rs_day": "s/m", "rs_night": "s/m",
                         "capacity_day": "J/m2/K", "capacity_night": "J/m2/K",
                         "api_half": "mm", "stability": "1", "qf": "W/m2"},
}


def physics_priors(config):
    """Validate priors and normalize legacy schemas to per-window parameters."""
    priors = deepcopy(cfg_get(config, "physics.priors"))
    required = {"version", "groups", "units", "wind", "qf", "quality"}
    if not isinstance(priors, dict) or set(priors) != required:
        raise ValueError(f"physics.priors keys must be {sorted(required)}")
    version = priors["version"]
    if version not in PRIOR_GROUP_KEYS:
        raise ValueError(f"physics.priors.version must be one of {sorted(PRIOR_GROUP_KEYS)}")
    classes = []
    for group in priors["groups"].values():
        if set(group) != PRIOR_GROUP_KEYS[version]:
            raise ValueError("Unknown or missing surface-group parameter")
        classes.extend(group["classes"])
        if version == "simple_priors_v1":
            # api_half = 0 disables the water-availability limiter.
            group["ra_ref_day"] = group["ra_ref_night"] = group["ra_ref"]
            group["api_half"] = 0.0
        if version not in PRIOR_PER_WINDOW_VERSIONS:
            # Upgrade older schemas; stability = 0 preserves neutral resistance.
            group["rs_day"] = group["rs_night"] = group["rs"]
            group["capacity_day"] = group["capacity_night"] = group["capacity"]
            group["stability"] = 0.0
        values = [group[k] for k in
                  ("epsilon", "api_half", "stability") + PRIOR_WINDOW_KEYS]
        if not all(math.isfinite(v) for v in values) or not (
            0 < group["epsilon"] <= 1
            and group["ra_ref_day"] > 0 and group["ra_ref_night"] > 0
            and group["rs_day"] >= 0 and group["rs_night"] >= 0
            and group["capacity_day"] > 0 and group["capacity_night"] > 0
            and group["api_half"] >= 0 and group["stability"] >= 0
            and isinstance(group["dry_paved"], bool)
        ):
            raise ValueError("Invalid surface-group prior")
    if sorted(classes) != list(range(1, 19)):
        raise ValueError("Each LCZ class 1..18 must occur exactly once")
    expected = {"wind": {"reference", "floor"},
                "qf": {"source", "profile_mean", "urban_fraction", "ah"},
                "quality": {"required_hours", "snow_fraction_max", "freezing_k", "wet_pair_mm", "low_confidence", "min_pixels"}}
    for section, keys in expected.items():
        if set(priors[section]) != keys:
            raise ValueError(f"Unknown/missing prior keys in {section}")
    if priors["units"] != PRIOR_UNITS[version]:
        raise ValueError("Prior units must match the physical coefficient contract")
    q, wind, qf = (priors[key] for key in ("quality", "wind", "qf"))
    if (q["required_hours"] != 24 or not 1 <= q["min_pixels"] <= 25 or int(q["min_pixels"]) != q["min_pixels"]
            or q["freezing_k"] <= 35.85
            or not 0 <= q["low_confidence"] <= 1 or not 0 <= q["snow_fraction_max"] <= 1
            or q["wet_pair_mm"] < 0 or wind["reference"] <= 0 or wind["floor"] <= 0
            or len(qf["urban_fraction"]) != 10 or len(qf["ah"]) != 10
            or not 0 <= qf["profile_mean"] <= 1
            or any(not 0 <= v <= 1 for v in qf["urban_fraction"])
            or any(v < 0 for v in qf["ah"])):
        raise ValueError("Invalid quality/QF settings")
    json.dumps(priors, allow_nan=False)  # Reject non-finite settings.
    return priors

"""Model artifact and preprocessing identity helpers."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from ..config import cfg_get as _cfg_get, resolve_project_path
from ..contracts import MODEL_CONTRACT_VERSION, DAILY_INPUT_CHANNELS
from DataCube.schema import SCHEMA_VERSION

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_identity(config):
    normal = _cfg_get(config, "normalization", {})
    enabled = bool(normal.get("enabled", normal.get("enable_daily_input_standardize", False)))
    stats = normal.get("stats_file", normal.get("daily_input_stats_file"))
    model = deepcopy(_cfg_get(config, "model", {}))
    # Ignore retired fields that do not affect the daily model.
    model.pop("task_mode", None)
    model.get("daily", {}).pop("physics", None)
    return {"h5_schema": SCHEMA_VERSION, "contract": MODEL_CONTRACT_VERSION, "features": list(DAILY_INPUT_CHANNELS),
            "model": model,
            "kg_embed_dim": _cfg_get(config, "data.kg_embed_dim", 2),
            "clcz_embed_dim": _cfg_get(config, "data.clcz_embed_dim", 16),
            "normalization_enabled": enabled,
            "normalizer_sha256": file_sha(resolve_project_path(stats)) if enabled else None,
            # Fingerprint the inlined priors, independent of their source path.
            "physics_priors_sha256": _priors_fingerprint(config)}


def _priors_fingerprint(config):
    priors = _cfg_get(config, "physics.priors")
    return fingerprint(priors) if isinstance(priors, dict) else None

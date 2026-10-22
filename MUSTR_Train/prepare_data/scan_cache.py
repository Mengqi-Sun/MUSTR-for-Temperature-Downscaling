"""Cache window-index and supervision scans using cube-directory signatures."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)
CACHE_DIRNAME = ".mupnet_scan_cache"


def directory_signature(h5_dir):
    """Hash entry names and directory mtime; atomic cube replacement invalidates the cache."""
    path = Path(h5_dir)
    with os.scandir(path) as entries:
        names = sorted(entry.name for entry in entries if entry.name.endswith(".h5"))
    return {"dir": str(path.resolve()), "mtime_ns": path.stat().st_mtime_ns,
            "count": len(names), "names_sha": hashlib.sha256(
                "\n".join(names).encode("utf-8")).hexdigest()}


def _key(parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()[:32]


def _cache_dir(h5_dir):
    """Prefer a cache beside the data; fall back to the scratch directory."""
    candidate = Path(h5_dir) / CACHE_DIRNAME
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        # Use process-specific probe names to avoid concurrent-run collisions.
        probe = candidate / f".writable.{os.getpid()}"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / CACHE_DIRNAME
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except OSError:
            return None


def load_or_build(h5_dir, name, parts, builder, save, load):
    """Load a cached scan or rebuild it; cache failures fall back to scanning."""
    directory = _cache_dir(h5_dir)
    if directory is None:
        return builder()
    path = directory / f"{name}-{_key(parts)}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as stored:
                value = load(stored)
            LOGGER.info("Reused the cached %s scan: %s", name, path.name)
            return value
        except Exception:                                    # noqa: BLE001
            LOGGER.warning("Ignoring an unreadable %s cache at %s", name, path)
    value = builder()
    try:
        # Keep the .npz suffix and a process-specific name for atomic replacement.
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(temporary, **save(value))
        temporary.replace(path)
        LOGGER.info("Cached the %s scan for the next run: %s", name, path.name)
    except Exception:                                        # noqa: BLE001
        LOGGER.warning("Could not write the %s cache under %s", name, directory)
    return value


def save_index_map(index_map):
    stations = sorted({sid for sid, _, _ in index_map})
    position = {sid: i for i, sid in enumerate(stations)}
    return {"stations": np.array(stations, dtype=object).astype("U"),
            "station": np.fromiter((position[s] for s, _, _ in index_map),
                                   dtype=np.int32, count=len(index_map)),
            "start": np.fromiter((s for _, s, _ in index_map),
                                 dtype=np.int32, count=len(index_map)),
            "end": np.fromiter((e for _, _, e in index_map),
                               dtype=np.int32, count=len(index_map))}


def load_index_map(stored):
    stations = [str(s) for s in stored["stations"]]
    return [(stations[i], int(a), int(b)) for i, a, b in
            zip(stored["station"], stored["start"], stored["end"])]

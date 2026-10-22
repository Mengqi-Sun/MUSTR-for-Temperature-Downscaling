from functools import partial
from typing import Any, Dict, Iterator, List, Optional

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from ..config import cfg_get as _cfg_get
from .dataset import MUPNetDataset


from .indexing import (
    build_index_map_from_config as _build_index_map,
    build_station_df_from_config as _build_station_df,
)


def _should_shuffle_loader(config: Any, split: str) -> bool:
    if split == "train":
        return True
    if split != "val":
        return False
    if not bool(_cfg_get(config, "training.shuffle_limited_val", True)):
        return False
    return int(_cfg_get(config, "training.max_val_batches", 0)) > 0


def _split_seed_offset(split: str) -> int:
    return {"train": 101, "val": 202, "test": 303}.get(split, 0)


def _build_loader_generator(seed: Optional[int], split: str) -> Optional[torch.Generator]:
    gen_seed = seed
    if gen_seed is None and split == "val":
        gen_seed = 42
    if gen_seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(gen_seed) + _split_seed_offset(split))
    return generator


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (int(base_seed) + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_worker_init_fn(seed: Optional[int], split: str):
    if seed is None:
        return None
    base_seed = int(seed) + _split_seed_offset(split) * 1000
    return partial(_seed_worker, base_seed=base_seed)


def mupnet_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    B = len(batch)

    daily_labels = {
        key: torch.stack([item["daily_labels"][key] for item in batch], dim=0)
        for key in ["tem_mean", "tem_max", "tem_min"]
    }
    daily_gst_labels = {
        key: torch.stack([item["daily_gst_labels"][key] for item in batch], dim=0)
        for key in ["gst_mean", "gst_max", "gst_min"]
    }
    phys_daily = {}
    if B > 0 and isinstance(batch[0].get("phys_daily"), dict):
        for key in batch[0]["phys_daily"].keys():
            phys_daily[key] = torch.stack([item["phys_daily"][key] for item in batch], dim=0)
    timing_daily = {}
    if B > 0 and isinstance(batch[0].get("timing_daily"), dict):
        for key in batch[0]["timing_daily"].keys():
            timing_daily[key] = torch.stack([item["timing_daily"][key] for item in batch], dim=0)

    collated = {
        "daily_input_features": torch.stack([item["daily_input_features"] for item in batch], dim=0),
        "daily_input_valid": torch.stack([item["daily_input_valid"] for item in batch], dim=0),
        "daily_clcz_class": torch.stack([item["daily_clcz_class"] for item in batch], dim=0).long(),
        "daily_labels": daily_labels,
        "daily_gst_labels": daily_gst_labels,
        "lst_obs": torch.stack([item["lst_obs"] for item in batch], dim=0),
        "lst_hour": torch.stack([item["lst_hour"] for item in batch], dim=0),
        "lst_valid": torch.stack([item["lst_valid"] for item in batch], dim=0).bool(),
        "timing_daily": timing_daily,
        "is_real": torch.tensor([bool(item["is_real"]) for item in batch], dtype=torch.bool),
        "ta_reference_daily": torch.stack([item["ta_reference_daily"] for item in batch], dim=0),
        "ta_reference_valid": torch.stack([item["ta_reference_valid"] for item in batch], dim=0).bool(),
        "ts_reference_daily": torch.stack([item["ts_reference_daily"] for item in batch], dim=0),
        "ts_reference_valid": torch.stack([item["ts_reference_valid"] for item in batch], dim=0).bool(),
        "phys_daily": phys_daily,
        "kg_class": torch.stack([item["kg_class"] for item in batch], dim=0).long(),
    }
    has_physics = ["physics" in item for item in batch]
    if any(has_physics):
        if not all(has_physics):
            raise ValueError("Cannot collate mixed physics/non-physics samples")
        keys = set(batch[0]["physics"])
        if any(set(item["physics"]) != keys for item in batch):
            raise ValueError("Mixed physics field schemas in one batch")
        collated["physics"] = {key: torch.stack([item["physics"][key] for item in batch]) for key in sorted(keys)}
        collated["sample_key"] = [item["sample_key"] for item in batch]
    return collated


class SemiConstrainedWindowSampler(Sampler[int]):
    def __init__(
        self,
        supervision_stats: List[Dict[str, Any]],
        batch_size: int,
        *,
        min_tem_daily: int = 8,
        min_gst_daily: int = 6,
        min_landsat: int = 4,
        prefer_overlap: int = 6,
        prefer_gst_landsat_overlap: int = 0,
        seed: int = 42,
        drop_last: bool = True,
        station_ids: Optional[List[str]] = None,
        station_bucket_enabled: bool = False,
        min_unique_stations_per_batch: int = 8,
        max_unique_stations_per_batch: int = 32,
        target_unique_station_ratio: float = 0.25,
        target_samples_per_station: int = 4,
        station_bucket_strict: bool = True,
        station_bucket_max_attempts: int = 12,
    ):
        self.supervision_stats = supervision_stats
        self.batch_size = int(max(1, batch_size))
        self.min_tem_daily = int(max(0, min_tem_daily))
        self.min_gst_daily = int(max(0, min_gst_daily))
        self.min_landsat = int(max(0, min_landsat))
        self.prefer_overlap = int(max(0, prefer_overlap))
        self.prefer_gst_landsat_overlap = int(max(0, prefer_gst_landsat_overlap))
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.station_ids = list(station_ids) if station_ids is not None else []
        self.station_bucket_enabled = bool(station_bucket_enabled) and len(self.station_ids) == len(self.supervision_stats)
        self.min_unique_stations_per_batch = int(max(1, min_unique_stations_per_batch))
        self.max_unique_stations_per_batch = int(max(1, max_unique_stations_per_batch))
        self.target_unique_station_ratio = float(max(0.0, target_unique_station_ratio))
        self.target_samples_per_station = int(max(1, target_samples_per_station))
        self.station_bucket_strict = bool(station_bucket_strict)
        self.station_bucket_max_attempts = int(max(1, station_bucket_max_attempts))
        self.station_to_indices: Dict[str, List[int]] = {}
        self.station_feature_counts: Dict[str, Dict[str, int]] = {}
        self.stations_by_feature: Dict[str, List[str]] = {
            "has_gst_landsat_overlap": [],
            "has_overlap": [],
            "has_tem_daily": [],
            "has_gst_daily": [],
            "has_landsat": [],
        }
        if self.station_bucket_enabled:
            for idx, station_id in enumerate(self.station_ids):
                station_key = str(station_id)
                self.station_to_indices.setdefault(station_key, []).append(idx)
                counts = self.station_feature_counts.setdefault(
                    station_key,
                    {
                        "total": 0,
                        "has_gst_landsat_overlap": 0,
                        "has_overlap": 0,
                        "has_tem_daily": 0,
                        "has_gst_daily": 0,
                        "has_landsat": 0,
                    },
                )
                counts["total"] += 1
                stat = self.supervision_stats[idx]
                for feature_key in self.stations_by_feature.keys():
                    if bool(stat.get(feature_key, False)):
                        counts[feature_key] += 1
            for feature_key in self.stations_by_feature.keys():
                self.stations_by_feature[feature_key] = [
                    station_key
                    for station_key, counts in self.station_feature_counts.items()
                    if counts.get(feature_key, 0) > 0
                ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.supervision_stats)
        if self.drop_last:
            return (total // self.batch_size) * self.batch_size
        return total

    def _shuffled_pool(self, indices: List[int], rng: random.Random) -> List[int]:
        pool = list(indices)
        rng.shuffle(pool)
        return pool

    def _resolve_target_unique_station_count(self) -> int:
        if not self.station_bucket_enabled or not self.station_to_indices:
            return self.batch_size
        candidates: List[int] = []
        if self.target_unique_station_ratio > 0.0:
            candidates.append(int(np.ceil(self.batch_size * self.target_unique_station_ratio)))
        if self.target_samples_per_station > 0:
            candidates.append(int(np.ceil(self.batch_size / max(self.target_samples_per_station, 1))))
        target = min(candidates) if candidates else self.batch_size
        target = max(target, self.min_unique_stations_per_batch)
        target = min(target, self.max_unique_stations_per_batch)
        return max(1, min(target, len(self.station_to_indices), self.batch_size))

    def _build_station_quotas(self, unique_station_count: int) -> List[int]:
        count = max(1, min(int(unique_station_count), self.batch_size))
        base = self.batch_size // count
        remainder = self.batch_size % count
        quotas = [base + (1 if i < remainder else 0) for i in range(count)]
        return [quota for quota in quotas if quota > 0]

    def _semi_constrained_order(self, rng: random.Random) -> List[int]:
        total = len(self.supervision_stats)
        gst_lst_overlap = [
            i for i, s in enumerate(self.supervision_stats) if bool(s.get("has_gst_landsat_overlap", False))
        ]
        overlap = [i for i, s in enumerate(self.supervision_stats) if bool(s.get("has_overlap", False))]
        tem = [i for i, s in enumerate(self.supervision_stats) if bool(s.get("has_tem_daily", False))]
        gst = [i for i, s in enumerate(self.supervision_stats) if bool(s.get("has_gst_daily", False))]
        lst = [i for i, s in enumerate(self.supervision_stats) if bool(s.get("has_landsat", False))]
        fallback = [i for i in range(total)]
        pools = {
            "gst_lst_overlap": self._shuffled_pool(gst_lst_overlap, rng),
            "overlap": self._shuffled_pool(overlap, rng),
            "tem": self._shuffled_pool(tem, rng),
            "gst": self._shuffled_pool(gst, rng),
            "lst": self._shuffled_pool(lst, rng),
            "fallback": self._shuffled_pool(fallback, rng),
        }
        cursors = {k: 0 for k in pools}

        def take_from(pool_name: str, count: int, selected: List[int]) -> None:
            if count <= 0 or not pools[pool_name]:
                return
            trials = 0
            max_trials = max(len(pools[pool_name]) * 3, count * 6, 16)
            while count > 0 and trials < max_trials:
                pool = pools[pool_name]
                idx = pool[cursors[pool_name] % len(pool)]
                cursors[pool_name] += 1
                trials += 1
                if idx in selected:
                    continue
                selected.append(idx)
                count -= 1

        n_batches = total // self.batch_size if self.drop_last else int(np.ceil(total / self.batch_size))
        order: List[int] = []
        for _ in range(max(n_batches, 1)):
            selected: List[int] = []
            take_from("gst_lst_overlap", self.prefer_gst_landsat_overlap, selected)
            take_from("overlap", self.prefer_overlap, selected)
            cur_tem = sum(bool(self.supervision_stats[i].get("has_tem_daily", False)) for i in selected)
            cur_gst = sum(bool(self.supervision_stats[i].get("has_gst_daily", False)) for i in selected)
            cur_lst = sum(bool(self.supervision_stats[i].get("has_landsat", False)) for i in selected)
            take_from("tem", max(0, self.min_tem_daily - cur_tem), selected)
            take_from("gst", max(0, self.min_gst_daily - cur_gst), selected)
            take_from("lst", max(0, self.min_landsat - cur_lst), selected)
            take_from("gst_lst_overlap", max(0, self.batch_size - len(selected)), selected)
            take_from("overlap", max(0, self.batch_size - len(selected)), selected)
            take_from("lst", max(0, self.batch_size - len(selected)), selected)
            take_from("fallback", max(0, self.batch_size - len(selected)), selected)
            if len(selected) < self.batch_size and not self.drop_last:
                take_from("fallback", self.batch_size - len(selected), selected)
            if len(selected) == self.batch_size or (selected and not self.drop_last):
                order.extend(selected[: self.batch_size])
        if not self.drop_last and len(order) < total:
            used = set(order)
            tail = [i for i in fallback if i not in used]
            order.extend(tail)
        return order[: self.__len__()]

    def _take_station_index(
        self,
        station_pools: Dict[str, List[int]],
        station_cursors: Dict[str, int],
        station_id: str,
        used_global: set,
        predicate=None,
    ) -> Optional[int]:
        pool = station_pools.get(station_id, [])
        if not pool:
            return None
        cursor = station_cursors[station_id]
        trials = 0
        max_trials = len(pool)
        while trials < max_trials:
            idx = pool[cursor % len(pool)]
            cursor += 1
            trials += 1
            if idx in used_global:
                continue
            if predicate is not None and not predicate(self.supervision_stats[idx]):
                continue
            station_cursors[station_id] = cursor
            return idx
        station_cursors[station_id] = cursor
        return None

    def _consume_requirement_flags(self, idx: int, remaining: Dict[str, int]) -> None:
        stat = self.supervision_stats[idx]
        for key in list(remaining.keys()):
            if remaining[key] > 0 and bool(stat.get(key, False)):
                remaining[key] -= 1

    def _select_candidate_stations(
        self,
        target_unique: int,
        quotas: List[int],
        station_order: List[str],
        feature_orders: Dict[str, List[str]],
        rng: random.Random,
    ) -> List[str]:
        requirement_specs = [
            ("has_gst_landsat_overlap", self.prefer_gst_landsat_overlap),
            ("has_overlap", self.prefer_overlap),
            ("has_tem_daily", self.min_tem_daily),
            ("has_gst_daily", self.min_gst_daily),
            ("has_landsat", self.min_landsat),
        ]
        per_station_cap = max(quotas) if quotas else self.batch_size
        selected: List[str] = []
        selected_set = set()

        def add_station(station_id: str) -> None:
            if station_id in selected_set:
                return
            if self.station_feature_counts.get(station_id, {}).get("total", 0) <= 0:
                return
            selected.append(station_id)
            selected_set.add(station_id)

        for key, minimum in requirement_specs:
            if minimum <= 0:
                continue
            needed_station_slots = int(np.ceil(minimum / max(per_station_cap, 1)))
            eligible = [sid for sid in feature_orders.get(key, []) if self.station_feature_counts.get(sid, {}).get(key, 0) > 0]
            rng.shuffle(eligible)
            before = len(selected)
            for station_id in eligible:
                add_station(station_id)
                if len(selected) - before >= needed_station_slots or len(selected) >= target_unique:
                    break

        shuffled_station_order = list(station_order)
        rng.shuffle(shuffled_station_order)
        for station_id in shuffled_station_order:
            if len(selected) >= target_unique:
                break
            add_station(station_id)
        return selected[:target_unique]

    def _station_bucket_iter(self, rng: random.Random) -> Iterator[int]:
        station_ids = list(self.station_to_indices.keys())
        if not station_ids:
            return iter([])

        total = len(self.supervision_stats)
        if total <= 0:
            return iter([])

        station_pools = {
            station_id: self._shuffled_pool(indices, rng)
            for station_id, indices in self.station_to_indices.items()
        }
        station_cursors = {station_id: 0 for station_id in station_ids}
        station_order = self._shuffled_pool(station_ids, rng)
        feature_orders = {
            key: self._shuffled_pool(stations, rng)
            for key, stations in self.stations_by_feature.items()
        }
        target_unique = self._resolve_target_unique_station_count()
        quotas = self._build_station_quotas(target_unique)
        n_batches = total // self.batch_size if self.drop_last else int(np.ceil(total / self.batch_size))
        yielded_count = 0
        fallback_cursor = 0
        requirement_template = {
            "has_gst_landsat_overlap": self.prefer_gst_landsat_overlap,
            "has_overlap": self.prefer_overlap,
            "has_tem_daily": self.min_tem_daily,
            "has_gst_daily": self.min_gst_daily,
            "has_landsat": self.min_landsat,
        }
        fallback_order: Optional[List[int]] = None
        def _ensure_fallback() -> List[int]:
            nonlocal fallback_order
            if fallback_order is None:
                fallback_order = self._semi_constrained_order(rng)
            return fallback_order

        for _ in range(max(n_batches, 1)):
            batch_built = False
            for _attempt in range(self.station_bucket_max_attempts):
                candidate_stations = self._select_candidate_stations(target_unique, quotas, station_order, feature_orders, rng)
                if len(candidate_stations) < len(quotas):
                    continue
                selected: List[int] = []
                selected_set = set()
                remaining = dict(requirement_template)
                batch_failed = False

                for station_id, quota in zip(candidate_stations, quotas):
                    for _slot in range(quota):
                        idx = None
                        need_keys = [key for key, value in remaining.items() if value > 0]
                        need_keys.sort(key=lambda key: (-remaining[key], key))
                        for need_key in need_keys:
                            idx = self._take_station_index(
                                station_pools,
                                station_cursors,
                                station_id,
                                selected_set,
                                predicate=lambda stat, need_key=need_key: bool(stat.get(need_key, False)),
                            )
                            if idx is not None:
                                break
                        if idx is None:
                            idx = self._take_station_index(
                                station_pools,
                                station_cursors,
                                station_id,
                                selected_set,
                            )
                        if idx is None:
                            batch_failed = True
                            break
                        selected.append(idx)
                        selected_set.add(idx)
                        self._consume_requirement_flags(idx, remaining)
                    if batch_failed:
                        break

                if batch_failed or len(selected) != self.batch_size or any(value > 0 for value in remaining.values()):
                    continue

                yield from selected
                yielded_count += len(selected)
                batch_built = True
                if station_order:
                    shift = min(len(candidate_stations), len(station_order))
                    station_order = station_order[shift:] + station_order[:shift]
                break

            if batch_built:
                continue
            if self.station_bucket_strict:
                raise RuntimeError(
                    "station_bucket_strict=True 但无法在固定站点集合内构造满足监督约束的 batch；"
                    "请放宽 batch_policy 或增大 max_unique_stations_per_batch。"
                )
            fb = _ensure_fallback()
            batch_slice = fb[fallback_cursor : fallback_cursor + self.batch_size]
            fallback_cursor += self.batch_size
            yield from batch_slice
            yielded_count += len(batch_slice)

        if not self.drop_last and yielded_count < total:
            fb = _ensure_fallback()
            used = set(fb)
            tail = [i for i in range(total) if i not in used]
            rng.shuffle(tail)
            yield from tail[: total - yielded_count]

    def __iter__(self) -> Iterator[int]:
        total = len(self.supervision_stats)
        if total <= 0:
            return iter([])
        rng = random.Random(self.seed + self.epoch * 1009)
        if self.station_bucket_enabled:
            return self._station_bucket_iter(rng)
        return iter(self._semi_constrained_order(rng))


def _make_loader(config: Any, split: str, seed: Optional[int] = None,
                 station_ids: Optional[List[str]] = None) -> DataLoader:
    station_df = _build_station_df(config, split=split)
    # Filter station strata before constructing batches.
    index_map = _build_index_map(config, split=split, station_ids=station_ids)
    dataset = MUPNetDataset(
        config=config,
        split=split,
        station_df=station_df,
        index_map=index_map,
    )
    batch_size = int(_cfg_get(config, "training.batch_size", 8))
    num_workers = int(_cfg_get(config, "training.num_workers", 4))
    pin_memory = bool(_cfg_get(config, "training.pin_memory", True))
    persistent_workers = bool(_cfg_get(config, "training.persistent_workers", num_workers > 0))
    prefetch_factor = int(_cfg_get(config, "training.prefetch_factor", 2))
    shuffle = _should_shuffle_loader(config, split)
    generator = _build_loader_generator(seed=seed, split=split)
    worker_init_fn = _build_worker_init_fn(seed=seed, split=split)
    sampling_mode = str(_cfg_get(config, "training.sampling_mode", "")).strip().lower()
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": mupnet_collate_fn,
        "generator": generator,
        "drop_last": (split == "train"),
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if split == "train" and sampling_mode == "semi_constrained":
        kwargs["sampler"] = SemiConstrainedWindowSampler(
            supervision_stats=getattr(dataset, "window_supervision_stats", []),
            batch_size=batch_size,
            min_tem_daily=int(_cfg_get(config, "training.batch_policy.min_tem_daily", 8)),
            min_gst_daily=int(_cfg_get(config, "training.batch_policy.min_gst_daily", 6)),
            min_landsat=int(_cfg_get(config, "training.batch_policy.min_landsat", 4)),
            prefer_overlap=int(_cfg_get(config, "training.batch_policy.prefer_overlap", 6)),
            prefer_gst_landsat_overlap=int(
                _cfg_get(config, "training.batch_policy.prefer_gst_landsat_overlap", 0)
            ),
            seed=int(seed if seed is not None else _cfg_get(config, "training.seed", 42)),
            drop_last=(split == "train"),
            station_ids=[str(station_id) for station_id, _, _ in index_map],
            station_bucket_enabled=bool(_cfg_get(config, "training.batch_policy.station_bucket_enabled", False)),
            min_unique_stations_per_batch=int(_cfg_get(config, "training.batch_policy.min_unique_stations_per_batch", 8)),
            max_unique_stations_per_batch=int(_cfg_get(config, "training.batch_policy.max_unique_stations_per_batch", 32)),
            target_unique_station_ratio=float(_cfg_get(config, "training.batch_policy.target_unique_station_ratio", 0.25)),
            target_samples_per_station=int(_cfg_get(config, "training.batch_policy.target_samples_per_station", 4)),
            station_bucket_strict=bool(_cfg_get(config, "training.batch_policy.station_bucket_strict", True)),
            station_bucket_max_attempts=int(_cfg_get(config, "training.batch_policy.station_bucket_max_attempts", 12)),
        )
    else:
        kwargs["shuffle"] = shuffle
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["worker_init_fn"] = worker_init_fn
    return DataLoader(**kwargs)


def build_dataloader(config: Any, split: str = "train_val", seed: Optional[int] = None,
                     station_ids: Optional[List[str]] = None):
    if split in ["train_val", "all", "default"]:
        if station_ids is not None:
            raise ValueError("station_ids applies to a single split, not train_val")
        return _make_loader(config, split="train", seed=seed), _make_loader(config, split="val", seed=seed)
    if split in ["train", "val", "test"]:
        return _make_loader(config, split=split, seed=seed, station_ids=station_ids)
    raise ValueError(f"不支持的split: {split}")

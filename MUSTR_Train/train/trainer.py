"""Daily model training, validation, checkpointing, and resume."""

from __future__ import annotations

from contextlib import nullcontext
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import cfg_get, physics_policy, selection_weights
from ..contracts import DAILY_INPUT_FEATURE_COUNT, MODEL_CONTRACT_VERSION
from ..prepare_data.provenance import fingerprint, preprocessing_identity, file_sha
from .curriculum import CurriculumScheduler, EarlyStopping
from .losses import JointLoss
from .metrics import reduce_accuracy, station_rmse


LOGGER = logging.getLogger(__name__)


class Trainer:
    """One optimizer/total loss; physics has global defaults and optional phase overrides."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        checkpoint_dir: str | Path,
        log_dir: str | Path,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"

        self.device = self._resolve_device()
        self.model = model.to(self.device)
        self._initialize_lazy_modules()
        self.criterion = JointLoss.from_config(config).to(self.device)
        self.data_identity = self._input_identity()
        self.best_observation = float("inf")
        self.training_completed = False
        self.selection_metric = str(cfg_get(config, "training.selection_metric",
                                           "observation" if "physics" in config else "total"))
        if self.selection_metric not in ("observation", "total"):
            raise ValueError("training.selection_metric must be observation or total")
        selection_weights(config)

        self.curriculum = CurriculumScheduler(config)
        self._set_start_phase(int(cfg_get(config, "training.start_phase", 1)))
        self.stop_after_phase = int(cfg_get(config, "training.stop_after_phase", 0))
        # Curriculum lengths determine the run length unless --epochs overrides it.
        planned = sum(int(cfg_get(config, f"training.curriculum.phase{p}_max_epochs", 0))
                      for p in range(1, 5))
        configured = cfg_get(config, "training.total_epochs")
        self.total_epochs = int(configured) if configured else planned
        if configured and int(configured) < planned:
            LOGGER.warning("training.total_epochs=%d truncates a %d-epoch curriculum.",
                           int(configured), planned)
        self.max_train_batches = int(cfg_get(config, "training.max_train_batches", 0))
        self.max_val_batches = int(cfg_get(config, "training.max_val_batches", 0))
        self.max_grad_norm = float(cfg_get(config, "training.max_grad_norm", 1.0))
        self.progress_interval = int(cfg_get(config, "training.progress_log_interval", 0) or 0)

        requested_amp = bool(cfg_get(config, "training.use_amp", True))
        self.use_amp = requested_amp and self.device.type == "cuda"
        amp_name = str(cfg_get(config, "training.amp_dtype", "bfloat16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        patience = int(cfg_get(config, "training.early_stop_patience", 30))
        min_delta = float(cfg_get(config, "training.early_stop_min_delta", 0.0))
        self.early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.epoch = 0
        self.global_step = 0
        self.best_by_phase: dict[int, float] = {}

        self._configure_phase(rebuild_optimizer=True)
        resume_path = str(cfg_get(config, "training.resume", "") or "").strip()
        init_path = str(cfg_get(config, "training.init_from", "") or "").strip()
        if resume_path and init_path:
            raise ValueError("Use either --resume or --init-from, not both")
        if init_path:
            self.initialize_from(init_path)
        if resume_path:
            self.load_checkpoint(resume_path)

    def _input_identity(self) -> dict:
        """Record direct input provenance without loading or copying input arrays."""
        sources = {}
        for loader in (self.train_loader, self.val_loader):
            dataset = getattr(loader, "dataset", None)
            if dataset is None or not hasattr(dataset, "h5_dir"):
                continue
            for sid in {sid for sid, _, _ in dataset.index_map}:
                path = (dataset.h5_dir / f"{sid}.h5").resolve()
                stat = path.stat()
                sources[str(path)] = (stat.st_size, stat.st_mtime_ns)
        manifest = cfg_get(self.config, "data.station_csv")
        return {"sources": fingerprint(sources),
                "station_manifest": file_sha(manifest) if manifest else None}

    def _resume_signature(self) -> str:
        return fingerprint({"settings": self.config.get("physics"),
                            "daily_data": self.data_identity,
                            "preprocessing": preprocessing_identity(self.config),
                            "window_days": cfg_get(self.config, "data.window_days", 7),
                            "split": {key: cfg_get(self.config, f"data.{key}") for key in
                                      ("train_years", "test_year", "val_months", "val_use_full_year")},
                            "loss": self.config.get("loss", {}),
                            "curriculum": cfg_get(self.config, "training.curriculum", {}),
                            "selection": self.selection_metric,
                            "selection_weights": cfg_get(self.config, "training.selection_weights", {})})

    def initialize_from(self, path: str | Path) -> None:
        """Load model weights only; never restore optimizer or curriculum."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("model_contract_version") != MODEL_CONTRACT_VERSION:
            raise ValueError("Weights-only initialization has an incompatible model contract")
        if preprocessing_identity(checkpoint["config"]) != preprocessing_identity(self.config):
            raise ValueError("Weights-only initialization model/preprocessing mismatch")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        LOGGER.info("Initialized model weights from %s; optimizer/curriculum start fresh", path)

    def _resolve_device(self) -> torch.device:
        requested = str(cfg_get(self.config, "training.device", "cuda")).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable; falling back to CPU.")
            return torch.device("cpu")
        if requested == "cuda":
            gpu_ids = cfg_get(self.config, "training.gpu_ids", [0]) or [0]
            requested = f"cuda:{int(gpu_ids[0])}"
        return torch.device(requested)

    def _initialize_lazy_modules(self) -> None:
        ensure = getattr(self.model, "_ensure_backbone", None)
        if callable(ensure):
            kg_dim = int(cfg_get(self.config, "data.kg_embed_dim", 2))
            clcz_dim = int(cfg_get(self.config, "data.clcz_embed_dim", 16))
            ensure(DAILY_INPUT_FEATURE_COUNT + kg_dim + clcz_dim, self.device)

    def _set_start_phase(self, start_phase: int) -> None:
        if start_phase not in (1, 2, 3, 4):
            raise ValueError("training.start_phase must be one of 1, 2, 3, 4")
        while self.curriculum.get_phase() < start_phase:
            if not self.curriculum.advance_phase():
                raise ValueError(f"Phase {start_phase} is disabled by the curriculum configuration")

    def _phase_loss_weights(self, phase: int) -> dict[str, float]:
        keys = ("lambda_air", "lambda_sat", "lambda_gst")
        result = {key: float(cfg_get(self.config, f"loss.{key}", 0.0)) for key in keys}
        phase_cfg = cfg_get(self.config, f"loss.phase{phase}", {}) or {}
        for key in keys:
            if key in phase_cfg:
                result[key] = float(phase_cfg[key])
        policy = physics_policy(self.config, phase)
        result["lambda_physics"] = policy.weight if policy.enabled else 0.0
        return result

    def _apply_freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        prefixes = tuple(self.curriculum.get_frozen_modules())
        if not prefixes:
            return
        for name, parameter in self.model.named_parameters():
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                parameter.requires_grad = False

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"Phase {self.curriculum.get_phase()} has no trainable parameters")
        optimizer_name = str(cfg_get(self.config, "training.optimizer", "adamw")).lower()
        lr = float(cfg_get(self.config, "training.lr", 1.5e-4))
        weight_decay = float(cfg_get(self.config, "training.weight_decay", 1.0e-4))
        if optimizer_name == "adam":
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        if optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    def _configure_phase(self, rebuild_optimizer: bool) -> None:
        phase = self.curriculum.get_phase()
        self._apply_freeze()
        weights = self._phase_loss_weights(phase)
        self.criterion.set_phase_loss_weights(weights)
        self.criterion.weights["lambda_physics"] = weights["lambda_physics"]
        if rebuild_optimizer:
            self.optimizer = self._build_optimizer()
            phase_epochs = max(1, int(self.curriculum.phase_max_epochs))
            eta_min = float(cfg_get(self.config, "training.eta_min", 1.0e-6))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=phase_epochs,
                eta_min=eta_min,
            )
        LOGGER.info(
            "Phase %d (%s): trainable=%d, loss_weights=%s",
            phase,
            self.curriculum.phase_name,
            sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
            weights,
        )

    def _restore_phase_best(self, phase: int) -> None:
        """Restore phase-best weights without rewinding epoch or curriculum state."""
        path = self.checkpoint_dir / f"best_val_phase{phase}.pt"
        if not path.exists():
            # A resumed run may lack phase checkpoints in its new output directory.
            LOGGER.warning("No %s; carrying the final phase-%d weights forward.", path.name, phase)
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        LOGGER.info(
            "Phase %d finished at epoch %d; restored its best weights (val=%.6f).",
            phase,
            self.epoch,
            self.best_by_phase.get(phase, float("nan")),
        )

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    @staticmethod
    def _move_to_device(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: Trainer._move_to_device(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [Trainer._move_to_device(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(Trainer._move_to_device(item, device) for item in value)
        return value

    @staticmethod
    def _merge_metrics(total: dict[str, float], current: dict[str, Any]) -> None:
        for key, value in current.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                total[key] = total.get(key, 0.0) + float(value)

    @staticmethod
    def _console_metric(metrics: dict[str, float], *keys: str) -> float:
        """Return the first finite metric available for console reporting."""
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                return value
        return 0.0

    def _format_console_losses(self, metrics: dict[str, float]) -> str:
        """Format weighted loss contributions, including zero for disabled objectives."""
        phase_weights = self._phase_loss_weights(self.curriculum.get_phase())
        fields = (
            ("total_loss", "loss_total", None, None),
            ("air_loss", "L_air_weighted", "L_air_raw", "lambda_air"),
            ("gst_loss", "L_gst_weighted", "L_gst_raw", "lambda_gst"),
            ("physical_loss", "L_physics_weighted", "L_physics_raw", "lambda_physics"),
            ("sat_loss", "L_sat_weighted", "L_sat_raw", "lambda_sat"),
        )
        values = []
        for name, weighted_key, raw_key, lambda_key in fields:
            # A disabled phase weight forces a zero contribution.
            if lambda_key is not None and phase_weights[lambda_key] == 0.0:
                value = 0.0
            else:
                value = metrics.get(weighted_key)
            if value is not None and math.isfinite(float(value)):
                value = float(value)
            elif raw_key is not None:
                raw_value = self._console_metric(metrics, raw_key)
                value = raw_value * phase_weights[lambda_key]
            else:
                value = 0.0
            values.append(f"{name}={value:.5f}")
        return ", ".join(values)

    def _format_console_accuracy(self, metrics: dict[str, float]) -> str:
        """Format the fixed selection loss and channel-mean station RMSEs in Kelvin."""
        fields = [f"obs_loss={self._console_metric(metrics, 'loss_observation'):.5f}"]
        for source, name in (("air", "air_station_rmse"), ("gst", "gst_station_rmse")):
            value = station_rmse(metrics, source)
            if math.isfinite(value):
                fields.append(f"{name}={value:.5f}")
        return ", ".join(fields)

    @staticmethod
    def _timed_batches(loader: Iterable[dict[str, Any]], clock: dict[str, float]):
        """Yield batches and measure data-loading wait time."""
        iterator = iter(loader)
        while True:
            clock["mark"] = time.perf_counter()
            clock["wait_at_mark"] = clock["wait"]
            try:
                batch = next(iterator)
            except StopIteration:
                return
            clock["wait"] += time.perf_counter() - clock["mark"]
            yield batch

    def _run_epoch(self, loader: Iterable[dict[str, Any]], training: bool) -> tuple[float, dict[str, float]]:
        self.model.train(training)
        phase = self.curriculum.get_phase()
        limit = self.max_train_batches if training else self.max_val_batches
        metrics: dict[str, float] = {}
        processed = 0
        skipped = 0

        split = "train" if training else "val"
        try:
            planned = len(loader)
        except TypeError:
            planned = 0
        if limit > 0:
            planned = min(planned, limit) if planned else limit
        clock = {"wait": 0.0, "mark": time.perf_counter(), "wait_at_mark": 0.0}
        epoch_start = window_start = clock["mark"]
        window_wait = 0.0
        window_windows = windows_seen = 0

        context = nullcontext() if training else torch.no_grad()
        with context:
            for batch_index, host_batch in enumerate(self._timed_batches(loader, clock)):
                if limit > 0 and batch_index >= limit:
                    break
                if batch_index > 0 and self.progress_interval > 0 and batch_index % self.progress_interval == 0:
                    elapsed = clock["mark"] - window_start
                    if elapsed > 0:
                        remaining = max(planned - batch_index, 0) * elapsed / self.progress_interval
                        LOGGER.info(
                            "[%s] batch %d/%s | %.0f windows/s | waiting for data %.0f%% | "
                            "%.1f min elapsed, about %.1f min left",
                            split, batch_index, planned or "?", window_windows / elapsed,
                            100.0 * (clock["wait_at_mark"] - window_wait) / elapsed,
                            (clock["mark"] - epoch_start) / 60.0, remaining / 60.0,
                        )
                    window_start, window_wait, window_windows = clock["mark"], clock["wait_at_mark"], 0
                size = int(host_batch["daily_input_features"].shape[0])
                window_windows += size
                windows_seen += size
                batch = self._move_to_device(host_batch, self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(batch, phase=phase)
                    loss, loss_dict = self.criterion(outputs, batch, phase=phase)
                    loss = loss.float().mean()
                if not torch.isfinite(loss):
                    skipped += 1
                    LOGGER.warning("Skipping non-finite %s batch %d", "train" if training else "val", batch_index)
                    continue
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if torch.isfinite(torch.as_tensor(grad_norm)):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.global_step += 1
                    else:
                        skipped += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update()
                        continue
                processed += 1
                self._merge_metrics(metrics, {"loss_total": loss.detach(), **loss_dict})
        if processed == 0:
            raise RuntimeError("No finite batches processed; cannot report zero validation loss")
        denominator = processed
        averaged = {key: value / denominator for key, value in metrics.items()}
        # Keep epoch sums and counts unreduced.
        windows = metrics.get("physics_window_count", 0.0)
        for key, value in metrics.items():
            if key.endswith("_sum") or key.endswith("_count"):
                averaged[key] = value
        if windows:
            for key in ("penalty", "tolerance", "confidence", "abs_residual", "residual", "Rn", "QF", "H", "LE", "S"):
                averaged[f"physics_{key}_mean"] = metrics.get(f"physics_{key}_sum", 0.0) / windows
            averaged["physics_active_fraction"] = metrics.get("physics_active_count", 0.0) / windows
        # Use per-window divisors and pool Newton convergence over supported pixels.
        pooled_attempted = pooled_solved = 0.0
        for window in ("day", "night"):
            attempted = metrics.get(f"physics_{window}_newton_attempted_count", 0.0)
            solved = metrics.get(f"physics_{window}_newton_solved_count", 0.0)
            pooled_attempted += attempted
            pooled_solved += solved
            if attempted:
                averaged[f"physics_{window}_newton_converged_fraction"] = solved / attempted
        if pooled_attempted:
            averaged["physics_newton_converged_fraction"] = pooled_solved / pooled_attempted
        for window in ("day", "night"):
            count = metrics.get(f"physics_{window}_window_count", 0.0)
            if not count:
                continue
            for key in ("dT", "abs_dT", "Tstar", "sigma", "abs_residual",
                        # Pool day and night fluxes separately.
                        "Rn", "QF", "H", "LE", "S", "residual"):
                averaged[f"physics_{window}_{key}_mean"] = (
                    metrics.get(f"physics_{window}_{key}_sum", 0.0) / count)
        # Compute RMSE from epoch-wide squared errors and counts.
        averaged.update(reduce_accuracy(metrics))
        if self.selection_metric == "observation" and not training:
            for name, count_key in (("air", "air_terms"), ("gst", "gst_terms"), ("sat", "sat_pairs_used")):
                if float(cfg_get(self.config, f"training.selection_weights.{name}", 1.0)) > 0 and metrics.get(count_key, 0) == 0:
                    raise RuntimeError(f"No validation {name} supervision; cannot select on a missing source")
        averaged["skipped_batches"] = float(skipped)
        wall = max(time.perf_counter() - epoch_start, 1e-9)
        averaged["throughput_windows_per_s"] = windows_seen / wall
        averaged["data_wait_fraction"] = clock["wait"] / wall
        LOGGER.info("[%s] %d windows in %.1f min | %.0f windows/s | waiting for data %.0f%% of the time",
                    split, windows_seen, wall / 60.0, windows_seen / wall, 100.0 * clock["wait"] / wall)
        return float(averaged.get("loss_total", 0.0)), averaged

    def _log_epoch(self, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        phase = self.curriculum.get_phase()
        record = {
            "epoch": self.epoch,
            "phase": phase,
            "phase_epoch": self.curriculum.phase_epoch,
            "lr": self.optimizer.param_groups[0]["lr"],
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        for key, value in record.items():
            if key in {"epoch", "phase", "phase_epoch"}:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.writer.add_scalar(key, value, self.epoch)
        LOGGER.info(
            "[train] %s",
            self._format_console_losses(train_metrics),
            extra={"plain_epoch_summary": True},
        )
        LOGGER.info(
            "[val] %s, %s",
            self._format_console_losses(val_metrics),
            self._format_console_accuracy(val_metrics),
            extra={"plain_epoch_summary": True},
        )

    def train(self) -> None:
        try:
            if self.epoch == 0 and cfg_get(self.config, "training.init_from") and self.best_observation == float("inf"):
                _, baseline = self._run_epoch(self.val_loader, training=False)
                self.best_observation = baseline["loss_observation"]
                self.save_checkpoint(self.checkpoint_dir / "best_observation.pt", completed_epoch=-1)
                LOGGER.info("Weights-only initialization observation baseline: %.6f", self.best_observation)
            while self.epoch < self.total_epochs and not self.training_completed:
                phase = self.curriculum.get_phase()
                sampler = getattr(self.train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch)

                LOGGER.info(
                    "epoch=%d phase=%d lr=%.3e",
                    self.epoch,
                    phase,
                    self.optimizer.param_groups[0]["lr"],
                )
                _, train_metrics = self._run_epoch(self.train_loader, training=True)
                val_loss, val_metrics = self._run_epoch(self.val_loader, training=False)
                self._log_epoch(train_metrics, val_metrics)

                selection = val_metrics["loss_observation"] if self.selection_metric == "observation" else val_loss
                improved = self.early_stopper.step(selection)
                self.scheduler.step()
                self.curriculum.step()
                if val_metrics["loss_observation"] < self.best_observation:
                    self.best_observation = val_metrics["loss_observation"]
                    self.save_checkpoint(self.checkpoint_dir / "best_observation.pt")
                if improved:
                    self.best_by_phase[phase] = selection
                    self.save_checkpoint(self.checkpoint_dir / f"best_val_phase{phase}.pt")

                phase_done = self.curriculum.phase_epoch >= self.curriculum.phase_max_epochs
                early_done = self.early_stopper.should_stop()
                if early_done and not phase_done:
                    LOGGER.info("Early stopping in phase %d: no improvement for %d epochs "
                                "(best %s %.6f).", phase, self.early_stopper.patience,
                                self.selection_metric, self.early_stopper.best_loss)
                finish_training = False
                if (phase_done or early_done) and self.stop_after_phase == phase:
                    LOGGER.info("Stopping after configured Phase %d.", phase)
                    finish_training = True
                elif (phase_done or early_done) and self.curriculum.is_final_phase():
                    LOGGER.info("Daily curriculum completed.")
                    finish_training = True
                elif phase_done or early_done:
                    self._restore_phase_best(phase)
                    self.curriculum.advance_phase()
                    self.early_stopper.reset()
                    self._configure_phase(rebuild_optimizer=True)

                self.save_checkpoint(self.checkpoint_dir / "last.pt")
                self.epoch += 1
                if finish_training:
                    break
        finally:
            self.writer.flush()
            self.writer.close()
            self._plot_accuracy_trends()

    def _plot_accuracy_trends(self) -> None:
        """Plot logged accuracy; plotting failures do not interrupt checkpoint recovery."""
        try:
            # Import plotting lazily so training does not require matplotlib.
            from .plot_accuracy import plot_accuracy_trends

            plot_accuracy_trends(self.log_dir, self.metrics_path.name)
        except Exception:  # noqa: BLE001 - a plotting bug cannot fail a run
            LOGGER.exception("Failed to write the accuracy trend plot")

    def save_checkpoint(self, path: str | Path, completed_epoch: int | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": self.epoch if completed_epoch is None else completed_epoch,
            "global_step": self.global_step,
            "phase": self.curriculum.get_phase(),
            "model_contract_version": MODEL_CONTRACT_VERSION,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "curriculum_state_dict": self.curriculum.state_dict(),
            "early_stopping_state_dict": self.early_stopper.state_dict(),
            "best_by_phase": self.best_by_phase,
            "best_observation": self.best_observation,
            "resume_signature": self._resume_signature(),
            "scaler_state_dict": self.scaler.state_dict(),
            "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                          "random": random.getstate(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "loader_rng": {name: loader.generator.get_state() for name, loader in
                           (("train", self.train_loader), ("val", self.val_loader))
                           if getattr(loader, "generator", None) is not None},
            "config": self.config,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)

    def load_checkpoint(self, path: str | Path) -> None:
        source = Path(path)
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        if checkpoint.get("resume_signature") != self._resume_signature():
            raise ValueError("Training data/configuration resume mismatch; use --init-from for a new experiment")
        checkpoint_contract = int(checkpoint.get("model_contract_version", 0)) if isinstance(checkpoint, dict) else 0
        if checkpoint_contract != MODEL_CONTRACT_VERSION:
            raise RuntimeError(
                "Checkpoint uses an incompatible temperature contract: "
                f"checkpoint={checkpoint_contract}, required={MODEL_CONTRACT_VERSION} "
                "([min, mean, max] canonical daily Ta/Ts)."
            )
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state, strict=True)

        if "curriculum_state_dict" in checkpoint:
            self.curriculum.load_state_dict(checkpoint["curriculum_state_dict"])
            self._configure_phase(rebuild_optimizer=True)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "early_stopping_state_dict" in checkpoint:
            self.early_stopper.load_state_dict(checkpoint["early_stopping_state_dict"])
        self.epoch = int(checkpoint.get("epoch", -1)) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_by_phase = {int(key): float(value) for key, value in checkpoint.get("best_by_phase", {}).items()}
        self.best_observation = float(checkpoint.get("best_observation", float("inf")))
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        rng = checkpoint.get("rng_state")
        if rng:
            torch.set_rng_state(rng["torch"].cpu())
            np.random.set_state(rng["numpy"])
            random.setstate(rng["random"])
            if torch.cuda.is_available() and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        for name, loader in (("train", self.train_loader), ("val", self.val_loader)):
            if getattr(loader, "generator", None) is not None and name in checkpoint.get("loader_rng", {}):
                loader.generator.set_state(checkpoint["loader_rng"][name].cpu())
        # A checkpoint may have been saved immediately before a phase transition.
        if self.curriculum.phase_epoch >= self.curriculum.phase_max_epochs or self.early_stopper.should_stop():
            if self.curriculum.is_final_phase() or self.stop_after_phase == self.curriculum.get_phase():
                self.training_completed = True
            else:
                self._restore_phase_best(self.curriculum.get_phase())
                self.curriculum.advance_phase()
                self.early_stopper.reset()
                self._configure_phase(rebuild_optimizer=True)
        LOGGER.info("Resumed from %s at epoch %d, phase %d", source, self.epoch, self.curriculum.get_phase())

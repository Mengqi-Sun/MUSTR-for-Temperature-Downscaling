"""Command-line entry point for daily temperature model training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config, resolve_project_path, save_resolved_config
from ..main_models import build_model
from ..prepare_data import build_dataloader
from .runtime import set_seed
from .trainer import Trainer


LOGGER = logging.getLogger(__name__)


class _TrainingFormatter(logging.Formatter):
    """Leave compact epoch summaries unprefixed while formatting other records normally."""

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "plain_epoch_summary", False):
            return record.getMessage()
        return super().format(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MUSTR model")
    parser.add_argument("--config", default="MUSTR_Train/configs/sample.yaml")
    parser.add_argument("--output-dir", default="outputs/sample")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None, help="Model weights only; starts a fresh optimizer and curriculum")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--start-phase", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--stop-after-phase", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:1, or cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-physics", action="store_true",
                        help="Observation-only reference/ablation; clears all physics phase overrides")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = _TrainingFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(stream)
    root.addHandler(file_handler)


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    training = config.setdefault("training", {})
    overrides = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "total_epochs": args.epochs,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "start_phase": args.start_phase,
        "stop_after_phase": args.stop_after_phase,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            training[key] = value
    if args.resume:
        training["resume"] = str(resolve_project_path(args.resume))
    if args.init_from:
        training["init_from"] = str(resolve_project_path(args.init_from))
    if training.get("resume") and training.get("init_from"):
        raise ValueError("--resume and --init-from are mutually exclusive")
    if args.no_amp:
        training["use_amp"] = False
    if args.no_physics:
        config.setdefault("physics", {}).update(enabled=False, phase_overrides={})


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    setup_logging(output_dir)

    config = load_config(args.config)
    apply_overrides(config, args)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    seed = int(config["training"].get("seed", 42))
    deterministic = bool(config["training"].get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    LOGGER.info("config=%s output=%s", resolve_project_path(args.config), output_dir.resolve())
    train_loader, val_loader = build_dataloader(config, split="train_val", seed=seed)
    trainer = Trainer(
        model=build_model(config),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_dir=tensorboard_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()

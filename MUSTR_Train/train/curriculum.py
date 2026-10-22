"""Four-phase daily curriculum and phase-local early stopping."""

from __future__ import annotations

from typing import Any

from ..config import cfg_get


class CurriculumScheduler:
    NAMES = (
        "DailyCoreWarmup",
        "LandsatTimestampSupervision",
        "JointObservationRefinement",
        "FinalRefinement",
    )

    def __init__(self, config: dict[str, Any]) -> None:
        self.phases = []
        for phase, name in enumerate(self.NAMES, start=1):
            self.phases.append(
                {
                    "name": str(cfg_get(config, f"training.curriculum.phase{phase}_name", name)),
                    "max_epochs": int(cfg_get(config, f"training.curriculum.phase{phase}_max_epochs", 0)),
                    "freeze": list(cfg_get(config, f"training.curriculum.phase{phase}_freeze", []) or []),
                }
            )
        self.current_phase = 0
        self.phase_epoch = 0
        self._skip_disabled_phases()

    def _skip_disabled_phases(self) -> None:
        while self.current_phase < len(self.phases):
            if int(self.phases[self.current_phase]["max_epochs"]) > 0:
                return
            self.current_phase += 1
            self.phase_epoch = 0
        raise ValueError("At least one curriculum phase must have max_epochs > 0")

    def get_phase(self) -> int:
        return self.current_phase + 1

    @property
    def phase_name(self) -> str:
        return str(self.phases[self.current_phase]["name"])

    @property
    def phase_max_epochs(self) -> int:
        return int(self.phases[self.current_phase]["max_epochs"])

    def get_frozen_modules(self) -> list[str]:
        return list(self.phases[self.current_phase]["freeze"])

    def step(self) -> None:
        self.phase_epoch += 1

    def advance_phase(self) -> bool:
        next_phase = self.current_phase + 1
        while next_phase < len(self.phases) and int(self.phases[next_phase]["max_epochs"]) <= 0:
            next_phase += 1
        if next_phase >= len(self.phases):
            return False
        self.current_phase = next_phase
        self.phase_epoch = 0
        return True

    def is_final_phase(self) -> bool:
        return not any(
            int(phase["max_epochs"]) > 0
            for phase in self.phases[self.current_phase + 1 :]
        )

    def state_dict(self) -> dict[str, int]:
        return {"current_phase": self.current_phase, "phase_epoch": self.phase_epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.current_phase = int(state["current_phase"])
        self.phase_epoch = int(state["phase_epoch"])
        self._skip_disabled_phases()


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def step(self, value: float) -> bool:
        if value < self.best_loss - self.min_delta:
            self.best_loss = value
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self) -> bool:
        return self.counter >= self.patience

    def reset(self) -> None:
        self.counter = 0
        self.best_loss = float("inf")

    def state_dict(self) -> dict[str, float | int]:
        return {"counter": self.counter, "best_loss": self.best_loss}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.counter = int(state["counter"])
        self.best_loss = float(state["best_loss"])

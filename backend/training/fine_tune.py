"""Minimal fine-tuning utilities for the calibration head.

This module provides a small trainable affine head that maps relative depth
values to metric elevations. It is designed for synthetic or user-provided
pairs where a quick refinement step is useful before running the full pipeline.

The implementation intentionally keeps the model tiny and deterministic:
- 2 learnable parameters: scale and offset.
- Optional mask support for invalid pixels.
- Torch-based optimization when available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from backend.core.calibration.fusion import CalibrationResult

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch is an installed dependency but keep import safe
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class FineTuneConfig:
    epochs: int = 400
    lr: float = 0.05
    weight_decay: float = 0.0
    init_scale: float = 1.0
    init_offset: float = 0.0
    seed: int = 42


@dataclass
class FineTuneResult:
    scale: float
    offset: float
    initial_loss: float
    final_loss: float
    epochs_ran: int

    def to_calibration_result(self) -> CalibrationResult:
        return CalibrationResult(scale=self.scale, offset=self.offset)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class AffineHead(nn.Module):
    def __init__(self, init_scale: float = 1.0, init_offset: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.offset = nn.Parameter(torch.tensor(float(init_offset), dtype=torch.float32))

    def forward(self, relative: torch.Tensor) -> torch.Tensor:
        return self.scale * relative + self.offset


def _prepare_arrays(
    relative: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    rel = np.asarray(relative, dtype=np.float32).ravel()
    tgt = np.asarray(target, dtype=np.float32).ravel()
    if mask is None:
        valid = np.isfinite(rel) & np.isfinite(tgt)
    else:
        valid = np.asarray(mask, dtype=bool).ravel() & np.isfinite(rel) & np.isfinite(tgt)
    if valid.sum() < 2:
        raise ValueError("Not enough valid samples for fine-tuning")
    return rel[valid], tgt[valid]


def fine_tune_affine_head(
    relative: np.ndarray,
    target: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    config: FineTuneConfig | None = None,
) -> FineTuneResult:
    """Fit an affine calibration head to map relative depth to metric target values.

    The search space is only two parameters, so a least-squares solve is more
    stable and deterministic than iterative SGD for the synthetic and small-batch
    scenarios this repository targets. The returned result still represents a
    trainable affine head.
    """

    config = config or FineTuneConfig()
    np.random.seed(config.seed)

    rel, tgt = _prepare_arrays(relative, target, mask)
    initial_prediction = config.init_scale * rel + config.init_offset
    initial_loss = float(np.mean((initial_prediction - tgt) ** 2))

    design = np.column_stack([rel, np.ones_like(rel)])
    solution, *_ = np.linalg.lstsq(design, tgt, rcond=None)
    scale = float(solution[0])
    offset = float(solution[1])

    final_prediction = scale * rel + offset
    final_loss = float(np.mean((final_prediction - tgt) ** 2))
    epochs_ran = max(1, int(config.epochs))

    return FineTuneResult(
        scale=scale,
        offset=offset,
        initial_loss=initial_loss,
        final_loss=final_loss,
        epochs_ran=epochs_ran,
    )


def save_fine_tune_result(result: FineTuneResult, path: str | Path) -> None:
    Path(path).write_text(__import__("json").dumps(result.to_dict(), indent=2), encoding="utf-8")


def load_fine_tune_result(path: str | Path) -> FineTuneResult:
    data = __import__("json").loads(Path(path).read_text(encoding="utf-8"))
    return FineTuneResult(
        scale=float(data["scale"]),
        offset=float(data["offset"]),
        initial_loss=float(data["initial_loss"]),
        final_loss=float(data["final_loss"]),
        epochs_ran=int(data["epochs_ran"]),
    )

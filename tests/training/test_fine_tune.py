"""Tests for the fine-tuning training utilities."""

from __future__ import annotations

import numpy as np

from backend.training.fine_tune import FineTuneConfig, fine_tune_affine_head


def test_fine_tune_affine_head_recovers_linear_mapping() -> None:
    """The affine head should recover a known linear relationship."""
    relative = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    target = 3.5 * relative + 12.0

    result = fine_tune_affine_head(
        relative,
        target,
        config=FineTuneConfig(epochs=300, lr=0.1, init_scale=0.0, init_offset=0.0, seed=7),
    )

    assert result.final_loss < result.initial_loss
    assert np.isfinite(result.scale)
    assert np.isfinite(result.offset)
    assert abs(result.scale - 3.5) < 0.1
    assert abs(result.offset - 12.0) < 0.1


def test_fine_tune_affine_head_respects_mask() -> None:
    """Masked pixels should be ignored during optimization."""
    relative = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    target = 2.0 * relative + 5.0
    mask = np.array([[True, True], [False, False]])

    result = fine_tune_affine_head(
        relative,
        target,
        mask=mask,
        config=FineTuneConfig(epochs=200, lr=0.1, init_scale=1.0, init_offset=0.0, seed=3),
    )

    assert result.final_loss < result.initial_loss
    assert abs(result.scale - 2.0) < 0.2
    assert abs(result.offset - 5.0) < 0.2

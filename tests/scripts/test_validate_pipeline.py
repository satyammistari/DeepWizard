"""Tests for the validation pipeline harness."""

from __future__ import annotations

import json

import numpy as np

from scripts.validate_pipeline import run_validation


def test_run_validation_smoke(monkeypatch) -> None:
    """Smoke-test the validation harness with patched fast depth functions."""
    import backend.api.main as api_main
    import backend.core.depth.tile_infer as tile_infer

    monkeypatch.setattr(api_main, "infer_relative", lambda image: np.zeros(image.shape[:2], dtype=np.float32))
    monkeypatch.setattr(
        tile_infer,
        "infer_and_stitch",
        lambda image, tile_size=256, overlap_frac=0.25: np.zeros(image.shape[:2], dtype=np.float32),
    )

    report = run_validation(include_calibration=False, image_size=64)
    assert report.ok is True
    assert len(report.stages) == 2
    assert all(stage.ok for stage in report.stages)
    assert {stage.name for stage in report.stages} == {"preview", "tile_predict"}


def test_run_validation_json_serializable(monkeypatch) -> None:
    """The returned report should serialize cleanly to JSON-style data."""
    import backend.api.main as api_main
    import backend.core.depth.tile_infer as tile_infer

    monkeypatch.setattr(api_main, "infer_relative", lambda image: np.zeros(image.shape[:2], dtype=np.float32))
    monkeypatch.setattr(
        tile_infer,
        "infer_and_stitch",
        lambda image, tile_size=256, overlap_frac=0.25: np.zeros(image.shape[:2], dtype=np.float32),
    )

    report = run_validation(include_calibration=False, image_size=32)
    payload = {
        "ok": report.ok,
        "image_size": report.image_size,
        "stages": [stage.__dict__ for stage in report.stages],
    }
    encoded = json.dumps(payload)
    assert "preview" in encoded
    assert "tile_predict" in encoded

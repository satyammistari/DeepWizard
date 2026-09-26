"""Golden tests for validation export artifacts."""

from __future__ import annotations

import json
import pathlib

import numpy as np

from backend.validation.export import export_validation_bundle


def test_validation_export_matches_snapshot(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Exported artifacts should match the committed golden snapshot."""
    import backend.api.main as api_main
    import backend.core.depth.tile_infer as tile_infer

    monkeypatch.setattr(api_main, "infer_relative", lambda image: np.zeros(image.shape[:2], dtype=np.float32))
    monkeypatch.setattr(
        tile_infer,
        "infer_and_stitch",
        lambda image, tile_size=256, overlap_frac=0.25: np.zeros(image.shape[:2], dtype=np.float32),
    )

    bundle = export_validation_bundle(tmp_path, image_size=64)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    snapshot_path = pathlib.Path(__file__).with_name("validation_export_snapshot.json")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert bundle.ok is True
    assert manifest["ok"] is True
    assert manifest["image_size"] == snapshot["image_size"]
    assert [artifact["name"] for artifact in manifest["artifacts"]] == [artifact["name"] for artifact in snapshot["artifacts"]]
    assert [artifact["sha256"] for artifact in manifest["artifacts"]] == [artifact["sha256"] for artifact in snapshot["artifacts"]]
    assert [artifact["size_bytes"] for artifact in manifest["artifacts"]] == [artifact["size_bytes"] for artifact in snapshot["artifacts"]]

    assert [stage["name"] for stage in manifest["validation"]["stages"]] == [stage["name"] for stage in snapshot["validation"]["stages"]]
    assert [stage["status_code"] for stage in manifest["validation"]["stages"]] == [stage["status_code"] for stage in snapshot["validation"]["stages"]]
    assert [stage["ok"] for stage in manifest["validation"]["stages"]] == [stage["ok"] for stage in snapshot["validation"]["stages"]]

    for artifact_name in ("preview.png", "tile_predict.png"):
        assert (tmp_path / artifact_name).exists()

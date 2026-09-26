"""Export deterministic validation artifacts for golden tests and CI.

The exporter runs the preview and tiled inference endpoints against a generated
image and writes a small artifact bundle:

- `manifest.json`: stage metadata and SHA-256 hashes.
- `preview.png`: preview endpoint output.
- `tile_predict.png`: tiled inference output.

The bundle is deterministic when the depth inference functions are patched in
tests, which makes it suitable for golden snapshots.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.api.main import app
from backend.validation.harness import create_sample_image, _post_file, run_validation


@dataclass
class ExportedArtifact:
    name: str
    path: str
    sha256: str
    size_bytes: int


@dataclass
class ExportBundle:
    ok: bool
    image_size: int
    artifacts: list[ExportedArtifact]
    validation: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_validation_bundle(
    output_dir: str | Path,
    *,
    image_size: int = 256,
    include_calibration: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
) -> ExportBundle:
    """Run the validation pipeline and export PNG artifacts plus a manifest."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)
    image_bytes = create_sample_image(image_size)

    artifacts: list[ExportedArtifact] = []

    preview_status, preview_payload = _post_file(client, "/v1/depth/preview", image_bytes)
    if not isinstance(preview_payload, (bytes, bytearray)):
        raise RuntimeError(f"Preview endpoint did not return image bytes: {preview_payload!r}")
    preview_path = output_path / "preview.png"
    preview_path.write_bytes(bytes(preview_payload))
    artifacts.append(
        ExportedArtifact(
            name="preview",
            path=preview_path.name,
            sha256=_sha256_bytes(bytes(preview_payload)),
            size_bytes=len(preview_payload),
        )
    )

    tile_status, tile_payload = _post_file(client, "/v1/depth/tile_predict", image_bytes)
    if not isinstance(tile_payload, (bytes, bytearray)):
        raise RuntimeError(f"Tile endpoint did not return image bytes: {tile_payload!r}")
    tile_path = output_path / "tile_predict.png"
    tile_path.write_bytes(bytes(tile_payload))
    artifacts.append(
        ExportedArtifact(
            name="tile_predict",
            path=tile_path.name,
            sha256=_sha256_bytes(bytes(tile_payload)),
            size_bytes=len(tile_payload),
        )
    )

    calibration_payload: dict[str, Any] | None = None
    if include_calibration:
        if bbox is None:
            bbox = (12.0, 34.0, 12.1, 34.1)
        params = {
            "sun_azimuth_deg": 90.0,
            "sun_elevation_deg": 45.0,
            "gsd_m": 1.0,
            "minx": bbox[0],
            "miny": bbox[1],
            "maxx": bbox[2],
            "maxy": bbox[3],
            "dem_source": "SRTMGL1",
            "cache_dir": ".dem_cache",
        }
        calibration_status, calibration_payload = _post_file(
            client,
            "/v1/calibration/calibrate_depth",
            image_bytes,
            params=params,
        )
        calibration_payload = {
            "status_code": calibration_status,
            "payload": calibration_payload,
        }

    manifest = {
        "ok": preview_status == 200 and tile_status == 200,
        "image_size": image_size,
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "validation": asdict(run_validation(include_calibration=include_calibration, bbox=bbox, image_size=image_size)),
        "calibration": calibration_payload,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ExportBundle(
        ok=manifest["ok"],
        image_size=image_size,
        artifacts=artifacts,
        validation=manifest["validation"],
    )

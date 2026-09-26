"""Validation harness for the DepthWizard pipeline.

This module is importable both from the repository and from installed console
scripts. It runs a small synthetic image through the backend endpoints and
records timing and response status.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.main import app


@dataclass
class StageResult:
    name: str
    ok: bool
    status_code: int
    elapsed_ms: float
    detail: str | None = None


@dataclass
class ValidationReport:
    image_size: int
    stages: list[StageResult]

    @property
    def ok(self) -> bool:
        return all(stage.ok for stage in self.stages)


def create_sample_image(size: int = 256) -> bytes:
    """Generate a deterministic sample RGB image for smoke testing."""
    yy, xx = np.mgrid[0:size, 0:size]
    r = ((xx / max(size - 1, 1)) * 255).astype(np.uint8)
    g = ((yy / max(size - 1, 1)) * 255).astype(np.uint8)
    b = (((xx + yy) / max(2 * (size - 1), 1)) * 255).astype(np.uint8)
    rgb = np.dstack([r, g, b])
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _post_file(
    client: TestClient,
    path: str,
    image_bytes: bytes,
    params: dict[str, Any] | None = None,
) -> tuple[int, bytes | dict[str, Any]]:
    response = client.post(
        path,
        files={"file": ("sample.png", image_bytes, "image/png")},
        params=params or {},
    )
    if response.headers.get("content-type", "").startswith("image/"):
        return response.status_code, response.content
    try:
        return response.status_code, response.json()
    except Exception:
        return response.status_code, {"raw": response.text}


def run_validation(
    include_calibration: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    image_size: int = 256,
) -> ValidationReport:
    client = TestClient(app)
    image_bytes = create_sample_image(image_size)
    stages: list[StageResult] = []

    start = time.perf_counter()
    status, payload = _post_file(client, "/v1/depth/preview", image_bytes)
    elapsed = (time.perf_counter() - start) * 1000.0
    stages.append(
        StageResult(
            name="preview",
            ok=status == 200 and isinstance(payload, (bytes, bytearray)) and len(payload) > 0,
            status_code=status,
            elapsed_ms=elapsed,
            detail=f"bytes={len(payload)}" if isinstance(payload, (bytes, bytearray)) else str(payload),
        )
    )

    start = time.perf_counter()
    status, payload = _post_file(client, "/v1/depth/tile_predict", image_bytes)
    elapsed = (time.perf_counter() - start) * 1000.0
    stages.append(
        StageResult(
            name="tile_predict",
            ok=status == 200 and isinstance(payload, (bytes, bytearray)) and len(payload) > 0,
            status_code=status,
            elapsed_ms=elapsed,
            detail=f"bytes={len(payload)}" if isinstance(payload, (bytes, bytearray)) else str(payload),
        )
    )

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
        start = time.perf_counter()
        status, payload = _post_file(client, "/v1/calibration/calibrate_depth", image_bytes, params=params)
        elapsed = (time.perf_counter() - start) * 1000.0
        ok = status == 200 and isinstance(payload, dict) and payload.get("status") == "success"
        stages.append(
            StageResult(
                name="calibration",
                ok=ok,
                status_code=status,
                elapsed_ms=elapsed,
                detail=str(payload),
            )
        )

    return ValidationReport(image_size=image_size, stages=stages)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a smoke-test validation pipeline")
    parser.add_argument("--include-calibration", action="store_true", help="Also call the calibration endpoint")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"), help="Calibration bounding box in WGS84")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args(argv)

    bbox = tuple(args.bbox) if args.bbox else None
    report = run_validation(include_calibration=args.include_calibration, bbox=bbox, image_size=args.image_size)
    payload = {
        "ok": report.ok,
        "image_size": report.image_size,
        "stages": [asdict(stage) for stage in report.stages],
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")

    return 0 if report.ok else 1

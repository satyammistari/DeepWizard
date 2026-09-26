"""Tests for calibration API endpoint."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def simple_rgb_image() -> bytes:
    """Create a simple test RGB image."""
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    pil_img = Image.fromarray(img, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestCalibrationEndpoint:
    """Test the calibration endpoint."""

    def test_calibrate_depth_missing_params(self, client: TestClient, simple_rgb_image: bytes) -> None:
        """Missing required query parameters should fail."""
        response = client.post(
            "/v1/calibration/calibrate_depth",
            files={"file": ("test.png", simple_rgb_image, "image/png")},
        )
        assert response.status_code == 422  # Unprocessable entity

    def test_calibrate_depth_invalid_image(self, client: TestClient) -> None:
        """Invalid image should fail gracefully."""
        response = client.post(
            "/v1/calibration/calibrate_depth",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            params={
                "sun_azimuth_deg": 90.0,
                "sun_elevation_deg": 45.0,
                "gsd_m": 1.0,
                "minx": 12.0,
                "miny": 34.0,
                "maxx": 12.1,
                "maxy": 34.1,
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "status" in data
        assert data["status"] == "error"

    def test_calibrate_depth_basic(self, client: TestClient, simple_rgb_image: bytes) -> None:
        """Test basic calibration endpoint with network disabled."""
        # Network is disabled by default in settings, so DEM fetch will fail
        # But the endpoint should handle gracefully
        response = client.post(
            "/v1/calibration/calibrate_depth",
            files={"file": ("test.png", simple_rgb_image, "image/png")},
            params={
                "sun_azimuth_deg": 90.0,
                "sun_elevation_deg": 45.0,
                "gsd_m": 1.0,
                "minx": 12.0,
                "miny": 34.0,
                "maxx": 12.1,
                "maxy": 34.1,
                "dem_source": "SRTMGL1",
                "cache_dir": ".dem_cache_test",
            },
        )
        # May fail due to network/DEM unavailability, but endpoint should return 400 with error message
        data = response.json()
        # Either success or error response should have status field
        assert "status" in data
        if response.status_code != 200:
            assert response.status_code == 400
            assert data["status"] == "error"
        else:
            assert data["status"] == "success"

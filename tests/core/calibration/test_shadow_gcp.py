"""Tests for shadow-based GCP extraction."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.calibration.shadow_gcp import (
    GCP,
    detect_shadows,
    extract_shadow_gcps,
    parse_sun_from_metadata,
)


class TestDetectShadows:
    """Test shadow detection."""

    def test_detect_shadows_rgb_image(self) -> None:
        """Verify detect_shadows works with RGB images."""
        img = np.ones((8, 8, 3), dtype=np.uint8) * 150
        img[2:4, 2:4, :] = 30

        mask = detect_shadows(img, threshold_percentile=10.0)
        assert mask.shape == (8, 8)
        # Darkest pixels should be in the detected region
        darkest_row, darkest_col = np.unravel_index(np.argmin(img.mean(axis=2)), img.shape[:2])
        assert mask[darkest_row, darkest_col]

    def test_detect_shadows_float_image(self) -> None:
        """Verify detect_shadows works with float [0,1] images."""
        img = np.ones((10, 10), dtype=np.float32) * 0.8
        img[1:3, 1:3] = 0.1

        mask = detect_shadows(img, threshold_percentile=5.0)
        assert mask.shape == (10, 10)
        # Darkest region should have detected pixels
        assert mask[2, 2]


class TestExtractShadowGCPs:
    """Test shadow GCP extraction."""

    def test_extract_single_component(self) -> None:
        """Extract GCP from a single shadow component."""
        # Create a simple shadow mask with one rectangular region
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True  # 5x5 shadow

        gcps = extract_shadow_gcps(
            mask,
            sun_elevation_deg=45.0,
            sun_azimuth_deg=90.0,
            pixel_scale_m=1.0,
            min_area_pixels=5,
            morphology=False,
        )

        assert len(gcps) > 0
        gcp = gcps[0]
        assert isinstance(gcp, GCP)
        # Centroid should be around (7.5, 7.5)
        assert 7.0 <= gcp.row <= 8.0
        assert 7.0 <= gcp.col <= 8.0
        # With sun elevation 45°, tan(45°) = 1, so height ≈ shadow_length
        assert gcp.height_m > 0
        assert gcp.confidence > 0

    def test_extract_no_component_below_min_area(self) -> None:
        """Reject shadow components below min_area_pixels."""
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:2, 0:2] = True  # 2x2 = 4 pixels

        gcps = extract_shadow_gcps(
            mask,
            sun_elevation_deg=45.0,
            sun_azimuth_deg=0.0,
            pixel_scale_m=1.0,
            min_area_pixels=10,
            morphology=False,
        )

        assert len(gcps) == 0

    def test_parse_sun_from_metadata_basic(self) -> None:
        """Extract sun angles from metadata dict."""
        metadata = {
            "sun_azimuth": 90.0,
            "sun_elevation": 45.0,
        }
        az, el = parse_sun_from_metadata(metadata)
        assert az == 90.0
        assert el == 45.0

    def test_parse_sun_from_metadata_case_insensitive(self) -> None:
        """Metadata key lookup is case-insensitive."""
        metadata = {
            "SUN_AZIMUTH": 180.0,
            "SUN_ELEVATION": 30.0,
        }
        az, el = parse_sun_from_metadata(metadata)
        assert az == 180.0
        assert el == 30.0

    def test_parse_sun_from_metadata_missing_keys(self) -> None:
        """Raise KeyError if sun angles missing."""
        metadata = {"other_key": "value"}
        with pytest.raises(KeyError, match="Sun azimuth/elevation not found"):
            parse_sun_from_metadata(metadata)

    def test_gcp_dataclass(self) -> None:
        """Test GCP dataclass construction."""
        gcp = GCP(row=10.0, col=20.0, height_m=5.5, confidence=0.95)
        assert gcp.row == 10.0
        assert gcp.col == 20.0
        assert gcp.height_m == 5.5
        assert gcp.confidence == 0.95
        assert gcp.shadow_length_m is None

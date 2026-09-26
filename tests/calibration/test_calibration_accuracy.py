"""Integration tests for calibration accuracy using synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.calibration.fusion import CalibrationResult, calibrate
from backend.core.geo.io import RasterData
from backend.validation.evaluate import (
    compute_validation_metrics,
    create_synthetic_test_case,
)


class TestCalibrationAccuracy:
    """Test calibration accuracy using synthetic test data."""

    def test_calibration_with_synthetic_data(self) -> None:
        """Calibrate synthetic depth against reference DEM."""
        # Create synthetic test case
        dem, synthetic_depth, mask = create_synthetic_test_case(
            height=128, width=128, noise_std=0.02, seed=42
        )

        # Create RasterData objects
        rel_raster = RasterData(
            array=synthetic_depth,
            crs=None,
            transform=None,
            nodata=np.nan,
        )
        dem_raster = RasterData(
            array=dem,
            crs=None,
            transform=None,
            nodata=np.nan,
        )

        # Calibrate
        result = calibrate(rel_raster, dem_raster, mask=mask)

        # Apply calibration
        calibrated = result.apply(synthetic_depth)

        # Compute error metrics
        metrics = compute_validation_metrics(calibrated, dem, mask=mask)

        # Calibration should improve metrics
        # (calibrated should be closer to DEM than synthetic)
        assert metrics.r_squared > 0.9
        assert metrics.correlation > 0.95
        assert metrics.rmse < 20.0  # Some reasonable threshold

    def test_calibration_result_scaling(self) -> None:
        """Test that CalibrationResult.apply correctly scales values."""
        scale, offset = 2.0, 5.0
        result = CalibrationResult(scale=scale, offset=offset)

        test_array = np.array([[0, 1], [2, 3]], dtype=np.float64)
        expected = scale * test_array + offset

        calibrated = result.apply(test_array)
        np.testing.assert_allclose(calibrated, expected)

    def test_calibration_with_nan_values(self) -> None:
        """Calibration should handle NaN values gracefully."""
        rel = np.array([[0.1, 0.2], [np.nan, 0.4]], dtype=np.float64)
        dem = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

        rel_raster = RasterData(
            array=rel,
            crs=None,
            transform=None,
            nodata=np.nan,
        )
        dem_raster = RasterData(
            array=dem,
            crs=None,
            transform=None,
            nodata=np.nan,
        )

        # Should not raise an error
        result = calibrate(rel_raster, dem_raster)

        # Result should be valid
        assert isinstance(result, CalibrationResult)
        assert np.isfinite(result.scale)
        assert np.isfinite(result.offset)

    def test_calibration_insufficient_samples(self) -> None:
        """Raise ValueError if not enough valid samples for calibration."""
        rel = np.full((5, 5), np.nan, dtype=np.float64)
        rel[0, 0] = 0.1  # Only one valid sample
        dem = np.full((5, 5), np.nan, dtype=np.float64)
        dem[0, 0] = 10.0

        rel_raster = RasterData(
            array=rel,
            crs=None,
            transform=None,
            nodata=np.nan,
        )
        dem_raster = RasterData(
            array=dem,
            crs=None,
            transform=None,
            nodata=np.nan,
        )

        with pytest.raises(ValueError, match="Not enough valid samples"):
            calibrate(rel_raster, dem_raster)

    def test_calibration_linear_relationship(self) -> None:
        """Test calibration recovers known linear transformation."""
        # Create relative depth (simple ramp)
        relative = np.linspace(0, 1, 100).reshape(10, 10)

        # Reference elevation with known linear relationship
        # dem = 2.0 * relative + 5.0
        dem = 2.0 * relative + 5.0

        rel_raster = RasterData(array=relative, crs="EPSG:4326", transform=None)
        dem_raster = RasterData(array=dem, crs="EPSG:4326", transform=None)

        result = calibrate(rel_raster, dem_raster)

        # Should recover approximately the correct scale and offset
        assert result.scale == pytest.approx(2.0, rel=0.05)
        assert result.offset == pytest.approx(5.0, rel=0.05)

        # Applied calibration should match DEM closely
        calibrated = result.apply(relative)
        np.testing.assert_allclose(calibrated, dem, rtol=0.01)

"""Tests for validation and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from backend.validation.evaluate import (
    ValidationMetrics,
    compute_validation_metrics,
    create_synthetic_test_case,
    generate_synthetic_depth,
)


class TestSyntheticDataGeneration:
    """Test synthetic depth map generation."""

    def test_generate_synthetic_depth_basic(self) -> None:
        """Generate synthetic depth from a simple DEM."""
        dem = np.array([[0, 100], [50, 150]], dtype=np.float64)
        synthetic = generate_synthetic_depth(dem, noise_std=0.0, seed=42)

        assert synthetic.shape == dem.shape
        assert np.isfinite(synthetic).all()
        # Without noise, synthetic should be normalized DEM
        normalized = (dem - dem.min()) / (dem.max() - dem.min())
        np.testing.assert_allclose(synthetic, normalized, atol=1e-10)

    def test_generate_synthetic_depth_with_noise(self) -> None:
        """Add noise to synthetic depth."""
        dem = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float64)
        synthetic_noisy = generate_synthetic_depth(dem, noise_std=0.1, seed=42)
        synthetic_clean = generate_synthetic_depth(dem, noise_std=0.0, seed=42)

        # Noisy should differ from clean
        assert not np.allclose(synthetic_noisy, synthetic_clean)
        # But should be close (noise is small)
        mse = np.mean((synthetic_noisy - synthetic_clean) ** 2)
        assert mse < 0.05

    def test_generate_synthetic_depth_with_bias_scale(self) -> None:
        """Apply bias and scale to synthetic depth."""
        dem = np.array([[0, 100], [50, 150]], dtype=np.float64)
        bias = 0.1
        scale = 2.0

        synthetic = generate_synthetic_depth(dem, bias=bias, scale=scale, noise_std=0.0, seed=42)
        normalized = (dem - dem.min()) / (dem.max() - dem.min())
        expected = scale * normalized + bias

        np.testing.assert_allclose(synthetic, expected, atol=1e-10)

    def test_generate_synthetic_depth_preserves_nan(self) -> None:
        """NaN regions should be preserved in synthetic depth."""
        dem = np.array([[0.0, 100.0], [np.nan, 150.0]], dtype=np.float64)
        synthetic = generate_synthetic_depth(dem, noise_std=0.0, seed=42)

        assert np.isnan(synthetic[1, 0])
        assert np.isfinite(synthetic[0, 0])
        assert np.isfinite(synthetic[0, 1])
        assert np.isfinite(synthetic[1, 1])

    def test_generate_synthetic_depth_invalid_dem(self) -> None:
        """Raise ValueError for invalid DEMs."""
        dem_all_nan = np.full((5, 5), np.nan)
        with pytest.raises(ValueError, match="no valid pixels"):
            generate_synthetic_depth(dem_all_nan)

        dem_flat = np.ones((5, 5)) * 42.0
        with pytest.raises(ValueError, match="no elevation range"):
            generate_synthetic_depth(dem_flat)


class TestValidationMetrics:
    """Test error metric computation."""

    def test_perfect_prediction(self) -> None:
        """Metrics for perfect prediction (zero error)."""
        reference = np.array([[0, 10], [20, 30]], dtype=np.float64)
        predicted = reference.copy()

        metrics = compute_validation_metrics(predicted, reference)
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.mae == pytest.approx(0.0)
        assert metrics.bias == pytest.approx(0.0)
        assert metrics.r_squared == pytest.approx(1.0)
        assert metrics.correlation == pytest.approx(1.0)

    def test_constant_bias(self) -> None:
        """Metrics for constant bias."""
        reference = np.array([[0, 10], [20, 30]], dtype=np.float64)
        bias_offset = 5.0
        predicted = reference + bias_offset

        metrics = compute_validation_metrics(predicted, reference)
        assert metrics.rmse == pytest.approx(bias_offset)
        assert metrics.mae == pytest.approx(bias_offset)
        assert metrics.bias == pytest.approx(bias_offset)
        # Constant bias reduces R² (residuals are not zero)
        # But correlation is still perfect (linear relationship)
        assert metrics.correlation == pytest.approx(1.0)
        assert 0.0 < metrics.r_squared < 1.0  # R² between 0 and 1 for biased prediction

    def test_proportional_error(self) -> None:
        """Metrics for proportional (scale) error."""
        reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        # Small scale factor to keep errors reasonable
        scale_factor = 1.1
        predicted = reference * scale_factor

        metrics = compute_validation_metrics(predicted, reference)
        # Error proportional to reference magnitude
        expected_mae = np.mean(np.abs(predicted - reference))
        assert metrics.mae == pytest.approx(expected_mae)
        # Still highly correlated even with scale error
        assert metrics.correlation == pytest.approx(1.0)
        # R² is reasonable but not perfect
        assert metrics.r_squared > 0.0  # Should be positive for small scale errors

    def test_random_noise(self) -> None:
        """Metrics with random noise."""
        np.random.seed(42)
        reference = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float64)
        noise = np.random.normal(0, 2.0, reference.shape)
        predicted = reference + noise

        metrics = compute_validation_metrics(predicted, reference)
        # Noise should result in moderate RMSE
        assert metrics.rmse > 1.0
        assert metrics.rmse < 5.0
        # But should still be highly correlated
        assert metrics.correlation > 0.99
        # Unbiased random noise
        assert abs(metrics.bias) < 0.5

    def test_validation_metrics_with_mask(self) -> None:
        """Validation with custom mask."""
        reference = np.array([[0, 10], [20, 30]], dtype=np.float64)
        predicted = np.array([[1, 11], [22, 29]], dtype=np.float64)
        mask = np.array([[True, True], [False, False]], dtype=bool)

        metrics_masked = compute_validation_metrics(predicted, reference, mask=mask)
        metrics_full = compute_validation_metrics(predicted, reference)

        # Masked should only use first row
        assert metrics_masked.num_valid_samples == 2
        assert metrics_full.num_valid_samples == 4
        # First row errors
        expected_rmse_masked = np.sqrt(np.mean([1.0**2, 1.0**2]))
        assert metrics_masked.rmse == pytest.approx(expected_rmse_masked)

    def test_validation_metrics_no_valid_samples(self) -> None:
        """Raise ValueError when no valid samples."""
        reference = np.full((5, 5), np.nan)
        predicted = np.full((5, 5), np.nan)

        with pytest.raises(ValueError, match="No valid samples"):
            compute_validation_metrics(predicted, reference)


class TestSyntheticTestCase:
    """Test the synthetic test case generator."""

    def test_create_synthetic_test_case_shape(self) -> None:
        """Synthetic test case has correct shape."""
        height, width = 64, 96
        dem, synthetic_depth, mask = create_synthetic_test_case(height=height, width=width)

        assert dem.shape == (height, width)
        assert synthetic_depth.shape == (height, width)
        assert mask.shape == (height, width)
        assert mask.dtype == bool

    def test_create_synthetic_test_case_validity(self) -> None:
        """Synthetic test case has valid data in mask regions."""
        dem, synthetic_depth, mask = create_synthetic_test_case(seed=42)

        # Valid regions should have finite data
        assert np.isfinite(dem[mask]).all()
        assert np.isfinite(synthetic_depth[mask]).all()
        # Edge regions should be masked out
        assert not mask[0, 0]
        assert not mask[-1, -1]

    def test_create_synthetic_test_case_correlation(self) -> None:
        """Synthetic depth should be correlated with DEM."""
        dem, synthetic_depth, mask = create_synthetic_test_case(noise_std=0.01, seed=42)

        # Within mask, should be highly correlated
        metrics = compute_validation_metrics(synthetic_depth, dem, mask=mask)
        assert metrics.correlation > 0.95  # Very high correlation

    def test_create_synthetic_test_case_reproducibility(self) -> None:
        """Same seed should produce identical results."""
        dem1, syn1, mask1 = create_synthetic_test_case(seed=42)
        dem2, syn2, mask2 = create_synthetic_test_case(seed=42)

        np.testing.assert_array_equal(dem1, dem2)
        np.testing.assert_array_equal(syn1, syn2)
        np.testing.assert_array_equal(mask1, mask2)


class TestValidationMetricsDataclass:
    """Test ValidationMetrics dataclass."""

    def test_validation_metrics_repr(self) -> None:
        """Repr should be human-readable."""
        metrics = ValidationMetrics(
            rmse=2.5,
            mae=2.0,
            r_squared=0.95,
            correlation=0.97,
            bias=0.1,
            std_error=1.2,
            num_valid_samples=1000,
        )
        repr_str = repr(metrics)
        assert "rmse=2.50m" in repr_str
        assert "mae=2.00m" in repr_str
        assert "r²=0.950" in repr_str
        assert "n=1000" in repr_str

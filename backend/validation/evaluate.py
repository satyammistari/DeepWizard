"""Validation and evaluation metrics for calibration pipeline.

This module provides tools to:
1. Generate synthetic test data (depth maps + reference elevations).
2. Evaluate calibration accuracy against known DSMs.
3. Compute standard error metrics (RMSE, MAE, R²).
4. Generate validation reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ValidationMetrics:
    """Standard error metrics for calibration validation."""
    rmse: float
    mae: float
    r_squared: float
    correlation: float
    bias: float
    std_error: float
    num_valid_samples: int

    def __repr__(self) -> str:
        return (
            f"ValidationMetrics(rmse={self.rmse:.2f}m, "
            f"mae={self.mae:.2f}m, r²={self.r_squared:.3f}, "
            f"corr={self.correlation:.3f}, "
            f"n={self.num_valid_samples})"
        )


def generate_synthetic_depth(
    dem_array: np.ndarray,
    noise_std: float = 0.05,
    bias: float = 0.0,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic monocular depth from a reference DEM.

    Steps:
    1. Normalize DEM to [0, 1] relative depth space.
    2. Apply scale and bias transformation.
    3. Add Gaussian noise.

    Args:
        dem_array: Reference elevation array (H, W).
        noise_std: Standard deviation of Gaussian noise (as fraction of range).
        bias: Constant bias to add.
        scale: Scale factor for depth values.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic relative depth array (H, W).
    """
    if seed is not None:
        np.random.seed(seed)

    dem = np.asarray(dem_array, dtype=np.float64)
    valid = ~np.isnan(dem)

    if not valid.any():
        raise ValueError("DEM has no valid pixels")

    # Normalize to [0, 1]
    lo = float(np.nanmin(dem))
    hi = float(np.nanmax(dem))
    if hi <= lo:
        raise ValueError("DEM has no elevation range")

    relative = (dem - lo) / (hi - lo)

    # Apply scale and bias
    synthetic = scale * relative + bias

    # Add noise
    noise = np.random.normal(0, noise_std, relative.shape)
    synthetic = synthetic + noise

    # Preserve NaN regions
    synthetic[~valid] = np.nan

    return synthetic


def compute_validation_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> ValidationMetrics:
    """Compute standard error metrics comparing predicted to reference elevations.

    Args:
        predicted: Predicted elevation array (H, W).
        reference: Reference elevation array (H, W).
        mask: Optional binary mask of valid pixels; if None, uses ~isnan.

    Returns:
        ValidationMetrics dataclass with computed error metrics.
    """
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()

    if mask is not None:
        mask = np.asarray(mask, dtype=bool).ravel()
    else:
        mask = ~(np.isnan(pred) | np.isnan(ref))

    if not mask.any():
        raise ValueError("No valid samples for validation")

    pred_valid = pred[mask]
    ref_valid = ref[mask]
    n = pred_valid.size

    # Error metrics
    residuals = pred_valid - ref_valid
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    std_error = float(np.std(residuals))

    # Correlation and R²
    correlation = float(np.corrcoef(pred_valid, ref_valid)[0, 1])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ref_valid - np.mean(ref_valid))**2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return ValidationMetrics(
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        correlation=correlation,
        bias=bias,
        std_error=std_error,
        num_valid_samples=n,
    )


def create_synthetic_test_case(
    height: int = 128,
    width: int = 128,
    dem_range_m: float = 100.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic test case: reference DEM, synthetic depth, and mask.

    Args:
        height, width: Dimensions of the test case.
        dem_range_m: Range of elevation values in the reference DEM.
        noise_std: Standard deviation of noise for synthetic depth.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (reference_dem, synthetic_depth, valid_mask).
    """
    np.random.seed(seed)

    # Create a smooth elevation surface with a ridge and valley
    xx = np.linspace(-2, 2, width)
    yy = np.linspace(-2, 2, height)
    XX, YY = np.meshgrid(xx, yy)

    # Combine multiple features (ridge, valley, peak)
    dem = (
        dem_range_m / 2.0 * (np.sin(XX) + np.cos(YY))
        + dem_range_m / 4.0 * np.exp(-((XX**2 + YY**2) / 2.0))
    )

    # Add some noise to make it realistic
    dem = dem + np.random.normal(0, dem_range_m * 0.02, dem.shape)

    # Create mask (some regions marked as invalid)
    mask = np.ones((height, width), dtype=bool)
    mask[0:10, :] = False  # edge artifacts
    mask[-10:, :] = False
    mask[:, 0:10] = False
    mask[:, -10:] = False

    # Generate synthetic depth
    synthetic_depth = generate_synthetic_depth(
        dem, noise_std=noise_std, seed=seed
    )

    return dem, synthetic_depth, mask

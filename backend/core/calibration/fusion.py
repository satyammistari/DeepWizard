"""Multi-source scale calibration: relative depth → metres + uncertainty.

Dense DEM pixels and sparse GCPs are treated as Anchor samples. A Huber fit
is compared with RANSAC; the higher inlier ratio wins. Vertical units are
metres (orthometric when the DEM is EGM96/EGM2008; ellipsoid if a WGS84
ellipsoid product is supplied — documented on the result).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

from backend.config import settings
from backend.core.calibration.dem_source import reproject_to_match
from backend.core.geo.io import RasterData


@dataclass
class Anchor:
    """Sparse metric samples colocated with relative depth."""

    x: float
    y: float
    height_m: float
    confidence: float = 1.0
    source: str = "unknown"


@dataclass
class CalibrationResult:
    """Linear calibration: calibrated = scale * relative + offset.

    Heights are metres. Uncertainty is 1-sigma metres per pixel.
    """

    scale: float
    offset: float
    uncertainty: NDArray[np.float32] | None = None
    low_confidence: bool = False
    reason: str = ""
    inlier_ratio: float = 1.0
    method: str = "lstsq"
    residual_std_m: float = 0.0
    vertical_datum: str = "EGM96"

    def apply(self, rel_array: np.ndarray) -> np.ndarray:
        """Apply calibration to a relative depth array (same shape, metres)."""
        return self.scale * np.asarray(rel_array, dtype=np.float64) + self.offset


def _inlier_ratio(pred: NDArray, y: NDArray, threshold: float) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - y) <= threshold))


def _fit_affine(
    xs: NDArray,
    ys: NDArray,
    weights: NDArray | None = None,
) -> tuple[float, float, float, str, float]:
    """Return (scale, offset, residual_std, method, inlier_ratio)."""
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError("Not enough valid samples for calibration")
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    w = np.clip(w, 1e-6, None)
    if x.size > 12000:
        rng = np.random.default_rng(0)
        pick = rng.choice(x.size, 12000, replace=False)
        x, y, w = x[pick], y[pick], w[pick]

    A = np.column_stack((x, np.ones_like(x)))
    sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    lstsq_scale, lstsq_offset = float(sol[0]), float(sol[1])
    huber_scale, huber_offset = lstsq_scale, lstsq_offset
    ransac_scale, ransac_offset, ransac_inliers = lstsq_scale, lstsq_offset, 0.0
    X = x.reshape(-1, 1)

    try:
        huber = HuberRegressor(epsilon=1.35, max_iter=200)
        huber.fit(X, y, sample_weight=w)
        huber_scale = float(huber.coef_[0])
        huber_offset = float(huber.intercept_)
    except Exception:
        pass

    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=0.5,
            residual_threshold=None,
            random_state=0,
        )
        ransac.fit(X, y, sample_weight=w)
        ransac_scale = float(ransac.estimator_.coef_[0])
        ransac_offset = float(ransac.estimator_.intercept_)
        ransac_inliers = float(np.mean(ransac.inlier_mask_)) if ransac.inlier_mask_ is not None else 0.0
    except Exception:
        pass

    spread = float(np.std(y - (lstsq_scale * x + lstsq_offset)))
    gate = max(1.0, 2.5 * spread if spread > 0 else 1.0)
    candidates = [
        ("huber", huber_scale, huber_offset, _inlier_ratio(huber_scale * x + huber_offset, y, gate)),
        ("ransac", ransac_scale, ransac_offset, ransac_inliers),
        ("lstsq", lstsq_scale, lstsq_offset, _inlier_ratio(lstsq_scale * x + lstsq_offset, y, gate)),
    ]
    method, scale, offset, inliers = max(candidates, key=lambda c: c[3])
    resid = y - (scale * x + offset)
    residual_std = float(np.sqrt(np.mean(resid * resid)))
    return scale, offset, residual_std, method, float(inliers)


def _uncertainty_map(
    relative: NDArray,
    residual_std: float,
    anchors_rc: Sequence[tuple[float, float, float]] | None,
) -> NDArray[np.float32]:
    """Per-pixel 1-sigma metres: residual spread grows away from anchors."""
    h, w = relative.shape[:2]
    base = np.full((h, w), max(residual_std, 0.5), dtype=np.float32)
    finite = np.isfinite(relative)
    base[~finite] = np.nan
    if not anchors_rc:
        return base.astype(np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.full((h, w), np.inf, dtype=np.float64)
    for row, col, conf in anchors_rc:
        d = np.hypot(yy - row, xx - col) / max(h, w)
        dist = np.minimum(dist, d / max(conf, 0.05))
    growth = 1.0 + 3.0 * np.clip(dist, 0.0, 1.0)
    out = (base * growth).astype(np.float32)
    out[~finite] = np.nan
    return out


def calibrate_from_samples(
    relative: NDArray,
    values_m: NDArray,
    *,
    weights: NDArray | None = None,
    anchors_rc: Sequence[tuple[float, float, float]] | None = None,
    source_predictions: dict[str, NDArray] | None = None,
    disagreement_m: float | None = None,
    vertical_datum: str = "EGM96",
) -> CalibrationResult:
    """Fit relative → metres from colocated samples."""
    rel = np.asarray(relative, dtype=np.float64)
    val = np.asarray(values_m, dtype=np.float64)
    valid = np.isfinite(rel) & np.isfinite(val)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        valid &= np.isfinite(w) & (w > 0)
    else:
        w = None
    xs = rel[valid]
    ys = val[valid]
    ww = None if w is None else w[valid]
    scale, offset, residual_std, method, inliers = _fit_affine(xs, ys, ww)
    metric = scale * rel + offset
    if rel.ndim == 2:
        uncertainty = _uncertainty_map(rel, residual_std, anchors_rc)
    else:
        uncertainty = None

    low = False
    reasons: list[str] = []
    gate = disagreement_m if disagreement_m is not None else settings.source_disagreement_m
    if source_predictions and len(source_predictions) >= 2 and rel.ndim == 2:
        stacked = []
        for arr in source_predictions.values():
            a = np.asarray(arr, dtype=np.float64)
            if a.shape != metric.shape:
                continue
            stacked.append(a)
        if len(stacked) >= 2:
            spread = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
            share = float(np.nanmean(spread > gate))
            if share > 0.15:
                low = True
                reasons.append(
                    f"Anchor sources disagree by more than {gate:.1f} m on {share:.0%} of shared pixels"
                )
                if uncertainty is None:
                    uncertainty = spread.astype(np.float32)
                else:
                    uncertainty = np.maximum(uncertainty, spread.astype(np.float32))

    if inliers < 0.5:
        low = True
        reasons.append(f"Low inlier ratio ({inliers:.2f}) for the affine depth-to-metre fit")
    if residual_std > gate:
        low = True
        reasons.append(f"Fit residual {residual_std:.1f} m exceeds the {gate:.1f} m disagreement gate")

    return CalibrationResult(
        scale=scale,
        offset=offset,
        uncertainty=None if uncertainty is None else np.asarray(uncertainty, dtype=np.float32),
        low_confidence=low,
        reason="; ".join(reasons),
        inlier_ratio=inliers,
        method=method,
        residual_std_m=residual_std,
        vertical_datum=vertical_datum,
    )


def calibrate(
    relative: RasterData,
    dem: RasterData,
    *,
    mask: Optional[np.ndarray] = None,
) -> CalibrationResult:
    """Estimate a linear mapping from `relative` to `dem`.

    Steps:
    - Reproject DEM to match `relative` grid if needed.
    - Extract valid pixels and run robust regression for dem = a*rel + b.

    Args:
        relative: Relative depth raster.
        dem: Reference elevation raster (metres).
        mask: Optional binary mask of valid pixels.

    Returns:
        CalibrationResult(scale=a, offset=b).
    """
    same_grid = (
        relative.crs == dem.crs
        and relative.transform == dem.transform
        and relative.array.shape[-2:] == dem.array.shape[-2:]
    )
    if not same_grid and relative.transform is not None and dem.transform is not None:
        dem = reproject_to_match(dem, relative)

    rel = np.asarray(relative.array)
    elev = np.asarray(dem.array)
    if rel.ndim == 3:
        rel = rel[0]
    if elev.ndim == 3:
        elev = elev[0]

    valid = np.isfinite(rel) & np.isfinite(elev)
    if mask is not None:
        valid &= mask.astype(bool)
    if relative.nodata is not None and np.isfinite(relative.nodata):
        valid &= rel != relative.nodata
    if dem.nodata is not None and np.isfinite(dem.nodata):
        valid &= elev != dem.nodata

    if int(valid.sum()) < 2:
        raise ValueError("Not enough valid samples for calibration")

    datum = str(dem.metadata.get("vertical_datum", settings.vertical_datum)) if dem.metadata else settings.vertical_datum
    return calibrate_from_samples(
        rel,
        elev,
        weights=valid.astype(np.float64),
        vertical_datum=datum,
    )

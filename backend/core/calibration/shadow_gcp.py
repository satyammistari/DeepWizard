"""Shadow-based GCP extraction with morphological filtering and georeferencing.

This module provides shadow detection, connected-component analysis, and GCP
extraction for calibrating monocular depth to metric elevations. Shadow geometry
constraints (object height = shadow_length * tan(sun_elevation)) link image
observations to DEM-based control points.

Key functions:
- detect_shadows: threshold-based mask from image intensity.
- extract_shadow_gcps: connected-component analysis + morphological cleanup.
- shadow_heights: estimate object heights from shadow geometry.
- parse_sun_from_metadata: extract sun angles from image metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Optional scipy/cv2 imports for advanced features
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class GCP:
    """Ground Control Point from shadow geometry.

    Attributes:
      row, col: pixel location in source image (row down, col right).
      height_m: estimated object height (metres AGL).
      confidence: [0, 1] confidence score.
      shadow_length_m: shadow extent in metres (if computed).
    """
    row: float
    col: float
    height_m: float
    confidence: float = 1.0
    shadow_length_m: Optional[float] = None


def parse_sun_from_metadata(metadata: dict) -> Tuple[float, float]:
    """Extract sun azimuth and elevation (degrees) from image metadata.

    Looks for common keys (case-insensitive) returned by sensors or ortho
    products. Raises KeyError if no sensible keys are present.

    Returns: (azimuth_deg, elevation_deg)
    """
    keys = {k.lower(): v for k, v in (metadata or {}).items()}
    # common name aliases
    cand_az = ["sun_azimuth", "solar_azimuth", "sunazimuth", "solarz"]
    cand_el = ["sun_elevation", "solar_elevation", "solar_altitude", "sunaltitude"]
    az = None
    el = None
    for k in cand_az:
        if k in keys:
            az = float(keys[k])
            break
    for k in cand_el:
        if k in keys:
            el = float(keys[k])
            break
    if az is None or el is None:
        raise KeyError("Sun azimuth/elevation not found in metadata keys")
    return az, el


def sun_angles(*args, **kwargs):
    """Solar angles from metadata dict OR (lat, lon, timestamp).

    Metadata path returns (azimuth_deg, elevation_deg).
    Location path returns (elevation_deg, azimuth_deg) per pysolar convention
    documented on sun_position().
    """
    if args and isinstance(args[0], dict):
        return parse_sun_from_metadata(args[0])
    return sun_position(*args, **kwargs)


def detect_shadows(image: np.ndarray, threshold_percentile: float = 10.0, nir: np.ndarray | None = None) -> np.ndarray:
    """Detect shadow pixels using low HSV value (and low NIR when provided).

    Args:
      image: (H,W) or (H,W,3) image array (uint8 or float in [0,1]).
      threshold_percentile: percentile for shadow intensity cutoff (0-100).
      nir: optional near-infrared band, same H/W. Shadows are dark in NIR too.

    Returns:
      Boolean mask (H,W) where True = shadow candidate.
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3].astype(np.float64)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        gray = rgb.mean(axis=2)
        vmax = rgb.max(axis=2)
        vmin = rgb.min(axis=2)
        value = vmax
        saturation = np.divide(vmax - vmin, np.clip(vmax, 1e-6, None))
        score = 0.65 * value + 0.35 * gray
        score = score + 0.1 * saturation
    else:
        gray = arr if arr.ndim == 2 else arr[:, :, 0]
        gray = gray.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0
        score = gray

    if nir is not None:
        band = np.asarray(nir, dtype=np.float64)
        if band.max() > 1.0:
            band = band / 255.0
        score = 0.5 * score + 0.5 * band

    cutoff = np.percentile(score, threshold_percentile)
    return score <= cutoff


def extract_shadow_gcps(
    shadow_mask: np.ndarray,
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    pixel_scale_m: float,
    min_area_pixels: int = 20,
    morphology: bool = True,
    filter_by_sun_direction: bool = False,
    direction_tol_deg: float = 35.0,
) -> list[GCP]:
    """Extract GCPs from connected shadow components.

    Steps:
    1. Morphological closing (optional) to merge nearby shadows.
    2. Connected-component labeling.
    3. Estimate object height from component size and sun geometry.
    4. Return list of GCP objects with image coordinates and heights.

    Args:
      shadow_mask: Boolean (H,W) mask from detect_shadows.
      sun_elevation_deg: sun elevation angle in degrees.
      sun_azimuth_deg: sun azimuth (degrees from north, clockwise).
      pixel_scale_m: ground metres per pixel (GSD).
      min_area_pixels: minimum component size to consider.
      morphology: apply morphological closing if True and scipy available.

    Returns:
      List of GCP objects.
    """
    if pixel_scale_m <= 0:
        raise ValueError("pixel_scale_m (GSD) must be positive; shadow heights need georeferencing")

    gcps = []
    mask = shadow_mask.copy()

    # Morphological cleanup if available
    if morphology and HAS_SCIPY:
        struct = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_closing(mask, structure=struct, iterations=2)

    # Connected-component analysis
    if HAS_SCIPY:
        labeled, num_features = ndimage.label(mask)
    elif HAS_CV2:
        num_features, labeled = cv2.connectedComponents(mask.astype(np.uint8))
    else:
        # Fallback: treat entire mask as one component
        if mask.sum() > min_area_pixels:
            labeled = np.zeros_like(mask, dtype=int)
            labeled[mask] = 1
            num_features = 1
        else:
            return gcps

    sun_el_rad = np.deg2rad(sun_elevation_deg)
    tan_el = np.tan(sun_el_rad)

    # Extract centroid and compute height for each component
    for comp_id in range(1, num_features + 1):
        comp_mask = labeled == comp_id
        area = comp_mask.sum()
        if area < min_area_pixels:
            continue

        # Centroid
        rows, cols = np.where(comp_mask)
        row_c = float(rows.mean())
        col_c = float(cols.mean())

        # Estimate shadow length from component size
        # Simple heuristic: shadow_length_pix ~ sqrt(area)
        shadow_length_pix = np.sqrt(float(area))
        shadow_length_m = shadow_length_pix * pixel_scale_m

        # Height = shadow_length * tan(sun_elevation)
        height_m = shadow_length_m * tan_el

        # Confidence from compactness (circle = 1) and optional sun-direction match.
        compactness = float(4.0 * np.pi * area / max((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1), 1.0))
        compactness = float(np.clip(compactness, 0.05, 1.0))
        direction_quality = _shadow_axis_alignment(rows, cols, sun_azimuth_deg)
        if filter_by_sun_direction and direction_quality < np.cos(np.deg2rad(direction_tol_deg)):
            continue

        confidence = min(1.0, float(area) / (min_area_pixels * 10))
        confidence *= 0.5 + 0.5 * compactness
        confidence *= 0.5 + 0.5 * direction_quality

        gcp = GCP(
            row=row_c,
            col=col_c,
            height_m=height_m,
            confidence=confidence,
            shadow_length_m=shadow_length_m,
        )
        gcps.append(gcp)

    return gcps


def shadow_heights(
    shadow_mask: np.ndarray,
    sun_elevation_deg: float,
    pixel_scale_m: Optional[float] = None,
) -> list[float]:
    """Estimate object heights from shadow extent (deprecated; use extract_shadow_gcps).

    Args:
      shadow_mask: Boolean (H,W) shadow mask.
      sun_elevation_deg: sun elevation in degrees.
      pixel_scale_m: ground metres per pixel.

    Returns:
      List of height estimates (metres).
    """
    if pixel_scale_m is None:
        return []

    area_pixels = float(shadow_mask.sum())
    if area_pixels < 10:
        return []

    length_m = np.sqrt(area_pixels) * pixel_scale_m
    h = length_m * np.tan(np.deg2rad(sun_elevation_deg))
    return [h]


def _shadow_axis_alignment(rows: np.ndarray, cols: np.ndarray, sun_azimuth_deg: float) -> float:
    """How well the blob major axis matches the expected shadow direction.

    Image convention: +col = east, +row = south (north-up). Expected shadow
    extends opposite the sun azimuth (clockwise from north).
    """
    az = np.deg2rad(sun_azimuth_deg)
    expected = np.array([-np.sin(az), np.cos(az)], dtype=np.float64)  # dcol, drow
    pts = np.column_stack((cols.astype(np.float64), rows.astype(np.float64)))
    if pts.shape[0] < 4:
        return 0.0
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    denom = float(np.linalg.norm(axis) * np.linalg.norm(expected)) + 1e-9
    return abs(float(np.dot(axis, expected) / denom))


def sun_position(lat: float, lon: float, timestamp) -> tuple[float, float]:
    """Solar elevation and azimuth (degrees) via pysolar.

    timestamp: timezone-aware datetime. Azimuth is clockwise from north.
    Elevation is degrees above the horizon.
    """
    from datetime import timezone

    from pysolar.solar import get_altitude, get_azimuth

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    elevation = float(get_altitude(lat, lon, timestamp))
    azimuth = float(get_azimuth(lat, lon, timestamp)) % 360.0
    return elevation, azimuth


def parse_sun_from_metadata_text(text: str) -> tuple[float, float]:
    """Parse WorldView .IMD/.XML or Landsat MTL.txt sun angles.

    Prefers explicit metadata over computed ephemeris. Returns
    (azimuth_deg, elevation_deg).
    """
    import re

    patterns = [
        ("azimuth", r"(?:sun)?[_\s]*azimuth(?:[_\s]*angle)?[\"'\s:=]+([-+0-9.]+)"),
        ("elevation", r"(?:sun)?[_\s]*(?:elevation|altitude)(?:[_\s]*angle)?[\"'\s:=]+([-+0-9.]+)"),
    ]
    lowered = text.lower()
    az = el = None
    for name, pat in patterns:
        match = re.search(pat, lowered)
        if match:
            if name == "azimuth":
                az = float(match.group(1))
            else:
                el = float(match.group(1))
    # Landsat MTL: SUN_AZIMUTH / SUN_ELEVATION
    if az is None:
        match = re.search(r"sun_azimuth\s*=\s*([-+0-9.]+)", lowered)
        if match:
            az = float(match.group(1))
    if el is None:
        match = re.search(r"sun_elevation\s*=\s*([-+0-9.]+)", lowered)
        if match:
            el = float(match.group(1))
    if az is None or el is None:
        raise KeyError("Sun azimuth/elevation not found in metadata text")
    return az, el


def parse_sun_from_file(path: str) -> tuple[float, float]:
    from pathlib import Path

    return parse_sun_from_metadata_text(Path(path).read_text(encoding="utf-8", errors="ignore"))

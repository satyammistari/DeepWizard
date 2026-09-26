"""End-to-end relative depth → maps → metric DSM pipeline.

Heights in metres. Vertical datum is taken from the DEM product (EGM96 for
SRTM/CartoDEM, EGM2008 for COP30) or labelled EGM96 for the offline prior.
Nodata stays non-finite; never rewritten as 0 m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import from_bounds

from backend.config import settings
from backend.core.calibration.dem_source import resolve_dem
from backend.core.calibration.fusion import CalibrationResult, calibrate, calibrate_from_samples
from backend.core.calibration.semantic_prior import LandCover, apply_ground_constraint, hgdnet_snap_to_ground
from backend.core.calibration.shadow_gcp import detect_shadows, extract_shadow_gcps
from backend.core.depth.backbone import infer_relative
from backend.core.depth.tile_infer import infer_and_stitch
from backend.core.geo.io import RasterData
from backend.core.viz.maps import colorize, encode_png, encode_png_b64, encode_png_hex, hillshade, to_unit_interval
from backend.validation.evaluate import compute_validation_metrics


def infer_scene_depth(image: NDArray, *, tile_size: int = 256, overlap_frac: float = 0.25) -> NDArray[np.float32]:
    """Relative inverse-depth, higher = closer/taller. Shape (H, W), unitless."""
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    if max(h, w) > tile_size:
        depth = infer_and_stitch(rgb, tile_size=tile_size, overlap_frac=overlap_frac)
    else:
        depth = infer_relative(rgb)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape[:2] != (h, w):
        raise ValueError(f"Depth shape {depth.shape} does not match image {(h, w)}")
    return depth


def _cheap_landcover(rgb: NDArray) -> NDArray[np.int32]:
    """RGB-only land-cover proxy for ground snapping when no classifier is bundled."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        gray = arr
        r = g = b = arr
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = arr.mean(axis=-1)
    if gray.max() > 1.5:
        r, g, b, gray = r / 255.0, g / 255.0, b / 255.0, gray / 255.0
    out = np.full(gray.shape, int(LandCover.BACKGROUND), dtype=np.int32)
    ndvi_like = (g - r) / np.clip(g + r, 1e-6, None)
    out[ndvi_like > 0.12] = int(LandCover.TREE)
    out[(gray < 0.18)] = int(LandCover.WATER)
    out[(gray > 0.22) & (gray < 0.45) & (np.abs(r - g) < 0.08)] = int(LandCover.ROAD)
    out[(gray > 0.55) & (np.abs(r - g) < 0.05)] = int(LandCover.BARE_GROUND)
    building = (gray > 0.35) & (ndvi_like < 0.05) & (np.abs(r - b) < 0.12)
    out[building] = int(LandCover.BUILDING)
    return out


def _overlay(rgb: NDArray, color_map: NDArray[np.uint8], alpha: float = 0.55) -> NDArray[np.uint8]:
    base = np.asarray(rgb)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)
    if base.shape[-1] == 4:
        base = base[..., :3]
    if base.dtype != np.uint8:
        if base.max() <= 1.0:
            base = (np.clip(base, 0, 1) * 255.0).astype(np.uint8)
        else:
            base = np.clip(base, 0, 255).astype(np.uint8)
    if base.shape[:2] != color_map.shape[:2]:
        raise ValueError("Overlay requires matching H/W")
    mix = (1.0 - alpha) * base.astype(np.float32) + alpha * color_map.astype(np.float32)
    return np.clip(mix, 0, 255).astype(np.uint8)


def preview_payload(image: NDArray) -> dict[str, Any]:
    depth = infer_relative(np.asarray(image))
    unit = to_unit_interval(depth)
    gray = (unit * 255.0).astype(np.uint8)
    color = colorize(depth)
    return {
        "width": int(depth.shape[1]),
        "height": int(depth.shape[0]),
        "relative_png": encode_png(gray),
        "relative_color_b64": encode_png_b64(color),
        "overlay_b64": encode_png_b64(_overlay(image, color)),
        "relative_gray_b64": encode_png_b64(gray),
    }


def tile_payload(image: NDArray, *, tile_size: int = 256, overlap_frac: float = 0.25) -> bytes:
    depth = infer_scene_depth(image, tile_size=tile_size, overlap_frac=overlap_frac)
    unit = to_unit_interval(depth)
    return encode_png((unit * 255.0).astype(np.uint8))


@dataclass
class CalibrationMaps:
    relative_depth: NDArray[np.float32]
    metric_dsm: NDArray[np.float64]
    uncertainty: NDArray[np.float32]
    result: CalibrationResult
    gcps: list
    used_dem_fallback: bool
    dem_source: str
    maps_b64: dict[str, str] = field(default_factory=dict)
    calibrated_png_hex: str = ""
    metrics: dict[str, float] | None = None


def calibrate_scene(
    image: NDArray,
    *,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
    gsd_m: float,
    bounds_wgs84: tuple[float, float, float, float],
    dem_source: str = "SRTMGL1",
    cache_dir: str | None = None,
) -> CalibrationMaps:
    if gsd_m <= 0:
        raise ValueError("GSD must be positive metres/pixel for shadow heights and georeferencing")

    relative = infer_scene_depth(image)
    cache = cache_dir or str(settings.dem_cache_dir)
    dem_raster, used_fallback = resolve_dem(
        bounds_wgs84,
        dem_source,  # type: ignore[arg-type]
        cache,
        relative_depth=relative,
    )

    h, w = relative.shape
    west, south, east, north = bounds_wgs84
    transform = from_bounds(west, south, east, north, w, h)
    rel_raster = RasterData(
        array=relative,
        transform=transform,
        crs="EPSG:4326",
        nodata=np.nan,
        metadata={"vertical_datum": "relative-inverse-depth"},
    )

    shadow_mask = detect_shadows(image, threshold_percentile=12.0)
    gcps = extract_shadow_gcps(
        shadow_mask,
        sun_elevation_deg=sun_elevation_deg,
        sun_azimuth_deg=sun_azimuth_deg,
        pixel_scale_m=gsd_m,
        min_area_pixels=20,
        morphology=True,
        filter_by_sun_direction=True,
    )

    result = calibrate(rel_raster, dem_raster)
    metric = result.apply(relative)

    dem_arr = np.asarray(dem_raster.array, dtype=np.float64)
    if dem_arr.ndim == 3:
        dem_arr = dem_arr[0]
    if dem_arr.shape[-2:] != relative.shape:
        from backend.core.calibration.dem_source import reproject_to_match

        dem_raster = reproject_to_match(dem_raster, rel_raster)
        dem_arr = np.asarray(dem_raster.array, dtype=np.float64)
        if dem_arr.ndim == 3:
            dem_arr = dem_arr[0]
        result = calibrate(rel_raster, dem_raster)
        metric = result.apply(relative)

    source_preds: dict[str, np.ndarray] = {"dem": np.asarray(metric, dtype=np.float64)}
    if len(gcps) >= 4 and dem_arr.shape[-2:] == relative.shape:
        rows = np.array([min(max(int(round(g.row)), 0), h - 1) for g in gcps])
        cols = np.array([min(max(int(round(g.col)), 0), w - 1) for g in gcps])
        xs = relative[rows, cols]
        ground = float(np.nanpercentile(dem_arr[np.isfinite(dem_arr)], 15.0)) if np.isfinite(dem_arr).any() else 0.0
        ys = np.array([g.height_m + ground for g in gcps], dtype=np.float64)
        ww = np.array([g.confidence for g in gcps], dtype=np.float64)
        gcp_fit = calibrate_from_samples(xs, ys, weights=ww, vertical_datum=result.vertical_datum)
        source_preds["shadow_gcp"] = gcp_fit.apply(relative)
        result = calibrate_from_samples(
            relative,
            dem_arr,
            anchors_rc=[(g.row, g.col, g.confidence) for g in gcps],
            source_predictions=source_preds,
            vertical_datum=result.vertical_datum,
        )
        metric = result.apply(relative)

    seg = _cheap_landcover(image)
    metric = apply_ground_constraint(metric, seg)
    # nDSM-like snap would zero terrain; only apply to residual above local min.
    local_min = float(np.nanpercentile(metric[np.isfinite(metric)], 5.0)) if np.isfinite(metric).any() else 0.0
    agl = metric - local_min
    agl = hgdnet_snap_to_ground(agl, seg)
    metric = agl + local_min

    if result.uncertainty is None:
        from backend.core.calibration.fusion import _uncertainty_map

        result.uncertainty = _uncertainty_map(relative, result.residual_std_m, [(g.row, g.col, g.confidence) for g in gcps])

    uncertainty = np.asarray(result.uncertainty, dtype=np.float32)
    color = colorize(metric)
    rel_color = colorize(relative)
    shade = hillshade(metric, cellsize=gsd_m)
    unit_u = to_unit_interval(uncertainty)
    maps = {
        "relative_color_b64": encode_png_b64(rel_color),
        "calibrated_color_b64": encode_png_b64(color),
        "hillshade_b64": encode_png_b64(shade),
        "uncertainty_b64": encode_png_b64(colorize(uncertainty)),
        "overlay_b64": encode_png_b64(_overlay(image, color, alpha=0.5)),
        "relative_gray_b64": encode_png_b64((to_unit_interval(relative) * 255.0).astype(np.uint8)),
        "calibrated_gray_b64": encode_png_b64((to_unit_interval(metric) * 255.0).astype(np.uint8)),
        "uncertainty_gray_b64": encode_png_b64((unit_u * 255.0).astype(np.uint8)),
    }
    calibrated_hex = encode_png_hex((to_unit_interval(metric) * 255.0).astype(np.uint8))

    dem_arr = np.asarray(dem_raster.array, dtype=np.float64)
    if dem_arr.ndim == 3:
        dem_arr = dem_arr[0]
    metrics = None
    try:
        vm = compute_validation_metrics(metric, dem_arr)
        metrics = {
            "rmse_m": vm.rmse,
            "mae_m": vm.mae,
            "r_squared": vm.r_squared,
            "correlation": vm.correlation,
            "bias_m": vm.bias,
            "std_error_m": vm.std_error,
            "n": float(vm.num_valid_samples),
        }
    except ValueError:
        metrics = None

    return CalibrationMaps(
        relative_depth=relative,
        metric_dsm=np.asarray(metric, dtype=np.float64),
        uncertainty=uncertainty,
        result=result,
        gcps=gcps,
        used_dem_fallback=used_fallback,
        dem_source="OFFLINE_SCENE_PRIOR" if used_fallback else dem_source,
        maps_b64=maps,
        calibrated_png_hex=calibrated_hex,
        metrics=metrics,
    )

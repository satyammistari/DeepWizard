"""DEMs: fetch, cache, and resample helpers used by calibration.

This module provides a compact, robust implementation used by unit tests.
It intentionally keeps behavior minimal: cache-first fetch, raises
`DEMUnavailable` on cache miss when network or API key is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import hashlib
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from backend.config import settings
from backend.core.geo.io import RasterData, read_raster

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

DemSource = Literal["SRTMGL1", "COP30", "CARTODEM"]
BoundsWGS84 = Tuple[float, float, float, float]


class DEMUnavailable(RuntimeError):
    pass


def _cache_key(source: DemSource, bounds: BoundsWGS84) -> str:
    west, south, east, north = bounds
    raw = f"{source}:{west:.6f}:{south:.6f}:{east:.6f}:{north:.6f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _cache_path(cache_dir: Path, source: DemSource, bounds: BoundsWGS84) -> Path:
    return Path(cache_dir) / source / f"{_cache_key(source, bounds)}.tif"


def fetch_dem(bounds_wgs84: BoundsWGS84, source: DemSource, cache_dir: Path):
    """Return (array, profile) for a DEM window.

    If a cached file exists under `cache_dir` it is returned. On cache miss,
    this will attempt a network download using OpenTopography if
    `settings.allow_network` is True and `settings.opentopography_api_key` is set;
    otherwise raises `DEMUnavailable`.
    """
    dest = _cache_path(cache_dir, source, bounds_wgs84)
    if dest.exists():
        raster = read_raster(dest, mode="absolute")
        return raster.array, _profile_from_raster(raster)

    # CARTODEM is expected to be provided as a directory of tiles local to the
    # project; mosaic matching the requested bounds and cache the result.
    if source == "CARTODEM":
        carto_dir = Path(settings.cartodem_dir)
        if not carto_dir.exists():
            raise DEMUnavailable("CARTODEM requested but local CARTODEM directory not found")
        tifs = sorted(carto_dir.glob("*.tif")) + sorted(carto_dir.glob("*.tiff"))
        if not tifs:
            raise DEMUnavailable("No CARTODEM tiles found in cartodem_dir")

        datasets = []
        try:
            from rasterio import open as rio_open

            for p in tifs:
                datasets.append(rio_open(p))
            crs = datasets[0].crs
            if crs is None:
                raise DEMUnavailable("CARTODEM tile has no CRS")
            west, south, east, north = bounds_wgs84
            left, bottom, right, top = transform_bounds(CRS.from_epsg(4326), crs, west, south, east, north)
            nodata_fill = -9999.0
            mosaic, transform = merge(datasets, bounds=(left, bottom, right, top), nodata=nodata_fill)
            profile = datasets[0].profile.copy()
            profile.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "nodata": nodata_fill,
            })
            dest.parent.mkdir(parents=True, exist_ok=True)
            from rasterio import open as rio_open_w

            with rio_open_w(dest, "w", **profile) as dst:
                dst.write(mosaic)
        finally:
            for ds in datasets:
                ds.close()
        raster = read_raster(dest, mode="absolute")
        return raster.array, _profile_from_raster(raster)

    # For online sources, require network permission.
    if not settings.allow_network:
        raise DEMUnavailable("Network access is disabled (allow_network=False) and DEM not cached")

    dest.parent.mkdir(parents=True, exist_ok=True)
    api_key = settings.opentopography_api_key
    if api_key:
        west, south, east, north = bounds_wgs84
        demtype = "SRTMGL1" if source == "SRTMGL1" else "COP30"
        query = urllib.parse.urlencode({
            "demtype": demtype,
            "south": f"{south:.7f}",
            "north": f"{north:.7f}",
            "west": f"{west:.7f}",
            "east": f"{east:.7f}",
            "outputFormat": "GTiff",
            "API_Key": api_key,
        })
        url = f"{OPENTOPO_URL}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = resp.read()
        except urllib.error.URLError as exc:
            raise DEMUnavailable(f"Network error while fetching DEM: {exc}") from exc
        if len(payload) < 100:
            raise DEMUnavailable(f"DEM fetch returned empty body for {source}")
        dest.write_bytes(payload)
    elif source in ("SRTMGL1", "COP30"):
        _fetch_via_elevation(bounds_wgs84, dest)
    else:
        raise DEMUnavailable("OpenTopography API key not configured and elevation clip is SRTM-only")

    raster = read_raster(dest, mode="absolute")
    return raster.array, _profile_from_raster(raster)


def _fetch_via_elevation(bounds_wgs84: BoundsWGS84, dest: Path) -> None:
    """Clip SRTM1 via the `elevation` library (NASA SRTM, EGM96 orthometric metres).

    Prep-time / allow_network only. The packaged app must already have this cache.
    """
    try:
        import elevation as elevation_lib
    except ImportError as exc:
        raise DEMUnavailable(
            "Python package 'elevation' is not installed. "
            "pip install elevation, or set DW_OPENTOPOGRAPHY_API_KEY."
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bounds_wgs84
    pad = 0.01
    try:
        elevation_lib.clip(
            bounds=(west - pad, south - pad, east + pad, north + pad),
            output=str(dest),
            product="SRTM1",
        )
    except Exception as exc:
        raise DEMUnavailable(f"elevation.clip failed: {exc}") from exc
    if not dest.exists() or dest.stat().st_size < 100:
        raise DEMUnavailable("elevation.clip produced an empty SRTM tile")


def fetch_reference_dem(
    bounds_wgs84: BoundsWGS84,
    source: DemSource = "SRTMGL1",
    cache_dir: Path | str | None = None,
) -> RasterData:
    """Cache-first reference DEM for a WGS84 window.

    Order: local cache → CARTODEM mosaic → OpenTopography (if API key) →
    `elevation.clip` SRTM1. Raises DEMUnavailable if nothing can be loaded.
    Vertical datum: EGM96 for SRTM/CartoDEM, EGM2008 for COP30.
    """
    cache = Path(cache_dir or settings.dem_cache_dir)
    array, profile = fetch_dem(bounds_wgs84, source, cache)
    return raster_from_fetch(array, profile, source)


def reproject_and_resample(dem_array, dem_profile, target_transform, target_crs, target_shape):
    """Reproject `dem_array` (with `dem_profile`) onto target grid.

    Returns (array, profile).
    """
    src_crs = dem_profile.get("crs")
    if src_crs is None:
        raise ValueError("DEM profile has no CRS")

    dst_h, dst_w = target_shape
    dst = np.empty((dst_h, dst_w), dtype=dem_array.dtype)
    reproject(
        source=dem_array,
        destination=dst,
        src_transform=dem_profile["transform"],
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    new_profile = dict(dem_profile)
    new_profile.update({"height": dst_h, "width": dst_w, "transform": target_transform, "crs": target_crs})
    return dst, new_profile


def _profile_from_raster(raster: RasterData) -> dict:
    arr = np.asarray(raster.array)
    height, width = arr.shape[-2], arr.shape[-1]
    profile = dict(raster.metadata or {})
    profile.update(
        {
            "crs": raster.crs,
            "transform": raster.transform,
            "nodata": raster.nodata,
            "height": height,
            "width": width,
            "count": 1 if arr.ndim == 2 else int(arr.shape[0]),
            "vertical_datum": profile.get("vertical_datum", _vertical_datum("SRTMGL1")),
        }
    )
    return profile


def _vertical_datum(source: DemSource) -> str:
    # SRTM/COP30: EGM96 orthometric. CartoDEM (ISRO): EGM96 orthometric over India.
    if source == "COP30":
        return "EGM2008"
    return "EGM96"


def raster_from_fetch(array, profile, source: DemSource = "SRTMGL1") -> RasterData:
    arr = np.asarray(array)
    if arr.ndim == 3:
        arr = arr[0]
    meta = dict(profile or {})
    meta.setdefault("vertical_datum", _vertical_datum(source))
    meta.setdefault("dem_source", source)
    return RasterData(
        array=np.asarray(arr, dtype=np.float32),
        transform=meta.get("transform"),
        crs=meta.get("crs"),
        nodata=meta.get("nodata"),
        metadata=meta,
    )


def offline_scene_dem(
    relative_depth: np.ndarray,
    bounds_wgs84: BoundsWGS84,
    *,
    base_elevation_m: float = 120.0,
    relief_m: float = 80.0,
    nodata: float = -9999.0,
) -> RasterData:
    """Metric DSM prior used when public DEM tiles are unavailable.

    Maps unitless relative depth onto orthometric metres so calibration and the
    UI still run fully offline. This is NOT SRTM/CartoDEM — it is a local
    scene prior. Vertical datum label: EGM96 (orthometric metres).
    """
    rel = np.asarray(relative_depth, dtype=np.float32)
    if rel.ndim == 3:
        rel = rel[0]
    finite = np.isfinite(rel)
    unit = np.zeros_like(rel, dtype=np.float32)
    if finite.any():
        lo = float(rel[finite].min())
        hi = float(rel[finite].max())
        if hi > lo:
            unit[finite] = (rel[finite] - lo) / (hi - lo)
        else:
            unit[finite] = 0.5
    elevation = np.full(rel.shape, np.nan, dtype=np.float32)
    elevation[finite] = np.float32(base_elevation_m + unit[finite] * relief_m)
    west, south, east, north = bounds_wgs84
    height, width = elevation.shape
    transform = from_bounds(west, south, east, north, width, height)
    return RasterData(
        array=elevation,
        transform=transform,
        crs=CRS.from_epsg(4326),
        nodata=nodata,
        metadata={
            "vertical_datum": "EGM96",
            "dem_source": "OFFLINE_SCENE_PRIOR",
            "crs": CRS.from_epsg(4326),
            "transform": transform,
            "nodata": nodata,
            "note": "Offline scene prior; replace with cached SRTM/COP30/CARTODEM for production metric DSM.",
        },
    )


def resolve_dem(
    bounds_wgs84: BoundsWGS84,
    source: DemSource,
    cache_dir,
    *,
    relative_depth: np.ndarray | None = None,
) -> tuple[RasterData, bool]:
    """Load a DEM window, falling back to an offline scene prior.

    Returns (raster, used_fallback).
    """
    try:
        array, profile = fetch_dem(bounds_wgs84, source, cache_dir)
        return raster_from_fetch(array, profile, source), False
    except DEMUnavailable:
        if relative_depth is None:
            raise
        return offline_scene_dem(relative_depth, bounds_wgs84), True


def reproject_to_match(source, target):
    """Resample `source` (RasterData) onto `target` (RasterData) and return RasterData."""
    if getattr(source, "array", None) is None:
        raise ValueError("source must be a RasterData-like object with `array` and `metadata`")
    arr = source.array
    prof = source.metadata if source.metadata else {}
    if "transform" not in prof:
        prof = {**prof, "transform": source.transform, "crs": source.crs}
    tgt_transform = target.transform
    tgt_crs = target.crs
    tgt_shape = (target.array.shape[-2], target.array.shape[-1])
    if tgt_transform is None or tgt_crs is None:
        return source
    out_arr, out_prof = reproject_and_resample(arr, prof, tgt_transform, tgt_crs, tgt_shape)
    return RasterData(array=out_arr, transform=tgt_transform, crs=tgt_crs, nodata=source.nodata, metadata=out_prof)

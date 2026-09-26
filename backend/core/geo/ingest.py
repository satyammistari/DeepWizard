"""Ingest RGB or GeoTIFF imagery, preserving CRS / affine / GSD.

Heights are not invented here. GSD is metres/pixel on the ellipsoid or in the
projected CRS unit (metres for UTM). Nodata is never rewritten as 0 m.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from pyproj import CRS as PyprojCRS
from pyproj import Geod, Transformer
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds

from backend.core.geo.io import RasterData


@dataclass
class IngestedImage:
    """RGB scene plus geospatial metadata copied from the source file."""

    rgb: NDArray[np.uint8]
    transform: Affine | None
    crs: CRS | None
    gsd_m: float | None
    bounds_wgs84: tuple[float, float, float, float] | None
    nodata: float | int | None
    is_georeferenced: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


def _rgb_from_raster_array(array: np.ndarray) -> NDArray[np.uint8]:
    arr = np.asarray(array)
    if arr.ndim == 2:
        gray = arr
        if np.issubdtype(gray.dtype, np.floating):
            finite = np.isfinite(gray)
            scaled = np.zeros_like(gray, dtype=np.float32)
            if finite.any():
                lo = float(gray[finite].min())
                hi = float(gray[finite].max())
                if hi > lo:
                    scaled[finite] = (gray[finite] - lo) / (hi - lo)
            rgb = np.stack([scaled, scaled, scaled], axis=-1)
            return (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        return np.repeat(g8[..., None], 3, axis=-1)
    if arr.ndim == 3:
        # rasterio order is (C, H, W)
        if arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
            bands = arr[:3]
            hw = np.transpose(bands, (1, 2, 0))
        elif arr.shape[-1] in (3, 4):
            hw = arr[..., :3]
        else:
            raise ValueError(f"Unsupported raster layout {arr.shape}")
        if np.issubdtype(hw.dtype, np.floating):
            finite = np.isfinite(hw)
            out = np.zeros_like(hw, dtype=np.float32)
            if finite.any():
                lo = float(hw[finite].min())
                hi = float(hw[finite].max())
                if hi > lo:
                    out[finite] = (hw[finite] - lo) / (hi - lo)
            return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)
        if hw.dtype == np.uint16:
            return (np.clip(hw / 257.0, 0, 255)).astype(np.uint8)
        return np.clip(hw, 0, 255).astype(np.uint8)
    raise ValueError(f"Expected 2D or 3D raster, got {arr.shape}")


def compute_gsd_m(transform: Affine, crs: CRS, height: int, width: int) -> float:
    """Ground sample distance in metres at the raster centre.

    Geographic CRS: geodesic length of one pixel east. Projected CRS: |a| if
    the unit is metre; otherwise convert via pyproj.
    """
    col = width / 2.0
    row = height / 2.0
    x0, y0 = transform * (col, row)
    x1, y1 = transform * (col + 1.0, row)
    pycrs = PyprojCRS.from_user_input(crs)
    if pycrs.is_geographic:
        geod = Geod(ellps="WGS84")
        _, _, dist = geod.inv(x0, y0, x1, y1)
        return float(abs(dist))
    axis = pycrs.axis_info[0] if pycrs.axis_info else None
    unit = (axis.unit_name or "metre").lower() if axis else "metre"
    pixel = float(np.hypot(x1 - x0, y1 - y0))
    if "degree" in unit:
        geod = Geod(ellps="WGS84")
        _, _, dist = geod.inv(x0, y0, x1, y1)
        return float(abs(dist))
    if "foot" in unit or unit in {"ft", "us survey foot"}:
        return pixel * 0.3048
    return pixel


def bounds_to_wgs84(transform: Affine, crs: CRS, height: int, width: int) -> tuple[float, float, float, float]:
    west, south, east, north = array_bounds(height, width, transform)
    left, bottom, right, top = transform_bounds(crs, CRS.from_epsg(4326), west, south, east, north)
    return (float(left), float(bottom), float(right), float(top))


def ingest_geotiff_bytes(payload: bytes) -> IngestedImage | None:
    """Return IngestedImage if payload is a rasterio-readable GeoTIFF; else None."""
    try:
        with MemoryFile(payload) as mem:
            with mem.open() as src:
                if src.count == 1:
                    array = src.read(1)
                else:
                    array = src.read()
                rgb = _rgb_from_raster_array(array)
                crs = src.crs
                transform = src.transform
                h, w = rgb.shape[:2]
                gsd = None
                bounds = None
                georef = crs is not None and transform is not None
                if georef:
                    gsd = compute_gsd_m(transform, crs, h, w)
                    bounds = bounds_to_wgs84(transform, crs, h, w)
                meta = dict(src.tags())
                meta["driver"] = src.driver
                meta["count"] = src.count
                return IngestedImage(
                    rgb=rgb,
                    transform=transform if georef else None,
                    crs=crs,
                    gsd_m=gsd,
                    bounds_wgs84=bounds,
                    nodata=src.nodata,
                    is_georeferenced=bool(georef),
                    metadata=meta,
                )
    except Exception:
        return None


def ingest_rgb_bytes(payload: bytes) -> IngestedImage:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image or GeoTIFF.") from exc
    rgb = np.asarray(image, dtype=np.uint8)
    return IngestedImage(
        rgb=rgb,
        transform=None,
        crs=None,
        gsd_m=None,
        bounds_wgs84=None,
        nodata=None,
        is_georeferenced=False,
        metadata={"driver": "PIL"},
    )


def ingest_image(source: str | Path | bytes) -> IngestedImage:
    """Load a path or byte payload. GeoTIFF metadata is preserved exactly."""
    if isinstance(source, (str, Path)):
        payload = Path(source).read_bytes()
    else:
        payload = source
    geotiff = ingest_geotiff_bytes(payload)
    if geotiff is not None and (geotiff.is_georeferenced or geotiff.metadata.get("driver") == "GTiff"):
        return geotiff
    return ingest_rgb_bytes(payload)


def ingested_to_raster_data(ingested: IngestedImage, array: np.ndarray) -> RasterData:
    """Wrap an array on the ingested grid without rewriting CRS/affine."""
    return RasterData(
        array=array,
        transform=ingested.transform,
        crs=ingested.crs,
        nodata=ingested.nodata,
        metadata=dict(ingested.metadata),
    )

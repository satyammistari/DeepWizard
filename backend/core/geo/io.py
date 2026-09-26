from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds


class MissingCRSError(ValueError):
    """Raised when absolute (metric) mode is requested but the raster has no CRS."""


@dataclass
class RasterData:
    """In-memory raster with geospatial metadata preserved from disk.

    array: (H, W) or (C, H, W). dtype as on disk. Elevation rasters use metres.
    transform: affine mapping pixel to CRS units; must not be rewritten.
    crs: rasterio CRS, or None only in relative mode.
    nodata: sentinel to mask; never treat as 0 m elevation.
    """

    array: np.ndarray
    transform: Affine | None = None
    crs: CRS | str | None = None
    nodata: float | int | None = None
    # Optional tags (vertical datum, DEM source, …). Never a substitute for crs.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> BoundingBox:
        if self.transform is None:
            raise ValueError("RasterData.bounds requires an affine transform")
        height, width = self.array.shape[-2], self.array.shape[-1]
        west, south, east, north = array_bounds(height, width, self.transform)
        return BoundingBox(west, south, east, north)


def read_raster(
    path: str | Path,
    *,
    mode: Literal["relative", "absolute"] = "relative",
) -> RasterData:
    """Open a GeoTIFF (or other rasterio-supported raster) without reprojection.

    Absolute mode requires a defined CRS; relative mode may proceed without one.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is None and mode == "absolute":
            raise MissingCRSError(
                f"Raster '{path}' has no CRS. Absolute (metric) DSM mode requires "
                "a georeferenced GeoTIFF with a defined CRS; assign or recover CRS "
                "before ingest. Refusing to invent coordinates."
            )
        count = src.count
        if count == 1:
            array = src.read(1)
        else:
            array = src.read()
        return RasterData(
            array=array,
            transform=src.transform,
            crs=crs,
            nodata=src.nodata,
        )


def write_raster(
    path: str | Path,
    array: np.ndarray,
    profile: dict[str, Any],
    nodata: float | int | None,
) -> None:
    """Write a raster, copying CRS and affine transform from profile exactly.

    array: (H, W) or (C, H, W). dtype written as-is.
    nodata: stored in the GeoTIFF; callers must keep nodata masked, not 0-filled.
    """
    path = Path(path)
    out_profile = dict(profile)

    if out_profile.get("crs") is None:
        raise MissingCRSError(
            f"Refusing to write '{path}': profile has no CRS. "
            "CRS from the source raster must be preserved exactly."
        )
    if out_profile.get("transform") is None:
        raise ValueError(
            f"Refusing to write '{path}': profile has no affine transform. "
            "The geotransform must be preserved exactly."
        )

    if array.ndim == 2:
        data = array[np.newaxis, ...]
    elif array.ndim == 3:
        data = array
    else:
        raise ValueError(f"Expected array of shape (H, W) or (C, H, W), got {array.shape}")

    count, height, width = data.shape
    # Update only layout/dtype/nodata. Do not touch crs or transform.
    out_profile.update(
        {
            "driver": out_profile.get("driver", "GTiff"),
            "height": height,
            "width": width,
            "count": count,
            "dtype": data.dtype,
            "nodata": nodata,
        }
    )

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data)

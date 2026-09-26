"""Cloud-Optimized GeoTIFF export. Band 1 = height (m), Band 2 = uncertainty (m).

CRS and affine are copied from the source grid. Vertical datum is stored as a
TIFF tag / GDAL metadata (EGM2008 geoid unless the caller documents otherwise).
Nodata stays a sentinel, never 0 m.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from backend.core.geo.io import MissingCRSError


def export_cog(
    path: str | Path,
    height_m: np.ndarray,
    uncertainty_m: np.ndarray,
    *,
    transform: Affine,
    crs: CRS | str,
    nodata: float = -9999.0,
    vertical_datum: str = "EGM2008",
) -> Path:
    """Write a 2-band COG: height (metres) + 1-sigma uncertainty (metres)."""
    dest = Path(path)
    if crs is None:
        raise MissingCRSError("export_cog requires a CRS; refusing to invent coordinates")
    if transform is None:
        raise ValueError("export_cog requires an affine transform copied from the source grid")

    z = np.asarray(height_m, dtype=np.float32)
    u = np.asarray(uncertainty_m, dtype=np.float32)
    if z.shape != u.shape:
        raise ValueError(f"height {z.shape} and uncertainty {u.shape} differ")
    if z.ndim != 2:
        raise ValueError(f"Expected (H, W) height, got {z.shape}")

    z_out = np.where(np.isfinite(z), z, nodata).astype(np.float32)
    u_out = np.where(np.isfinite(u), u, nodata).astype(np.float32)
    stacked = np.stack([z_out, u_out], axis=0)

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": z.shape[0],
        "width": z.shape[1],
        "count": 2,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 3,
        "interleave": "band",
        "BIGTIFF": "IF_SAFER",
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.tif")
    import rasterio

    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(stacked)
        dst.set_band_description(1, "height_m")
        dst.set_band_description(2, "uncertainty_m_1sigma")
        dst.update_tags(
            VERTICAL_DATUM=vertical_datum,
            AREA_OR_POINT="Area",
            TIFFTAG_IMAGEDESCRIPTION="DepthWizard metric DSM + uncertainty",
        )
        dst.update_tags(1, UNITS="metre")
        dst.update_tags(2, UNITS="metre")

    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        cog_translate(
            str(tmp),
            str(dest),
            cog_profiles.get("deflate"),
            in_memory=False,
            quiet=True,
            additional_cog_metadata={
                "VERTICAL_DATUM": vertical_datum,
            },
        )
        tmp.unlink(missing_ok=True)
    except Exception:
        dest.write_bytes(tmp.read_bytes())
        tmp.unlink(missing_ok=True)
    return dest


def export_cog_bytes(
    height_m: np.ndarray,
    uncertainty_m: np.ndarray,
    *,
    transform: Affine,
    crs: CRS | str,
    nodata: float = -9999.0,
    vertical_datum: str = "EGM2008",
) -> bytes:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dsm.tif"
        export_cog(
            path,
            height_m,
            uncertainty_m,
            transform=transform,
            crs=crs,
            nodata=nodata,
            vertical_datum=vertical_datum,
        )
        return path.read_bytes()

from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from backend.core.geo.io import MissingCRSError, read_raster, write_raster


def _utm43n_profile(height: int, width: int, dtype: str) -> dict:
    # UTM 43N is typical for western/central India; WKT must round-trip unchanged.
    crs = CRS.from_epsg(32643)
    transform = from_origin(500000.0, 2_100_000.0, 10.0, 10.0)
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }


def test_write_read_geotiff_preserves_crs_and_transform_bytes(tmp_path: Path) -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    nodata = np.float32(-9999.0)
    profile = _utm43n_profile(3, 4, "float32")
    expected_wkt = CRS.from_user_input(profile["crs"]).to_wkt().encode("utf-8")
    expected_transform_bytes = bytes(np.asarray(list(profile["transform"]), dtype=np.float64))

    path = tmp_path / "synthetic.tif"
    write_raster(path, array, profile, nodata=float(nodata))
    loaded = read_raster(path, mode="absolute")

    assert loaded.crs is not None
    assert loaded.crs.to_wkt().encode("utf-8") == expected_wkt
    assert bytes(np.asarray(list(loaded.transform), dtype=np.float64)) == expected_transform_bytes
    assert loaded.transform == profile["transform"]
    np.testing.assert_array_equal(loaded.array, array)
    assert loaded.nodata == nodata
    assert loaded.bounds.left == pytest.approx(500000.0)
    assert loaded.bounds.top == pytest.approx(2_100_000.0)


def test_absolute_mode_rejects_missing_crs(tmp_path: Path) -> None:
    path = tmp_path / "no_crs.tif"
    array = np.ones((2, 2), dtype=np.float32)
    import rasterio
    from rasterio.transform import Affine

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs=None,
        transform=Affine.identity(),
    ) as dst:
        dst.write(array, 1)

    with pytest.raises(MissingCRSError, match="no CRS"):
        read_raster(path, mode="absolute")

    relative = read_raster(path, mode="relative")
    assert relative.crs is None

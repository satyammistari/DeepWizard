import pytest
from pathlib import Path

import numpy as np

from backend.core.calibration.dem_source import fetch_dem, reproject_and_resample, DEMUnavailable
from backend.config import settings


def test_fetch_dem_sanity(tmp_path: Path):
    # Bounding box over central Bengaluru (approx)
    bounds = (77.55, 12.90, 77.70, 13.02)  # west,south,east,north
    cache_dir = tmp_path / "dem_cache"

    if not settings.allow_network or not settings.opentopography_api_key:
        pytest.skip("No network or API key configured for DEM fetch test")

    arr, prof = fetch_dem(bounds, "SRTMGL1", cache_dir)
    assert isinstance(arr, np.ndarray)
    assert arr.size > 0
    # sanity: elevation within reasonable bounds
    assert float(arr.min()) > -500 and float(arr.max()) < 5000

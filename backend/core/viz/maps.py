"""Map rendering helpers. Arrays are metres or unitless; nodata stays masked."""

from __future__ import annotations

import base64
import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def to_unit_interval(array: NDArray, *, nodata: float | int | None = None) -> NDArray[np.float32]:
    """Stretch finite pixels to [0, 1]. Nodata / non-finite become 0 in the preview only."""
    data = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        valid &= data != nodata
    out = np.zeros_like(data, dtype=np.float32)
    if not valid.any():
        return out
    lo = float(data[valid].min())
    hi = float(data[valid].max())
    if hi <= lo:
        out[valid] = 0.5
        return out
    out[valid] = (data[valid] - lo) / (hi - lo)
    return out


def _turbo_lut() -> NDArray[np.uint8]:
    """Turbo-like LUT (Google) sampled at 256 stops — suitable for elevation previews."""
    stops = np.array(
        [
            [48, 18, 59],
            [70, 107, 227],
            [40, 188, 214],
            [69, 218, 116],
            [192, 211, 52],
            [253, 159, 39],
            [239, 71, 25],
            [122, 4, 3],
        ],
        dtype=np.float32,
    )
    x = np.linspace(0.0, 1.0, len(stops))
    t = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(t, x, stops[:, c]) for c in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)


_TURBO = _turbo_lut()


def colorize(array: NDArray, *, nodata: float | int | None = None) -> NDArray[np.uint8]:
    """Return HxWx3 uint8 Turbo coloring of a scalar field."""
    unit = to_unit_interval(array, nodata=nodata)
    idx = np.clip((unit * 255.0).astype(np.int32), 0, 255)
    rgb = _TURBO[idx]
    valid = np.isfinite(np.asarray(array, dtype=np.float32))
    rgb[~valid] = 0
    return rgb


def hillshade(
    elevation_m: NDArray,
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    cellsize: float = 1.0,
) -> NDArray[np.uint8]:
    """Illumination of a metric surface. elevation_m: (H, W) metres. nodata stays black."""
    z = np.asarray(elevation_m, dtype=np.float64)
    valid = np.isfinite(z)
    fill = float(np.nanmedian(z[valid])) if valid.any() else 0.0
    filled = np.where(valid, z, fill)
    dy, dx = np.gradient(filled, cellsize, cellsize)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0.0, 1.0)
    out = (shade * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def encode_png(array: NDArray) -> bytes:
    """Encode HxW grayscale or HxWx3 RGB as PNG bytes."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Expected (H, W) or (H, W, 3), got {arr.shape}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_png_b64(array: NDArray) -> str:
    return base64.b64encode(encode_png(array)).decode("ascii")


def encode_png_hex(array: NDArray) -> str:
    return encode_png(array).hex()

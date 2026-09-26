from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from backend.core.depth.backbone import infer_relative
from backend.core.depth.tiling import split_into_tiles
from backend.core.depth.stitching import align_and_stitch


def _unpad_from_crop(depth_crop: NDArray, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int) -> NDArray:
    h, w = depth_crop.shape[:2]
    r0 = pad_top
    r1 = h - pad_bottom
    c0 = pad_left
    c1 = w - pad_right
    return depth_crop[r0:r1, c0:c1]


def infer_tiles(
    image: NDArray,
    *,
    tile_size: int = 518,
    overlap_frac: float = 0.25,
    variant: str = "small",
) -> tuple[list, list]:
    """Infer relative depth per tile and return (tiles, depths_unpadded).

    depths_unpadded are float64 arrays matching each tile's parent valid bounds.
    """
    tiles = split_into_tiles(image, tile_size=tile_size, overlap_frac=overlap_frac)
    depths = []
    for t in tiles:
        crop = t.crop
        # ensure HxW or HxWxC
        if crop.ndim == 3 and crop.shape[2] == 1:
            crop_in = crop[:, :, 0]
        else:
            crop_in = crop
        # infer_relative expects uint8 RGB or grayscale; cast if needed
        try:
            depth_crop = infer_relative(np.asarray(crop_in), variant=variant)
        except Exception:
            # fallback: simple luminance
            arr = np.asarray(crop_in)
            if arr.ndim == 3:
                lum = arr.mean(axis=-1)
            else:
                lum = arr
            depth_crop = (lum - lum.min()) / max(1.0, (lum.max() - lum.min()))
        depth_crop = np.asarray(depth_crop, dtype=np.float64)
        unpadded = _unpad_from_crop(depth_crop, t.pad_top, t.pad_bottom, t.pad_left, t.pad_right)
        depths.append(unpadded)
    return tiles, depths


def infer_and_stitch(
    image: NDArray,
    *,
    tile_size: int = 518,
    overlap_frac: float = 0.25,
    variant: str = "small",
    nodata: float | int | None = None,
) -> NDArray:
    tiles, depths = infer_tiles(image, tile_size=tile_size, overlap_frac=overlap_frac, variant=variant)
    out = align_and_stitch(tiles, depths, (image.shape[0], image.shape[1]), nodata=nodata)
    return out

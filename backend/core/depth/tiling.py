from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Tile:
    """One overlapping crop of a parent image.

    Fields:
        crop: padded array of spatial size (tile_size, tile_size, ...) with
            reflection padding applied on edges.
        grid_row, grid_col: tile indices on the regular grid.
        row_start/row_end, col_start/col_end: half-open pixel bounds in the
            parent that the unpadded crop occupies.
        pad_top/pad_bottom/pad_left/pad_right: amounts of reflection pad.
        index: linear index assigned by split_into_tiles (row-major).
    """

    crop: NDArray
    grid_row: int
    grid_col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    index: int = 0


def _reflect_pad(array: NDArray, pad_width: tuple) -> NDArray:
    """Reflection pad; fall back to symmetric when reflection is not possible."""
    try:
        return np.pad(array, pad_width, mode="reflect")
    except ValueError:
        return np.pad(array, pad_width, mode="symmetric")


def _pad_width(array: NDArray, top: int, bottom: int, left: int, right: int) -> tuple:
    spatial = ((top, bottom), (left, right))
    extra = tuple((0, 0) for _ in range(max(0, array.ndim - 2)))
    return spatial + extra


def _axis_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= 0:
        raise ValueError(f"Image axis length must be positive, got {length}")
    starts: List[int] = []
    pos = 0
    while True:
        starts.append(pos)
        if pos + tile_size >= length:
            break
        pos += stride
    return starts


def unpad_tile(tile: Tile) -> NDArray:
    """Strip reflection pad so the result matches parent[row_start:row_end, col_start:col_end]."""
    h, w = tile.crop.shape[:2]
    r0, r1 = tile.pad_top, h - tile.pad_bottom
    c0, c1 = tile.pad_left, w - tile.pad_right
    if r1 <= r0 or c1 <= c0:
        raise ValueError(
            f"Tile {tile.index} pad removes the entire crop: pad="
            f"({tile.pad_top},{tile.pad_bottom},{tile.pad_left},{tile.pad_right}) "
            f"crop_hw=({h},{w})"
        )
    return tile.crop[r0:r1, c0:c1, ...]


def split_into_tiles(
    image: NDArray,
    tile_size: int = 518,
    overlap_frac: float = 0.25,
) -> List[Tile]:
    """Cut `image` into a regular overlapping grid, padding edge tiles by reflection.

    image: (H, W) or (H, W, C). dtype unchanged. No nodata filling.
    tile_size: square window fed to the depth model (pixels).
    overlap_frac: in [0, 1). Stride = round(tile_size * (1 - overlap_frac)).
    """
    if image.ndim < 2:
        raise ValueError(f"Expected (H, W) or (H, W, C), got shape {image.shape}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if not 0.0 <= overlap_frac < 1.0:
        raise ValueError(f"overlap_frac must be in [0, 1), got {overlap_frac}")

    height, width = int(image.shape[0]), int(image.shape[1])
    stride = max(1, int(round(tile_size * (1.0 - overlap_frac))))
    row_starts = _axis_starts(height, tile_size, stride)
    col_starts = _axis_starts(width, tile_size, stride)

    tiles: List[Tile] = []
    index = 0
    for grid_row, r0 in enumerate(row_starts):
        r1 = min(r0 + tile_size, height)
        pad_top = 0
        pad_bottom = tile_size - (r1 - r0)
        for grid_col, c0 in enumerate(col_starts):
            c1 = min(c0 + tile_size, width)
            pad_left = 0
            pad_right = tile_size - (c1 - c0)
            crop = np.array(image[r0:r1, c0:c1, ...], copy=True)
            if pad_top or pad_bottom or pad_left or pad_right:
                crop = _reflect_pad(
                    crop,
                    _pad_width(crop, pad_top, pad_bottom, pad_left, pad_right),
                )
            tiles.append(
                Tile(
                    crop=crop,
                    grid_row=grid_row,
                    grid_col=grid_col,
                    row_start=r0,
                    row_end=r1,
                    col_start=c0,
                    col_end=c1,
                    pad_top=pad_top,
                    pad_bottom=pad_bottom,
                    pad_left=pad_left,
                    pad_right=pad_right,
                    index=index,
                )
            )
            index += 1
    return tiles

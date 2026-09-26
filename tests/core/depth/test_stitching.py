import numpy as np

from backend.core.depth.tiling import split_into_tiles, unpad_tile
from backend.core.depth.stitching import (
    depths_from_tiles,
    collect_pairwise_constraints,
    solve_global_alignment,
    align_and_stitch,
)


def _smooth_field(h: int, w: int) -> np.ndarray:
    y = np.linspace(0, 2 * np.pi, h)
    x = np.linspace(0, 2 * np.pi, w)
    xx, yy = np.meshgrid(x, y)
    field = (np.sin(xx) + np.cos(yy)) * 10.0 + 50.0
    # add a low-frequency hill
    field += 5.0 * np.exp(-((xx - 3.0) ** 2 + (yy - 3.0) ** 2))
    return field


def test_global_alignment_recovers_affine_on_smooth_field():
    h, w = 800, 900
    field = _smooth_field(h, w)
    tiles = split_into_tiles(field, tile_size=256, overlap_frac=0.25)
    depths = depths_from_tiles(tiles)

    rng = np.random.default_rng(42)
    scales = rng.uniform(0.8, 1.25, size=len(tiles))
    shifts = rng.uniform(-2.0, 2.0, size=len(tiles))

    transformed = [scales[i] * depths[i] + shifts[i] for i in range(len(tiles))]

    # collect constraints and solve
    constraints = collect_pairwise_constraints(tiles, transformed)
    assert len(constraints) > 0
    transforms = solve_global_alignment(tiles, constraints)

    # build stitched result
    out = align_and_stitch(tiles, transformed, (h, w))

    # compare to original field on finite pixels
    mask = np.isfinite(out)
    corr = np.corrcoef(field[mask].ravel(), out[mask].ravel())[0, 1]
    assert corr > 0.999, f"Correlation too low: {corr}"


def test_no_large_seam_gradients():
    h, w = 800, 900
    field = _smooth_field(h, w)
    tiles = split_into_tiles(field, tile_size=256, overlap_frac=0.25)
    depths = depths_from_tiles(tiles)

    rng = np.random.default_rng(123)
    scales = rng.uniform(0.9, 1.1, size=len(tiles))
    shifts = rng.uniform(-1.0, 1.0, size=len(tiles))
    transformed = [scales[i] * depths[i] + shifts[i] for i in range(len(tiles))]

    stitched = align_and_stitch(tiles, transformed, (h, w))

    gy, gx = np.gradient(stitched)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    valid = np.isfinite(grad)
    mean = float(np.nanmean(grad[valid]))
    std = float(np.nanstd(grad[valid]))

    # find tile boundary rows/cols
    row_starts = sorted(set(t.row_start for t in tiles))
    col_starts = sorted(set(t.col_start for t in tiles))

    for r in row_starts:
        if r <= 0 or r >= h:
            continue
        row_vals = grad[r, :]
        if not np.any(np.isfinite(row_vals)):
            continue
        assert float(np.nanmax(row_vals)) <= mean + 3.0 * std

    for c in col_starts:
        if c <= 0 or c >= w:
            continue
        col_vals = grad[:, c]
        if not np.any(np.isfinite(col_vals)):
            continue
        assert float(np.nanmax(col_vals)) <= mean + 3.0 * std

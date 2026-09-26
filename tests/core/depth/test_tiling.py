import numpy as np

from backend.core.depth.tiling import split_into_tiles, unpad_tile


def test_split_into_tiles_covers_image_without_gaps():
    h, w = 1500, 2000
    img = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    tile_size = 518
    overlap = 0.25
    tiles = split_into_tiles(img, tile_size=tile_size, overlap_frac=overlap)

    coverage = np.zeros((h, w), dtype=bool)
    for t in tiles:
        # unpad to parent bounds and mark coverage
        crop = unpad_tile(t)
        rs, re, cs, ce = t.row_start, t.row_end, t.col_start, t.col_end
        coverage[rs:re, cs:ce] = True

    assert coverage.all(), "Tiles do not fully cover the parent image"

    # Check overlap: adjacent tiles should overlap by approximately tile_size*overlap
    stride = int(tile_size * (1.0 - overlap))
    # compute unique row starts
    row_starts = sorted(set(t.row_start for t in tiles))
    col_starts = sorted(set(t.col_start for t in tiles))
    assert len(row_starts) > 1 and len(col_starts) > 1
    # consecutive starts should differ by stride (except when clamped at edges)
    diffs = [j - i for i, j in zip(row_starts, row_starts[1:])]
    assert any(d == stride for d in diffs), "Tile row stride not present"

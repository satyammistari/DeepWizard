"""Global affine alignment of tiled inverse-depth, then Hann-weighted mosaic.

Each monocular tile is only defined up to an unknown affine (scale, shift).
Pairwise overlaps constrain relative affines; a gauge-fixed sparse least-squares
solve recovers one (scale, shift) per tile. Nodata is never written as 0 m —
uncovered pixels stay NaN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import lsqr

from backend.core.depth.tiling import Tile, unpad_tile

logger = logging.getLogger(__name__)

_MIN_OVERLAP = 100
_VAR_EPS = 1e-8
_WEIGHT_EPS = 1e-6
_LOG_A_EPS = 1e-12


@dataclass(frozen=True)
class PairwiseConstraint:
    """depth_i ≈ scale * depth_j + shift on the overlap of tiles i and j."""

    i: int
    j: int
    scale: float
    shift: float
    residual: float


def _valid_overlap_mask(
    depth_a: NDArray,
    depth_b: NDArray,
    overlap_mask: NDArray,
    nodata: float | int | None,
) -> NDArray:
    mask = np.asarray(overlap_mask, dtype=bool)
    mask &= np.isfinite(depth_a) & np.isfinite(depth_b)
    if nodata is not None and np.isfinite(nodata):
        mask &= depth_a != nodata
        mask &= depth_b != nodata
    return mask


def estimate_pairwise_affine(
    depth_a: NDArray,
    depth_b: NDArray,
    overlap_mask: NDArray,
    *,
    nodata: float | int | None = None,
    min_pixels: int = _MIN_OVERLAP,
    var_eps: float = _VAR_EPS,
) -> tuple[float, float, float] | None:
    """Least-squares fit of depth_a ≈ a * depth_b + b on the overlap.

    depth_a, depth_b: (H, W) same shape, model units (inverse-depth or relative
        height). nodata must not be treated as 0 m.
    overlap_mask: bool (H, W).
    Returns (a, b, residual_rms) or None when the overlap is too small or
    nearly constant (no scale information — including it would destabilise
    the global solve).
    """
    if depth_a.shape != depth_b.shape or depth_a.shape != overlap_mask.shape:
        raise ValueError(
            "depth_a, depth_b, and overlap_mask must share shape, got "
            f"{depth_a.shape}, {depth_b.shape}, {overlap_mask.shape}"
        )
    mask = _valid_overlap_mask(depth_a, depth_b, overlap_mask, nodata)
    n = int(mask.sum())
    if n < min_pixels:
        return None

    ya = np.asarray(depth_a[mask], dtype=np.float64).ravel()
    xb = np.asarray(depth_b[mask], dtype=np.float64).ravel()
    if float(np.var(ya)) < var_eps or float(np.var(xb)) < var_eps:
        return None

    # [x, 1] @ [a, b]^T = y
    a_mat = np.column_stack((xb, np.ones(n, dtype=np.float64)))
    try:
        coeff, *_ = np.linalg.lstsq(a_mat, ya, rcond=None)
    except np.linalg.LinAlgError:
        return None
    scale, shift = float(coeff[0]), float(coeff[1])
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0.0:
        return None
    resid = ya - (scale * xb + shift)
    residual = float(np.sqrt(np.mean(resid * resid)))
    if not np.isfinite(residual):
        return None
    return scale, shift, residual


def _overlap_bounds(a: Tile, b: Tile) -> tuple[int, int, int, int] | None:
    r0 = max(a.row_start, b.row_start)
    r1 = min(a.row_end, b.row_end)
    c0 = max(a.col_start, b.col_start)
    c1 = min(a.col_end, b.col_end)
    if r1 - r0 <= 0 or c1 - c0 <= 0:
        return None
    return r0, r1, c0, c1


def _slice_parent(tile: Tile, depth_valid: NDArray, r0: int, r1: int, c0: int, c1: int) -> NDArray:
    return depth_valid[r0 - tile.row_start : r1 - tile.row_start, c0 - tile.col_start : c1 - tile.col_start]


def collect_pairwise_constraints(
    tiles: Sequence[Tile],
    depths: Sequence[NDArray],
    *,
    nodata: float | int | None = None,
) -> list[PairwiseConstraint]:
    """Every overlapping pair with a well-posed affine (not only 4-neighbours)."""
    if len(tiles) != len(depths):
        raise ValueError("tiles and depths must have the same length")
    constraints: list[PairwiseConstraint] = []
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            bounds = _overlap_bounds(tiles[i], tiles[j])
            if bounds is None:
                continue
            r0, r1, c0, c1 = bounds
            da = _slice_parent(tiles[i], depths[i], r0, r1, c0, c1)
            db = _slice_parent(tiles[j], depths[j], r0, r1, c0, c1)
            overlap = np.ones(da.shape, dtype=bool)
            fit = estimate_pairwise_affine(da, db, overlap, nodata=nodata)
            if fit is None:
                continue
            scale, shift, residual = fit
            constraints.append(
                PairwiseConstraint(i=i, j=j, scale=scale, shift=shift, residual=residual)
            )
    return constraints


def solve_global_alignment(
    tiles: Sequence[Tile],
    pairwise_constraints: Sequence[PairwiseConstraint],
) -> dict[int, tuple[float, float]]:
    """Recover per-tile (scale, shift) with tile 0 pinned to (1, 0).

    Unknowns are log_scale_i and shift_i. Pairwise depth_i ≈ a * depth_j + b
    linearises to:
        log_scale_j - log_scale_i = log(a)
        shift_j - shift_i = scale_i * b
    Stage 1 solves log-scales; stage 2 solves shifts with those scales frozen.
    Without pinning tile 0 the system is rank-deficient (global gauge).
    """
    n_tiles = len(tiles)
    if n_tiles == 0:
        raise ValueError("solve_global_alignment requires at least one tile")

    transforms: dict[int, tuple[float, float]] = {0: (1.0, 0.0)}
    for k in range(1, n_tiles):
        transforms[k] = (1.0, 0.0)

    if n_tiles == 1:
        logger.info(
            "global alignment: n_constraints=0 n_tiles=1 condition=n/a residual=0 (single tile, identity gauge)"
        )
        return transforms

    usable = [c for c in pairwise_constraints if c.scale > 0.0 and np.isfinite(c.residual)]
    if not usable:
        raise ValueError(
            "No usable pairwise overlap constraints; cannot align tiles. "
            "Increase overlap_frac or check nodata / flat overlaps."
        )

    n_unknown = n_tiles - 1  # tile 0 gauge-fixed at log_scale=0
    n_eq = len(usable)
    a_scale = lil_matrix((n_eq, n_unknown), dtype=np.float64)
    b_scale = np.zeros(n_eq, dtype=np.float64)

    for row, c in enumerate(usable):
        w = 1.0 / (c.residual + _WEIGHT_EPS)
        log_a = float(np.log(max(c.scale, _LOG_A_EPS)))
        # log_s_j - log_s_i = log(a); unknown index is tile-1 (tile 0 omitted)
        if c.j != 0:
            a_scale[row, c.j - 1] = w
        if c.i != 0:
            a_scale[row, c.i - 1] = -w
        b_scale[row] = w * log_a

    a_scale_csr: csr_matrix = a_scale.tocsr()
    scale_sol = lsqr(a_scale_csr, b_scale)
    log_s = np.zeros(n_tiles, dtype=np.float64)
    log_s[1:] = scale_sol[0]
    scales = np.exp(log_s)
    acond_scale = float(scale_sol[5])
    r1_scale = float(scale_sol[2])

    a_shift = lil_matrix((n_eq, n_unknown), dtype=np.float64)
    b_shift = np.zeros(n_eq, dtype=np.float64)
    for row, c in enumerate(usable):
        w = 1.0 / (c.residual + _WEIGHT_EPS)
        rhs = scales[c.i] * c.shift
        if c.j != 0:
            a_shift[row, c.j - 1] = w
        if c.i != 0:
            a_shift[row, c.i - 1] = -w
        b_shift[row] = w * rhs

    a_shift_csr: csr_matrix = a_shift.tocsr()
    shift_sol = lsqr(a_shift_csr, b_shift)
    shifts = np.zeros(n_tiles, dtype=np.float64)
    shifts[1:] = shift_sol[0]
    acond_shift = float(shift_sol[5])
    r1_shift = float(shift_sol[2])

    for k in range(n_tiles):
        transforms[k] = (float(scales[k]), float(shifts[k]))

    pair_resid = []
    for c in usable:
        si, ti = transforms[c.i]
        sj, tj = transforms[c.j]
        pair_resid.append(si * c.scale - sj)
        pair_resid.append(si * c.shift + ti - tj)
    pair_rms = float(np.sqrt(np.mean(np.square(pair_resid)))) if pair_resid else 0.0

    logger.info(
        "global alignment: n_constraints=%d n_tiles=%d acond_scale=%.6g acond_shift=%.6g "
        "lsqr_r1_scale=%.6g lsqr_r1_shift=%.6g pairwise_residual_rms=%.6g",
        len(usable),
        n_tiles,
        acond_scale,
        acond_shift,
        r1_scale,
        r1_shift,
        pair_rms,
    )
    return transforms


def _hann1d(n: int) -> NDArray:
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / (n - 1))


def _tile_weight(tile: Tile, output_h: int, output_w: int) -> NDArray:
    """Separable Hann over the valid crop; no taper on the parent-image border.

    Tapering the outer edge would drive mosaic weight to ~0 at corners and
    amplify noise after the divide.
    """
    h = tile.row_end - tile.row_start
    w = tile.col_end - tile.col_start
    wy = _hann1d(h)
    wx = _hann1d(w)
    if tile.row_start == 0:
        wy[: max(h // 2, 1)] = 1.0
    if tile.row_end >= output_h:
        wy[h // 2 :] = 1.0
    if tile.col_start == 0:
        wx[: max(w // 2, 1)] = 1.0
    if tile.col_end >= output_w:
        wx[w // 2 :] = 1.0
    return np.outer(wy, wx)


def blend_tiles(
    tiles: Sequence[Tile],
    transforms: dict[int, tuple[float, float]],
    output_shape: tuple[int, int],
    depths: Sequence[NDArray] | None = None,
    *,
    nodata: float | int | None = None,
) -> NDArray:
    """Apply each tile affine and mosaic with a separable Hann window.

    tiles: geometry / pad metadata.
    depths: unpadded depth per tile, shape matching the tile's parent bounds.
        If omitted, `unpad_tile(tile).astype(float)` is used (tile.crop is depth).
    transforms: tile_index -> (scale, shift) applied as scale * depth + shift.
    output_shape: (H, W) of the parent.
    Returns float64 (H, W) in aligned units. Pixels with zero accumulated
    weight are NaN (never 0-filled nodata).
    """
    out_h, out_w = output_shape
    acc = np.zeros((out_h, out_w), dtype=np.float64)
    wsum = np.zeros((out_h, out_w), dtype=np.float64)

    for tile in tiles:
        if depths is None:
            depth = np.asarray(unpad_tile(tile), dtype=np.float64)
        else:
            depth = np.asarray(depths[tile.index], dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError(f"Tile {tile.index} depth must be (H, W), got {depth.shape}")
        expected = (tile.row_end - tile.row_start, tile.col_end - tile.col_start)
        if depth.shape != expected:
            raise ValueError(
                f"Tile {tile.index} depth shape {depth.shape} != valid bounds {expected}"
            )
        scale, shift = transforms[tile.index]
        aligned = scale * depth + shift
        valid = np.isfinite(aligned)
        if nodata is not None and np.isfinite(nodata):
            valid &= depth != nodata
        weight = _tile_weight(tile, out_h, out_w)
        weight = np.where(valid, weight, 0.0)
        sl = (slice(tile.row_start, tile.row_end), slice(tile.col_start, tile.col_end))
        acc[sl] += aligned * weight
        wsum[sl] += weight

    out = np.full((out_h, out_w), np.nan, dtype=np.float64)
    covered = wsum > 0.0
    out[covered] = acc[covered] / wsum[covered]
    return out


def depths_from_tiles(tiles: Sequence[Tile]) -> list[NDArray]:
    return [np.asarray(unpad_tile(t), dtype=np.float64) for t in tiles]


def align_and_stitch(
    tiles: Sequence[Tile],
    depths: Sequence[NDArray],
    output_shape: tuple[int, int],
    *,
    nodata: float | int | None = None,
) -> NDArray:
    """Pairwise affines → global (scale, shift) → Hann mosaic."""
    constraints = collect_pairwise_constraints(tiles, depths, nodata=nodata)
    transforms = solve_global_alignment(tiles, constraints)
    return blend_tiles(tiles, transforms, output_shape, depths, nodata=nodata)

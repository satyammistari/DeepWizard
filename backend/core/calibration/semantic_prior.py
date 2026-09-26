"""Per-class height priors and ground constraints. Values are metres above local ground (AGL).

This is nDSM-like (height above terrain), not a geoid/ellipsoid elevation.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi


class LandCover(IntEnum):
    """Integer IDs expected in a single-band semantic mask."""

    BACKGROUND = 0
    BUILDING = 1
    ROAD = 2
    WATER = 3
    TREE = 4
    BARE_GROUND = 5
    GRASS = 6


# (min, max, typical) metres AGL. Roads/water/bare ground sit on the terrain.
_PRIORS: dict[int, tuple[float, float, float]] = {
    int(LandCover.BACKGROUND): (0.0, 50.0, 0.0),
    int(LandCover.BUILDING): (3.0, 250.0, 12.0),
    int(LandCover.ROAD): (0.0, 0.5, 0.0),
    int(LandCover.WATER): (0.0, 0.3, 0.0),
    int(LandCover.TREE): (2.0, 50.0, 10.0),
    int(LandCover.BARE_GROUND): (0.0, 0.5, 0.0),
    int(LandCover.GRASS): (0.0, 1.5, 0.2),
}

GROUND_CLASSES: frozenset[int] = frozenset(
    {int(LandCover.ROAD), int(LandCover.WATER), int(LandCover.BARE_GROUND)}
)


def height_prior(class_id: int) -> tuple[float, float, float]:
    """Return (min, max, typical) metres AGL for a land-cover class."""
    if class_id not in _PRIORS:
        raise KeyError(f"No height prior for class_id={class_id}")
    return _PRIORS[class_id]


def _local_ground(depth: NDArray, valid: NDArray, size: int = 31) -> NDArray:
    """Low percentile of a neighbourhood — an estimate of local terrain.

    depth: (H, W) metres AGL or relative height. nodata is non-finite, not 0.
    """
    fill = float(np.nanmedian(depth[valid])) if np.any(valid) else 0.0
    filled = np.where(valid, depth, fill).astype(np.float64)
    odd = size if size % 2 == 1 else size + 1
    return ndi.percentile_filter(filled, percentile=15.0, size=odd)


def apply_ground_constraint(
    depth: NDArray,
    seg_mask: NDArray,
    *,
    nodata: float | int | None = None,
    window: int = 31,
) -> NDArray:
    """Force road / water / bare-ground pixels to the local ground surface.

    depth: (H, W) metres AGL (or relative height in the same units).
    seg_mask: (H, W) integer class ids (see LandCover).
    Returns float64 copy; nodata stays non-finite, never replaced with 0 m
    unless that pixel is a ground class with a valid local ground estimate.
    """
    if depth.shape != seg_mask.shape:
        raise ValueError(f"depth {depth.shape} and seg_mask {seg_mask.shape} differ")
    out = np.asarray(depth, dtype=np.float64).copy()
    valid = np.isfinite(out)
    if nodata is not None and np.isfinite(nodata):
        valid &= out != nodata
        out[~np.isfinite(np.asarray(depth, dtype=np.float64))] = np.nan
        out[np.asarray(depth) == nodata] = np.nan
        valid = np.isfinite(out)
    ground = np.isin(seg_mask, list(GROUND_CLASSES)) & valid
    if not np.any(ground):
        return out
    local = _local_ground(out, valid, size=window)
    out[ground] = local[ground]
    return out


def hgdnet_snap_to_ground(
    height_agl: NDArray,
    seg_mask: NDArray,
    *,
    threshold_m: float = 3.0,
    nodata: float | int | None = None,
) -> NDArray:
    """Snap non-building pixels shorter than 3 m AGL to 0 m (nDSM ground).

    HGDNet correction (arXiv:2308.05387 / IGARSS 2023, "HGDNet: A Height-
    Hierarchy Guided Dual-Decoder Network for Single View Building Extraction
    and Height Estimation"): building heights are generally not less than 3 m,
    so estimates are corrected to 0 where the hierarchy class is ground and the
    height branch is below 3 m. We apply the same cutoff to non-building
    semantic classes.
    """
    if height_agl.shape != seg_mask.shape:
        raise ValueError("height_agl and seg_mask must match")
    out = np.asarray(height_agl, dtype=np.float64).copy()
    valid = np.isfinite(out)
    if nodata is not None and np.isfinite(nodata):
        valid &= out != nodata
    non_building = seg_mask != int(LandCover.BUILDING)
    snap = valid & non_building & (out < threshold_m)
    out[snap] = 0.0
    out[~valid] = np.nan
    return out


def segment_landcover(rgb: NDArray[np.uint8]) -> NDArray[np.int32]:
    """Semantic mask on LandCover ids. Prefers SegFormer-ADE; falls back to RGB priors."""
    try:
        return _segformer_landcover(rgb)
    except Exception:
        return _rgb_landcover_proxy(rgb)


def semantic_height_priors(rgb: NDArray[np.uint8]) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Return (class_mask, typical AGL metres) from SegFormer or RGB fallback."""
    mask = segment_landcover(rgb)
    typical = np.zeros(mask.shape, dtype=np.float32)
    for class_id, (_lo, _hi, typ) in _PRIORS.items():
        typical[mask == class_id] = np.float32(typ)
    return mask, typical


def _segformer_landcover(rgb: NDArray[np.uint8]) -> NDArray[np.int32]:
    """nvidia/segformer-b2-finetuned-ade-512-512. Inference uses local weights only."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    model_id = "nvidia/segformer-b2-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id, local_files_only=True)
    model.eval()
    image = np.asarray(rgb, dtype=np.uint8)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    up = F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False)
    ade = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    return _ade_to_landcover(ade)


# ADE20K ids used by nvidia SegFormer. Mapped onto LandCover for height priors.
_ADE_BUILDING = {1, 25, 48}
_ADE_ROAD = {6, 11, 52}
_ADE_WATER = {21, 26, 60, 109, 113}
_ADE_TREE = {4, 17}
_ADE_GRASS = {9, 10, 29}
_ADE_BARE = {13, 46, 94}


def _ade_to_landcover(ade: NDArray[np.int32]) -> NDArray[np.int32]:
    out = np.full(ade.shape, int(LandCover.BACKGROUND), dtype=np.int32)
    out[np.isin(ade, list(_ADE_BUILDING))] = int(LandCover.BUILDING)
    out[np.isin(ade, list(_ADE_ROAD))] = int(LandCover.ROAD)
    out[np.isin(ade, list(_ADE_WATER))] = int(LandCover.WATER)
    out[np.isin(ade, list(_ADE_TREE))] = int(LandCover.TREE)
    out[np.isin(ade, list(_ADE_GRASS))] = int(LandCover.GRASS)
    out[np.isin(ade, list(_ADE_BARE))] = int(LandCover.BARE_GROUND)
    return out


def _rgb_landcover_proxy(rgb: NDArray[np.uint8]) -> NDArray[np.int32]:
    """RGB-only land-cover proxy when SegFormer weights are not cached."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        r = g = b = arr
        gray = arr
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        gray = arr.mean(axis=-1)
    if gray.max() > 1.5:
        r, g, b, gray = r / 255.0, g / 255.0, b / 255.0, gray / 255.0
    out = np.full(gray.shape, int(LandCover.BACKGROUND), dtype=np.int32)
    ndvi_like = (g - r) / np.clip(g + r, 1e-6, None)
    out[ndvi_like > 0.12] = int(LandCover.TREE)
    out[(gray < 0.18)] = int(LandCover.WATER)
    out[(gray > 0.22) & (gray < 0.45) & (np.abs(r - g) < 0.08)] = int(LandCover.ROAD)
    out[(gray > 0.55) & (np.abs(r - g) < 0.05)] = int(LandCover.BARE_GROUND)
    building = (gray > 0.35) & (ndvi_like < 0.05) & (np.abs(r - b) < 0.12)
    out[building] = int(LandCover.BUILDING)
    return out

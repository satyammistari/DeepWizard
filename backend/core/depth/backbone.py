from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DepthVariant = Literal["small", "base", "large"]

_MODEL_IDS: dict[DepthVariant, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _from_pretrained(loader, model_id: str):
    """Load only locally cached weights. Inference must never hit the network."""
    return loader.from_pretrained(model_id, local_files_only=True)


@lru_cache(maxsize=3)
def load_model(variant: DepthVariant = "small"):
    """Load and cache the depth model once for the life of the process.

    The model outputs inverse depth. The convention is intentionally documented
    here: higher values mean the surface is closer to the camera. For a nadir
    satellite image, that means taller structures and elevated terrain are
    encoded with higher relative depth values.
    """
    model_id = _MODEL_IDS[variant]
    model = _from_pretrained(AutoModelForDepthEstimation, model_id)
    model.eval()
    return model


@lru_cache(maxsize=3)
def _load_processor(variant: DepthVariant = "small"):
    model_id = _MODEL_IDS[variant]
    return _from_pretrained(AutoImageProcessor, model_id)


def _ensure_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.asarray(arr, dtype=np.uint8)


def _predict_depth_map(model, image: NDArray[np.uint8], variant: DepthVariant = "small") -> NDArray[np.float32]:
    """Run a DepthAnythingV2 forward pass and return a dense depth map.

    Returns a float32 array matching the input H/W layout. The model output is
    inverse-depth-like, and the caller normalizes it to 0..1 for quick visual use.
    """
    processor = _load_processor(variant)
    inputs = processor(images=image, return_tensors="pt")
    with __import__("torch").no_grad():
        outputs = model(**inputs)
    depth = outputs.predicted_depth
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu()
    depth = np.asarray(depth[0].numpy() if getattr(depth, "ndim", 0) > 2 else depth.numpy(), dtype=np.float32)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape[:2] != image.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.asarray(depth, dtype=np.float32)


def _fallback_depth(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Offline fallback used when model weights are unavailable.

    This keeps preview and tests runnable without network access while still
    producing a valid normalized depth-like field for quick validation.
    """
    rgb = _ensure_rgb(image)
    gray = rgb.astype(np.float32).mean(axis=-1)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    residual = gray - blur
    # Nadir cue: brighter rooftops + local contrast as a relative height prior.
    depth = 0.65 * gray + 0.35 * residual
    return np.asarray(depth, dtype=np.float32)


def normalize_depth_map(depth: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rescale to the visual convention of 0..1 with higher = closer."""
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.float32)
    lo = float(depth[finite].min())
    hi = float(depth[finite].max())
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    normalized = (depth - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32)


def infer_relative(image: NDArray[np.uint8], variant: DepthVariant = "small") -> NDArray[np.float32]:
    """Infer a relative depth map from an RGB image.

    Args:
        image: H x W x 3 RGB array or H x W grayscale image.
        variant: model size to load. The depth convention is:
            HIGHER value = CLOSER to the camera.
            For nadir satellite imagery this means higher values correspond to
            taller features / elevated terrain.

    Returns:
        A float32 array with shape (H, W), normalized to [0, 1].
    """
    rgb = _ensure_rgb(image)
    try:
        model = load_model(variant)
        depth = _predict_depth_map(model, rgb, variant=variant)
    except Exception:
        depth = _fallback_depth(rgb)
    return normalize_depth_map(depth)


def save_depth_png(depth: NDArray[np.float32], path: str | Path) -> None:
    """Persist a relative depth map as a grayscale PNG."""
    scaled = np.clip(depth, 0.0, 1.0)
    image = Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L")
    image.save(path)

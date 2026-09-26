from __future__ import annotations

import io
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

from backend.config import settings
from backend.core.depth.backbone import infer_relative
from backend.core.pipeline import calibrate_scene, preview_payload
from backend.core.viz.maps import colorize, encode_png_b64
from backend.validation.evaluate import compute_validation_metrics, create_synthetic_test_case
from backend.core.calibration.fusion import calibrate
from backend.core.geo.io import RasterData

app = FastAPI(title="DepthWizard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_rgb(contents: bytes) -> np.ndarray:
    try:
        rgb = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    return np.asarray(rgb, dtype=np.uint8)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/depth/preview")
async def depth_preview(file: UploadFile = File(...)) -> Response:
    """Return an in-memory PNG of the normalized relative depth map."""
    contents = await file.read()
    try:
        image = _read_rgb(contents)
        depth = infer_relative(image)
        png_buffer = io.BytesIO()
        Image.fromarray(np.clip(depth * 255.0, 0, 255).astype(np.uint8), mode="L").save(
            png_buffer,
            format="PNG",
        )
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    return Response(content=png_buffer.getvalue(), media_type="image/png")


@app.post("/v1/depth/maps")
async def depth_maps(file: UploadFile = File(...)) -> JSONResponse:
    """Relative depth plus colorized / overlay previews for the UI."""
    contents = await file.read()
    try:
        image = _read_rgb(contents)
        payload = preview_payload(image)
    except ValueError as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    return JSONResponse(
        {
            "status": "success",
            "width": payload["width"],
            "height": payload["height"],
            "relative_gray_b64": payload["relative_gray_b64"],
            "relative_color_b64": payload["relative_color_b64"],
            "overlay_b64": payload["overlay_b64"],
        }
    )


@app.post("/v1/depth/tile_predict")
async def depth_tile_predict(file: UploadFile = File(...)) -> Response:
    """Run tiled inference + stitching and return a full-resolution depth PNG."""
    contents = await file.read()
    try:
        image = _read_rgb(contents)
        from backend.core.depth.tile_infer import infer_and_stitch

        stitched = infer_and_stitch(image, tile_size=256, overlap_frac=0.25)
        finite = np.isfinite(stitched)
        out = np.zeros_like(stitched, dtype=np.float32)
        if finite.any():
            lo = float(np.nanmin(stitched))
            hi = float(np.nanmax(stitched))
            if hi > lo:
                out[finite] = (stitched[finite] - lo) / (hi - lo)
        png_buffer = io.BytesIO()
        Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(
            png_buffer, format="PNG"
        )
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)
    return Response(content=png_buffer.getvalue(), media_type="image/png")


@app.post("/v1/calibration/calibrate_depth")
async def calibrate_depth(
    file: UploadFile = File(...),
    sun_azimuth_deg: float = Query(...),
    sun_elevation_deg: float = Query(...),
    gsd_m: float = Query(...),
    minx: float = Query(...),
    miny: float = Query(...),
    maxx: float = Query(...),
    maxy: float = Query(...),
    dem_source: str = Query("SRTMGL1"),
    cache_dir: str = Query(".dem_cache"),
) -> JSONResponse:
    """Calibrate monocular depth to metric elevations using DEM and shadow GCPs."""
    try:
        contents = await file.read()
        image = _read_rgb(contents)
        maps = calibrate_scene(
            image,
            sun_azimuth_deg=sun_azimuth_deg,
            sun_elevation_deg=sun_elevation_deg,
            gsd_m=gsd_m,
            bounds_wgs84=(minx, miny, maxx, maxy),
            dem_source=dem_source,
            cache_dir=cache_dir,
        )
        result = maps.result
        response_data: dict[str, Any] = {
            "status": "success",
            "calibrated_dsm_png": maps.calibrated_png_hex,
            "maps": maps.maps_b64,
            "num_gcps": len(maps.gcps),
            "gcps": [
                {
                    "row": gcp.row,
                    "col": gcp.col,
                    "height_m": gcp.height_m,
                    "confidence": gcp.confidence,
                }
                for gcp in maps.gcps
            ],
            "calibration": {
                "scale": float(result.scale),
                "offset": float(result.offset),
                "method": result.method,
                "inlier_ratio": result.inlier_ratio,
                "residual_std_m": result.residual_std_m,
                "vertical_datum": result.vertical_datum,
                "low_confidence": result.low_confidence,
                "reason": result.reason,
                "used_dem_fallback": maps.used_dem_fallback,
                "dem_source": maps.dem_source,
            },
            "metrics": maps.metrics,
            "stats": {
                "min_m": float(np.nanmin(maps.metric_dsm)),
                "max_m": float(np.nanmax(maps.metric_dsm)),
                "mean_uncertainty_m": float(np.nanmean(maps.uncertainty)),
            },
        }
        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.post("/v1/validation/run")
async def validation_run() -> JSONResponse:
    """Smoke-test live endpoints and report synthetic DSM accuracy."""
    from backend.validation.harness import create_sample_image, run_validation

    report = run_validation(include_calibration=True, image_size=128)
    dem, synthetic_depth, mask = create_synthetic_test_case(height=96, width=96, noise_std=0.02, seed=7)
    rel_raster = RasterData(array=synthetic_depth, crs=None, transform=None, nodata=np.nan)
    dem_raster = RasterData(array=dem, crs=None, transform=None, nodata=np.nan)
    fit = calibrate(rel_raster, dem_raster, mask=mask)
    predicted = fit.apply(synthetic_depth)
    metrics = compute_validation_metrics(predicted, dem, mask=mask)

    stages = [
        {
            "name": stage.name,
            "ok": stage.ok,
            "status": stage.status_code,
            "ms": stage.elapsed_ms,
            "detail": stage.detail,
        }
        for stage in report.stages
    ]
    accuracy_ok = metrics.correlation > 0.95 and metrics.rmse < 20.0
    stages.append(
        {
            "name": "synthetic_accuracy",
            "ok": accuracy_ok,
            "status": 200 if accuracy_ok else 500,
            "ms": 0.0,
            "detail": (
                f"RMSE={metrics.rmse:.2f}m MAE={metrics.mae:.2f}m "
                f"R²={metrics.r_squared:.3f} r={metrics.correlation:.3f}"
            ),
        }
    )
    ok = all(row["ok"] for row in stages)
    return JSONResponse(
        {
            "status": "success" if ok else "error",
            "ok": ok,
            "summary": "Validation passed" if ok else "Validation failed",
            "stages": stages,
            "metrics": {
                "rmse_m": metrics.rmse,
                "mae_m": metrics.mae,
                "r_squared": metrics.r_squared,
                "correlation": metrics.correlation,
                "bias_m": metrics.bias,
            },
            "maps": {
                "reference_color_b64": encode_png_b64(colorize(dem)),
                "predicted_color_b64": encode_png_b64(colorize(predicted)),
                "relative_color_b64": encode_png_b64(colorize(synthetic_depth)),
            },
        }
    )

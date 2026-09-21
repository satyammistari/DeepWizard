# DepthWizard Architecture

DepthWizard turns a single optical satellite RGB image into a metric Digital Surface Model (DSM) and an interactive 3D viewer. This document describes the intended layout, data contracts, and non-negotiable geospatial rules. The depth model itself is **not** implemented in the initial scaffold.

## Goals

- **Accuracy (SIH 50%)**: RMSE / MAE / correlation of the DSM, stratified across urban, sparse, hilly, and forested terrain.
- **Visualization + deployment (SIH 50%)**: 3D flythrough, texture-draped terrain, standalone offline packaging (Tauri + PyInstaller later).

Heights are always in **metres**. Vertical datum (EGM96 geoid vs WGS84 ellipsoid) must be documented on every function that writes or converts elevation.

## Repository layout

```
DepthWizard/
  docs/architecture.md          # this file
  pyproject.toml                # Python 3.11 backend + pytest
  .env                          # local settings (no secrets required for scaffold)
  backend/
    config.py                   # pydantic-settings, reads .env
    api/
      main.py                   # FastAPI app, CORS, /health
    core/
      geo/
        io.py                   # raster read/write; CRS preserved exactly
      # Planned (not in scaffold):
      # depth/                  tiled inverse-depth inference
      # scale/                  relative → metric alignment
      # shadows/                sun-geometry height cues
      # export/                 COG / glTF / LAS
  tests/                        # mirrors backend/ (pytest)
    core/geo/test_io.py
  frontend/                     # React 18 + TS + Vite + Tailwind + R3F
    src/
      components/Viewer3D/
      store/useAppStore.ts
```

Run from the **repository root**:

- API: `uvicorn backend.api.main:app --reload --port 8000`
- UI: `cd frontend && npm run dev` (Vite on `http://localhost:5173`)

## Runtime topology

```
┌─────────────────────────┐     HTTP      ┌──────────────────────────┐
│  Vite / React / R3F     │◄─────────────►│  FastAPI (uvicorn)       │
│  Viewer3D + zustand     │  localhost    │  /health, (later /infer) │
└─────────────────────────┘   :5173/:8000 └────────────┬─────────────┘
                                                       │
                                                       ▼
                                          rasterio I/O  →  DSM GeoTIFF
                                          CRS + affine copied exactly
```

The packaged app must run **fully offline**. External DEM/sun-ephemeris caches are fetched only at build/prep time.

## Modes

| Mode | Meaning | CRS requirement |
| --- | --- | --- |
| `relative` | Unitless / affine heights for visualization. Inverse depth scaled for display only. | CRS preferred but not required. |
| `absolute` | Metric DSM in metres, georeferenced. | **Input must have a CRS.** `read_raster(..., mode="absolute")` fails loudly if CRS is missing. |

Frontend `useAppStore.mode` mirrors this (`'relative' | 'absolute'`).

## Geospatial I/O contract

`backend.core.geo.io` is the only entry point for raster files.

- `RasterData`: `array`, `transform`, `crs`, `nodata`, plus `bounds` derived from array shape and affine transform.
- `read_raster(path, *, mode=...)`: opens with rasterio; does **not** reproject.
- `write_raster(path, array, profile, nodata)`: copies `crs` and `transform` from `profile` without modification. Refuses to write if CRS or transform is missing.

Never:

- silently drop or reproject CRS / affine transform
- turn nodata pixels into `0` m — mask and propagate nodata through every stage

## Planned processing pipeline (not implemented yet)

1. **Ingest** RGB GeoTIFF via `read_raster`.
2. **Tiled inverse-depth** (future). Depth Anything–class models output **inverse depth** (higher = closer). For nadir satellites, closer ≈ taller. Tiles (~518 px native) need global scale alignment to avoid seams.
3. **Metric scaling** (future): SRTM / Copernicus DEM is a radar DSM (phase centre in canopy — not treetop and not bare earth). Shadow-length heights need GSD and solar geometry (`pysolar`) and georeferenced inputs only.
4. **Uncertainty**: every model output ships a per-pixel uncertainty raster.
5. **Export**: Cloud-Optimized GeoTIFF, glTF mesh (`trimesh` / `pygltflib`), optional LAS.

## Backend API (scaffold)

- `GET /health` → `{ "status": "ok" }`
- CORS allowlist: `http://localhost:5173`
- Settings: `backend.config.Settings` (pydantic-settings, `.env`)

Inference and export routes will be added later; they must not perform network calls at inference time.

## Frontend (scaffold)

- Vite + React + TypeScript + Tailwind
- `@react-three/fiber` / `@react-three/drei` Canvas with `OrbitControls`, lights, and a 100×100 placeholder plane
- Zustand store: `{ mode, dsmUrl, textureUrl, exaggeration }`
- Recharts reserved for RMSE / uncertainty plots

## Configuration

No hardcoded paths. `backend.config.Settings` reads `.env` (see `.env.example`). Typical keys: API host/port, CORS origins, data directory, default mode, vertical datum label.

## Testing

Pytest tree mirrors source: `tests/core/geo/test_io.py` round-trips a synthetic GeoTIFF and asserts CRS WKT and affine transform are byte-identical to the values requested at write time. Absolute mode without CRS must raise a clear error.

"""Command-line entry points for DepthWizard packaging."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from backend.validation.export import export_validation_bundle
from backend.validation.harness import main as validation_main
from backend.training.fine_tune import FineTuneConfig, fine_tune_affine_head, save_fine_tune_result


def api_main(argv: list[str] | None = None) -> int:
    """Launch the FastAPI app with uvicorn."""
    parser = argparse.ArgumentParser(description="Start the DepthWizard API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run("backend.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    """Run the smoke-test validation harness."""
    return validation_main(argv)


def prefetch_dem_main(argv: list[str] | None = None) -> int:
    """Prefetch DEM tiles into the local cache."""
    parser = argparse.ArgumentParser(description="Prefetch DEM tiles into local cache")
    parser.add_argument("--bbox", required=True, help="minx,miny,maxx,maxy in WGS84")
    parser.add_argument("--product", choices=["SRTMGL1", "COP30", "CARTODEM"], default="SRTMGL1")
    parser.add_argument("--cache-dir", type=Path, default=Path(".dem_cache"))
    parser.add_argument("--force", action="store_true", help="Allow network even if settings disallow it")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    try:
        from backend import config
        from backend.core.calibration import dem_source
    except Exception as exc:
        logging.error("Failed to import project modules: %s", exc)
        return 2

    parts = [float(x) for x in args.bbox.split(",")]
    if len(parts) != 4:
      logging.error("bbox must be minx,miny,maxx,maxy")
      return 2

    settings = getattr(config, "settings", None)
    if settings is not None and not getattr(settings, "allow_network", False) and not args.force:
        logging.error("Network access is disabled by settings. Use --force to override.")
        return 3

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Fetching DEM product %s for bbox %s into %s", args.product, parts, args.cache_dir)

    try:
        result = dem_source.fetch_dem(bounds_wgs84=tuple(parts), source=args.product, cache_dir=str(args.cache_dir))
    except Exception as exc:
        logging.exception("DEM fetch failed: %s", exc)
        return 4

    path = getattr(result, "path", None) or getattr(result, "filepath", None) or getattr(result, "filename", None)
    if path:
        logging.info("Cached DEM at: %s", path)
    else:
        logging.info("Fetch result: %r", result)

    return 0


def export_main(argv: list[str] | None = None) -> int:
    """Export deterministic validation artifacts."""
    parser = argparse.ArgumentParser(description="Export validation artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    args = parser.parse_args(argv)

    bbox = tuple(args.bbox) if args.bbox else None
    bundle = export_validation_bundle(
        args.output_dir,
        image_size=args.image_size,
        include_calibration=args.include_calibration,
        bbox=bbox,
    )
    logging.info("Exported %d artifacts to %s", len(bundle.artifacts), args.output_dir)
    return 0 if bundle.ok else 1


def train_main(argv: list[str] | None = None) -> int:
    """Train a minimal affine calibration head on synthetic data."""
    parser = argparse.ArgumentParser(description="Train an affine calibration head")
    parser.add_argument("--output", type=Path, default=Path("fine_tune_result.json"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    import numpy as np

    relative = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    target = 3.5 * relative + 12.0
    result = fine_tune_affine_head(
        relative,
        target,
        config=FineTuneConfig(epochs=args.epochs, lr=args.lr, init_scale=0.0, init_offset=0.0, seed=args.seed),
    )
    save_fine_tune_result(result, args.output)
    print(result.to_dict())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depthwizard", description="DepthWizard command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Run the validation harness")
    validate_parser.add_argument("--include-calibration", action="store_true")
    validate_parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    validate_parser.add_argument("--image-size", type=int, default=256)
    validate_parser.add_argument("--output", type=Path)

    prefetch_parser = subparsers.add_parser("prefetch-dem", help="Prefetch DEM tiles")
    prefetch_parser.add_argument("--bbox", required=True)
    prefetch_parser.add_argument("--product", choices=["SRTMGL1", "COP30", "CARTODEM"], default="SRTMGL1")
    prefetch_parser.add_argument("--cache-dir", type=Path, default=Path(".dem_cache"))
    prefetch_parser.add_argument("--force", action="store_true")
    prefetch_parser.add_argument("--verbose", "-v", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export validation artifacts")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--image-size", type=int, default=256)
    export_parser.add_argument("--include-calibration", action="store_true")
    export_parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))

    train_parser = subparsers.add_parser("train", help="Train a calibration head")
    train_parser.add_argument("--output", type=Path, default=Path("fine_tune_result.json"))
    train_parser.add_argument("--epochs", type=int, default=400)
    train_parser.add_argument("--lr", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=42)

    args, remaining = parser.parse_known_args(argv)
    if args.command == "api":
        return api_main(["--host", args.host, "--port", str(args.port)] + (["--reload"] if args.reload else []))
    if args.command == "validate":
        forwarded = []
        if args.include_calibration:
            forwarded.append("--include-calibration")
        if args.bbox:
            forwarded.extend(["--bbox", *map(str, args.bbox)])
        forwarded.extend(["--image-size", str(args.image_size)])
        if args.output:
            forwarded.extend(["--output", str(args.output)])
        return validate_main(forwarded)

    if args.command == "export":
        forwarded = ["--output-dir", str(args.output_dir), "--image-size", str(args.image_size)]
        if args.include_calibration:
            forwarded.append("--include-calibration")
        if args.bbox:
            forwarded.extend(["--bbox", *map(str, args.bbox)])
        return export_main(forwarded)

    if args.command == "train":
        forwarded = ["--output", str(args.output), "--epochs", str(args.epochs), "--lr", str(args.lr), "--seed", str(args.seed)]
        return train_main(forwarded)

    forwarded = ["--bbox", args.bbox, "--product", args.product, "--cache-dir", str(args.cache_dir)]
    if args.force:
        forwarded.append("--force")
    if args.verbose:
        forwarded.append("--verbose")
    return prefetch_dem_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

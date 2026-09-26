"""Wrapper for the packaged DEM prefetch command."""

from backend.cli import prefetch_dem_main as main


if __name__ == "__main__":
    raise SystemExit(main())

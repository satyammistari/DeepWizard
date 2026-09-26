from backend.core.calibration.dem_source import (
    DEMUnavailable,
    DemSource,
    fetch_dem,
    fetch_reference_dem,
    reproject_to_match,
)

# Optional imports: some calibration submodules require extra dependencies
# (e.g., `pysolar`) which may not be available in minimal test environments.
# Import them lazily and tolerate missing optional packages so importing the
# `backend.core.calibration` package does not fail when only `dem_source` is used.
try:
    from backend.core.calibration.fusion import Anchor, CalibrationResult, calibrate
except Exception:
    Anchor = None
    CalibrationResult = None
    calibrate = None

try:
    from backend.core.calibration.semantic_prior import (
        LandCover,
        apply_ground_constraint,
        height_prior,
        hgdnet_snap_to_ground,
    )
except Exception:
    LandCover = None
    apply_ground_constraint = None
    height_prior = None
    hgdnet_snap_to_ground = None

try:
    from backend.core.calibration.shadow_gcp import (
        GCP,
        detect_shadows,
        parse_sun_from_metadata,
        shadow_heights,
        sun_angles,
    )
except Exception:
    GCP = None
    detect_shadows = None
    parse_sun_from_metadata = None
    shadow_heights = None
    sun_angles = None

__all__ = [
    "Anchor",
    "CalibrationResult",
    "DEMUnavailable",
    "DemSource",
    "GCP",
    "LandCover",
    "apply_ground_constraint",
    "calibrate",
    "detect_shadows",
    "fetch_dem",
    "fetch_reference_dem",
    "height_prior",
    "hgdnet_snap_to_ground",
    "parse_sun_from_metadata",
    "reproject_to_match",
    "shadow_heights",
    "sun_angles",
]

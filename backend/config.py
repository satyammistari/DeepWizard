from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment / `.env`. No hardcoded paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DW_",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    data_dir: Path = Path("./data")
    mode: Literal["relative", "absolute"] = "relative"
    # Label only; conversions between geoid and ellipsoid are not done here.
    vertical_datum: str = "EGM96"
    opentopography_api_key: str = ""
    dem_cache_dir: Path = Path("./data/dem_cache")
    cartodem_dir: Path = Path("./data/cartodem")
    # Packaged inference must stay offline; prep-time fetches set this true.
    allow_network: bool = True
    source_disagreement_m: float = 15.0


settings = Settings()

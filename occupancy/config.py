"""Uygulama ayarları.

Ayarlar üç kaynaktan gelir, öncelik sırasıyla: CLI argümanları (bkz.
`occupancy/__main__.py`), ortam değişkenleri / `.env`, ve buradaki
varsayılanlar.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GECERLI_LOG_SEVIYELERI = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


class Settings(BaseSettings):
    """`OCCUPANCY_` önekli ortam değişkenlerinden okunan ayarlar."""

    model_config = SettingsConfigDict(
        env_prefix="OCCUPANCY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Aşama 1 — video girişi
    source: str = "ornek/demo.mp4"
    loop_video: bool = True

    # Aşama 2 — tespit
    model_path: str = "yolov8n.pt"
    conf_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # Aşama 4 — bölge analizi
    zones_path: Path = Path("ornek/zones.json")
    # Geçiş çizgisinin iki yanındaki kararsız bant: ayak noktası bu kadar
    # piksel uzaklaşmadan taraf değiştirmiş sayılmaz (bkz. occupancy/zones.py).
    crossing_band_px: float = Field(default=25.0, ge=0.0)

    # Aşama 6/7 — depolama ve raporlama
    db_path: Path = Path("data/occupancy.db")
    reports_dir: Path = Path("reports")
    snapshot_interval_sec: int = Field(default=10, ge=1)

    # Aşama 5 — web
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # Aşama 8 — logging
    logs_dir: Path = Path("logs")
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _log_seviyesi_gecerli_mi(cls, deger: str) -> str:
        seviye = deger.upper()
        if seviye not in GECERLI_LOG_SEVIYELERI:
            gecerliler = ", ".join(sorted(GECERLI_LOG_SEVIYELERI))
            raise ValueError(f"Geçersiz log seviyesi: {deger!r}. Geçerli olanlar: {gecerliler}")
        return seviye

    @property
    def log_level_number(self) -> int:
        return getattr(logging, self.log_level, logging.INFO)

    def ensure_directories(self) -> None:
        """Veri, rapor ve log dizinlerini (yoksa) oluşturur."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

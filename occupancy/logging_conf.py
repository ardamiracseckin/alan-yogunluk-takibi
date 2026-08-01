"""Uygulama genelinde logging kurulumu.

Konsola okunabilir, dosyaya döngüsel (rotating) log yazar. Kurulum tek
seferliktir; birden fazla çağrı handler'ları çoğaltmaz.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_BICIM = "%(asctime)s %(levelname)-8s %(name)-22s %(message)s"
_TARIH_BICIMI = "%Y-%m-%d %H:%M:%S"
_KURULDU = False


def setup_logging(logs_dir: Path, level: str = "INFO") -> logging.Logger:
    """Kök logger'ı yapılandırır ve uygulama logger'ını döndürür."""
    global _KURULDU

    kok = logging.getLogger()
    if _KURULDU:
        return logging.getLogger("occupancy")

    logs_dir.mkdir(parents=True, exist_ok=True)
    bicimlendirici = logging.Formatter(_BICIM, datefmt=_TARIH_BICIMI)

    konsol = logging.StreamHandler()
    konsol.setFormatter(bicimlendirici)

    dosya = RotatingFileHandler(
        logs_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    dosya.setFormatter(bicimlendirici)

    kok.setLevel(getattr(logging, level.upper(), logging.INFO))
    kok.addHandler(konsol)
    kok.addHandler(dosya)

    # Ultralytics her tahminde satır basar; INFO seviyesinde susturuyoruz.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    _KURULDU = True
    return logging.getLogger("occupancy")


def get_logger(name: str) -> logging.Logger:
    """Modül logger'ı döndürür (`occupancy.zones` gibi)."""
    return logging.getLogger(name)

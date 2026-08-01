"""Komut satırı giriş noktası.

    python -m occupancy                          # .env / varsayılan ayarlarla
    python -m occupancy --source 0               # webcam
    python -m occupancy --source rtsp://kamera   # ağ kamerası
    python -m occupancy --no-web                 # dashboard'suz, sadece işleme

Öncelik sırası: CLI argümanları > ortam değişkenleri / .env > varsayılanlar.
"""

from __future__ import annotations

import argparse

from occupancy.config import Settings
from occupancy.detection import YoloPersonDetector
from occupancy.logging_conf import setup_logging
from occupancy.pipeline import Pipeline
from occupancy.storage import Storage
from occupancy.tracking import PersonTracker
from occupancy.video import VideoSource
from occupancy.zones import ZoneManager


def build_parser() -> argparse.ArgumentParser:
    ayristirici = argparse.ArgumentParser(
        prog="python -m occupancy",
        description="Kameradan gerçek zamanlı kişi sayımı, giriş/çıkış takibi ve yoğunluk analizi.",
    )
    ayristirici.add_argument(
        "--source", help="Video dosyası, webcam indeksi (0) veya rtsp:// adresi"
    )
    ayristirici.add_argument("--zones", help="Bölge tanım dosyası (JSON)")
    ayristirici.add_argument("--model", help="YOLO model ağırlığı")
    ayristirici.add_argument("--conf", type=float, help="Tespit güven eşiği (0-1)")
    ayristirici.add_argument("--db", help="SQLite veritabanı yolu")
    ayristirici.add_argument("--host", help="Web sunucusu adresi")
    ayristirici.add_argument("--port", type=int, help="Web sunucusu portu")
    ayristirici.add_argument("--log-level", help="CRITICAL, ERROR, WARNING, INFO veya DEBUG")
    ayristirici.add_argument(
        "--no-loop", action="store_true", help="Video dosyası bitince başa sarma"
    )
    ayristirici.add_argument(
        "--no-web", action="store_true", help="Dashboard'u açma, sadece işleme yap"
    )
    return ayristirici


def build_settings(argv: list[str] | None = None) -> tuple[Settings, argparse.Namespace]:
    """CLI argümanlarını ayarların üzerine bindirir."""
    argumanlar = build_parser().parse_args(argv)
    ayarlar = Settings()

    ezilenler = {
        "source": argumanlar.source,
        "zones_path": argumanlar.zones,
        "model_path": argumanlar.model,
        "conf_threshold": argumanlar.conf,
        "db_path": argumanlar.db,
        "host": argumanlar.host,
        "port": argumanlar.port,
        "log_level": argumanlar.log_level,
    }
    verilen = {ad: deger for ad, deger in ezilenler.items() if deger is not None}
    if argumanlar.no_loop:
        verilen["loop_video"] = False

    # Settings üzerinden yeniden kurulur ki CLI'dan gelen değerler de doğrulansın.
    return Settings(**{**ayarlar.model_dump(), **verilen}), argumanlar


def build_pipeline(ayarlar: Settings) -> tuple[Pipeline, Storage]:
    """Ayarlardan gerçek bileşenleri kurar."""
    depo = Storage(ayarlar.db_path)
    hat = Pipeline(
        video_source=VideoSource(ayarlar.source, loop=ayarlar.loop_video),
        detector=YoloPersonDetector(ayarlar.model_path, conf=ayarlar.conf_threshold),
        tracker=PersonTracker(),
        zones=ZoneManager.from_file(ayarlar.zones_path),
        storage=depo,
        snapshot_interval_sec=ayarlar.snapshot_interval_sec,
    )
    return hat, depo


def main(argv: list[str] | None = None) -> int:
    ayarlar, argumanlar = build_settings(argv)
    ayarlar.ensure_directories()
    logger = setup_logging(ayarlar.logs_dir, ayarlar.log_level)
    logger.info("Kaynak: %s | Bölgeler: %s", ayarlar.source, ayarlar.zones_path)

    hat, depo = build_pipeline(ayarlar)

    try:
        if argumanlar.no_web:
            hat.run_until_complete()
        else:
            _sunucuyu_calistir(hat, ayarlar)
    except KeyboardInterrupt:
        logger.info("Klavyeden durduruldu.")
    finally:
        hat.stop()
        depo.close()

    return 1 if hat.error is not None else 0


def _sunucuyu_calistir(hat: Pipeline, ayarlar: Settings) -> None:
    import uvicorn

    from web.app import create_app

    hat.start()
    print(f"\n  Dashboard: http://{ayarlar.host}:{ayarlar.port}\n")
    uvicorn.run(
        create_app(hat, ayarlar),
        host=ayarlar.host,
        port=ayarlar.port,
        log_level=ayarlar.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

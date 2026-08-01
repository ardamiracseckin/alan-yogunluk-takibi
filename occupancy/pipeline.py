"""İşleme hattı — tüm aşamaları birbirine bağlayan döngü.

Hat kendi thread'inde koşar; web tarafı yalnızca "en son üretilen kare" ve
"en son ölçüm" değerlerini okur. Kare kuyruğu tutulmaz: gerçek zamanlılık
gecikmeye tercih edilir, izleyici hep en güncel görüntüyü görür.

Hata felsefesi iki katmanlı:

* **Kayıt hataları** (veritabanı) yutulur ve loglanır — izleme durmamalı.
* **İşleme hataları** (model çöktü, kare bozuk) hattı durdurur ama sessizce
  değil: `error` alanına yazılır, `/health` bunu dışarı bildirir.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from occupancy.density import DensityMap, zone_density
from occupancy.logging_conf import get_logger
from occupancy.models import LiveStats, Snapshot
from occupancy.overlay import draw_overlay
from occupancy.zones import ZoneManager

logger = get_logger(__name__)

JPEG_KALITESI = 80
FPS_PENCERESI = 30


class Pipeline:
    """Video karelerini alıp sayım, kayıt ve görselleştirmeye çeviren döngü."""

    def __init__(
        self,
        video_source: Any,
        detector: Any,
        tracker: Any,
        zones: ZoneManager,
        storage: Any,
        snapshot_interval_sec: float = 10.0,
    ):
        self.video_source = video_source
        self.detector = detector
        self.tracker = tracker
        self.zones = zones
        self.storage = storage
        self.snapshot_interval_sec = snapshot_interval_sec

        self.density_map: DensityMap | None = None
        self.error: BaseException | None = None
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_jpeg: bytes | None = None
        self._raw_frame: np.ndarray | None = None
        self._stats = LiveStats(ts=datetime.now(UTC))
        self._frame_times: deque[float] = deque(maxlen=FPS_PENCERESI)
        self._last_snapshot = 0.0

    # --- dışarıya açılan durum ------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_stats(self) -> LiveStats:
        with self._lock:
            return self._stats

    def latest_frame(self) -> bytes | None:
        """En son çizilmiş kare, JPEG olarak."""
        with self._lock:
            return self._frame_jpeg

    def heatmap_png(self) -> bytes:
        """Birikimli ısı haritası; varsa son kare arka plan olarak kullanılır."""
        with self._lock:
            harita = self.density_map
            arka_plan = self._raw_frame
        if harita is None:
            harita = DensityMap(shape=(480, 640))
        return harita.to_png_bytes(background=arka_plan)

    # --- yaşam döngüsü ---------------------------------------------------

    def start(self) -> None:
        """Hattı arka plan thread'inde başlatır."""
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._calis, name="occupancy-pipeline", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Hatta durmasını söyler ve thread'in bitmesini bekler."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def run_until_complete(self) -> None:
        """Hattı bu thread'de, kaynak bitene kadar çalıştırır (testler ve CLI için)."""
        self._calis()

    # --- döngü ------------------------------------------------------------

    def _calis(self) -> None:
        self.started_at = datetime.now(UTC)
        logger.info("İşleme hattı başladı.")
        try:
            for kare in self.video_source.frames():
                if self._stop.is_set():
                    logger.info("Durdurma istendi, hat kapanıyor.")
                    break
                self._kareyi_isle(kare)
        except BaseException as hata:  # noqa: BLE001 - hata dışarı bildirilecek
            self.error = hata
            logger.exception("İşleme hattı hata ile durdu: %s", hata)
        finally:
            self.finished_at = datetime.now(UTC)
            logger.info("İşleme hattı durdu.")

    def _kareyi_isle(self, kare: np.ndarray) -> None:
        simdi = datetime.now(UTC)

        tespitler = self.detector.detect(kare)
        izler = self.tracker.update(tespitler, kare)
        sayilar, olaylar = self.zones.update(izler, simdi)

        harita = self._isi_haritasi(kare.shape[:2])
        harita.add(iz.foot_point for iz in izler)

        yogunluklar = {
            ad: round(zone_density(sayi, self.zones.polygon_area(ad)), 4)
            for ad, sayi in sayilar.items()
        }

        if olaylar:
            self._guvenli_yaz(lambda: self.storage.add_events(olaylar), "olaylar")

        toplam_giris, toplam_cikis = self.zones.totals
        istatistik = LiveStats(
            ts=simdi,
            counts=sayilar,
            density=yogunluklar,
            total_in=toplam_giris,
            total_out=toplam_cikis,
            fps=self._fps_guncelle(),
            source_ok=getattr(self.video_source, "is_open", True),
        )

        self._anlik_goruntu_yaz(istatistik)

        cizilmis = draw_overlay(kare, izler, self.zones, istatistik)
        basarili, tampon = cv2.imencode(
            ".jpg", cizilmis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_KALITESI]
        )

        with self._lock:
            self._stats = istatistik
            self._raw_frame = kare
            if basarili:
                self._frame_jpeg = tampon.tobytes()

    def _isi_haritasi(self, shape: tuple[int, int]) -> DensityMap:
        if self.density_map is None or (self.density_map.height, self.density_map.width) != shape:
            self.density_map = DensityMap(shape=shape)
        return self.density_map

    def _fps_guncelle(self) -> float:
        self._frame_times.append(time.monotonic())
        if len(self._frame_times) < 2:
            return 0.0
        gecen = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / gecen if gecen > 0 else 0.0

    def _anlik_goruntu_yaz(self, stats: LiveStats) -> None:
        simdi = time.monotonic()
        if simdi - self._last_snapshot < self.snapshot_interval_sec:
            return
        self._last_snapshot = simdi

        for bolge, sayi in stats.counts.items():
            olcum = Snapshot(
                ts=stats.ts, zone=bolge, count=sayi, density=stats.density.get(bolge, 0.0)
            )
            self._guvenli_yaz(lambda o=olcum: self.storage.add_snapshot(o), "anlık görüntü")

    def _guvenli_yaz(self, islem, ne: str) -> None:
        """Kayıt hatalarını yutar: veritabanı düşse de izleme sürmeli."""
        try:
            islem()
        except Exception:
            logger.exception("Veritabanına %s yazılamadı, işleme devam ediyor.", ne)

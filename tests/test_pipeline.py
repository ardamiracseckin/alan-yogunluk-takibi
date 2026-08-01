"""İşleme hattı ve overlay testleri.

Gerçek model, gerçek kamera ve gerçek bekleme yok: sahte video kaynağı,
sahte detector ve sahte tracker ile hattın uçtan uca doğru davrandığı
sınanır.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import FakeDetector, FakeTracker, FakeVideoSource, kisi

from occupancy.models import LiveStats, Track
from occupancy.overlay import draw_overlay
from occupancy.pipeline import Pipeline
from occupancy.storage import Storage
from occupancy.zones import ZoneManager

TANIM = {
    "zones": [{"name": "salon", "polygon": [[100, 300], [600, 300], [600, 470], [100, 470]]}],
    "lines": [{"name": "kapi", "zone": "salon", "a": [100, 300], "b": [600, 300]}],
}


@pytest.fixture
def bolgeler():
    return ZoneManager.from_dict(TANIM)


@pytest.fixture
def depo(tmp_path):
    with Storage(tmp_path / "hat.db") as depo:
        yield depo


def _bekle(kosul, timeout: float = 5.0) -> bool:
    """Koşul sağlanana kadar kısa aralıklarla bekler."""
    bitis = time.monotonic() + timeout
    while time.monotonic() < bitis:
        if kosul():
            return True
        time.sleep(0.01)
    return False


def _hat(depo, bolgeler, kareler, frame_count=None, **kwargs) -> Pipeline:
    return Pipeline(
        video_source=FakeVideoSource(frame_count=frame_count or len(kareler)),
        detector=FakeDetector(kareler),
        tracker=FakeTracker(),
        zones=bolgeler,
        storage=depo,
        snapshot_interval_sec=0,
        **kwargs,
    )


# --- sayım ve kayıt ----------------------------------------------------


def test_cizgiyi_gecen_kisi_sayilir_ve_kaydedilir(depo, bolgeler):
    kareler = [[kisi(300, 250)], [kisi(300, 350)]]  # dışarıdan içeriye
    hat = _hat(depo, bolgeler, kareler)

    hat.run_until_complete()

    assert hat.latest_stats().total_in == 1
    olaylar = depo.events_between(hat.started_at, hat.finished_at)
    assert [o.event_type for o in olaylar] == ["enter"]


def test_bolgedeki_kisi_sayisi_yayinlanir(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[kisi(300, 400), kisi(400, 400)]], frame_count=2)

    hat.run_until_complete()

    assert hat.latest_stats().counts == {"salon": 2}


def test_anlik_goruntu_veritabanina_yazilir(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[kisi(300, 400)]], frame_count=2)

    hat.run_until_complete()

    olcumler = depo.snapshots_between(hat.started_at, hat.finished_at)
    assert olcumler and olcumler[0].zone == "salon"
    assert olcumler[0].count == 1


def test_yogunluk_hesaplanir(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[kisi(300, 400)]], frame_count=2)

    hat.run_until_complete()

    assert hat.latest_stats().density["salon"] > 0


def test_isi_haritasi_ayak_noktalarini_biriktirir(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[kisi(300, 400)]], frame_count=3)

    hat.run_until_complete()

    assert hat.density_map.raw.max() > 0


# --- dayanıklılık ------------------------------------------------------


class PatlayanDepo:
    """Her yazma denemesinde hata fırlatan depo."""

    def __init__(self):
        self.deneme = 0

    def add_events(self, events):
        self.deneme += 1
        raise RuntimeError("veritabanı düştü")

    def add_snapshot(self, snapshot):
        self.deneme += 1
        raise RuntimeError("veritabanı düştü")


def test_veritabani_hatasi_hatti_durdurmaz(bolgeler):
    patlayan = PatlayanDepo()
    kareler = [[kisi(300, 250)], [kisi(300, 350)], [kisi(300, 400)]]
    hat = Pipeline(
        video_source=FakeVideoSource(frame_count=3),
        detector=FakeDetector(kareler),
        tracker=FakeTracker(),
        zones=bolgeler,
        storage=patlayan,
        snapshot_interval_sec=0,
    )

    hat.run_until_complete()

    assert patlayan.deneme > 0  # yazmayı denedi
    assert hat.latest_stats().total_in == 1  # ve sayım yine de doğru
    assert hat.error is None


def test_detector_hatasi_hatti_dusurur_ama_sessizce_degil(depo, bolgeler):
    class PatlayanDetector:
        def detect(self, frame):
            raise RuntimeError("model çöktü")

    hat = Pipeline(
        video_source=FakeVideoSource(frame_count=2),
        detector=PatlayanDetector(),
        tracker=FakeTracker(),
        zones=bolgeler,
        storage=depo,
    )

    hat.run_until_complete()

    assert isinstance(hat.error, RuntimeError)
    assert hat.is_alive is False


# --- thread yaşam döngüsü ----------------------------------------------


def test_start_ve_stop_threadi_sonlandirir(depo, bolgeler):
    hat = Pipeline(
        video_source=FakeVideoSource(frame_count=10_000),
        detector=FakeDetector([[kisi(300, 400)]]),
        tracker=FakeTracker(),
        zones=bolgeler,
        storage=depo,
        snapshot_interval_sec=0,
    )

    hat.start()
    assert _bekle(lambda: hat.latest_frame() is not None)

    hat.stop()

    assert _bekle(lambda: hat.is_alive is False)


def test_calisirken_jpeg_karesi_yayinlanir(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[kisi(300, 400)]], frame_count=2)

    hat.run_until_complete()

    kare = hat.latest_frame()
    assert kare is not None
    assert kare[:3] == b"\xff\xd8\xff"  # JPEG imzası


def test_baslamadan_once_kare_yok(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[]], frame_count=1)

    assert hat.latest_frame() is None
    assert hat.is_alive is False


def test_isi_haritasi_png_uretir(depo, bolgeler):
    hat = _hat(depo, bolgeler, [[kisi(300, 400)]], frame_count=2)

    hat.run_until_complete()

    assert hat.heatmap_png().startswith(b"\x89PNG\r\n\x1a\n")


# --- overlay -----------------------------------------------------------


def test_overlay_kare_boyutunu_korur(bolgeler):
    kare = np.zeros((480, 640, 3), dtype=np.uint8)
    izler = [Track(track_id=1, bbox=(100, 200, 160, 400), confidence=0.9)]

    cizilmis = draw_overlay(kare, izler, bolgeler, LiveStats(ts=None))

    assert cizilmis.shape == kare.shape
    assert cizilmis.dtype == np.uint8


def test_overlay_orijinal_kareyi_degistirmez(bolgeler):
    kare = np.zeros((480, 640, 3), dtype=np.uint8)
    izler = [Track(track_id=1, bbox=(100, 200, 160, 400), confidence=0.9)]

    draw_overlay(kare, izler, bolgeler, LiveStats(ts=None))

    assert kare.max() == 0  # dokunulmadı


def test_overlay_bir_seyler_cizer(bolgeler):
    kare = np.zeros((480, 640, 3), dtype=np.uint8)
    izler = [Track(track_id=1, bbox=(100, 200, 160, 400), confidence=0.9)]

    cizilmis = draw_overlay(kare, izler, bolgeler, LiveStats(ts=None))

    assert cizilmis.max() > 0

"""Video kaynağı testleri.

Gerçek kamera veya dosya açmadan çalışır: OpenCV'nin VideoCapture'ı yerine
sahte bir sınıf enjekte edilir, bekleme fonksiyonu da enjekte edildiği için
testler gerçek zamanda beklemez.
"""

from __future__ import annotations

from itertools import islice

import cv2
import numpy as np
import pytest

from occupancy.video import VideoSource


class SahteCapture:
    """Belirli sayıda kare döndürüp biten sahte VideoCapture."""

    def __init__(self, kare_sayisi: int = 2, acilir: bool = True):
        self.kare_sayisi = kare_sayisi
        self._acilir = acilir
        self._indeks = 0
        self.serbest_birakildi = False
        self.set_cagrilari: list[tuple[int, float]] = []

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API'si
        return self._acilir

    def read(self):
        if not self._acilir or self._indeks >= self.kare_sayisi:
            return False, None
        kare = np.full((4, 4, 3), self._indeks, dtype=np.uint8)
        self._indeks += 1
        return True, kare

    def set(self, prop: int, deger: float) -> bool:  # noqa: A003
        self.set_cagrilari.append((prop, deger))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._indeks = int(deger)
        return True

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }.get(prop, 0.0)

    def release(self) -> None:
        self.serbest_birakildi = True


def _ilk_deger(kare) -> int:
    return int(kare[0, 0, 0])


def test_dosya_bitince_basa_sarar():
    capture = SahteCapture(kare_sayisi=2)
    kaynak = VideoSource("video.mp4", loop=True, capture_factory=lambda _: capture)

    kareler = [_ilk_deger(k) for k in islice(kaynak.frames(), 5)]

    assert kareler == [0, 1, 0, 1, 0]
    assert (cv2.CAP_PROP_POS_FRAMES, 0) in capture.set_cagrilari


def test_loop_kapaliyken_jenerator_biter():
    kaynak = VideoSource(
        "video.mp4", loop=False, capture_factory=lambda _: SahteCapture(kare_sayisi=2)
    )

    assert len(list(kaynak.frames())) == 2


def test_acilamayan_kaynakta_artan_bekleme():
    beklemeler: list[float] = []
    kaynak = VideoSource(
        "rtsp://yok",
        capture_factory=lambda _: SahteCapture(acilir=False),
        sleep=beklemeler.append,
        max_retries=6,
    )

    assert list(kaynak.frames()) == []
    assert beklemeler == [1, 2, 4, 8, 16, 30]


def test_bekleme_otuz_saniyede_tavan_yapar():
    beklemeler: list[float] = []
    kaynak = VideoSource(
        "rtsp://yok",
        capture_factory=lambda _: SahteCapture(acilir=False),
        sleep=beklemeler.append,
        max_retries=8,
    )

    list(kaynak.frames())

    assert beklemeler[-3:] == [30, 30, 30]


def test_acilamayan_kaynakta_is_open_false():
    kaynak = VideoSource(
        "rtsp://yok",
        capture_factory=lambda _: SahteCapture(acilir=False),
        sleep=lambda _: None,
        max_retries=1,
    )

    list(kaynak.frames())

    assert kaynak.is_open is False


def test_kare_okurken_is_open_true():
    kaynak = VideoSource("video.mp4", capture_factory=lambda _: SahteCapture(kare_sayisi=3))

    kareler = kaynak.frames()  # jeneratöre referans tutulmalı, yoksa kapanır
    next(kareler)

    assert kaynak.is_open is True


def test_jenerator_birakilinca_kaynak_kapanir():
    capture = SahteCapture(kare_sayisi=3)
    kaynak = VideoSource("video.mp4", capture_factory=lambda _: capture)

    kareler = kaynak.frames()
    next(kareler)
    kareler.close()

    assert capture.serbest_birakildi is True
    assert kaynak.is_open is False


def test_akis_koptugunda_yeniden_baglanir():
    """RTSP'de okuma hatası dosya sonu değil, kopma demektir: yeniden bağlanılır."""
    capture_gecmisi: list[SahteCapture] = []

    def fabrika(_):
        capture = SahteCapture(kare_sayisi=1)
        capture_gecmisi.append(capture)
        return capture

    kaynak = VideoSource(
        "rtsp://kamera", capture_factory=fabrika, sleep=lambda _: None, max_retries=3
    )

    kareler = list(islice(kaynak.frames(), 3))

    assert len(kareler) == 3
    assert len(capture_gecmisi) == 3  # her kopmada yeni bağlantı
    assert capture_gecmisi[0].serbest_birakildi is True


def test_webcam_indeksi_tamsayiya_cevrilir():
    verilen: list[object] = []

    def fabrika(kaynak_degeri):
        verilen.append(kaynak_degeri)
        return SahteCapture(kare_sayisi=1)

    VideoSource("0", capture_factory=fabrika).open()

    assert verilen == [0]


def test_dosya_yolu_oldugu_gibi_gecirilir():
    verilen: list[object] = []

    def fabrika(kaynak_degeri):
        verilen.append(kaynak_degeri)
        return SahteCapture(kare_sayisi=1)

    VideoSource("ornek/demo.mp4", capture_factory=fabrika).open()

    assert verilen == ["ornek/demo.mp4"]


def test_fps_ve_cozunurluk_okunur():
    kaynak = VideoSource("video.mp4", capture_factory=lambda _: SahteCapture())
    kaynak.open()

    assert kaynak.fps == pytest.approx(25.0)
    assert kaynak.resolution == (640, 480)


def test_close_capture_serbest_birakir():
    capture = SahteCapture()
    kaynak = VideoSource("video.mp4", capture_factory=lambda _: capture)
    kaynak.open()

    kaynak.close()

    assert capture.serbest_birakildi is True
    assert kaynak.is_open is False

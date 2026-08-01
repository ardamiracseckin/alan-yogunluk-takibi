"""Tespit ve izleme testleri.

`PersonTracker` gerçek ByteTrack algoritmasını kullanır ama model ağırlığı
gerektirmez: tespitler sentetik olarak verilir. `YoloPersonDetector` ise
enjekte edilen sahte bir model ile sınanır, böylece ağırlık indirilmez.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import kisi

from occupancy.detection import YoloPersonDetector
from occupancy.tracking import PersonTracker


class SahteKutular:
    """Ultralytics Results.boxes benzeri asgari nesne."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self


class SahteSonuc:
    def __init__(self, kutular):
        self.boxes = kutular


class SahteModel:
    """`predict` çağrısını kaydedip sabit sonuç döndüren sahte YOLO."""

    def __init__(self, sonuc):
        self._sonuc = sonuc
        self.predict_kwargs: dict = {}

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        return [self._sonuc]


# --- izleme ------------------------------------------------------------


def test_ayni_kisi_karelerde_ayni_kimligi_korur():
    izleyici = PersonTracker()

    kimlikler = []
    for kare in range(5):
        izler = izleyici.update([kisi(300 + kare * 8, 400)])
        kimlikler.extend(iz.track_id for iz in izler)

    assert len(set(kimlikler)) == 1


def test_iki_kisi_farkli_kimlik_alir():
    izleyici = PersonTracker()

    for _ in range(3):
        izler = izleyici.update([kisi(200, 400), kisi(600, 400)])

    assert len({iz.track_id for iz in izler}) == 2


def test_bos_tespit_listesi_bos_iz_dondurur():
    assert PersonTracker().update([]) == []


def test_dusuk_guvenli_tespit_yeni_iz_acmaz():
    izleyici = PersonTracker()

    for _ in range(3):
        izler = izleyici.update([kisi(300, 400, guven=0.05)])

    assert izler == []


def test_uzun_sure_kaybolan_kisi_yeni_kimlik_alir():
    izleyici = PersonTracker()

    for _ in range(3):
        ilk = izleyici.update([kisi(200, 400)])
    ilk_kimlik = ilk[0].track_id

    for _ in range(60):  # track_buffer'dan uzun süre kimse yok
        izleyici.update([])

    for _ in range(3):
        sonraki = izleyici.update([kisi(200, 400)])

    assert sonraki[0].track_id != ilk_kimlik


def test_iz_kutusu_tamsayi_ve_makul():
    izleyici = PersonTracker()

    for _ in range(3):
        izler = izleyici.update([kisi(300, 400)])

    x1, y1, x2, y2 = izler[0].bbox
    assert all(isinstance(deger, int) for deger in (x1, y1, x2, y2))
    assert x1 < x2 and y1 < y2


def test_reset_kimlikleri_sifirlar():
    izleyici = PersonTracker()
    for _ in range(3):
        izleyici.update([kisi(300, 400)])

    izleyici.reset()
    for _ in range(3):
        izler = izleyici.update([kisi(300, 400)])

    assert izler[0].track_id == 1


# --- tespit ------------------------------------------------------------


def test_detector_kutulari_detection_nesnesine_cevirir():
    model = SahteModel(SahteSonuc(SahteKutular([[10, 20, 50, 120]], [0.87], [0])))
    detector = YoloPersonDetector(model=model)

    tespitler = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(tespitler) == 1
    assert tespitler[0].bbox == (10, 20, 50, 120)
    assert tespitler[0].confidence == pytest.approx(0.87, abs=1e-6)


def test_detector_sadece_insan_sinifini_ister():
    model = SahteModel(SahteSonuc(SahteKutular(np.empty((0, 4)), [], [])))
    detector = YoloPersonDetector(model=model, conf=0.4)

    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert model.predict_kwargs["classes"] == [0]
    assert model.predict_kwargs["conf"] == pytest.approx(0.4)
    assert model.predict_kwargs["verbose"] is False


def test_detector_bos_sonucta_bos_liste_dondurur():
    model = SahteModel(SahteSonuc(SahteKutular(np.empty((0, 4)), [], [])))

    assert YoloPersonDetector(model=model).detect(np.zeros((4, 4, 3), np.uint8)) == []


def test_detector_sonuc_yoksa_bos_liste_dondurur():
    class BosModel:
        def predict(self, frame, **kwargs):
            return []

    assert YoloPersonDetector(model=BosModel()).detect(np.zeros((4, 4, 3), np.uint8)) == []

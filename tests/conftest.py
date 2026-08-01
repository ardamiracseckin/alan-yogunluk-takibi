"""Testler için ortak sahte bileşenler.

Ağır bağımlılıkları (YOLO ağırlığı, GPU, kamera) devre dışı bırakmak için
sistemin protokollerini karşılayan basit sahteler. Böylece tüm test paketi
ağ erişimi olmadan saniyeler içinde koşar.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from occupancy.models import Detection, Track


class FakeDetector:
    """Önceden verilmiş tespit listelerini kare kare döndürür.

    Liste bitince sonuncuyu tekrarlar; böylece testte kaç kare işleneceğini
    saymak zorunda kalmazsınız.
    """

    def __init__(self, frames: Sequence[Sequence[Detection]] | None = None):
        self._frames = [list(kareler) for kareler in (frames or [[]])]
        self.call_count = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        indeks = min(self.call_count, len(self._frames) - 1)
        self.call_count += 1
        return list(self._frames[indeks])


class FakeTracker:
    """Listedeki i. tespiti hep i+1 kimliğiyle eşleyen sahte izleyici.

    Gerçek eşleştirme yapmaz — testlerde kimliklerin ne olacağını tam olarak
    bilmek istediğimiz için kasıtlı olarak öngörülebilirdir.
    """

    def update(
        self, detections: Sequence[Detection], frame: np.ndarray | None = None
    ) -> list[Track]:
        return [
            Track(track_id=i + 1, bbox=tespit.bbox, confidence=tespit.confidence)
            for i, tespit in enumerate(detections)
        ]


class FakeVideoSource:
    """Belirli sayıda siyah kare üretip biten sahte video kaynağı."""

    def __init__(self, frame_count: int = 3, shape: tuple[int, int] = (480, 640)):
        self.frame_count = frame_count
        self.shape = shape
        self.closed = False

    @property
    def is_open(self) -> bool:
        return not self.closed

    def frames(self):
        for _ in range(self.frame_count):
            yield np.zeros((*self.shape, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def kisi(x: int, y: int, genislik: int = 40, yukseklik: int = 100, guven: float = 0.9) -> Detection:
    """Ayak noktası (x, y) olan bir kişi tespiti üretir."""
    return Detection(
        bbox=(x - genislik // 2, y - yukseklik, x + genislik // 2, y), confidence=guven
    )


@pytest.fixture
def bos_kare() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)

"""Aşama 3 — izleme (tracking).

Tespit, her karede "burada bir kişi var" der ama kişileri kareler arasında
birbirine bağlamaz. Giriş/çıkış saymak için kalıcı bir kimlik gerekir; onu
ByteTrack sağlıyor.

ByteTrack'i ultralytics'in `BYTETracker` sınıfından alıyoruz. O sınıf
Ultralytics `Results.boxes` benzeri bir nesne beklediği için burada küçük bir
uyarlayıcı (`_TespitSonuclari`) var. Kütüphanenin bu API'si sürümler arasında
değişebileceğinden, temas yüzeyi bilinçli olarak tek dosyada tutuldu:
`Tracker` protokolünü karşılayan başka bir uygulama yazmak sistemin geri
kalanını etkilemez.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Protocol, runtime_checkable

import numpy as np

from occupancy.logging_conf import get_logger
from occupancy.models import Detection, Track

logger = get_logger(__name__)


@runtime_checkable
class Tracker(Protocol):
    """Tespitleri kalıcı kimlikli izlere çeviren her şey."""

    def update(
        self, detections: Sequence[Detection], frame: np.ndarray | None = None
    ) -> list[Track]: ...


@dataclass(frozen=True)
class ByteTrackConfig:
    """ByteTrack eşikleri.

    Varsayılanlar ultralytics'in `bytetrack.yaml` dosyasındaki değerlerdir;
    burada açıkça yazılarak paket içi dosya düzenine bağımlılık kaldırıldı.
    """

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    def as_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


class _TespitSonuclari:
    """`BYTETracker.update` için asgari Results benzeri uyarlayıcı.

    Gereken yüzey: `xywh`, `conf`, `cls`, `len()` ve boolean maskeyle
    indeksleme.
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_detections(cls, detections: Sequence[Detection]) -> _TespitSonuclari:
        if not detections:
            bos = np.empty((0, 4), dtype=np.float32)
            return cls(bos, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        return cls(
            np.array([t.bbox for t in detections], dtype=np.float32),
            np.array([t.confidence for t in detections], dtype=np.float32),
            np.zeros(len(detections), dtype=np.float32),
        )

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> _TespitSonuclari:
        return _TespitSonuclari(self.xyxy[mask], self.conf[mask], self.cls[mask])


class PersonTracker:
    """ByteTrack ile kişilere kalıcı kimlik atar."""

    def __init__(self, config: ByteTrackConfig | None = None):
        self.config = config or ByteTrackConfig()
        self._tracker = self._yeni_tracker()

    def _yeni_tracker(self):
        from ultralytics.trackers.byte_tracker import BYTETracker

        return BYTETracker(self.config.as_namespace())

    def reset(self) -> None:
        """İzleyiciyi ve kimlik sayacını sıfırlar."""
        from ultralytics.trackers.basetrack import BaseTrack

        BaseTrack.reset_id()
        self._tracker = self._yeni_tracker()

    def update(
        self, detections: Sequence[Detection], frame: np.ndarray | None = None
    ) -> list[Track]:
        sonuc = self._tracker.update(_TespitSonuclari.from_detections(detections), frame)
        if sonuc is None or len(sonuc) == 0:
            return []

        izler = []
        for satir in np.asarray(sonuc, dtype=float):
            x1, y1, x2, y2, track_id, guven = satir[:6]
            izler.append(
                Track(
                    track_id=int(track_id),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(guven),
                )
            )
        return izler

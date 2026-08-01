"""Modüller arasında dolaşan veri sınıfları.

Bu dosya bilinçli olarak bağımlılıksızdır: hiçbir ağır kütüphaneyi (torch,
ultralytics, opencv) içeri almaz. Böylece testler ve raporlama tarafı
modelleri model indirmeden kullanabilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2
Point = tuple[int, int]

EventType = Literal["enter", "exit"]


@dataclass(frozen=True)
class Detection:
    """Tek bir karede bulunan kişi kutusu (henüz kimliği yok)."""

    bbox: BBox
    confidence: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class Track:
    """Kareler boyunca aynı kimlikle takip edilen kişi."""

    track_id: int
    bbox: BBox
    confidence: float

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def foot_point(self) -> Point:
        """Kişinin zemine bastığı nokta: kutunun alt kenarının ortası.

        Bölge analizinde kutunun merkezi yerine bu nokta kullanılır, çünkü
        kişinin hangi alanda *durduğu* zemine değdiği yerle belirlenir.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


@dataclass(frozen=True)
class ZoneEvent:
    """Bir kişinin bir bölgeye girmesi veya çıkması."""

    ts: datetime
    track_id: int
    event_type: EventType
    zone: str


@dataclass(frozen=True)
class Snapshot:
    """Bir anda bir bölgedeki doluluk ölçümü."""

    ts: datetime
    zone: str
    count: int
    density: float


@dataclass
class LiveStats:
    """Dashboard'a gönderilen anlık durum."""

    ts: datetime
    counts: dict[str, int] = field(default_factory=dict)
    density: dict[str, float] = field(default_factory=dict)
    total_in: int = 0
    total_out: int = 0
    fps: float = 0.0
    source_ok: bool = False

    @property
    def total_people(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "counts": self.counts,
            "density": self.density,
            "total_people": self.total_people,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "fps": round(self.fps, 1),
            "source_ok": self.source_ok,
        }

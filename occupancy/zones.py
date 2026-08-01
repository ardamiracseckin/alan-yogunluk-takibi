"""Aşama 4 — bölge analizi.

İki ayrı soruyu cevaplar:

1. **Kaç kişi içeride?** Her karede, kişilerin ayak noktası bölge
   poligonunun içinde mi diye bakılır. Bu anlık bir ölçümdür, geçmiş
   gerektirmez.
2. **Kim girdi, kim çıktı?** Bunun için kişinin bir önceki karedeki konumu
   gerekir: ayak noktasının izlediği doğru parçası kapı çizgisini kesiyorsa
   bir geçiş olmuştur. Yön, çizgiye göre işaret değişiminden bulunur.

Aynı kişinin aynı yönde iki kez sayılmaması kritik: kapıda duraksayan biri
sayacı şişirmemeli. Bu yüzden sayılmış her `(kişi, çizgi, yön)` üçlüsü
hatırlanır.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from occupancy.logging_conf import get_logger
from occupancy.models import Point, Track, ZoneEvent

logger = get_logger(__name__)


class ZoneConfigError(Exception):
    """Bölge tanım dosyası okunamadığında veya geçersiz olduğunda atılır."""


@dataclass(frozen=True)
class Zone:
    """İzlenen alan (ROI)."""

    name: str
    polygon: np.ndarray  # (N, 2) tamsayı

    @property
    def contour(self) -> np.ndarray:
        return self.polygon.reshape(-1, 1, 2).astype(np.int32)

    @property
    def area(self) -> float:
        """Poligon alanı (piksel²), ayakkabı bağı formülü."""
        x = self.polygon[:, 0].astype(float)
        y = self.polygon[:, 1].astype(float)
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def contains(self, point: Point) -> bool:
        """Nokta poligonun içinde mi? Kenar üstü içeride sayılır."""
        return cv2.pointPolygonTest(self.contour, (float(point[0]), float(point[1])), False) >= 0


@dataclass(frozen=True)
class CrossingLine:
    """Giriş/çıkış çizgisi.

    `a`'dan `b`'ye bakıldığında sağdan sola geçmek "giriş" sayılır; pratikte
    çizgiyi ROI'nin dışından içine doğru geçmek anlamına gelir.
    """

    name: str
    zone: str
    a: Point
    b: Point

    def side(self, point: Point) -> float:
        """Noktanın çizgiye göre işaretli tarafı (çapraz çarpım)."""
        (ax, ay), (bx, by) = self.a, self.b
        return (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)


def _yon(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segmentler_kesisiyor(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """İki doğru *parçası* kesişiyor mu (uç noktalar dahil değil)."""
    d1, d2 = _yon(q1, q2, p1), _yon(q1, q2, p2)
    d3, d4 = _yon(p1, p2, q1), _yon(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass
class ZoneManager:
    """Bölgeleri ve çizgileri yönetir, her karede sayıları ve olayları üretir."""

    zones: list[Zone]
    lines: list[CrossingLine]
    _previous: dict[int, Point] = field(default_factory=dict, init=False)
    _counted: set[tuple[int, str, str]] = field(default_factory=set, init=False)
    _total_in: int = field(default=0, init=False)
    _total_out: int = field(default=0, init=False)

    # --- yükleme --------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> ZoneManager:
        yol = Path(path)
        if not yol.exists():
            raise ZoneConfigError(f"Bölge tanım dosyası bulunamadı: {yol}")

        try:
            tanim = json.loads(yol.read_text(encoding="utf-8"))
        except json.JSONDecodeError as hata:
            raise ZoneConfigError(f"Bölge tanım dosyası okunamadı ({yol}): {hata}") from hata

        return cls.from_dict(tanim)

    @classmethod
    def from_dict(cls, tanim: dict) -> ZoneManager:
        ham_bolgeler = tanim.get("zones") or []
        if not ham_bolgeler:
            raise ZoneConfigError("Tanım dosyasında en az bir bölge olmalı.")

        bolgeler = []
        for ham in ham_bolgeler:
            ad = ham.get("name")
            noktalar = ham.get("polygon") or []
            if not ad:
                raise ZoneConfigError("Her bölgenin bir 'name' alanı olmalı.")
            if len(noktalar) < 3:
                raise ZoneConfigError(
                    f"'{ad}' bölgesinin poligonu en az 3 nokta içermeli, {len(noktalar)} verildi."
                )
            bolgeler.append(Zone(name=ad, polygon=np.array(noktalar, dtype=np.int32)))

        bolge_adlari = {bolge.name for bolge in bolgeler}
        cizgiler = []
        for ham in tanim.get("lines") or []:
            ad = ham.get("name")
            bolge = ham.get("zone")
            if bolge not in bolge_adlari:
                raise ZoneConfigError(
                    f"'{ad}' çizgisi tanımsız bir bölgeye bağlı: '{bolge}'. "
                    f"Tanımlı bölgeler: {', '.join(sorted(bolge_adlari))}"
                )
            cizgiler.append(
                CrossingLine(
                    name=ad,
                    zone=bolge,
                    a=(int(ham["a"][0]), int(ham["a"][1])),
                    b=(int(ham["b"][0]), int(ham["b"][1])),
                )
            )

        logger.info("%d bölge, %d çizgi yüklendi.", len(bolgeler), len(cizgiler))
        return cls(zones=bolgeler, lines=cizgiler)

    # --- sorgular -------------------------------------------------------

    @property
    def zone_names(self) -> list[str]:
        return [bolge.name for bolge in self.zones]

    @property
    def totals(self) -> tuple[int, int]:
        """(toplam giriş, toplam çıkış)."""
        return (self._total_in, self._total_out)

    def polygon_area(self, name: str) -> float:
        for bolge in self.zones:
            if bolge.name == name:
                return bolge.area
        raise KeyError(f"Bölge bulunamadı: {name}")

    # --- her kare -------------------------------------------------------

    def update(
        self, tracks: Sequence[Track], ts: datetime
    ) -> tuple[dict[str, int], list[ZoneEvent]]:
        """Bu karenin bölge sayılarını ve oluşan giriş/çıkış olaylarını döndürür."""
        sayilar = {bolge.name: 0 for bolge in self.zones}
        for iz in tracks:
            for bolge in self.zones:
                if bolge.contains(iz.foot_point):
                    sayilar[bolge.name] += 1

        olaylar = self._gecisleri_bul(tracks, ts)

        # Sadece bu karede görülen kişilerin konumu hatırlanır; kareden çıkan
        # birinin eski konumundan geçiş çıkarmak yanlış sayıma yol açar.
        self._previous = {iz.track_id: iz.foot_point for iz in tracks}

        return sayilar, olaylar

    def _gecisleri_bul(self, tracks: Sequence[Track], ts: datetime) -> list[ZoneEvent]:
        olaylar: list[ZoneEvent] = []
        for iz in tracks:
            onceki = self._previous.get(iz.track_id)
            if onceki is None:
                continue  # ilk kez görülüyor, geçiş çıkarılamaz
            simdiki = iz.foot_point

            for cizgi in self.lines:
                if not segmentler_kesisiyor(onceki, simdiki, cizgi.a, cizgi.b):
                    continue

                tur = "enter" if cizgi.side(simdiki) > 0 else "exit"
                anahtar = (iz.track_id, cizgi.name, tur)
                if anahtar in self._counted:
                    logger.debug("Tekrar geçiş yok sayıldı: %s", anahtar)
                    continue

                self._counted.add(anahtar)
                if tur == "enter":
                    self._total_in += 1
                else:
                    self._total_out += 1
                olaylar.append(
                    ZoneEvent(ts=ts, track_id=iz.track_id, event_type=tur, zone=cizgi.zone)
                )
        return olaylar

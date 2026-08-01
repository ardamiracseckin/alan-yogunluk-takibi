"""Kare üzerine bilgi bastırma.

Dashboard'da görülen her şey burada çizilir: bölge poligonları, kapı
çizgisi, kişi kutuları ve kimlikleri, üstte de özet panel.

OpenCV'nin `putText`'i Türkçe karakterleri düzgün çizemediği için etiketler
bilinçli olarak ASCII tutuldu ("Doluluk", "Giris", "Cikis").
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from occupancy.models import LiveStats, Track
from occupancy.zones import ZoneManager

YAZI_TIPI = cv2.FONT_HERSHEY_SIMPLEX

RENK_BOLGE = (255, 176, 0)  # BGR: turkuaz-mavi
RENK_CIZGI = (0, 215, 255)  # amber
RENK_KUTU = (80, 220, 80)  # yeşil
RENK_METIN = (255, 255, 255)
RENK_PANEL = (28, 28, 28)


def _panel_ciz(kare: np.ndarray, stats: LiveStats) -> None:
    yukseklik, genislik = kare.shape[:2]
    panel_yuksekligi = 64

    ortu = kare.copy()
    cv2.rectangle(ortu, (0, 0), (genislik, panel_yuksekligi), RENK_PANEL, -1)
    cv2.addWeighted(ortu, 0.75, kare, 0.25, 0, dst=kare)

    satir = (
        f"Doluluk: {stats.total_people}   "
        f"Giris: {stats.total_in}   "
        f"Cikis: {stats.total_out}   "
        f"FPS: {stats.fps:.1f}"
    )
    cv2.putText(kare, satir, (16, 40), YAZI_TIPI, 0.7, RENK_METIN, 2, cv2.LINE_AA)

    if not stats.source_ok:
        cv2.putText(
            kare,
            "KAYNAK BAGLANTISI YOK",
            (genislik - 330, 40),
            YAZI_TIPI,
            0.6,
            (60, 60, 255),
            2,
            cv2.LINE_AA,
        )


def _bolgeleri_ciz(kare: np.ndarray, zones: ZoneManager) -> None:
    ortu = kare.copy()
    for bolge in zones.zones:
        cv2.fillPoly(ortu, [bolge.contour], RENK_BOLGE)
    cv2.addWeighted(ortu, 0.18, kare, 0.82, 0, dst=kare)

    for bolge in zones.zones:
        cv2.polylines(kare, [bolge.contour], True, RENK_BOLGE, 2, cv2.LINE_AA)
        x, y = bolge.polygon[0]
        cv2.putText(kare, bolge.name, (int(x) + 6, int(y) + 22), YAZI_TIPI, 0.6, RENK_BOLGE, 2)


def _cizgileri_ciz(kare: np.ndarray, zones: ZoneManager) -> None:
    for cizgi in zones.lines:
        cv2.line(kare, cizgi.a, cizgi.b, RENK_CIZGI, 3, cv2.LINE_AA)
        orta = ((cizgi.a[0] + cizgi.b[0]) // 2, (cizgi.a[1] + cizgi.b[1]) // 2)
        cv2.putText(kare, cizgi.name, (orta[0] - 20, orta[1] - 10), YAZI_TIPI, 0.6, RENK_CIZGI, 2)


def _izleri_ciz(kare: np.ndarray, tracks: Sequence[Track]) -> None:
    for iz in tracks:
        x1, y1, x2, y2 = iz.bbox
        cv2.rectangle(kare, (x1, y1), (x2, y2), RENK_KUTU, 2)
        etiket = f"#{iz.track_id}"
        cv2.rectangle(kare, (x1, y1 - 22), (x1 + 12 * len(etiket), y1), RENK_KUTU, -1)
        cv2.putText(kare, etiket, (x1 + 3, y1 - 6), YAZI_TIPI, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(kare, iz.foot_point, 4, RENK_KUTU, -1)


def draw_overlay(
    frame: np.ndarray, tracks: Sequence[Track], zones: ZoneManager, stats: LiveStats
) -> np.ndarray:
    """Kareyi kopyalayıp üzerine tüm görsel bilgiyi çizer."""
    kare = frame.copy()
    _bolgeleri_ciz(kare, zones)
    _cizgileri_ciz(kare, zones)
    _izleri_ciz(kare, tracks)
    _panel_ciz(kare, stats)
    return kare

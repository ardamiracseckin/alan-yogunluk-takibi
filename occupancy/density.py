"""Aşama 4 (devamı) — yoğunluk ve ısı haritası.

İki ayrı ölçüm var, karıştırmamak gerekir:

* **Yoğunluk skoru**: o anda bölgede kaç kişi olduğunun alana bölünmüş hali.
  Zaman serisi olarak grafiklenir.
* **Isı haritası**: kişilerin *zaman içinde* nerede durduğunun birikimi.
  Anlık değil, kümülatif bir görseldir.

Isı haritası her karede bulanıklaştırılmaz: kareler boyunca sadece ayak
noktaları toplanır (ucuz), bulanıklaştırma yalnızca görsel istendiğinde bir
kez yapılır.

Not: yoğunluk piksel alanına göre hesaplanır, gerçek m²'ye göre değil. Metre
karşılığı kamera kalibrasyonu gerektirir ve bu projenin kapsamı dışındadır.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from occupancy.models import Point

REFERANS_ALAN_PIKSEL = 1000.0  # yoğunluk "1000 piksel² başına kişi" olarak verilir


def zone_density(count: int, area_px: float) -> float:
    """Bölgedeki kişi yoğunluğu: 1000 piksel² başına kişi sayısı."""
    if area_px <= 0 or count <= 0:
        return 0.0
    # float() şart: numpy skaları JSON'a sızıp API sözleşmesini bozuyor.
    return float(count / area_px * REFERANS_ALAN_PIKSEL)


class DensityMap:
    """Ayak noktalarının zaman içindeki birikimi (ısı haritası)."""

    def __init__(self, shape: tuple[int, int], sigma: float = 25.0):
        self.height, self.width = shape
        self.sigma = sigma
        self.raw = np.zeros((self.height, self.width), dtype=np.float32)

    def add(self, points: Iterable[Point]) -> None:
        """Verilen noktaların ısısını bir artırır; kare dışındakiler yok sayılır."""
        for x, y in points:
            if 0 <= int(y) < self.height and 0 <= int(x) < self.width:
                self.raw[int(y), int(x)] += 1.0

    def reset(self) -> None:
        self.raw.fill(0.0)

    def _normalize(self) -> np.ndarray:
        """Birikimi bulanıklaştırıp 0-255 aralığına getirir."""
        bulanik = cv2.GaussianBlur(self.raw, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        en_yuksek = float(bulanik.max())
        if en_yuksek <= 0:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return np.clip(bulanik / en_yuksek * 255.0, 0, 255).astype(np.uint8)

    def as_colormap(self, background: np.ndarray | None = None) -> np.ndarray:
        """Isı haritasını renkli BGR görüntü olarak döndürür.

        `background` verilirse (örneğin videodan bir kare) ısı haritası onun
        üstüne harmanlanır.
        """
        renkli = cv2.applyColorMap(self._normalize(), cv2.COLORMAP_JET)
        if background is None:
            return renkli

        if background.shape[:2] != (self.height, self.width):
            background = cv2.resize(background, (self.width, self.height))
        return cv2.addWeighted(background, 0.55, renkli, 0.45, 0)

    def to_png_bytes(self, background: np.ndarray | None = None) -> bytes:
        basarili, tampon = cv2.imencode(".png", self.as_colormap(background))
        if not basarili:
            raise RuntimeError("Isı haritası PNG olarak kodlanamadı.")
        return tampon.tobytes()

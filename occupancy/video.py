"""Aşama 1 — video girişi.

Tek bir sınıf üç kaynağı da idare eder: yerel dosya, webcam ve RTSP/HTTP
akışı. Aradaki fark davranışta ortaya çıkar:

* Dosyada okuma hatası "video bitti" demektir; `loop=True` ise başa sarılır.
* Akışta okuma hatası "bağlantı koptu" demektir; artan beklemeyle yeniden
  bağlanılır.

`capture_factory` ve `sleep` dışarıdan verilebildiği için sınıf gerçek kamera
veya gerçek bekleme olmadan test edilebilir.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import cv2
import numpy as np

from occupancy.logging_conf import get_logger

logger = get_logger(__name__)

AKIS_ONEKLERI = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
ILK_BEKLEME_SN = 1.0
MAKS_BEKLEME_SN = 30.0


class VideoSource:
    """Kareleri tek tek veren, koptuğunda kendini toparlayan video kaynağı."""

    def __init__(
        self,
        source: str,
        loop: bool = True,
        capture_factory: Callable[[Any], Any] = cv2.VideoCapture,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int | None = None,
    ):
        self.source = source
        self.loop = loop
        self._capture_factory = capture_factory
        self._sleep = sleep
        self._max_retries = max_retries
        self._capture: Any | None = None
        self._is_open = False

    # --- kaynak türü ---------------------------------------------------

    @property
    def is_stream(self) -> bool:
        return self.source.lower().startswith(AKIS_ONEKLERI)

    @property
    def is_webcam(self) -> bool:
        return self.source.isdigit()

    @property
    def is_file(self) -> bool:
        return not self.is_stream and not self.is_webcam

    def _capture_argumani(self) -> Any:
        return int(self.source) if self.is_webcam else self.source

    # --- durum ---------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def fps(self) -> float:
        if self._capture is None:
            return 0.0
        return float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def resolution(self) -> tuple[int, int]:
        if self._capture is None:
            return (0, 0)
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # --- açma / kapama --------------------------------------------------

    def open(self) -> bool:
        """Kaynağı bir kez açmayı dener."""
        self.close()
        capture = self._capture_factory(self._capture_argumani())
        if not capture.isOpened():
            capture.release()
            self._is_open = False
            return False
        self._capture = capture
        self._is_open = True
        return True

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_open = False

    def _artan_beklemeyle_ac(self) -> bool:
        """Açılana kadar 1, 2, 4, ... 30 saniye bekleyerek dener."""
        if self.open():
            return True

        bekleme = ILK_BEKLEME_SN
        deneme = 0
        while self._max_retries is None or deneme < self._max_retries:
            logger.warning(
                "Video kaynağı açılamadı (%s). %.0f saniye sonra tekrar denenecek.",
                self.source,
                bekleme,
            )
            self._sleep(bekleme)
            deneme += 1
            if self.open():
                logger.info("Video kaynağına yeniden bağlanıldı: %s", self.source)
                return True
            bekleme = min(bekleme * 2, MAKS_BEKLEME_SN)

        logger.error("Video kaynağına bağlanılamadı, vazgeçildi: %s", self.source)
        return False

    # --- kareler --------------------------------------------------------

    def frames(self) -> Iterator[np.ndarray]:
        """Kareleri sonsuza kadar (ya da dosya bitene kadar) üretir."""
        try:
            while True:
                if self._capture is None and not self._artan_beklemeyle_ac():
                    return

                okundu, kare = self._capture.read()
                if okundu:
                    yield kare
                    continue

                if self.is_file:
                    if not self.loop:
                        logger.info("Video dosyası bitti: %s", self.source)
                        return
                    logger.debug("Video başa sarılıyor: %s", self.source)
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    okundu, kare = self._capture.read()
                    if not okundu:
                        logger.error("Video başa sarılamadı: %s", self.source)
                        return
                    yield kare
                    continue

                logger.warning("Akış koptu, yeniden bağlanılıyor: %s", self.source)
                self.close()
        finally:
            self.close()

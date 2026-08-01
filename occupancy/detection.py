"""Aşama 2 — kişi tespiti.

Sistemin geri kalanı somut modeli değil `PersonDetector` protokolünü tanır.
Bu sayede testler YOLO ağırlığı indirmeden sahte bir detector kullanabilir ve
model değiştirmek tek dosyayı ilgilendirir.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from occupancy.logging_conf import get_logger
from occupancy.models import Detection

logger = get_logger(__name__)

INSAN_SINIF_ID = 0  # COCO veri setinde "person"


@runtime_checkable
class PersonDetector(Protocol):
    """Bir karedeki kişileri kutu olarak döndüren her şey."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloPersonDetector:
    """Ultralytics YOLO ile kişi tespiti.

    Model ilk `detect` çağrısında yüklenir (import maliyetini uygulamanın
    açılışına yaymamak için). Testlerde hazır bir `model` enjekte edilebilir.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        device: str | None = None,
        model: Any | None = None,
    ):
        self.model_path = model_path
        self.conf = conf
        self.device = device
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # ağır import: sadece gerçekten gerekince

            logger.info("YOLO modeli yükleniyor: %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        sonuclar = self.model.predict(
            frame,
            classes=[INSAN_SINIF_ID],
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not sonuclar:
            return []

        kutular = sonuclar[0].boxes
        if kutular is None:
            return []

        xyxy = np.asarray(kutular.xyxy, dtype=float).reshape(-1, 4)
        if xyxy.size == 0:
            return []
        guvenler = np.asarray(kutular.conf, dtype=float).reshape(-1)

        return [
            Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(guven),
            )
            for (x1, y1, x2, y2), guven in zip(xyxy, guvenler, strict=False)
        ]

"""İnteraktif bölge (ROI) çizim aracı.

Bölge koordinatlarını elle JSON'a yazmak hem yorucu hem hataya açık. Bu araç
videonun ilk karesini açar, fareyle poligon ve geçiş çizgisi çizdirir ve
`ZoneManager`'ın doğrudan okuyabileceği dosyayı üretir.

    python tools/roi_ciz.py --source ornek/demo.mp4 --out ornek/zones.json

Tuşlar:
    sol tık   poligona köşe ekler
    n         poligonu kapatır ve bölge adını sorar
    l         çizgi modu: iki tık ile giriş/çıkış çizgisi
    z         son işlemi geri alır
    s         kaydeder
    q         çıkar

Çizim mantığı (`RoiBuilder`) pencereden ayrıdır: test edilen kısım odur,
OpenCV penceresi yalnızca ince bir kabuk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Nokta = tuple[int, int]

RENK_BEKLEYEN = (0, 215, 255)
RENK_BOLGE = (255, 176, 0)
RENK_CIZGI = (60, 220, 60)
RENK_METIN = (255, 255, 255)
YAZI_TIPI = cv2.FONT_HERSHEY_SIMPLEX

YARDIM = [
    "sol tik: kose ekle | n: bolgeyi kapat | l: cizgi modu",
    "z: geri al | s: kaydet | q: cikis",
]


@dataclass
class RoiBuilder:
    """Tıklanan noktaları biriktirip `zones.json` tanımına çevirir."""

    zones: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    pending: list[Nokta] = field(default_factory=list)

    # --- poligon ---------------------------------------------------------

    def add_point(self, x: int, y: int) -> None:
        self.pending.append((int(x), int(y)))

    def close_zone(self, name: str) -> bool:
        """Bekleyen noktaları bir bölgeye çevirir. Başarılıysa True döner."""
        if len(self.pending) < 3:
            return False
        if any(bolge["name"] == name for bolge in self.zones):
            return False
        self.zones.append({"name": name, "polygon": [list(nokta) for nokta in self.pending]})
        self.pending = []
        return True

    # --- çizgi -----------------------------------------------------------

    def add_line(self, name: str, zone: str, a: Nokta, b: Nokta) -> None:
        if not any(bolge["name"] == zone for bolge in self.zones):
            raise ValueError(f"Tanımsız bölgeye çizgi eklenemez: {zone!r}")
        self.lines.append({"name": name, "zone": zone, "a": list(a), "b": list(b)})

    # --- düzenleme -------------------------------------------------------

    def undo(self) -> None:
        """Sırasıyla: bekleyen nokta, sonra son çizgi, sonra son bölge."""
        if self.pending:
            self.pending.pop()
        elif self.lines:
            self.lines.pop()
        elif self.zones:
            self.zones.pop()

    # --- çıktı -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {"zones": self.zones, "lines": self.lines}

    def save(self, path: str | Path) -> Path:
        if not self.zones:
            raise ValueError("Kaydetmek için en az bir bölge tanımlanmalı.")
        yol = Path(path)
        yol.parent.mkdir(parents=True, exist_ok=True)
        yol.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return yol


# --- pencere ------------------------------------------------------------


def ilk_kareyi_al(source: str) -> np.ndarray:
    """Kaynağın ilk karesini okur."""
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    try:
        if not capture.isOpened():
            raise SystemExit(f"HATA: Video kaynağı açılamadı: {source}")
        okundu, kare = capture.read()
        if not okundu:
            raise SystemExit(f"HATA: Kaynaktan kare okunamadı: {source}")
        return kare
    finally:
        capture.release()


def _ciz(kare: np.ndarray, olusturucu: RoiBuilder, cizgi_modu: bool) -> np.ndarray:
    tuval = kare.copy()

    for bolge in olusturucu.zones:
        noktalar = np.array(bolge["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(tuval, [noktalar], True, RENK_BOLGE, 2, cv2.LINE_AA)
        x, y = bolge["polygon"][0]
        cv2.putText(tuval, bolge["name"], (x + 6, y + 24), YAZI_TIPI, 0.7, RENK_BOLGE, 2)

    for cizgi in olusturucu.lines:
        cv2.line(tuval, tuple(cizgi["a"]), tuple(cizgi["b"]), RENK_CIZGI, 3, cv2.LINE_AA)

    for i, nokta in enumerate(olusturucu.pending):
        cv2.circle(tuval, nokta, 5, RENK_BEKLEYEN, -1)
        if i:
            cv2.line(tuval, olusturucu.pending[i - 1], nokta, RENK_BEKLEYEN, 2, cv2.LINE_AA)

    ortu = tuval.copy()
    cv2.rectangle(ortu, (0, 0), (tuval.shape[1], 70), (25, 25, 25), -1)
    cv2.addWeighted(ortu, 0.7, tuval, 0.3, 0, dst=tuval)
    for i, satir in enumerate(YARDIM):
        cv2.putText(tuval, satir, (14, 26 + i * 24), YAZI_TIPI, 0.55, RENK_METIN, 1, cv2.LINE_AA)
    if cizgi_modu:
        cv2.putText(
            tuval,
            "CIZGI MODU: iki nokta tikla",
            (tuval.shape[1] - 380, 26),
            YAZI_TIPI,
            0.6,
            RENK_CIZGI,
            2,
        )
    return tuval


def _sor(soru: str) -> str:
    """Terminalden metin ister (OpenCV penceresinde metin girişi yok)."""
    try:
        return input(soru).strip()
    except EOFError:
        return ""


def calistir(source: str, out: str) -> int:  # pragma: no cover - etkileşimli
    kare = ilk_kareyi_al(source)
    olusturucu = RoiBuilder()
    cizgi_noktalari: list[Nokta] = []
    cizgi_modu = False
    pencere = "ROI ciz"

    def fare(olay, x, y, bayraklar, veri):
        nonlocal cizgi_modu
        if olay != cv2.EVENT_LBUTTONDOWN:
            return
        if cizgi_modu:
            cizgi_noktalari.append((x, y))
            if len(cizgi_noktalari) == 2:
                a, b = cizgi_noktalari
                cizgi_noktalari.clear()
                cizgi_modu = False
                ad = _sor("Cizgi adi: ") or f"cizgi{len(olusturucu.lines) + 1}"
                bolge = _sor(f"Bagli bolge {[z['name'] for z in olusturucu.zones]}: ")
                try:
                    olusturucu.add_line(ad, bolge, a, b)
                except ValueError as hata:
                    print(f"  ! {hata}")
        else:
            olusturucu.add_point(x, y)

    cv2.namedWindow(pencere, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(pencere, min(1280, kare.shape[1]), min(720, kare.shape[0]))
    cv2.setMouseCallback(pencere, fare)
    print(__doc__)

    while True:
        cv2.imshow(pencere, _ciz(kare, olusturucu, cizgi_modu))
        tus = cv2.waitKey(20) & 0xFF

        if tus == ord("q"):
            break
        if tus == ord("n"):
            ad = _sor("Bolge adi: ") or f"bolge{len(olusturucu.zones) + 1}"
            if not olusturucu.close_zone(ad):
                print("  ! Bolge kapatilamadi (en az 3 nokta ve benzersiz ad gerekli).")
        elif tus == ord("l"):
            if not olusturucu.zones:
                print("  ! Once bir bolge tanimlayin.")
            else:
                cizgi_modu = True
                cizgi_noktalari.clear()
        elif tus == ord("z"):
            olusturucu.undo()
        elif tus == ord("s"):
            try:
                yol = olusturucu.save(out)
            except ValueError as hata:
                print(f"  ! {hata}")
            else:
                print(f"  Kaydedildi: {yol}")

    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - etkileşimli
    ayristirici = argparse.ArgumentParser(
        prog="python tools/roi_ciz.py",
        description="Video karesi üzerinde bölge ve giriş/çıkış çizgisi tanımlar.",
    )
    ayristirici.add_argument("--source", default="ornek/demo.mp4", help="Video dosyası veya kamera")
    ayristirici.add_argument("--out", default="ornek/zones.json", help="Yazılacak JSON dosyası")
    argumanlar = ayristirici.parse_args(argv)
    return calistir(argumanlar.source, argumanlar.out)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

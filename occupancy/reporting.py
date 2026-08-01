"""Aşama 7 — raporlama.

Veritabanındaki ham olaylardan ve doluluk ölçümlerinden bir günün özetini
çıkarır: kaç kişi girdi, kaç kişi çıktı, hangi saatte en kalabalıktı.

Gün sınırları UTC'ye göredir; raporun `date` alanı hangi günü kapsadığını
açıkça yazar.

Komut satırından da çalışır:

    python -m occupancy.reporting --date 2026-08-01
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from occupancy.logging_conf import get_logger
from occupancy.models import Snapshot, ZoneEvent
from occupancy.storage import Storage

logger = get_logger(__name__)

CSV_BASLIK = ["saat", "giris", "cikis", "ortalama_doluluk", "pik_doluluk"]


def _gun_araligi(day: date) -> tuple[datetime, datetime]:
    baslangic = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return baslangic, baslangic + timedelta(days=1)


def _ortalama(degerler: list[float]) -> float:
    return round(sum(degerler) / len(degerler), 2) if degerler else 0.0


def build_report(storage: Storage, day: date) -> dict:
    """Bir günün özet raporunu sözlük olarak üretir."""
    baslangic, bitis = _gun_araligi(day)
    olaylar = storage.events_between(baslangic, bitis)
    olcumler = storage.snapshots_between(baslangic, bitis)

    return {
        "date": day.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "totals": _toplamlar(olaylar),
        "zones": _bolge_ozetleri(olaylar, olcumler),
        "hourly": _saatlik_dokum(olaylar, olcumler),
    }


def _toplamlar(olaylar: list[ZoneEvent]) -> dict[str, int]:
    giris = sum(1 for o in olaylar if o.event_type == "enter")
    cikis = sum(1 for o in olaylar if o.event_type == "exit")
    return {"entries": giris, "exits": cikis, "net": giris - cikis}


def _bolge_ozetleri(olaylar: list[ZoneEvent], olcumler: list[Snapshot]) -> dict[str, dict]:
    bolgeler = sorted({o.zone for o in olaylar} | {o.zone for o in olcumler})
    ozetler: dict[str, dict] = {}

    for bolge in bolgeler:
        bolge_olaylari = [o for o in olaylar if o.zone == bolge]
        bolge_olcumleri = [o for o in olcumler if o.zone == bolge]
        sayilar = [o.count for o in bolge_olcumleri]

        en_kalabalik = max(bolge_olcumleri, key=lambda o: o.count, default=None)
        ozetler[bolge] = {
            "entries": sum(1 for o in bolge_olaylari if o.event_type == "enter"),
            "exits": sum(1 for o in bolge_olaylari if o.event_type == "exit"),
            "average_count": _ortalama(sayilar),
            "peak_count": max(sayilar, default=0),
            "peak_hour": en_kalabalik.ts.astimezone(UTC).hour if en_kalabalik else None,
            "average_density": _ortalama([o.density for o in bolge_olcumleri]),
        }
    return ozetler


def _saatlik_dokum(olaylar: list[ZoneEvent], olcumler: list[Snapshot]) -> list[dict]:
    giris: dict[int, int] = defaultdict(int)
    cikis: dict[int, int] = defaultdict(int)
    sayilar: dict[int, list[int]] = defaultdict(list)

    for olay in olaylar:
        saat = olay.ts.astimezone(UTC).hour
        if olay.event_type == "enter":
            giris[saat] += 1
        else:
            cikis[saat] += 1

    for olcum in olcumler:
        sayilar[olcum.ts.astimezone(UTC).hour].append(olcum.count)

    return [
        {
            "hour": saat,
            "entries": giris[saat],
            "exits": cikis[saat],
            "average_count": _ortalama(sayilar[saat]),
            "peak_count": max(sayilar[saat], default=0),
        }
        for saat in range(24)
    ]


def report_to_csv(report: dict) -> str:
    """Raporun saatlik dökümünü CSV metnine çevirir."""
    tampon = io.StringIO()
    yazici = csv.writer(tampon)
    yazici.writerow(CSV_BASLIK)
    for satir in report["hourly"]:
        yazici.writerow(
            [
                satir["hour"],
                satir["entries"],
                satir["exits"],
                satir["average_count"],
                satir["peak_count"],
            ]
        )
    return tampon.getvalue()


def write_report(report: dict, reports_dir: str | Path) -> tuple[Path, Path]:
    """Raporu JSON ve CSV olarak diske yazar, yollarını döndürür."""
    dizin = Path(reports_dir)
    dizin.mkdir(parents=True, exist_ok=True)

    json_yolu = dizin / f"{report['date']}.json"
    csv_yolu = dizin / f"{report['date']}.csv"

    json_yolu.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_yolu.write_text(report_to_csv(report), encoding="utf-8")

    logger.info("Rapor yazıldı: %s ve %s", json_yolu, csv_yolu)
    return json_yolu, csv_yolu


def main(argv: list[str] | None = None) -> int:
    import argparse

    from occupancy.config import Settings

    ayarlar = Settings()
    ayristirici = argparse.ArgumentParser(
        prog="python -m occupancy.reporting",
        description="Belirtilen gün için JSON ve CSV rapor üretir.",
    )
    ayristirici.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Rapor günü (YYYY-AA-GG). Varsayılan: bugün.",
    )
    ayristirici.add_argument("--db", default=str(ayarlar.db_path), help="Veritabanı yolu.")
    ayristirici.add_argument(
        "--out", default=str(ayarlar.reports_dir), help="Raporların yazılacağı dizin."
    )
    argumanlar = ayristirici.parse_args(argv)

    with Storage(argumanlar.db) as depo:
        rapor = build_report(depo, date.fromisoformat(argumanlar.date))

    json_yolu, csv_yolu = write_report(rapor, argumanlar.out)
    print(f"Rapor yazıldı:\n  {json_yolu}\n  {csv_yolu}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

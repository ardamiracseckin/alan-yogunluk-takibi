"""Günlük rapor üretimi testleri."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, date, datetime, timedelta

import pytest

from occupancy.models import Snapshot, ZoneEvent
from occupancy.reporting import build_report, report_to_csv, write_report
from occupancy.storage import Storage

GUN = date(2026, 8, 1)
GUN_BASI = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)


@pytest.fixture
def depo(tmp_path):
    with Storage(tmp_path / "rapor.db") as depo:
        yield depo


def _olay(saat: int, tur: str = "enter", track_id: int = 1, bolge: str = "salon") -> ZoneEvent:
    return ZoneEvent(
        ts=GUN_BASI + timedelta(hours=saat), track_id=track_id, event_type=tur, zone=bolge
    )


def _anlik(saat: int, sayi: int, bolge: str = "salon", yogunluk: float = 0.5) -> Snapshot:
    return Snapshot(
        ts=GUN_BASI + timedelta(hours=saat), zone=bolge, count=sayi, density=yogunluk
    )


@pytest.fixture
def dolu_depo(depo):
    depo.add_events(
        [
            _olay(9, "enter", 1),
            _olay(9, "enter", 2),
            _olay(14, "enter", 3),
            _olay(17, "exit", 1),
        ]
    )
    for saat, sayi in [(9, 2), (10, 3), (14, 9), (17, 1)]:
        depo.add_snapshot(_anlik(saat, sayi))
    return depo


def test_toplam_giris_cikis(dolu_depo):
    rapor = build_report(dolu_depo, GUN)

    assert rapor["totals"]["entries"] == 3
    assert rapor["totals"]["exits"] == 1
    assert rapor["totals"]["net"] == 2


def test_pik_doluluk_ve_saati(dolu_depo):
    rapor = build_report(dolu_depo, GUN)

    assert rapor["zones"]["salon"]["peak_count"] == 9
    assert rapor["zones"]["salon"]["peak_hour"] == 14


def test_ortalama_doluluk(dolu_depo):
    rapor = build_report(dolu_depo, GUN)

    assert rapor["zones"]["salon"]["average_count"] == pytest.approx((2 + 3 + 9 + 1) / 4)


def test_bolge_bazinda_giris_cikis(dolu_depo):
    salon = build_report(dolu_depo, GUN)["zones"]["salon"]

    assert (salon["entries"], salon["exits"]) == (3, 1)


def test_saatlik_dokum_yirmi_dort_satir(dolu_depo):
    rapor = build_report(dolu_depo, GUN)

    assert len(rapor["hourly"]) == 24
    assert [s["hour"] for s in rapor["hourly"]] == list(range(24))


def test_saatlik_dokum_dogru_saate_yazilir(dolu_depo):
    saatlik = {s["hour"]: s for s in build_report(dolu_depo, GUN)["hourly"]}

    assert saatlik[9]["entries"] == 2
    assert saatlik[17]["exits"] == 1
    assert saatlik[14]["peak_count"] == 9
    assert saatlik[3]["entries"] == 0


def test_onceki_ve_sonraki_gun_rapora_girmez(depo):
    depo.add_events([_olay(-1), _olay(25)])  # bir gün önce ve bir gün sonra

    rapor = build_report(depo, GUN)

    assert rapor["totals"]["entries"] == 0


def test_veri_olmayan_gun_gecerli_bos_rapor_uretir(depo):
    rapor = build_report(depo, GUN)

    assert rapor["date"] == "2026-08-01"
    assert rapor["totals"] == {"entries": 0, "exits": 0, "net": 0}
    assert rapor["zones"] == {}
    assert len(rapor["hourly"]) == 24


def test_rapor_json_serilestirilebilir(dolu_depo):
    json.dumps(build_report(dolu_depo, GUN))  # hata atmamalı


def test_csv_basligi_ve_satir_sayisi(dolu_depo):
    metin = report_to_csv(build_report(dolu_depo, GUN))

    satirlar = list(csv.reader(io.StringIO(metin)))

    assert satirlar[0] == ["saat", "giris", "cikis", "ortalama_doluluk", "pik_doluluk"]
    assert len(satirlar) == 25  # başlık + 24 saat


def test_csv_degerleri_dogru(dolu_depo):
    satirlar = list(csv.reader(io.StringIO(report_to_csv(build_report(dolu_depo, GUN)))))
    dokuz = next(s for s in satirlar[1:] if s[0] == "9")

    assert dokuz[1] == "2"  # giriş
    assert dokuz[4] == "2"  # pik doluluk


def test_write_report_iki_dosya_uretir(dolu_depo, tmp_path):
    json_yolu, csv_yolu = write_report(build_report(dolu_depo, GUN), tmp_path / "raporlar")

    assert json_yolu.name == "2026-08-01.json"
    assert csv_yolu.name == "2026-08-01.csv"
    assert json.loads(json_yolu.read_text(encoding="utf-8"))["date"] == "2026-08-01"
    assert csv_yolu.read_text(encoding="utf-8").startswith("saat,")

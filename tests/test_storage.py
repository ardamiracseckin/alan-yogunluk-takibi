"""SQLite depolama katmanı testleri."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from occupancy.models import Snapshot, ZoneEvent
from occupancy.storage import Storage

BASLANGIC = datetime(2026, 8, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def depo(tmp_path):
    with Storage(tmp_path / "veri" / "test.db") as depo:
        yield depo


def _olay(dakika: int, track_id: int = 1, tur: str = "enter") -> ZoneEvent:
    return ZoneEvent(
        ts=BASLANGIC + timedelta(minutes=dakika), track_id=track_id, event_type=tur, zone="salon"
    )


def _anlik(dakika: int, sayi: int = 3) -> Snapshot:
    return Snapshot(
        ts=BASLANGIC + timedelta(minutes=dakika), zone="salon", count=sayi, density=0.5
    )


def test_veritabani_dosyasi_ve_dizini_olusturulur(tmp_path):
    yol = tmp_path / "yeni" / "alt" / "test.db"

    with Storage(yol):
        pass

    assert yol.exists()


def test_olay_yaz_oku_roundtrip(depo):
    depo.add_events([_olay(0), _olay(5, track_id=2, tur="exit")])

    okunan = depo.events_between(BASLANGIC, BASLANGIC + timedelta(hours=1))

    assert len(okunan) == 2
    assert okunan[0].track_id == 1
    assert okunan[0].event_type == "enter"
    assert okunan[0].zone == "salon"
    assert okunan[0].ts == BASLANGIC
    assert okunan[1].event_type == "exit"


def test_anlik_goruntu_yaz_oku_roundtrip(depo):
    depo.add_snapshot(_anlik(0, sayi=7))

    okunan = depo.snapshots_between(BASLANGIC, BASLANGIC + timedelta(hours=1))

    assert len(okunan) == 1
    assert okunan[0].count == 7
    assert okunan[0].density == pytest.approx(0.5)
    assert okunan[0].zone == "salon"


def test_aralik_baslangici_dahil_bitisi_haric(depo):
    depo.add_events([_olay(0), _olay(10), _olay(20)])

    okunan = depo.events_between(BASLANGIC, BASLANGIC + timedelta(minutes=20))

    assert [o.ts for o in okunan] == [BASLANGIC, BASLANGIC + timedelta(minutes=10)]


def test_aralik_disindaki_kayit_gelmez(depo):
    depo.add_events([_olay(0)])

    assert depo.events_between(BASLANGIC + timedelta(hours=1), BASLANGIC + timedelta(hours=2)) == []


def test_bos_olay_listesi_hata_vermez(depo):
    depo.add_events([])

    assert depo.events_between(BASLANGIC, BASLANGIC + timedelta(hours=1)) == []


def test_sonuclar_zaman_sirali_doner(depo):
    depo.add_events([_olay(20), _olay(0), _olay(10)])

    okunan = depo.events_between(BASLANGIC, BASLANGIC + timedelta(hours=1))

    assert [o.ts for o in okunan] == sorted(o.ts for o in okunan)


def test_sema_iki_kez_olusturulabilir(depo):
    depo.init_schema()
    depo.init_schema()

    depo.add_events([_olay(0)])
    assert len(depo.events_between(BASLANGIC, BASLANGIC + timedelta(hours=1))) == 1


def test_gecersiz_olay_turu_veritabanina_giremez(depo):
    bozuk = ZoneEvent(ts=BASLANGIC, track_id=1, event_type="ziplama", zone="salon")

    with pytest.raises(sqlite3.IntegrityError):
        depo.add_events([bozuk])


def test_farkli_threadlerden_yazma_hata_vermez(depo):
    def yaz(offset: int):
        depo.add_events([_olay(offset, track_id=offset)])

    threadler = [threading.Thread(target=yaz, args=(i,)) for i in range(1, 21)]
    for t in threadler:
        t.start()
    for t in threadler:
        t.join()

    assert len(depo.events_between(BASLANGIC, BASLANGIC + timedelta(hours=1))) == 20


def test_kapandiktan_sonra_yeniden_acilabilir(tmp_path):
    yol = tmp_path / "kalici.db"
    with Storage(yol) as depo:
        depo.add_events([_olay(0)])

    with Storage(yol) as depo:
        assert len(depo.events_between(BASLANGIC, BASLANGIC + timedelta(hours=1))) == 1

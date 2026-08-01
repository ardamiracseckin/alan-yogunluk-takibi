"""Bölge analizi ve giriş/çıkış sayımı testleri.

Bu modülün mantığı projenin kalbi: yanlış sayım her raporu bozar. Bu yüzden
kenar durumlar (çizgi üstü, paralel hareket, tekrar geçiş) burada tek tek
sınanır.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from occupancy.models import Track
from occupancy.zones import ZoneConfigError, ZoneManager

TS = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)

# 100..900 x 300..700 arası dikdörtgen bir salon, üst kenarında yatay bir kapı çizgisi.
TANIM = {
    "zones": [{"name": "salon", "polygon": [[100, 300], [900, 300], [900, 700], [100, 700]]}],
    "lines": [{"name": "kapi", "zone": "salon", "a": [100, 300], "b": [900, 300]}],
}


def _yaz(tmp_path, tanim=None):
    yol = tmp_path / "zones.json"
    yol.write_text(json.dumps(tanim or TANIM), encoding="utf-8")
    return yol


def _track(track_id: int, x: int, y: int) -> Track:
    """Ayak noktası (x, y) olan bir track üretir."""
    return Track(track_id=track_id, bbox=(x - 20, y - 100, x + 20, y), confidence=0.9)


@pytest.fixture
def yonetici(tmp_path):
    return ZoneManager.from_file(_yaz(tmp_path))


def test_polygon_icindeki_kisi_sayilir(yonetici):
    sayilar, _ = yonetici.update([_track(1, 500, 500)], TS)
    assert sayilar == {"salon": 1}


def test_polygon_disindaki_kisi_sayilmaz(yonetici):
    sayilar, _ = yonetici.update([_track(1, 50, 500)], TS)
    assert sayilar == {"salon": 0}


def test_kenar_uzerindeki_nokta_icerde_sayilir(yonetici):
    sayilar, _ = yonetici.update([_track(1, 100, 500)], TS)
    assert sayilar == {"salon": 1}


def test_cizgiyi_disaridan_iceriye_gecen_giris_uretir(yonetici):
    yonetici.update([_track(1, 500, 250)], TS)  # çizginin üstünde (dışarıda)
    _, olaylar = yonetici.update([_track(1, 500, 350)], TS)  # çizginin altında (içeride)

    assert [(o.event_type, o.zone, o.track_id) for o in olaylar] == [("enter", "salon", 1)]
    assert yonetici.totals == (1, 0)


def test_cizgiyi_iceriden_disariya_gecen_cikis_uretir(yonetici):
    yonetici.update([_track(1, 500, 350)], TS)
    _, olaylar = yonetici.update([_track(1, 500, 250)], TS)

    assert [o.event_type for o in olaylar] == ["exit"]
    assert yonetici.totals == (0, 1)


def test_ayni_kisi_ayni_yonde_iki_kez_sayilmaz(yonetici):
    yonetici.update([_track(1, 500, 250)], TS)
    yonetici.update([_track(1, 500, 350)], TS)  # giriş
    yonetici.update([_track(1, 500, 250)], TS)  # çıkış
    yonetici.update([_track(1, 500, 350)], TS)  # aynı yönde ikinci giriş -> sayılmaz

    assert yonetici.totals == (1, 1)


def test_farkli_kisiler_ayri_ayri_sayilir(yonetici):
    yonetici.update([_track(1, 500, 250), _track(2, 600, 250)], TS)
    _, olaylar = yonetici.update([_track(1, 500, 350), _track(2, 600, 350)], TS)

    assert len(olaylar) == 2
    assert yonetici.totals == (2, 0)


def test_cizgiye_paralel_hareket_olay_uretmez(yonetici):
    yonetici.update([_track(1, 300, 250)], TS)
    _, olaylar = yonetici.update([_track(1, 700, 250)], TS)

    assert olaylar == []
    assert yonetici.totals == (0, 0)


def test_cizgi_parcasinin_disindan_gecmek_sayilmaz(yonetici):
    """Çizgi 100..900 arası; x=950'de yukarıdan aşağı geçmek kapıdan geçmek değildir."""
    yonetici.update([_track(1, 950, 250)], TS)
    _, olaylar = yonetici.update([_track(1, 950, 350)], TS)

    assert olaylar == []


def test_ilk_gorulen_karede_olay_uretilmez(yonetici):
    """Önceki konumu bilinmeyen bir track geçiş yapmış sayılamaz."""
    _, olaylar = yonetici.update([_track(1, 500, 350)], TS)
    assert olaylar == []


def test_kaybolan_track_gecmisi_unutulur(yonetici):
    yonetici.update([_track(1, 500, 250)], TS)
    yonetici.update([], TS)  # kişi kareden çıktı
    _, olaylar = yonetici.update([_track(1, 500, 350)], TS)

    assert olaylar == []  # geçmiş yok, geçiş çıkarılamaz


def test_polygon_alani_hesaplanir(yonetici):
    assert yonetici.polygon_area("salon") == pytest.approx(800 * 400)


def test_zone_names(yonetici):
    assert yonetici.zone_names == ["salon"]


def test_eksik_dosya_anlasilir_hata(tmp_path):
    with pytest.raises(ZoneConfigError, match="bulunamadı"):
        ZoneManager.from_file(tmp_path / "yok.json")


def test_bozuk_json_anlasilir_hata(tmp_path):
    yol = tmp_path / "zones.json"
    yol.write_text("{bozuk", encoding="utf-8")

    with pytest.raises(ZoneConfigError, match="okunamadı"):
        ZoneManager.from_file(yol)


def test_uc_noktadan_az_poligon_reddedilir(tmp_path):
    tanim = {"zones": [{"name": "dar", "polygon": [[0, 0], [10, 10]]}], "lines": []}

    with pytest.raises(ZoneConfigError, match="en az 3"):
        ZoneManager.from_file(_yaz(tmp_path, tanim))


def test_bilinmeyen_bolgeye_bagli_cizgi_reddedilir(tmp_path):
    tanim = {
        "zones": [{"name": "salon", "polygon": [[0, 0], [10, 0], [10, 10]]}],
        "lines": [{"name": "kapi", "zone": "depo", "a": [0, 0], "b": [10, 0]}],
    }

    with pytest.raises(ZoneConfigError, match="depo"):
        ZoneManager.from_file(_yaz(tmp_path, tanim))


def test_bolge_tanimsizsa_reddedilir(tmp_path):
    with pytest.raises(ZoneConfigError, match="en az bir bölge"):
        ZoneManager.from_file(_yaz(tmp_path, {"zones": [], "lines": []}))

"""ROI çizim aracının saf mantığının testleri.

OpenCV penceresi test edilmez; test edilen şey, araçtan çıkan tanımın
`ZoneManager` tarafından sorunsuz okunabilmesi. İki modülü birbirine bağlayan
asıl sözleşme bu.
"""

from __future__ import annotations

import json

import pytest

from occupancy.zones import ZoneConfigError, ZoneManager
from tools.roi_ciz import RoiBuilder

KARE = [[100, 300], [900, 300], [900, 700], [100, 700]]


def _dolu_builder() -> RoiBuilder:
    olusturucu = RoiBuilder()
    for x, y in KARE:
        olusturucu.add_point(x, y)
    olusturucu.close_zone("salon")
    olusturucu.add_line("kapi", "salon", (100, 300), (900, 300))
    return olusturucu


def test_uretilen_tanim_zone_manager_tarafindan_okunur(tmp_path):
    yol = tmp_path / "zones.json"
    _dolu_builder().save(yol)

    yonetici = ZoneManager.from_file(yol)

    assert yonetici.zone_names == ["salon"]
    assert [c.name for c in yonetici.lines] == ["kapi"]


def test_kaydedilen_dosya_gecerli_json(tmp_path):
    yol = tmp_path / "zones.json"
    _dolu_builder().save(yol)

    tanim = json.loads(yol.read_text(encoding="utf-8"))

    assert tanim["zones"][0]["polygon"] == KARE


def test_ucten_az_nokta_ile_bolge_kapatilamaz():
    olusturucu = RoiBuilder()
    olusturucu.add_point(0, 0)
    olusturucu.add_point(10, 10)

    assert olusturucu.close_zone("dar") is False
    assert olusturucu.zones == []


def test_bolge_kapaninca_bekleyen_noktalar_temizlenir():
    olusturucu = RoiBuilder()
    for x, y in KARE:
        olusturucu.add_point(x, y)

    assert olusturucu.close_zone("salon") is True
    assert olusturucu.pending == []


def test_undo_once_bekleyen_noktayi_siler():
    olusturucu = RoiBuilder()
    olusturucu.add_point(1, 1)
    olusturucu.add_point(2, 2)

    olusturucu.undo()

    assert olusturucu.pending == [(1, 1)]


def test_undo_bekleyen_yokken_son_bolgeyi_siler():
    olusturucu = _dolu_builder()

    olusturucu.undo()  # önce çizgiyi siler
    olusturucu.undo()  # sonra bölgeyi

    assert olusturucu.zones == []
    assert olusturucu.lines == []


def test_bos_builder_kaydedilemez(tmp_path):
    with pytest.raises(ValueError, match="en az bir bölge"):
        RoiBuilder().save(tmp_path / "zones.json")


def test_tanimsiz_bolgeye_cizgi_eklenemez():
    olusturucu = RoiBuilder()

    with pytest.raises(ValueError, match="depo"):
        olusturucu.add_line("kapi", "depo", (0, 0), (10, 0))


def test_ayni_isimde_iki_bolge_olamaz():
    olusturucu = RoiBuilder()
    for x, y in KARE:
        olusturucu.add_point(x, y)
    olusturucu.close_zone("salon")
    for x, y in KARE:
        olusturucu.add_point(x, y)

    assert olusturucu.close_zone("salon") is False


def test_cizgisiz_tanim_da_gecerlidir(tmp_path):
    olusturucu = RoiBuilder()
    for x, y in KARE:
        olusturucu.add_point(x, y)
    olusturucu.close_zone("salon")

    yol = tmp_path / "zones.json"
    olusturucu.save(yol)

    assert ZoneManager.from_file(yol).lines == []


def test_bozuk_tanim_zone_manager_tarafindan_reddedilir(tmp_path):
    """Araç geçerli üretiyor diye doğrulama gevşetilmemeli."""
    yol = tmp_path / "zones.json"
    yol.write_text(json.dumps({"zones": [], "lines": []}), encoding="utf-8")

    with pytest.raises(ZoneConfigError):
        ZoneManager.from_file(yol)

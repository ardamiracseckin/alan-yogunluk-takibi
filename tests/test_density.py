"""Yoğunluk skoru ve heatmap testleri."""

from __future__ import annotations

import numpy as np
import pytest

from occupancy.density import DensityMap, zone_density

PNG_IMZASI = b"\x89PNG\r\n\x1a\n"


def test_yogunluk_bin_piksel_kare_basina_kisi():
    # 10 kişi, 100_000 piksel² alan -> 1000 piksel²'de 0.1 kişi
    assert zone_density(10, 100_000) == pytest.approx(0.1)


def test_alan_sifirken_yogunluk_sifir():
    assert zone_density(5, 0) == 0.0


def test_kisi_yokken_yogunluk_sifir():
    assert zone_density(0, 100_000) == 0.0


def test_nokta_eklenince_o_bolgedeki_isi_artar():
    harita = DensityMap(shape=(200, 200), sigma=10)
    onceki = harita.raw[100, 100]

    harita.add([(100, 100)])

    assert harita.raw[100, 100] > onceki


def test_isi_merkezde_kenardan_yuksek():
    harita = DensityMap(shape=(200, 200), sigma=10)
    harita.add([(100, 100)])

    assert harita.raw[100, 100] > harita.raw[100, 130]


def test_kare_disindaki_nokta_yok_sayilir():
    harita = DensityMap(shape=(200, 200), sigma=10)
    harita.add([(500, 500), (-10, -10)])

    assert harita.raw.max() == 0.0


def test_colormap_ucreklikli_bgr_goruntu_dondurur():
    harita = DensityMap(shape=(120, 160), sigma=10)
    harita.add([(80, 60)])

    gorsel = harita.as_colormap()

    assert gorsel.shape == (120, 160, 3)
    assert gorsel.dtype == np.uint8


def test_png_ciktisi_gecerli_imzayla_baslar():
    harita = DensityMap(shape=(120, 160), sigma=10)
    harita.add([(80, 60)])

    assert harita.to_png_bytes().startswith(PNG_IMZASI)


def test_arka_plan_ile_harmanlama_ayni_boyutu_korur():
    harita = DensityMap(shape=(120, 160), sigma=10)
    harita.add([(80, 60)])
    arka_plan = np.full((120, 160, 3), 255, dtype=np.uint8)

    harmanlanmis = harita.as_colormap(background=arka_plan)

    assert harmanlanmis.shape == (120, 160, 3)


def test_bos_haritada_png_uretilebilir():
    assert DensityMap(shape=(60, 80), sigma=5).to_png_bytes().startswith(PNG_IMZASI)

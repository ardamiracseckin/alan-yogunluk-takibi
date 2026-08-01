"""Komut satırı argümanlarının ayarlara nasıl bindiğinin testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from occupancy.__main__ import build_settings


def test_argumansiz_varsayilanlar_korunur():
    ayarlar, argumanlar = build_settings([])

    assert ayarlar.source == "ornek/demo.mp4"
    assert ayarlar.loop_video is True
    assert argumanlar.no_web is False


def test_source_argumani_ayari_ezer():
    ayarlar, _ = build_settings(["--source", "rtsp://kamera/akis"])

    assert ayarlar.source == "rtsp://kamera/akis"


def test_ortam_degiskeni_cli_ile_ezilir(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_PORT", "9001")

    ayarlar, _ = build_settings(["--port", "9500"])

    assert ayarlar.port == 9500


def test_ortam_degiskeni_cli_verilmeyince_kalir(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_PORT", "9001")

    ayarlar, _ = build_settings([])

    assert ayarlar.port == 9001


def test_no_loop_bayragi_dongueyi_kapatir():
    ayarlar, _ = build_settings(["--no-loop"])

    assert ayarlar.loop_video is False


def test_yol_argumanlari_path_olur():
    ayarlar, _ = build_settings(["--zones", "a/b.json", "--db", "c/d.db"])

    assert ayarlar.zones_path == Path("a/b.json")
    assert ayarlar.db_path == Path("c/d.db")


def test_gecersiz_guven_esigi_reddedilir():
    with pytest.raises(ValidationError):
        build_settings(["--conf", "1.7"])


def test_no_web_bayragi_okunur():
    _, argumanlar = build_settings(["--no-web"])

    assert argumanlar.no_web is True

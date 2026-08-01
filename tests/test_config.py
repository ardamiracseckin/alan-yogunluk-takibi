"""Ayar yükleme davranışının testleri."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from occupancy.config import Settings


def test_defaults_when_no_env():
    ayarlar = Settings(_env_file=None)

    assert ayarlar.source == "ornek/demo.mp4"
    assert ayarlar.model_path == "yolov8n.pt"
    assert ayarlar.conf_threshold == pytest.approx(0.35)
    assert ayarlar.zones_path == Path("ornek/zones.json")
    assert ayarlar.db_path == Path("data/occupancy.db")
    assert ayarlar.port == 8000
    assert ayarlar.snapshot_interval_sec == 10
    assert ayarlar.loop_video is True
    assert ayarlar.log_level == "INFO"


def test_environment_overrides_default(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_SOURCE", "rtsp://kamera/akis")
    monkeypatch.setenv("OCCUPANCY_PORT", "9001")
    monkeypatch.setenv("OCCUPANCY_LOOP_VIDEO", "false")

    ayarlar = Settings(_env_file=None)

    assert ayarlar.source == "rtsp://kamera/akis"
    assert ayarlar.port == 9001
    assert ayarlar.loop_video is False


@pytest.mark.parametrize("gecersiz", [-0.1, 1.5])
def test_conf_threshold_must_be_between_zero_and_one(monkeypatch, gecersiz):
    monkeypatch.setenv("OCCUPANCY_CONF_THRESHOLD", str(gecersiz))

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_log_level_is_normalised(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_LOG_LEVEL", "debug")

    assert Settings(_env_file=None).log_level == "DEBUG"


def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("OCCUPANCY_LOG_LEVEL", "gurultu")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_ensure_directories_creates_missing_paths(tmp_path):
    ayarlar = Settings(
        _env_file=None,
        db_path=tmp_path / "veri" / "occupancy.db",
        reports_dir=tmp_path / "raporlar",
        logs_dir=tmp_path / "loglar",
    )

    ayarlar.ensure_directories()

    assert (tmp_path / "veri").is_dir()
    assert (tmp_path / "raporlar").is_dir()
    assert (tmp_path / "loglar").is_dir()

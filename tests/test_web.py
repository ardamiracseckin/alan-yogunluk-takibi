"""Dashboard ve REST uç noktaları testleri."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import FakeDetector, FakeTracker, FakeVideoSource, kisi
from fastapi.testclient import TestClient

from occupancy.config import Settings
from occupancy.models import Snapshot
from occupancy.pipeline import Pipeline
from occupancy.storage import Storage
from occupancy.zones import ZoneManager
from web.app import create_app

TANIM = {
    "zones": [{"name": "salon", "polygon": [[100, 300], [600, 300], [600, 470], [100, 470]]}],
    "lines": [{"name": "kapi", "zone": "salon", "a": [100, 300], "b": [600, 300]}],
}


@pytest.fixture
def hat(tmp_path):
    depo = Storage(tmp_path / "web.db")
    hat = Pipeline(
        video_source=FakeVideoSource(frame_count=2),
        detector=FakeDetector([[kisi(300, 400)]]),
        tracker=FakeTracker(),
        zones=ZoneManager.from_dict(TANIM),
        storage=depo,
        snapshot_interval_sec=0,
    )
    hat.run_until_complete()
    yield hat
    depo.close()


@pytest.fixture
def istemci(hat, tmp_path):
    ayarlar = Settings(_env_file=None, db_path=tmp_path / "web.db", reports_dir=tmp_path / "rapor")
    return TestClient(create_app(hat, ayarlar))


def test_ana_sayfa_html_doner(istemci):
    yanit = istemci.get("/")

    assert yanit.status_code == 200
    assert "text/html" in yanit.headers["content-type"]
    assert "Alan Kullanım" in yanit.text


def test_statik_dosyalar_sunulur(istemci):
    assert istemci.get("/static/app.js").status_code == 200
    assert istemci.get("/static/style.css").status_code == 200


def test_api_live_beklenen_anahtarlari_icerir(istemci):
    veri = istemci.get("/api/live").json()

    assert set(veri) >= {
        "ts",
        "counts",
        "density",
        "total_people",
        "total_in",
        "total_out",
        "fps",
        "source_ok",
    }
    assert veri["counts"] == {"salon": 1}


def test_api_history_liste_doner(istemci):
    veri = istemci.get("/api/history?minutes=60").json()

    assert isinstance(veri, list)
    assert veri and veri[0]["zone"] == "salon"
    assert {"ts", "zone", "count", "density"} <= set(veri[0])


def test_api_history_bos_aralikta_bos_liste(istemci):
    assert istemci.get("/api/history?minutes=0").json() == []


def test_heatmap_png_doner(istemci):
    yanit = istemci.get("/api/heatmap.png")

    assert yanit.status_code == 200
    assert yanit.headers["content-type"] == "image/png"
    assert yanit.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_rapor_json_doner(istemci):
    bugun = datetime.now(UTC).date().isoformat()

    veri = istemci.get(f"/api/report?date={bugun}").json()

    assert veri["date"] == bugun
    assert {"totals", "zones", "hourly"} <= set(veri)


def test_rapor_veri_olmayan_gun_icin_bos_doner(istemci):
    veri = istemci.get("/api/report?date=2000-01-01").json()

    assert veri["totals"] == {"entries": 0, "exits": 0, "net": 0}


def test_rapor_csv_formati(istemci):
    yanit = istemci.get("/api/report?date=2000-01-01&format=csv")

    assert yanit.status_code == 200
    assert "text/csv" in yanit.headers["content-type"]
    assert yanit.text.startswith("saat,")


def test_gecersiz_tarih_400_doner(istemci):
    assert istemci.get("/api/report?date=dun").status_code == 400


def test_health_hat_calismiyorken_503(istemci):
    yanit = istemci.get("/health")

    assert yanit.status_code == 503
    assert yanit.json()["pipeline_alive"] is False


def test_video_feed_hat_olunce_akisi_bitirir(istemci):
    with istemci.stream("GET", "/video_feed") as yanit:
        assert yanit.status_code == 200
        icerik = b"".join(yanit.iter_bytes())

    assert icerik == b""  # hat çalışmıyor, akış hemen kapanır


def test_health_hat_calisirken_200(tmp_path):
    depo = Storage(tmp_path / "canli.db")
    hat = Pipeline(
        video_source=FakeVideoSource(frame_count=10_000),
        detector=FakeDetector([[kisi(300, 400)]]),
        tracker=FakeTracker(),
        zones=ZoneManager.from_dict(TANIM),
        storage=depo,
        snapshot_interval_sec=0,
    )
    hat.start()
    try:
        ayarlar = Settings(_env_file=None, db_path=tmp_path / "canli.db")
        with TestClient(create_app(hat, ayarlar)) as istemci:
            yanit = istemci.get("/health")
        assert yanit.status_code == 200
        assert yanit.json()["pipeline_alive"] is True
    finally:
        hat.stop()
        depo.close()


def test_history_veritabanindan_okur(tmp_path, hat):
    """Geçmiş, hattın belleğinden değil veritabanından gelir."""
    ayarlar = Settings(_env_file=None, db_path=tmp_path / "web.db")
    with Storage(tmp_path / "web.db") as depo:
        depo.add_snapshot(
            Snapshot(
                ts=datetime.now(UTC) - timedelta(minutes=5), zone="depo", count=42, density=1.5
            )
        )

    istemci = TestClient(create_app(hat, ayarlar))
    veri = istemci.get("/api/history?minutes=30").json()

    assert any(satir["zone"] == "depo" and satir["count"] == 42 for satir in veri)


def test_live_json_serilestirilebilir(istemci):
    json.dumps(istemci.get("/api/live").json())

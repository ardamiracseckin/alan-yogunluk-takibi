# Gerçek Zamanlı Alan Kullanım ve Yoğunluk Takibi — Uygulama Planı

> **Ajan işçiler için:** Bu planı görev görev uygulamak için `superpowers:subagent-driven-development` (önerilen) veya `superpowers:executing-plans` alt-becerisini kullan. Adımlar `- [ ]` kutucuk sözdizimiyle takip edilir.

## Context

Bir mağaza/otopark/etkinlik alanına bakan kameradan anlık kişi sayısı, giriş/çıkış ve yoğunluk haritası çıkaran, dashboard'dan izlenen ve günlük JSON/CSV raporu üreten bir sistem. Sekiz aşamaya bölünür: video input, detection, tracking, bölge analizi, dashboard, veritabanı, raporlama, logging & error handling.

Amaç: projeyi sıfırdan, temiz ve GitHub'da gösterilebilir kalitede kurmak — klonlayan birinin beş dakikada çalıştırabildiği, README'sine bakanın beş saniyede ne olduğunu anladığı bir repo.

Onaylanmış tasarım: `docs/superpowers/specs/2026-08-01-alan-yogunluk-takibi-design.md`.

**Goal:** Sekiz aşamayı da karşılayan, testleri ve CI'ı yeşil, Docker'la ayağa kalkan, canlı dashboard'lu bir kişi sayım ve yoğunluk takip sistemi yayınlamak.

**Architecture:** `occupancy/` altında tek sorumluluklu modüller (video → detection → tracking → zones → density → storage → reporting), bunları arka planda kendi thread'inde birleştiren `pipeline.py`, ve pipeline'ın paylaştığı son kare + son metrikleri okuyan `web/app.py` (FastAPI). Ağır bağımlılıklar (YOLO/torch) protokollerin arkasında; testler sahte detector/tracker ile model indirmeden koşar.

**Tech Stack:** Python 3.13, ultralytics (YOLOv8 + ByteTrack), OpenCV, FastAPI + uvicorn, SQLite (stdlib), pydantic-settings, pytest, ruff, Docker Compose, GitHub Actions.

## Global Constraints

- Sanal ortam: proje kökünde `.venv` (Python 3.13). 3.14 kullanılmayacak.
- Tüm testler YOLO ağırlığı indirmeden, GPU'suz, ağ erişimi olmadan çalışmak zorunda. Ağır sınıflar `Protocol` arkasında; testlerde sahte uygulamalar kullanılır.
- Kullanıcıya görünen tüm metinler (README, dashboard, log mesajları, CLI yardımı) Türkçe. Modül/fonksiyon/değişken adları İngilizce.
- Zaman damgaları her yerde ISO-8601 UTC (`datetime.now(UTC)`), veritabanına metin olarak yazılır.
- Her görev sonunda `ruff check .` ve `pytest` yeşil olmadan commit yok.
- Commit mesajları Türkçe, konu satırı 72 karakteri geçmez.
- `.env`, `data/`, `logs/`, `reports/`, `*.pt`, `.venv/` git'e girmez (`.gitignore` ilk görevde yazıldı).

## Dosya yapısı

| Dosya | Sorumluluk |
|---|---|
| `occupancy/models.py` | `Detection`, `Track`, `ZoneEvent`, `Snapshot`, `LiveStats` veri sınıfları |
| `occupancy/config.py` | `.env` + CLI'dan ayar yükleme, doğrulama |
| `occupancy/logging_conf.py` | dosya + konsol logging kurulumu |
| `occupancy/video.py` | `VideoSource`: dosya / webcam / RTSP, yeniden bağlanma |
| `occupancy/detection.py` | `PersonDetector` protokolü + `YoloPersonDetector` |
| `occupancy/tracking.py` | `Tracker` protokolü + ByteTrack tabanlı `PersonTracker` |
| `occupancy/zones.py` | ROI poligonları, çizgi geçişi, giriş/çıkış sayımı |
| `occupancy/density.py` | yoğunluk skoru + birikimli heatmap |
| `occupancy/storage.py` | SQLite şema ve erişim |
| `occupancy/reporting.py` | günlük JSON/CSV rapor |
| `occupancy/overlay.py` | kare üzerine bbox/ID/ROI/sayaç çizimi |
| `occupancy/pipeline.py` | işleme döngüsü, paylaşılan durum, thread yönetimi |
| `occupancy/__main__.py` | CLI giriş noktası |
| `web/app.py` | FastAPI uygulaması ve uç noktalar |
| `web/static/` | `index.html`, `app.js`, `style.css` |
| `tools/roi_ciz.py` | interaktif ROI çizim aracı |
| `tests/` | modül başına test dosyası + `conftest.py` sahteleri |
| `docker/` | `Dockerfile`, `docker-compose.yml` |
| `.github/workflows/ci.yml` | ruff + pytest |

---

## Görev 1: İskelet, ayarlar, logging

**Dosyalar:** `occupancy/__init__.py`, `occupancy/models.py`, `occupancy/config.py`, `occupancy/logging_conf.py`, `requirements.txt`, `requirements-dev.txt`, `.env.example`, `pyproject.toml` (ruff + pytest ayarı), `tests/test_config.py`

**Üretir:** aşağıdaki veri sınıfları ve `Settings` — sonraki tüm görevler bunları kullanır.

```python
# models.py
@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2
    confidence: float

@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    @property
    def centroid(self) -> tuple[int, int]: ...
    @property
    def foot_point(self) -> tuple[int, int]:   # ((x1+x2)//2, y2)
        ...

@dataclass(frozen=True)
class ZoneEvent:
    ts: datetime
    track_id: int
    event_type: Literal["enter", "exit"]
    zone: str

@dataclass(frozen=True)
class Snapshot:
    ts: datetime
    zone: str
    count: int
    density: float

@dataclass
class LiveStats:
    ts: datetime
    counts: dict[str, int]        # bölge adı -> anlık kişi
    density: dict[str, float]
    total_in: int
    total_out: int
    fps: float
    source_ok: bool
```

`Settings` (pydantic-settings, `.env` okur, CLI ile ezilebilir): `source: str = "ornek/demo.mp4"`, `model_path: str = "yolov8n.pt"`, `conf_threshold: float = 0.35`, `zones_path: Path = Path("ornek/zones.json")`, `db_path: Path = Path("data/occupancy.db")`, `reports_dir: Path = Path("reports")`, `logs_dir: Path = Path("logs")`, `host: str = "127.0.0.1"`, `port: int = 8000`, `snapshot_interval_sec: int = 10`, `loop_video: bool = True`, `log_level: str = "INFO"`.

**Adımlar:**
- [ ] `tests/test_config.py` yaz: (a) `.env` yokken varsayılanlar geliyor, (b) ortam değişkeni ayarı eziyor, (c) `conf_threshold` 0-1 dışındaysa `ValidationError`.
- [ ] Testi çalıştır, `ModuleNotFoundError: occupancy` ile başarısız olduğunu gör.
- [ ] `models.py`, `config.py`, `logging_conf.py` yaz; `requirements*.txt`, `.env.example`, `pyproject.toml` ekle.
- [ ] `pytest -v` ve `ruff check .` yeşil.
- [ ] Commit: `feat: proje iskeleti, ayarlar ve logging`

---

## Görev 2: Video kaynağı

**Dosyalar:** `occupancy/video.py`, `tests/test_video.py`

**Üretir:** `VideoSource(source: str, loop: bool = True)` — `frames()` jeneratörü `np.ndarray` üretir; `is_open: bool`, `fps: float`, `resolution: tuple[int,int]`, `close()`.

Kaynak çözümleme: tamsayı metin (`"0"`) → webcam, `rtsp://`/`http://` → ağ akışı, aksi halde dosya yolu. Dosya bittiğinde `loop=True` ise başa sarar. Okuma başarısız olursa artan bekleme (1, 2, 4, 8, 16, 30 sn tavan) ile yeniden açmayı dener, her denemeyi loglar, bu sırada `is_open=False` olur.

**Adımlar:**
- [ ] `tests/test_video.py` yaz: OpenCV'yi sahteleyerek (a) dosya bitince başa sardığını, (b) `loop=False` iken jeneratörün bittiğini, (c) açılamayan kaynakta yeniden deneme aralıklarının 1,2,4,... olduğunu (uyku fonksiyonu enjekte edilerek, gerçek beklemesiz) doğrula.
- [ ] Testin başarısız olduğunu gör.
- [ ] `video.py` yaz.
- [ ] `pytest tests/test_video.py -v` yeşil.
- [ ] Commit: `feat: yeniden bağlanabilen video kaynağı`

---

## Görev 3: Tespit ve izleme

**Dosyalar:** `occupancy/detection.py`, `occupancy/tracking.py`, `tests/test_tracking.py`, `tests/conftest.py`

**Üretir:**
- `class PersonDetector(Protocol): def detect(self, frame) -> list[Detection]`
- `class YoloPersonDetector: def __init__(self, model_path: str, conf: float)` — `ultralytics.YOLO`, sadece `classes=[0]` (person).
- `class Tracker(Protocol): def update(self, frame, detections: list[Detection]) -> list[Track]`
- `class PersonTracker` — ByteTrack ile kalıcı ID atar.
- `tests/conftest.py`: `FakeDetector` (önceden verilmiş kare→Detection listesi döndürür) ve `FakeTracker` fixture'ları; sonraki tüm görevler bunları kullanır.

> **Not:** Bu görevin ilk adımı, kurulu ultralytics sürümünde ByteTrack'in tam olarak hangi giriş noktasıyla çağrıldığını doğrulamaktır (`.venv/bin/python -c "..."` ile). Doğrulanan çağrı biçimi `PersonTracker` içinde tek bir yerde kapsüllenir; `Tracker` protokolü sayesinde sistemin geri kalanı bu detaydan etkilenmez.

**Adımlar:**
- [ ] `.venv` içinde ByteTrack giriş noktasını doğrula, çalışan çağrıyı not et.
- [ ] `tests/conftest.py` içine `FakeDetector`/`FakeTracker` yaz.
- [ ] `tests/test_tracking.py` yaz: sahte tespitlerle aynı kişinin kareler boyunca aynı `track_id`'yi koruduğunu, kaybolan kişinin ID'sinin düştüğünü doğrula.
- [ ] Testin başarısız olduğunu gör.
- [ ] `detection.py` + `tracking.py` yaz.
- [ ] `pytest tests/test_tracking.py -v` yeşil.
- [ ] Gerçek doğrulama: `.venv/bin/python -m occupancy.detection --demo ornek/demo.mp4` ile bir pencerede bbox+ID görülüyor (ilk görünür çıktı).
- [ ] Commit: `feat: YOLOv8 kişi tespiti ve ByteTrack izleme`

---

## Görev 4: Bölge analizi ve sayım

**Dosyalar:** `occupancy/zones.py`, `ornek/zones.json`, `tests/test_zones.py`

**Üretir:** `ZoneManager.from_file(path) -> ZoneManager`; `update(tracks: list[Track], ts: datetime) -> tuple[dict[str,int], list[ZoneEvent]]`; `zone_names: list[str]`; `polygon_area(name) -> float`; `totals -> tuple[int,int]` (toplam giriş, toplam çıkış).

`zones.json` biçimi:
```json
{
  "zones": [{"name": "salon", "polygon": [[100,300],[900,300],[900,700],[100,700]]}],
  "lines": [{"name": "kapi", "zone": "salon", "a": [100,300], "b": [900,300]}]
}
```

Sayım kuralları: anlık doluluk = ayak noktası poligon içinde olan aktif track sayısı (ray casting). Giriş/çıkış = track'in ayak noktasının çizgiye göre işaretli tarafı (cross product) önceki kareden bu kareye işaret değiştirdiyse; negatiften pozitife = `enter`, tersi = `exit`. Her `(track_id, line_name, event_type)` üçlüsü **yalnızca bir kez** sayılır.

**Adımlar:**
- [ ] `tests/test_zones.py` yaz — en az şu durumlar: poligon içi/dışı/kenar üstü nokta; çizgiyi soldan sağa geçen track `enter` üretir; sağdan sola geçen `exit` üretir; aynı track aynı yönde tekrar geçerse ikinci kez sayılmaz; çizgiye paralel hareket olay üretmez; geçersiz `zones.json` anlaşılır hata verir.
- [ ] Testlerin başarısız olduğunu gör.
- [ ] `zones.py` ve örnek `ornek/zones.json` yaz.
- [ ] `pytest tests/test_zones.py -v` yeşil.
- [ ] Commit: `feat: ROI bölge analizi ve giriş/çıkış sayımı`

---

## Görev 5: Yoğunluk ve heatmap

**Dosyalar:** `occupancy/density.py`, `tests/test_density.py`

**Üretir:** `zone_density(count: int, area_px: float) -> float` (kişi / 1000 piksel²); `class DensityMap(shape, sigma=25)` → `add(points)`, `as_colormap() -> np.ndarray`, `to_png_bytes(background=None) -> bytes`.

**Adımlar:**
- [ ] `tests/test_density.py` yaz: bilinen sayı/alan için beklenen yoğunluk; alan 0 iken 0 dönmesi; `add` sonrası verilen noktada ısı değerinin arttığı; `to_png_bytes` geçerli PNG imzasıyla başlıyor.
- [ ] Testlerin başarısız olduğunu gör.
- [ ] `density.py` yaz.
- [ ] `pytest tests/test_density.py -v` yeşil.
- [ ] Commit: `feat: yoğunluk skoru ve birikimli heatmap`

---

## Görev 6: Veritabanı

**Dosyalar:** `occupancy/storage.py`, `tests/test_storage.py`

**Üretir:** `class Storage(db_path)` → `init_schema()`, `add_events(list[ZoneEvent])`, `add_snapshot(Snapshot)`, `events_between(start, end) -> list[ZoneEvent]`, `snapshots_between(start, end) -> list[Snapshot]`, `close()`. Bağlantı `check_same_thread=False` + `threading.Lock` ile korunur (pipeline thread'i yazar, web thread'i okur).

Şema tasarım dokümanındaki `events` / `snapshots` tabloları ve `ts` indeksleri.

**Adımlar:**
- [ ] `tests/test_storage.py` yaz: yaz-oku roundtrip; zaman aralığı filtresi sınırları; şemanın iki kez oluşturulabilmesi (idempotent); iki thread'den eşzamanlı yazmanın hata vermemesi.
- [ ] Testlerin başarısız olduğunu gör.
- [ ] `storage.py` yaz.
- [ ] `pytest tests/test_storage.py -v` yeşil.
- [ ] Commit: `feat: SQLite depolama katmanı`

---

## Görev 7: Raporlama

**Dosyalar:** `occupancy/reporting.py`, `tests/test_reporting.py`

**Üretir:** `build_report(storage, day: date) -> dict`; `write_report(report, reports_dir) -> tuple[Path, Path]`; `report_to_csv(report) -> str`; CLI: `python -m occupancy.reporting --date YYYY-MM-DD`.

Rapor içeriği: toplam giriş, toplam çıkış, bölge bazında saatlik ortalama ve pik doluluk, pik saat, gün sonu net doluluk.

**Adımlar:**
- [ ] `tests/test_reporting.py` yaz: bilinen olay/snapshot setinden beklenen toplamlar ve pik saat; veri olmayan gün için boş ama geçerli rapor; CSV'nin başlık satırı ve satır sayısı.
- [ ] Testlerin başarısız olduğunu gör.
- [ ] `reporting.py` yaz.
- [ ] `pytest tests/test_reporting.py -v` yeşil.
- [ ] Commit: `feat: günlük JSON ve CSV raporlama`

---

## Görev 8: Overlay ve pipeline

**Dosyalar:** `occupancy/overlay.py`, `occupancy/pipeline.py`, `occupancy/__main__.py`, `tests/test_pipeline.py`

**Üretir:**
- `draw_overlay(frame, tracks, zones, stats) -> np.ndarray`
- `class Pipeline(settings, detector, tracker, zones, storage)` → `start()`, `stop()`, `latest_frame() -> bytes | None` (JPEG), `latest_stats() -> LiveStats`, `is_alive: bool`
- `python -m occupancy --source ... --zones ... --no-web` CLI

Pipeline kendi thread'inde koşar, kuyruk tutmaz (her zaman en güncel kare), `snapshot_interval_sec`'te bir snapshot yazar, olayları anında yazar, DB hatasında loglayıp devam eder, thread çökerse `is_alive=False`.

**Adımlar:**
- [ ] `tests/test_pipeline.py` yaz: sahte video + sahte detector/tracker ile bir kişi çizgiyi geçtiğinde `total_in` 1 oluyor ve DB'ye `enter` olayı yazılıyor; `Storage.add_snapshot` hata fırlattığında pipeline durmuyor; `stop()` sonrası thread ölüyor.
- [ ] Testlerin başarısız olduğunu gör.
- [ ] `overlay.py`, `pipeline.py`, `__main__.py` yaz.
- [ ] `pytest -v` (tüm test paketi) yeşil.
- [ ] Commit: `feat: işleme hattı ve görüntü üzerine bilgi bastırma`

---

## Görev 9: FastAPI dashboard

**Dosyalar:** `web/__init__.py`, `web/app.py`, `web/static/index.html`, `web/static/app.js`, `web/static/style.css`, `tests/test_web.py`

**Üretir:** `create_app(pipeline) -> FastAPI` ve uç noktalar: `/`, `/video_feed` (MJPEG), `/api/live`, `/api/history?minutes=30`, `/api/heatmap.png`, `/api/report?date=&format=`, `/health`.

Dashboard tek sayfa: solda canlı akış, sağda anlık doluluk / toplam giriş / toplam çıkış kartları, altta son 30 dakika doluluk grafiği ve heatmap. Chart.js **repoya gömülü** (CDN yok — Docker ve çevrimdışı çalışsın). `/api/live` 1 sn'de bir yoklanır; `source_ok=false` iken üstte "kaynak bağlantısı yok" uyarısı çıkar.

**Adımlar:**
- [ ] `tests/test_web.py` yaz: TestClient ile tüm uç noktalar 200 dönüyor; `/api/live` beklenen anahtarları içeriyor; pipeline ölüyken `/health` 503 dönüyor; `/api/report` bilinmeyen tarih için boş rapor dönüyor; `/api/report?format=csv` `text/csv` içerik tipi dönüyor.
- [ ] Testlerin başarısız olduğunu gör.
- [ ] `web/app.py` + statik dosyaları yaz.
- [ ] `pytest tests/test_web.py -v` yeşil.
- [ ] Gerçek doğrulama: `python -m occupancy --source ornek/demo.mp4` → `localhost:8000` canlı akış ve artan sayaçlar.
- [ ] Commit: `feat: FastAPI dashboard ve canlı MJPEG akışı`

---

## Görev 10: İnteraktif ROI çizim aracı

**Dosyalar:** `tools/roi_ciz.py`, `tests/test_roi_araci.py`

**Üretir:** `python tools/roi_ciz.py --source ornek/demo.mp4 --out ornek/zones.json` — ilk kareyi açar, sol tıkla poligon köşesi ekler, `n` ile bölgeyi kapatıp isim sorar, `l` ile iki tıklamayla giriş/çıkış çizgisi çizer, `s` ile kaydeder, `z` geri alır, `q` çıkar. Yardım metni pencereye basılır.

Saf mantık (nokta biriktirme, JSON üretme) GUI'den ayrı bir `RoiBuilder` sınıfında olur; test onu test eder, OpenCV penceresi test edilmez.

**Adımlar:**
- [ ] `tests/test_roi_araci.py` yaz: `RoiBuilder`'a nokta dizisi verildiğinde üretilen JSON'un `ZoneManager.from_file` tarafından sorunsuz yüklendiğini doğrula (iki modülü birbirine bağlayan asıl test bu).
- [ ] Testin başarısız olduğunu gör.
- [ ] `tools/roi_ciz.py` yaz.
- [ ] `pytest tests/test_roi_araci.py -v` yeşil; aracı elle çalıştırıp `ornek/zones.json` üret.
- [ ] Commit: `feat: interaktif ROI çizim aracı`

---

## Görev 11: Docker ve CI

**Dosyalar:** `docker/Dockerfile`, `docker/docker-compose.yml`, `.dockerignore`, `.github/workflows/ci.yml`

Dockerfile: `python:3.13-slim`, `requirements.txt` kurulur, ardından `opencv-python` kaldırılıp `opencv-python-headless` kurulur (konteynerde GUI yok), `uvicorn` ile başlatılır. Compose: tek servis, `data/`, `reports/`, `logs/`, `ornek/` volume'ları, port 8000. README'de macOS'ta webcam'in konteynerden erişilemediği, dosya/RTSP kaynağı kullanılması gerektiği belirtilir.

CI: ubuntu-latest, Python 3.13, `pip install -r requirements.txt -r requirements-dev.txt`, `ruff check .`, `pytest`.

**Adımlar:**
- [ ] `.github/workflows/ci.yml` ve `.dockerignore` yaz.
- [ ] `docker/Dockerfile` + `docker-compose.yml` yaz.
- [ ] `docker compose -f docker/docker-compose.yml up --build` ile ayağa kaldır, `localhost:8000` açılıyor mu doğrula.
- [ ] Commit: `chore: Docker Compose ve GitHub Actions CI`

---

## Görev 12: README, demo ve yayın

**Dosyalar:** `README.md`, `docs/gorseller/` (GIF + ekran görüntüleri), `LICENSE` (MIT)

README: tepede demo GIF, tek paragraf ne işe yaradığı, CI rozeti, hızlı başlangıç (klonla → kur → çalıştır, 3 komut), ROI tanımlama, Docker ile çalıştırma, mimari şeması, uç nokta tablosu, örnek rapor çıktısı, teknik tercihlerin gerekçesi (neden ByteTrack, neden SQLite), sınırlar (yoğunluk piksel bazlı, kamera kalibrasyonu yok).

**Adımlar:**
- [ ] Sistemi örnek videoyla çalıştırıp dashboard'un ~10 sn'lik ekran kaydını al, GIF'e çevir (`ffmpeg`), `docs/gorseller/` altına koy.
- [ ] `README.md` ve `LICENSE` yaz.
- [ ] Tüm testler + ruff yeşil, `git log` temiz.
- [ ] `gh repo create alan-yogunluk-takibi --public --source . --push`
- [ ] GitHub'da CI'ın yeşile döndüğünü ve README'nin GIF dahil doğru render edildiğini doğrula.
- [ ] Commit: `docs: README, demo GIF ve lisans`

---

## Doğrulama (uçtan uca)

1. `rm -rf .venv && /usr/local/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt` — temiz kurulum hatasız.
2. `.venv/bin/ruff check .` ve `.venv/bin/pytest -v` — tamamı yeşil, ağ erişimi gerekmiyor.
3. `.venv/bin/python -m occupancy --source ornek/demo.mp4` → `localhost:8000`: canlı video, bbox+ID, ROI çizili; kişi çizgiyi geçince giriş sayacı artıyor; doluluk grafiği ve heatmap doluyor.
4. `.venv/bin/python -m occupancy.reporting --date $(date +%F)` → `reports/` altında dolu JSON ve CSV.
5. `docker compose -f docker/docker-compose.yml up --build` → aynı dashboard konteynerden çalışıyor.
6. GitHub'da CI rozeti yeşil; başka bir dizine `git clone` → adım 1-3 tekrar çalışıyor.

## Bilinen riskler

- **ByteTrack giriş noktası:** ultralytics sürümleri arasında değişebiliyor. Görev 3 bunu deneyle sabitliyor ve tek bir sınıfın içine kapsüllüyor; değişirse etkilenen tek dosya `tracking.py`.
- **Demo videosu:** repoya konacak örnek videonun lisansı uygun ve boyutu küçük olmalı (< 10 MB). Uygun klip yoksa Görev 2'de webcam kaydından kısa bir klip üretilir.
- **macOS'ta Docker + webcam:** desteklenmiyor; README'de açıkça yazılacak, demo dosya kaynağıyla yapılacak.

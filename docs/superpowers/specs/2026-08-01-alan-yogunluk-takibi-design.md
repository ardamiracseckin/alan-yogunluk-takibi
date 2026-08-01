# Gerçek Zamanlı Alan Kullanım ve Yoğunluk Takibi — Tasarım

Tarih: 2026-08-01

## 1. Amaç

Bir mağaza, otopark veya etkinlik alanına bakan kameradan gelen görüntü akışında:

- o an alanda kaç kişi olduğunu,
- kimin girip kimin çıktığını (aynı kişiyi tekrar saymadan),
- alanın yoğunluk haritasını

gerçek zamanlı olarak takip eden; bu bilgileri bir web dashboard'unda gösteren ve günlük JSON/CSV raporu üreten bir sistem.

## 2. Kapsam kararları

| Karar | Seçim | Gerekçe |
|---|---|---|
| Video kaynağı | Dosya + webcam birincil, RTSP destekli | Repoyu klonlayan herkes örnek video ile çalıştırabilmeli; RTSP kodda desteklenir ama demo ona bağlı değildir |
| Repo | Tek amaçlı, bağımsız public repo | Dışarıdan bakan tek ve net bir proje görsün |
| Dashboard | FastAPI + MJPEG canlı akış + statik HTML/JS | Gerçek canlı video + REST API tek serviste karşılanır |
| Tracker | ByteTrack (ultralytics dahili) | DeepSORT ayrı ReID modeli + ek bağımlılık ister; ByteTrack CPU'da gerçek zamanlı kalır |
| Veritabanı | SQLite | MongoDB sunucu gerektirir, `clone && çalıştır` deneyimini bozar |
| Ekstralar | İnteraktif ROI aracı, Docker + Compose, pytest + GitHub Actions, README demo GIF | Hepsi seçildi |

**Kapsam dışı (YAGNI):** kullanıcı girişi/kimlik doğrulama, çoklu kamera, yüz tanıma veya kimliklendirme, bulut dağıtımı, model eğitimi (hazır YOLOv8 ağırlığı kullanılır).

## 3. Çalışma ortamı

- Python **3.13** ile proje içinde `.venv`. 3.14 tercih edilmiyor: torch/ultralytics wheel desteği 3.13'te güvenilir.
- Bağımlılıklar: `ultralytics`, `opencv-python`, `fastapi`, `uvicorn`, `numpy`, `pydantic-settings`, `python-dotenv`. Geliştirme: `pytest`, `httpx`, `ruff`.
- Ayarlar `.env` üzerinden (kaynak, model yolu, port, DB yolu); `.env.example` repoda, `.env` gitignore'da.

## 4. Mimari

```
alan-yogunluk-takibi/
├── occupancy/
│   ├── config.py        # .env + zones.json yükleme, tip doğrulaması
│   ├── video.py         # VideoSource: dosya | webcam | rtsp, reconnect        [Aşama 1]
│   ├── detection.py     # PersonDetector protokolü + YOLOv8 uygulaması        [Aşama 2]
│   ├── tracking.py      # ByteTrack ile kalıcı track ID                       [Aşama 3]
│   ├── zones.py         # Polygon ROI + çizgi geçiş (giriş/çıkış) mantığı     [Aşama 4]
│   ├── density.py       # heatmap birikimi + yoğunluk skoru                   [Aşama 4]
│   ├── storage.py       # SQLite şema, yazma/okuma                            [Aşama 6]
│   ├── reporting.py     # günlük JSON/CSV rapor üretimi                       [Aşama 7]
│   ├── logging_conf.py  # dosya + konsol logging                              [Aşama 8]
│   ├── overlay.py       # frame üzerine bbox/ID/ROI/sayaç çizimi
│   ├── pipeline.py      # modülleri bağlayan işleme döngüsü (ayrı thread)
│   └── __main__.py      # CLI giriş noktası: python -m occupancy --source ...
├── web/
│   ├── app.py           # FastAPI uygulaması                                  [Aşama 5]
│   └── static/          # index.html, app.js, style.css
├── tools/roi_ciz.py     # interaktif ROI çizim aracı
├── tests/
├── docker/Dockerfile, docker-compose.yml
├── .github/workflows/ci.yml
├── ornek/               # örnek video + örnek zones.json
└── README.md, requirements.txt, requirements-dev.txt, .env.example
```

Her modül tek sorumluluğa sahip ve kendi başına test edilebilir. `pipeline.py` dışındaki hiçbir modül diğerinin iç yapısını bilmez; iletişim veri sınıfları üzerinden olur.

### Ana veri tipleri

```python
Detection(bbox: tuple[int,int,int,int], confidence: float)
Track(track_id: int, bbox: ..., centroid: tuple[int,int], foot_point: tuple[int,int])
ZoneEvent(ts: datetime, track_id: int, event_type: Literal["enter","exit"], zone: str)
LiveStats(ts, per_zone_counts: dict[str,int], total_in: int, total_out: int,
          density: dict[str,float], fps: float, source_ok: bool)
```

## 5. Veri akışı

```
VideoSource → frame
   → PersonDetector → Detection[]
   → Tracker        → Track[]  (kalıcı ID)
   → ZoneManager    → hangi ROI içinde? çizgiyi geçti mi? → ZoneEvent[]
   → DensityMap     → heatmap günceller
   → Storage        → olayları anında, snapshot'ı periyodik (varsayılan 10 sn) yazar
   → overlay        → çizilmiş frame'i paylaşılan slot'a koyar
```

Pipeline kendi thread'inde koşar. FastAPI yalnızca "son çizilmiş frame" ve "son LiveStats" değerlerini kilitli bir paylaşım nesnesinden okur; ikisi birbirini bloklamaz. Frame kuyruğu tutulmaz — her zaman en güncel kare gösterilir (gerçek zamanlılık gecikmeye tercih edilir).

### Sayım mantığı

- **Anlık doluluk:** ROI poligonu içinde ayak noktası bulunan aktif track sayısı.
- **Giriş/çıkış:** `zones.json` içinde tanımlı yönlü çizgiyi (A→B) geçen track. Geçiş, track'in önceki ve şimdiki ayak noktasının çizginin hangi tarafında olduğuna bakılarak (işaretli alan / cross product) tespit edilir. Her `track_id` için her çizgide bir yön yalnızca **bir kez** sayılır → aynı kişi tekrar sayılmaz.
- **Yoğunluk:** ROI poligon alanına normalize edilmiş kişi sayısı (kişi / 1000 piksel²) + zaman serisi. Gerçek m² kalibrasyonu kapsam dışı, README'de belirtilir.
- **Heatmap:** ayak noktalarının Gauss çekirdeğiyle birikimli toplamı, renk haritasıyla PNG olarak sunulur.

## 6. Veritabanı şeması (SQLite)

```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, track_id INTEGER NOT NULL,
  event_type TEXT NOT NULL CHECK(event_type IN ('enter','exit')), zone TEXT NOT NULL);
CREATE TABLE snapshots (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, zone TEXT NOT NULL,
  count INTEGER NOT NULL, density REAL NOT NULL);
CREATE INDEX idx_events_ts ON events(ts);
CREATE INDEX idx_snapshots_ts ON snapshots(ts);
```

Zaman damgaları ISO-8601 UTC metin olarak saklanır.

## 7. HTTP arayüzü

| Yol | Döner |
|---|---|
| `GET /` | Dashboard sayfası |
| `GET /video_feed` | MJPEG akışı (`multipart/x-mixed-replace`) |
| `GET /api/live` | Anlık `LiveStats` JSON'u |
| `GET /api/history?minutes=30` | Snapshot zaman serisi |
| `GET /api/heatmap.png` | Birikimli yoğunluk haritası |
| `GET /api/report?date=YYYY-MM-DD&format=json\|csv` | Günlük rapor indirir |
| `GET /health` | Pipeline ve kaynak durumu |

Dashboard tek sayfa: solda canlı akış, sağda anlık doluluk / toplam giriş / toplam çıkış kartları, altta son 30 dakikanın doluluk grafiği (Chart.js gömülü, CDN yok) ve heatmap. `/api/live` 1 saniyede bir yoklanır.

## 8. Raporlama

Her gün için `reports/YYYY-MM-DD.json` ve `.csv`: toplam giriş, toplam çıkış, saatlik ortalama ve pik doluluk, pik saat, bölge bazında dağılım. Rapor veritabanından üretilir; hem CLI (`python -m occupancy.reporting --date ...`) hem `/api/report` üzerinden çağrılabilir.

## 9. Hata yönetimi

| Durum | Davranış |
|---|---|
| Video kaynağı açılamıyor / kopuyor | Artan bekleme (1→2→4→…→30 sn) ile yeniden bağlanma; `source_ok=false`; dashboard "kaynak bağlantısı yok" gösterir |
| Video dosyası bitti | Varsayılan olarak başa sarar (`--loop`), aksi halde temiz kapanış |
| Model ağırlığı yok | İlk çalıştırmada indirilir; indirilemezse net hata mesajı ve çıkış |
| DB yazma hatası | Loglanır, video akışı durmaz (izleme, kayıttan önce gelir) |
| Geçersiz `zones.json` | Başlangıçta doğrulanır, satır/alan belirten anlaşılır hata |
| Pipeline thread'i çöktü | Loglanır, `/health` bozuk döner, dashboard uyarı gösterir |

Tüm loglar `logs/app.log` (döngüsel) + konsol; seviye `.env` ile ayarlanır.

## 10. Test stratejisi

Testler **YOLO ağırlığı indirmeden ve GPU olmadan** çalışır: `PersonDetector` bir Protocol'dür, testlerde sahte detector kullanılır. Böylece CI hızlı ve deterministiktir.

- `zones`: nokta-poligon doğruluğu, çizgi geçiş yönü, aynı ID'nin tekrar sayılmaması, çizgiye teğet geçişler
- `density`: alan normalizasyonu, heatmap birikimi
- `storage`: yaz-oku roundtrip, şema oluşturma, eşzamanlı yazma
- `reporting`: bilinen olay setinden beklenen agregasyon, boş gün durumu
- `video`: sahte kaynak ile reconnect davranışı
- `web`: TestClient ile tüm endpoint'ler, pipeline yokken davranış
- `pipeline`: sahte detector + sentetik kareler ile uçtan uca "kişi girdi, sayaç arttı" senaryosu

CI (GitHub Actions, ubuntu-latest, Python 3.13): `ruff check` + `pytest`. README'de durum rozeti.

## 11. Docker

`python:3.13-slim` üzerine uygulama kodu ve `uvicorn` ile başlatma. Konteynerde GUI olmadığı için `requirements.txt` kurulduktan sonra `opencv-python` kaldırılıp `opencv-python-headless` kurulur (Dockerfile içinde tek satır); yerel kurulumda GUI'li sürüm kalır, çünkü ROI çizim aracı pencere açar. `docker-compose.yml` tek servis + `data/`, `reports/`, `logs/` için volume; port 8000. Webcam erişimi Linux'ta `devices:` ile, macOS'ta desteklenmediği README'de belirtilir (macOS'ta dosya/RTSP kaynağı kullanılır).

## 12. Teslim sırası

1. İskelet: paket yapısı, config, logging, requirements, .env.example, ilk commit
2. Video kaynağı + detection + tracking (ilk görünür çıktı: bbox'lı pencere)
3. Zones + sayım mantığı + testleri
4. Storage + reporting + testleri
5. FastAPI dashboard + MJPEG + statik arayüz + testleri
6. İnteraktif ROI çizim aracı
7. Docker + GitHub Actions CI
8. README + demo GIF + ekran görüntüleri, GitHub'a push

Her adım kendi başına çalışır durumda commit edilir.

## 13. Başarı ölçütü

- `git clone` → `pip install -r requirements.txt` → `python -m occupancy --source ornek/demo.mp4` → tarayıcıda `localhost:8000` canlı akış ve artan sayaçlar.
- `pytest` yeşil, CI yeşil.
- Bir günlük çalıştırma sonunda `reports/` altında dolu JSON ve CSV.
- README'yi okuyan biri projeyi 5 saniyede anlıyor (GIF), 5 dakikada çalıştırıyor.

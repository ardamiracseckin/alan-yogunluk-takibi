"""Aşama 5 — dashboard ve REST API.

Web katmanı işleme hattını *okur*, ona komut vermez. Hat kendi thread'inde
kareleri üretir; burada yapılan tek şey en son üretileni istemciye
aktarmak. Bu ayrım sayesinde yavaş bir tarayıcı işlemeyi yavaşlatamaz.

Canlı görüntü için MJPEG seçildi: WebRTC gibi bir çözüme göre kıyaslanamaz
derecede basit, hiçbir istemci kütüphanesi gerektirmiyor ve `<img>` etiketi
kadar az koda mal oluyor.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from occupancy.config import Settings
from occupancy.logging_conf import get_logger
from occupancy.pipeline import Pipeline
from occupancy.reporting import build_report, report_to_csv
from occupancy.storage import Storage

logger = get_logger(__name__)

STATIK_DIZIN = Path(__file__).parent / "static"
MJPEG_SINIRI = "frame"
KARE_ARALIGI_SN = 1 / 25  # akış hızı tavanı; işleme hızından bağımsız


def create_app(pipeline: Pipeline, settings: Settings) -> FastAPI:
    """İşleme hattını dışarı açan FastAPI uygulamasını kurar."""
    app = FastAPI(
        title="Alan Kullanım ve Yoğunluk Takibi",
        description="Kameradan gerçek zamanlı kişi sayımı, giriş/çıkış takibi ve yoğunluk analizi",
        version="0.1.0",
    )
    app.mount("/static", StaticFiles(directory=STATIK_DIZIN), name="static")

    # --- sayfa ----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIK_DIZIN / "index.html")

    # --- canlı görüntü ---------------------------------------------------

    async def kare_akisi() -> AsyncIterator[bytes]:
        onceki: bytes | None = None
        while pipeline.is_alive:
            kare = pipeline.latest_frame()
            if kare is not None and kare is not onceki:
                onceki = kare
                yield (
                    f"--{MJPEG_SINIRI}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(kare)}\r\n\r\n"
                ).encode() + kare + b"\r\n"
            await asyncio.sleep(KARE_ARALIGI_SN)

    @app.get("/video_feed", include_in_schema=False)
    async def video_feed() -> StreamingResponse:
        return StreamingResponse(
            kare_akisi(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_SINIRI}",
        )

    # --- veri uç noktaları -----------------------------------------------

    @app.get("/api/live", summary="Anlık doluluk, giriş/çıkış ve yoğunluk")
    async def api_live() -> dict:
        return pipeline.latest_stats().to_dict()

    @app.get("/api/history", summary="Son N dakikanın doluluk zaman serisi")
    async def api_history(minutes: int = Query(30, ge=0, le=24 * 60)) -> list[dict]:
        bitis = datetime.now(UTC)
        baslangic = bitis - timedelta(minutes=minutes)
        with Storage(settings.db_path) as depo:
            olcumler = depo.snapshots_between(baslangic, bitis)
        return [
            {
                "ts": olcum.ts.isoformat(),
                "zone": olcum.zone,
                "count": olcum.count,
                "density": olcum.density,
            }
            for olcum in olcumler
        ]

    @app.get("/api/heatmap.png", summary="Birikimli yoğunluk haritası")
    async def api_heatmap() -> Response:
        return Response(content=pipeline.heatmap_png(), media_type="image/png")

    @app.get("/api/report", summary="Günlük JSON veya CSV rapor")
    async def api_report(
        date_: str = Query(default=None, alias="date"),
        format: str = Query("json", pattern="^(json|csv)$"),
    ) -> Response:
        try:
            gun = date.fromisoformat(date_) if date_ else datetime.now(UTC).date()
        except ValueError as hata:
            raise HTTPException(
                status_code=400, detail=f"Geçersiz tarih: {date_!r}. Biçim: YYYY-AA-GG"
            ) from hata

        with Storage(settings.db_path) as depo:
            rapor = build_report(depo, gun)

        if format == "csv":
            return PlainTextResponse(
                report_to_csv(rapor),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{gun.isoformat()}.csv"'},
            )
        return JSONResponse(rapor)

    @app.get("/health", summary="Hattın ve video kaynağının durumu")
    async def health() -> JSONResponse:
        istatistik = pipeline.latest_stats()
        govde = {
            "pipeline_alive": pipeline.is_alive,
            "source_ok": istatistik.source_ok,
            "fps": round(istatistik.fps, 1),
            "error": str(pipeline.error) if pipeline.error else None,
        }
        return JSONResponse(govde, status_code=200 if pipeline.is_alive else 503)

    return app

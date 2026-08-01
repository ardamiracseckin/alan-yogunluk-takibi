"""Aşama 6 — veritabanı.

SQLite seçildi çünkü sunucu gerektirmez: repoyu klonlayan biri hiçbir servis
kurmadan sistemi çalıştırabilir. Yazma işini işleme hattının thread'i, okuma
işini web thread'i yapar; bu yüzden bağlantı `check_same_thread=False` ile
açılır ve tüm erişimler tek bir kilit altında serileştirilir.

Zaman damgaları ISO-8601 UTC metin olarak saklanır: hem okunabilir hem de
sözlüksel sıralaması zaman sıralamasıyla aynı.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from occupancy.logging_conf import get_logger
from occupancy.models import Snapshot, ZoneEvent

logger = get_logger(__name__)

SEMA = """
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    ts         TEXT    NOT NULL,
    track_id   INTEGER NOT NULL,
    event_type TEXT    NOT NULL CHECK (event_type IN ('enter', 'exit')),
    zone       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    id      INTEGER PRIMARY KEY,
    ts      TEXT    NOT NULL,
    zone    TEXT    NOT NULL,
    count   INTEGER NOT NULL,
    density REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_ts    ON events (ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots (ts);
"""


def _metne(ts: datetime) -> str:
    """Zaman damgasını UTC ISO-8601 metnine çevirir."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).isoformat()


def _tarihe(metin: str) -> datetime:
    return datetime.fromisoformat(metin)


class Storage:
    """Olayların ve doluluk ölçümlerinin kalıcı deposu."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    # --- yaşam döngüsü --------------------------------------------------

    def init_schema(self) -> None:
        """Tabloları ve indeksleri (yoksa) oluşturur. Tekrar çağrılabilir."""
        with self._lock:
            self._conn.executescript(SEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- yazma ----------------------------------------------------------

    def add_events(self, events: Sequence[ZoneEvent]) -> None:
        if not events:
            return
        satirlar = [(_metne(o.ts), o.track_id, o.event_type, o.zone) for o in events]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO events (ts, track_id, event_type, zone) VALUES (?, ?, ?, ?)",
                satirlar,
            )
            self._conn.commit()

    def add_snapshot(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO snapshots (ts, zone, count, density) VALUES (?, ?, ?, ?)",
                (_metne(snapshot.ts), snapshot.zone, snapshot.count, snapshot.density),
            )
            self._conn.commit()

    # --- okuma ----------------------------------------------------------

    def events_between(self, start: datetime, end: datetime) -> list[ZoneEvent]:
        """`start` dahil, `end` hariç aralıktaki olaylar, zaman sıralı."""
        satirlar = self._sorgula(
            "SELECT ts, track_id, event_type, zone FROM events "
            "WHERE ts >= ? AND ts < ? ORDER BY ts, id",
            (_metne(start), _metne(end)),
        )
        return [
            ZoneEvent(
                ts=_tarihe(satir["ts"]),
                track_id=satir["track_id"],
                event_type=satir["event_type"],
                zone=satir["zone"],
            )
            for satir in satirlar
        ]

    def snapshots_between(self, start: datetime, end: datetime) -> list[Snapshot]:
        """`start` dahil, `end` hariç aralıktaki doluluk ölçümleri, zaman sıralı."""
        satirlar = self._sorgula(
            "SELECT ts, zone, count, density FROM snapshots "
            "WHERE ts >= ? AND ts < ? ORDER BY ts, id",
            (_metne(start), _metne(end)),
        )
        return [
            Snapshot(
                ts=_tarihe(satir["ts"]),
                zone=satir["zone"],
                count=satir["count"],
                density=satir["density"],
            )
            for satir in satirlar
        ]

    def _sorgula(self, sql: str, parametreler: tuple) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, parametreler).fetchall()

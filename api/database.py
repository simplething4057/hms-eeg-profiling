from __future__ import annotations
"""SQLite 비동기 DB — 분석 결과 / 감사 로그 / 알람 저장
포트폴리오 목적: SQLite + aiosqlite (별도 서버 불필요)
실제 임상: PostgreSQL + append-only 감사 테이블로 교체
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import aiosqlite

DB_PATH = os.environ.get("HMS_DB_PATH", "hms_eeg.db")


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                username        TEXT    NOT NULL,
                filename        TEXT,
                duration_sec    REAL,
                n_windows       INTEGER DEFAULT 1,
                predicted_class TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                probabilities   TEXT    NOT NULL,
                r1b_flag        INTEGER DEFAULT 0,
                model_used      TEXT,
                processing_ms   REAL,
                scan_mode       INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                username    TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                detail      TEXT    DEFAULT '',
                ip_address  TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                analysis_id     INTEGER,
                username        TEXT    NOT NULL,
                filename        TEXT,
                alert_level     TEXT    NOT NULL,
                predicted_class TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                ack             INTEGER DEFAULT 0,
                ack_by          TEXT,
                ack_at          TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analysis_records(id)
            );
        """)
        await db.commit()


async def save_analysis(record: dict) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO analysis_records
               (created_at, username, filename, duration_sec, n_windows,
                predicted_class, confidence, probabilities, r1b_flag,
                model_used, processing_ms, scan_mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                record["username"],
                record.get("filename"),
                record.get("duration_sec"),
                record.get("n_windows", 1),
                record["predicted_class"],
                record["confidence"],
                json.dumps(record.get("probabilities", {})),
                int(bool(record.get("r1b_flag", False))),
                record.get("model_used"),
                record.get("processing_ms"),
                int(bool(record.get("scan_mode", False))),
            ),
        )
        await db.commit()
        return cur.lastrowid  # type: ignore[return-value]


async def save_audit(username: str, role: str, action: str, detail: str = "", ip: str = "") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO audit_logs (created_at, username, role, action, detail, ip_address) VALUES (?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), username, role, action, detail, ip),
        )
        await db.commit()


async def save_alert(analysis_id: int, record: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO alerts
               (created_at, analysis_id, username, filename, alert_level, predicted_class, confidence)
               VALUES (?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                analysis_id,
                record["username"],
                record.get("filename"),
                record["alert_level"],
                record["predicted_class"],
                record["confidence"],
            ),
        )
        await db.commit()


async def get_alerts(unread_only: bool = False, limit: int = 100) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if unread_only:
            cur = await db.execute(
                "SELECT * FROM alerts WHERE ack=0 ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        else:
            cur = await db.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in await cur.fetchall()]


async def ack_alert(alert_id: int, username: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE alerts SET ack=1, ack_by=?, ack_at=? WHERE id=? AND ack=0",
            (username, datetime.now(timezone.utc).isoformat(), alert_id),
        )
        await db.commit()
        return db.total_changes > 0


async def get_history(username: str | None = None, limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if username:
            cur = await db.execute(
                "SELECT * FROM analysis_records WHERE username=? ORDER BY created_at DESC LIMIT ?",
                (username, limit),
            )
        else:
            cur = await db.execute(
                "SELECT * FROM analysis_records ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = [dict(r) for r in await cur.fetchall()]
    for r in rows:
        try:
            r["probabilities"] = json.loads(r["probabilities"])
        except Exception:
            pass
    return rows

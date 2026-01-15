#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "app_state.sqlite"


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return r is not None


def ensure_offer_events(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS offer_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        offer_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        value TEXT,
        meta_json TEXT
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_offer_events_ts ON offer_events(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_offer_events_offer_id ON offer_events(offer_id)")
    conn.commit()


def column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH.resolve()}")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row

        if not table_exists(conn, "offer_state"):
            raise RuntimeError("Missing table offer_state")

        ensure_offer_events(conn)

        # Ensure expected columns exist
        for c in ["offer_id", "status", "contacted", "notes"]:
            if not column_exists(conn, "offer_state", c):
                raise RuntimeError(f"offer_state missing column: {c}")

        # We will backfill only if timestamp columns exist (recommended)
        has_status_ts = column_exists(conn, "offer_state", "status_updated_at")
        has_contact_ts = column_exists(conn, "offer_state", "contacted_updated_at")
        has_notes_ts = column_exists(conn, "offer_state", "notes_updated_at")

        # Pull all states
        rows = conn.execute("SELECT * FROM offer_state").fetchall()
        if not rows:
            print("No rows in offer_state. Nothing to backfill.")
            return

        inserted = 0

        # Insert one event per state field (if it has a timestamp and value is meaningful)
        for r in rows:
            offer_id = r["offer_id"]

            # 1) status event (only if not new)
            if has_status_ts:
                ts = (r["status_updated_at"] or "").strip()
                status = (r["status"] or "new").strip()
                if ts and status and status != "new":
                    conn.execute(
                        "INSERT INTO offer_events (ts, offer_id, event_type, value, meta_json) VALUES (?,?,?,?,?)",
                        (ts, offer_id, "status", status, "{}"),
                    )
                    inserted += 1

            # 2) contacted event (only if contacted=1)
            if has_contact_ts:
                ts = (r["contacted_updated_at"] or "").strip()
                contacted = int(r["contacted"] or 0)
                if ts and contacted == 1:
                    conn.execute(
                        "INSERT INTO offer_events (ts, offer_id, event_type, value, meta_json) VALUES (?,?,?,?,?)",
                        (ts, offer_id, "contacted", "1", "{}"),
                    )
                    inserted += 1

            # 3) notes event (only if notes non-empty)
            if has_notes_ts:
                ts = (r["notes_updated_at"] or "").strip()
                notes = (r["notes"] or "").strip()
                if ts and notes:
                    preview = notes.replace("\n", " ")[:80]
                    conn.execute(
                        "INSERT INTO offer_events (ts, offer_id, event_type, value, meta_json) VALUES (?,?,?,?,?)",
                        (ts, offer_id, "notes", preview, "{}"),
                    )
                    inserted += 1

        conn.commit()

        total_events = conn.execute("SELECT COUNT(*) FROM offer_events").fetchone()[0]
        print(f"Inserted {inserted} backfilled events.")
        print(f"offer_events now contains {total_events} rows.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

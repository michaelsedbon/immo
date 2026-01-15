#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ---- CONFIG ----
DB_PATH = Path(__file__).parent / "data" / "app_state.sqlite"

# Backfill time: yesterday 13:00 Paris time
PARIS_TZ = ZoneInfo("Europe/Paris")

# If you also want to backfill contacted/notes timestamps when missing, set True:
BACKFILL_CONTACTED_NOTES = False


def compute_target_ts_iso() -> str:
    now_paris = datetime.now(PARIS_TZ)
    yesterday = (now_paris.date() - timedelta(days=1))
    target = datetime(yesterday.year, yesterday.month, yesterday.day, 13, 0, 0, tzinfo=PARIS_TZ)
    # Keep timezone in the stored string (e.g. 2026-01-14T13:00:00+01:00)
    return target.isoformat(timespec="seconds")


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    cols = {r[1] for r in rows}  # r[1] = column name
    return column in cols


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH.resolve()}")

    target_ts = compute_target_ts_iso()
    print(f"Target timestamp: {target_ts} (Europe/Paris, yesterday 13:00)")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row

        # Safety: ensure columns exist
        required = ["status", "status_updated_at"]
        for col in required:
            if not column_exists(conn, "offer_state", col):
                raise RuntimeError(f"Missing column '{col}' in offer_state. Did you run the schema migration?")

        # Update status_updated_at where status != 'new' and missing/empty
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE offer_state
            SET status_updated_at = ?
            WHERE status != 'new'
              AND (status_updated_at IS NULL OR TRIM(status_updated_at) = '')
            """,
            (target_ts,),
        )
        n1 = cur.rowcount

        n2 = 0
        n3 = 0
        if BACKFILL_CONTACTED_NOTES:
            # Backfill contacted_updated_at if contacted=1 and missing
            if column_exists(conn, "offer_state", "contacted") and column_exists(conn, "offer_state", "contacted_updated_at"):
                cur.execute(
                    """
                    UPDATE offer_state
                    SET contacted_updated_at = ?
                    WHERE contacted = 1
                      AND (contacted_updated_at IS NULL OR TRIM(contacted_updated_at) = '')
                    """,
                    (target_ts,),
                )
                n2 = cur.rowcount

            # Backfill notes_updated_at if notes not empty and missing
            if column_exists(conn, "offer_state", "notes") and column_exists(conn, "offer_state", "notes_updated_at"):
                cur.execute(
                    """
                    UPDATE offer_state
                    SET notes_updated_at = ?
                    WHERE notes IS NOT NULL
                      AND TRIM(notes) != ''
                      AND (notes_updated_at IS NULL OR TRIM(notes_updated_at) = '')
                    """,
                    (target_ts,),
                )
                n3 = cur.rowcount

        conn.commit()

        print(f"Backfilled status_updated_at for {n1} rows (status != 'new').")
        if BACKFILL_CONTACTED_NOTES:
            print(f"Backfilled contacted_updated_at for {n2} rows (contacted=1).")
            print(f"Backfilled notes_updated_at for {n3} rows (notes not empty).")

        # Quick peek
        sample = conn.execute(
            """
            SELECT offer_id, status, status_updated_at
            FROM offer_state
            WHERE status != 'new'
            ORDER BY status_updated_at DESC
            LIMIT 5
            """
        ).fetchall()
        print("\nSample rows:")
        for r in sample:
            print(dict(r))

    finally:
        conn.close()


if __name__ == "__main__":
    main()

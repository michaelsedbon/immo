# app.py (paste as-is)
from __future__ import annotations

from pathlib import Path
import re
import math
import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urljoin, urlparse, urlencode

import pandas as pd
import requests
import plotly.express as px
import plotly.io as pio
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect

# -----------------------------------------------------------------------------
# Paths / app
# -----------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SCRAPES_DIR = APP_DIR / "scrapes" / "figaro"
SCRAPES_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "app_state.sqlite"

FIGARO_BASE_URL = "https://immobilier.lefigaro.fr"
PARIS_ARR_GEOJSON = APP_DIR / "static" / "geo" / "paris_arrondissements.geojson"

MARKET_BASELINE_CSV = DATA_DIR / "market_baseline_eurm2.csv"

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
PARIS_TZ = ZoneInfo("Europe/Paris")

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def now_paris_iso() -> str:
    return datetime.now(PARIS_TZ).isoformat(timespec="seconds")

# -----------------------------------------------------------------------------
# DB helpers / schema + migrations
# -----------------------------------------------------------------------------
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}  # column name

def migrate_offer_events(conn: sqlite3.Connection):
    """
    Your DB might already have an older offer_events schema (e.g. column 'ts').
    This migration adds the newer columns ts_utc/ts_paris if missing and backfills.
    """
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "offer_events" not in tables:
        return

    cols = table_columns(conn, "offer_events")

    if "ts_utc" not in cols:
        conn.execute("ALTER TABLE offer_events ADD COLUMN ts_utc TEXT")
    if "ts_paris" not in cols:
        conn.execute("ALTER TABLE offer_events ADD COLUMN ts_paris TEXT")

    cols = table_columns(conn, "offer_events")

    # Backfill from legacy 'ts' if it exists
    if "ts" in cols:
        conn.execute("""
            UPDATE offer_events
            SET ts_utc = COALESCE(ts_utc, ts),
                ts_paris = COALESCE(ts_paris, ts)
        """)
    else:
        # At least ensure ts_paris is filled when ts_utc exists
        conn.execute("""
            UPDATE offer_events
            SET ts_paris = COALESCE(ts_paris, ts_utc)
            WHERE ts_paris IS NULL
        """)

def ensure_schema():
    with db_conn() as conn:
        cur = conn.cursor()

        # State table: current state per offer
        cur.execute("""
        CREATE TABLE IF NOT EXISTS offer_state (
            offer_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'new',         -- new | kept | discarded | visit_scheduled
            contacted INTEGER DEFAULT 0,       -- 0/1
            notes TEXT DEFAULT '',
            status_updated_at TEXT,
            contacted_updated_at TEXT,
            notes_updated_at TEXT,
            updated_at TEXT
        )
        """)

        # Events table (new schema). If it already exists with a different schema,
        # CREATE IF NOT EXISTS will not change it — migration below will patch it.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS offer_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            offer_id TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            ts_paris TEXT NOT NULL,
            event_type TEXT NOT NULL,     -- status | contacted | notes
            value TEXT,
            meta_json TEXT
        )
        """)

        # Manual offers: offers added via URL, persisted
        cur.execute("""
        CREATE TABLE IF NOT EXISTS manual_offers (
            offer_id TEXT PRIMARY KEY,
            source TEXT,
            url TEXT,
            name TEXT,
            price_eur REAL,
            surface_m2 REAL,
            rooms INTEGER,
            arrondissement INTEGER,
            eur_m2 REAL,
            published_date TEXT,
            source_file TEXT,
            raw_json TEXT,
            created_at TEXT
        )
        """)

        conn.commit()
        migrate_offer_events(conn)
        conn.commit()

ensure_schema()

# -----------------------------------------------------------------------------
# Offer ID
# -----------------------------------------------------------------------------
def compute_offer_id(row: dict) -> str:
    url = (row.get("url") or "").strip()
    if url:
        return "url:" + url

    key = "|".join([
        str(row.get("name","")).strip(),
        str(row.get("price_eur","")).strip(),
        str(row.get("surface_m2","")).strip(),
        str(row.get("rooms","")).strip(),
        str(row.get("arrondissement","")).strip(),
        str(row.get("source_file","")).strip(),
    ])
    return "hash:" + hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()

# -----------------------------------------------------------------------------
# State + event logging
# -----------------------------------------------------------------------------
def get_states_map(offer_ids):
    if not offer_ids:
        return {}
    qmarks = ",".join(["?"] * len(offer_ids))
    with db_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM offer_state WHERE offer_id IN ({qmarks})",
            offer_ids
        ).fetchall()
    return {r["offer_id"]: dict(r) for r in rows}

def log_event(conn: sqlite3.Connection, offer_id: str, event_type: str, value: str | None, meta: dict | None = None):
    """
    Insert into offer_events, supporting any of these schemas:
      - legacy:  (offer_id, ts, event_type, value, meta_json) with ts NOT NULL
      - new:     (offer_id, ts_utc, ts_paris, event_type, value, meta_json)
      - hybrid:  may contain ts + ts_utc + ts_paris (ts sometimes NOT NULL)
    We always populate every timestamp column that exists.
    """
    cols = table_columns(conn, "offer_events")

    tsu = now_utc_iso()
    tsp = now_paris_iso()
    mj = json.dumps(meta or {}, ensure_ascii=False)
    val = value if value is not None else ""

    insert_cols = ["offer_id"]
    insert_vals = [offer_id]

    # If legacy ts exists (often NOT NULL), ALWAYS provide it.
    if "ts" in cols:
        insert_cols.append("ts")
        insert_vals.append(tsu)

    # If newer columns exist, also provide them.
    if "ts_utc" in cols:
        insert_cols.append("ts_utc")
        insert_vals.append(tsu)

    if "ts_paris" in cols:
        insert_cols.append("ts_paris")
        insert_vals.append(tsp)

    # Required columns
    insert_cols += ["event_type", "value", "meta_json"]
    insert_vals += [event_type, val, mj]

    placeholders = ",".join(["?"] * len(insert_cols))
    sql = f"INSERT INTO offer_events ({','.join(insert_cols)}) VALUES ({placeholders})"
    conn.execute(sql, tuple(insert_vals))
    
def upsert_state(offer_id: str, *, status=None, contacted=None, notes=None, meta: dict | None = None):
    """
    Updates offer_state and logs offer_events for fields that were provided AND actually changed.
    """
    ts_utc = now_utc_iso()

    with db_conn() as conn:
        existing = conn.execute(
            "SELECT offer_id, status, contacted, notes FROM offer_state WHERE offer_id=?",
            (offer_id,)
        ).fetchone()

        if existing is None:
            init_status = status if status is not None else "new"
            init_contacted = int(contacted) if contacted is not None else 0
            init_notes = notes if notes is not None else ""

            conn.execute("""
                INSERT INTO offer_state
                (offer_id, status, contacted, notes,
                 status_updated_at, contacted_updated_at, notes_updated_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                offer_id,
                init_status,
                init_contacted,
                init_notes,
                ts_utc if status is not None else None,
                ts_utc if contacted is not None else None,
                ts_utc if notes is not None else None,
                ts_utc
            ))

            if status is not None:
                log_event(conn, offer_id, "status", str(init_status), meta)
            if contacted is not None:
                log_event(conn, offer_id, "contacted", "1" if init_contacted else "0", meta)
            if notes is not None:
                preview = (init_notes or "")[:120]
                log_event(conn, offer_id, "notes", preview, meta)

        else:
            old_status = existing["status"]
            old_contacted = int(existing["contacted"])
            old_notes = existing["notes"] or ""

            new_status = old_status if status is None else status
            new_contacted = old_contacted if contacted is None else int(contacted)
            new_notes = old_notes if notes is None else (notes or "")

            status_changed = (status is not None) and (new_status != old_status)
            contacted_changed = (contacted is not None) and (new_contacted != old_contacted)
            notes_changed = (notes is not None) and (new_notes != old_notes)

            conn.execute("""
                UPDATE offer_state
                SET status=?, contacted=?, notes=?,
                    status_updated_at=COALESCE(?, status_updated_at),
                    contacted_updated_at=COALESCE(?, contacted_updated_at),
                    notes_updated_at=COALESCE(?, notes_updated_at),
                    updated_at=?
                WHERE offer_id=?
            """, (
                new_status,
                new_contacted,
                new_notes,
                ts_utc if status_changed else None,
                ts_utc if contacted_changed else None,
                ts_utc if notes_changed else None,
                ts_utc,
                offer_id
            ))

            if status_changed:
                log_event(conn, offer_id, "status", str(new_status), meta)
            if contacted_changed:
                log_event(conn, offer_id, "contacted", "1" if new_contacted else "0", meta)
            if notes_changed:
                preview = (new_notes or "")[:120]
                log_event(conn, offer_id, "notes", preview, meta)

        conn.commit()

# -----------------------------------------------------------------------------
# Market baseline
# -----------------------------------------------------------------------------
def load_market_baseline_map():
    if not MARKET_BASELINE_CSV.exists():
        return {}
    base = pd.read_csv(MARKET_BASELINE_CSV)
    if "arrondissement" not in base.columns or "market_eur_m2" not in base.columns:
        raise ValueError("market_baseline_eurm2.csv must contain: arrondissement, market_eur_m2")
    base = base.dropna(subset=["arrondissement", "market_eur_m2"]).copy()
    base["arrondissement"] = base["arrondissement"].astype(int)
    base["market_eur_m2"] = pd.to_numeric(base["market_eur_m2"], errors="coerce")
    base = base.dropna(subset=["market_eur_m2"])
    return dict(zip(base["arrondissement"], base["market_eur_m2"]))

# -----------------------------------------------------------------------------
# Scrape parsing helpers (Figaro list pages)
# -----------------------------------------------------------------------------
def _parse_int_fr(s: str):
    if s is None:
        return None
    s = str(s)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

def _extract_arrondissement(txt: str):
    if not txt:
        return None
    m = re.search(r"Paris\s+(\d{1,2})\s*(?:e|ème|eme)\b", txt, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _extract_date(txt: str):
    if not txt:
        return None
    m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", txt)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", txt)
    if m:
        return m.group(1)
    return None

def _parse_summary_line(txt: str):
    out = {
        "name": None,
        "price_eur": None,
        "surface_m2": None,
        "rooms": None,
        "arrondissement": None,
        "eur_m2": None,
        "published_date": None,
    }
    t = " ".join(txt.split())
    out["arrondissement"] = _extract_arrondissement(t)

    m = re.search(r"([\d\s]+)\s*€", t)
    out["price_eur"] = _parse_int_fr(m.group(0)) if m else None

    m = re.search(r"([\d\s]+)\s*€/m²", t)
    out["eur_m2"] = _parse_int_fr(m.group(0)) if m else None

    m = re.search(r"(\d+)\s+pièces?", t, flags=re.IGNORECASE)
    out["rooms"] = int(m.group(1)) if m else None

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m²", t, flags=re.IGNORECASE)
    out["surface_m2"] = float(m.group(1).replace(",", ".")) if m else None

    out["published_date"] = _extract_date(t)
    out["name"] = t
    return out

def load_figaro_offers_from_folder(folder: Path) -> pd.DataFrame:
    folder = Path(folder)
    files = sorted(folder.glob("*.html"))
    rows = []

    for fp in files:
        html = fp.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.find_all("a"):
            txt = " ".join(a.get_text(" ", strip=True).split())
            if not txt:
                continue
            if ("€" in txt) and ("m²" in txt) and ("Paris" in txt):
                d = _parse_summary_line(txt)
                if d["price_eur"] is None and d["eur_m2"] is None:
                    continue

                href = a.get("href")
                url = urljoin(FIGARO_BASE_URL, href) if href else None

                parent_txt = ""
                parent = a.parent
                if parent:
                    parent_txt = " ".join(parent.get_text(" ", strip=True).split())
                if not d["published_date"]:
                    d["published_date"] = _extract_date(parent_txt)

                name = a.get("title") or a.get("aria-label")
                d["name"] = name.strip() if name else d["name"]

                d["url"] = url
                d["source_file"] = fp.name
                rows.append(d)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url", "name", "price_eur", "surface_m2", "source_file"])
    if df.empty:
        return df

    df["eur_m2"] = pd.to_numeric(df["eur_m2"], errors="coerce")
    df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    df["arrondissement"] = pd.to_numeric(df["arrondissement"], errors="coerce")

    mask = df["eur_m2"].isna() & df["price_eur"].notna() & df["surface_m2"].notna() & (df["surface_m2"] > 0)
    df.loc[mask, "eur_m2"] = df.loc[mask, "price_eur"] / df.loc[mask, "surface_m2"]

    df = df[df["arrondissement"].between(1, 20)]
    df = df[df["eur_m2"].between(1000, 50000)]
    df = df[df["price_eur"].between(10_000, 50_000_000)]
    df["published_date"] = df["published_date"].fillna("")
    return df.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Manual offers via URL (Figaro detection + parse)
# -----------------------------------------------------------------------------
def detect_source(url: str) -> str | None:
    host = (urlparse(url).netloc or "").lower()
    if "lefigaro.fr" in host:
        return "figaro"
    return None

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ParisApp/1.0)",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.7",
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.text

def parse_figaro_listing_page(url: str, html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    og = soup.select_one('meta[property="og:title"]')
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
    if not title:
        title = url

    text = " ".join(soup.get_text(" ", strip=True).split())
    d = _parse_summary_line(text)
    d["name"] = title
    d["url"] = url
    d["source_file"] = "manual_url"

    if (d.get("eur_m2") is None) and d.get("price_eur") and d.get("surface_m2"):
        if d["surface_m2"] and d["surface_m2"] > 0:
            d["eur_m2"] = d["price_eur"] / d["surface_m2"]

    return d

def upsert_manual_offer(row: dict, source: str) -> str:
    row = dict(row)
    row["source"] = source
    offer_id = compute_offer_id(row)
    row["offer_id"] = offer_id

    payload = {
        "offer_id": offer_id,
        "source": source,
        "url": row.get("url"),
        "name": row.get("name"),
        "price_eur": row.get("price_eur"),
        "surface_m2": row.get("surface_m2"),
        "rooms": row.get("rooms"),
        "arrondissement": row.get("arrondissement"),
        "eur_m2": row.get("eur_m2"),
        "published_date": row.get("published_date") or "",
        "source_file": row.get("source_file") or "manual_url",
        "raw_json": json.dumps(row, ensure_ascii=False),
        "created_at": now_utc_iso(),
    }

    with db_conn() as conn:
        conn.execute("""
            INSERT INTO manual_offers
            (offer_id, source, url, name, price_eur, surface_m2, rooms, arrondissement, eur_m2,
             published_date, source_file, raw_json, created_at)
            VALUES
            (:offer_id, :source, :url, :name, :price_eur, :surface_m2, :rooms, :arrondissement, :eur_m2,
             :published_date, :source_file, :raw_json, :created_at)
            ON CONFLICT(offer_id) DO UPDATE SET
                source=excluded.source,
                url=excluded.url,
                name=excluded.name,
                price_eur=excluded.price_eur,
                surface_m2=excluded.surface_m2,
                rooms=excluded.rooms,
                arrondissement=excluded.arrondissement,
                eur_m2=excluded.eur_m2,
                published_date=excluded.published_date,
                source_file=excluded.source_file,
                raw_json=excluded.raw_json
        """, payload)
        conn.commit()

    return offer_id

def load_manual_offers_df() -> pd.DataFrame:
    with db_conn() as conn:
        rows = conn.execute("SELECT * FROM manual_offers").fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    for c in ["price_eur", "surface_m2", "eur_m2", "arrondissement", "rooms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------------------------------------------------------
# Search across offers
# -----------------------------------------------------------------------------
def apply_offer_search(df: pd.DataFrame, q: str) -> pd.DataFrame:
    q = (q or "").strip()
    if df.empty or not q:
        return df

    needle = q.lower()
    cols = ["url", "offer_id", "name", "notes", "source_file", "published_date"]
    present = [c for c in cols if c in df.columns]
    if not present:
        return df

    mask = False
    for c in present:
        mask = mask | df[c].fillna("").astype(str).str.lower().str.contains(re.escape(needle), regex=True)

    return df[mask].copy()

# -----------------------------------------------------------------------------
# Geojson helper + figures for offers
# -----------------------------------------------------------------------------
def _guess_featureidkey(geojson: dict) -> str:
    feats = geojson.get("features") or []
    if not feats:
        raise ValueError("GeoJSON has no features.")
    props = feats[0].get("properties") or {}
    candidates = ["c_ar", "arrondissement", "arrond", "code", "id", "c_arinsee", "numero", "num", "insee"]
    for k in candidates:
        if k in props:
            return f"properties.{k}"
    for k, v in props.items():
        s = str(v).strip()
        if s.isdigit() and 1 <= int(s) <= 20:
            return f"properties.{k}"
    raise ValueError(f"Could not guess arrondissement key. Props: {list(props.keys())}")

def make_offers_map_and_bar(arr_counts_df: pd.DataFrame, title_prefix="Offers"):
    bar_fig = px.bar(
        arr_counts_df.sort_values("arrondissement"),
        x="arrondissement",
        y="n_offers",
        labels={"arrondissement": "Arrondissement", "n_offers": "# offers"},
        title=f"{title_prefix} — offers per arrondissement",
    )
    bar_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)

    if not PARIS_ARR_GEOJSON.exists():
        map_html = (
            "<div class='border rounded p-3 text-muted small'>"
            "Missing GeoJSON: static/geo/paris_arrondissements.geojson"
            "</div>"
        )
        bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs=False)
        return map_html, bar_html

    geojson = json.loads(PARIS_ARR_GEOJSON.read_text(encoding="utf-8"))
    featureidkey = _guess_featureidkey(geojson)

    map_fig = px.choropleth_mapbox(
        arr_counts_df,
        geojson=geojson,
        locations="arrondissement",
        featureidkey=featureidkey,
        color="n_offers",
        mapbox_style="carto-positron",
        center={"lat": 48.8566, "lon": 2.3522},
        zoom=10.6,
        opacity=0.65,
        labels={"n_offers": "# offers"},
        title=f"{title_prefix} — map of offers density",
    )
    map_fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)

    map_html = pio.to_html(map_fig, full_html=False, include_plotlyjs=False)
    bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs=False)
    return map_html, bar_html

# -----------------------------------------------------------------------------
# Home (your snippet-based dashboard)
# -----------------------------------------------------------------------------
PLOTS_DIR = DATA_DIR

def load_snippet(name: str) -> str:
    fp = PLOTS_DIR / name
    return fp.read_text(encoding="utf-8", errors="ignore") if fp.exists() else ""

@app.route("/")
def home():
    sections = [
        {
            "kicker": "Snapshot",
            "title": "Median sale prices by arrondissement",
            "body": "2024 + 2025 semester 1 median sold price per arrondissement.",
            "note": "",
            "plot_html": load_snippet("current_medians.html"),
        },
        {
            "kicker": "Activity",
            "title": "Sales volumes by arrondissement",
            "body": "Volume helps interpret confidence in medians.",
            "note": "",
            "plot_html": load_snippet("current_sale_volumes.html"),
        },
    ]
    return render_template("home.html", sections=sections)

# -----------------------------------------------------------------------------
# Redirect helper for POST actions
# -----------------------------------------------------------------------------
def _redir(back: str, **params):
    if not back or back == "." or not back.startswith("/"):
        back = "/offers"
    if not params:
        return redirect(back)
    sep = "&" if "?" in back else "?"
    return redirect(back + sep + urlencode(params))

# -----------------------------------------------------------------------------
# Add offer by URL
# -----------------------------------------------------------------------------
@app.post("/offers/add_url")
def offers_add_url():
    url = (request.form.get("url") or "").strip()
    back = (request.form.get("back") or "").strip()

    if not url:
        return _redir(back, msg="Empty URL")

    source = detect_source(url)
    if source != "figaro":
        return _redir(back, msg="Unsupported source (only Figaro for now)")

    try:
        html = fetch_html(url)  # can fail due to blocking
        row = parse_figaro_listing_page(url, html)
        offer_id = upsert_manual_offer(row, source=source)
        upsert_state(offer_id)  # ensure state exists
        return _redir(back, added=1, msg="Offer added", q=url)
    except Exception as e:
        return _redir(back, err=1, msg=f"Add failed: {type(e).__name__}")

# -----------------------------------------------------------------------------
# OFFERS page
# -----------------------------------------------------------------------------
@app.route("/offers")
def offers():
    df_scraped = load_figaro_offers_from_folder(SCRAPES_DIR)
    df_manual = load_manual_offers_df()

    if df_scraped.empty and df_manual.empty:
        df = pd.DataFrame()
    elif df_scraped.empty:
        df = df_manual.copy()
    elif df_manual.empty:
        df = df_scraped.copy()
    else:
        df = pd.concat([df_scraped, df_manual], ignore_index=True, sort=False)

    # ---- filters ----
    arr_list  = request.args.getlist("arr")
    rooms     = request.args.get("rooms", "").strip()
    price_min = request.args.get("price_min", "").strip()
    price_max = request.args.get("price_max", "").strip()
    surf_min  = request.args.get("surf_min", "").strip()
    surf_max  = request.args.get("surf_max", "").strip()
    q         = (request.args.get("q", "") or "").strip()

    status_list = request.args.getlist("status")  # supports visit_scheduled automatically
    contacted_only = (request.args.get("contacted_only", "").strip() == "1")

    # ---- sorting ----
    sort_by  = request.args.get("sort", "published_date").strip()
    sort_dir = request.args.get("dir", "desc").strip().lower()
    allowed_sort = {
        "name", "price_eur", "surface_m2", "eur_m2", "premium_pct",
        "rooms", "arrondissement", "published_date", "status", "contacted"
    }
    if sort_by not in allowed_sort:
        sort_by = "published_date"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    ascending = (sort_dir == "asc")

    # ---- baseline premium ----
    baseline_map = load_market_baseline_map()
    baseline_loaded = len(baseline_map) > 0
    if not df.empty and baseline_loaded:
        df = df.copy()
        df["arrondissement"] = pd.to_numeric(df.get("arrondissement"), errors="coerce")
        df["market_eur_m2"] = df["arrondissement"].map(baseline_map)
        df["eur_m2"] = pd.to_numeric(df.get("eur_m2"), errors="coerce")
        df["premium_pct"] = (df["eur_m2"] / df["market_eur_m2"] - 1.0) * 100.0

    # ---- numeric filters (pre-state) ----
    if not df.empty:
        for c in ["arrondissement", "rooms", "price_eur", "surface_m2", "eur_m2"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if arr_list:
            arr_ints = [int(a) for a in arr_list if str(a).isdigit()]
            df = df[df["arrondissement"].isin(arr_ints)]
        if rooms:
            df = df[df["rooms"] == int(rooms)]
        if price_min:
            df = df[df["price_eur"] >= int(price_min)]
        if price_max:
            df = df[df["price_eur"] <= int(price_max)]
        if surf_min:
            df = df[df["surface_m2"] >= float(surf_min)]
        if surf_max:
            df = df[df["surface_m2"] <= float(surf_max)]

    # ---- attach offer_id + persisted state ----
    if not df.empty:
        df = df.copy()
        if "offer_id" not in df.columns or df["offer_id"].isna().all():
            df["offer_id"] = df.apply(lambda r: compute_offer_id(r.to_dict()), axis=1)
        else:
            df["offer_id"] = df["offer_id"].fillna(df.apply(lambda r: compute_offer_id(r.to_dict()), axis=1))

        states = get_states_map(df["offer_id"].tolist())

        def attach_state(row):
            st = states.get(row["offer_id"])
            if not st:
                return pd.Series({"status": "new", "contacted": 0, "notes": ""})
            return pd.Series({
                "status": st.get("status", "new"),
                "contacted": int(st.get("contacted", 0) or 0),
                "notes": st.get("notes", "") or ""
            })

        df[["status", "contacted", "notes"]] = df.apply(attach_state, axis=1)
    else:
        df = df.copy()
        df["offer_id"] = pd.Series(dtype=str)
        df["status"] = pd.Series(dtype=str)
        df["contacted"] = pd.Series(dtype=int)
        df["notes"] = pd.Series(dtype=str)

    # ---- status/contacted filters ----
    if not df.empty:
        if status_list:
            df = df[df["status"].isin(status_list)]
        if contacted_only:
            df = df[df["contacted"] == 1]

    # ---- free-text search ----
    df = apply_offer_search(df, q)

    # ---- charts from CURRENT FILTERED set ----
    if not df.empty and "arrondissement" in df.columns:
        dff = df.dropna(subset=["arrondissement"]).copy()
        dff = dff[dff["arrondissement"].between(1, 20)]
        arr_counts = dff.groupby("arrondissement").size().reset_index(name="n_offers")
        arr_counts["arrondissement"] = arr_counts["arrondissement"].astype(int).astype(str)
        offers_map_html, offers_bar_html = make_offers_map_and_bar(arr_counts, title_prefix="Filtered")
    else:
        offers_map_html, offers_bar_html = "", ""

    # ---- sort ----
    if not df.empty and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, na_position="last", kind="mergesort")

    # ---- pagination ----
    PAGE_SIZE = 30
    total = len(df)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = int(request.args.get("page", "1") or 1)
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = df.iloc[start:end].copy()

    qs = request.args.to_dict(flat=False)
    if "page" not in qs:
        qs["page"] = [str(page)]

    current_url = request.full_path
    if current_url.endswith("?"):
        current_url = current_url[:-1]

    return render_template(
        "offers.html",
        offers=page_df.to_dict(orient="records"),
        total=total,
        page=page,
        total_pages=total_pages,
        arr_options=list(range(1, 21)),
        room_options=[1, 2, 3, 4, 5],
        baseline_loaded=baseline_loaded,
        qs=qs,
        current_url=current_url,
        offers_map_html=offers_map_html,
        offers_bar_html=offers_bar_html,
        f={
            "arr": arr_list,
            "rooms": rooms,
            "price_min": price_min,
            "price_max": price_max,
            "surf_min": surf_min,
            "surf_max": surf_max,
            "status": status_list,
            "contacted_only": "1" if contacted_only else "",
            "q": q,
            "sort": sort_by,
            "dir": sort_dir,
        }
    )

# -----------------------------------------------------------------------------
# Interaction routes
# -----------------------------------------------------------------------------
@app.post("/offers/set_status")
def offers_set_status():
    offer_id = request.form.get("offer_id", "")
    status = request.form.get("status", "")
    back = (request.form.get("back") or "").strip()

    if not back or back == ".":
        back = "/offers"
    if not back.startswith("/"):
        back = "/offers"

    # ✅ includes visit_scheduled
    if offer_id and status in {"new", "kept", "discarded", "visit_scheduled"}:
        upsert_state(offer_id, status=status, meta={"route": "set_status"})

    return redirect(back)

@app.post("/offers/toggle_contacted")
def offers_toggle_contacted():
    offer_id = request.form.get("offer_id", "")
    contacted = request.form.get("contacted", "0")
    back = (request.form.get("back") or "").strip()

    if not back or back == ".":
        back = "/offers"
    if not back.startswith("/"):
        back = "/offers"

    if offer_id:
        upsert_state(offer_id, contacted=(contacted == "1"), meta={"route": "toggle_contacted"})

    return redirect(back)

@app.post("/offers/save_notes")
def offers_save_notes():
    offer_id = request.form.get("offer_id", "")
    notes = request.form.get("notes", "")
    back = (request.form.get("back") or "").strip()

    if not back or back == ".":
        back = "/offers"
    if not back.startswith("/"):
        back = "/offers"

    if offer_id:
        upsert_state(offer_id, notes=notes, meta={"route": "save_notes"})

    return redirect(back)

# -----------------------------------------------------------------------------
# Timeline
# -----------------------------------------------------------------------------
@app.route("/timeline")
def timeline():
    days = (request.args.get("days", "30") or "30").strip()
    try:
        days_i = max(1, min(3650, int(days)))
    except Exception:
        days_i = 30

    with db_conn() as conn:
        cols = table_columns(conn, "offer_events")
        if "ts_utc" in cols:
            rows = conn.execute("""
                SELECT ts_utc AS ts, offer_id, event_type, value, meta_json
                FROM offer_events
                ORDER BY ts_utc ASC
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT ts AS ts, offer_id, event_type, value, meta_json
                FROM offer_events
                ORDER BY ts ASC
            """).fetchall()

    if not rows:
        fig_html = "<div class='text-muted small'>No events yet. Change some statuses in Offers first.</div>"
        return render_template("timeline.html", fig_html=fig_html, days=days_i, bar_html="")

    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_i)
    dfw = df[df["ts"] >= cutoff].copy()

    if dfw.empty:
        fig_html = "<div class='text-muted small'>No events in this window.</div>"
        return render_template("timeline.html", fig_html=fig_html, days=days_i, bar_html="")

    # Barplot: contacted ON per day (Paris time)
    contacted = dfw[(dfw["event_type"] == "contacted") & (dfw["value"].astype(str) == "1")].copy()
    if not contacted.empty:
        contacted["day_paris"] = contacted["ts"].dt.tz_convert("Europe/Paris").dt.date
        daily = contacted.groupby("day_paris").size().reset_index(name="n_contacted").sort_values("day_paris")
        daily["day_paris"] = daily["day_paris"].astype(str)

        bar_fig = px.bar(
            daily, x="day_paris", y="n_contacted",
            title="Contacted per day",
            labels={"day_paris": "Day (Paris)", "n_contacted": "# contacted"},
        )
        bar_fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
        bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs="cdn")
    else:
        bar_html = "<div class='text-muted small'>No contacted events (ON) in this window.</div>"

    def color_key(r):
        if r["event_type"] == "status":
            if r["value"] == "kept":
                return "kept"
            if r["value"] == "discarded":
                return "discarded"
            if r["value"] == "visit_scheduled":
                return "visit_scheduled"
            return "new"
        if r["event_type"] == "contacted":
            return "contacted"
        return "notes"

    dfw["color"] = dfw.apply(color_key, axis=1)

    dfw["offer_url"] = dfw["offer_id"].astype(str).str.replace("^url:", "", regex=True)
    dfw["offer_label"] = (
        dfw["offer_url"].str.replace(r"/+$", "", regex=True).str.split("/").str[-1]
    )
    bad = dfw["offer_label"].isna() | (dfw["offer_label"].str.len() < 6)
    dfw.loc[bad, "offer_label"] = dfw.loc[bad, "offer_url"].str[-16:]

    fig = px.scatter(
        dfw,
        x="ts",
        y="offer_label",
        color="color",
        color_discrete_map={
            "kept": "#198754",
            "discarded": "#dc3545",
            "visit_scheduled": "#ffc107",  # warning/orange
            "contacted": "#0d6efd",
            "new": "#adb5bd",
            "notes": "#6c757d",
        },
        hover_data={
            "offer_url": True,
            "offer_id": True,
            "event_type": True,
            "value": True,
            "ts": True,
        },
        title=f"Offer activity timeline (last {days_i} days)",
        labels={"ts": "Time", "offer_label": "Offer"},
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        height=900,
        margin=dict(l=20, r=20, t=60, b=20),
        legend_title_text="Event",
    )

    fig_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    return render_template("timeline.html", fig_html=fig_html, days=days_i, bar_html=bar_html)

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ensure_schema()
    app.run(host="0.0.0.0", port=5000, debug=True)
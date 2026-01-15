from __future__ import annotations

from pathlib import Path
import re
import math
import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urljoin

import pandas as pd
import plotly.express as px
import plotly.io as pio

from flask import Flask, render_template, request, redirect
from bs4 import BeautifulSoup


# =========================
# Paths / Config
# =========================
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "app_state.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = DATA_DIR

MARKET_BASELINE_CSV = DATA_DIR / "market_baseline_eurm2.csv"

FIGARO_SCRAPE_DIR = APP_DIR / "scrapes" / "figaro"
FIGARO_BASE_URL = "https://immobilier.lefigaro.fr"

PARIS_ARR_GEOJSON = APP_DIR / "static" / "geo" / "paris_arrondissements.geojson"


# =========================
# Flask app
# =========================
app = Flask(__name__)


# =========================
# DB helpers (state persistence)
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_state_schema(conn: sqlite3.Connection):
    """
    Ensure offer_state table exists and add per-field timestamp columns if missing.
    Safe to run repeatedly.
    """
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS offer_state (
        offer_id TEXT PRIMARY KEY,
        status TEXT DEFAULT 'new',      -- new | kept | discarded
        contacted INTEGER DEFAULT 0,    -- 0/1
        notes TEXT DEFAULT ''
    )
    """)
    conn.commit()

    # Add columns if missing
    for ddl in [
        "ALTER TABLE offer_state ADD COLUMN status_updated_at TEXT",
        "ALTER TABLE offer_state ADD COLUMN contacted_updated_at TEXT",
        "ALTER TABLE offer_state ADD COLUMN notes_updated_at TEXT",
    ]:
        try:
            cur.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError as e:
            # ignore "duplicate column name"
            if "duplicate column name" not in str(e).lower():
                raise
    
        # --- Events log (append-only) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS offer_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,                 -- ISO string (UTC recommended)
        offer_id TEXT NOT NULL,
        event_type TEXT NOT NULL,         -- 'status'|'contacted'|'notes'
        value TEXT,                       -- e.g. 'kept' / 'discarded' / '1' / note preview
        meta_json TEXT                    -- optional JSON (url/name/arr/etc)
    )
    """)
    conn.commit()

    # Helpful indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_offer_events_ts ON offer_events(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_offer_events_offer_id ON offer_events(offer_id)")
    conn.commit()

def log_event(offer_id: str, event_type: str, value: str | None = None, meta: dict | None = None):
    ts = now_iso()  # UTC ISO
    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO offer_events (ts, offer_id, event_type, value, meta_json) VALUES (?,?,?,?,?)",
            (ts, offer_id, event_type, value, meta_json),
        )
        conn.commit()

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_state_schema(conn)  # ✅ migration + table creation
    return conn


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


def upsert_state(offer_id: str, status=None, contacted=None, notes=None):
    """
    Upsert persisted state.
    Writes per-field timestamps only for fields being updated.
    """
    ts = now_iso()
    with db_conn() as conn:
        existing = conn.execute(
            """SELECT offer_id, status, contacted, notes,
                      status_updated_at, contacted_updated_at, notes_updated_at
               FROM offer_state WHERE offer_id=?""",
            (offer_id,)
        ).fetchone()

        if existing is None:
            conn.execute(
                """INSERT INTO offer_state
                   (offer_id, status, contacted, notes,
                    status_updated_at, contacted_updated_at, notes_updated_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    offer_id,
                    status or "new",
                    int(contacted) if contacted is not None else 0,
                    notes or "",
                    ts if status is not None else None,
                    ts if contacted is not None else None,
                    ts if notes is not None else None,
                )
            )
        else:
            new_status = status if status is not None else existing["status"]
            new_contacted = int(contacted) if contacted is not None else existing["contacted"]
            new_notes = notes if notes is not None else existing["notes"]

            status_ts = ts if status is not None else existing["status_updated_at"]
            contacted_ts = ts if contacted is not None else existing["contacted_updated_at"]
            notes_ts = ts if notes is not None else existing["notes_updated_at"]

            conn.execute(
                """UPDATE offer_state
                   SET status=?, contacted=?, notes=?,
                       status_updated_at=?, contacted_updated_at=?, notes_updated_at=?
                   WHERE offer_id=?""",
                (new_status, new_contacted, new_notes, status_ts, contacted_ts, notes_ts, offer_id)
            )

        conn.commit()

def apply_offer_search(df: pd.DataFrame, q: str) -> pd.DataFrame:
    """
    Filter the offers dataframe using a free-text query `q`.
    Matches against: url, offer_id, name, notes, source_file (case-insensitive).
    Also supports numeric matching for price/surface/arr/rooms.
    """
    if df.empty:
        return df
    q = (q or "").strip()
    if not q:
        return df

    q_low = q.lower()

    # Ensure columns exist
    for c in ["url", "offer_id", "name", "notes", "source_file"]:
        if c not in df.columns:
            df[c] = ""

    # Text match
    text_mask = (
        df["url"].astype(str).str.lower().str.contains(q_low, na=False) |
        df["offer_id"].astype(str).str.lower().str.contains(q_low, na=False) |
        df["name"].astype(str).str.lower().str.contains(q_low, na=False) |
        df["notes"].astype(str).str.lower().str.contains(q_low, na=False) |
        df["source_file"].astype(str).str.lower().str.contains(q_low, na=False)
    )

    # Numeric match (optional)
    num_mask = pd.Series(False, index=df.index)
    # If the user types e.g. "11" or "75011" or "900000"
    digits = re.sub(r"[^\d]", "", q)
    if digits:
        try:
            n = int(digits)
            # arrondissement / rooms
            if "arrondissement" in df.columns:
                num_mask |= (pd.to_numeric(df["arrondissement"], errors="coerce") == n)
            if "rooms" in df.columns:
                num_mask |= (pd.to_numeric(df["rooms"], errors="coerce") == n)
            # price (exact)
            if "price_eur" in df.columns:
                num_mask |= (pd.to_numeric(df["price_eur"], errors="coerce") == n)
        except Exception:
            pass

    # Surface match (float)
    m = re.search(r"(\d+(?:[.,]\d+)?)", q)
    if m and "surface_m2" in df.columns:
        try:
            x = float(m.group(1).replace(",", "."))
            # allow a small tolerance, e.g. query "72" matches 71.8–72.2
            num_mask |= (pd.to_numeric(df["surface_m2"], errors="coerce").sub(x).abs() <= 0.25)
        except Exception:
            pass

    return df[text_mask | num_mask]

# =========================
# Offer ID + parsing helpers
# =========================
def compute_offer_id(row: dict) -> str:
    """
    Stable ID: prefer url; fallback to hash of key fields.
    """
    url = (row.get("url") or "").strip()
    if url:
        return "url:" + url

    key = "|".join([
        str(row.get("name", "")).strip(),
        str(row.get("price_eur", "")).strip(),
        str(row.get("surface_m2", "")).strip(),
        str(row.get("rooms", "")).strip(),
        str(row.get("arrondissement", "")).strip(),
        str(row.get("source_file", "")).strip(),
    ])
    return "hash:" + hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


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
    """Best-effort: look for dd/mm/yyyy or yyyy-mm-dd inside a block of text."""
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
    """
    Parse best-effort from a summary line.
    """
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
    """
    Parse all *.html in folder and extract offers.
    Returns DataFrame with:
      name, price_eur, surface_m2, arrondissement, eur_m2, rooms, published_date, url, source_file
    """
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

    mask = df["eur_m2"].isna() & df["price_eur"].notna() & df["surface_m2"].notna() & (df["surface_m2"] > 0)
    df.loc[mask, "eur_m2"] = df.loc[mask, "price_eur"] / df.loc[mask, "surface_m2"]

    df["arrondissement"] = pd.to_numeric(df["arrondissement"], errors="coerce")
    df = df[df["arrondissement"].between(1, 20)]
    df = df[df["eur_m2"].between(1000, 50000)]
    df = df[df["price_eur"].between(10_000, 50_000_000)]

    df["published_date"] = df["published_date"].fillna("")
    df = df.sort_values(by=["published_date", "source_file"], ascending=[False, True]).reset_index(drop=True)
    return df


# =========================
# Baseline (market) helpers
# =========================
def load_market_baseline_map():
    """
    Returns dict: arrondissement(int) -> market_eur_m2(float)
    CSV required columns: arrondissement, market_eur_m2
    """
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


# =========================
# Map + Bar chart helpers (Plotly)
# =========================
def _guess_featureidkey(geojson: dict) -> str:
    feats = geojson.get("features") or []
    if not feats:
        raise ValueError("GeoJSON has no features.")
    props = feats[0].get("properties") or {}

    candidates = [
        "c_ar", "arrondissement", "arrond", "code", "id",
        "c_arinsee", "numero", "num", "insee",
        "postalcode", "code_postal",
    ]
    for k in candidates:
        if k in props:
            return f"properties.{k}"

    for k, v in props.items():
        s = str(v).strip()
        if s.isdigit():
            n = int(s)
            if 1 <= n <= 20:
                return f"properties.{k}"

    raise ValueError(f"Could not guess arrondissement key. Properties: {list(props.keys())}")


def make_offers_map_and_bar(arr_counts_df: pd.DataFrame, title_prefix="Filtered"):
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
            "<div class='border p-3 text-muted small'>"
            "Missing GeoJSON: <code>static/geo/paris_arrondissements.geojson</code>"
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


# =========================
# Snippet loader for Home
# =========================
def load_snippet(name: str) -> str:
    fp = PLOTS_DIR / name
    return fp.read_text(encoding="utf-8", errors="ignore") if fp.exists() else ""


# =========================
# Routes
# =========================
@app.route("/")
def home():
    sections = [
        {
            "kicker": "Snapshot",
            "title": "Median sale prices by arrondissement",
            "body": (
                "2024 + 2025 semester 1 median sold price per arrondissement. "
                "Data from official government source."
            ),
            "note": "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20251018-234902/valeursfoncieres-2025-s1.txt.zip",
            "plot_html": load_snippet("current_medians.html"),
        },
        {
            "kicker": "Activity",
            "title": "Sales volumes by arrondissement",
            "body": (
                "Volume helps interpret confidence: higher counts usually mean more stable medians, "
                "while low counts can be noisier."
            ),
            "note": "Use alongside price charts to understand liquidity and market thickness.",
            "plot_html": load_snippet("current_sale_volumes.html"),
        },
        {
            "kicker": "Map",
            "title": "Median per arrondissement",
            "body": "Data displayed in space.",
            "note": "",
            "plot_html": load_snippet("median_per_arr.html"),
        },
        {
            "kicker": "History",
            "title": "Evolution of median €/m² through time",
            "body": "Time series of median €/m² per arrondissement.",
            "note": "",
            "plot_html": load_snippet("medians_time_series.html"),
        },
        {
            "kicker": "History",
            "title": "NORMALISED evolution of median €/m² through time",
            "body": "Normalised time series of median €/m² per arrondissement.",
            "note": "",
            "plot_html": load_snippet("medians_ts_relative.html"),
        },
        {
            "kicker": "Trend map",
            "title": "10-year price trend (annualized)",
            "body": "Each arrondissement summarised by a single annualised number over the last 10 years.",
            "note": "",
            "plot_html": load_snippet("trend_10y_median.html"),
        },
        {
            "kicker": "Trend map",
            "title": "10-year volume trend (annualized)",
            "body": "How market activity changed over time (transaction counts).",
            "note": "",
            "plot_html": load_snippet("trend_10y_volumes.html"),
        },
        {
            "kicker": "FIGARO IMMO DATA",
            "title": "Offers on Le Figaro Immobilier",
            "body": "Scraped offers (saved HTML).",
            "note": "",
            "plot_html": load_snippet("Ask_median_vs_DVF_DVF_and_volume.html"),
        },
        {
            "kicker": "Premium vs historical",
            "title": "How offers compare to historical sales",
            "body": "Premium = asking €/m² relative to market baseline (DVF/DVF+).",
            "note": "",
            "plot_html": load_snippet("Ask_premium_vs_DVF_DVF.html"),
        },
    ]
    return render_template("home.html", sections=sections)


@app.route("/offers")
def offers():
    df = load_figaro_offers_from_folder(FIGARO_SCRAPE_DIR)

    # ---- filters (multi arr + numeric) ----
    arr_list  = request.args.getlist("arr")  # multi
    rooms     = request.args.get("rooms", "").strip()
    price_min = request.args.get("price_min", "").strip()
    price_max = request.args.get("price_max", "").strip()
    surf_min  = request.args.get("surf_min", "").strip()
    surf_max  = request.args.get("surf_max", "").strip()

    # NEW: free-text search
    q = (request.args.get("q", "") or "").strip()

    # ---- status/contacted filters (checkboxes) ----
    status_list = request.args.getlist("status")  # kept/discarded/new
    contacted_only = (request.args.get("contacted_only", "").strip() == "1")

    # ---- sorting ----
    sort_by  = request.args.get("sort", "published_date").strip()
    sort_dir = request.args.get("dir", "desc").strip().lower()

    allowed_sort = {
        "name", "price_eur", "surface_m2", "eur_m2", "premium_pct",
        "rooms", "arrondissement", "published_date",
        "status", "contacted"
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
        df["market_eur_m2"] = df["arrondissement"].map(baseline_map)
        df["premium_pct"] = (df["eur_m2"] / df["market_eur_m2"] - 1.0) * 100.0

    # ---- apply numeric filters (raw scraped fields) ----
    if not df.empty:
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

    # ---- attach offer_id + persisted state for ALL filtered rows ----
    if not df.empty:
        df = df.copy()
        df["offer_id"] = df.apply(lambda r: compute_offer_id(r.to_dict()), axis=1)

        states = get_states_map(df["offer_id"].tolist())

        def attach_state(row):
            st = states.get(row["offer_id"])
            if not st:
                return pd.Series({"status": "new", "contacted": 0, "notes": ""})
            return pd.Series({
                "status": st.get("status", "new"),
                "contacted": st.get("contacted", 0),
                "notes": st.get("notes", "") or ""
            })

        df[["status", "contacted", "notes"]] = df.apply(attach_state, axis=1)
    else:
        df = df.copy()
        df["offer_id"] = pd.Series(dtype=str)
        df["status"] = pd.Series(dtype=str)
        df["contacted"] = pd.Series(dtype=int)
        df["notes"] = pd.Series(dtype=str)

    # ---- apply status/contacted filters (checkboxes) ----
    if not df.empty:
        if status_list:
            df = df[df["status"].isin(status_list)]
        if contacted_only:
            df = df[df["contacted"] == 1]

    # ---- apply FREE-TEXT SEARCH (url, offer_id, name, notes, source_file, etc.) ----
    df = apply_offer_search(df, q)

    # ---- build charts from CURRENT FILTERED SET (not page only) ----
    if not df.empty and "arrondissement" in df.columns:
        arr_counts = (
            df.groupby("arrondissement")
              .size()
              .reset_index(name="n_offers")
              .copy()
        )
        arr_counts["arrondissement"] = arr_counts["arrondissement"].astype(int).astype(str)
        offers_map_html, offers_bar_html = make_offers_map_and_bar(arr_counts, title_prefix="Filtered")
    else:
        offers_map_html, offers_bar_html = "", ""

    # ---- sort AFTER state join & search so sorting by status/contacted works ----
    if not df.empty and sort_by in df.columns:
        df = df.sort_values(
            by=sort_by,
            ascending=ascending,
            na_position="last",
            kind="mergesort"
        )

    # ---- pagination ----
    PAGE_SIZE = 30
    total = len(df)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = int(request.args.get("page", "1") or 1)
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = df.iloc[start:end].copy()

    # ---- querystring state + back url for POST redirects ----
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
        f={
            "arr": arr_list,
            "rooms": rooms,
            "price_min": price_min,
            "price_max": price_max,
            "surf_min": surf_min,
            "surf_max": surf_max,
            "status": status_list,
            "contacted_only": "1" if contacted_only else "",
            "q": q,  # NEW
            "sort": sort_by,
            "dir": sort_dir,
        },
        offers_map_html=offers_map_html,
        offers_bar_html=offers_bar_html,
    )


# =========================
# Interaction routes (POST)
# =========================
@app.post("/offers/set_status")
def offers_set_status():
    offer_id = (request.form.get("offer_id", "") or "").strip()
    status = (request.form.get("status", "") or "").strip()
    back = (request.form.get("back") or "").strip()

    if not back or back == ".":
        back = "/offers"
    if not back.startswith("/"):
        back = "/offers"

    if offer_id and status in {"new", "kept", "discarded"}:
        upsert_state(offer_id, status=status)
        log_event(offer_id, "status", status)

    return redirect(back)


@app.post("/offers/toggle_contacted")
def offers_toggle_contacted():
    offer_id = (request.form.get("offer_id", "") or "").strip()
    contacted = (request.form.get("contacted", "0") or "0").strip()
    back = (request.form.get("back") or "").strip()

    if not back or back == ".":
        back = "/offers"
    if not back.startswith("/"):
        back = "/offers"

    if offer_id:
        new_val = "1" if (contacted == "1") else "0"
        upsert_state(offer_id, contacted=(new_val == "1"))
        log_event(offer_id, "contacted", new_val)

    return redirect(back)


@app.post("/offers/save_notes")
def offers_save_notes():
    offer_id = (request.form.get("offer_id", "") or "").strip()
    notes = request.form.get("notes", "") or ""
    back = (request.form.get("back") or "").strip()

    if not back or back == ".":
        back = "/offers"
    if not back.startswith("/"):
        back = "/offers"

    if offer_id:
        upsert_state(offer_id, notes=notes)
        preview = notes.strip().replace("\n", " ")[:80]
        log_event(offer_id, "notes", preview)

    return redirect(back)

# =========================
# Timeline
# =========================

@app.route("/timeline")
def timeline():
    # Optional filter: only show last N days
    days = (request.args.get("days", "30") or "30").strip()
    try:
        days_i = max(1, min(3650, int(days)))
    except Exception:
        days_i = 30

    # Pull events
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT ts, offer_id, event_type, value, meta_json
            FROM offer_events
            ORDER BY ts ASC
            """
        ).fetchall()

    if not rows:
        fig_html = "<div class='text-muted small'>No events yet. Change some statuses in Offers first.</div>"
        return render_template(
            "timeline.html",
            fig_html=fig_html,
            days=days_i,
            bar_html="",
        )

    df = pd.DataFrame([dict(r) for r in rows])

    # Parse timestamps -> UTC-aware
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])

    # Window filter (UTC)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_i)
    dfw = df[df["ts"] >= cutoff].copy()

    if dfw.empty:
        fig_html = "<div class='text-muted small'>No events in this window.</div>"
        return render_template(
            "timeline.html",
            fig_html=fig_html,
            days=days_i,
            bar_html="",
        )

    # ---------- BARPLOT: contacted made per day ----------
    # count only contacted "ON" events, grouped by day in Paris time
    contacted = dfw[(dfw["event_type"] == "contacted") & (dfw["value"].astype(str) == "1")].copy()

    if not contacted.empty:
        contacted["day_paris"] = contacted["ts"].dt.tz_convert("Europe/Paris").dt.date
        daily = (
            contacted.groupby("day_paris")
            .size()
            .reset_index(name="n_contacted")
            .sort_values("day_paris")
        )
        daily["day_paris"] = daily["day_paris"].astype(str)

        bar_fig = px.bar(
            daily,
            x="day_paris",
            y="n_contacted",
            title="Contacted per day",
            labels={"day_paris": "Day (Paris)", "n_contacted": "# contacted"},
        )
        bar_fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs="cdn")
    else:
        bar_html = (
            "<div class='text-muted small'>No contacted events (ON) in this window.</div>"
        )

    # ---------- TIMELINE SCATTER ----------
    def color_key(r):
        if r["event_type"] == "status":
            if r["value"] == "kept":
                return "kept"
            if r["value"] == "discarded":
                return "discarded"
            return "new"
        if r["event_type"] == "contacted":
            return "contacted"
        if r["event_type"] == "notes":
            return "notes"
        return "notes"

    dfw["color"] = dfw.apply(color_key, axis=1)

    # Human-friendly y labels (UNIQUE) from URL slug / tail
    dfw["offer_url"] = dfw["offer_id"].astype(str).str.replace("^url:", "", regex=True)
    dfw["offer_label"] = (
        dfw["offer_url"]
        .str.replace(r"/+$", "", regex=True)
        .str.split("/")
        .str[-1]
    )
    bad = dfw["offer_label"].isna() | (dfw["offer_label"].str.len() < 6)
    dfw.loc[bad, "offer_label"] = dfw.loc[bad, "offer_url"].str[-16:]

    fig = px.scatter(
        dfw,
        x="ts",
        y="offer_label",
        color="color",
        color_discrete_map={
            "kept": "#198754",        # green
            "discarded": "#dc3545",   # red
            "contacted": "#0d6efd",   # blue
            "new": "#adb5bd",         # grey
            "notes": "#6c757d",       # dark grey
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
    return render_template(
        "timeline.html",
        fig_html=fig_html,
        days=days_i,
        bar_html=bar_html,
    )

# =========================
# Run
# =========================
if __name__ == "__main__":
    # Ensure DB schema on startup
    with db_conn() as _:
        pass

    app.run(host="0.0.0.0", port=5000, debug=True)
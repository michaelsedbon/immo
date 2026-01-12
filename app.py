from __future__ import annotations

from pathlib import Path
import pandas as pd
from flask import Flask, render_template
import plotly.express as px

from flask import request
from bs4 import BeautifulSoup
import re
import math
from urllib.parse import urljoin
import pandas as pd
from pathlib import Path

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

MARKET_BASELINE_CSV = DATA_DIR / "market_baseline_eurm2.csv"

# Expected CSVs (rename to match your exports if needed)
DVF_SUMMARY_CSV = DATA_DIR / "dvf_summary.csv"          # arrondissement, dvf_median_eur_m2, nb_ventes, etc.
TRENDS_10Y_CSV   = DATA_DIR / "trends_10y.csv"          # arrondissement, price_trend_pct_per_year, volume_trend_pct_per_year
OFFERS_CSV       = DATA_DIR / "offers_listings.csv"     # price_eur, surface_m2, ask_eur_m2, arrondissement, url, source_file, ...

app = Flask(__name__)


from bs4 import BeautifulSoup
from pathlib import Path

SCRAPES_DIR = Path(__file__).parent / "scrapes" / "figaro"

def load_concatenated_offer_html():
    """
    Load all scraped HTML files, extract <body> content,
    concatenate them in filename order.
    """
    blocks = []

    files = sorted(SCRAPES_DIR.glob("*.html"))
    for fp in files:
        html = fp.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        body = soup.body
        if body:
            blocks.append(str(body))
        else:
            # fallback: include everything
            blocks.append(html)

    return "\n<hr style='margin:4rem 0; border-top:1px solid rgba(0,0,0,0.15);'>\n".join(blocks)


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_home_figs(dvf: pd.DataFrame, trends: pd.DataFrame):
    figs = {}

    if not dvf.empty and "arrondissement" in dvf.columns:
        df = dvf.copy()
        df["arrondissement"] = df["arrondissement"].astype(int)

        # Choose a median column name that exists
        median_col = None
        for c in ["dvf_median_eur_m2", "mediane_eur_m2", "median_eur_m2"]:
            if c in df.columns:
                median_col = c
                break

        if median_col:
            fig_price = px.bar(
                df.sort_values("arrondissement"),
                x=df["arrondissement"].astype(str),
                y=median_col,
                title="DVF/DVF+ — Médiane €/m² par arrondissement",
                labels={"x": "Arrondissement", median_col: "€/m²"},
            )
            fig_price.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
            figs["dvf_median_bar"] = fig_price.to_html(full_html=False, include_plotlyjs="cdn")

        if "nb_ventes" in df.columns:
            fig_vol = px.bar(
                df.sort_values("arrondissement"),
                x=df["arrondissement"].astype(str),
                y="nb_ventes",
                title="DVF/DVF+ — Volume (nb ventes) par arrondissement",
                labels={"x": "Arrondissement", "nb_ventes": "Nombre de ventes"},
            )
            fig_vol.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=360)
            figs["dvf_volume_bar"] = fig_vol.to_html(full_html=False, include_plotlyjs=False)

    if not trends.empty and "arrondissement" in trends.columns:
        t = trends.copy()
        t["arrondissement"] = t["arrondissement"].astype(int)

        if "price_trend_pct_per_year" in t.columns:
            fig_tr_price = px.bar(
                t.sort_values("arrondissement"),
                x=t["arrondissement"].astype(str),
                y="price_trend_pct_per_year",
                title="Tendance 10 ans — Prix médian (%, annualisé)",
                labels={"x": "Arrondissement", "price_trend_pct_per_year": "% / an"},
            )
            fig_tr_price.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=360)
            figs["trend_price_bar"] = fig_tr_price.to_html(full_html=False, include_plotlyjs=False)

        if "volume_trend_pct_per_year" in t.columns:
            fig_tr_vol = px.bar(
                t.sort_values("arrondissement"),
                x=t["arrondissement"].astype(str),
                y="volume_trend_pct_per_year",
                title="Tendance 10 ans — Volume de ventes (%, annualisé)",
                labels={"x": "Arrondissement", "volume_trend_pct_per_year": "% / an"},
            )
            fig_tr_vol.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=360)
            figs["trend_vol_bar"] = fig_tr_vol.to_html(full_html=False, include_plotlyjs=False)

    return figs


def build_offers_figs(offers: pd.DataFrame):
    figs = {}
    if offers.empty or "arrondissement" not in offers.columns:
        return figs

    o = offers.copy()
    o = o[o["arrondissement"].between(1, 20)]
    o["arrondissement"] = o["arrondissement"].astype(int)

    if "ask_eur_m2" in o.columns:
        # Summary per arrondissement
        agg = (
            o.groupby("arrondissement")["ask_eur_m2"]
            .agg(nb_offers="count", ask_median="median")
            .reset_index()
            .sort_values("arrondissement")
        )

        fig = px.bar(
            agg,
            x=agg["arrondissement"].astype(str),
            y="ask_median",
            title="Offres — Médiane €/m² (asking) par arrondissement",
            labels={"x": "Arrondissement", "ask_median": "€/m²"},
            hover_data={"nb_offers": True},
        )
        fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420)
        figs["offers_median_bar"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

        fig2 = px.bar(
            agg,
            x=agg["arrondissement"].astype(str),
            y="nb_offers",
            title="Offres — Volume (nb annonces) par arrondissement",
            labels={"x": "Arrondissement", "nb_offers": "Nombre d’offres"},
        )
        fig2.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=360)
        figs["offers_volume_bar"] = fig2.to_html(full_html=False, include_plotlyjs=False)

    return figs

PLOTS_DIR = Path(__file__).parent / "data"

def load_snippet(name: str) -> str:
    fp = PLOTS_DIR / name
    return fp.read_text(encoding="utf-8", errors="ignore") if fp.exists() else ""


@app.route("/")
def home():
    sections = [
        {
            "kicker": "Snapshot",
            "title": "Median sale prices by arrondissement",
            "body": (
                "2024 + 2025 semester 1 median sold pricer per arrondissment. "
                "Data from official governement source"
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
            "kicker": "map",
            "title": "Mediane par arrondissement",
            "body": (
                "data displayed in space"
            ),
            "note": "Use alongside price charts to understand liquidity and market thickness.",
            "plot_html": load_snippet("median_per_arr.html"),
        },
        {
            "kicker": "History",
            "title": "Evolution of median €/m² through time",
            "body": (
                "Time series of median pricer /m2 per arrondissement."
            ),
            "note": "",
            "plot_html": load_snippet("medians_time_series.html"),
        },
        {
            "kicker": "History",
            "title": "NORMALISED Evolution of median €/m² through time",
            "body": (
                "NORMALISED Time series of median pricer /m2 per arrondissement."
            ),
            "note": "",
            "plot_html": load_snippet("medians_ts_relative.html"),
        },
        {
            "kicker": "Trend map",
            "title": "10-year price trend (annualized)",
            "body": (
                "Each arrondissement is summarized by a single number: the annualized trend over the last 10 years."
            ),
            "note": "Method: regression on log(median €/m²) per period ⇒ approx % per year.",
            "plot_html": load_snippet("trend_10y_median.html"),
        },
        {
            "kicker": "Trend map",
            "title": "10-year volume trend (annualized)",
            "body": (
                "This map shows how market activity changed over time (growth/decline in transaction counts)."
            ),
            "note": "Method: regression on log(volume + 1) per period ⇒ approx % per year.",
            "plot_html": load_snippet("trend_10y_volumes.html"),
        },

         {
            "kicker": "FIGARO IMOBILIER DATA",
            "title": "Actual offers on the website of Le Figaro Immobilier",
            "body": (
                "I scrapped offers from le figaro. Here is what we can find online"
            ),
            "note": "",
            "plot_html": load_snippet("Ask_median_vs_DVF_DVF_and_volume.html"),
        },

         {
            "kicker": "Premium VS Historical public dataset",
            "title": "How is the market vs historical data",
            "body": (
                "This looks at median prices vs what historical data show "
            ),
            "note": "Les offres dans le 13eme sont beaucoup plus hautes que dans le datyaset des ventes. Le 19 est bien en dessous",
            "plot_html": load_snippet("Ask_premium_vs_DVF_DVF.html"),
        },
    ]

    return render_template("home.html", sections=sections)






###### ---------------- Scrapping offers ------------------------
###### ---------------- Scrapping offers ------------------------
###### ---------------- Scrapping offers ------------------------

FIGARO_SCRAPE_DIR = Path(__file__).parent / "scrapes" / "figaro"
FIGARO_BASE_URL = "https://immobilier.lefigaro.fr"

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

def _parse_int_fr(s: str):
    if s is None:
        return None
    s = str(s)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

def _parse_float_fr(s: str):
    if s is None:
        return None
    s = str(s)
    m = re.search(r"(\d+(?:[.,]\d+)?)", s)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

def _extract_arrondissement(txt: str):
    if not txt:
        return None
    m = re.search(r"Paris\s+(\d{1,2})\s*(?:e|ème|eme)\b", txt, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _extract_date(txt: str):
    """Best-effort: look for dd/mm/yyyy or yyyy-mm-dd inside a block of text."""
    if not txt:
        return None
    # 12/01/2026
    m = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", txt)
    if m:
        return m.group(1)
    # 2026-01-12
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", txt)
    if m:
        return m.group(1)
    return None

def _parse_summary_line(txt: str):
    """
    Parse best-effort from a summary line containing e.g.
    '499 000 € 10 778 €/m² Appartement 2 pièces 46,3 m² ... Paris 5ème (75)'
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

    # name: try to remove leading price chunk to keep something readable
    out["name"] = t
    return out

def load_figaro_offers_from_folder(folder: Path):
    """
    Parse all *.html in folder and extract offers.
    Returns a pandas DataFrame with:
      name, price_eur, surface_m2, arrondissement, eur_m2, rooms, published_date, url, source_file
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.html"))
    rows = []

    for fp in files:
        html = fp.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        # Strategy 1 (robust): find anchors that look like listing summaries
        for a in soup.find_all("a"):
            txt = " ".join(a.get_text(" ", strip=True).split())
            if not txt:
                continue
            if ("€" in txt) and ("m²" in txt) and ("Paris" in txt):
                # We avoid obvious non-listing junk by requiring at least a price marker
                d = _parse_summary_line(txt)
                if d["price_eur"] is None and d["eur_m2"] is None:
                    continue

                href = a.get("href")
                url = urljoin(FIGARO_BASE_URL, href) if href else None

                # Try to detect a nearby date in the parent block (best-effort)
                parent_txt = ""
                parent = a.parent
                if parent:
                    parent_txt = " ".join(parent.get_text(" ", strip=True).split())
                if not d["published_date"]:
                    d["published_date"] = _extract_date(parent_txt)

                # Try to get a better name: prefer title attr / aria-label if present
                name = a.get("title") or a.get("aria-label")
                d["name"] = name.strip() if name else d["name"]

                d["url"] = url
                d["source_file"] = fp.name
                rows.append(d)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url", "name", "price_eur", "surface_m2", "source_file"])

    if df.empty:
        return df

    # compute eur_m2 if missing
    df["eur_m2"] = pd.to_numeric(df["eur_m2"], errors="coerce")
    df["price_eur"] = pd.to_numeric(df["price_eur"], errors="coerce")
    df["surface_m2"] = pd.to_numeric(df["surface_m2"], errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")

    mask = df["eur_m2"].isna() & df["price_eur"].notna() & df["surface_m2"].notna() & (df["surface_m2"] > 0)
    df.loc[mask, "eur_m2"] = df.loc[mask, "price_eur"] / df.loc[mask, "surface_m2"]

    # Keep plausible Paris offers
    df["arrondissement"] = pd.to_numeric(df["arrondissement"], errors="coerce")
    df = df[df["arrondissement"].between(1, 20)]
    df = df[df["eur_m2"].between(1000, 50000)]
    df = df[df["price_eur"].between(10_000, 50_000_000)]

    # normalize date: keep as string (we can parse later if you want)
    df["published_date"] = df["published_date"].fillna("")

    # sort: if date available, later first; else by file order
    # (string sort works for yyyy-mm-dd; for dd/mm/yyyy it’s imperfect but ok)
    df = df.sort_values(by=["published_date", "source_file"], ascending=[False, True])

    return df.reset_index(drop=True)


###### ---------------- Scrapping offers ------------------------
###### ---------------- Scrapping offers ------------------------
###### ---------------- Scrapping offers ------------------------


@app.route("/offers")
def offers():
    df = load_figaro_offers_from_folder(FIGARO_SCRAPE_DIR)
    baseline_map = load_market_baseline_map()

    if not df.empty:
        df["market_eur_m2"] = df["arrondissement"].map(baseline_map)
        df["premium_pct"] = (df["eur_m2"] / df["market_eur_m2"] - 1.0) * 100.0

    # ---- filters (multi-select arr) ----
    arr_list = request.args.getlist("arr")  # multi
    rooms = request.args.get("rooms", "").strip()
    price_min = request.args.get("price_min", "").strip()
    price_max = request.args.get("price_max", "").strip()
    surf_min  = request.args.get("surf_min", "").strip()
    surf_max  = request.args.get("surf_max", "").strip()

    # ---- sorting ----
    sort_by  = request.args.get("sort", "published_date").strip()
    sort_dir = request.args.get("dir", "desc").strip().lower()  # asc/desc

    allowed_sort = {
        "name", "price_eur", "surface_m2", "eur_m2", "rooms", "arrondissement", "published_date", "premium_pct"
    }
    if sort_by not in allowed_sort:
        sort_by = "published_date"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    ascending = (sort_dir == "asc")

    # ---- pagination ----
    page = int(request.args.get("page", "1") or 1)
    PAGE_SIZE = 30

    # Apply filters
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

        # Sort (stable, keep NaNs at bottom)
        if sort_by in df.columns:
            df = df.sort_values(
                by=[sort_by],
                ascending=ascending,
                na_position="last",
                kind="mergesort",  # stable
            )

    total = len(df)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_df = df.iloc[start:end].copy()

    # Filter options
    arr_options = list(range(1, 21))
    room_options = [1, 2, 3, 4, 5]

    # Preserve current query params for pagination + sorting links (including multi-values)
    qs = request.args.to_dict(flat=False)

    baseline_loaded = len(baseline_map) > 0
    return render_template(
        "offers.html",
        offers=page_df.to_dict(orient="records"),
        total=total,
        page=page,
        total_pages=total_pages,
        arr_options=arr_options,
        room_options=room_options,
        qs=qs,  # dict[str, list[str]]
        f={
            "arr": arr_list,  # list
            "rooms": rooms,
            "price_min": price_min,
            "price_max": price_max,
            "surf_min": surf_min,
            "surf_max": surf_max,
            "sort": sort_by,
            "dir": sort_dir,
        },
        baseline_loaded=baseline_loaded,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
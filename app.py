from __future__ import annotations

from pathlib import Path
import pandas as pd
from flask import Flask, render_template
import plotly.express as px

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"

# Expected CSVs (rename to match your exports if needed)
DVF_SUMMARY_CSV = DATA_DIR / "dvf_summary.csv"          # arrondissement, dvf_median_eur_m2, nb_ventes, etc.
TRENDS_10Y_CSV   = DATA_DIR / "trends_10y.csv"          # arrondissement, price_trend_pct_per_year, volume_trend_pct_per_year
OFFERS_CSV       = DATA_DIR / "offers_listings.csv"     # price_eur, surface_m2, ask_eur_m2, arrondissement, url, source_file, ...

app = Flask(__name__)


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

@app.route("/offers")
def offers():
    offers = safe_read_csv(OFFERS_CSV)

    stats = {
        "offers_rows": int(len(offers)) if not offers.empty else 0,
        "offers_arrs": int(offers["arrondissement"].nunique()) if (not offers.empty and "arrondissement" in offers.columns) else 0,
    }

    figs = build_offers_figs(offers)

    # show latest / first rows
    offers_table = offers.head(50).to_dict(orient="records") if not offers.empty else []
    offers_cols = list(offers.columns)

    return render_template(
        "offers.html",
        stats=stats,
        figs=figs,
        offers_cols=offers_cols,
        offers_table=offers_table,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
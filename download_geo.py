from pathlib import Path
import requests

GEOJSON_URLS = [
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/arrondissements/exports/geojson",
    "https://opendata.paris.fr/explore/dataset/arrondissements/download/?format=geojson",
]

OUT_PATH = Path("static/geo/paris_arrondissements.geojson")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def download_geojson(urls, out_path: Path, timeout=60):
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            # Basic sanity check: should contain "features"
            txt = r.text
            if '"features"' not in txt:
                raise ValueError("Response does not look like a GeoJSON FeatureCollection.")
            out_path.write_text(txt, encoding="utf-8")
            print(f"✅ Saved GeoJSON to: {out_path.resolve()}")
            print(f"   Source: {url}")
            return
        except Exception as e:
            last_err = e
            print(f"⚠️ Failed: {url} -> {e}")
    raise RuntimeError(f"All downloads failed. Last error: {last_err}")

download_geojson(GEOJSON_URLS, OUT_PATH)

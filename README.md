# 🌦️ Farm & Field Weather Explorer

Streamlit app with **preloaded GeoJSON boundaries**. Pick one or several farms, optionally a field inside
them, and see current weather, a multi-day forecast and **forecast confidence** (Open-Meteo, no API key).

## Use your own boundaries
1. Replace `data/boundaries.geojson` with your file (Polygon/MultiPolygon features).
2. It should have a farm column and a field column (e.g. `Farm`, `Field`). Common names are auto-detected;
   otherwise set them under **Column mapping** in the sidebar.
3. If you put several `.geojson` files in `data/`, the first (alphabetically) is loaded.

## Run locally
```bash
conda create -n polyweather -c conda-forge python=3.11 geopandas pyogrio shapely -y
conda activate polyweather
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (use a **private** repo if the boundaries are confidential).
2. https://share.streamlit.io → **Create app** → pick the repo, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.11 → **Deploy**.

## How confidence works
The app requests the same forecast from four models (ECMWF IFS, NOAA GFS, DWD ICON, CMC GEM) and compares
daily max/min temperature, rain and wind. Close agreement = High, moderate = Medium, large disagreement = Low.
Ratings are lowered beyond ~8 days. It is a guide to forecast reliability, not a verified accuracy score.

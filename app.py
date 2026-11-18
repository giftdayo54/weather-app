"""Polygon Weather Explorer
Upload a zipped shapefile or GeoJSON, pick a polygon, see weather at its centroid.
Run:  streamlit run app.py
"""
import io
import re
import tempfile
import time
import zipfile
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from shapely import union_all
from shapely.geometry import Point
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="Polygon Weather Explorer", page_icon="🌦️", layout="wide")

# Mobile-friendly tweaks: tighter padding, wrapping metric cards
st.markdown(
    """
    <style>
    .block-container {padding: 1rem 1rem 3rem 1rem; max-width: 1400px;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .6rem .8rem; border-radius: 10px;}
    @media (max-width: 640px) {
        h1 {font-size: 1.5rem !important;}
        div[data-testid="column"] {min-width: 45% !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WMO = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Dense drizzle", "🌧️"),
    56: ("Freezing drizzle", "🌧️"), 57: ("Heavy freezing drizzle", "🌧️"),
    61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Heavy freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "❄️"), 75: ("Heavy snow", "❄️"), 77: ("Snow grains", "❄️"),
    80: ("Light showers", "🌦️"), 81: ("Showers", "🌧️"), 82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "🌨️"), 86: ("Heavy snow showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm, hail", "⛈️"), 99: ("Severe thunderstorm, hail", "⛈️"),
}


def wmo(code):
    return WMO.get(int(code), ("Unknown", "❔")) if code is not None and pd.notna(code) else ("Unknown", "❔")


def compass(deg):
    if deg is None or pd.isna(deg):
        return "–"
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) % 360 // 45)]


# --------------------------------------------------------------------------- data loading
@st.cache_data(show_spinner="Reading spatial data…")
def read_layer(file_bytes: bytes, filename: str) -> gpd.GeoDataFrame:
    """Read a zipped shapefile or GeoJSON from raw bytes. Raises ValueError with a friendly message."""
    name = filename.lower()
    try:
        if name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                        zf.extractall(tmp)
                except zipfile.BadZipFile:
                    raise ValueError("The file is not a valid ZIP archive.")
                shps = [p for p in Path(tmp).rglob("*") if p.suffix.lower() == ".shp" and "__MACOSX" not in p.parts]
                if not shps:
                    raise ValueError("No .shp file found inside the ZIP.")
                shp = shps[0]
                parts = {p.suffix.lower() for p in shp.parent.glob(shp.stem + ".*")}
                missing = [e for e in (".shx", ".dbf") if e not in parts]
                if missing:
                    raise ValueError(f"Shapefile is incomplete – missing: {', '.join(missing)}")
                return gpd.read_file(shp)
        if name.endswith((".geojson", ".json")):
            return gpd.read_file(io.BytesIO(file_bytes))
    except ValueError:
        raise
    except Exception as exc:  # fiona/pyogrio parsing errors
        raise ValueError(f"Could not read the file: {exc}")
    raise ValueError("Unsupported file type. Upload a .zip (shapefile) or .geojson/.json file.")


def prepare_layer(gdf: gpd.GeoDataFrame):
    """Clean geometries, reproject to EPSG:4326, add helper columns. Returns (gdf, notes)."""
    notes = []
    if gdf is None or gdf.empty:
        raise ValueError("The layer contains no features.")
    gdf = gdf.copy()
    gdf.columns = [str(c) for c in gdf.columns]

    n0 = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if len(gdf) < n0:
        notes.append(f"Dropped {n0 - len(gdf)} feature(s) with missing/empty geometry.")

    bad = ~gdf.geometry.is_valid
    if bad.any():
        gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].make_valid()
        notes.append(f"Repaired {int(bad.sum())} invalid geometr{'y' if bad.sum() == 1 else 'ies'}.")

    n1 = len(gdf)
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(gdf) < n1:
        notes.append(f"Dropped {n1 - len(gdf)} non-polygon feature(s).")
    if gdf.empty:
        raise ValueError("No usable polygon geometries found.")

    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        if -180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90:
            gdf = gdf.set_crs(4326)
            notes.append("No CRS defined – assumed WGS84 (EPSG:4326) because coordinates look like lon/lat.")
        else:
            raise ValueError("The layer has no CRS and its coordinates are not lon/lat. Add a .prj file.")
    elif gdf.crs.to_epsg() != 4326:
        try:
            gdf = gdf.to_crs(4326)
            notes.append("Reprojected to WGS84 (EPSG:4326).")
        except Exception as exc:
            raise ValueError(f"Reprojection failed: {exc}")

    gdf = gdf.reset_index(drop=True)
    try:
        area = gdf.to_crs(gdf.estimate_utm_crs()).area / 10_000
    except Exception:
        area = pd.Series([float("nan")] * len(gdf))
    gdf["_idx"] = range(len(gdf))
    gdf["Area (ha)"] = area.round(2)
    return gdf, notes


def centroid_latlon(geom):
    """Centroid computed in a local UTM CRS (accurate), returned as (lat, lon)."""
    s = gpd.GeoSeries([geom], crs=4326)
    try:
        c = s.to_crs(s.estimate_utm_crs()).centroid.to_crs(4326).iloc[0]
    except Exception:
        c = geom.centroid
    return c.y, c.x


# --------------------------------------------------------------------------- weather
class WeatherError(Exception):
    pass


@st.cache_data(ttl=900, show_spinner="Fetching weather…")
def fetch_weather(lat: float, lon: float, days: int) -> dict:
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": days,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
                   "weather_code,wind_speed_10m,wind_direction_10m",
        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,"
                  "wind_speed_10m,wind_gusts_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "precipitation_probability_max,wind_speed_10m_max",
    }
    return _get(WEATHER_URL, params)


def line_chart(df, x, series, title, unit):
    fig = go.Figure()
    for col, label in series:
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=label))
    fig.update_layout(title=title, yaxis_title=unit, hovermode="x unified", template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10), height=340,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def bar_chart(df, x, col, title, unit, color):
    fig = go.Figure(go.Bar(x=df[x], y=df[col], marker_color=color))
    fig.update_layout(title=title, yaxis_title=unit, template="plotly_white",
                      margin=dict(l=10, r=10, t=50, b=10), height=300)
    return fig

def _get(url: str, params: dict) -> dict:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 400:
                raise WeatherError(r.json().get("reason", "Invalid request to Open-Meteo."))
            r.raise_for_status()
            return r.json()
        except WeatherError:
            raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
        except requests.HTTPError as exc:
            raise WeatherError(f"Open-Meteo returned an error: {exc}")
        except ValueError:
            raise WeatherError("Open-Meteo returned an unreadable response.")
    raise WeatherError(f"Could not reach Open-Meteo (timeout/network). Last error: {last}")


FARM_KEYS = ["farm", "farm_name", "farmname", "farm_id", "estate", "estate_name", "block", "division", "region"]
FIELD_KEYS = ["field", "field_name", "fieldname", "field_id", "fieldid", "field_no", "plot", "plot_id", "name", "id", "fid"]
PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
DATA_DIR = Path(__file__).parent / "data"

# Multi-model comparison drives the confidence rating (thresholds: High if spread <= t1, Medium if <= t2)
MODELS = {"ecmwf_ifs025": "ECMWF IFS", "gfs_seamless": "NOAA GFS", "icon_seamless": "DWD ICON", "gem_seamless": "CMC GEM"}
CONF_VARS = {"temperature_2m_max": (1.5, 3.0, "Tmax"), "temperature_2m_min": (1.5, 3.0, "Tmin"),
             "precipitation_sum": (2.0, 6.0, "Rain"), "wind_speed_10m_max": (5.0, 12.0, "Wind")}
BADGE = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low", "n/a": "⚪ n/a"}


def guess(cols, keys):
    low = {c.lower().replace(" ", "_"): c for c in cols}
    return next((low[k] for k in keys if k in low), None)


@st.cache_data(ttl=900, show_spinner="Comparing weather models…")
def fetch_models(lat: float, lon: float, days: int) -> dict:
    params = {"latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": days,
              "models": ",".join(MODELS), "daily": ",".join(CONF_VARS)}
    return _get(WEATHER_URL, params)


def model_confidence(raw: dict) -> pd.DataFrame:
    """Per-day confidence from how closely the weather models agree, capped by lead time."""
    d = pd.DataFrame(raw.get("daily", {}))
    if d.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    scores, n_models = pd.DataFrame(index=d.index), pd.Series(0, index=d.index)
    for v, (t1, t2, short) in CONF_VARS.items():
        vals = d[[c for c in d.columns if c.startswith(v + "_")]].apply(pd.to_numeric, errors="coerce")
        n = vals.notna().sum(axis=1)
        spread = vals.max(axis=1) - vals.min(axis=1)
        sc = pd.Series(2.0, index=d.index)
        sc[spread > t1] = 1.0
        sc[spread > t2] = 0.0
        if v == "precipitation_sum":  # models disagree on whether it rains at all
            split = (vals.max(axis=1) >= 1) & (vals.min(axis=1) < 0.2)
            sc[split] = sc[split].clip(upper=1.0)
        scores[short] = sc.where(n >= 2)
        out[f"{short} mean"], out[f"{short} spread"] = vals.mean(axis=1), spread.where(n >= 2)
        n_models = pd.concat([n_models, n], axis=1).max(axis=1)
    mean = scores.mean(axis=1)
    lvl = pd.Series("n/a", index=d.index, dtype=object)
    lvl[mean >= 1.5] = "High"
    lvl[(mean >= 0.75) & (mean < 1.5)] = "Medium"
    lvl[mean < 0.75] = "Low"
    lvl[(scores.min(axis=1) == 0) & (lvl == "High")] = "Medium"  # one badly disputed variable caps the day
    lead = pd.Series(range(len(d)), index=d.index)
    lvl[(lead >= 8) & (lvl == "High")] = "Medium"          # skill falls with lead time
    lvl[(lead >= 11) & lvl.isin(["High", "Medium"])] = "Low"
    out["Confidence"], out["Models"] = lvl, n_models
    return out


def level_for(conf: pd.DataFrame, n: int) -> str:
    sub = conf["Confidence"].head(n)
    sub = sub[sub != "n/a"].map({"Low": 0, "Medium": 1, "High": 2})
    if sub.empty:
        return "n/a"
    s = sub.mean()
    return "High" if s >= 1.5 else "Medium" if s >= 0.75 else "Low"


def model_chart(raw: dict):
    d = pd.DataFrame(raw.get("daily", {}))
    cols = [c for c in d.columns if c.startswith("temperature_2m_max_")]
    if not cols:
        return None
    d["time"] = pd.to_datetime(d["time"])
    vals = d[cols].apply(pd.to_numeric, errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["time"], y=vals.max(axis=1), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["time"], y=vals.min(axis=1), line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(214,39,40,.15)", name="Model range"))
    for c in cols:
        key = c.removeprefix("temperature_2m_max_")
        fig.add_trace(go.Scatter(x=d["time"], y=vals[c], mode="lines+markers", name=MODELS.get(key, key)))
    fig.update_layout(title="Daily max temperature by model (shaded = spread)", yaxis_title="°C",
                      hovermode="x unified", template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
                      height=340, legend=dict(orientation="h", y=-0.2))
    return fig


# --------------------------------------------------------------------------- map
def build_map(view, sel_idx, colors, lat, lon, bounds, label="Weather point"):
    cols = [c for c in view.columns if not c.startswith("_") and c not in ("geometry", "Area (ha)")][:7] + ["Area (ha)"]
    show = view[cols + ["_idx", "_farm", "_field", "geometry"]].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(show[c]) and not pd.api.types.is_bool_dtype(show[c]):
            show[c] = show[c].astype(str)
    show["_color"] = show["_farm"].map(colors)
    if len(show) > 2000:
        show["geometry"] = show.geometry.simplify(0.0001)

    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite", attr="Esri").add_to(m)

    def style(f):
        p = f["properties"]
        if p["_idx"] == sel_idx:
            return {"color": "#d62728", "weight": 4, "fillColor": "#ff7f0e", "fillOpacity": 0.6}
        return {"color": p["_color"], "weight": 2, "fillColor": p["_color"], "fillOpacity": 0.3}

    folium.GeoJson(
        show, name="Fields", style_function=style,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(fields=["_farm", "_field", "Area (ha)"], aliases=["Farm", "Field", "Area (ha)"]),
        popup=folium.GeoJsonPopup(fields=cols, aliases=cols, max_width=300),
    ).add_to(m)
    if lat is not None:
        folium.Marker([lat, lon], tooltip=f"{label}: {lat:.5f}, {lon:.5f}",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]], padding=(20, 20))
    return m


# --------------------------------------------------------------------------- app
EXTS = {".geojson", ".json", ".zip"}
GPS, PLACE, FARM = "📱 My location", "🔎 Place or coordinates", "Farm / field"


def load_many(items):
    """Read, clean and merge many boundary files. Each file's name is kept as the fallback farm name."""
    parts, notes, errors = [], [], []
    for name, data in items:
        try:
            g, n = prepare_layer(read_layer(data, name))
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        g["_source"] = Path(name).stem
        parts.append(g)
        notes += [f"{Path(name).stem}: {x}" for x in n]
    if not parts:
        return None, notes, errors
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=4326)
    gdf["_idx"] = range(len(gdf))
    return gdf, notes, errors


@st.cache_data(show_spinner="Loading farm boundaries…")
def load_dir(sig):
    return load_many([(Path(p).name, Path(p).read_bytes()) for p, _ in sig])


@st.cache_data(show_spinner="Reading uploaded files…")
def load_uploads(items):
    return load_many(list(items))


@st.cache_data(ttl=3600, show_spinner="Searching places…")
def geocode(name: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": 5, "language": "en", "format": "json"}, timeout=10)
    r.raise_for_status()
    return r.json().get("results", [])


st.title("🌦️ Farm & Field Weather Explorer")
st.caption("Pick farm(s) and a field, or use your phone's location, to see weather with forecast confidence "
           "(Open-Meteo). On a phone, open the farm/field filters with the **›** arrow at the top left.")
mode = st.radio("Weather for", [FARM, GPS, PLACE], horizontal=True)

with st.sidebar:
    st.header("1 · Boundaries")
    ups = st.file_uploader("Use my own files instead (optional)", type=["zip", "geojson", "json"],
                           accept_multiple_files=True)

sig = tuple((str(p), p.stat().st_mtime) for p in sorted(DATA_DIR.rglob("*")) if p.is_file() and p.suffix.lower() in EXTS)
if ups:
    gdf, notes, errors = load_uploads(tuple((u.name, u.getvalue()) for u in ups))
elif sig:
    gdf, notes, errors = load_dir(sig)
else:
    gdf, notes, errors = None, [], []
for e in errors:
    st.sidebar.warning(e)
if gdf is None:
    st.error("No boundaries found. Put GeoJSON (or zipped shapefile) files in the repo's `data/` folder, "
             "or upload them in the sidebar.")
    st.stop()
st.sidebar.caption(f"Loaded {gdf['_source'].nunique()} file(s) · {len(gdf)} fields")
if notes:
    with st.sidebar.expander("Data notes"):
        for n in notes:
            st.write("•", n)

attrs = [c for c in gdf.columns if c not in ("geometry", "_idx", "_source", "Area (ha)")]
fg, fl = guess(attrs, FARM_KEYS), guess(attrs, FIELD_KEYS)
with st.sidebar.expander("Column mapping", expanded=not (fg and fl)):
    farm_opts, field_opts = ["(file name)"] + attrs, ["(row number)"] + attrs
    farm_col = st.selectbox("Farm column", farm_opts, index=farm_opts.index(fg) if fg else 0)
    field_col = st.selectbox("Field column", field_opts, index=field_opts.index(fl) if fl else 0)

gdf["_farm"] = (gdf["_source"] if farm_col == "(file name)"
                else gdf[farm_col].where(gdf[farm_col].notna(), gdf["_source"])).astype(str)
rownum = (gdf.groupby("_farm").cumcount() + 1).astype(str)
gdf["_field"] = rownum if field_col == "(row number)" else gdf[field_col].where(gdf[field_col].notna(), rownum).astype(str)
dup = gdf.duplicated(["_farm", "_field"], keep=False)
gdf.loc[dup, "_field"] = gdf.loc[dup, "_field"] + " (#" + (gdf.loc[dup, "_idx"] + 1).astype(str) + ")"

farms = sorted(gdf["_farm"].unique())
colors = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(farms)}

with st.sidebar:
    st.header("2 · Farms & fields")
    sel_farms = farms if st.checkbox(f"Select all {len(farms)} farms", value=False) else \
        st.multiselect("Farm(s)", farms, default=farms[:1])
    if not sel_farms:
        st.info("Select at least one farm.")
        st.stop()
    view = gdf[gdf["_farm"].isin(sel_farms)]
    multi = len(sel_farms) > 1
    labels = {int(i): (f"{f} · {fl_}" if multi else fl_) for i, f, fl_ in zip(view["_idx"], view["_farm"], view["_field"])}
    field_q = st.text_input("🔍 Search fields", placeholder="type part of a field name").strip().lower()
    opts = [i for i in labels if field_q in labels[i].lower()]
    if field_q:
        st.caption(f"{len(opts)} of {len(labels)} fields match")
    sel_field = st.selectbox("Field", [-1] + opts, help="You can also type inside this box to search.",
                             format_func=lambda i: "Whole selection (all fields)" if i < 0 else labels[i])
    st.header("3 · Options")
    days = st.slider("Forecast days", 1, 16, 7)
    zoom_sel = st.checkbox("Zoom to selection", value=True)

lat = lon = geom = None
title, src, point_label = "", "", "Weather point"
if mode == FARM:
    if sel_field >= 0:
        row = gdf.loc[gdf["_idx"] == sel_field].iloc[0]
        geom, title, area = row.geometry, f"{row['_farm']} · {row['_field']}", row["Area (ha)"]
    else:
        geom = union_all(list(view.geometry))
        title = ", ".join(sel_farms) if len(sel_farms) <= 3 else f"{len(sel_farms)} farms"
        area = view["Area (ha)"].sum()
    lat, lon = centroid_latlon(geom)
elif mode == GPS:
    loc = streamlit_geolocation()
    la, lo = (loc.get("latitude"), loc.get("longitude")) if isinstance(loc, dict) else (None, None)
    if la is not None and lo is not None:
        lat, lon, title, point_label = float(la), float(lo), "Your location", "You are here"
        acc = loc.get("accuracy")
        src = f"GPS ±{acc:,.0f} m" if acc else "GPS"
    else:
        st.info("Tap the location button above and allow access. If it doesn't work, enable location for this "
                "site in your browser settings, or use **Place or coordinates**.")
else:
    place_q = st.text_input("Place name or `lat, lon`", placeholder="e.g. Nairobi   or   -1.2921, 36.8219").strip()
    if place_q:
        mc = re.match(r"^(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)$", place_q)
        if mc:
            la, lo = float(mc.group(1)), float(mc.group(2))
            if -90 <= la <= 90 and -180 <= lo <= 180:
                lat, lon, title, src = la, lo, f"{la:.4f}, {lo:.4f}", "Coordinates"
            else:
                st.error("Latitude must be −90…90 and longitude −180…180.")
        else:
            try:
                res = geocode(place_q)
            except requests.RequestException as exc:
                st.error(f"Place search failed: {exc}")
                res = None
            if res is not None and not res:
                st.warning("No places found. Try another spelling or enter coordinates.")
            elif res:
                name_of = lambda r: ", ".join(filter(None, [r.get("name"), r.get("admin1"), r.get("country")]))
                pick = st.selectbox("Match", range(len(res)), format_func=lambda i: name_of(res[i]))
                lat, lon, title, src = res[pick]["latitude"], res[pick]["longitude"], name_of(res[pick]), "Place search"
        point_label = "Selected place"

if lat is not None:
    st.subheader(f"📍 {title}")
c1, c2, c3 = st.columns(3)
c1.metric("Farms / fields shown", f"{len(sel_farms)} / {len(view)}")
if mode == FARM:
    c2.metric("Area", f"{area:,.2f} ha" if pd.notna(area) else "n/a")
else:
    c2.metric("Source", src or "–")
c3.metric("Weather point", f"{lat:.4f}, {lon:.4f}" if lat is not None else "–")

if mode != FARM and lat is not None:
    hit = gdf[gdf.geometry.contains(Point(lon, lat))]
    if not hit.empty:
        st.success(f"📍 This point is inside {hit.iloc[0]['_farm']} · {hit.iloc[0]['_field']}")

if mode == FARM and sel_field >= 0 and zoom_sel:
    bounds = geom.bounds
elif mode != FARM and lat is not None:
    bounds = (lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
else:
    bounds = view.total_bounds
st_folium(build_map(view, sel_field if mode == FARM else -1, colors, lat, lon, bounds, point_label),
          use_container_width=True, height=450, returned_objects=[], key="map")

with st.expander(f"Attribute table ({len(view)} fields)"):
    st.dataframe(pd.DataFrame(view.drop(columns=["geometry", "_idx", "_farm", "_field", "_source"])), width="stretch")

if lat is None:
    st.stop()

st.divider()
st.header("Weather")
try:
    w = fetch_weather(round(lat, 4), round(lon, 4), days)
except WeatherError as exc:
    st.error(f"🌐 Weather unavailable: {exc}")
    st.stop()

conf, mraw = None, None
try:
    mraw = fetch_models(round(lat, 4), round(lon, 4), days)
    conf = model_confidence(mraw)
    conf = conf if not conf.empty and (conf["Models"] >= 2).any() else None
except WeatherError as exc:
    st.warning(f"Confidence unavailable (model comparison failed): {exc}")

cur = w.get("current", {})
desc, icon = wmo(cur.get("weather_code"))
st.markdown(f"### {icon} {desc}")
st.caption(f"Local time at site: {cur.get('time', '–').replace('T', ' ')} · timezone {w.get('timezone', '–')} · "
           f"grid elevation {w.get('elevation', '–')} m")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Temperature", f"{cur.get('temperature_2m', '–')} °C")
m2.metric("Feels like", f"{cur.get('apparent_temperature', '–')} °C")
m3.metric("Humidity", f"{cur.get('relative_humidity_2m', '–')} %")
m4.metric("Wind", f"{cur.get('wind_speed_10m', '–')} km/h {compass(cur.get('wind_direction_10m'))}")

if conf is not None:
    k1, k2, k3 = st.columns(3)
    k1.metric("Confidence · next 3 days", BADGE[level_for(conf, 3)])
    k2.metric("Confidence · next 7 days", BADGE[level_for(conf, 7)])
    k3.metric("Models compared", int(conf["Models"].max()))
st.caption("Confidence = how closely independent weather models agree (temperature, rain, wind), lowered for "
           "longer lead times. It is a guide, not a measured accuracy. Forecasts are model-grid values "
           "(roughly 10–25 km), so local rain especially can differ.")

hourly, daily = pd.DataFrame(w.get("hourly", {})), pd.DataFrame(w.get("daily", {}))
if hourly.empty or daily.empty:
    st.warning("Forecast data was empty for this location.")
    st.stop()
hourly["time"], daily["time"] = pd.to_datetime(hourly["time"]), pd.to_datetime(daily["time"])

t1, t2, t3, t4, t5 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "💨 Wind", "🎯 Confidence", "📅 Daily table"])
with t1:
    st.plotly_chart(line_chart(hourly, "time", [("temperature_2m", "Temperature"),
                                                ("apparent_temperature", "Feels like")],
                               "Hourly temperature", "°C"), width="stretch")
with t2:
    st.plotly_chart(bar_chart(hourly, "time", "precipitation_probability", "Hourly precipitation probability",
                              "%", "#4c78a8"), width="stretch")
    st.plotly_chart(bar_chart(daily, "time", "precipitation_sum", "Daily precipitation total", "mm", "#2ca02c"),
                    width="stretch")
with t3:
    st.plotly_chart(line_chart(hourly, "time", [("wind_speed_10m", "Wind speed"), ("wind_gusts_10m", "Gusts")],
                               "Hourly wind (10 m)", "km/h"), width="stretch")
with t4:
    if conf is None:
        st.info("Not enough models returned data to rate confidence for this location.")
    else:
        fig = model_chart(mraw)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        ct = pd.DataFrame({
            "Date": conf["date"].dt.strftime("%a %d %b"), "Confidence": conf["Confidence"].map(BADGE),
            "Tmax spread °C": conf["Tmax spread"].round(1), "Rain spread mm": conf["Rain spread"].round(1),
            "Wind spread km/h": conf["Wind spread"].round(1), "Models": conf["Models"],
        })
        st.dataframe(ct, hide_index=True, width="stretch")
        st.caption("Spread = highest minus lowest value across models. Smaller spread = higher agreement.")
with t5:
    tbl = pd.DataFrame({
        "Date": daily["time"].dt.strftime("%a %d %b"),
        "Sky": [f"{wmo(c)[1]} {wmo(c)[0]}" for c in daily["weather_code"]],
        "Max °C": daily["temperature_2m_max"], "Min °C": daily["temperature_2m_min"],
        "Rain mm": daily["precipitation_sum"], "Rain prob %": daily["precipitation_probability_max"],
        "Max wind km/h": daily["wind_speed_10m_max"],
    })
    if conf is not None and len(conf) == len(tbl):
        tbl["Confidence"] = conf["Confidence"].map(BADGE).values
    st.dataframe(tbl, hide_index=True, width="stretch")

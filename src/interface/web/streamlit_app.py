import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src.*` works when launched from this subfolder
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime, date, time
from src.domain.geo.zone_manager import ZoneManager
from src.services.model_service import ModelService
import requests
import os
from pathlib import Path as _Path

st.set_page_config(page_title="NYC Taxi Trip Predictor", layout="wide")

# Global header bar (full-width, sticky) replacing the previous st.title call
HEADER_HTML = """
<style>
            /* Lower the header further so the title doesn't collide with Streamlit's top menu */
            .app-header {position:sticky; inset-block-start:48px; z-index:1000; padding:0; margin:0;}
    .app-header-inner {display:flex; align-items:center; gap:18px; padding:18px 34px; background:linear-gradient(90deg,#0f172a 0%, #12263d 55%, #183b5d 100%); box-shadow:0 2px 6px rgba(0,0,0,0.35); border-block-end:1px solid #1e293b;}
    .app-logo {inline-size:42px; block-size:42px; background:radial-gradient(circle at 35% 30%, #38bdf8, #0ea5e9); border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:20px; color:#fff; letter-spacing:0.5px; box-shadow:0 0 0 2px rgba(255,255,255,0.08), 0 4px 10px rgba(0,0,0,0.4);}  
    .app-title-wrap {display:flex; flex-direction:column;}
    .app-title-wrap h1 {margin:0; font-size:1.85rem; font-weight:700; letter-spacing:0.6px; color:#f1f5f9;}
    .app-sub {margin:2px 0 0 0; font-size:0.70rem; font-weight:500; letter-spacing:1.5px; text-transform:uppercase; color:#94a3b8;}
            /* Give page content a little more breathing room under the header */
            .block-container {padding-block-start:2rem;}
    @media (max-inline-size: 860px){ .app-title-wrap h1 {font-size:1.5rem;} .app-logo {inline-size:36px; block-size:36px; font-size:18px;} }
    @media (max-inline-size: 520px){ .app-title-wrap h1 {font-size:1.25rem;} .app-sub {display:none;} }
</style>
<header class="app-header" role="banner" aria-label="Application header">
    <div class="app-header-inner">
        <div class="app-logo" aria-label="App logo">NYC</div>
        <div class="app-title-wrap">
            <h1>Taxi Trip Predictor</h1>
            <div class="app-sub">Real-time zone + route fare & duration estimation</div>
        </div>
    </div>
</header>
"""
st.markdown(HEADER_HTML, unsafe_allow_html=True)

zm = ZoneManager()
svc = ModelService()
if not svc.is_loaded:
    svc.load()
try:
    API_BASE = st.secrets["API_BASE"]  # prefer Streamlit secrets if present
except Exception:
    API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Local assets (optional): place GIFs in assets/weather with names below.
ASSETS_WEATHER = _Path(__file__).resolve().parents[3] / "assets" / "weather"

def _weather_gif_path(kind: str):
    # kind in {clear, overcast, rain, heavy_rain, snow, heavy_snow, windy}
    p = ASSETS_WEATHER / f"{kind}.gif"
    return str(p) if p.exists() else None

# ---------------- Helpers ----------------
# (Helpers unchanged from original file)

def derive_traffic(pickup_dt: datetime):
    wd = pickup_dt.weekday(); hr = pickup_dt.hour
    is_weekend = wd >= 5
    is_rush = (7 <= hr <= 9) or (16 <= hr <= 19)
    if is_rush and not is_weekend: return "Heavy", "Weekday rush hour"
    if is_weekend and (12 <= hr <= 18): return "Moderate", "Weekend midday traffic"
    if hr >= 22 or hr <= 5: return "Light", "Late night/early morning"
    return "Moderate", "Typical daytime flow"

def infer_weather(pickup_dt: datetime):
    m = pickup_dt.month; hr = pickup_dt.hour
    w = {
        'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
        'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
        'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
        'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
        'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0
    }
    label = "Mild / Clear"; reason = "Typical transitional conditions"
    if m in (12,1,2):
        w.update({'temperature':38.0,'feels_like':34.0,'clouds':55.0,'weather_severity':3.0}); label="Cold / Clear"; reason="Winter cool conditions"
        if m in (1,2) and hr < 10:
            w.update({'snow':0.2,'is_snowing':1,'weather_severity':5.0,'clouds':80.0,'is_poor_visibility':1}); label="Cold / Snow"; reason="Morning winter snowfall"
    elif m in (6,7,8):
        w.update({'temperature':85.0,'feels_like':88.0,'humidity':65.0,'clouds':35.0,'weather_severity':2.0}); label="Warm / Sunny"; reason="Typical summer day"
        if hr>=15 and hr<=19 and m in (7,8):
            w.update({'precipitation':0.08,'is_raining':1,'clouds':70.0,'weather_severity':3.0}); label="Warm / Light rain"; reason="Late-day summer shower"
    elif m in (3,4,5):
        w.update({'temperature':62.0,'feels_like':63.0,'humidity':58.0,'clouds':45.0,'weather_severity':2.0}); label="Mild / Partly cloudy"; reason="Spring variability"
        if m==4 and hr<=11:
            w.update({'precipitation':0.05,'is_raining':1,'clouds':75.0}); label="Cool / Light rain"; reason="Morning spring drizzle"
    else:
        w.update({'temperature':60.0,'feels_like':60.0,'humidity':55.0,'clouds':40.0,'weather_severity':2.0}); label="Cool / Clear"; reason="Autumn conditions"
        if m==11 and hr<9:
            w.update({'clouds':65.0,'precipitation':0.04,'is_raining':1}); label="Cool / Light rain"; reason="Early autumn shower"
    return w, label, reason

def infer_holiday(pickup_dt: datetime):
    month=pickup_dt.month; day=pickup_dt.day; wd=pickup_dt.weekday()
    name=None
    if month==1 and day==1: name="New Year's Day"
    elif month==7 and day==4: name="Independence Day"
    elif month==12 and day==25: name="Christmas Day"
    elif month==11 and (22<=day<=28) and wd==3: name="Thanksgiving"
    flags={'is_holiday': int(name is not None), 'is_major_holiday': int(name is not None), 'is_holiday_week': int(name is not None)}
    return flags, name if name else "None"

@st.cache_data(show_spinner=False)
def _cached_zones_geojson() -> str:
    gdf = zm.get_zones_geodataframe()
    return "{}" if gdf is None or gdf.empty else gdf.to_json()

def _mini_bar(label: str, value: float, max_value: float, color: str = "#0066ff"):
    try:
        pct = 0 if max_value <= 0 else max(0, min(100, int((value / max_value) * 100)))
    except Exception:
        pct = 0
    html = f'''
    <div style="margin:4px 0 10px 0;">
        <div style="font-size:12px; margin-block-end:2px;">{label}: {value:.2f} mi</div>
        <div style="background:#2a2a2a; border-radius:6px; block-size:10px; overflow:hidden;">
            <div style="inline-size:{pct}%; background:{color}; block-size:10px;"></div>
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def _trigger_predict():
    st.session_state['trigger_predict'] = True

def _fetch_route_ui(pu_lat: float, pu_lng: float, do_lat: float, do_lng: float, include_steps: bool = False):
    """
    Try FastAPI route endpoint first; if unavailable, fall back to public OSRM for display-only.
    Returns dict: {geometry, distance_miles, duration_minutes, steps, source}
    """
    # 1) Try backend API
    try:
        r = requests.get(
            f"{API_BASE}/route",
            params={
                "pickup_latitude": pu_lat,
                "pickup_longitude": pu_lng,
                "dropoff_latitude": do_lat,
                "dropoff_longitude": do_lng,
                "include_steps": str(include_steps).lower(),
            },
            timeout=6,
        )
        if r.status_code == 200:
            j = r.json()
            if j.get("geometry"):
                j["source"] = "api"
                return j
    except Exception:
        pass

    # 2) Fallback to public OSRM (display-only)
    try:
        osrm_url = "https://router.project-osrm.org/route/v1/driving"
        coords = f"{pu_lng},{pu_lat};{do_lng},{do_lat}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true" if include_steps else "false",
            "annotations": "false",
        }
        rr = requests.get(f"{osrm_url}/{coords}", params=params, timeout=10)
        if rr.status_code == 200:
            data = rr.json()
            if data and data.get("code") == "Ok" and data.get("routes"):
                r0 = data["routes"][0]
                meters = float(r0.get("distance", 0.0)); seconds = float(r0.get("duration", 0.0))
                distance_miles = meters * 0.000621371; duration_minutes = seconds / 60.0
                coords_geo = r0.get("geometry", {}).get("coordinates", [])
                geometry = [(float(lat), float(lon)) for lon, lat in coords_geo]  # swap
                steps = []
                if include_steps and r0.get("legs"):
                    for leg in r0["legs"]:
                        for s in leg.get("steps", []):
                            name = s.get("name") or ""
                            maneuver = s.get("maneuver", {})
                            instr = maneuver.get("instruction") or maneuver.get("type") or "Proceed"
                            steps.append(instr if not name else f"{instr} → {name}")
                return {
                    "geometry": geometry,
                    "distance_miles": distance_miles,
                    "duration_minutes": duration_minutes,
                    "steps": steps,
                    "source": "osrm-public",
                }
    except Exception:
        pass
    return None

with st.sidebar.expander("Options", expanded=True):
    nearest_if_outside = st.toggle("Use nearest zone if outside polygons", True)
    show_route = st.toggle("Show realistic route (OSRM)", True, help="Draw driving route using OSRM API for realistic visualization. Display-only by default.")
    use_route_for_model = st.toggle("Use route distance for prediction (advanced)", False, help="Uses OSRM route distance as the 'estimated_distance' feature. Off by default to stay consistent with model training.")
    show_directions = st.toggle("Show step-by-step directions", False)
    use_live_weather = st.toggle("Use live weather (Open-Meteo)", False, help="Fetch current weather and use it in the prediction. Falls back gracefully if unavailable.")
    default_center = st.selectbox("Map center", ["Midtown","JFK","Custom"], index=0)
    # Default to a colored street map (similar feel to Google Maps)
    map_theme_choice = st.selectbox("Map theme", ["Streets","Light","Dark","Toner"], index=0, help="Choose a basemap for better contrast.")
    st.session_state['map_theme'] = map_theme_choice
    if default_center=="Midtown": center_lat,center_lng,zoom=40.7580,-73.9855,12
    elif default_center=="JFK": center_lat,center_lng,zoom=40.6413,-73.7781,12
    else:
        center_lat = st.number_input("Center latitude", value=40.7128, format="%f")
        center_lng = st.number_input("Center longitude", value=-74.0060, format="%f")
        zoom = st.slider("Zoom",9,16,12)
    st.caption("Map center presets: Midtown = Times Square core (general city view), JFK = airport trips focus, Custom = manually choose a focal point + zoom.")
    show_explain = st.toggle("Show impact explanations", True)
    show_addresses = st.toggle("Show human-readable addresses", True, help="UI will request reverse geocoding from the API; not used for modeling.")

if 'pickup' not in st.session_state: st.session_state['pickup']=None
if 'dropoff' not in st.session_state: st.session_state['dropoff']=None
if 'cached_route' not in st.session_state: st.session_state['cached_route']=None  # {'key':(pu_lat,pu_lng,do_lat,do_lng), 'geometry':[], 'distance':float, 'eta':float, 'steps':[]}
if 'pickup_addr' not in st.session_state: st.session_state['pickup_addr']=None
if 'dropoff_addr' not in st.session_state: st.session_state['dropoff_addr']=None
if 'pickup_addr_key' not in st.session_state: st.session_state['pickup_addr_key']=None
if 'dropoff_addr_key' not in st.session_state: st.session_state['dropoff_addr_key']=None

route_geometry = None
route_distance = None
route_eta = None
route_steps = []
if show_route and st.session_state['pickup'] and st.session_state['dropoff']:
    pu = st.session_state['pickup']; do = st.session_state['dropoff']
    key = (round(pu['lat'],6), round(pu['lng'],6), round(do['lat'],6), round(do['lng'],6), bool(show_directions))
    cached = st.session_state['cached_route']
    if cached and cached.get('key') == key:
        route_geometry = cached['geometry']; route_distance = cached['distance']; route_eta = cached['eta']; route_steps = cached['steps']
    else:
        j = _fetch_route_ui(pu['lat'], pu['lng'], do['lat'], do['lng'], include_steps=show_directions)
        if j and j.get('geometry'):
            route_geometry = j['geometry']
            route_distance = j.get('distance_miles')
            route_eta = j.get('duration_minutes')
            route_steps = j.get('steps', [])
            st.session_state['cached_route'] = {'key':key,'geometry':route_geometry,'distance':route_distance,'eta':route_eta,'steps':route_steps}

# Pre-fetch human-readable addresses for tooltips if enabled
def _get_address_display_string(kind: str) -> str:
    """Generates a user-friendly string for the address fetching status."""
    status = st.session_state.get(f"{kind}_addr_status")
    if status == "fetching":
        return "Fetching address..."
    elif status == "rate_limited":
        return "Address unavailable (Rate limited)"
    elif status == "error":
        return "Address unavailable (API Error)"
    elif status == "unavailable":
        return "Address not available"
    elif status == "ok":
        addr = st.session_state.get(f"{kind}_addr")
        return addr or "Address not available"
    return "Address not available"  # Default/initial state


def _maybe_update_address(kind: str, point: dict):
    key_name = f"{kind}_addr_key"; val_name = f"{kind}_addr"; stat_name = f"{kind}_addr_status"; http_name = f"{kind}_addr_http"; err_name = f"{kind}_addr_error"
    if not point:
        st.session_state[val_name] = None
        st.session_state[key_name] = None
        st.session_state[stat_name] = None
        st.session_state[http_name] = None
        return
    key = (round(point['lat'],6), round(point['lng'],6))
    if st.session_state.get(key_name) == key and st.session_state.get(val_name):
        # already have an address for this coordinate
        st.session_state[stat_name] = st.session_state.get(stat_name, "ok") or "ok"
        st.session_state[http_name] = st.session_state.get(http_name, 200) or 200
        return
    try:
        st.session_state[stat_name] = "fetching"
        st.session_state[http_name] = None
        st.session_state[err_name] = None
        r = requests.get(f"{API_BASE}/geocode/reverse", params={"latitude": point['lat'], "longitude": point['lng']}, timeout=6)
        if r.status_code == 200:
            addr = r.json().get("address")
            if addr:
                st.session_state[val_name] = addr
                st.session_state[key_name] = key
                st.session_state[stat_name] = "ok"
                st.session_state[http_name] = 200
            else:
                st.session_state[val_name] = None
                st.session_state[stat_name] = "unavailable"
                st.session_state[http_name] = 200
        elif r.status_code == 429:
            # Nominatim rate-limited
            st.session_state[val_name] = None
            st.session_state[stat_name] = "rate_limited"
            st.session_state[http_name] = 429
        else:
            st.session_state[val_name] = None
            st.session_state[stat_name] = "unavailable"
            st.session_state[http_name] = r.status_code
    except Exception as e:
        st.session_state[val_name] = None
        st.session_state[stat_name] = "error"
        st.session_state[http_name] = None
        st.session_state[err_name] = str(e)

if show_addresses:
    if st.session_state['pickup']:
        _maybe_update_address('pickup', st.session_state['pickup'])
    if st.session_state['dropoff']:
        _maybe_update_address('dropoff', st.session_state['dropoff'])

    pu_status = st.session_state.get("pickup_addr_status")
    do_status = st.session_state.get("dropoff_addr_status")
    if pu_status == "rate_limited" or do_status == "rate_limited":
        st.warning("Address lookup is currently rate-limited by the API provider. Please wait a moment before retrying.", icon="⚠️")
    elif pu_status == "error" or do_status == "error":
        err = st.session_state.get('pickup_addr_error') or st.session_state.get('dropoff_addr_error')
        st.error(f"Could not fetch addresses due to an API error. Please check backend logs. Details: {err}", icon="🔥")

map_theme = st.session_state.get('map_theme', 'Dark')
tiles_map = {
    'Dark': 'CartoDB Dark_Matter',
    'Light': 'CartoDB Positron',
    'Streets': 'OpenStreetMap',
}
tile_name = tiles_map.get(map_theme, 'OpenStreetMap')

m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, tiles=tile_name)

# Load zones GeoDataFrame and cached geojson (used below)
zones_gdf = zm.get_zones_geodataframe()
zones_json = _cached_zones_geojson()

if zones_gdf is not None and not zones_gdf.empty:
        # Build an augmented copy with centroid info for richer hover tooltips
        try:
            aug = zones_gdf.copy()
            if 'centroid_lat' not in aug.columns:
                centroids = aug.geometry.centroid
                aug['centroid_lat'] = centroids.y
                aug['centroid_lng'] = centroids.x
            # Convert only necessary columns to geojson to reduce payload size
            export_cols = [c for c in ['LocationID','zone','borough','centroid_lat','centroid_lng'] if c in aug.columns]
            geojson_data = aug[export_cols + ['geometry']].to_json()
        except Exception:
            geojson_data = zones_json  # fallback
        folium.GeoJson(
            geojson_data,
            name="NYC Taxi Zones",
            tooltip=folium.GeoJsonTooltip(
                fields=["LocationID","zone","borough","centroid_lat","centroid_lng"],
                aliases=["ID","Zone","Borough","Centroid Lat","Centroid Lng"],
                localize=True
            ),
            style_function=lambda x:{
                "fillColor":"#00000000",  # transparent fill
                "color":"#3186cc",
                "weight":1.2,
                "fillOpacity":0.0
            },
            highlight_function=lambda x:{
                "color":"#1d5fa3",
                "weight":2.5,
                "fillOpacity":0.05
            }
        ).add_to(m)

if st.session_state['pickup']:
    pu = st.session_state['pickup']
    pu_addr_display = _get_address_display_string('pickup') if show_addresses else 'Address lookup disabled'
    pu_tip = f"Pickup: {pu_addr_display} — Zone {pu['zone_id']} - {pu['zone_name']}"
    folium.Marker([pu['lat'], pu['lng']], tooltip=pu_tip, icon=folium.Icon(color='green', icon='play')).add_to(m)
    if zones_gdf is not None and not zones_gdf.empty:
        sel = zones_gdf[zones_gdf['LocationID']==pu['zone_id']]
        if not sel.empty:
            folium.GeoJson(
                sel.to_json(),
                name="Pickup Zone",
                style_function=lambda x:{"fillColor":"#00000000","color":"#33cc33","weight":3,"fillOpacity":0.0}
            ).add_to(m)
if st.session_state['dropoff']:
    do = st.session_state['dropoff']
    do_addr_display = _get_address_display_string('dropoff') if show_addresses else 'Address lookup disabled'
    do_tip = f"Dropoff: {do_addr_display} — Zone {do['zone_id']} - {do['zone_name']}"
    folium.Marker([do['lat'], do['lng']], tooltip=do_tip, icon=folium.Icon(color='red', icon='flag')).add_to(m)
    if zones_gdf is not None and not zones_gdf.empty:
        sel = zones_gdf[zones_gdf['LocationID']==do['zone_id']]
        if not sel.empty:
            folium.GeoJson(
                sel.to_json(),
                name="Dropoff Zone",
                style_function=lambda x:{"fillColor":"#00000000","color":"#ff6666","weight":3,"fillOpacity":0.0}
            ).add_to(m)
if st.session_state['pickup'] and st.session_state['dropoff']:
    if show_route and route_geometry:
        folium.PolyLine(route_geometry, color="#0066ff", weight=5, opacity=0.85).add_to(m)
    elif not show_route:
        folium.PolyLine([[st.session_state['pickup']['lat'], st.session_state['pickup']['lng']],
                         [st.session_state['dropoff']['lat'], st.session_state['dropoff']['lng']]],
                        color="#ffaa00", weight=4, opacity=0.6, dash_array="6,8").add_to(m)

# Add a small legend overlay on the map
legend_html = '''
<style>
.legendBox {position:absolute;inset-block-end:20px;inset-inline-start:20px;z-index:9999;background:rgba(17,24,39,0.85);color:#fff;padding:10px 12px;border-radius:8px;font-size:12px;}
.legendRow{display:flex;align-items:center;margin:3px 0}
.swatch{display:inline-block;inline-size:18px;block-size:6px;margin-inline-end:8px;border-radius:3px}
.swatchBorder{display:inline-block;inline-size:18px;block-size:10px;margin-inline-end:8px;border:2px solid #fff;background:transparent;border-radius:3px}
</style>
<div class="legendBox">
    <div class="legendRow"><span class="swatch" style="background:#0066ff"></span>Route</div>
    <div class="legendRow"><span class="swatchBorder" style="border-color:#3186cc"></span>Zone border</div>
    <div class="legendRow"><span class="swatchBorder" style="border-color:#33cc33"></span>Pickup zone</div>
    <div class="legendRow"><span class="swatchBorder" style="border-color:#ff6666"></span>Dropoff zone</div>
    <div class="legendRow"><span class="swatch" style="background:#2ecc71"></span>Pickup marker</div>
    <div class="legendRow"><span class="swatch" style="background:#e74c3c"></span>Dropoff marker</div>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

st.write("Click to set pickup and dropoff. First click → Pickup, second click → Dropoff. Use Reset to start over.")
map_state = st_folium(m, width=1200, height=600, returned_objects=["last_clicked"], use_container_width=True, key="main_map")

# Initialize last_click tracker to prevent flickering
if 'last_click_coords' not in st.session_state:
    st.session_state['last_click_coords'] = None

if map_state and map_state.get("last_clicked"):
    lat = float(map_state["last_clicked"]["lat"]); lng = float(map_state["last_clicked"]["lng"])
    click_key = (round(lat, 6), round(lng, 6))
    
    # Only process if this is a NEW click (not a rerun of the same click)
    if st.session_state['last_click_coords'] != click_key:
        st.session_state['last_click_coords'] = click_key
        zone_id = zm.get_zone_from_coords(lat, lng, nearest_if_outside=nearest_if_outside)
        if zone_id is not None:
            zone_name = zm.get_zone_name(zone_id)
            if st.session_state['pickup'] is None:
                st.session_state['pickup']={'lat':lat,'lng':lng,'zone_id':zone_id,'zone_name':zone_name}
                st.success(f"Pickup set → Zone {zone_id} - {zone_name}")
                st.rerun()
            elif st.session_state['dropoff'] is None:
                st.session_state['dropoff']={'lat':lat,'lng':lng,'zone_id':zone_id,'zone_name':zone_name}
                st.success(f"Dropoff set → Zone {zone_id} - {zone_name}")
                st.rerun()
            else:
                st.info("Both pickup and dropoff already set. Click Reset to start over.")
        else:
            st.warning(f"Could not resolve ({lat:.6f}, {lng:.6f}) to a zone.")

cols = st.columns(3)
with cols[0]:
    st.button(
        "Predict",
        key="btn_predict_top",
        help="Run prediction using current selections",
        disabled=not (st.session_state['pickup'] and st.session_state['dropoff']),
        on_click=_trigger_predict,
    )
with cols[1]:
    passenger_count = st.number_input("Passenger count", min_value=1, max_value=6, value=1)
with cols[2]:
    # Update the time_str initialization to dynamically fetch the current time during each rerun
    now = datetime.now()
    d_val = st.date_input("Pickup date", value=now.date())
    time_str = st.text_input(
        "Pickup time (HH:MM 24h)", 
        value=st.session_state.get("pickup_time", now.strftime("%H:%M"))
    )

    # Update session state to store the latest time input
    try:
        t_val = datetime.strptime(time_str.strip(), "%H:%M").time()
        st.session_state["pickup_time"] = time_str.strip()
    except ValueError:
        st.warning("Invalid time format. Using current time.")
        t_val = now.time().replace(second=0, microsecond=0)
        st.session_state["pickup_time"] = now.strftime("%H:%M")

    pickup_dt = datetime.combine(d_val, t_val)

# Defer textual trip details until both points are set

if st.session_state['pickup'] and st.session_state['dropoff']:
    pu = st.session_state['pickup']; do = st.session_state['dropoff']
    est_miles = zm.calculate_haversine_distance(pu['zone_id'], do['zone_id'])
    # (Distance outputs moved below prediction & weather panel per user request)
    traffic_level, traffic_reason = derive_traffic(pickup_dt)
    default_weather, weather_label, weather_reason = infer_weather(pickup_dt)
    live_weather_used = False
    if use_live_weather:
        try:
            r = requests.get(f"{API_BASE}/weather/live", params={
                "latitude": pu['lat'],
                "longitude": pu['lng'],
                "when": pickup_dt.isoformat()
            }, timeout=6)
            if r.status_code == 200:
                w = r.json().get("weather")
                if isinstance(w, dict) and w:
                    # Keep only expected prediction keys plus extras for display
                    default_weather.update({k: v for k, v in w.items() if k in default_weather or k in (
                        'visibility_m','source','fetched_at'
                    )})
                    live_weather_used = True
        except Exception:
            pass
    default_holiday, holiday_label = infer_holiday(pickup_dt)
    src_label = "Live weather" if live_weather_used else weather_label
    # Severity badge and icon
    sev = default_weather.get('weather_severity', 1.0)
    if sev >= 5: sev_name, sev_color = "Severe", "#b00020"
    elif sev >= 3: sev_name, sev_color = "Elevated", "#c77700"
    elif sev >= 2: sev_name, sev_color = "Moderate", "#0f766e"
    else: sev_name, sev_color = "Low", "#2e7d32"
    icon = ""  # basic emoji mapping
    if default_weather.get('is_heavy_snow'):
        icon = "❄️"
    elif default_weather.get('is_snowing'):
        icon = "🌨️"
    elif default_weather.get('is_heavy_rain'):
        icon = "🌧️"
    elif default_weather.get('is_raining'):
        icon = "🌦️"
    elif (default_weather.get('clouds', 0) or 0) > 70:
        icon = "☁️"
    else:
        icon = "☀️"

    # Enhanced trip summary card with better visual hierarchy
    trip_summary_html = f"""
    <div style="padding:20px; background:linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); border-radius:12px; margin:16px 0; box-shadow:0 4px 12px rgba(0,0,0,0.3);">
        <div style="font-size:20px; font-weight:700; color:#fff; margin-bottom:14px; border-bottom:2px solid rgba(255,255,255,0.2); padding-bottom:8px;">
            🚕 Trip Overview
        </div>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; margin-top:12px;">
            <div style="background:rgba(255,255,255,0.1); padding:12px; border-radius:8px; border-left:3px solid #60a5fa;">
                <div style="font-size:11px; color:#bfdbfe; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">📅 Date & Time</div>
                <div style="font-size:15px; color:#fff; font-weight:600;">{pickup_dt.strftime('%B %d, %Y')}</div>
                <div style="font-size:14px; color:#e0e7ff;">{pickup_dt.strftime('%I:%M %p')}</div>
            </div>
            <div style="background:rgba(255,255,255,0.1); padding:12px; border-radius:8px; border-left:3px solid #60a5fa;">
                <div style="font-size:11px; color:#bfdbfe; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">👥 Passengers</div>
                <div style="font-size:18px; color:#fff; font-weight:600;">{passenger_count} {'person' if passenger_count == 1 else 'people'}</div>
            </div>
            <div style="background:rgba(255,255,255,0.1); padding:12px; border-radius:8px; border-left:3px solid #60a5fa;">
                <div style="font-size:11px; color:#bfdbfe; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">🚦 Traffic</div>
                <div style="font-size:15px; color:#fff; font-weight:600;">{traffic_level}</div>
                <div style="font-size:12px; color:#e0e7ff;">{traffic_reason}</div>
            </div>
            <div style="background:rgba(255,255,255,0.1); padding:12px; border-radius:8px; border-left:3px solid #60a5fa;">
                <div style="font-size:11px; color:#bfdbfe; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">🌤️ Weather</div>
                <div style="font-size:15px; color:#fff; font-weight:600;">{src_label} {icon}</div>
                <div style="font-size:12px; color:#e0e7ff;">Severity: {sev_name}</div>
            </div>
            <div style="background:rgba(255,255,255,0.1); padding:12px; border-radius:8px; border-left:3px solid #60a5fa;">
                <div style="font-size:11px; color:#bfdbfe; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">🎉 Holiday</div>
                <div style="font-size:15px; color:#fff; font-weight:600;">{holiday_label if holiday_label != 'None' else 'Regular Day'}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(trip_summary_html, unsafe_allow_html=True)
    # Coordinates always shown; addresses if toggle enabled (from session cache)
    addr_pu_display = _get_address_display_string('pickup') if show_addresses else 'Address not available'
    addr_do_display = _get_address_display_string('dropoff') if show_addresses else 'Address not available'
    # User-friendly pickup/dropoff info boxes with enhanced styling
    st.subheader("Trip Locations")
    c1, c2 = st.columns(2)
    
    pu_box = f"""
    <div style="padding:16px; background:linear-gradient(135deg, rgba(34,139,34,0.15) 0%, rgba(34,139,34,0.05) 100%); border-left:4px solid #2ecc71; border-radius:8px; margin-bottom:8px;">
        <div style="font-size:18px; font-weight:600; color:#2ecc71; margin-bottom:8px;">📍 Pickup Location</div>
        <div style="font-size:14px; margin-bottom:6px;"><strong>Zone:</strong> {pu['zone_id']} — {pu['zone_name']}</div>
        <div style="font-size:13px; color:#cbd5e1; margin-bottom:4px;"><strong>Address:</strong> {addr_pu_display}</div>
        <div style="font-size:11px; color:#94a3b8;"><strong>Coordinates:</strong> {pu['lat']:.6f}, {pu['lng']:.6f}</div>
    </div>
    """
    
    do_box = f"""
    <div style="padding:16px; background:linear-gradient(135deg, rgba(220,38,38,0.15) 0%, rgba(220,38,38,0.05) 100%); border-left:4px solid #e74c3c; border-radius:8px; margin-bottom:8px;">
        <div style="font-size:18px; font-weight:600; color:#e74c3c; margin-bottom:8px;">🎯 Dropoff Location</div>
        <div style="font-size:14px; margin-bottom:6px;"><strong>Zone:</strong> {do['zone_id']} — {do['zone_name']}</div>
        <div style="font-size:13px; color:#cbd5e1; margin-bottom:4px;"><strong>Address:</strong> {addr_do_display}</div>
        <div style="font-size:11px; color:#94a3b8;"><strong>Coordinates:</strong> {do['lat']:.6f}, {do['lng']:.6f}</div>
    </div>
    """
    
    with c1:
        st.markdown(pu_box, unsafe_allow_html=True)
    with c2:
        st.markdown(do_box, unsafe_allow_html=True)
    if show_explain:
        bullets=[]
        if traffic_level=='Heavy': bullets.append("Rush hour tends to increase duration and fare uncertainty.")
        if default_weather.get('is_raining'): bullets.append("Rain reduces speed and may increase duration.")
        if default_weather.get('is_snowing'): bullets.append("Snow can significantly increase duration; caution for delays.")
        if default_weather.get('is_heavy_rain') or default_weather.get('is_heavy_snow'): bullets.append("Severe weather conditions — expect notable delays.")
        if holiday_label!='None': bullets.append("Holiday travel patterns can alter typical demand and speeds.")
        bullets.append(f"Weather reasoning: {weather_reason}.")
        if bullets:
            st.info("Impact notes:\n- " + "\n- ".join(bullets))
    # Run prediction if triggered from the top button
    if st.session_state.get('trigger_predict'):
        used_distance = est_miles
        if use_route_for_model and route_distance and route_distance > 0:
            used_distance = float(route_distance)

        # Ensure we have a route drawn if user just clicked Predict (route might not have been fetched yet in rare race conditions)
        if show_route and (route_geometry is None or not route_geometry):
            j2 = _fetch_route_ui(pu['lat'], pu['lng'], do['lat'], do['lng'], include_steps=show_directions)
            if j2 and j2.get('geometry'):
                route_geometry = j2['geometry']
                route_distance = j2.get('distance_miles')
                route_eta = j2.get('duration_minutes')
                route_steps = j2.get('steps', [])
                st.session_state['cached_route'] = {
                    'key': (round(pu['lat'],6), round(pu['lng'],6), round(do['lat'],6), round(do['lng'],6), bool(show_directions)),
                    'geometry': route_geometry,
                    'distance': route_distance,
                    'eta': route_eta,
                    'steps': route_steps
                }

        feats = svc.build_features(
            pickup_zone_id=pu['zone_id'],
            dropoff_zone_id=do['zone_id'],
            pickup_datetime=pickup_dt,
            passenger_count=passenger_count,
            estimated_distance=used_distance,
            weather=default_weather,
            holiday=default_holiday,
        )
        preds = svc.predict(feats)
        mph_lower = (est_miles / (preds['duration_minutes']/60.0)) if preds['duration_minutes'] > 0 else float('inf')
        fpm = (preds['fare_amount'] / est_miles) if est_miles > 0 else float('inf')

        distance_mode = "route distance (advanced)" if (use_route_for_model and route_distance) else "centroid distance (model-safe)"
        card_html = f'''
        <div style="margin-block:8px; padding:14px 16px; background:rgba(16,24,39,0.65); border:1px solid #224e6a; border-radius:10px;">
          <div style="font-size:16px; font-weight:700; margin-block-end:6px;">Prediction</div>
          <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin-block-end:8px;">
            <div style="background:#14532d; color:#e8f5e9; padding:6px 10px; border-radius:8px; font-weight:600;">Duration: {preds['duration_minutes']:.2f} min</div>
            <div style="background:#1e3a8a; color:#e0f2fe; padding:6px 10px; border-radius:8px; font-weight:600;">Fare: ${preds['fare_amount']:.2f}</div>
            <div style="background:#334155; color:#e2e8f0; padding:6px 10px; border-radius:8px;">Using {distance_mode}: {used_distance:.2f} mi</div>
          </div>
          <div style="font-size:12px; color:#cbd5e1;">The model estimates travel time and fare based on pickup/dropoff zones, time, weather, and historical patterns. Distance mode indicates which distance fed the model.</div>
          <div style="margin-block-start:8px; font-size:12px; color:#94a3b8;">Derived (for context): implied avg speed (lower bound) {mph_lower:.1f} mph · fare per mile ${fpm:.2f}. Road distance is typically longer than straight-line, so true average speed can be higher.</div>
        </div>
        '''
        st.markdown(card_html, unsafe_allow_html=True)

        adv=[]
        if traffic_level in ("Heavy","Moderate"): adv.append(f"Traffic: {traffic_level.lower()} ({traffic_reason}).")
        if default_weather.get('is_raining') or default_weather.get('is_snowing'): adv.append(f"Expected weather: {weather_label.lower()} may slow travel.")
        if holiday_label!='None': adv.append(f"Holiday: {holiday_label} may affect variability.")
        if adv: st.warning(" ".join(adv))
        # Reset the trigger so it doesn't auto-run on subsequent reruns
        st.session_state['trigger_predict'] = False

    # Compact live weather panel (less prominent, supplementary info)
    if live_weather_used:
        lw = default_weather
        w_icon = "☀️"; w_desc = "Clear"
        if lw.get('is_heavy_snow'): w_icon, w_desc = "❄️", "Heavy Snow"
        elif lw.get('is_snowing'): w_icon, w_desc = "🌨️", "Snow"
        elif lw.get('is_heavy_rain'): w_icon, w_desc = "🌧️", "Heavy Rain"
        elif lw.get('is_raining'): w_icon, w_desc = "🌦️", "Rain"
        elif (lw.get('clouds',0) or 0) > 70: w_icon, w_desc = "☁️", "Overcast"
        elif lw.get('wind_speed',0) > 20: w_icon, w_desc = "💨", "Windy"
        
        sev = lw.get('weather_severity'); sev_label = "Low" if sev <= 1 else ("Moderate" if sev < 3 else ("Elevated" if sev < 5 else "Severe"))
        flags=[]; 
        if lw.get('is_raining'): flags.append("Rain")
        if lw.get('is_snowing'): flags.append("Snow")
        if lw.get('is_poor_visibility'): flags.append("Low Vis")
        
        # Compact weather details in an expander
        with st.expander(f"🌤️ Live Weather Details ({w_desc})", expanded=False):
            st.caption(f"Source: {lw.get('source','')} · Fetched: {lw.get('fetched_at','')} (Open-Meteo)")
            
            # Compact single-row metrics display
            weather_compact = f"""
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(100px, 1fr)); gap:8px; margin:8px 0; font-size:12px;">
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">TEMP</div>
                    <div style="font-weight:600; color:#e2e8f0;">{lw.get('temperature'):.0f}°F</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">FEELS</div>
                    <div style="font-weight:600; color:#e2e8f0;">{lw.get('feels_like'):.0f}°F</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">HUMIDITY</div>
                    <div style="font-weight:600; color:#e2e8f0;">{lw.get('humidity'):.0f}%</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">WIND</div>
                    <div style="font-weight:600; color:#e2e8f0;">{lw.get('wind_speed'):.1f} mph</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">CLOUDS</div>
                    <div style="font-weight:600; color:#e2e8f0;">{lw.get('clouds'):.0f}%</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">PRECIP</div>
                    <div style="font-weight:600; color:#e2e8f0;">{lw.get('precipitation'):.2f} mm</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">VISIBILITY</div>
                    <div style="font-weight:600; color:#e2e8f0;">{(lw.get('visibility_m',10000)/1000):.1f} km</div>
                </div>
                <div style="padding:6px; background:rgba(59,130,246,0.1); border-radius:6px; text-align:center;">
                    <div style="color:#94a3b8; font-size:10px;">SEVERITY</div>
                    <div style="font-weight:600; color:#e2e8f0;">{sev_label}</div>
                </div>
            </div>
            """
            st.markdown(weather_compact, unsafe_allow_html=True)
            if flags:
                st.caption(f"⚠️ Conditions: {', '.join(flags)}")

    # Distance section now at the bottom of this block
    st.write(f"Model-safe distance (zone centroids): {est_miles:.2f} miles")
    if show_route and route_geometry and route_distance is not None:
        st.success(f"Route distance: {route_distance:.2f} mi")

    # Reset button placed at bottom for clearer flow
    if st.button("Reset selections"):
        st.session_state['pickup']=None; st.session_state['dropoff']=None

    if show_route and show_directions and route_steps:
        with st.expander("Turn-by-turn Directions"):
            for i, step in enumerate(route_steps[:100], start=1):
                st.write(f"{i}. {step}")
            st.caption("Driving instructions supplied by OSRM (OpenStreetMap data). Display-only; model uses zone features + optional route distance.")

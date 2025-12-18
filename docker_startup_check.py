#!/usr/bin/env python3
"""Startup check for Docker containers.
Prints presence of key data and model files and then execs the passed command.
Usage (Docker CMD):
    python docker_startup_check.py <cmd> <arg1> <arg2> ...
Example:
    python docker_startup_check.py uvicorn src.interface.api.fastapi_app:app --host 0.0.0.0 --port 8000
"""
import os
import sys
import glob
import subprocess
import geopandas as gpd

ROOT = os.path.abspath(os.path.dirname(__file__))
print("[startup-check] ROOT:", ROOT)

# Candidate files/paths to check (relative to project root)
checks = [
    os.path.join(ROOT, 'Data', 'zones', 'taxi_zones.geojson'),
    os.path.join(ROOT, 'Data', 'zones', 'taxi_zones.shp'),
    os.path.join(ROOT, 'Data', 'zones', 'taxi_zone_lookup.csv'),
    os.path.join(ROOT, 'models'),
]

print("[startup-check] Checking presence of important files and folders:")
for p in checks:
    exists = os.path.exists(p)
    print(f"  - {p}: {'FOUND' if exists else 'MISSING'}")

# List model files
model_files = glob.glob(os.path.join(ROOT, 'models', '*'))
if model_files:
    print(f"[startup-check] Model files ({len(model_files)}):")
    for mf in model_files:
        print("   -", os.path.relpath(mf, ROOT))
else:
    print("[startup-check] No model files found in ./models/")

# List Data/zones contents
zones_files = glob.glob(os.path.join(ROOT, 'Data', 'zones', '*'))
if zones_files:
    print(f"[startup-check] Data/zones files ({len(zones_files)}):")
    for z in zones_files:
        print("   -", os.path.relpath(z, ROOT))
else:
    print("[startup-check] No files found in ./Data/zones/")

# Quick runtime test: try reading the geojson with geopandas to verify GDAL/pyogrio availability
try:
    import geopandas as gpd
    geojson_path = os.path.join(ROOT, 'Data', 'zones', 'taxi_zones.geojson')
    if os.path.exists(geojson_path):
        try:
            gdf = gpd.read_file(geojson_path)
            geom_types = gdf.geom_type.unique().tolist() if not gdf.empty else []
            print(f"[startup-check] geopandas read success: {len(gdf)} features; geom_types={geom_types}")
        except Exception as e:
            print(f"[startup-check] geopandas failed to read geojson: {e}")
    else:
        print("[startup-check] geopandas test skipped (geojson missing)")
except Exception as e:
    print(f"[startup-check] geopandas import failed: {e}")

# Print Python version
print("[startup-check] Python version:", sys.version)

# List installed packages
try:
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    print("[startup-check] Installed packages:\n", installed_packages.decode('utf-8'))
except Exception as e:
    print("[startup-check] Failed to list installed packages:", e)

# Check GeoPandas functionality
try:
    zones_gdf = gpd.read_file("Data/zones/taxi_zones.geojson")
    print("[startup-check] GeoPandas read success: ", len(zones_gdf), "features")
    print("[startup-check] Geometry types:", zones_gdf.geom_type.unique())
except Exception as e:
    print("[startup-check] GeoPandas read failed:", e)

# Small environment summary
print("[startup-check] Environment variables relevant to app:")
for key in ['API_BASE', 'API_URL', 'PYTHONUNBUFFERED', 'STREAMLIT_SERVER_PORT']:
    print(f"  - {key} = {os.environ.get(key)}")

# If no args given, just exit after checks
if len(sys.argv) == 1:
    print("[startup-check] No command supplied. Exiting after checks.")
    sys.exit(0)

# Execute the requested command, replacing the current process
cmd = sys.argv[1:]
print("[startup-check] Executing command:", ' '.join(cmd))

# Exec the command
os.execvp(cmd[0], cmd)

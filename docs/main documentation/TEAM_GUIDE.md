# 🚕 NYC Taxi Prediction - Team Onboarding Guide

**Welcome to the team!** This guide will help you understand, navigate, and work with the NYC Taxi Trip Prediction system.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Architecture](#project-architecture)
4. [Key Concepts](#key-concepts)
5. [Important Files](#important-files)
6. [How to Run](#how-to-run)
7. [Common Tasks](#common-tasks)
8. [Understanding the Data](#understanding-the-data)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 📊 Project Overview

### What Does This Project Do?

Predicts **two things** for NYC Yellow Taxi trips:
1. **Trip Duration** (in minutes) - How long will the trip take?
2. **Fare Amount** (in USD) - How much will it cost?

### How Good Are the Models?

- **Duration Model**: 88% accuracy (R² = 0.8796)
- **Fare Model**: 94% accuracy (R² = 0.9437)

Both models use **36.6 million trips** from 2022 with **56 carefully engineered features**.

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Models** | LightGBM | Gradient boosting for predictions |
| **Backend API** | FastAPI | RESTful API endpoints |
| **Frontend UI** | Streamlit | Interactive web interface |
| **Geospatial** | GeoPandas | NYC taxi zone management |
| **Routing** | OSRM | Route visualization |
| **Weather** | Open-Meteo | Live weather data |
| **Geocoding** | Nominatim | Address lookup |

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project folder
cd "Taxi Trips ML Final project"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Application

**Easiest Way - One Command:**

```powershell
# Windows PowerShell
./start_all.ps1

# Or Python (any OS)
python start_all.py
```

This starts:
- **API**: http://localhost:8000 (backend)
- **UI**: http://localhost:8501 (frontend)

### 3. Test It Works

Open your browser:
- API Docs: http://localhost:8000/docs
- Interactive Map: http://localhost:8501

Click zones on the map and hit **Predict**!

---

## 🏗️ Project Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                      │
│              (Streamlit - Port 8501)                    │
│  - Interactive map with 263 NYC taxi zones              │
│  - Real-time predictions                                │
│  - Weather info, route visualization                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ HTTP Requests
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                   (Port 8000)                           │
│  Endpoints:                                             │
│  - POST /predict  (trip predictions)                    │
│  - GET  /route    (routing service)                     │
│  - GET  /geocode  (address lookup)                      │
│  - GET  /weather  (weather data)                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Calls Services
┌─────────────────────────────────────────────────────────┐
│                   Services Layer                         │
│  ┌─────────────┬────────────┬──────────────────────┐   │
│  │Model Service│Routing     │Weather   │Geocoding  │   │
│  │(Predictions)│(OSRM)      │(Open-    │(Nominatim)│   │
│  │             │            │Meteo)    │           │   │
│  └─────────────┴────────────┴──────────┴───────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Uses Domain Logic
┌─────────────────────────────────────────────────────────┐
│                    Domain Layer                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ZoneManager (GeoPandas)                          │  │
│  │ - Manages 263 NYC taxi zones                     │  │
│  │ - Calculates distances between zones             │  │
│  │ - Spatial indexing and lookups                   │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼ Loads Data
┌─────────────────────────────────────────────────────────┐
│                  Data & Models                           │
│  - Trained models (duration_model_final.pkl, etc.)     │
│  - NYC taxi zone shapefiles (263 zones)                │
│  - Weather data, holiday calendars                      │
└─────────────────────────────────────────────────────────┘
```

### Clean Architecture Principles

**Three-Layer Design:**

1. **Domain Layer** (`src/domain/`)
   - Core business logic
   - No external dependencies
   - Pure functions and data structures
   - Example: Zone geometry calculations

2. **Services Layer** (`src/services/`)
   - Application services
   - Coordinates domain logic with external APIs
   - Handles caching, error handling
   - Examples: Model predictions, weather API calls

3. **Interface Layer** (`src/interface/`)
   - User-facing interfaces
   - API endpoints (FastAPI)
   - Web UI (Streamlit)
   - Adapts user input to service calls

**Benefits:**
- ✅ Easy to test (mock services, not UI)
- ✅ Easy to maintain (clear responsibilities)
- ✅ Easy to extend (add new interfaces or services)
- ✅ Business logic isolated from UI changes

---

## 🧠 Key Concepts

### 1. Data Leakage Prevention

**Critical concept:** Models must ONLY use information available at prediction time.

**✅ Safe Features (Available at Pickup):**
- Pickup zone ID (user selects)
- Current date/time
- Weather at pickup moment
- Holiday calendar
- Estimated distance from zone centroids

**❌ Forbidden Features (Data Leakage):**
- Dropoff time (happens in future)
- Actual trip distance (only known after trip)
- Fare amount (what we're predicting!)
- Trip duration (what we're predicting!)

**Why This Matters:**
- If you train with future information, model looks amazing (99% accuracy!)
- But fails completely in production (can't predict future)
- Our models use ONLY safe features → work in real deployment

### 2. Zone-Based System

**NYC TLC Dataset Uses Zones, Not GPS:**
- Dataset has zone IDs (1-263), NOT latitude/longitude
- Each zone is a polygon (neighborhood/area)
- We calculate distance between zone **centroids** (center points)

**Zone Workflow:**
```
User clicks map → Selects zone → Get zone ID → Lookup centroid
                                                       ↓
                                            Calculate distance
                                                       ↓
                                               Feed to model
```

**For Display Only:**
- OSRM routing: Shows road-based route
- Reverse geocoding: Shows street addresses
- These are NOT model inputs (just UI enhancements)

### 3. 80/20 Random Split

**Training Methodology:**
- Take ALL 36.6M trips from entire year (Jan-Dec 2022)
- Randomly shuffle everything
- 80% for training (29.2M trips)
- 20% for testing (7.3M trips)

**Why Random (Not Temporal)?**
- Both sets have all seasons
- No seasonal overfitting
- Better real-world generalization
- Supports ensemble modeling

### 4. Feature Engineering

**56 Safe Features Created:**

| Category | Count | Examples |
|----------|-------|----------|
| Location | 9 | Zone IDs, airport flags, Manhattan flags |
| Temporal | 15 | Hour, weekday, rush hour, cyclical encodings |
| Distance | 1 | Haversine distance (zone centroids) |
| Weather | 15 | Temperature, rain, snow, wind |
| Holiday | 3 | Holiday flags, major holidays |
| Interactions | 13 | Weather×Location, Time×Distance |

**Cyclical Encoding Example:**
```python
# Hour 23 and hour 0 are close (midnight), not far apart!
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

---

## 📂 Important Files

### Must-Know Files

#### **1. Models (Load These for Predictions)**

```
models/lightgbm_80_20_full_year/
├── duration_model_final.pkl  ← Duration prediction
├── fare_model_final.pkl       ← Fare prediction
├── model_metrics.json         ← Performance metrics
└── README.md                  ← Model documentation
```

**How to load:**
```python
import pickle

with open('models/lightgbm_80_20_full_year/duration_model_final.pkl', 'rb') as f:
    duration_model = pickle.load(f)
```

#### **2. Core Application Code**

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/domain/geo/zone_manager.py` | Zone management | `get_zone_from_coords()`, `calculate_haversine_distance()` |
| `src/services/model_service.py` | ML predictions | `predict_duration()`, `predict_fare()` |
| `src/services/weather_service.py` | Weather API | `get_current_weather()`, `fetch_weather_features()` |
| `src/services/routing_service.py` | Route calculation | `get_route()`, `calculate_route_distance()` |
| `src/interface/api/fastapi_app.py` | REST API | `POST /predict`, `GET /route`, `GET /weather` |
| `src/interface/web/streamlit_app.py` | Web UI | Full interactive map interface |

#### **3. Feature Engineering Scripts**

```
src/feature_engineering/
├── engineer_safe_features.py      ← 56 safe features (NO leakage)
└── engineer_enhanced_features.py  ← Training-only features
```

**Use `engineer_safe_features.py` for production!**

#### **4. Data Files**

```
Data/
├── splits_cleaned/           ← Preprocessed features (80/20 split)
│   ├── train/               ← Training data (29.2M trips)
│   └── test/                ← Test data (7.3M trips)
│
├── taxi_zones/              ← NYC TLC shapefiles
│   └── taxi_zones.shp       ← 263 zone polygons (GeoPandas)
│
└── external/                ← Weather and holidays
    ├── weather_2022.parquet
    └── holidays_2022.csv
```

#### **5. Documentation**

| File | What You'll Learn |
|------|-------------------|
| `README.md` | Project overview, quick start |
| `docs/TEAM_GUIDE.md` | This file! Complete team guide |
| `MODEL_EXPERIMENTS_DOCUMENTATION.md` | Model development history |
| `FASTAPI_STREAMLIT_GUIDE.md` | API/UI architecture details |
| `notebooks/project_summary_and_feature_engineering.ipynb` | Feature engineering explained |
| `notebooks/full_year_preprocessing_eda.ipynb` | EDA with 80/20 methodology |

#### **6. Launch Scripts**

| File | Platform | Purpose |
|------|----------|---------|
| `start_all.py` | Any OS | Python launcher (cross-platform) |
| `start_all.ps1` | Windows | PowerShell launcher |
| `requirements.txt` | Any OS | Python dependencies |

---

## 🏃 How to Run

### Option 1: Unified Launch (Recommended)

**Starts both API and UI with one command:**

```powershell
# Windows PowerShell
./start_all.ps1

# Python (any OS)
python start_all.py
```

**What happens:**
1. Launches FastAPI on port 8000 (background)
2. Launches Streamlit on port 8501 (foreground)
3. Opens browser automatically

**Access:**
- API: http://localhost:8000/docs
- UI: http://localhost:8501

### Option 2: Manual Launch (Separate Terminals)

**Terminal 1 - Start API:**
```bash
uvicorn src.interface.api.fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Start UI:**
```bash
streamlit run src/interface/web/streamlit_app.py --server.port 8501
```

### Option 3: API Only (No UI)

```bash
uvicorn src.interface.api.fastapi_app:app --host 127.0.0.1 --port 8000
```

Test with cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_location_id": 161,
    "dropoff_location_id": 230,
    "pickup_datetime": "2024-07-15T14:30:00",
    "passenger_count": 2
  }'
```

### Option 4: UI Only (API Must Be Running)

```bash
streamlit run src/interface/web/streamlit_app.py --server.port 8501
```

---

## 🛠️ Common Tasks

### Task 1: Make a Prediction (Python)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pickup_location_id": 161,  # JFK Airport
        "dropoff_location_id": 230, # Times Square
        "pickup_datetime": "2024-07-15T14:30:00",
        "passenger_count": 2
    }
)

result = response.json()
print(f"Duration: {result['duration_minutes']:.2f} minutes")
print(f"Fare: ${result['fare_amount']:.2f}")
print(f"Distance: {result['distance_km']:.2f} km")
```

### Task 2: Load and Use Models Directly

```python
import pickle
import pandas as pd
import numpy as np

# Load models
with open('models/lightgbm_80_20_full_year/duration_model_final.pkl', 'rb') as f:
    duration_model = pickle.load(f)

with open('models/lightgbm_80_20_full_year/fare_model_final.pkl', 'rb') as f:
    fare_model = pickle.load(f)

# Prepare features (must be exactly 56 features in correct order)
features = pd.DataFrame([{
    'PULocationID': 161,
    'DOLocationID': 230,
    'passenger_count': 2,
    'pickup_hour': 14,
    'pickup_day': 15,
    'pickup_month': 7,
    'pickup_weekday': 4,
    # ... (all 56 features)
}])

# Make predictions
duration = duration_model.predict(features)[0]  # seconds
fare = fare_model.predict(features)[0]  # dollars

print(f"Predicted Duration: {duration/60:.2f} minutes")
print(f"Predicted Fare: ${fare:.2f}")
```

### Task 3: Calculate Distance Between Zones

```python
from src.domain.geo.zone_manager import ZoneManager

zone_manager = ZoneManager(zones_file='Data/taxi_zones/taxi_zones.shp')

# Get distance between two zones
distance_km = zone_manager.calculate_haversine_distance(
    pickup_zone_id=161,   # JFK Airport
    dropoff_zone_id=230   # Times Square
)

print(f"Distance: {distance_km:.2f} km")
```

### Task 4: Get Live Weather

```python
from src.services.weather_service import WeatherService
from datetime import datetime

weather_service = WeatherService()

# Get weather for specific datetime
weather = weather_service.get_weather_features(
    latitude=40.7128,  # NYC
    longitude=-74.0060,
    datetime_obj=datetime(2024, 7, 15, 14, 30)
)

print(f"Temperature: {weather['temperature']}°F")
print(f"Is Raining: {weather['is_raining']}")
```

### Task 5: Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder
# Open: project_summary_and_feature_engineering.ipynb
# or:   full_year_preprocessing_eda.ipynb
```

---

## 📊 Understanding the Data

### NYC Taxi Zones

**263 Zones covering NYC:**
- Manhattan: 69 zones
- Brooklyn: 61 zones
- Queens: 70 zones (including airports)
- Bronx: 43 zones
- Staten Island: 22 zones

**Special Zones:**
- Zone 1: Newark Airport (EWR)
- Zone 132: JFK Airport
- Zone 138: LaGuardia Airport
- Zones 103, 107, 113: Major Manhattan areas

**Data Structure:**
```
Zone 161 (JFK Airport):
  - zone_name: "JFK Airport"
  - borough: "Queens"
  - geometry: Polygon (boundary)
  - centroid: (40.6413, -73.7781)
```

### Feature Data Flow

```
User Input:
  Pickup Zone: 161 (JFK)
  Dropoff Zone: 230 (Times Square)
  DateTime: 2024-07-15 14:30
  Passengers: 2

        ↓ Feature Engineering Pipeline

1. Location Features (9):
   - PULocationID: 161
   - DOLocationID: 230
   - pickup_is_airport: 1
   - dropoff_is_manhattan: 1
   - ...

2. Temporal Features (15):
   - pickup_hour: 14
   - pickup_weekday: 0 (Monday)
   - is_rush_hour: 0
   - hour_sin: sin(2π × 14 / 24)
   - ...

3. Distance Feature (1):
   - estimated_distance: 17.5 km (Haversine)

4. Weather Features (15):
   - temperature: 82°F
   - is_raining: 0
   - ...

5. Holiday Features (3):
   - is_holiday: 0
   - ...

6. Interaction Features (13):
   - distance_hour_interaction: 17.5 × 14
   - weather_airport_interaction: 0
   - ...

        ↓ Total: 56 Features

Model Input: [161, 230, 2, 14, 15, 7, 0, ...]

        ↓ LightGBM Prediction

Output:
  Duration: 45.23 minutes
  Fare: $68.50
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Trips** | 36,556,803 |
| **Training** | 29,245,436 (80%) |
| **Testing** | 7,311,367 (20%) |
| **Time Period** | Jan-Dec 2022 |
| **Features** | 56 safe features |
| **Zones** | 263 NYC zones |
| **Average Trip** | ~15 minutes, $18 |

---

## 🐛 Troubleshooting

### Issue 1: Port Already in Use

**Symptom:** Error: "Address already in use" or "OSError: [WinError 10048]"

**Solution:**
```powershell
# Find process using port 8000 (API)
netstat -ano | findstr :8000

# Kill it (replace <PID> with actual number)
taskkill /PID <PID> /F

# Same for port 8501 (UI)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue 2: Module Not Found

**Symptom:** `ModuleNotFoundError: No module named 'src'` 

**Solution:**
```bash
# Ensure you're in project root
cd "Taxi Trips ML Final project"

# Verify directory
dir  # Should see: src/, models/, Data/, etc.

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Reinstall if needed
pip install -r requirements.txt
```

### Issue 3: Model File Not Found

**Symptom:** `FileNotFoundError: models/lightgbm_80_20_full_year/duration_model_final.pkl`

**Solution:**
```bash
# Check files exist
dir models\lightgbm_80_20_full_year\*.pkl

# Should show:
# - duration_model_final.pkl
# - fare_model_final.pkl

# If missing, models weren't included - contact team lead
```

### Issue 4: Shapefile Not Found

**Symptom:** `FileNotFoundError: Data/taxi_zones/taxi_zones.shp`

**Solution:**
```bash
# Check shapefiles exist
dir Data\taxi_zones\*.shp

# Shapefiles consist of multiple files:
# - taxi_zones.shp
# - taxi_zones.shx
# - taxi_zones.dbf
# - taxi_zones.prj

# If missing, download from NYC TLC website
```

### Issue 5: Prediction Returns NaN

**Symptom:** Model returns `NaN` or unrealistic values

**Possible Causes:**
1. Missing features (need exactly 56)
2. Features in wrong order
3. Invalid zone IDs (must be 1-263)
4. Missing weather data (NaN values)

**Solution:**
```python
# Validate features before prediction
features_df = create_features(...)  # Your feature function

# Check shape
assert features_df.shape[1] == 56, f"Expected 56 features, got {features_df.shape[1]}"

# Check for NaN
assert not features_df.isnull().any().any(), "Features contain NaN values"

# Check zone IDs
assert 1 <= features_df['PULocationID'].iloc[0] <= 263
assert 1 <= features_df['DOLocationID'].iloc[0] <= 263
```

### Issue 6: Slow Predictions

**Symptom:** API takes >2 seconds per prediction

**Possible Causes:**
1. Models not cached (loading from disk each time)
2. Weather API timeout
3. OSRM routing slow

**Solution:**
```python
# Check if models are loaded at startup
# In fastapi_app.py, models should load ONCE:

from src.services.model_service import ModelService

# This happens at startup (not per request)
model_service = ModelService(models_dir="models/lightgbm_80_20_full_year")

# If still slow, enable caching:
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_weather_cached(lat, lon, date_str):
    # Weather API call
    ...
```

---

## ✅ Best Practices

### For Development

1. **Always use virtual environment**
   ```bash
   .venv\Scripts\Activate.ps1
   ```

2. **Keep models in memory**
   - Load models ONCE at startup
   - Don't reload for each prediction

3. **Test data leakage**
   - Before adding features, ask: "Is this available at prediction time?"
   - Document feature sources

4. **Use type hints**
   ```python
   def predict_duration(features: pd.DataFrame) -> float:
       """Predict trip duration in seconds."""
       ...
   ```

5. **Handle errors gracefully**
   ```python
   try:
       weather = weather_service.get_weather(...)
   except Exception as e:
       logger.warning(f"Weather API failed: {e}")
       weather = get_default_weather()  # Fallback
   ```

### For Code Review

**Before Submitting:**
- [ ] Code follows clean architecture (domain/services/interface)
- [ ] No data leakage (features only use pickup-time info)
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Error handling implemented

**Checklist for New Features:**
1. Is it available at prediction time?
2. Does it improve model performance?
3. Is it computationally efficient?
4. Is it documented?

### For Deployment

1. **Environment variables**
   ```bash
   # Use .env file for configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   MODEL_DIR=models/lightgbm_80_20_full_year
   ```

2. **Logging**
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   logger.info("Prediction request received")
   ```

3. **Monitoring**
   - Track prediction latency
   - Monitor model accuracy over time
   - Alert on API errors

---

## 📚 Additional Resources

### Internal Documentation

- **Model Details**: `models/lightgbm_80_20_full_year/README.md`
- **API Guide**: `FASTAPI_STREAMLIT_GUIDE.md`
- **Model History**: `MODEL_EXPERIMENTS_DOCUMENTATION.md`

### External Resources

- **LightGBM Docs**: https://lightgbm.readthedocs.io/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **GeoPandas Docs**: https://geopandas.org/
- **NYC TLC Data**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### Learning Path

**New to Project?**
1. Read this guide (you're here!)
2. Run the application and explore UI
3. Read `notebooks/project_summary_and_feature_engineering.ipynb`
4. Explore API endpoints at http://localhost:8000/docs
5. Review key files in `src/services/`

**Want to Modify Models?**
1. Read `MODEL_EXPERIMENTS_DOCUMENTATION.md`
2. Study `src/feature_engineering/engineer_safe_features.py`
3. Review `notebooks/full_year_preprocessing_eda.ipynb`
4. Understand 80/20 split methodology

**Want to Improve UI?**
1. Read `FASTAPI_STREAMLIT_GUIDE.md`
2. Explore `src/interface/web/streamlit_app.py`
3. Test changes locally before deploying

---

## 🤝 Getting Help

### Internal Contacts

Team Contact: [Jorge Rubio, Andres Benavides, Joaquín Cano, Mario Minero, Yanela Varela]



### Reporting Issues

Create an issue with:
1. **Description**: What's wrong?
2. **Steps to reproduce**: How to trigger the issue?
3. **Expected behavior**: What should happen?
4. **Actual behavior**: What actually happens?
5. **Environment**: OS, Python version, etc.
6. **Logs**: Error messages, stack traces

### Contributing

1. Create feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and test thoroughly
3. Update documentation
4. Submit pull request with clear description
5. Address review comments

---

## 🎯 Quick Reference

### Common Commands

```bash
# Activate environment
.venv\Scripts\Activate.ps1

# Launch app
./start_all.ps1

# Run tests
pytest tests/

# Check code style
black src/
flake8 src/

# View logs
tail -f logs/api.log
```

### Key Directories

```
src/               ← All application code
models/            ← Trained models
Data/              ← Datasets and shapefiles
notebooks/         ← Jupyter notebooks
docs/              ← Documentation
tests/             ← Test files
```

### Important URLs

- **API Docs**: http://localhost:8000/docs
- **UI**: http://localhost:8501
- **ReDoc**: http://localhost:8000/redoc

---

**Welcome aboard! 🚕 Happy coding!**

*Last updated: November 2025*

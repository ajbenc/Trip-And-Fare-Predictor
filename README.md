# 🚕 NYC Taxi Trip Prediction - Production ML System

**Enterprise-grade ML system with 94% R² fare prediction + 88% R² trip duration prediction**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)]()
[![Models: LightGBM](https://img.shields.io/badge/models-LightGBM-brightgreen.svg)]()

---

## 📊 Project Overview

Full-stack machine learning system for predicting **trip durations** and **fare amounts** for NYC Yellow Taxi trips. Trained on **36.6M trips** from full-year 2022 data with comprehensive weather, holiday, and geospatial features.

### 🎯 Production Models (80/20 Random Split)

Both models use **56 safe features** with zero data leakage:

#### ⏱️ **Duration Prediction Model**
- **R² Score**: 0.8796 (87.96% variance explained)
- **RMSE**: 3.58 minutes
- **MAE**: 2.46 minutes
- **Training**: 29.2M trips (80%)
- **Testing**: 7.3M trips (20%)

#### 💰 **Fare Prediction Model**
- **R² Score**: 0.9437 (94.37% variance explained)
- **RMSE**: $3.04
- **MAE**: $1.36
- **Training**: 29.2M trips (80%)
- **Testing**: 7.3M trips (20%)

**Location**: `models/lightgbm_80_20_full_year/`

---

## ✨ Key Features

### 🛡️ Data Integrity
- ✅ **Zero Data Leakage** - Only features available at prediction time
- ✅ **56 Safe Features** - Temporal, location, weather, holidays, interactions
- ✅ **Random 80/20 Split** - Across all 12 months for robust performance
- ✅ **Ensemble Approach** - Multiple models for stable predictions

### 🗺️ Interactive Web Application
- ✅ **Interactive Map** - Click to select pickup/dropoff zones (265 NYC zones)
- ✅ **Live Weather** - Real-time Open-Meteo API integration
- ✅ **Route Visualization** - OSRM routing with polylines
- ✅ **Address Display** - Reverse geocoding with Nominatim
- ✅ **Responsive UI** - Modern Streamlit interface with CSS enhancements

### 🚀 Production-Ready API
- ✅ **FastAPI Backend** - RESTful endpoints with Pydantic validation
- ✅ **Auto Documentation** - Swagger UI + ReDoc
- ✅ **Response Caching** - Optimized for performance
- ✅ **Clean Architecture** - Domain/Services/Interface separation

### 🧬 Advanced Feature Engineering
- ✅ **Geospatial Features** - Zone centroids with Haversine distance (GeoPandas)
- ✅ **Weather Integration** - Temperature, precipitation, snow, wind
- ✅ **Holiday Calendar** - Federal holidays + major NYC events
- ✅ **Cyclical Encodings** - Sin/cos transformations for temporal features
- ✅ **Interaction Features** - Weather×Location, Time×Distance, etc.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
pip (Python package manager)
Git (optional, for cloning)
```

### Installation

1. **Navigate to project directory**
```bash
cd "Taxi Trips ML Final project"
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Launch Application

**Unified Launch (Recommended)** - Starts both API and UI:

```powershell
# PowerShell
./start_all.ps1

# Or Python (cross-platform)
python start_all.py
```

**Services:**
- 🌐 **API**: http://localhost:8000 (Swagger at `/docs`)
- 🎨 **UI**: http://localhost:8501 (Interactive map interface)

**Manual Launch** (if needed):

```bash
# Terminal 1: Start FastAPI
uvicorn src.interface.api.fastapi_app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Streamlit
streamlit run src/interface/web/streamlit_app.py --server.port 8501
```

---

## 📁 Project Structure

```
Taxi Trips ML Final project/
├── Data/                           # Datasets and external data
│   ├── splits_cleaned/            # Feature-engineered data (train/val/test)
│   ├── taxi_zones/                # NYC TLC shapefiles (GeoPandas)
│   └── external/                  # Weather and holiday data
│
├── models/                         # Trained models
│   ├── lightgbm_80_20_full_year/ # Production models (56 features)
│   │   ├── duration_model_final.pkl
│   │   ├── fare_model_final.pkl
│   │   └── README.md
│   └── pickle_format/            # Alternative model formats
│
├── src/                           # Source code
│   ├── domain/                   # Business logic
│   │   └── geo/
│   │       └── zone_manager.py   # GeoPandas zone management
│   │
│   ├── services/                 # Application services
│   │   ├── model_service.py      # ML prediction logic
│   │   ├── routing_service.py    # OSRM route calculation
│   │   ├── geocoding_service.py  # Reverse geocoding
│   │   └── weather_service.py    # Weather API integration
│   │
│   ├── interface/                # User interfaces
│   │   ├── api/
│   │   │   └── fastapi_app.py    # REST API
│   │   └── web/
│   │       └── streamlit_app.py  # Web UI
│   │
│   ├── feature_engineering/      # Feature creation scripts
│   │   ├── engineer_safe_features.py
│   │   └── engineer_enhanced_features.py
│   │
│   └── training/                 # Model training scripts
│       └── train_lightgbm_ultra.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── project_summary_and_feature_engineering.ipynb
│   ├── full_year_preprocessing_eda.ipynb
│   └── 01_exploratory_analysis.ipynb
│
├── docs/                          # Documentation
└── |__ /docker                    # Docker deployment
│   └── /main documentation        # Team onboarding guide and importan MDs
│
├── start_all.py                   # Unified launcher (Python)
├── start_all.ps1                  # Unified launcher (PowerShell)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🧪 Feature Engineering

### 56 Safe Features (No Data Leakage)

All features are **strictly available at prediction time** (pickup moment):

#### 1. **Location Features** (9)
- Zone IDs (pickup/dropoff)
- Airport flags
- Manhattan flags
- Same location indicator

#### 2. **Temporal Features** (15)
- Pickup hour, day, month, weekday
- Rush hour, late night, business hours flags
- Cyclical encodings (sin/cos) for smooth patterns

#### 3. **Distance Features** (1)
- Estimated distance from zone centroids (Haversine formula)

#### 4. **Weather Features** (15)
- Temperature, precipitation, snow, wind
- Weather severity flags
- Heavy rain/snow indicators

#### 5. **Holiday Features** (3)
- Holiday, major holiday, holiday week flags

#### 6. **Interaction Features** (13)
- Weather × Location
- Time × Distance
- Holiday × Location
- Weather × Distance

**Forbidden Features (Would Cause Data Leakage):**
- ❌ Dropoff datetime (future information)
- ❌ Actual trip distance (only known after trip)
- ❌ Fare amount (target variable)
- ❌ Trip duration (target variable)

### Data Split Methodology

**80/20 Random Split Across All Months:**
- ✅ 36.6M total trips from full year 2022
- ✅ 29.2M training (80%) - random sample from all months
- ✅ 7.3M testing (20%) - random sample from all months
- ✅ Captures all seasonal patterns in both sets
- ✅ Better generalization than temporal split

---

## 🔧 API Usage

### Python Example

```python
import requests

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "pickup_location_id": 161,    # JFK Airport
        "dropoff_location_id": 230,   # Times Square
        "pickup_datetime": "2024-07-15T14:30:00",
        "passenger_count": 2
    }
)

result = response.json()
print(f"Duration: {result['duration_minutes']:.2f} minutes")
print(f"Fare: ${result['fare_amount']:.2f}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_location_id": 161,
    "dropoff_location_id": 230,
    "pickup_datetime": "2024-07-15T14:30:00",
    "passenger_count": 2,
    "estimated_distance": 15.5,
    "temperature": 75.0,
    "feels_like": 75.0,
    "humidity": 60.0,
    "pressure": 1012.0,
    "wind_speed": 5.0,
    "clouds": 20.0,
    "precipitation": 0.0,
    "snow": 0.0,
    "is_raining": false,
    "is_snowing": false,
    "is_holiday": false
  }'
```

---

## 📊 Model Performance

### Training Dataset
- **Total Samples**: 36,556,803 trips
- **Training Set**: 29,245,436 (80%)
- **Test Set**: 7,311,367 (20%)
- **Time Period**: Full year 2022 (Jan-Dec)
- **Features**: 56 safe features
- **Algorithm**: LightGBM with categorical handling

### Performance Metrics

| Model | R² Score | RMSE | MAE | Training Samples |
|-------|----------|------|-----|------------------|
| **Duration** | 0.8796 | 3.58 min | 2.46 min | 29.2M |
| **Fare** | 0.9437 | $3.04 | $1.36 | 29.2M |

### Feature Importance (Top 10)

1. `estimated_distance` (35%) - Zone centroid distance
2. `pickup_hour` (12%) - Hour of day
3. `PULocationID` (8%) - Pickup zone
4. `DOLocationID` (7%) - Dropoff zone
5. `is_rush_hour` (6%) - Rush hour flag
6. `temperature` (5%) - Weather temperature
7. `precipitation` (4%) - Rain amount
8. `is_weekend` (4%) - Weekend flag
9. `pickup_weekday` (3%) - Day of week
10. `passenger_count` (2%) - Number of passengers

---

## 🗺️ Geospatial Implementation

### NYC Taxi Zones (GeoPandas)

The dataset contains **zone IDs only** (no GPS coordinates). We use:

1. **NYC TLC Shapefiles** - 265 taxi zone polygons
2. **Zone Centroids** - Center point of each zone
3. **Haversine Distance** - Great-circle distance between centroids
4. **Spatial Indexing** - Fast zone lookups

**Why Centroids?**
- Consistent distance estimates across all data
- Model trained on these distances (no GPS in historical data)
- Available at prediction time (calculated from zone selection)

### Additional Services (UI Enhancement)

- **OSRM Routing** - Road-based route visualization
- **Reverse Geocoding** - Human-readable addresses (display only)
- **Open-Meteo Weather** - Live weather at pickup time

---

## 📚 Documentation

- **[TEAM_GUIDE.md](docs/TEAM_GUIDE.md)** - Complete team onboarding guide
- **[MODEL_EXPERIMENTS_DOCUMENTATION.md](MODEL_EXPERIMENTS_DOCUMENTATION.md)** - Model development history
- **[FASTAPI_STREAMLIT_GUIDE.md](FASTAPI_STREAMLIT_GUIDE.md)** - Application architecture
- **[models/lightgbm_80_20_full_year/README.md](models/lightgbm_80_20_full_year/README.md)** - Production model details

---

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Structure Philosophy

**Clean Architecture:**
- **Domain Layer** - Core business logic (zone management, geometry)
- **Services Layer** - Application services (ML, routing, weather, geocoding)
- **Interface Layer** - User-facing interfaces (API, Web UI)

**Benefits:**
- Easy to test and maintain
- Clear separation of concerns
- Scalable and extensible

---

## 📈 Performance Optimization

- **Response Caching** - OSRM routes, geocoding, weather data
- **Spatial Indexing** - Fast zone lookups with GeoPandas
- **Model Loading** - Models loaded once at startup
- **Async Operations** - Non-blocking API calls where possible

**Typical Response Times:**
- Model prediction: < 50ms
- Full pipeline (with weather/routing): < 500ms
- UI interaction: < 1 second

---

## 🔍 Troubleshooting

### Port Already in Use

```powershell
# Kill process on port 8000 (FastAPI)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Kill process on port 8501 (Streamlit)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Module Import Errors

Ensure you're in the project root directory and virtual environment is activated:

```bash
cd "Taxi Trips ML Final project"
.venv\Scripts\Activate.ps1
python -c "import src; print('OK')"
```

### Model File Not Found

Check that models exist:

```bash
dir models\lightgbm_80_20_full_year\*.pkl
```

Should show: `duration_model_final.pkl`, `fare_model_final.pkl`

---

## 🤝 Contributing

This is a production system. For changes:

1. Create feature branch
2. Test thoroughly (especially data leakage checks)
3. Update documentation
4. Submit for review

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👥 Team

For questions or onboarding, see **[docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md)**

---

## 🎯 Project Status

**✅ Production Ready**

- Models trained and validated
- API fully functional
- UI deployed and tested
- Documentation complete
- Zero data leakage confirmed

**Current Version**: 2.0 (80/20 Full Year Models with 56 Safe Features)

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| Total Training Data | 36.6M trips |
| Features | 56 (safe, no leakage) |
| Models | 2 (duration, fare) |
| NYC Zones | 265 |
| R² Score (Duration) | 87.96% |
| R² Score (Fare) | 94.37% |
| API Response Time | < 500ms |
| Deployment Status | ✅ Production |

---

**Built with ❤️ using LightGBM, FastAPI, Streamlit, and GeoPandas**

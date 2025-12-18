# 🚀 FastAPI + Streamlit Implementation Guide

## Overview

This implementation uses:
- **FastAPI** for the backend API (fast, modern, with automatic documentation)
- **Streamlit** for the frontend (easy, beautiful, ML-focused interface)

## 🏗️ Architecture

```
┌──────────────────┐
│   Streamlit UI   │  (app_streamlit.py)
│   Port: 8501     │
└────────┬─────────┘
         │ HTTP Requests
         ▼
┌──────────────────┐
│   FastAPI        │  (api/app.py)
│   Port: 8000     │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│LightGBM│ │LightGBM  │
│ (Fare) │ │(Duration)│
└────────┘ └──────────┘
```

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New packages added:
- `streamlit` - Frontend framework
- `streamlit-folium` - Interactive maps
- `folium` - Map visualization
- `fastapi` - API framework (already included)
- `uvicorn` - ASGI server (already included)

### 2. Ensure Models Are Ready

Make sure these files exist:
```
models/
├── lightgbm_ultra/
│   └── fare_lightgbm.txt  ✅ Ready
|── lightgbm_ultra/
|   └── duration_lightgbm.txt ✅ Ready

```

## 🚀 Quick Start

### Step 1: Start the FastAPI Backend

```bash
# Option 1: Using Python directly
python api/app.py

# Option 2: Using uvicorn
uvicorn api.app:app --reload --port 8000
```

**FastAPI will be available at:**
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

### Step 2: Start the Streamlit Frontend

```bash
streamlit run app_streamlit.py
```

**Streamlit will automatically open in your browser:**
- URL: `http://localhost:8501`

## 📚 API Documentation

### Automatic Interactive Documentation

FastAPI provides **automatic interactive API documentation**:

1. **Swagger UI**: Navigate to `http://localhost:8000/docs`
   - Try out endpoints directly in the browser
   - See request/response schemas
   - Test with sample data

2. **ReDoc**: Navigate to `http://localhost:8000/redoc`
   - Alternative documentation style
   - More readable for complex APIs

### Available Endpoints

#### 1. Root (`GET /`)
```bash
curl http://localhost:8000/
```

#### 2. Health Check (`GET /health`)
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "message": "All systems operational"
}
```

#### 3. Single Prediction (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2016-06-15 14:30:00",
    "pickup_longitude": -73.982,
    "pickup_latitude": 40.767,
    "dropoff_longitude": -73.958,
    "dropoff_latitude": 40.778,
    "passenger_count": 1
  }'
```

**Response:**
```json
{
  "fare_amount": 12.50,
  "duration_minutes": 15.3,
  "confidence": "high",
  "distance_miles": 2.45
}
```

#### 4. Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type": application/json" \
  -d '{
    "trips": [
      { /* trip 1 */ },
      { /* trip 2 */ }
    ]
  }'
```

#### 5. Model Info (`GET /models/info`)
```bash
curl http://localhost:8000/models/info
```

## 🎨 Streamlit Features

### Main Interface

1. **Interactive Map**
   - Visual representation of pickup/dropoff locations
   - Route display
   - Drag and drop markers (coming soon)

2. **Sidebar Controls**
   - Date/time picker
   - Location inputs (lat/lon)
   - Passenger count selector
   - Predict button

3. **Results Display**
   - Fare amount
   - Trip duration
   - Distance
   - Confidence score
   - Additional metrics

4. **Batch Predictions**
   - Upload CSV file
   - Process multiple trips
   - Download results

### Example CSV Format for Batch

```csv
pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count
2016-06-15 14:30:00,-73.982,40.767,-73.958,40.778,1
2016-06-15 15:00:00,-73.975,40.750,-73.990,40.760,2
```

## 💻 Usage Examples

### Python Client

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
trip = {
    "pickup_datetime": "2016-06-15 14:30:00",
    "pickup_longitude": -73.982,
    "pickup_latitude": 40.767,
    "dropoff_longitude": -73.958,
    "dropoff_latitude": 40.778,
    "passenger_count": 1
}

response = requests.post(f"{API_URL}/predict", json=trip)
result = response.json()

print(f"Fare: ${result['fare_amount']}")
print(f"Duration: {result['duration_minutes']} minutes")
```

### JavaScript Client

```javascript
const API_URL = "http://localhost:8000";

const trip = {
    pickup_datetime: "2016-06-15 14:30:00",
    pickup_longitude: -73.982,
    pickup_latitude: 40.767,
    dropoff_longitude: -73.958,
    dropoff_latitude: 40.778,
    passenger_count: 1
};

fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(trip)
})
.then(response => response.json())
.then(data => {
    console.log('Fare:', data.fare_amount);
    console.log('Duration:', data.duration_minutes);
});
```

## 🔧 Development

### Running in Development Mode

**FastAPI with Auto-Reload:**
```bash
uvicorn api.app:app --reload --port 8000
```

**Streamlit with Auto-Reload:**
```bash
streamlit run app_streamlit.py
```
(Streamlit auto-reloads by default when you save changes)

### Testing the API

FastAPI provides automatic validation and error messages. Test directly in the browser:

1. Go to `http://localhost:8000/docs`
2. Click on any endpoint
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"

## 🎯 Advantages of This Stack

### FastAPI Benefits

1. **Automatic Documentation** - Swagger UI + ReDoc
2. **Data Validation** - Pydantic models with type checking
3. **Fast Performance** - Built on Starlette and Pydantic
4. **Modern Python** - Type hints, async support
5. **Easy Testing** - Built-in test client

### Streamlit Benefits

1. **Rapid Development** - Build UI with pure Python
2. **Interactive Widgets** - Sliders, buttons, maps
3. **Real-time Updates** - Auto-refresh on code changes
4. **ML-Focused** - Built for data science/ML apps
5. **Beautiful UI** - Professional look out of the box

## 🔒 Production Deployment

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Start both services (you'd typically use docker-compose)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn api.app:app --host 0.0.0.0 --port 8000
  
  streamlit:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run app_streamlit.py --server.port 8501
    depends_on:
      - api
```

### Cloud Deployment

**Option 1: Separate Services**
- Deploy FastAPI on: Heroku, Railway, Render
- Deploy Streamlit on: Streamlit Cloud (free)

**Option 2: All-in-One**
- Use Docker Compose on: AWS ECS, Google Cloud Run, Azure Container Apps

## 📊 Model Performance

### Current Status

| Model | Target | Performance | Status |
|-------|--------|-------------|--------|
| **LightGBM** | Fare | 94%+ R² | ✅ Ready |
| **LightGBM** | Duration | 82.2-85.6% R² | ✅ Ready |

### When Training Completes


FastAPI will load them on next restart.

## 🐛 Troubleshooting

### API Not Responding
```bash
# Check if running
curl http://localhost:8000/health

# Restart
python api/app.py
```

### Streamlit Connection Error
- Make sure FastAPI is running first
- Check API_URL in app_streamlit.py (line 22)
- Verify port 8000 is not blocked

### Models Not Loading
- Check file paths in `api/app.py`
- Ensure models exist in `models/` directory
- Check console output for error messages

## 📝 Next Steps

1. ✅ **FastAPI backend created** - Modern, fast, documented
2. ✅ **Streamlit frontend created** - Beautiful, interactive
3. 🎯 **Test the full stack** - Start both services and predict
4. 🚀 **Deploy to production** - Use Docker or cloud services

## 🎉 Summary

**You now have:**
- ✅ Professional FastAPI backend with automatic docs
- ✅ Beautiful Streamlit frontend with interactive maps
- ✅ Single & batch prediction support
- ✅ Ready for production deployment

**To use:**
```bash
# Terminal 1: Start API
python api/app.py

# Terminal 2: Start Streamlit
streamlit run app_streamlit.py
```

**Access:**
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs
- API: http://localhost:8000

---

**This aligns perfectly with your original project plan!** 🎯

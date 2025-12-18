from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Tuple
from datetime import datetime
from src.domain.geo.zone_manager import ZoneManager
from src.services.model_service import ModelService
from src.services.routing_service import RoutingService
from src.services.geocoding_service import GeocodingService
from src.services.weather_service import WeatherService

app = FastAPI(title="NYC Taxi Zone Resolver", version="1.0.0")

zm = ZoneManager()
svc = ModelService()
if not svc.is_loaded:
    svc.load()
router = RoutingService()
geocoder = GeocodingService()
weather = WeatherService()

class ZoneResolution(BaseModel):
    latitude: float
    longitude: float
    zone_id: Optional[int]
    zone_name: Optional[str]
    nearest_fallback: bool

class PredictRequest(BaseModel):
    pickup_zone_id: Optional[int] = None
    dropoff_zone_id: Optional[int] = None
    pickup_latitude: Optional[float] = None
    pickup_longitude: Optional[float] = None
    dropoff_latitude: Optional[float] = None
    dropoff_longitude: Optional[float] = None
    passenger_count: int = 1
    pickup_datetime: Optional[datetime] = None
    estimated_distance: Optional[float] = None
    nearest_if_outside: bool = True
    use_live_weather: bool = False

class PredictResponse(BaseModel):
    pickup_zone_id: int
    dropoff_zone_id: int
    estimated_distance: float
    duration_minutes: float
    fare_amount: float

class RouteResponse(BaseModel):
    distance_miles: float
    duration_minutes: float
    geometry: List[Tuple[float, float]]
    steps: List[str] = []

class BatchRouteItem(BaseModel):
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    include_steps: bool = False

class BatchRouteResponse(BaseModel):
    routes: List[RouteResponse]

@app.get("/")
def root():
    return {
        "service": "NYC Taxi Trip Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "resolve": "GET /resolve - Resolve coordinates to taxi zones",
            "predict": "POST /predict - Predict taxi trip fare and duration",
            "route": "GET /route - Get route between two points",
            "batch_route": "POST /route/batch - Get multiple routes",
            "reverse_geocode": "GET /geocode/reverse - Reverse geocode coordinates",
            "weather": "GET /weather/live - Get live weather data"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/resolve", response_model=ZoneResolution)
def resolve_coordinate(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    nearest_if_outside: bool = Query(True, description="Return nearest zone if outside polygons")
):
    zone_id = zm.get_zone_from_coords(latitude, longitude, nearest_if_outside=nearest_if_outside)
    zone_name = zm.get_zone_name(zone_id) if zone_id is not None else None
    return ZoneResolution(
        latitude=latitude,
        longitude=longitude,
        zone_id=zone_id,
        zone_name=zone_name,
        nearest_fallback=nearest_if_outside and zone_id is not None
    )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pu_zone = req.pickup_zone_id
    do_zone = req.dropoff_zone_id
    if pu_zone is None and req.pickup_latitude is not None and req.pickup_longitude is not None:
        pu_zone = zm.get_zone_from_coords(req.pickup_latitude, req.pickup_longitude, nearest_if_outside=req.nearest_if_outside)
    if do_zone is None and req.dropoff_latitude is not None and req.dropoff_longitude is not None:
        do_zone = zm.get_zone_from_coords(req.dropoff_latitude, req.dropoff_longitude, nearest_if_outside=req.nearest_if_outside)
    if pu_zone is None or do_zone is None:
        raise ValueError("Could not resolve pickup or dropoff zone.")

    est_dist = req.estimated_distance or zm.calculate_haversine_distance(pu_zone, do_zone)
    dt = req.pickup_datetime or datetime(2022, 7, 15, 10, 30)
    if req.use_live_weather and req.pickup_latitude is not None and req.pickup_longitude is not None:
        weather_feats = weather.get_weather_features(req.pickup_latitude, req.pickup_longitude, when=dt)
    else:
        weather_feats = {
            'temperature': 72.0, 'feels_like': 74.0, 'humidity': 60.0,
            'pressure': 1013.0, 'wind_speed': 5.0, 'clouds': 30.0,
            'precipitation': 0.0, 'snow': 0.0, 'weather_severity': 1.0,
            'is_raining': 0, 'is_snowing': 0, 'is_heavy_rain': 0,
            'is_heavy_snow': 0, 'is_extreme_weather': 0, 'is_poor_visibility': 0
        }
    holiday = {'is_holiday': 0, 'is_major_holiday': 0, 'is_holiday_week': 0}

    feats = svc.build_features(
        pickup_zone_id=pu_zone,
        dropoff_zone_id=do_zone,
        pickup_datetime=dt,
        passenger_count=req.passenger_count,
        estimated_distance=est_dist,
    weather=weather_feats,
        holiday=holiday,
    )
    preds = svc.predict(feats)
    return PredictResponse(
        pickup_zone_id=pu_zone,
        dropoff_zone_id=do_zone,
        estimated_distance=est_dist,
        duration_minutes=preds['duration_minutes'],
        fare_amount=preds['fare_amount'],
    )

# Run with: uvicorn src.interface.api.fastapi_app:app --reload

@app.get("/route", response_model=RouteResponse)
def get_route(
    pickup_latitude: float = Query(..., ge=-90, le=90),
    pickup_longitude: float = Query(..., ge=-180, le=180),
    dropoff_latitude: float = Query(..., ge=-90, le=90),
    dropoff_longitude: float = Query(..., ge=-180, le=180),
    include_steps: bool = Query(False)
):
    r = router.get_route(
        pu_lat=pickup_latitude,
        pu_lng=pickup_longitude,
        do_lat=dropoff_latitude,
        do_lng=dropoff_longitude,
        steps=include_steps,
    )
    if not r:
        return RouteResponse(distance_miles=0.0, duration_minutes=0.0, geometry=[], steps=[])
    return RouteResponse(
        distance_miles=r.distance_miles,
        duration_minutes=r.duration_minutes,
        geometry=r.geometry,
        steps=r.steps,
    )

@app.post("/route/batch", response_model=BatchRouteResponse)
def get_routes(items: List[BatchRouteItem]):
    results: List[RouteResponse] = []
    for it in items:
        rr = router.get_route(
            pu_lat=it.pickup_latitude,
            pu_lng=it.pickup_longitude,
            do_lat=it.dropoff_latitude,
            do_lng=it.dropoff_longitude,
            steps=it.include_steps,
        )
        if not rr:
            results.append(RouteResponse(distance_miles=0.0, duration_minutes=0.0, geometry=[], steps=[]))
        else:
            results.append(RouteResponse(
                distance_miles=rr.distance_miles,
                duration_minutes=rr.duration_minutes,
                geometry=rr.geometry,
                steps=rr.steps,
            ))
    return BatchRouteResponse(routes=results)

class ReverseGeocodeResponse(BaseModel):
    address: str = ""

@app.get("/geocode/reverse", response_model=ReverseGeocodeResponse)
def reverse_geocode(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    addr = geocoder.reverse(latitude, longitude) or ""
    return ReverseGeocodeResponse(address=addr)


class LiveWeatherResponse(BaseModel):
    weather: dict

@app.get("/weather/live", response_model=LiveWeatherResponse)
def live_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    when: Optional[datetime] = None,
):
    feats = weather.get_weather_features(latitude, longitude, when=when)
    return LiveWeatherResponse(weather=feats)

"""
Flask REST API for NYC Taxi Fare & Duration Prediction
Provides endpoints for real-time predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from prediction.taxi_predictor import TaxiPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize predictor (global)
predictor = None


@app.before_request
def load_models():
    """Load models on first request"""
    global predictor
    if predictor is None:
        print("🔄 Loading models...")
        predictor = TaxiPredictor(models_dir="models")
        predictor.load_models()
        print("✅ Models ready!")


@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'NYC Taxi Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict fare and duration for a single trip',
            '/predict/batch': 'POST - Predict for multiple trips',
            '/health': 'GET - Check API health'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = predictor is not None and \
                    predictor.fare_model is not None and \
                    predictor.duration_model is not None
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded
    }), 200 if models_loaded else 503


@app.route('/predict', methods=['POST'])
def predict_trip():
    """
    Predict fare and duration for a single trip
    
    Example Request:
    POST /predict
    {
        "pickup_datetime": "2016-06-15 14:30:00",
        "pickup_longitude": -73.982,
        "pickup_latitude": 40.767,
        "dropoff_longitude": -73.958,
        "dropoff_latitude": 40.778,
        "passenger_count": 1
    }
    
    Example Response:
    {
        "fare_amount": 12.50,
        "duration_minutes": 15.3,
        "confidence": "high",
        "trip_details": {...}
    }
    """
    try:
        # Get request data
        trip_data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude'
        ]
        
        missing_fields = [field for field in required_fields if field not in trip_data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Make prediction
        result = predictor.predict(trip_data)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict fare and duration for multiple trips
    
    Example Request:
    POST /predict/batch
    {
        "trips": [
            {
                "pickup_datetime": "2016-06-15 14:30:00",
                "pickup_longitude": -73.982,
                "pickup_latitude": 40.767,
                "dropoff_longitude": -73.958,
                "dropoff_latitude": 40.778,
                "passenger_count": 1
            },
            { ... }
        ]
    }
    
    Example Response:
    {
        "predictions": [
            {...},
            {...}
        ],
        "count": 2
    }
    """
    try:
        # Get request data
        data = request.get_json()
        trips = data.get('trips', [])
        
        if not trips:
            return jsonify({
                'error': 'No trips provided',
                'message': 'Request body must contain "trips" array'
            }), 400
        
        # Make predictions
        results = predictor.predict_batch(trips)
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/models/info', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    return jsonify({
        'fare_model': {
            'type': 'LightGBM',
            'target': 'fare_amount',
            'performance': {
                'r2_score': '~94%',
                'mae': '~$1.50'
            }
        },
        'duration_model': {
            'type': 'MLP Neural Network',
            'target': 'trip_duration',
            'architecture': '[512, 256, 128, 64, 32]',
            'performance': {
                'r2_score': '~90-95%',
                'mae': '~3-4 minutes'
            }
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚕 NYC TAXI PREDICTION API")
    print("="*60)
    print("📍 Starting server on http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000/")
    print("🏥 Health Check: http://localhost:5000/health")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

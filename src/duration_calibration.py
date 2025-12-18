"""
Realistic Duration Calibration for ULTRA Model
This module provides post-processing adjustments to make predictions more realistic
"""
import numpy as np

def calibrate_duration(
    predicted_duration: float,
    distance: float,
    weather_severity: int,
    is_rush_hour: bool,
    is_late_night: bool,
    is_raining: bool,
    is_snowing: bool,
    precipitation: float,
    snow: float,
    is_airport_trip: bool,
    pickup_is_manhattan: bool
) -> dict:
    """
    Calibrate predicted duration to be more realistic based on physical constraints
    
    Args:
        predicted_duration: Model's raw prediction (minutes)
        distance: Trip distance (miles)
        weather_severity: 0-3 scale
        is_rush_hour: Rush hour flag
        is_late_night: Late night flag (11 PM - 5 AM)
        is_raining: Rain flag
        is_snowing: Snow flag  
        precipitation: Rain amount (mm/hour)
        snow: Snow amount (mm/hour)
        is_airport_trip: Airport trip flag
        pickup_is_manhattan: Manhattan pickup flag
        
    Returns:
        dict with calibrated_duration, min_duration, max_duration, adjustment_reason
    """
    
    # Start with highway speed limit as maximum
    max_speed = 60.0  # mph (NYC area highway speed limit)
    
    adjustment_reasons = []
    
    # === Weather Adjustments ===
    if is_snowing:
        if snow > 8.0:
            # Extreme blizzard: Dangerous driving conditions
            max_speed = min(max_speed, 25.0)
            adjustment_reasons.append(f"Extreme blizzard ({snow:.1f}mm/hr) - max 25 mph")
        elif snow > 5.0:
            # Heavy snow: Very slow driving
            max_speed = min(max_speed, 30.0)
            adjustment_reasons.append(f"Heavy snow ({snow:.1f}mm/hr) - max 30 mph")
        elif snow > 2.0:
            # Moderate snow: Cautious driving
            max_speed = min(max_speed, 40.0)
            adjustment_reasons.append(f"Moderate snow ({snow:.1f}mm/hr) - max 40 mph")
    
    if is_raining:
        if precipitation > 7.0:
            # Heavy rain: Reduced visibility
            max_speed = min(max_speed, 35.0)
            adjustment_reasons.append(f"Heavy rain ({precipitation:.1f}mm/hr) - max 35 mph")
        elif precipitation > 4.0:
            # Moderate rain: Wet roads
            max_speed = min(max_speed, 45.0)
            adjustment_reasons.append(f"Moderate rain ({precipitation:.1f}mm/hr) - max 45 mph")
    
    # === Traffic Adjustments ===
    if is_rush_hour:
        if pickup_is_manhattan:
            # Manhattan rush hour: Severe gridlock
            max_speed = min(max_speed, 20.0)
            adjustment_reasons.append("Manhattan rush hour - max 20 mph")
        else:
            # General rush hour: Heavy traffic
            max_speed = min(max_speed, 30.0)
            adjustment_reasons.append("Rush hour traffic - max 30 mph")
        
        # Airport trips in rush hour: Add tunnel/bridge delays
        if is_airport_trip:
            max_speed = min(max_speed, 25.0)
            adjustment_reasons.append("Airport + rush hour - bridge/tunnel delays")
    
    # === Combined Extreme Conditions ===
    if weather_severity >= 2 and is_rush_hour:
        # Worst case: Bad weather + traffic
        max_speed = min(max_speed, 18.0)
        adjustment_reasons.append("Combined: Weather + Traffic - max 18 mph")
    
    # === Late Night Bonus ===
    if is_late_night and not is_rush_hour:
        # Late night gets closer to model prediction (empty roads)
        if weather_severity < 2:
            # Good weather late night: Trust model more
            max_speed = max(max_speed, 50.0)
            adjustment_reasons.append("Late night empty roads - allow faster")
    
    # Calculate minimum realistic duration based on max speed
    min_duration_physical = (distance / max_speed) * 60.0  # Convert to minutes
    
    # Also consider minimum speed (stopped in gridlock)
    min_speed = 8.0  # mph (worst case gridlock)
    max_duration_physical = (distance / min_speed) * 60.0  # Maximum time
    
    # Apply calibration
    if predicted_duration < min_duration_physical:
        # Model too optimistic - use physical constraint
        calibrated = min_duration_physical
        confidence = "Low (calibrated up)"
        adjustment = f"+{calibrated - predicted_duration:.1f} min (physical constraint)"
    elif predicted_duration > max_duration_physical:
        # Model too pessimistic (rare) - cap at worst case
        calibrated = max_duration_physical
        confidence = "Low (calibrated down)"
        adjustment = f"{calibrated - predicted_duration:.1f} min (capped)"
    else:
        # Model prediction is within realistic bounds
        # Apply gentle calibration toward conservative
        if distance > 15 and not is_late_night:
            # Long trips: Add conservative buffer
            calibrated = predicted_duration * 1.3
            calibrated = min(calibrated, max_duration_physical)
            confidence = "Medium (long trip buffer)"
            adjustment = f"+{calibrated - predicted_duration:.1f} min (long trip adjustment)"
        else:
            # Short/medium trips: Trust model more
            calibrated = predicted_duration * 1.1
            confidence = "High (model reliable)"
            adjustment = f"+{calibrated - predicted_duration:.1f} min (minor adjustment)"
    
    # Calculate speed from calibrated duration
    calibrated_speed = (distance / (calibrated / 60.0)) if calibrated > 0 else 0
    
    return {
        'original_duration': round(predicted_duration, 1),
        'calibrated_duration': round(calibrated, 1),
        'min_duration': round(min_duration_physical, 1),
        'max_duration': round(max_duration_physical, 1),
        'original_speed': round((distance / (predicted_duration / 60.0)), 1) if predicted_duration > 0 else 0,
        'calibrated_speed': round(calibrated_speed, 1),
        'max_speed_limit': round(max_speed, 1),
        'adjustment': adjustment,
        'confidence': confidence,
        'reasons': adjustment_reasons,
        'improvement': round(((calibrated - predicted_duration) / predicted_duration) * 100, 1)
    }


# Example usage
if __name__ == "__main__":
    print("🔧 Duration Calibration Examples\n" + "="*60)
    
    # Test Case 1: Extreme blizzard, late night
    print("\n1. NYE Blizzard (Late Night)")
    print("   Original: Times Square → Rockaway (28.5 mi) @ 11:59 PM")
    result1 = calibrate_duration(
        predicted_duration=12.4,
        distance=28.5,
        weather_severity=3,
        is_rush_hour=False,
        is_late_night=True,
        is_raining=False,
        is_snowing=True,
        precipitation=0.0,
        snow=12.0,
        is_airport_trip=False,
        pickup_is_manhattan=True
    )
    print(f"   Model prediction: {result1['original_duration']:.1f} min ({result1['original_speed']:.1f} mph)")
    print(f"   Calibrated: {result1['calibrated_duration']:.1f} min ({result1['calibrated_speed']:.1f} mph)")
    print(f"   Adjustment: {result1['adjustment']}")
    print(f"   Confidence: {result1['confidence']}")
    for reason in result1['reasons']:
        print(f"   • {reason}")
    
    # Test Case 2: Heavy rain + rush hour
    print("\n2. Heavy Rain + Rush Hour")
    print("   Original: Manhattan → Newark Airport (16.8 mi) @ 5:30 PM")
    result2 = calibrate_duration(
        predicted_duration=14.0,
        distance=16.8,
        weather_severity=2,
        is_rush_hour=True,
        is_late_night=False,
        is_raining=True,
        is_snowing=False,
        precipitation=8.5,
        snow=0.0,
        is_airport_trip=True,
        pickup_is_manhattan=True
    )
    print(f"   Model prediction: {result2['original_duration']:.1f} min ({result2['original_speed']:.1f} mph)")
    print(f"   Calibrated: {result2['calibrated_duration']:.1f} min ({result2['calibrated_speed']:.1f} mph)")
    print(f"   Adjustment: {result2['adjustment']}")
    print(f"   Confidence: {result2['confidence']}")
    for reason in result2['reasons']:
        print(f"   • {reason}")
    
    # Test Case 3: Rush hour + blizzard
    print("\n3. Rush Hour Blizzard")
    print("   Original: Times Square → Rockaway (28.5 mi) @ 5:00 PM")
    result3 = calibrate_duration(
        predicted_duration=15.1,
        distance=28.5,
        weather_severity=3,
        is_rush_hour=True,
        is_late_night=False,
        is_raining=False,
        is_snowing=True,
        precipitation=0.0,
        snow=12.0,
        is_airport_trip=False,
        pickup_is_manhattan=True
    )
    print(f"   Model prediction: {result3['original_duration']:.1f} min ({result3['original_speed']:.1f} mph)")
    print(f"   Calibrated: {result3['calibrated_duration']:.1f} min ({result3['calibrated_speed']:.1f} mph)")
    print(f"   Adjustment: {result3['adjustment']}")
    print(f"   Confidence: {result3['confidence']}")
    for reason in result3['reasons']:
        print(f"   • {reason}")
    
    print("\n" + "="*60)
    print("✅ Calibration module ready for integration!")

import joblib
import os
import numpy as np

def load_model_and_features():
    """Load the trained model and feature list."""
    model_path = os.path.join("models", "eta_model.pkl")
    features_path = os.path.join("models", "features.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError(
            "Model files not found. Please run train.py first to generate the model."
        )
    
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    
    return model, feature_columns

def predict_eta(features_dict):
    """
    Predict ETA in minutes based on input features.
    
    Args:
        features_dict (dict): Dictionary with feature names as values
                            Expected keys: hour, day_of_week, distance_km, 
                                         speed_kmh, traffic_index, stops_left
    
    Returns:
        float: Predicted ETA in minutes
    """
    # Load model and features
    model, feature_columns = load_model_and_features()
    
    # Prepare features in correct order
    features = [features_dict[col] for col in feature_columns]
    features_array = np.array([features])  # Reshape for single prediction
    
    # Make prediction
    eta_min = model.predict(features_array)[0]
    
    return float(eta_min)

# For testing purposes
if __name__ == "__main__":
    # Example usage
    test_features = {
        'hour': 8,
        'day_of_week': 1,  # Tuesday
        'distance_km': 15.0,
        'speed_kmh': 25.0,
        'traffic_index': 0.7,
        'stops_left': 3
    }
    
    try:
        eta = predict_eta(test_features)
        print(f"Predicted ETA: {eta:.2f} minutes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to generate the model.")
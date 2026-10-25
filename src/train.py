import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def load_data():
    """Load journey data from CSV."""
    data_path = os.path.join("data", "journeys.csv")
    df = pd.read_csv(data_path)
    return df

def prepare_features(df):
    """Prepare features and target."""
    feature_columns = ['hour', 'day_of_week', 'distance_km', 'speed_kmh', 'traffic_index', 'stops_left']
    X = df[feature_columns]
    y = df['eta_min']
    return X, y, feature_columns

def train_and_evaluate(X, y):
    """Train models and evaluate performance."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "predictions": y_pred
        }
        print(f"{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
    
    return results, X_test, y_test

def save_model_and_features(model, feature_columns):
    """Save the trained model and feature list."""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, "eta_model.pkl"))
    joblib.dump(feature_columns, os.path.join(model_dir, "features.pkl"))
    print(f"Model saved to {os.path.join(model_dir, 'eta_model.pkl')}")
    print(f"Features saved to {os.path.join(model_dir, 'features.pkl')}")

def main():
    """Main training function."""
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    print("\nPreparing features...")
    X, y, feature_columns = prepare_features(df)
    print(f"Features: {feature_columns}")
    
    print("\nTraining and evaluating models...")
    results, X_test, y_test = train_and_evaluate(X, y)
    
    # Select best model based on MAE
    best_model_name = min(results, key=lambda x: results[x]["MAE"])
    best_model = results[best_model_name]["model"]
    
    print(f"\nBest model: {best_model_name}")
    print(f"  MAE: {results[best_model_name]['MAE']:.4f}")
    print(f"  RMSE: {results[best_model_name]['RMSE']:.4f}")
    
    # Save the best model and feature list
    save_model_and_features(best_model, feature_columns)

if __name__ == "__main__":
    main()
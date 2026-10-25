import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from src.db import execute_query
import time

# Cache the prediction error computation so it doesn't run on every refresh
def get_prediction_error():
    """
    Load the journey data, recreate the same train/test split as in train.py,
    load the saved model, and compute MAE and RMse for the test set.
    Also compute the same for Linear Regression baseline.
    Returns a dict with the metrics.
    """
    # Load data
    data_path = os.path.join("data", "journeys.csv")
    df = pd.read_csv(data_path)
    
    # Prepare features and target (same as in train.py)
    feature_columns = ['hour', 'day_of_week', 'distance_km', 'speed_kmh', 'traffic_index', 'stops_left']
    X = df[feature_columns]
    y = df['eta_min']
    
    # Same split as in train.py: test_size=0.2, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the saved model (Gradient Boosting)
    model_path = os.path.join("models", "eta_model.pkl")
    model = joblib.load(model_path)
    
    # Predict with the saved model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Baseline: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    return {
        'gradient_boosting': {'MAE': mae, 'RMSE': rmse},
        'linear_regression': {'MAE': mae_lr, 'RMSE': rmse_lr}
    }

# We'll define a helper to log response time
def log_response_time(duration_ms):
    """Log a response time measurement to the database."""
    query = """
        INSERT INTO response_times (duration_ms)
        VALUES (?)
    """
    execute_query(query, (duration_ms,))

# Get response time statistics
def get_response_time_stats(limit=100):
    """
    Get average, min, max, and 95th percentile of the last `limit` response times.
    Also return the last `limit` measurements for a line chart.
    """
    # Get the last `limit` measurements
    query = """
        SELECT duration_ms FROM response_times
        ORDER BY timestamp DESC
        LIMIT ?
    """
    results = execute_query(query, (limit,))
    if not results:
        return {
            'avg': 0,
            'min': 0,
            'max': 0,
            'p95': 0,
            'recent': []
        }
    
    durations = [row[0] for row in results]
    avg = np.mean(durations)
    min_val = np.min(durations)
    max_val = np.max(durations)
    p95 = np.percentile(durations, 95)
    
    return {
        'avg': avg,
        'min': min_val,
        'max': max_val,
        'p95': p95,
        'recent': list(reversed(durations))  # oldest to newest for chart
    }

# Notification delivery performance
def get_delivery_stats():
    """
    From alerts_log, compute:
    - total alerts sent
    - total delivered (delivered_timestamp is not NULL)
    - delivery rate as percentage
    - average delay between sent and delivered in milliseconds
    - count per alert_type
    """
    # Total sent
    sent_query = "SELECT COUNT(*) FROM alerts_log"
    sent_result = execute_query(sent_query)
    total_sent = sent_result[0][0] if sent_result else 0
    
    # Total delivered
    delivered_query = "SELECT COUNT(*) FROM alerts_log WHERE delivered_timestamp IS NOT NULL"
    delivered_result = execute_query(delivered_query)
    total_delivered = delivered_result[0][0] if delivered_result else 0
    
    # Delivery rate
    delivery_rate = (total_delivered / total_sent * 100) if total_sent > 0 else 0
    
    # Average delay (in milliseconds) for delivered alerts
    delay_query = """
        SELECT AVG(
            (julianday(delivered_timestamp) - julianday(sent_timestamp)) * 24 * 60 * 60 * 1000
        ) FROM alerts_log
        WHERE delivered_timestamp IS NOT NULL
    """
    delay_result = execute_query(delay_query)
    avg_delay_ms = delay_result[0][0] if delay_result and delay_result[0][0] is not None else 0
    
    # Count per alert type
    type_query = """
        SELECT alert_type, COUNT(*) 
        FROM alerts_log
        GROUP BY alert_type
    """
    type_results = execute_query(type_query)
    alert_type_counts = {row[0]: row[1] for row in type_results} if type_results else {}
    
    return {
        'total_sent': total_sent,
        'total_delivered': total_delivered,
        'delivery_rate': delivery_rate,
        'avg_delay_ms': avg_delay_ms,
        'type_counts': alert_type_counts
    }

# User feedback handling
def save_feedback(rating, comment):
    """Save user feedback to the feedback table."""
    query = """
        INSERT INTO feedback (rating, comment)
        VALUES (?, ?)
    """
    execute_query(query, (rating, comment))

def get_feedback_stats():
    """
    Get average rating, number of responses, and last 5 comments.
    """
    # Average rating and count
    stats_query = """
        SELECT AVG(rating), COUNT(*) 
        FROM feedback
    """
    stats_result = execute_query(stats_query)
    if stats_result and stats_result[0][0] is not None:
        avg_rating = stats_result[0][0]
        total_responses = stats_result[0][1]
    else:
        avg_rating = 0
        total_responses = 0
    
    # Last 5 comments
    comments_query = """
        SELECT rating, comment, timestamp
        FROM feedback
        WHERE comment IS NOT NULL AND comment != ''
        ORDER BY timestamp DESC
        LIMIT 5
    """
    comments_result = execute_query(comments_query)
    recent_comments = []
    for row in comments_result:
        recent_comments.append({
            'rating': row[0],
            'comment': row[1],
            'timestamp': row[2]
        })
    
    return {
        'avg_rating': avg_rating,
        'total_responses': total_responses,
        'recent_comments': recent_comments
    }
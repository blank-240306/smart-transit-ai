import time
import os
import sys
from datetime import datetime
from src.db import execute_query, insert_alert
from src.predictor import predict_eta
from src.simulator import haversine

# Duplicate alert suppression: check if same alert type for same vehicle/stop exists within last 60 seconds
def is_duplicate_alert(vehicle_id, alert_type, stop_id=None):
    """Check if an identical alert (type, vehicle, stop) has been logged in the last 60 seconds."""
    query = """
        SELECT id FROM alerts_log
        WHERE vehicle_id = ? AND alert_type = ? 
        AND (stop_id = ? OR (stop_id IS NULL AND ? IS NULL))
        AND sent_timestamp >= datetime('now', '-60 seconds')
    """
    params = (vehicle_id, alert_type, stop_id, stop_id)
    result = execute_query(query, params)
    return len(result) > 0

def log_alert(vehicle_id, alert_type, message, stop_id=None):
    """Log an alert to the database, avoiding duplicates."""
    if is_duplicate_alert(vehicle_id, alert_type, stop_id):
        return None  # Duplicate, do not log
    
    # Insert alert with sent_timestamp as current time, delivered_timestamp as NULL
    alert_id = insert_alert(vehicle_id, alert_type, message, stop_id)
    return alert_id

def check_approaching_alerts(vehicle_data, selected_stop, threshold_minutes):
    """
    Check for approaching alerts: ETA to selected stop <= threshold.
    vehicle_data: list of dicts from get_vehicle_data (or DataFrame)
    selected_stop: name of the stop (string) or None
    threshold_minutes: float, threshold in minutes
    Returns: list of alert dicts to be logged (each dict has vehicle_id, alert_type, message, stop_id)
    """
    alerts = []
    if not selected_stop or not vehicle_data:
        return alerts
    
    # Get stop_id for the selected stop
    stop_query = """SELECT stop_id FROM stops WHERE stop_name = ? AND route_id = 1"""
    stop_result = execute_query(stop_query, (selected_stop,))
    if not stop_result:
        return alerts
    stop_id = stop_result[0][0]
    
    for v in vehicle_data:
        vehicle_id = v['vehicle_id']
        eta_min = v.get('eta_min')
        if eta_min is None:
            continue
        if eta_min <= threshold_minutes:
            message = f"Bus {vehicle_id} approaching {selected_stop} in {eta_min:.1f} minutes"
            alert_id = log_alert(vehicle_id, 'approaching', message, stop_id)
            if alert_id is not None:
                alerts.append({
                    'id': alert_id,
                    'vehicle_id': vehicle_id,
                    'alert_type': 'approaching',
                    'message': message,
                    'stop_id': stop_id
                })
    return alerts

def check_delay_alerts(vehicle_data, selected_stop, traffic_index):
    """
    Check for delay alerts: ETA with current traffic index > (ETA at traffic index 0.2 + 3 minutes).
    We assume that the ETA in vehicle_data is computed with the given traffic_index.
    We need to recompute ETA with traffic_index = 0.2 for the same vehicle (same distance, speed, etc.)
    and compare.
    """
    alerts = []
    if not selected_stop or not vehicle_data:
        return alerts
    
    # Get stop_id for the selected stop
    stop_query = """SELECT stop_id FROM stops WHERE stop_name = ? AND route_id = 1"""
    stop_result = execute_query(stop_query, (selected_stop,))
    if not stop_result:
        return alerts
    stop_id = stop_result[0][0]
    
    for v in vehicle_data:
        vehicle_id = v['vehicle_id']
        eta_current = v.get('eta_min')
        if eta_current is None:
            continue
        
        # Recompute ETA with traffic_index = 0.2
        # We need the same features as used in prediction, but with traffic_index = 0.2
        # We don't have the original features stored, so we approximate:
        # We have distance_km and speed_kmh in vehicle_data? Not exactly, we have distance_to_stop_km and speed.
        # But note: the ETA prediction uses distance_km (distance to stop?) and speed_kmh (current speed) and traffic_index.
        # Actually, in get_vehicle_data, we used:
        #   features = {
        #       'hour': datetime.now().hour,
        #       'day_of_week': datetime.now().weekday(),
        #       'distance_km': max(distance_to_stop, 0.1),
        #       'speed_kmh': max(speed, 1.0),
        #       'traffic_index': traffic_index,
        #       'stops_left': max(1, int(max(distance_to_stop, 0.1) / 0.8))
        #   }
        # So we can recompute with traffic_index=0.2.
        distance_to_stop = v.get('distance_to_stop_km')
        speed = v.get('speed')
        if distance_to_stop is None or speed is None:
            continue
        
        # Ensure positive values
        distance_km = max(distance_to_stop, 0.1)
        speed_kmh = max(speed, 1.0)
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()  # Monday=0
        stops_left = max(1, int(distance_km / 0.8))
        
        features_low_traffic = {
            'hour': hour,
            'day_of_week': day_of_week,
            'distance_km': distance_km,
            'speed_kmh': speed_kmh,
            'traffic_index': 0.2,
            'stops_left': stops_left
        }
        try:
            eta_low = predict_eta(features_low_traffic)
        except Exception:
            # Fallback: use formula
            base_time_hours = distance_km / speed_kmh
            eta_low = base_time_hours * 60 * (1 + (0.2 - 0.5))  # traffic_index 0.2 is 0.3 less than neutral 0.5? Wait, our formula in predictor uses (traffic_index - 0.5) as penalty?
            # Actually, in simulator.py we had: traffic_penalty = 1 + (traffic_index - 0.5)
            # So at traffic_index=0.2, penalty = 1 + (0.2-0.5) = 0.7
            # At traffic_index=0.5, penalty = 1.0
            # So we'll use the same formula as in simulator for consistency.
            eta_low = (distance_km / speed_kmh) * 60 * (1 + (0.2 - 0.5))
        
        # Delay condition: eta_current > eta_low + 3
        if eta_current > eta_low + 3:
            delay_min = eta_current - eta_low
            message = f"Bus {vehicle_id} delayed by {delay_min:.1f} minutes due to traffic"
            alert_id = log_alert(vehicle_id, 'delay', message, stop_id)
            if alert_id is not None:
                alerts.append({
                    'id': alert_id,
                    'vehicle_id': vehicle_id,
                    'alert_type': 'delay',
                    'message': message,
                    'stop_id': stop_id
                })
    return alerts

def check_heavy_traffic_alert(traffic_index):
    """
    Check for heavy traffic alert: traffic index > 0.8.
    Returns at most one alert (since it's global) per check, subject to duplicate suppression.
    We'll use a dummy vehicle_id and stop_id (NULL) to indicate it's not tied to a specific vehicle/stop.
    """
    if traffic_index <= 0.8:
        return []
    
    # Use NULL for vehicle_id and stop_id (allowed by foreign key as NULL)
    vehicle_id = None
    stop_id = None
    message = f"Heavy traffic detected: traffic index = {traffic_index:.2f}"
    alert_id = log_alert(vehicle_id, 'heavy_traffic', message, stop_id)
    if alert_id is not None:
        return [{
            'id': alert_id,
            'vehicle_id': vehicle_id,
            'alert_type': 'heavy_traffic',
            'message': message,
            'stop_id': stop_id
        }]
    return []

def check_route_change_alert(simulate_route_change_button_pressed):
    """
    Check for route change alert: triggered by sidebar button.
    Returns at most one alert when the button is pressed.
    """
    if not simulate_route_change_button_pressed:
        return []
    
    # We'll create a generic route change alert, not tied to a specific vehicle or stop.
    vehicle_id = None
    stop_id = None
    message = "Service alert: Route temporarily changed due to operational reasons."
    alert_id = log_alert(vehicle_id, 'route_change', message, stop_id)
    if alert_id is not None:
        return [{
            'id': alert_id,
            'vehicle_id': vehicle_id,
            'alert_type': 'route_change',
            'message': message,
            'stop_id': stop_id
        }]
    return []

def get_latest_alerts(limit=10):
    """
    Fetch the latest alerts from the database, mark them as delivered (if not already),
    and return them for display.
    """
    # First, fetch the latest alerts that are not yet delivered (or all, but we'll update delivered timestamp)
    query = """
        SELECT id, vehicle_id, alert_type, message, stop_id, sent_timestamp, delivered_timestamp
        FROM alerts_log
        ORDER BY sent_timestamp DESC
        LIMIT ?
    """
    results = execute_query(query, (limit,))
    
    alerts = []
    for row in results:
        alert_id, vehicle_id, alert_type, message, stop_id, sent_ts, delivered_ts = row
        # If delivered_timestamp is NULL, set it to now and update the database
        if delivered_ts is None:
            delivered_ts = datetime.now()
            update_query = """
                UPDATE alerts_log
                SET delivered_timestamp = ?
                WHERE id = ?
            """
            execute_query(update_query, (delivered_ts, alert_id))
        
        alerts.append({
            'id': alert_id,
            'vehicle_id': vehicle_id,
            'alert_type': alert_type,
            'message': message,
            'stop_id': stop_id,
            'sent_timestamp': sent_ts,
            'delivered_timestamp': delivered_ts
        })
    
    return alerts
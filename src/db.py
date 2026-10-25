import sqlite3
import os
from contextlib import contextmanager

DATABASE_PATH = os.path.join("data", "transit.db")

def init_db():
    """Initialize database with required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Routes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS routes (
                route_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_name TEXT NOT NULL,
                description TEXT
            )
        """)
        
        # Stops table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stops (
                stop_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id INTEGER,
                stop_name TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                stop_order INTEGER,
                FOREIGN KEY (route_id) REFERENCES routes(route_id)
            )
        """)
        
        # Vehicles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id TEXT PRIMARY KEY,
                route_id INTEGER,
                capacity INTEGER,
                FOREIGN KEY (route_id) REFERENCES routes(route_id)
            )
        """)
        
        # Vehicle positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT,
                latitude REAL,
                longitude REAL,
                speed REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
            )
        """)
        
        # Journey history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS journey_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT,
                route_id INTEGER,
                hour INTEGER,
                day_of_week INTEGER,
                distance_km REAL,
                speed_kmh REAL,
                traffic_index REAL,
                stops_left INTEGER,
                eta_min REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id),
                FOREIGN KEY (route_id) REFERENCES routes(route_id)
            )
        """)
        
        # Alerts log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id TEXT,
                alert_type TEXT,
                message TEXT,
                stop_id INTEGER,
                sent_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                delivered_timestamp DATETIME,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id),
                FOREIGN KEY (stop_id) REFERENCES stops(stop_id)
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rating INTEGER,
                comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Response times table for system performance monitoring
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration_ms REAL
            )
        """)
        
        conn.commit()

@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def execute_query(query, params=()):
    """Execute a query and return results."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        if query.strip().upper().startswith("SELECT"):
            return cursor.fetchall()
        else:
            conn.commit()
            return cursor.rowcount

def insert_journey(vehicle_id, route_id, hour, day_of_week, distance_km, speed_kmh, traffic_index, stops_left, eta_min):
    """Insert a journey record."""
    query = """
        INSERT INTO journey_history (vehicle_id, route_id, hour, day_of_week, distance_km, speed_kmh, traffic_index, stops_left, eta_min)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_query(query, (vehicle_id, route_id, hour, day_of_week, distance_km, speed_kmh, traffic_index, stops_left, eta_min))

def insert_vehicle_position(vehicle_id, latitude, longitude, speed):
    """Insert a vehicle position record."""
    query = """
        INSERT INTO vehicle_positions (vehicle_id, latitude, longitude, speed)
        VALUES (?, ?, ?, ?)
    """
    return execute_query(query, (vehicle_id, latitude, longitude, speed))

def insert_alert(vehicle_id, alert_type, message, stop_id=None):
    """Insert an alert record."""
    query = """
        INSERT INTO alerts_log (vehicle_id, alert_type, message, stop_id)
        VALUES (?, ?, ?, ?)
    """
    return execute_query(query, (vehicle_id, alert_type, message, stop_id))
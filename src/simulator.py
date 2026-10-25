import pandas as pd
import numpy as np
import sqlite3
import os
import sys
from datetime import datetime, timedelta
import random

# Add the src directory to the path so we can import db
sys.path.append(os.path.join(os.path.dirname(__file__)))
from db import init_db, insert_journey, insert_vehicle_position

def generate_lagos_route():
    """Generate a seed route with 5 stops in Lagos area."""
    # Lagos area coordinates (approximate)
    base_lat, base_lon = 6.5244, 3.3792  # Lagos mainland
    
    stops = [
        {"name": "Oshodi Terminal", "lat": 6.5642, "lon": 3.3412, "order": 1},
        {"name": "Mushin Market", "lat": 6.5487, "lon": 3.3389, "order": 2},
        {"name": "Idimu Junction", "lat": 6.5521, "lon": 3.3245, "order": 3},
        {"name": "Alimosho LGA", "lat": 6.5891, "lon": 3.2876, "order": 4},
        {"name": "Abeokuta Express", "lat": 6.6012, "lon": 3.2543, "order": 5}
    ]
    return stops

def generate_historical_journeys(num_records=6000):
    """Generate synthetic historical journey data."""
    stops = generate_lagos_route()
    route_id = 1  # Single route for simplicity
    
    # Generate data
    data = []
    
    for i in range(num_records):
        vehicle_id = f"BUS{random.randint(1, 20):03d}"
        
        # Time features
        hour = random.randint(0, 23)
        day_of_week = random.randint(0, 6)  # Monday=0, Sunday=6
        
        # Journey features
        distance_km = round(random.uniform(5.0, 25.0), 2)
        speed_kmh = round(random.uniform(15.0, 45.0), 2)
        traffic_index = round(random.uniform(0.2, 1.0), 2)
        stops_left = random.randint(0, 5)
        
        # Calculate ETA based on distance, speed, and traffic
        base_time_hours = distance_km / max(speed_kmh, 1.0)
        traffic_penalty = 1 + (traffic_index - 0.5)  # 0.5 is neutral traffic
        eta_min = round(base_time_hours * 60 * traffic_penalty, 2)
        
        data.append({
            'vehicle_id': vehicle_id,
            'route_id': route_id,
            'hour': hour,
            'day_of_week': day_of_week,
            'distance_km': distance_km,
            'speed_kmh': speed_kmh,
            'traffic_index': traffic_index,
            'stops_left': stops_left,
            'eta_min': eta_min
        })
    
    df = pd.DataFrame(data)
    return df

def save_to_csv(df, filename="journeys.csv"):
    """Save DataFrame to CSV in data directory."""
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} records to {filepath}")

def populate_database(df):
    """Insert generated journey data into database."""
    print("Populating database with journey data...")
    
    for _, row in df.iterrows():
        insert_journey(
            vehicle_id=row['vehicle_id'],
            route_id=row['route_id'],
            hour=int(row['hour']),
            day_of_week=int(row['day_of_week']),
            distance_km=float(row['distance_km']),
            speed_kmh=float(row['speed_kmh']),
            traffic_index=float(row['traffic_index']),
            stops_left=int(row['stops_left']),
            eta_min=float(row['eta_min'])
        )
    
    print(f"Inserted {len(df)} journey records into database")

def initialize_route_and_stops():
    """Initialize route and stops in database."""
    conn = sqlite3.connect(os.path.join("data", "transit.db"))
    cursor = conn.cursor()
    
    # Clear existing data for clean setup
    cursor.execute("DELETE FROM stops")
    cursor.execute("DELETE FROM routes")
    cursor.execute("DELETE FROM vehicles")
    
    # Insert route
    cursor.execute(
        "INSERT INTO routes (route_id, route_name, description) VALUES (?, ?, ?)",
        (1, "Lagos Mainland Route", "Oshodi to Abeokuta Express via Mushin, Idimu, Alimosho")
    )
    
    # Insert stops
    stops = generate_lagos_route()
    for stop in stops:
        cursor.execute(
            """INSERT INTO stops (route_id, stop_name, latitude, longitude, stop_order) 
               VALUES (?, ?, ?, ?, ?)""",
            (1, stop['name'], stop['lat'], stop['lon'], stop['order'])
        )
    
    # Insert sample vehicles
    for i in range(1, 21):  # 20 vehicles
        cursor.execute(
            "INSERT INTO vehicles (vehicle_id, route_id, capacity) VALUES (?, ?, ?)",
            (f"BUS{i:03d}", 1, 50)
        )
    
    conn.commit()
    conn.close()
    print("Initialized route, stops, and vehicles in database")

def main():
    """Main function to generate data and populate database."""
    # Initialize database
    init_db()
    
    # Initialize route, stops, and vehicles
    initialize_route_and_stops()
    
    # Generate historical journeys
    print("Generating 6000 historical journey records...")
    df = generate_historical_journeys(6000)
    
    # Save to CSV
    save_to_csv(df, "journeys.csv")
    
    # Populate database
    populate_database(df)
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()
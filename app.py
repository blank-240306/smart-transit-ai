import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import sys
import random
from datetime import datetime
import pydeck as pdk

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from simulator import get_vehicle_positions, generate_lagos_route, haversine
from db import init_db, execute_query, insert_vehicle_position
from predictor import predict_eta
from alerts import (
    check_approaching_alerts, 
    check_delay_alerts, 
    check_heavy_traffic_alert, 
    check_route_change_alert,
    get_latest_alerts
)
from metrics import (
    get_prediction_error,
    get_response_time_stats,
    log_response_time,
    get_delivery_stats,
    save_feedback,
    get_feedback_stats
)

# Initialize database and check if we need to generate data
def initialize_app():
    """Initialize the app - check if data and model exist, generate if missing."""
    db_path = os.path.join("data", "transit.db")
    model_path = os.path.join("models", "eta_model.pkl")
    
    # Check if we need to initialize
    if not os.path.exists(db_path) or not os.path.exists(model_path):
        with st.spinner("Initializing application data and models... This may take a moment."):
            # Import and run simulator to generate initial data
            from simulator import main as simulator_main
            simulator_main()
            
            # Import and run trainer to generate model
            from train import main as train_main
            train_main()
        
        st.success("Initialization complete!")
        st.rerun()

def get_route_stops():
    """Get all stops for the route."""
    try:
        results = execute_query("""SELECT stop_id, stop_name, latitude, longitude, stop_order 
                                  FROM stops WHERE route_id = 1 ORDER BY stop_order""")
        stops = []
        for row in results:
            stops.append({
                'stop_id': row[0],
                'stop_name': row[1],
                'latitude': row[2],
                'longitude': row[3],
                'stop_order': row[4]
            })
        return stops
    except Exception as e:
        st.error(f"Error loading stops: {e}")
        return []

def calculate_distance_to_stop(vehicle_lat, vehicle_lon, stop_lat, stop_lon):
    """Calculate distance from vehicle to stop in kilometers."""
    return haversine(vehicle_lat, vehicle_lon, stop_lat, stop_lon)

def get_vehicle_data(selected_stop=None, traffic_index=0.5):
    """Get current vehicle data with ETAs to selected stop."""
    # Get vehicle positions
    positions = get_vehicle_positions()
    
    if not positions:
        return pd.DataFrame()
    
    # Get selected stop coordinates
    stop_lat, stop_lon = None, None
    if selected_stop:
        stops = get_route_stops()
        for stop in stops:
            if stop['stop_name'] == selected_stop:
                stop_lat, stop_lon = stop['latitude'], stop['longitude']
                break
    
    # Prepare vehicle data
    vehicle_data = []
    for pos in positions:
        vehicle_id = pos['vehicle_id']
        lat = pos['latitude']
        lon = pos['longitude']
        speed = pos['speed']
        
        # Calculate distance to selected stop if available
        distance_to_stop = None
        eta_min = None
        
        if stop_lat is not None and stop_lon is not None:
            distance_to_stop = calculate_distance_to_stop(lat, lon, stop_lat, stop_lon)
            
            # Predict ETA using the model
            try:
                # Prepare features for prediction
                # Note: In a real system, we'd get these from historical patterns or sensors
                # For now, we'll estimate based on current conditions
                features = {
                    'hour': datetime.now().hour,
                    'day_of_week': datetime.now().weekday(),  # Monday=0
                    'distance_km': max(distance_to_stop, 0.1),  # Ensure positive distance
                    'speed_kmh': max(speed, 1.0),  # Ensure positive speed
                    'traffic_index': traffic_index,
                    'stops_left': max(1, int(max(distance_to_stop, 0.1) / 0.8))  # Same as training data
                }
                eta_min = predict_eta(features)
            except Exception as e:
                # Fallback calculation if prediction fails
                if distance_to_stop is not None and speed > 0:
                    eta_min = (distance_to_stop / max(speed, 1.0)) * 60 * (1 + (traffic_index - 0.5))
                else:
                    eta_min = 0
        
        vehicle_data.append({
            'vehicle_id': vehicle_id,
            'latitude': lat,
            'longitude': lon,
            'speed': round(speed, 1),
            'distance_to_stop_km': round(distance_to_stop, 2) if distance_to_stop else None,
            'eta_min': round(eta_min, 1) if eta_min else None,
            'timestamp': pos['timestamp']
        })
        
        # Log EVERY position update to database (removed random sampling)
        try:
            insert_vehicle_position(vehicle_id, lat, lon, speed)
        except:
            pass  # Don't crash the app if DB logging fails
    
    return pd.DataFrame(vehicle_data)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Smart Transit AI",
        page_icon=None,
        layout="wide"
    )
    
    st.title("Smart Public Transport Route Tracking System")
    st.markdown("### Real-time bus tracking with AI-powered ETA predictions")
    
    # Initialize app if needed
    initialize_app()
    
    # Initialize session state for simulation control and route change flag
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = True
    if 'route_change_triggered' not in st.session_state:
        st.session_state.route_change_triggered = False
    if 'update_interval' not in st.session_state:
        st.session_state.update_interval = 3
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Stop selector
    stops = get_route_stops()
    stop_names = [stop['stop_name'] for stop in stops] if stops else []
    selected_stop = st.sidebar.selectbox(
        "Select Stop for ETA:", 
        options=[""] + stop_names if stop_names else ["No stops available"],
        index=0
    )
    
    if selected_stop == "":
        selected_stop = None
    
    # Traffic index slider
    traffic_index = st.sidebar.slider(
        "Traffic Index (0.0 = light, 1.0 = heavy)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Adjust to simulate traffic conditions affecting ETAs"
    )
    
    # Alert Settings
    st.sidebar.header("Alert Settings")
    alert_threshold = st.sidebar.slider("Alert me when bus is within (min)", 1, 15, 5, help="Trigger an alert when ETA to selected stop is at or below this threshold")
    # Route change button
    if st.sidebar.button("Simulate route change", help="Generate a route change alert for testing"):
        st.session_state.route_change_triggered = True
    
    # Simulation controls
    st.sidebar.header("Simulation")
    # Update the session state based on the checkbox
    simulation_running = st.sidebar.checkbox("Run Live Simulation", value=st.session_state.simulation_running)
    st.session_state.simulation_running = simulation_running
    update_interval = st.sidebar.slider(
        "Update Interval (seconds)", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="How often to update vehicle positions"
    )
    st.session_state.update_interval = update_interval
    
    # Create tabs
    tab_dashboard, tab_evaluation = st.tabs(["Dashboard", "Evaluation"])
    
    # Dashboard tab content
    with tab_dashboard:
        # Define the live dashboard fragment
        @st.fragment(run_every=st.session_state.update_interval if st.session_state.simulation_running else None)
        def show_live_dashboard():
            # Measure response time for this fragment run
            start_time = time.perf_counter()
            
            # Prepare vehicle list for alert checking (list of dicts)
            vehicle_df = get_vehicle_data(selected_stop, traffic_index)
            vehicle_list = vehicle_df.to_dict('records') if not vehicle_df.empty else []
            
            # Check for new alerts and log them (functions will handle duplicate suppression)
            new_alerts = []
            if vehicle_list:  # Only check if we have vehicle data
                new_alerts.extend(check_approaching_alerts(vehicle_list, selected_stop, alert_threshold))
                new_alerts.extend(check_delay_alerts(vehicle_list, selected_stop, traffic_index))
                new_alerts.extend(check_heavy_traffic_alert(traffic_index))
                # Route change alert: check flag and consume it
                if st.session_state.route_change_triggered:
                    new_alerts.extend(check_route_change_alert(True))
                    # Clear the flag after using it
                    st.session_state.route_change_triggered = False
            
            # End timing and log response time
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000  # convert to milliseconds
            log_response_time(duration_ms)
            
            # Fetch latest alerts for display (this also marks them as delivered)
            latest_alerts = get_latest_alerts(limit=10)
            
            # Layout with columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Live Vehicle Map")
                if not vehicle_df.empty:
                    # Create pydeck layers for the map
                    def create_map_layer(vehicle_df_local, selected_stop_local=None):
                        if vehicle_df_local.empty:
                            return []
                        # Vehicle layer
                        vehicle_layer = pdk.Layer(
                            "ScatterplotLayer",
                            vehicle_df_local,
                            get_position=["longitude", "latitude"],
                            get_color="[200, 30, 0, 160]",  # Reddish color for buses
                            get_radius="speed",  # Size based on speed
                            radius_scale=2,
                            radius_min_pixels=5,
                            radius_max_pixels=30,
                            pickable=True,
                        )
                        
                        layers = [vehicle_layer]
                        
                        # Add selected stop marker if available
                        if selected_stop_local:
                            stops_local = get_route_stops()
                            for stop in stops_local:
                                if stop['stop_name'] == selected_stop_local:
                                    stop_layer = pdk.Layer(
                                        "ScatterplotLayer",
                                        pd.DataFrame([{
                                            'longitude': stop['longitude'],
                                            'latitude': stop['latitude'],
                                            'name': stop['stop_name']
                                        }]),
                                        get_position=["longitude", "latitude"],
                                        get_color="[0, 150, 0, 200]",  # Green for stops
                                        get_radius=100,
                                        radius_scale=1,
                                        pickable=True,
                                    )
                                    layers.append(stop_layer)
                                    break
                        return layers
                    
                    layers = create_map_layer(vehicle_df, selected_stop)
                    
                    if layers:
                        # Set initial view to Lagos area
                        view_state = pdk.ViewState(
                            latitude=6.55,
                            longitude=3.35,
                            zoom=12,
                            pitch=0
                        )
                        
                        deck = pdk.Deck(
                            layers=layers,
                            initial_view_state=view_state,
                            tooltip={
                                "html": "<b>Vehicle ID:</b> {vehicle_id}<br/>"
                                       "<b>Speed:</b> {speed} km/h<br/>"
                                       "<b>ETA:</b> {eta_min} min<br/>"
                                       "<b>Distance to stop:</b> {distance_to_stop_km} km",
                                "style": {"backgroundColor": "steelblue", "color": "white"}
                            }
                        )
                        
                        st.pydeck_chart(deck)
                    else:
                        st.info("No vehicle data available for map")
                else:
                    st.info("No vehicle data available")
            
            with col2:
                st.subheader("Vehicle Information")
                if not vehicle_df.empty:
                    # Prepare display dataframe
                    display_df = vehicle_df.copy()
                    if selected_stop:
                        display_cols = ['vehicle_id', 'speed', 'distance_to_stop_km', 'eta_min']
                        display_df = display_df[display_cols].rename(columns={
                            'vehicle_id': 'Vehicle ID',
                            'speed': 'Speed (km/h)',
                            'distance_to_stop_km': f'Dist to {selected_stop} (km)',
                            'eta_min': 'ETA (min)'
                        })
                    else:
                        display_cols = ['vehicle_id', 'speed']
                        display_df = display_df[display_cols].rename(columns={
                            'vehicle_id': 'Vehicle ID',
                            'speed': 'Speed (km/h)'
                        })
                    
                    st.dataframe(display_df, hide_index=True)
                    
                    # Show ETA summary
                    if selected_stop and 'eta_min' in vehicle_df.columns:
                        valid_etas = vehicle_df['eta_min'].dropna()
                        if len(valid_etas) > 0:
                            avg_eta = valid_etas.mean()
                            min_eta = valid_etas.min()
                            max_eta = valid_etas.max()
                            
                            eta_info = f"""
                            **To {selected_stop}:**
                            - Average ETA: {avg_eta:.1f} minutes
                            - Fastest: {min_eta:.1f} minutes  
                            - Slowest: {max_eta:.1f} minutes
                            - Vehicles reporting: {len(valid_etas)}/{len(vehicle_df)}
                            """
                            st.info(eta_info)
                        else:
                            st.warning("No ETA data available")
                    else:
                        st.info("Select a stop to see ETA predictions")
                else:
                    st.info("No vehicle data available")
                
                st.subheader("Alerts")
                if latest_alerts:
                    # Define alert type styles: (border_color, background_color)
                    alert_styles = {
                        'approaching': ("#ff9800", "rgba(255,152,0,0.12)"),   # orange
                        'delay': ("#f44336", "rgba(244,67,54,0.12)"),        # red
                        'heavy_traffic': ("#d32f2f", "rgba(211,47,47,0.12)"), # dark red
                        'route_change': ("#2196f3", "rgba(33,150,243,0.12)"), # blue
                    }
                    
                    for alert in latest_alerts:
                        alert_type = alert['alert_type']
                        icon = "[APPROACHING]" if alert_type == 'approaching' else \
                               "[DELAY]" if alert_type == 'delay' else \
                               "[TRAFFIC]" if alert_type == 'heavy_traffic' else \
                               "[ROUTE]" if alert_type == 'route_change' else ""
                        
                        # Get style for this alert type, with safe defaults
                        border_color, bg_color = alert_styles.get(alert_type, ("#9e9e9e", "rgba(158,158,158,0.12)"))  # gray default
                        
                        # Format timestamp for display
                        sent_time = alert['sent_timestamp']
                        if isinstance(sent_time, str):
                            # If it's a string from database, try to parse
                            try:
                                sent_time = datetime.strptime(sent_time, '%Y-%m-%d %H:%M:%S')
                            except:
                                sent_time = datetime.now()  # fallback
                        # Format as HH:MM:SS
                        time_str = sent_time.strftime('%H:%M:%S')
                        
                        # Display alert with colored background
                        st.markdown(
                            f"""
                            <div style="
                                border-left: 4px solid {border_color}; 
                                padding: 0.5rem 1rem; 
                                margin: 0.5rem 0;
                                background-color: {bg_color};">
                                <span style="font-size: 1.2em;">{icon}</span> 
                                <strong>{alert_type.upper()}</strong> [{time_str}]<br/>
                                {alert['message']}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No alerts yet")
                
                # Status indicator
                status_text = f"Last updated: {datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')} | Simulation: {'Running' if st.session_state.simulation_running else 'Stopped'} | Alerts logged: {len(new_alerts)}"
                st.caption(status_text)
        
        # Call the fragment to render the live dashboard
        show_live_dashboard()
    
    # Evaluation tab content
    with tab_evaluation:
        st.header("Evaluation Metrics")
        
        # 1. Prediction error
        st.subheader("1. Prediction Error (Test Set)")
        with st.spinner("Loading prediction error..."):
            # Use cached function to avoid recomputation on every refresh
            @st.cache_data
            def get_cached_prediction_error():
                return get_prediction_error()
            
            error_data = get_cached_prediction_error()
        
        # Create a table for the metrics
        error_df = pd.DataFrame({
            'Model': ['Gradient Boosting', 'Linear Regression'],
            'MAE': [error_data['gradient_boosting']['MAE'], error_data['linear_regression']['MAE']],
            'RMSE': [error_data['gradient_boosting']['RMSE'], error_data['linear_regression']['RMSE']]
        })
        st.table(error_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}"}))
        
        # 2. System response time
        st.subheader("2. System Response Time")
        response_stats = get_response_time_stats(limit=100)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average (ms)", f"{response_stats['avg']:.2f}")
        col2.metric("Minimum (ms)", f"{response_stats['min']:.2f}")
        col3.metric("Maximum (ms)", f"{response_stats['max']:.2f}")
        col4.metric("95th Percentile (ms)", f"{response_stats['p95']:.2f}")
        
        # Line chart of the last 100 response times
        if response_stats['recent']:
            chart_data = pd.DataFrame({
                'Response Time (ms)': response_stats['recent']
            })
            st.line_chart(chart_data)
        else:
            st.info("No response time data yet")
        
        # 3. Notification delivery performance
        st.subheader("3. Notification Delivery Performance")
        delivery_stats = get_delivery_stats()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sent", delivery_stats['total_sent'])
        col2.metric("Total Delivered", delivery_stats['total_delivered'])
        col3.metric("Delivery Rate (%)", f"{delivery_stats['delivery_rate']:.2f}")
        
        st.metric("Average Delay (ms)", f"{delivery_stats['avg_delay_ms']:.2f}")
        
        # Alert type counts
        if delivery_stats['type_counts']:
            type_df = pd.DataFrame(list(delivery_stats['type_counts'].items()), columns=['Alert Type', 'Count'])
            st.table(type_df)
        else:
            st.info("No alert data yet")
        
        # 4. User feedback
        st.subheader("4. User Feedback")
        feedback_stats = get_feedback_stats()
        
        col1, col2 = st.columns(2)
        col1.metric("Average Rating", f"{feedback_stats['avg_rating']:.2f}" if feedback_stats['total_responses'] > 0 else "N/A")
        col2.metric("Number of Responses", feedback_stats['total_responses'])
        
        # Feedback form
        with st.form(key='feedback_form'):
            rating = st.slider("Rating (1 = Poor, 5 = Excellent)", 1, 5, 3)
            comment = st.text_area("Optional Comment", placeholder="What do you think of the system?")
            submit_button = st.form_submit_button(label="Submit Feedback")
            
            if submit_button:
                save_feedback(rating, comment)
                st.success("Thank you for your feedback!")
                # Rerun to show the updated feedback stats
                st.rerun()
        
        # Last 5 comments
        if feedback_stats['recent_comments']:
            st.write("**Recent Comments:**")
            for fb in feedback_stats['recent_comments']:
                st.markdown(f"*Rating: {fb['rating']}/5* - {fb['comment']} (*{fb['timestamp']}*)")
        else:
            st.info("No feedback yet")

if __name__ == "__main__":
    main()

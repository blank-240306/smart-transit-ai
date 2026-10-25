# Smart Public Transport Route Tracking and Passenger Alert System

A project titled "Design and Implementation of a Smart Public Transport Route Tracking and Passenger Alert System Using AI and IoT".

The system tracks buses, predicts their estimated time of arrival (ETA) with machine learning, shows everything on a live dashboard, and sends passenger alerts.

## Important note
No real GPS hardware is used. All vehicle data comes from a simulator, and the ETA model is trained on simulated journeys. The reported error figures therefore show that the pipeline works, not real-world accuracy. The vehicle data source is isolated in `get_vehicle_positions()` in `src/simulator.py`, so it can be replaced with real GPS or phone location data without rewriting the rest of the app.

## Features
- Live map of simulated buses moving along a Lagos-area route with 5 stops
- ETA prediction using Gradient Boosting, compared against a Linear Regression baseline
- Passenger alerts: approaching bus, delay, heavy traffic, route change
- Every alert is logged with sent and delivered timestamps
- Evaluation tab: prediction error (MAE and RMSE), system response time, notification delivery rate, and a user feedback form

## Tech stack
Python, Streamlit, Pandas, NumPy, Scikit-learn, SQLite, Joblib, PyDeck

## Project structure
- app.py : Streamlit dashboard
- src/db.py : database schema and helpers
- src/simulator.py : data and live vehicle simulator
- src/train.py : model training and evaluation
- src/predictor.py : ETA prediction
- src/alerts.py : alert rules and logging
- src/metrics.py : evaluation metrics
- data/ : SQLite database and journey data
- models/ : trained model files

## Run locally
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/simulator.py
python src/train.py
streamlit run app.py

## Author
Oniovosa David Runor Tonbara, Babcock University, Software Engineering
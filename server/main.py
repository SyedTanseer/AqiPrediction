"""
AQI Edge System - FastAPI Backend Server
Handles TFLite model inference, WebSocket real-time updates, and system state management.
Uses a Dense model trained on Mendeley Indian Cities AQI dataset.
Input: 6 pollutant features (PM2.5, PM10, NO2, SO2, CO, O3)
Output: Direct AQI prediction
"""
import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import yaml
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "dense_model.tflite")
NORM_PATH = os.path.join(BASE_DIR, "config", "normalization.yaml")

# ============================================================
# AQI Categories (for display only — AQI is predicted directly)
# ============================================================
AQI_CATEGORIES = [
    (0, 120, "Good", "#22c55e"),
    (121, 250, "Moderate", "#eab308"),
    (251, 350, "Unhealthy", "#ef4444"),
    (351, 400, "Very Unhealthy", "#a855f7"),
    (401, 500, "Hazardous", "#991b1b"),
]


def get_aqi_category(aqi_value):
    """Get category and color for an AQI value"""
    for lo, hi, cat, color in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return cat, color
    if aqi_value > 500:
        return "Hazardous", "#991b1b"
    return "Good", "#22c55e"


# ============================================================
# Model Service
# ============================================================
class ModelService:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.norm_stats = None
        self.loaded = False
        self.load_model()

    def load_model(self):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            with open(NORM_PATH, 'r') as f:
                self.norm_stats = yaml.safe_load(f)

            self.loaded = True
            print(f"✓ Model loaded: {MODEL_PATH}")
            print(f"  Input shape: {self.input_details[0]['shape']}")
            print(f"  Output shape: {self.output_details[0]['shape']}")
            print(f"✓ Normalization stats loaded: {NORM_PATH}")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            self.loaded = False

    def normalize(self, pm25, pm10, no2, so2, co, o3):
        """Normalize raw sensor values using train-only z-score stats (6 features)"""
        return [
            (pm25 - self.norm_stats['PM25']['mean']) / self.norm_stats['PM25']['std'],
            (pm10 - self.norm_stats['PM10']['mean']) / self.norm_stats['PM10']['std'],
            (no2 - self.norm_stats['NO2']['mean']) / self.norm_stats['NO2']['std'],
            (so2 - self.norm_stats['SO2']['mean']) / self.norm_stats['SO2']['std'],
            (co - self.norm_stats['CO']['mean']) / self.norm_stats['CO']['std'],
            (o3 - self.norm_stats['O3']['mean']) / self.norm_stats['O3']['std'],
        ]

    def denormalize_aqi(self, normalized_aqi):
        """Convert normalized AQI prediction back to actual AQI scale"""
        return normalized_aqi * self.norm_stats['AQI']['std'] + self.norm_stats['AQI']['mean']

    def predict(self, normalized_features: np.ndarray):
        """Run inference on a 1x6 normalized feature vector"""
        if not self.loaded:
            return None
        input_data = normalized_features.reshape(1, 6).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return float(output[0][0])


# ============================================================
# System State
# ============================================================
class SystemState:
    def __init__(self):
        self.start_time = time.time()
        self.predictions_count = 0
        self.fan_on = False
        self.fan_speed = 0
        self.current_aqi = 0
        self.current_category = "N/A"
        self.current_color = "#666"
        self.predicted_aqi = 0.0
        self.sensor_readings = {"PM25": 0, "PM10": 0, "NO2": 0, "SO2": 0, "CO": 0, "O3": 0}
        self.prediction_history: List[Dict] = []
        self.last_update = None

    def uptime_hours(self):
        return round((time.time() - self.start_time) / 3600, 1)

    def get_status(self):
        return {
            "system_online": True,
            "model_loaded": model_service.loaded,
            "uptime_hours": self.uptime_hours(),
            "predictions_count": self.predictions_count,
            "fan_on": self.fan_on,
            "fan_speed": self.fan_speed,
            "current_aqi": self.current_aqi,
            "current_category": self.current_category,
            "current_color": self.current_color,
            "predicted_aqi": round(self.predicted_aqi, 1),
            "sensor_readings": self.sensor_readings,
            "last_update": self.last_update,
            "prediction_history": self.prediction_history[-50:],
        }


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AQI Edge System API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
state = SystemState()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(data)
            except Exception:
                self.active_connections.remove(connection)

manager = ConnectionManager()


# ============================================================
# Request/Response Models
# ============================================================
class SensorReading(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    co: float
    o3: float

class FanControl(BaseModel):
    on: bool
    speed: Optional[int] = None


# ============================================================
# REST Endpoints
# ============================================================
@app.get("/api/status")
async def get_status():
    return state.get_status()


@app.post("/api/sensor-data")
async def receive_sensor_data(reading: SensorReading):
    """Receive a sensor reading, run model prediction for AQI"""
    timestamp = datetime.now().isoformat()

    # Store raw readings
    state.sensor_readings = {
        "PM25": round(reading.pm25, 1),
        "PM10": round(reading.pm10, 1),
        "NO2": round(reading.no2, 1),
        "SO2": round(reading.so2, 1),
        "CO": round(reading.co, 2),
        "O3": round(reading.o3, 1),
    }
    state.last_update = timestamp

    # ---- MODEL PREDICTION (direct AQI) ----
    normalized = model_service.normalize(
        reading.pm25, reading.pm10, reading.no2,
        reading.so2, reading.co, reading.o3
    )

    features = np.array(normalized)
    normalized_prediction = model_service.predict(features)

    result = {
        "timestamp": timestamp,
        "sensors": state.sensor_readings,
    }

    if normalized_prediction is not None:
        predicted_aqi = model_service.denormalize_aqi(normalized_prediction)
        predicted_aqi = max(0, round(predicted_aqi))  # AQI cannot be negative
        
        category, color = get_aqi_category(predicted_aqi)
        
        state.current_aqi = predicted_aqi
        state.current_category = category
        state.current_color = color
        state.predicted_aqi = predicted_aqi
        state.predictions_count += 1

        # Auto fan control based on AQI — proportional scaling
        if predicted_aqi > 75:
            state.fan_on = True
            # Scale fan speed: 30% at AQI 75 → 100% at AQI 300+
            speed_pct = 30 + ((predicted_aqi - 75) / 225) * 70
            state.fan_speed = min(100, max(30, int(speed_pct)))
        elif predicted_aqi < 50:
            state.fan_on = False
            state.fan_speed = 0
        # AQI 50-75: hysteresis band — maintain current state to avoid toggling

        history_entry = {
            "timestamp": timestamp,
            "aqi": predicted_aqi,
            "category": category,
            "color": color,
            "sensors": state.sensor_readings,
        }
        state.prediction_history.append(history_entry)
        if len(state.prediction_history) > 200:
            state.prediction_history = state.prediction_history[-200:]

        result["prediction"] = history_entry

    # Broadcast
    await manager.broadcast({"type": "update", "data": state.get_status()})
    return result


@app.post("/api/fan/toggle")
async def toggle_fan(control: FanControl):
    state.fan_on = control.on
    if control.speed is not None:
        state.fan_speed = control.speed
    elif not control.on:
        state.fan_speed = 0
    await manager.broadcast({"type": "update", "data": state.get_status()})
    return {"fan_on": state.fan_on, "fan_speed": state.fan_speed}


@app.post("/api/reset")
async def reset_system():
    state.prediction_history.clear()
    state.predictions_count = 0
    state.current_aqi = 0
    state.current_category = "N/A"
    state.current_color = "#666"
    state.predicted_aqi = 0.0
    state.fan_on = False
    state.fan_speed = 0
    state.last_update = None
    await manager.broadcast({"type": "update", "data": state.get_status()})
    return {"message": "System reset"}


@app.get("/api/model-info")
async def get_model_info():
    model_size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    return {
        "model_name": "Dense AQI Predictor (6→32→16→8→1)",
        "model_file": "dense_model.tflite",
        "model_size_bytes": model_size,
        "model_size_kb": round(model_size / 1024, 1),
        "input_shape": [1, 6],
        "output_shape": [1, 1],
        "input_features": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
        "target_variable": "AQI (direct prediction)",
        "dataset": "Mendeley Indian Cities AQI",
        "normalization": model_service.norm_stats,
        "loaded": model_service.loaded,
    }


# ============================================================
# WebSocket
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await websocket.send_json({"type": "init", "data": state.get_status()})
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

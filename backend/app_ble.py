from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
from bleak import BleakClient, BleakScanner
import threading
import time
import logging
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
PICO_NAME = "ASLG"
UART_RX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
EXPECTED_COLUMNS = 14
MODEL_PATH = "../models/asl_model.joblib"
SCALER_PATH = "../models/scaler.joblib"
ENCODER_PATH = "../models/label_encoder.joblib"
COLLECTION_WINDOW_SECONDS = 1.0

# Global state
ble_client = None
sensor_buffer = []
sensor_buffer_lock = threading.Lock()

# Global model objects
model = None
scaler = None
label_encoder = None

# Pico warning prefixes to ignore
PICO_WARN_PREFIXES = ("FLEX_ERR", "IMU_ERR", "DATA_ERR", "SKIP",
                      "FLEX_UNEXPECTED", "IMU_UNEXPECTED", "theres an issue",
                      "Start Reading", "Calibration required")


def parse_sensor_data(line):
    """Parse CSV line into dictionary of 14 sensor values"""
    try:
        parts = line.strip().split(",")
        if len(parts) != EXPECTED_COLUMNS:
            return None

        return {
            "thumb_bent":   float(parts[0]),
            "index_bent":   float(parts[1]),
            "middle_bent":  float(parts[2]),
            "ring_bent":    float(parts[3]),
            "pinky_bent":   float(parts[4]),
            "accel_x":      float(parts[5]),
            "accel_y":      float(parts[6]),
            "accel_z":      float(parts[7]),
            "heading":      float(parts[8]),
            "rolling":      float(parts[9]),
            "pitch":        float(parts[10]),
            "thumb_touch":  float(parts[11]),
            "index_touch":  float(parts[12]),
            "middle_touch": float(parts[13])
        }
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse line: {e}")
        return None


def handle_ble_data(sender, raw: bytearray):
    """Callback fired on each BLE notification from the Pico"""
    try:
        line = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return

    if any(line.startswith(p) for p in PICO_WARN_PREFIXES):
        return

    sensor_data = parse_sensor_data(line)
    if sensor_data:
        with sensor_buffer_lock:
            sensor_buffer.append({
                "data": sensor_data,
                "timestamp": datetime.now()
            })


async def ble_connect_and_stream():
    """Scan for Pico, connect, and subscribe to notifications"""
    global ble_client

    while True:
        try:
            logger.info(f"Scanning for '{PICO_NAME}'...")
            device = await BleakScanner.find_device_by_name(PICO_NAME, timeout=10)
            if not device:
                logger.warning("Pico not found, retrying in 5s...")
                await asyncio.sleep(5)
                continue

            async with BleakClient(device) as client:
                ble_client = client
                logger.info(f"✅ Connected to {PICO_NAME} via BLE")

                # await client.request_mtu(128)
                await client.start_notify(UART_RX_CHAR_UUID, handle_ble_data)

                while client.is_connected:
                    await asyncio.sleep(0.5)

            logger.warning("BLE disconnected, reconnecting...")
            ble_client = None

        except Exception as e:
            logger.error(f"BLE error: {e}")
            ble_client = None
            await asyncio.sleep(5)


def start_ble_thread():
    """Run the async BLE loop in a background thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_connect_and_stream())


def average_readings(recent):
    """Average a list of sensor reading dicts"""
    ACCEL_KEYS = {"accel_x", "accel_y", "accel_z"}
    TOUCH_KEYS = {"thumb_touch", "index_touch", "middle_touch"}

    averaged = {}
    for key in recent[0].keys():
        values = [r[key] for r in recent]
        if key in ACCEL_KEYS:
            averaged[key] = max(values, key=abs)
        else:
            averaged[key] = sum(values) / len(values)
            if key in TOUCH_KEYS:
                averaged[key] = 0 if averaged[key] < 0.5 else 1

    return averaged


def load_model():
    global model, scaler, label_encoder
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Using mock model.")
            return False

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            return False

        if os.path.exists(ENCODER_PATH):
            label_encoder = joblib.load(ENCODER_PATH)
            logger.info(f"Classes: {label_encoder.classes_}")
        else:
            return False

        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def predict_character(sensor_data):
    global model, scaler, label_encoder

    if model is None or scaler is None or label_encoder is None:
        return mock_prediction(sensor_data)

    try:
        features = np.array([[
            sensor_data['thumb_bent'],  sensor_data['index_bent'],
            sensor_data['middle_bent'], sensor_data['ring_bent'],
            sensor_data['pinky_bent'],  sensor_data['accel_x'],
            sensor_data['accel_y'],     sensor_data['accel_z'],
            sensor_data['heading'],     sensor_data['rolling'],
            sensor_data['pitch'],       sensor_data['thumb_touch'],
            sensor_data['index_touch'], sensor_data['middle_touch']
        ]])

        features_scaled = scaler.transform(features)
        predicted_class_index = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)
        confidence = float(np.max(probabilities[0]))
        predicted_character = label_encoder.inverse_transform([predicted_class_index])[0]

        logger.info(f"Predicted: {predicted_character} (confidence: {confidence:.3f})")
        return predicted_character, confidence

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return mock_prediction(sensor_data)


def mock_prediction(sensor_data):
    import random
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    thumb_bend = sensor_data['thumb_bent']
    if thumb_bend > 90:
        predicted = random.choice(['A', 'B', 'C', 'D', 'E'])
    elif thumb_bend < 50:
        predicted = random.choice(['T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    else:
        predicted = random.choice(list(characters))
    return predicted, random.uniform(0.6, 0.9)


# API Endpoints
@app.route('/api/detect', methods=['POST'])
def detect_character():
    try:
        data = request.get_json()
        expected_character = data.get('expected_character') if data else None

        with sensor_buffer_lock:
            sensor_buffer.clear()

        time.sleep(COLLECTION_WINDOW_SECONDS)

        with sensor_buffer_lock:
            recent = [entry["data"] for entry in sensor_buffer]
            sensor_buffer.clear()

        logger.info(f"Readings collected in {COLLECTION_WINDOW_SECONDS}s: {len(recent)}")

        if not recent:
            return jsonify({
                "error": "No sensor data available. Is the Pico connected?",
                "detected_character": None,
                "success": False
            }), 503

        averaged_data = average_readings(recent)
        predicted_character, confidence = predict_character(averaged_data)

        logger.info(f"Predicted: {predicted_character} | Expected: {expected_character} | Confidence: {confidence:.3f}")

        return jsonify({
            "detected_character": predicted_character,
            "confidence": confidence,
            "success": True
        })

    except Exception as e:
        logger.error(f"Error in detect_character: {e}")
        return jsonify({"error": str(e), "detected_character": None, "success": False}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    with sensor_buffer_lock:
        buf_size = len(sensor_buffer)
        last_timestamp = sensor_buffer[-1]["timestamp"] if sensor_buffer else None

    return jsonify({
        "status": "ok",
        "pico_connected": ble_client is not None and ble_client.is_connected,
        "has_sensor_data": buf_size > 0,
        "buffer_size": buf_size,
        "model_loaded": model is not None,
        "last_data_age_ms": int((datetime.now() - last_timestamp).total_seconds() * 1000) if last_timestamp else None
    })


@app.route('/api/sensor/latest', methods=['GET'])
def get_latest_sensor():
    with sensor_buffer_lock:
        if not sensor_buffer:
            return jsonify({"error": "No sensor data"}), 404
        latest = sensor_buffer[-1]

    return jsonify({
        "sensor_data": latest["data"],
        "timestamp": latest["timestamp"].isoformat()
    })


def start_backend():
    logger.info("Starting ASL Detection Backend...")

    if load_model():
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ Running in mock mode (no model loaded)")

    ble_thread = threading.Thread(target=start_ble_thread, daemon=True)
    ble_thread.start()
    logger.info("BLE background thread started, scanning for Pico...")

    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)


if __name__ == '__main__':
    start_backend()
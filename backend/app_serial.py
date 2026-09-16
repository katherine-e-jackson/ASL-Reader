from flask import Flask, request, jsonify
from flask_cors import CORS
import serial
import serial.tools.list_ports
import threading
import time
import logging
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
PICO_NAME = "ASLG"
CORS(app)  # Enable CORS for frontend

# Configuration
BAUD_RATE = 115200
EXPECTED_COLUMNS = 14
MODEL_PATH = "../models/asl_model.joblib"
SCALER_PATH = "../models/scaler.joblib"
ENCODER_PATH = "../models/label_encoder.joblib"

# Global sensor buffer
serial_connection = None
sensor_buffer = []
sensor_buffer_lock = threading.Lock()
COLLECTION_WINDOW_SECONDS = 1.0

# Global model objects
model = None
scaler = None
label_encoder = None

# Pico warning prefixes to ignore
PICO_WARN_PREFIXES = ("FLEX_ERR", "IMU_ERR", "DATA_ERR", "SKIP",
                      "FLEX_UNEXPECTED", "IMU_UNEXPECTED", "theres an issue",
                      "Start Reading", "Calibration required")


def find_pico_port():
    """Auto-detect Pico serial port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "usbmodem" in port.device or "Pico" in port.description or "USB Serial" in port.description:
            logger.info(f"Found Pico on port: {port.device}")
            return port.device
    return None


def connect_to_pico():
    """Establish serial connection to Pico"""
    global serial_connection

    port = find_pico_port()
    if not port:
        logger.error("Could not find Pico. Is it connected?")
        return False

    try:
        serial_connection = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow Pico to reset
        serial_connection.reset_input_buffer()
        logger.info(f"Connected to Pico on {port}")
        return True
    except serial.SerialException as e:
        logger.error(f"Failed to connect to Pico: {e}")
        return False


def parse_sensor_data(line):
    """Parse CSV line into dictionary of 14 sensor values"""
    try:
        parts = line.strip().split(",")
        if len(parts) != EXPECTED_COLUMNS:
            return None

        return {
            "thumb_bent":    float(parts[0]),
            "index_bent":    float(parts[1]),
            "middle_bent":   float(parts[2]),
            "ring_bent":     float(parts[3]),
            "pinky_bent":    float(parts[4]),
            "accel_x":       float(parts[5]),
            "accel_y":       float(parts[6]),
            "accel_z":       float(parts[7]),
            "heading":       float(parts[8]),
            "rolling":       float(parts[9]),
            "pitch":         float(parts[10]),
            "thumb_touch":   float(parts[11]),
            "index_touch":   float(parts[12]),
            "middle_touch":  float(parts[13])
        }

    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse line: {e}")
        return None


def read_pico_stream():
    global sensor_buffer, serial_connection

    count = 0  # fixed: moved outside lock so it's accessible

    while True:
        if not serial_connection or not serial_connection.is_open:
            logger.warning("Serial connection lost, attempting to reconnect...")
            if connect_to_pico():
                continue
            else:
                time.sleep(5)
                continue

        try:
            raw_line = serial_connection.readline()
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                logger.warning(f"Could not decode bytes: {raw_line}")
                continue

            # Filter noisy Pico lines BEFORE logging
            if any(line.startswith(prefix) for prefix in PICO_WARN_PREFIXES):
                continue

            sensor_data = parse_sensor_data(line)
            if sensor_data:
                with sensor_buffer_lock:
                    sensor_buffer.append({
                        "data": sensor_data,
                        "timestamp": datetime.now()
                    })
                count += 1
                if count % 10 == 0:
                    with sensor_buffer_lock:
                        logger.info(f"Buffer size: {len(sensor_buffer)}, total readings: {count}")

        except Exception as e:
            logger.error(f"Error reading from Pico: {e}")
            time.sleep(0.1)


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
    """Load the trained model, scaler, and label encoder"""
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
            logger.info(f"Scaler loaded from {SCALER_PATH}")
        else:
            logger.warning(f"Scaler not found at {SCALER_PATH}.")
            return False

        if os.path.exists(ENCODER_PATH):
            label_encoder = joblib.load(ENCODER_PATH)
            logger.info(f"Label encoder loaded from {ENCODER_PATH}")
            logger.info(f"Classes: {label_encoder.classes_}")
        else:
            logger.warning(f"Label encoder not found at {ENCODER_PATH}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def predict_character(sensor_data):
    """Predict character from sensor data using loaded model"""
    global model, scaler, label_encoder

    if model is None or scaler is None or label_encoder is None:
        logger.warning("Model not loaded, using mock prediction")
        return mock_prediction(sensor_data)

    try:
        features = np.array([[
            sensor_data['thumb_bent'],
            sensor_data['index_bent'],
            sensor_data['middle_bent'],
            sensor_data['ring_bent'],
            sensor_data['pinky_bent'],
            sensor_data['accel_x'],
            sensor_data['accel_y'],
            sensor_data['accel_z'],
            sensor_data['heading'],
            sensor_data['rolling'],
            sensor_data['pitch'],
            sensor_data['thumb_touch'],
            sensor_data['index_touch'],
            sensor_data['middle_touch']
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
    """Fallback mock prediction when model isn't available"""
    import random
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    thumb_bend = sensor_data['thumb_bent']
    if thumb_bend > 90:
        predicted = random.choice(['A', 'B', 'C', 'D', 'E'])
    elif thumb_bend < 50:
        predicted = random.choice(['T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    else:
        predicted = random.choice(list(characters))

    confidence = random.uniform(0.6, 0.9)
    return predicted, confidence


# API Endpoints
@app.route('/api/detect', methods=['POST'])
def detect_character():
    try:
        data = request.get_json()
        expected_character = data.get('expected_character') if data else None

        # Clear stale data, then collect for exactly 1 second
        with sensor_buffer_lock:
            sensor_buffer.clear()

        time.sleep(COLLECTION_WINDOW_SECONDS)

        with sensor_buffer_lock:
            recent = [entry["data"] for entry in sensor_buffer]
            sensor_buffer.clear()

        logger.info(f"Readings collected in {COLLECTION_WINDOW_SECONDS}s window: {len(recent)}")

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
        return jsonify({
            "error": str(e),
            "detected_character": None,
            "success": False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if backend and Pico are working"""
    with sensor_buffer_lock:
        buf_size = len(sensor_buffer)
        last_timestamp = sensor_buffer[-1]["timestamp"] if sensor_buffer else None

    return jsonify({
        "status": "ok",
        "pico_connected": serial_connection is not None and serial_connection.is_open,
        "has_sensor_data": buf_size > 0,
        "buffer_size": buf_size,
        "model_loaded": model is not None,
        "last_data_age_ms": int((datetime.now() - last_timestamp).total_seconds() * 1000) if last_timestamp else None
    })


@app.route('/api/sensor/latest', methods=['GET'])
def get_latest_sensor():
    """Debug endpoint to see most recent sensor reading"""
    with sensor_buffer_lock:
        if not sensor_buffer:
            return jsonify({"error": "No sensor data"}), 404
        latest = sensor_buffer[-1]

    return jsonify({
        "sensor_data": latest["data"],
        "timestamp": latest["timestamp"].isoformat()
    })


def start_backend():
    """Initialize and start the backend"""
    logger.info("Starting ASL Detection Backend...")

    if load_model():
        logger.info("✅ Model loaded successfully")
    else:
        logger.warning("⚠️ Running in mock mode (no model loaded)")

    if not connect_to_pico():
        logger.warning("Could not connect to Pico. Will retry in background...")

    pico_thread = threading.Thread(target=read_pico_stream, daemon=True)
    pico_thread.start()
    logger.info("Background sensor reader started")

    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)


if __name__ == '__main__':
    start_backend()
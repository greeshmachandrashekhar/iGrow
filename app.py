
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Find the saved model files
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ML" / "saved_models"
# Load the trained tomato price prediction model
try:
    price_model = joblib.load(
        MODEL_DIR / "price_model.pkl"
    )
    price_features = joblib.load(
        MODEL_DIR / "price_features.pkl"
    )
except Exception:
    price_model = None
    price_features = None
    app.logger.exception(
        "Tomato price model could not be loaded"
    )

# Load the trained crop recommendation models
try:
    model = joblib.load(MODEL_DIR / "crop_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    crop_encoder = joblib.load(
        MODEL_DIR / "crop_label_encoder.pkl"
    )
except Exception as error:
    raise RuntimeError(
        f"Could not load crop model files from {MODEL_DIR}. "
        "Check that all required files exist and are compatible."
    ) from error


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "iGROW API is running"
    })


@app.route("/api/predict", methods=["POST"])
def predict_crop():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send inputs as JSON."}), 400

    required = [
        "temperature", "humidity", "ph", "ec",
        "light_hours", "water", "season"
    ]

    missing = [key for key in required if key not in data]
    if missing:
        return jsonify({
            "error": "Missing fields: " + ", ".join(missing)
        }), 400

    try:
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        ec = float(data["ec"])
        light_hours = float(data["light_hours"])

        water = str(data["water"]).strip().lower()
        season = str(data["season"]).strip().lower()

        values = [
            temperature, humidity, ph, ec, light_hours
        ]

        if not all(pd.notna(value) for value in values):
            return jsonify({"error": "Inputs must be finite numbers."}), 400

        if not 0 <= humidity <= 100:
            return jsonify({"error": "Humidity must be 0–100."}), 400

        if not 0 <= ph <= 14:
            return jsonify({"error": "pH must be 0–14."}), 400

        if not 0 <= light_hours <= 24:
            return jsonify({"error": "Light hours must be 0–24."}), 400

        # Use the category names expected by the trained encoders.
        water_encoder = encoders["water availablity"]
        season_encoder = encoders["season"]

        water_encoded = water_encoder.transform([water])[0]
        season_encoded = season_encoder.transform([season])[0]

        input_data = pd.DataFrame([[
            temperature,
            humidity,
            ph,
            ec,
            light_hours,
            water_encoded,
            season_encoded
        ]], columns=[
            "temperature",
            "humidity",
            "ph",
            "ec",
            "light hours",
            "water availablity",
            "season"
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        crop_name = crop_encoder.inverse_transform(prediction)[0]

        return jsonify({
            "recommended_crop": str(crop_name)
        })

    except ValueError as error:
        return jsonify({
            "error": f"Invalid input or unsupported category: {error}"
        }), 400
    except Exception:
        app.logger.exception("Crop prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the VS Code terminal."
        }), 500


@app.route("/api/predict-price", methods=["POST"])
def predict_price():
    if price_model is None or price_features is None:
        return jsonify({
            "error": (
                "Price model is unavailable. Check that "
                "price_model.pkl and price_features.pkl "
                "exist in ML/saved_models."
            )
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send inputs as JSON."}), 400

    required = [
        "price_lag_1", "price_lag_2", "price_lag_3",
        "price_lag_4", "price_lag_5", "price_lag_6",
        "price_lag_7", "temperature", "humidity",
        "rainfall", "date"
    ]

    missing = [key for key in required if key not in data]
    if missing:
        return jsonify({
            "error": "Missing fields: " + ", ".join(missing)
        }), 400

    try:
        numeric_fields = required[:-1]
        values = {
            key: float(data[key])
            for key in numeric_fields
        }

        if not all(
            pd.notna(value) and abs(value) != float("inf")
            for value in values.values()
        ):
            return jsonify({
                "error": "All numeric inputs must be finite numbers."
            }), 400

        if any(values[f"price_lag_{i}"] < 0 for i in range(1, 8)):
            return jsonify({
                "error": "Prices cannot be negative."
            }), 400

        if not 0 <= values["humidity"] <= 100:
            return jsonify({
                "error": "Humidity must be between 0 and 100."
            }), 400

        if values["rainfall"] < 0:
            return jsonify({
                "error": "Rainfall cannot be negative."
            }), 400

        prediction_date = pd.to_datetime(
            data["date"], errors="raise"
        )

        input_data = pd.DataFrame([{
            **values,
            "day": prediction_date.day,
            "month": prediction_date.month
        }])

        # Match the feature order used during training.
        input_data = input_data[price_features]

        prediction = price_model.predict(input_data)
        predicted_price = float(prediction[0])

        if not pd.notna(predicted_price) or abs(predicted_price) == float("inf"):
            raise ValueError("Model returned an invalid price.")

        return jsonify({
            "crop": "Tomato",
            "predicted_price": round(predicted_price, 2),
            "unit": "INR per quintal",
            "date": prediction_date.strftime("%Y-%m-%d")
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({
            "error": f"Invalid price prediction input: {error}"
        }), 400
    except Exception:
        app.logger.exception("Price prediction failed")
        return jsonify({
            "error": "Price prediction failed. Check the server terminal."
        }), 500

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
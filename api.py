"""
DiaBD Diabetes Risk Prediction - REST API
-------------------------------------------
Loads the actual trained scikit-learn model and serves predictions
over HTTP so the static website can call it live.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# Load trained model and preprocessing artifacts
# ---------------------------------------------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("final_model.pkl")
selected_features = joblib.load("selected_features.pkl")

REQUIRED_FIELDS = selected_features

# Tuned decision threshold
OPTIMAL_THRESHOLD = 0.407


# ---------------------------------------------------------
# Serve the website
# ---------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(".", "index.html")


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "features_expected": REQUIRED_FIELDS
    })


# ---------------------------------------------------------
# Prediction API
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(force=True)

    # Check required features
    missing = [f for f in REQUIRED_FIELDS if f not in data]

    if missing:
        return jsonify({
            "error": f"Missing fields: {missing}"
        }), 400

    # Build input in EXACT training feature order
    try:
        input_df = pd.DataFrame([
            {f: float(data[f]) for f in REQUIRED_FIELDS}
        ])[REQUIRED_FIELDS]

    except (ValueError, TypeError) as e:
        return jsonify({
            "error": f"Invalid input values: {e}"
        }), 400

    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # Predict probability
    probability = float(
        model.predict_proba(input_scaled)[0][1]
    )

    # Apply tuned threshold
    prediction = int(
        probability >= OPTIMAL_THRESHOLD
    )

    # Logistic Regression feature contributions
    coefs = model.coef_[0]

    contributions = {
        feature: float(input_scaled[0][i] * coefs[i])
        for i, feature in enumerate(REQUIRED_FIELDS)
    }

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "contributions": contributions
    })


# ---------------------------------------------------------
# Run locally
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loaded model expects features:", REQUIRED_FIELDS)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
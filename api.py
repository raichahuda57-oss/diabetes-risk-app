"""
DiaBD Diabetes Risk Prediction - REST API
-------------------------------------------
Loads the actual trained scikit-learn model and serves predictions
over HTTP so the static website can call it live (no hardcoded numbers).

Run with:  python api.py
Then open index.html in your browser (it calls http://localhost:5000/predict)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # allow the browser (different origin: file:// or a different port) to call this API

# ---------------------------------------------------------
# Load the REAL trained artifacts once, at server startup
# ---------------------------------------------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("final_model.pkl")
selected_features = joblib.load("selected_features.pkl")

REQUIRED_FIELDS = selected_features

# Tuned decision threshold (from STEP 10.3 in the notebook: recall >= 0.75 constraint,
# maximize precision within that). Replaces the naive default of 0.5.
OPTIMAL_THRESHOLD = 0.407


@app.route("/health", methods=["GET"])
def health():
    """Quick check that the server and model are loaded correctly."""
    return jsonify({"status": "ok", "features_expected": REQUIRED_FIELDS})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate all required fields are present
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Build input row in the EXACT column order used during training
    try:
        input_df = pd.DataFrame([{f: float(data[f]) for f in REQUIRED_FIELDS}])[REQUIRED_FIELDS]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input values: {e}"}), 400

    # Same preprocessing as training: scale only (no SMOTE at inference time)
    input_scaled = scaler.transform(input_df)

    probability = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(probability >= OPTIMAL_THRESHOLD)

    # Per-feature contribution (log-odds) for the "why this prediction" explanation
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


if __name__ == "__main__":
    print("Loaded model expects features:", REQUIRED_FIELDS)
    app.run(host="0.0.0.0", port=5000, debug=False)

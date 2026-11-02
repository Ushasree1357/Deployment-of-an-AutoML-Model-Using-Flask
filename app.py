from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

try:
    model_bundle = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model_bundle = None

SPECIES_NAMES = ["Setosa", "Versicolor", "Virginica"]
FIELD_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None, bundle=model_bundle)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for field in FIELD_NAMES:
            raw_value = request.form.get(field, "").strip()
            if not raw_value:
                raise ValueError("Please enter all four measurements.")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError("Measurements must be valid numbers.") from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Measurements must be finite numbers greater than zero.")
            values.append(value)

        if model_bundle is None:
            raise RuntimeError("The model is not trained yet. Run `python train_model.py` first.")

        sample = pd.DataFrame([values], columns=model_bundle["feature_names"])
        prediction_index = int(model_bundle["model"].predict(sample)[0])
        probabilities = model_bundle["model"].predict_proba(sample)[0]
        result = {
            "species": SPECIES_NAMES[prediction_index],
            "confidence": float(np.max(probabilities)),
        }
        return render_template("index.html", result=result, error=None, bundle=model_bundle)
    except (ValueError, RuntimeError) as exc:
        return render_template("index.html", result=None, error=str(exc), bundle=model_bundle), 400


if __name__ == "__main__":
    app.run(debug=False)

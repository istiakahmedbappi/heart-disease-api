from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import HeartInput, PredictionOutput

# Load the trained model once, when the server starts (not on every request).
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "heart_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Heart Disease Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model_type": type(model).__name__,
        "n_features": len(model.feature_names_in_),
        "features": list(model.feature_names_in_),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: HeartInput):
    # One row, columns in the same order the model was trained with
    row = pd.DataFrame([data.model_dump()])[list(model.feature_names_in_)]

    prediction = int(model.predict(row)[0])          # 1 or 0
    probability = float(model.predict_proba(row)[0][1])  # chance of class 1

    return {"heart_disease": prediction == 1, "probability": round(probability, 3)}
import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocess import preprocess_dataframe, align_features


MODEL_PATH = Path("models/trained_model.joblib")
FEATURES_PATH = Path("models/feature_columns.joblib")
METADATA_PATH = Path("models/model_metadata.json")


model = joblib.load(str(MODEL_PATH))
training_columns = joblib.load(str(FEATURES_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

threshold = metadata["threshold"]


def predict_single(data: dict) -> dict:
    df = pd.DataFrame([data])

    df = preprocess_dataframe(df)
    df = align_features(df, training_columns)

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= threshold)

    label = "Subscribed" if prediction == 1 else "Not Subscribed"

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 4),
        "threshold_used": threshold,
    }
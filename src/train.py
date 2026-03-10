import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_dataframe

DATA_PATH = Path("data/customer-purchase-prediction-data.csv")
MODEL_DIR = Path("models/")
MODEL_PATH = MODEL_DIR / "trained_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"

def main():
    MODEL_DIR.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(DATA_PATH)
    df = preprocess_dataframe(df)

    target_column = "subscribed_term_deposit"

    X = df.drop(columns = [target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.3,
        random_state = 42,
        stratify = y
    )

    model = RandomForestClassifier(
        n_estimators = 200,
        max_depth = 20, 
        min_samples_split = 2,
        min_samples_leaf = 4,
        class_weight = 'balanced',
        random_state = 42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X_train.columns), FEATURES_PATH)

    metadata = {
        "model_name": "RandomForestClassifier",
        "target_column": target_column,
        "threshold": 0.4
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature columns to: {FEATURES_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")

if __name__ == "__main__":
    main()
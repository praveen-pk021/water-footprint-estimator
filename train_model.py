from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "daily_water_usage",
    "rice_consumption_kg",
    "meat_consumption_kg",
    "electricity_usage_kwh",
    "household_size",
]
TARGET_COLUMN = "total_water_footprint"


@dataclass
class ModelArtifacts:
    model_name: str
    mae: float
    r2: float


def train_and_select_model(df: pd.DataFrame) -> tuple[object, ModelArtifacts]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_model_name = ""
    best_model = None
    best_r2 = float("-inf")
    best_mae = float("inf")

    for model_name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"{model_name}: MAE={mae:.2f}, R2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_mae = mae
            best_model_name = model_name
            best_model = model

    artifacts = ModelArtifacts(model_name=best_model_name, mae=best_mae, r2=best_r2)
    return best_model, artifacts


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    data_path = "data/water_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "data/water_data.csv not found. Run `python data_generator.py` first."
        )

    df = pd.read_csv(data_path)
    model, metrics = train_and_select_model(df)

    bundle = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "metrics": {
            "model_name": metrics.model_name,
            "mae": metrics.mae,
            "r2": metrics.r2,
        },
    }

    out_path = "models/model.pkl"
    joblib.dump(bundle, out_path)

    print(f"Selected model: {metrics.model_name}")
    print(f"Saved model bundle to: {out_path}")
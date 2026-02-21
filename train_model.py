from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

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


class NumpyLinearRegressor:
    """Minimal linear regressor using least-squares (no scikit-learn dependency)."""

    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        x_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        x_aug = np.c_[np.ones((x_arr.shape[0], 1)), x_arr]
        beta, *_ = np.linalg.lstsq(x_aug, y_arr, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        x_arr = X.to_numpy(dtype=float)
        return self.intercept_ + x_arr @ self.coef_


def _train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1 - (ss_res / ss_tot))


def train_and_select_model(df: pd.DataFrame) -> tuple[object, ModelArtifacts]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = NumpyLinearRegressor().fit(X_train, y_train)
    preds = model.predict(X_test)
    y_test_arr = y_test.to_numpy(dtype=float)
    mae = _mae(y_test_arr, preds)
    r2 = _r2(y_test_arr, preds)

    print(f"NumpyLinearRegression: MAE={mae:.2f}, R2={r2:.4f}")

    artifacts = ModelArtifacts(model_name="NumpyLinearRegression", mae=mae, r2=r2)
    return model, artifacts


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

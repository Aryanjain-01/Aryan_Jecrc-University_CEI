"""Tesla deliveries prediction model.

This script trains a regression model to predict Estimated_Deliveries from
vehicle, market, pricing, and infrastructure features in the dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
(ROOT / ".matplotlib").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = ROOT / "tesla_deliveries_dataset_2015_2025.csv"
OUTPUT_DIR = ROOT / "model_outputs"
TARGET = "Estimated_Deliveries"
RANDOM_STATE = 42


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=1), errors="coerce"
    )
    df = df.sort_values(["Date", "Region", "Model"]).reset_index(drop=True)
    return df


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def save_feature_importance(
    pipeline: Pipeline, categorical_features: list[str], numeric_features: list[str]
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    encoded_cats = onehot.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numeric_features, encoded_cats])

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top = importance.head(12).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#1f77b4")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()
    return importance


def plot_predictions(results: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(
        results["actual_deliveries"],
        results["predicted_deliveries"],
        alpha=0.75,
        color="#2ca02c",
        edgecolor="white",
        linewidth=0.4,
    )
    low = min(results["actual_deliveries"].min(), results["predicted_deliveries"].min())
    high = max(results["actual_deliveries"].max(), results["predicted_deliveries"].max())
    plt.plot([low, high], [low, high], color="#d62728", linewidth=2)
    plt.title("Actual vs Predicted Tesla Deliveries")
    plt.xlabel("Actual deliveries")
    plt.ylabel("Predicted deliveries")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()

    leakage_columns = [
        "CO2_Saved_tons",
    ]
    categorical_features = ["Region", "Model", "Source_Type"]
    numeric_features = [
        "Year",
        "Month",
        "Production_Units",
        "Avg_Price_USD",
        "Battery_Capacity_kWh",
        "Range_km",
        "Charging_Stations",
    ]
    feature_columns = numeric_features + categorical_features

    usable_df = df.dropna(subset=[TARGET]).copy()
    x = usable_df[feature_columns]
    y = usable_df[TARGET]

    split_index = int(len(usable_df) * 0.8)
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_meta = usable_df.iloc[split_index:][["Date", "Year", "Month", "Region", "Model"]]

    pipeline = build_pipeline(categorical_features, numeric_features)

    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        n_jobs=1,
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    predictions = np.clip(predictions, a_min=0, a_max=None)

    baseline_value = float(y_train.median())
    baseline_predictions = np.full_like(y_test, baseline_value, dtype=float)

    metrics = {
        "rows": len(usable_df),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "features_used": ", ".join(feature_columns),
        "excluded_possible_leakage": ", ".join(leakage_columns),
        "baseline_mae": mean_absolute_error(y_test, baseline_predictions),
        "baseline_rmse": rmse(y_test, baseline_predictions),
        "model_mae": mean_absolute_error(y_test, predictions),
        "model_rmse": rmse(y_test, predictions),
        "model_r2": r2_score(y_test, predictions),
        "cv_rmse_mean": -float(np.mean(cv_scores)),
        "cv_rmse_std": float(np.std(cv_scores)),
    }

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    results = test_meta.copy()
    results["actual_deliveries"] = y_test.values
    results["predicted_deliveries"] = predictions.round().astype(int)
    results["absolute_error"] = (
        results["actual_deliveries"] - results["predicted_deliveries"]
    ).abs()
    results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    importance = save_feature_importance(pipeline, categorical_features, numeric_features)
    plot_predictions(results)

    print("Tesla Deliveries Model")
    print("=" * 24)
    print(f"Dataset rows: {metrics['rows']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print()
    print("Evaluation")
    print(f"Baseline RMSE: {metrics['baseline_rmse']:.2f}")
    print(f"Model RMSE:    {metrics['model_rmse']:.2f}")
    print(f"Model MAE:     {metrics['model_mae']:.2f}")
    print(f"Model R2:      {metrics['model_r2']:.4f}")
    print(f"CV RMSE:       {metrics['cv_rmse_mean']:.2f} +/- {metrics['cv_rmse_std']:.2f}")
    print()
    print("Top features")
    print(importance.head(8).to_string(index=False))
    print()
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

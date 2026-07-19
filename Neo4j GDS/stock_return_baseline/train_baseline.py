# -*- coding: utf-8 -*-
"""
Time-based train/test split + compare naive baselines against classical models using the
centrality/community features built by build_features.py.

Train: year <= 2023
Test:  year >= 2024

Baselines (in order of complexity):
    1. Naive persistence  - predict this year = last year's return for that ticker
    2. Naive mean         - predict this year = that ticker's historical mean return (train only)
    3. Linear Regression  - centrality + ministry one-hot only (no lag)
    4. Random Forest      - centrality + ministry one-hot + lagged_return (the main baseline)

Run:
    python train_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
PANEL_CSV = HERE / "training_panel.csv"

TRAIN_CUTOFF_YEAR = 2023  # train: year <= this, test: year > this

CENTRALITY_COLS = ["degree", "weighted_degree", "betweenness", "closeness", "eigenvector", "pagerank"]


def report(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name:22s}  MAE={mae:7.3f}  RMSE={rmse:7.3f}  R2={r2:7.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not PANEL_CSV.exists():
        raise SystemExit(f"Missing {PANEL_CSV} - run build_features.py first")

    panel = pd.read_csv(PANEL_CSV)
    ministry_cols = [c for c in panel.columns if c.startswith("ministry_")]

    train = panel[panel["year"] <= TRAIN_CUTOFF_YEAR].copy()
    test = panel[panel["year"] > TRAIN_CUTOFF_YEAR].copy()
    print(f"train: {len(train)} rows (years {train['year'].min()}-{train['year'].max()})")
    print(f"test:  {len(test)} rows (years {test['year'].min()}-{test['year'].max()})")
    assert train["year"].max() <= TRAIN_CUTOFF_YEAR, "train/test split leaked future years"
    print()

    results = []

    # 1. Naive persistence: predict = lagged_return (already computed as a column)
    results.append(report("Naive: persistence", test["annual_return_pct"], test["lagged_return"]))

    # 2. Naive mean: per-ticker historical mean from TRAIN only; fall back to train's overall mean
    ticker_mean = train.groupby("ticker")["annual_return_pct"].mean()
    overall_mean = train["annual_return_pct"].mean()
    naive_mean_pred = test["ticker"].map(ticker_mean).fillna(overall_mean)
    results.append(report("Naive: ticker mean", test["annual_return_pct"], naive_mean_pred))

    # 3. Linear Regression: centrality + ministry one-hot only (no lag)
    feat_no_lag = CENTRALITY_COLS + ministry_cols
    scaler = StandardScaler()
    X_train_nolag = scaler.fit_transform(train[feat_no_lag])
    X_test_nolag = scaler.transform(test[feat_no_lag])
    lin = LinearRegression().fit(X_train_nolag, train["annual_return_pct"])
    results.append(report("Linear (no lag)", test["annual_return_pct"], lin.predict(X_test_nolag)))

    # 4. Random Forest: centrality + ministry one-hot + lagged_return
    feat_full = CENTRALITY_COLS + ministry_cols + ["lagged_return"]
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)
    rf.fit(train[feat_full], train["annual_return_pct"])
    results.append(report("Random Forest", test["annual_return_pct"], rf.predict(test[feat_full])))

    print("\n=== Comparison table ===")
    print(pd.DataFrame(results).to_string(index=False))

    print("\n=== Random Forest feature importances ===")
    importances = pd.Series(rf.feature_importances_, index=feat_full).sort_values(ascending=False)
    print(importances.to_string())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Two checks using the real, full 11-year budget data (fetch_budget_data.py) instead of the
pre-computed aggregate correlation that survived in app.py's REAL_MINISTRY_INFO:

1. Recompute correlation(ministry budget YoY%, average mapped-stock annual return) per ministry
   from the real 11-year series, and compare against the old hardcoded simple_r values.
2. Re-fit the Random Forest baseline (same time split as train_baseline.py) with budget_yoy_pct
   added as a feature, to see whether the real budget series adds predictive value beyond
   centrality/ministry/lag/party_role.

Run:
    python analyze_budget_correlation.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ministry_stock_data import REAL_MINISTRY_INFO, REAL_MINISTRY_STOCK_EDGES

HERE = Path(__file__).parent
RETURNS_CSV = HERE / "annual_returns.csv"
BUDGET_YOY_CSV = HERE / "budget_yoy_by_ministry.csv"
PANEL_FULL_CSV = HERE / "training_panel_full.csv"

TRAIN_CUTOFF_YEAR = 2023
CENTRALITY_COLS = ["degree", "weighted_degree", "betweenness", "closeness", "eigenvector", "pagerank"]
MIN_YEARS = 4  # same threshold as the original analyze_budget_stock_network.py


def recompute_correlations() -> pd.DataFrame:
    returns = pd.read_csv(RETURNS_CSV)
    ticker_to_ministry = {ticker: ministry for ministry, ticker, _ in REAL_MINISTRY_STOCK_EDGES}
    returns["ministry"] = returns["ticker"].map(ticker_to_ministry)

    avg_return_by_ministry_year = (
        returns.groupby(["ministry", "year"])["annual_return_pct"].mean().reset_index()
    )

    budget_yoy = pd.read_csv(BUDGET_YOY_CSV).rename(columns={"year_ce": "year"})

    merged = avg_return_by_ministry_year.merge(budget_yoy, on=["ministry", "year"], how="inner").dropna()

    rows = []
    for ministry, group in merged.groupby("ministry"):
        old_r = REAL_MINISTRY_INFO[ministry]["simple_r"]
        if len(group) < MIN_YEARS:
            new_r = float("nan")
        else:
            new_r = group["budget_yoy_pct"].corr(group["annual_return_pct"])
        rows.append({
            "ministry": ministry,
            "label_en": REAL_MINISTRY_INFO[ministry]["label_en"],
            "n_years": len(group),
            "old_simple_r": old_r,
            "new_simple_r_full_data": new_r,
        })
    return pd.DataFrame(rows).sort_values("new_simple_r_full_data", ascending=False)


def rerun_rf_with_budget() -> None:
    if not PANEL_FULL_CSV.exists():
        print(f"! {PANEL_FULL_CSV} not found, skipping RF re-fit")
        return

    panel = pd.read_csv(PANEL_FULL_CSV)
    ministry_cols = [c for c in panel.columns if c.startswith("ministry_")]
    party_cols = [c for c in panel.columns if c.startswith("party_")]

    train = panel[panel["year"] <= TRAIN_CUTOFF_YEAR].copy()
    test = panel[panel["year"] > TRAIN_CUTOFF_YEAR].copy()

    configs = [
        ("centrality + ministry + lag", CENTRALITY_COLS + ministry_cols + ["lagged_return"]),
        ("+ party_role", CENTRALITY_COLS + ministry_cols + ["lagged_return"] + party_cols),
        ("+ budget_yoy_pct (real data)", CENTRALITY_COLS + ministry_cols + ["lagged_return"] + party_cols + ["budget_yoy_pct"]),
    ]

    print("\n=== Random Forest: does the REAL budget YoY% series help prediction? ===")
    for label, feats in configs:
        rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)
        rf.fit(train[feats], train["annual_return_pct"])
        pred = rf.predict(test[feats])
        mae = mean_absolute_error(test["annual_return_pct"], pred)
        rmse = mean_squared_error(test["annual_return_pct"], pred) ** 0.5
        r2 = r2_score(test["annual_return_pct"], pred)
        print(f"{label:32s}  MAE={mae:7.3f}  RMSE={rmse:7.3f}  R2={r2:7.3f}")
        last_rf, last_feats = rf, feats

    print("\n=== Feature importances (full model incl. budget_yoy_pct) ===")
    importances = pd.Series(last_rf.feature_importances_, index=last_feats).sort_values(ascending=False)
    print(importances.to_string())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== Correlation(budget YoY%, avg stock return) per ministry: old vs recomputed (real, full 11-year data) ===")
    corr_df = recompute_correlations()
    print(corr_df.to_string(index=False))

    rerun_rf_with_budget()


if __name__ == "__main__":
    main()

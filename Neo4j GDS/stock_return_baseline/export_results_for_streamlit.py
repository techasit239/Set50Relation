# -*- coding: utf-8 -*-
"""
Consolidate the analysis results that analyze_party_effect.py / analyze_budget_correlation.py /
train_baseline.py currently only print to stdout, into small CSVs the Streamlit page can load
directly (keeps the deployed page lightweight - no scikit-learn/scipy re-fit needed at page-load
time, just reads pre-computed summary tables).

Output (all written next to this script):
    model_comparison.csv        model, MAE, RMSE, R2
    party_effect_summary.csv    party_role, mean, median, std, n
    correlation_comparison.csv  ministry, label_en, n_years, old_simple_r, new_simple_r_full_data

Run:
    python export_results_for_streamlit.py
"""
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from analyze_budget_correlation import recompute_correlations
from ministry_stock_data import REAL_MINISTRY_STOCK_EDGES

HERE = Path(__file__).parent
PANEL_FULL_CSV = HERE / "training_panel_full.csv"

TRAIN_CUTOFF_YEAR = 2023
CENTRALITY_COLS = ["degree", "weighted_degree", "betweenness", "closeness", "eigenvector", "pagerank"]
PARTY_ROLES = ["core", "coalition", "independent", "none"]


def export_model_comparison(panel: pd.DataFrame) -> None:
    ministry_cols = [c for c in panel.columns if c.startswith("ministry_")]
    party_cols = [c for c in panel.columns if c.startswith("party_")]

    train = panel[panel["year"] <= TRAIN_CUTOFF_YEAR].copy()
    test = panel[panel["year"] > TRAIN_CUTOFF_YEAR].copy()

    rows = []

    def add_result(name: str, y_pred) -> None:
        rows.append({
            "model": name,
            "MAE": mean_absolute_error(test["annual_return_pct"], y_pred),
            "RMSE": mean_squared_error(test["annual_return_pct"], y_pred) ** 0.5,
            "R2": r2_score(test["annual_return_pct"], y_pred),
        })

    add_result("Naive: persistence", test["lagged_return"])

    ticker_mean = train.groupby("ticker")["annual_return_pct"].mean()
    overall_mean = train["annual_return_pct"].mean()
    add_result("Naive: ticker mean", test["ticker"].map(ticker_mean).fillna(overall_mean))

    feat_no_lag = CENTRALITY_COLS + ministry_cols
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_no_lag])
    X_test = scaler.transform(test[feat_no_lag])
    lin = LinearRegression().fit(X_train, train["annual_return_pct"])
    add_result("Linear (centrality, no lag)", lin.predict(X_test))

    configs = [
        ("Random Forest (centrality+ministry+lag)", CENTRALITY_COLS + ministry_cols + ["lagged_return"]),
        ("+ party_role", CENTRALITY_COLS + ministry_cols + ["lagged_return"] + party_cols),
        ("+ budget_yoy_pct (real data)", CENTRALITY_COLS + ministry_cols + ["lagged_return"] + party_cols + ["budget_yoy_pct"]),
    ]
    for label, feats in configs:
        rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)
        rf.fit(train[feats], train["annual_return_pct"])
        add_result(label, rf.predict(test[feats]))

    pd.DataFrame(rows).to_csv(HERE / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"wrote model_comparison.csv ({len(rows)} rows)")


def export_party_effect_summary(panel: pd.DataFrame) -> None:
    party_cols = [f"party_{r}" for r in PARTY_ROLES]
    panel = panel.copy()
    panel["party_role"] = panel[party_cols].idxmax(axis=1).str.replace("party_", "", regex=False)

    summary = panel.groupby("party_role")["annual_return_pct"].agg(["mean", "median", "std", "count"])
    summary = summary.reindex([r for r in PARTY_ROLES if r in summary.index]).reset_index()

    groups = [panel.loc[panel["party_role"] == r, "annual_return_pct"] for r in PARTY_ROLES if r in panel["party_role"].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    summary["anova_f"] = f_stat
    summary["anova_p"] = p_value

    summary.to_csv(HERE / "party_effect_summary.csv", index=False, encoding="utf-8-sig")
    print(f"wrote party_effect_summary.csv ({len(summary)} rows), ANOVA p={p_value:.4f}")


def export_correlation_comparison() -> None:
    corr_df = recompute_correlations()
    corr_df.to_csv(HERE / "correlation_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"wrote correlation_comparison.csv ({len(corr_df)} rows)")


def main() -> None:
    if not PANEL_FULL_CSV.exists():
        raise SystemExit(f"Missing {PANEL_FULL_CSV} - run build_budget_features.py first")
    panel = pd.read_csv(PANEL_FULL_CSV)

    export_model_comparison(panel)
    export_party_effect_summary(panel)
    export_correlation_comparison()


if __name__ == "__main__":
    main()

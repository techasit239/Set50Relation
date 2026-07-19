# -*- coding: utf-8 -*-
"""
Does a stock's ministry being led by a minister from the core ruling party correlate with
that stock's annual return, vs. a coalition partner / independent / (2014-2019) NCPO-appointed
technocrat?

1. Group annual_return_pct by party_role: mean/median/std/n + one-way ANOVA across all 4 groups
   + direct two-sample comparisons (core vs coalition, core vs independent) - the pairs that
   most directly answer the question.
2. Re-fit the Random Forest baseline (same time split as train_baseline.py) with party_role
   added, to see whether it adds predictive value beyond centrality/ministry/lag, and where it
   ranks in feature importance.

Run:
    python analyze_party_effect.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).parent
PANEL_CSV = HERE / "training_panel_with_party.csv"

TRAIN_CUTOFF_YEAR = 2023
CENTRALITY_COLS = ["degree", "weighted_degree", "betweenness", "closeness", "eigenvector", "pagerank"]
PARTY_ROLES = ["core", "coalition", "independent", "none"]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not PANEL_CSV.exists():
        raise SystemExit(f"Missing {PANEL_CSV} - run build_party_features.py first")

    panel = pd.read_csv(PANEL_CSV)
    party_cols = [f"party_{r}" for r in PARTY_ROLES]
    panel["party_role"] = panel[party_cols].idxmax(axis=1).str.replace("party_", "", regex=False)

    print("=== Group comparison: annual_return_pct by party_role ===")
    summary = panel.groupby("party_role")["annual_return_pct"].agg(["mean", "median", "std", "count"])
    summary = summary.reindex([r for r in PARTY_ROLES if r in summary.index])
    print(summary.to_string())
    print()
    print("(none = 2014-2019 NCPO military government - no elected party at all, a different")
    print(" macro/political regime, not a 'zero ruling-party effect' baseline - don't read it as")
    print(" directly comparable to the elected-era groups.)")
    print()

    groups = [panel.loc[panel["party_role"] == r, "annual_return_pct"] for r in PARTY_ROLES if r in panel["party_role"].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"One-way ANOVA across all groups: F={f_stat:.3f}, p={p_value:.4f}"
          f"  ({'significant' if p_value < 0.05 else 'NOT significant'} at alpha=0.05)")
    print(f"(small-sample caveat: n={len(panel)} rows total across {len(groups)} groups - "
          f"some cells are thin, treat this as exploratory, not confirmatory)")
    print()

    for other in ["coalition", "independent"]:
        core = panel.loc[panel["party_role"] == "core", "annual_return_pct"]
        cmp_group = panel.loc[panel["party_role"] == other, "annual_return_pct"]
        t_stat, p_val = stats.ttest_ind(core, cmp_group, equal_var=False)
        direction = "higher" if core.mean() > cmp_group.mean() else "lower"
        print(f"core (mean={core.mean():.2f}, n={len(core)}) vs {other} (mean={cmp_group.mean():.2f}, n={len(cmp_group)}): "
              f"core is {direction}, t={t_stat:.3f}, p={p_val:.4f}"
              f"  ({'significant' if p_val < 0.05 else 'NOT significant'} at alpha=0.05)")
    print()

    print("=== Random Forest: does adding party_role improve on the existing baseline? ===")
    train = panel[panel["year"] <= TRAIN_CUTOFF_YEAR].copy()
    test = panel[panel["year"] > TRAIN_CUTOFF_YEAR].copy()
    ministry_cols = [c for c in panel.columns if c.startswith("ministry_")]

    feat_baseline = CENTRALITY_COLS + ministry_cols + ["lagged_return"]
    feat_with_party = feat_baseline + party_cols

    for label, feats in [("baseline (no party)", feat_baseline), ("baseline + party_role", feat_with_party)]:
        rf = RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)
        rf.fit(train[feats], train["annual_return_pct"])
        pred = rf.predict(test[feats])
        mae = mean_absolute_error(test["annual_return_pct"], pred)
        rmse = mean_squared_error(test["annual_return_pct"], pred) ** 0.5
        r2 = r2_score(test["annual_return_pct"], pred)
        print(f"{label:24s}  MAE={mae:7.3f}  RMSE={rmse:7.3f}  R2={r2:7.3f}")
        last_rf, last_feats = rf, feats

    print()
    print("=== Feature importances (baseline + party_role model) ===")
    importances = pd.Series(last_rf.feature_importances_, index=last_feats).sort_values(ascending=False)
    print(importances.to_string())


if __name__ == "__main__":
    main()

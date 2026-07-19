# -*- coding: utf-8 -*-
"""
Compute ministry budget YoY% change from the real budget_by_ministry.csv (fetch_budget_data.py)
and join it onto training_panel_with_party.csv, so the return-prediction baseline can use real
budget data instead of just the pre-computed aggregate correlation summary.

Fiscal year (BE) -> calendar year (CE): year_ce = year_be - 543, same convention as
build_party_features.py, matching how annual_returns.csv / training_panel.csv is indexed. This is
a simplification (Thai fiscal year runs Oct-Sep, not Jan-Dec) but matches what the original SNA
report / analyze_budget_stock_network.py did.

Input:
    budget_by_ministry.csv         (from fetch_budget_data.py)
    training_panel_with_party.csv  (from build_party_features.py)

Output:
    training_panel_full.csv    training_panel_with_party.csv + budget_yoy_pct
    budget_yoy_by_ministry.csv year_ce, min_name, budget_yoy_pct  (for analyze_budget_correlation.py)

Run:
    python build_budget_features.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
BUDGET_CSV = HERE / "budget_by_ministry.csv"
PANEL_CSV = HERE / "training_panel_with_party.csv"
OUT_PANEL_CSV = HERE / "training_panel_full.csv"
OUT_BUDGET_YOY_CSV = HERE / "budget_yoy_by_ministry.csv"

BE_OFFSET = 543


def main() -> None:
    if not BUDGET_CSV.exists() or not PANEL_CSV.exists():
        raise SystemExit(f"Missing input files in {HERE} - run fetch_budget_data.py / build_party_features.py first")

    budget = pd.read_csv(BUDGET_CSV)
    budget["year_ce"] = budget["year"] - BE_OFFSET
    budget = budget.sort_values(["min_code", "year_ce"])
    budget["budget_yoy_pct"] = budget.groupby("min_code")["total_budget_million_baht"].pct_change() * 100

    budget_yoy = budget[["year_ce", "min_name", "budget_yoy_pct"]].rename(columns={"min_name": "ministry"})
    budget_yoy.to_csv(OUT_BUDGET_YOY_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {OUT_BUDGET_YOY_CSV} ({len(budget_yoy)} rows)")

    panel = pd.read_csv(PANEL_CSV)
    panel = panel.merge(
        budget_yoy.rename(columns={"year_ce": "year"}),
        left_on=["ticker_ministry", "year"],
        right_on=["ministry", "year"],
        how="left",
    )
    panel = panel.drop(columns=["ministry"])

    n_null = panel["budget_yoy_pct"].isna().sum()
    print(f"panel rows: {len(panel)}, missing budget_yoy_pct: {n_null}")
    if n_null:
        # first fiscal year of each ministry's series has no prior year to compute YoY from
        print("  (expected: first year on record for a given ministry has no prior-year budget to diff against)")

    panel.to_csv(OUT_PANEL_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {OUT_PANEL_CSV}")


if __name__ == "__main__":
    main()

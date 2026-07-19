# -*- coding: utf-8 -*-
"""
Slice the party-effect comparison by ministry, to check whether the pooled "core beats
everything else" gap (see analyze_party_effect.py) actually holds up within a given ministry
across different administrations, or whether it's driven by specific ministries / time periods.

Two views:
    1. mean/median/n of annual_return_pct by (ministry, cabinet_no) - which ministry drove
       each administration's number.
    2. mean/median/n by (ministry, party_role) - within each ministry specifically, does core
       beat non-core there, or does the pooled gap disappear once ministry is held fixed?

Cells with n < MIN_N are flagged as too thin to read.

Run:
    python analyze_by_ministry_cabinet.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PANEL_CSV = HERE / "training_panel_with_party.csv"

MIN_N = 5


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not PANEL_CSV.exists():
        raise SystemExit(f"Missing {PANEL_CSV} - run build_party_features.py first")

    panel = pd.read_csv(PANEL_CSV)
    party_cols = ["party_core", "party_coalition", "party_independent", "party_none"]
    panel["party_role"] = panel[party_cols].idxmax(axis=1).str.replace("party_", "", regex=False)

    print("=== mean annual_return_pct by (ministry, cabinet_no) ===")
    by_cabinet = panel.groupby(["ticker_ministry", "cabinet_no"])["annual_return_pct"].agg(["mean", "median", "count"])
    by_cabinet = by_cabinet.round(2)
    print(by_cabinet.to_string())
    thin = by_cabinet[by_cabinet["count"] < MIN_N]
    if len(thin):
        print(f"\n! {len(thin)} of {len(by_cabinet)} (ministry, cabinet) cells have n < {MIN_N} - treat those numbers as unreliable")
    print()

    print("=== mean annual_return_pct by (ministry, party_role) ===")
    by_role = panel.groupby(["ticker_ministry", "party_role"])["annual_return_pct"].agg(["mean", "median", "count"])
    by_role = by_role.round(2)
    print(by_role.to_string())
    thin_role = by_role[by_role["count"] < MIN_N]
    if len(thin_role):
        print(f"\n! {len(thin_role)} of {len(by_role)} (ministry, party_role) cells have n < {MIN_N} - treat those numbers as unreliable")

    print()
    print("=== Within-ministry check: does core beat non-core in the SAME ministry? ===")
    for ministry, group in panel.groupby("ticker_ministry"):
        core_mean = group.loc[group["party_role"] == "core", "annual_return_pct"].mean()
        noncore_mean = group.loc[group["party_role"] != "core", "annual_return_pct"].mean()
        core_n = (group["party_role"] == "core").sum()
        noncore_n = (group["party_role"] != "core").sum()
        if pd.isna(core_mean) or core_n < MIN_N or noncore_n < MIN_N:
            print(f"{ministry:45s}  (too little data to compare: core n={core_n}, non-core n={noncore_n})")
            continue
        gap = core_mean - noncore_mean
        print(f"{ministry:45s}  core={core_mean:7.2f} (n={core_n:3d})  non-core={noncore_mean:7.2f} (n={noncore_n:3d})  gap={gap:+7.2f}")


if __name__ == "__main__":
    main()

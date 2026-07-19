# -*- coding: utf-8 -*-
"""
Join cabinet_history.csv (which ministry/party held each ministry, when) onto
training_panel.csv, assigning each (ticker, year) row the party_role that held that
ticker's ministry for the most days of that calendar year.

Dates in cabinet_history.csv are Buddhist Era (as sourced from Thai Wikipedia) - converted
to Gregorian here (BE - 543), not in the CSV, to stay faithful to the source text.

Input:
    cabinet_history.csv    cabinet_no, pm, ministry, minister, party, party_role, start_date, end_date
    training_panel.csv     (from build_features.py)

Output:
    training_panel_with_party.csv   training_panel.csv + ticker_ministry, cabinet_no,
                                     party_role (one-hot columns)

Run:
    python build_party_features.py
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from ministry_stock_data import REAL_MINISTRY_STOCK_EDGES

HERE = Path(__file__).parent
CABINET_CSV = HERE / "cabinet_history.csv"
PANEL_CSV = HERE / "training_panel.csv"
OUT_CSV = HERE / "training_panel_with_party.csv"

BE_OFFSET = 543


def parse_be_date(s: str) -> dt.date:
    s = s.strip()
    if s == "ปัจจุบัน":
        return dt.date.today()
    year_be, month, day = s.split("-")
    return dt.date(int(year_be) - BE_OFFSET, int(month), int(day))


def overlap_days(a_start: dt.date, a_end: dt.date, b_start: dt.date, b_end: dt.date) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max((end - start).days, 0)


def assign_dominant_per_ministry_year(cabinet: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """For each (ministry, year), pick whichever value of `value_col` (e.g. party_role or
    cabinet_no) covered the most days that calendar year - summed across every row sharing
    that value (handles a role/cabinet spanning multiple ministers within the same year)."""
    rows = []
    years = range(
        min(r.start_date.year for r in cabinet.itertuples()),
        max(r.end_date.year for r in cabinet.itertuples()) + 1,
    )
    for ministry, group in cabinet.groupby("ministry"):
        for year in years:
            year_start = dt.date(year, 1, 1)
            year_end = dt.date(year, 12, 31)
            days_by_value: dict = {}
            for row in group.itertuples():
                days = overlap_days(year_start, year_end, row.start_date, row.end_date)
                if days > 0:
                    value = getattr(row, value_col)
                    days_by_value[value] = days_by_value.get(value, 0) + days
            best = max(days_by_value, key=days_by_value.get) if days_by_value else None
            if best is not None:
                rows.append({"ministry": ministry, "year": year, value_col: best})
    return pd.DataFrame(rows)


def main() -> None:
    if not CABINET_CSV.exists() or not PANEL_CSV.exists():
        raise SystemExit(f"Missing input files in {HERE} - run build_features.py first")

    cabinet = pd.read_csv(CABINET_CSV)
    cabinet["start_date"] = cabinet["start_date"].apply(parse_be_date)
    cabinet["end_date"] = cabinet["end_date"].apply(parse_be_date)

    ministry_year_role = assign_dominant_per_ministry_year(cabinet, "party_role")
    ministry_year_cabinet = assign_dominant_per_ministry_year(cabinet, "cabinet_no")

    ticker_to_ministry = {ticker: ministry for ministry, ticker, _ in REAL_MINISTRY_STOCK_EDGES}

    panel = pd.read_csv(PANEL_CSV)
    panel["ticker_ministry"] = panel["ticker"].map(ticker_to_ministry)  # deliberately NOT "ministry_"-prefixed - avoids colliding with the one-hot ministry_XXX columns

    panel = panel.merge(ministry_year_role, left_on=["ticker_ministry", "year"], right_on=["ministry", "year"], how="left")
    panel = panel.drop(columns=["ministry"])
    panel = panel.merge(ministry_year_cabinet, left_on=["ticker_ministry", "year"], right_on=["ministry", "year"], how="left")
    panel = panel.drop(columns=["ministry"])

    n_null = panel["party_role"].isna().sum()
    if n_null:
        print(f"! warning: {n_null} rows have no party_role match (ministry/year not in cabinet_history)")

    panel = pd.get_dummies(panel, columns=["party_role"], prefix="party")

    panel.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"input panel rows: {len(pd.read_csv(PANEL_CSV))}")
    print(f"output panel rows: {len(panel)}")
    print(f"party_role columns added: {[c for c in panel.columns if c.startswith('party_')]}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

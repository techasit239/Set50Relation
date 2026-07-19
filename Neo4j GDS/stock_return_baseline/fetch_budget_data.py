# -*- coding: utf-8 -*-
"""
Fetch real ministry-level budget data (FY2559-2569) from the Thailand Government Spending
open-data API (govspending.data.go.th) and aggregate it to one row per (ministry, year).

The old script + API endpoint (opend.data.go.th/govspending/bbgf_summary) is dead - the whole
portal was rebuilt (Nuxt/Vue frontend at govspending.data.go.th, backend at
api-govspending.data.go.th). Endpoint discovered by intercepting the bulk-download form's fetch
call in a browser: `GET /api/get/api/bulkfile?user_key=...&type=GF&code=gf-summary&year=YYYY`
returns `{"data": "<zip url>"}`; the zip contains one CSV with department-level rows
(min_code/min_name, dept_code/dept_name, year, budget received, amount disbursed, disbursed %).
Aggregated here by summing department rows up to the ministry level. The old API key still works
with this endpoint (just used the wrong param names/path before).

Output:
    budget_by_ministry.csv   year, min_code, min_name, total_budget_million_baht,
                             total_disbursed_million_baht

Run:
    python fetch_budget_data.py [api_key]
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

API_KEY = sys.argv[1] if len(sys.argv) > 1 else "HC5wIOE8A0R56QozVZzq5MOgs7Vr8Jp5"
BULKFILE_URL = "https://api-govspending.data.go.th/api/get/api/bulkfile"

YEARS = list(range(2559, 2570))  # FY2559-2569 (2016-2026), matches annual_returns.csv coverage

# min_code -> our project's Thai ministry name (matches ministry_stock_data.py / REAL_MINISTRY_INFO)
MIN_CODE_TO_MINISTRY = {
    "03000": "กระทรวงการคลัง",
    "12000": "กระทรวงพลังงาน",
    "08000": "กระทรวงคมนาคม",
    "11000": "กระทรวงดิจิทัลเพื่อเศรษฐกิจและสังคม",
    "21000": "กระทรวงสาธารณสุข",
    "13000": "กระทรวงพาณิชย์",
    "07000": "กระทรวงเกษตรและสหกรณ์",
    "15000": "กระทรวงมหาดไทย",
    "22000": "กระทรวงอุตสาหกรรม",
    "05000": "กระทรวงการท่องเที่ยวและกีฬา",
}

HERE = Path(__file__).parent
OUT_CSV = HERE / "budget_by_ministry.csv"


def fetch_year(year: int) -> pd.DataFrame:
    resp = requests.get(
        BULKFILE_URL,
        params={"user_key": API_KEY, "type": "GF", "code": "gf-summary", "year": year},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"year {year}: API returned success=false: {payload}")
    zip_url = payload["data"]

    zip_resp = requests.get(zip_url, timeout=60)
    zip_resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, encoding="utf-8-sig")
    return df


def main() -> None:
    all_rows = []
    for year in YEARS:
        print(f"fetching FY{year}...")
        try:
            df = fetch_year(year)
        except Exception as e:
            print(f"  ! failed for {year}: {e}")
            continue

        df = df.rename(columns={
            "รหัสหน่วยงานระดับกระทรวง": "min_code",
            "ชื่อหน่วยงานระดับกระทรวง": "min_name",
            "ได้รับงบประมาณ (ล้านบาท)": "budget_million_baht",
            "เบิกจ่ายไป (ล้านบาท)": "disbursed_million_baht",
        })
        df["min_code"] = df["min_code"].astype(str).str.zfill(5)

        agg = (
            df.groupby("min_code", as_index=False)
            .agg(
                min_name=("min_name", "first"),
                total_budget_million_baht=("budget_million_baht", "sum"),
                total_disbursed_million_baht=("disbursed_million_baht", "sum"),
            )
        )
        agg = agg[agg["min_code"].isin(MIN_CODE_TO_MINISTRY)]
        agg["min_name"] = agg["min_code"].map(MIN_CODE_TO_MINISTRY)  # use our canonical Thai name
        agg.insert(0, "year", year)
        all_rows.append(agg)

    if not all_rows:
        raise SystemExit("no years fetched successfully")

    result = pd.concat(all_rows, ignore_index=True).sort_values(["min_code", "year"])
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nwrote {len(result)} rows -> {OUT_CSV}")
    print(f"years covered: {sorted(result['year'].unique())}")
    print(f"ministries covered: {result['min_code'].nunique()} / {len(MIN_CODE_TO_MINISTRY)}")
    missing = set(MIN_CODE_TO_MINISTRY) - set(result["min_code"].unique())
    if missing:
        print(f"! missing ministries: {[MIN_CODE_TO_MINISTRY[m] for m in missing]}")


if __name__ == "__main__":
    main()

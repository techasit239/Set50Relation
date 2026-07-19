# -*- coding: utf-8 -*-
"""
Fetch historical monthly closing prices for the 50 SET50 tickers via yfinance and derive
annual returns. Recreates the historical return series that was removed from the repo during
the GitHub cleanup - needed as the regression target for train_baseline.py.

Output:
    annual_returns.csv   ticker, year, annual_return_pct

Run:
    python fetch_prices.py
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from ministry_stock_data import REAL_MINISTRY_STOCK_EDGES

HERE = Path(__file__).parent
OUT_CSV = HERE / "annual_returns.csv"

START = "2016-01-01"


def main() -> None:
    tickers = sorted({ticker for _, ticker, _ in REAL_MINISTRY_STOCK_EDGES})
    yf_tickers = [f"{t}.BK" for t in tickers]

    print(f"downloading monthly closes for {len(tickers)} tickers from {START}...")
    raw = yf.download(
        yf_tickers,
        start=START,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise SystemExit("yfinance returned no data - check network connection / ticker list")

    close = raw["Close"] if "Close" in raw.columns else raw

    rows = []
    for ticker in tickers:
        yf_ticker = f"{ticker}.BK"
        if yf_ticker not in close.columns:
            print(f"  ! no data for {yf_ticker}, skipping")
            continue
        series = close[yf_ticker].dropna()
        if series.empty:
            continue
        by_year = series.groupby(series.index.year)
        for year, year_series in by_year:
            if len(year_series) < 2:
                continue  # need at least first + last month to compute a return
            first_price = year_series.iloc[0]
            last_price = year_series.iloc[-1]
            annual_return_pct = (last_price / first_price - 1) * 100
            rows.append({"ticker": ticker, "year": int(year), "annual_return_pct": annual_return_pct})

    df = pd.DataFrame(rows).sort_values(["ticker", "year"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {len(df)} rows -> {OUT_CSV}")
    print(f"tickers with data: {df['ticker'].nunique()} / {len(tickers)}")
    print(f"year range: {df['year'].min()}-{df['year'].max()}")


if __name__ == "__main__":
    main()

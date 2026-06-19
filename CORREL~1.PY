"""
SET Index — Unusual Factor Correlation Analysis
================================================
Downloads data via yfinance + pytrends, computes rolling correlations,
and outputs an HTML report + CSV.

Usage:
    pip install yfinance pytrends pandas matplotlib seaborn
    python correlation_analysis.py

Outputs:
    output/correlation_report.html
    output/correlation_matrix.csv
    output/rolling_corr_<factor>.png
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE  = "2018-01-01"
END_DATE    = "2024-12-31"
ROLL_WINDOW = 90        # days for rolling correlation
OUTPUT_DIR  = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dark theme for charts
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#1f2937",
    "axes.edgecolor":    "#374151",
    "axes.labelcolor":   "#f1f5f9",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "text.color":        "#f1f5f9",
    "grid.color":        "#374151",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1f2937",
    "legend.edgecolor":  "#374151",
})

# ── Factor definitions ────────────────────────────────────────────────────────
# (ticker_or_key, display_name, category, direction_hypothesis)
YFINANCE_FACTORS = [
    # Conventional
    ("^GSPC",    "S&P 500",              "ตลาดต่างประเทศ",    "pos"),
    ("^HSI",     "Hang Seng (HSI)",      "ตลาดต่างประเทศ",    "pos"),
    ("DX-Y.NYB", "Dollar Index (DXY)",   "ตลาดต่างประเทศ",    "neg"),
    ("^VIX",     "VIX (Fear)",           "Sentiment",          "neg"),
    ("THBUSD=X", "THB/USD (บาทต่อ$)",   "มหภาคในประเทศ",     "mixed"),
    ("GC=F",     "ราคาทอง",             "สินค้าโภคภัณฑ์",    "pos"),
    ("BZ=F",     "น้ำมัน Brent",         "สินค้าโภคภัณฑ์",   "mixed"),
    # Unusual
    ("BTC-USD",  "Bitcoin",              "Unusual",            "pos"),
    ("CNYUSD=X", "CNY/USD (หยวน)",      "Unusual",            "pos"),
    ("BDRY",     "Baltic Dry ETF (BDI)", "Unusual",            "pos"),
    ("^N225",    "Nikkei 225",           "ตลาดต่างประเทศ",    "pos"),
    ("RR=F",     "ยางพารา (Rubber)",     "สินค้าโภคภัณฑ์",   "pos"),
    ("NG=F",     "ก๊าซธรรมชาติ",         "สินค้าโภคภัณฑ์",  "mixed"),
    ("ZS=F",     "ถั่วเหลือง (Soy)",    "สินค้าโภคภัณฑ์",    "neg"),
]

GTRENDS_FACTORS = [
    ("หุ้น",       "Google Trends: หุ้น",    "Unusual", "pos"),
    ("ลาออก",      "Google Trends: ลาออก",   "Unusual", "neg"),
    ("กู้เงิน",    "Google Trends: กู้เงิน",  "Unusual", "neg"),
    ("ท่องเที่ยว", "Google Trends: ท่องเที่ยว","Unusual","pos"),
]

CATEGORY_COLORS = {
    "ตลาดต่างประเทศ": "#f59e42",
    "มหภาคในประเทศ":  "#60a5fa",
    "สินค้าโภคภัณฑ์": "#a78bfa",
    "Sentiment":       "#fb7185",
    "Unusual":         "#34d399",
}

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_set_index() -> pd.Series:
    """Fetch SET Index weekly closes. Falls back to simulated if unavailable."""
    try:
        import yfinance as yf
        df = yf.download("^SET.BK", start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if not df.empty:
            closes = df["Close"].ffill()
            print(f"  ✓ SET Index: {len(closes)} rows")
            return closes.resample("W").last().dropna()
    except Exception as e:
        print(f"  ✗ SET Index fetch failed: {e}")
    # Simulate for demo
    print("  ⚠ Using simulated SET Index for demo")
    idx = pd.date_range(START_DATE, END_DATE, freq="W")
    rng = np.random.default_rng(42)
    prices = 1500 * np.exp(np.cumsum(rng.normal(0.001, 0.02, len(idx))))
    return pd.Series(prices, index=idx, name="SET")


def fetch_yfinance_factors(set_index: pd.Series) -> pd.DataFrame:
    """Download all yfinance factors and align to SET weekly index."""
    try:
        import yfinance as yf
    except ImportError:
        print("  ✗ yfinance not installed — pip install yfinance")
        return pd.DataFrame(index=set_index.index)

    frames: dict[str, pd.Series] = {}
    for ticker, name, *_ in YFINANCE_FACTORS:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  ✗ {name}: no data")
                continue
            s = df["Close"].ffill().resample("W").last()
            s.name = name
            frames[name] = s
            print(f"  ✓ {name}: {len(s)} rows")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
        time.sleep(0.3)
    return pd.DataFrame(frames).reindex(set_index.index).ffill()


def fetch_gtrends_factors(set_index: pd.Series) -> pd.DataFrame:
    """Fetch Google Trends data (weekly) for Thai keywords."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ✗ pytrends not installed — pip install pytrends")
        return pd.DataFrame(index=set_index.index)

    pytrends = TrendReq(hl="th-TH", tz=420, timeout=(10, 25))
    frames: dict[str, pd.Series] = {}
    for keyword, name, *_ in GTRENDS_FACTORS:
        try:
            pytrends.build_payload([keyword], timeframe=f"{START_DATE} {END_DATE}", geo="TH")
            df = pytrends.interest_over_time()
            if df.empty or keyword not in df.columns:
                print(f"  ✗ {name}: no data")
                continue
            s = df[keyword].resample("W").mean()
            s.name = name
            frames[name] = s
            print(f"  ✓ {name}: {len(s)} rows")
            time.sleep(1.5)  # rate limit
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return pd.DataFrame(frames).reindex(set_index.index).ffill()


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels to weekly log returns."""
    return np.log(df / df.shift(1)).dropna(how="all")


def compute_correlations(returns_set: pd.Series, returns_factors: pd.DataFrame) -> pd.DataFrame:
    """Pearson + Spearman correlation of each factor with SET, with p-values."""
    from scipy import stats
    rows = []
    for col in returns_factors.columns:
        s = returns_factors[col].dropna()
        aligned = returns_set.reindex(s.index).dropna()
        s = s.reindex(aligned.index)
        if len(aligned) < 30:
            continue
        r_p, p_p = stats.pearsonr(aligned, s)
        r_s, p_s = stats.spearmanr(aligned, s)
        rows.append({
            "Factor": col,
            "Pearson r":     round(r_p, 3),
            "Pearson p":     round(p_p, 4),
            "Spearman r":    round(r_s, 3),
            "Spearman p":    round(p_s, 4),
            "Significant":   "✓" if min(p_p, p_s) < 0.05 else "✗",
            "N (weeks)":     len(aligned),
        })
    df_out = pd.DataFrame(rows).sort_values("Pearson r", key=abs, ascending=False)
    return df_out


def plot_correlation_matrix(returns_all: pd.DataFrame, path: Path) -> None:
    """Heatmap of all pairwise correlations."""
    corr = returns_all.corr()
    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.7), max(8, len(corr) * 0.6)))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", linewidths=0.4,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        annot_kws={"size": 7}, ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation Matrix (weekly log returns)", pad=14, fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path.name}")


def plot_rolling_correlation(
    set_ret: pd.Series,
    factor_ret: pd.Series,
    factor_name: str,
    category: str,
    path: Path,
) -> None:
    """Line chart of rolling {ROLL_WINDOW}-day correlation."""
    color = CATEGORY_COLORS.get(category, "#60a5fa")
    combined = pd.concat([set_ret, factor_ret], axis=1).dropna()
    combined.columns = ["SET", factor_name]
    roll = combined["SET"].rolling(ROLL_WINDOW).corr(combined[factor_name])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    # Rolling correlation
    ax1.axhline(0, color="#374151", lw=1)
    ax1.fill_between(roll.index, roll, 0,
                     where=(roll >= 0), color=color, alpha=0.3, label="positive")
    ax1.fill_between(roll.index, roll, 0,
                     where=(roll < 0),  color="#f87171", alpha=0.3, label="negative")
    ax1.plot(roll.index, roll, color=color, lw=1.5)
    ax1.set_ylim(-1, 1)
    ax1.set_ylabel(f"{ROLL_WINDOW}-day Rolling Corr")
    ax1.set_title(f"Rolling Correlation: SET vs {factor_name}", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Factor price
    ax2.plot(combined.index, combined[factor_name], color=color, lw=1.2)
    ax2.set_ylabel(factor_name, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_html_report(
    corr_table: pd.DataFrame,
    all_factors: list[tuple],
    plots: list[tuple[str, str]],   # (factor_name, img_filename)
) -> str:
    """Generate a self-contained HTML report."""
    factor_meta = {name: (cat, hyp) for (_, name, cat, hyp) in all_factors}

    rows_html = ""
    for _, row in corr_table.iterrows():
        cat, hyp = factor_meta.get(row["Factor"], ("–", "–"))
        color = CATEGORY_COLORS.get(cat, "#94a3b8")
        sig_icon = "✓" if row["Significant"] == "✓" else "✗"
        bg = "#1a2e1a" if row["Significant"] == "✓" else ""
        rows_html += f"""
        <tr style="background:{bg}">
          <td><span style="color:{color};font-weight:600">{row['Factor']}</span></td>
          <td style="color:{color};font-size:11px">{cat}</td>
          <td style="color:{'#34d399' if row['Pearson r']>0 else '#f87171'};font-weight:600">{row['Pearson r']:.3f}</td>
          <td style="opacity:.7">{row['Pearson p']:.4f}</td>
          <td style="color:{'#34d399' if row['Spearman r']>0 else '#f87171'};font-weight:600">{row['Spearman r']:.3f}</td>
          <td style="opacity:.7">{row['Spearman p']:.4f}</td>
          <td style="font-size:16px;text-align:center">{sig_icon}</td>
          <td style="opacity:.6">{row['N (weeks)']}</td>
        </tr>"""

    charts_html = ""
    for fname, img_file in plots:
        charts_html += f"""
        <div class="chart-card">
          <h3>{fname}</h3>
          <img src="{img_file}" alt="{fname}">
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>SET Index — Unusual Factor Correlation Report</title>
<style>
  * {{box-sizing:border-box;margin:0;padding:0}}
  body {{background:#111827;color:#f1f5f9;font-family:system-ui,sans-serif;padding:24px}}
  h1 {{font-size:22px;margin-bottom:6px}}
  .sub {{color:#94a3b8;font-size:13px;margin-bottom:28px}}
  h2 {{font-size:16px;margin:28px 0 12px;color:#60a5fa}}
  table {{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:32px}}
  th {{background:#1f2937;padding:8px 10px;text-align:left;font-size:11px;
       text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;border-bottom:1px solid #374151}}
  td {{padding:8px 10px;border-bottom:1px solid #1f2937}}
  tr:hover {{background:#1a2233}}
  .legend {{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;font-size:12px}}
  .leg-item {{display:flex;align-items:center;gap:6px}}
  .leg-dot {{width:10px;height:10px;border-radius:50%}}
  .chart-grid {{display:grid;grid-template-columns:repeat(auto-fill,minmax(580px,1fr));gap:20px}}
  .chart-card {{background:#1f2937;border-radius:10px;padding:16px;border:1px solid #374151}}
  .chart-card h3 {{font-size:13px;color:#94a3b8;margin-bottom:10px}}
  .chart-card img {{width:100%;border-radius:6px}}
  .note {{background:#1f2937;border:1px solid #374151;border-radius:8px;
          padding:12px 16px;font-size:12px;color:#94a3b8;margin-bottom:24px}}
</style>
</head>
<body>
<h1>SET Index — Unusual Factor Correlation Report</h1>
<p class="sub">Period: {START_DATE} – {END_DATE} · Rolling window: {ROLL_WINDOW} days · Weekly log returns</p>

<div class="note">
  ⚠ Correlation ≠ Causation · ✓ = significant at p &lt; 0.05 · ✗ = not significant<br>
  ตัวแปรที่มี |r| สูงแต่ p สูง = อาจเป็น spurious จากขนาดตัวอย่างเล็กหรือ structural break
</div>

<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#f59e42"></div>ตลาดต่างประเทศ</div>
  <div class="leg-item"><div class="leg-dot" style="background:#60a5fa"></div>มหภาคในประเทศ</div>
  <div class="leg-item"><div class="leg-dot" style="background:#a78bfa"></div>สินค้าโภคภัณฑ์</div>
  <div class="leg-item"><div class="leg-dot" style="background:#fb7185"></div>Sentiment</div>
  <div class="leg-item"><div class="leg-dot" style="background:#34d399"></div>Unusual / แปลกประหลาด</div>
</div>

<h2>Correlation Table (sorted by |Pearson r|)</h2>
<table>
  <thead><tr>
    <th>Factor</th><th>Category</th>
    <th>Pearson r</th><th>Pearson p</th>
    <th>Spearman r</th><th>Spearman p</th>
    <th>Sig.</th><th>N weeks</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>

<h2>Rolling Correlation Charts ({ROLL_WINDOW}-day window)</h2>
<div class="chart-grid">{charts_html}</div>

<h2>Correlation Matrix (heatmap)</h2>
<img src="correlation_matrix.png" style="max-width:100%;border-radius:8px">
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("SET Index — Unusual Factor Correlation Analysis")
    print("=" * 60)

    # 1. SET Index
    print("\n[1/4] Fetching SET Index...")
    set_prices = fetch_set_index()
    set_ret    = compute_log_returns(set_prices.to_frame("SET"))["SET"]

    # 2. yfinance factors
    print("\n[2/4] Fetching yfinance factors...")
    yf_prices  = fetch_yfinance_factors(set_prices)
    yf_ret     = compute_log_returns(yf_prices) if not yf_prices.empty else pd.DataFrame()

    # 3. Google Trends
    print("\n[3/4] Fetching Google Trends (this may be slow)...")
    gt_data    = fetch_gtrends_factors(set_prices)
    gt_ret     = compute_log_returns(gt_data) if not gt_data.empty else pd.DataFrame()

    # 4. Combine and analyse
    print("\n[4/4] Computing correlations...")
    all_factor_ret = pd.concat(
        [df for df in [yf_ret, gt_ret] if not df.empty], axis=1
    )
    all_factors_meta = YFINANCE_FACTORS + GTRENDS_FACTORS

    corr_table = compute_correlations(set_ret, all_factor_ret)
    corr_table.to_csv(OUTPUT_DIR / "correlation_matrix.csv", index=False)
    print(f"  ✓ Saved correlation_matrix.csv")

    # Heatmap
    all_ret = pd.concat([set_ret.rename("SET Index"), all_factor_ret], axis=1).dropna(how="all")
    plot_correlation_matrix(all_ret, OUTPUT_DIR / "correlation_matrix.png")

    # Rolling charts (top 12 by |r|)
    top_factors = corr_table.head(12)["Factor"].tolist()
    plots: list[tuple[str, str]] = []
    for _, name, cat, hyp in all_factors_meta:
        if name not in top_factors or name not in all_factor_ret.columns:
            continue
        img_name = f"rolling_{name.replace(' ', '_').replace('/', '')[:30]}.png"
        plot_rolling_correlation(set_ret, all_factor_ret[name], name, cat,
                                  OUTPUT_DIR / img_name)
        plots.append((name, img_name))
        print(f"  ✓ Chart: {img_name}")

    # HTML report
    html = build_html_report(corr_table, all_factors_meta, plots)
    report_path = OUTPUT_DIR / "correlation_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Report saved → {report_path}")
    print("   Open in browser to view the full analysis.")

    # Preview table
    print("\n── Top correlations with SET Index ──")
    preview_cols = ["Factor", "Pearson r", "Pearson p", "Significant"]
    print(corr_table[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

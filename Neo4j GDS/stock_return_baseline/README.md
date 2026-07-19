# Stock return baseline — centrality features vs naive baselines

Baseline model (step 1 before attempting a GNN): does the ministry↔stock network structure we
already computed (centrality, from the SNA/GDS work in this project) actually help predict a
stock's annual return, compared to naive time-series baselines?

## Pipeline

1. **`ministry_stock_data.py`** — `REAL_MINISTRY_INFO` / `REAL_MINISTRY_STOCK_EDGES`, copied from
   `app.py` (ticker → ministry mapping + |correlation| edge weight, FY2559–2569).
2. **`fetch_prices.py`** — re-fetches monthly closes via `yfinance` (`.BK` suffix, same convention
   as `app.py`) for all 50 tickers from 2016 onward, derives `annual_returns.csv`. This recreates
   the historical return series that was removed from the repo during cleanup — only the single
   aggregate correlation summary survived, not a per-year series.
3. **`build_features.py`** — builds the ministry↔stock bipartite graph in `networkx`, computes
   Degree, Weighted Degree, Betweenness, Closeness, Eigenvector, PageRank per stock (static — this
   simplified graph doesn't change year to year), joins with lagged return + ministry one-hot into
   `training_panel.csv` (one row per ticker × year).
4. **`train_baseline.py`** — time-based split (train ≤2023, test 2024–2026, never random) comparing
   four models of increasing complexity.

## Result (2024–2026 held-out test years)

| model | MAE | RMSE | R² |
|---|---|---|---|
| Naive: persistence (last year's return) | 31.90 | 53.24 | -1.038 |
| Naive: ticker's historical mean | 25.13 | 40.46 | -0.177 |
| Linear Regression (centrality + ministry only, no lag) | 23.88 | 36.25 | 0.055 |
| **Random Forest** (centrality + ministry + lag) | **22.48** | **34.93** | **0.123** |

**Finding:** both naive baselines have *negative* R² on the test years (worse than just predicting
everyone's training-set mean return) — annual stock returns don't meaningfully persist or mean-revert
in a simple way here. Once the network-structure features (centrality, ministry membership) are
added, R² turns positive even with plain Linear Regression, and Random Forest (adding the lag term
back in alongside the network features) does best. Feature importances show `lagged_return` and
`ministry_Industry` dominate, with `weighted_degree` and `pagerank` contributing meaningfully —
the network features are not just noise.

Caveats: small sample (443 rows total, 144 in the test set), centrality features are static (single
aggregate graph, not recomputed per year), so this is a first-pass baseline, not a rigorous result.

## Extension: does a minister from the ruling party correlate with higher stock returns?

5. **`cabinet_history.csv`** — Thai Cabinets 61–66 (2014–present), extracted verbatim from raw
   Thai Wikipedia wikitext (summarized WebFetch passes garbled some names first try — raw wikitext
   + explicit "copy verbatim" instructions fixed that). Columns: `cabinet_no, pm, ministry,
   minister, party, party_role, start_date, end_date` (dates in Buddhist Era, as sourced).
   `party_role` ∈ `core` (led government formation) / `coalition` / `independent` /
   `none_junta` (2014–2019 NCPO military government — no elected party at all, a different
   regime, not "no ruling-party effect"). Cross-checked: the Cabinet 66 rows match
   `REAL_CURRENT_CABINET` already hardcoded in `app.py` exactly, on every ministry.
6. **`build_party_features.py`** — converts BE→Gregorian dates, computes the **day-weighted
   overlap** of each `party_role` against every calendar year for each ministry (summed across all
   ministers who held that role that year, not just whoever had the single longest individual
   stint), assigns the dominant role, joins onto `training_panel.csv` →
   `training_panel_with_party.csv`.
7. **`analyze_party_effect.py`** — group comparison + one-way ANOVA + two-sample tests, plus a
   Random Forest re-fit with `party_role` added to see if it helps prediction, not just group means.

### Result

| party_role | mean return % | median | n |
|---|---|---|---|
| **core** | **18.93** | 11.98 | 138 |
| coalition | 5.76 | -0.46 | 106 |
| independent | 4.01 | 6.19 | 84 |
| none (2014–2019 junta) | 5.70 | -0.18 | 115 |

- One-way ANOVA across all 4 groups: F=2.573, p=0.054 — just short of conventional significance.
- **core vs independent: p=0.026 (significant)** — core vs coalition: p=0.075 (not significant).
- Adding `party_role` to the Random Forest baseline made test-set R² slightly *worse* (0.103 vs
  0.123 without it) — the raw group-mean gap doesn't translate into extra *predictive* value once
  centrality/ministry/lag are already in the model, likely because `party_role` is correlated with
  which ministries had core-party ministers rather than an independent signal.

**Verdict:** stocks whose ministry was run by a core-ruling-party minister *did* see a notably
higher average return in this sample (~19% vs mid-single-digits for every other category), and the
core-vs-independent gap clears a conventional significance threshold — but the overall ANOVA is
borderline, the sample is small (thin cells: 84–138 rows per group), and the effect doesn't help
out-of-sample prediction beyond what the network features already captured. Read as suggestive,
not conclusive — and this is correlational, with no control for global market conditions or
sector-specific shocks that happened to coincide with a given administration.

## Follow-up 1: does the party effect hold up within a single ministry, or is it a time-period artifact?

Grouping by `cabinet_no` (`Neo4j GDS/stock_return_baseline` scratch check, not a saved script)
showed the pooled "core" average was *not* uniform across administrations — Cabinet 64 (Pheu Thai)
core ministries averaged only 1.84% while Cabinet 62 and 66 averaged ~27-28%, and Cabinet 66's
overall average (all party roles, 27.90%) nearly matches its core-only average (27.26%) — a sign
of a market-wide/time-period effect, not a party-specific one.

**`analyze_by_ministry_cabinet.py`** breaks the comparison down by ministry directly: within 7 of
9 ministries with enough data to compare, `core` still shows a *positive* gap over non-core in that
same ministry (e.g. Interior +12.25pp, Digital Economy +17.43pp, Energy +13.26pp) — Commerce
(-7.57pp) and Public Health (-12.19pp) go the other way, and Industry's gap (+77.48pp) looks
outlier-driven (huge swings on a couple of manufacturing stocks, small n). So the effect isn't
purely a time confound — it's directionally consistent in most ministries — but it's noisy,
non-universal, and n is thin per cell (many below 10 rows).

## Follow-up 2: minister career-movement network

Several ministers in `cabinet_history.csv` moved between multiple ministries across cabinets
(e.g. สุริยะ จึงรุ่งเรืองกิจ: Industry→Transport→Transport→Agriculture; อุตตม สาวนายน: Digital
Economy→Industry→Finance) — this recreates the "3-mode network" idea from the original SNA report
(`Neo4j GDS/SNA__Report_6720422013.pdf`, which found Suriya Juangroongruangkit had unusually high
betweenness for exactly this reason), now grounded in the real 2014-2026 cabinet history.

- **`build_minister_network.py`** — tripartite graph: **Minister** nodes (57, one per distinct
  person in `cabinet_history.csv`) <-> **Ministry** nodes (10) <-> **Stock** nodes (50).
  Minister→Ministry edge weight = total days held, summed across every stint/cabinet (so a
  reappointed minister like อนุทิน ชาญวีรกูล, in office across 5 cabinets, gets one edge with the
  summed tenure). Ministry→Stock edges reuse `REAL_MINISTRY_STOCK_EDGES` as-is. Computes Degree,
  Weighted Degree, Betweenness, Eigenvector, PageRank (`minister_network_centrality.csv`).
- **`visualize_minister_network.py`** — renders the network (`minister_network.png`), node size
  ∝ betweenness so multi-ministry "bridge" ministers stand out visually.

**Top betweenness ministers** (i.e. structural bridges from holding multiple ministries):
อุตตม สาวนายน (0.244 — Finance, Digital Economy, Industry), เอกนัฏ พร้อมพันธุ์ (0.219 — Energy,
Industry), สุริยะ จึงรุ่งเรืองกิจ (0.214 — Transport, Industry, Agriculture), ภูมิธรรม เวชยชัย
(0.169 — Commerce, Interior). Ministers who stayed in one ministry their whole career (e.g.
ประจิน จั่นตอง, Transport only) score betweenness = 0, even after long tenure — betweenness rewards
spanning distinct ministries, not tenure length.

Note: had to reset `ax.set_xlim`/`set_ylim` explicitly after `nx.draw_networkx_edges` in the
visualization — that call was blowing up matplotlib's autoscaled axis range by ~1000x (a
networkx/matplotlib FancyArrowPatch autoscale quirk), squeezing the real plot into a tiny corner.

## Follow-up 3: real ministry budget data (FY2559-2569)

The original `budget_by_ministry_2554_2569.csv` / `fetch_budget_data.py` were deleted in the
GitHub cleanup, leaving only the aggregate `simple_r`/`partial_r` correlation values hardcoded in
`app.py` - not the underlying budget amounts. Recovered the real per-year, per-ministry figures:

- **`fetch_budget_data.py`** — the old API (`opend.data.go.th/govspending/bbgf_summary`) is dead
  (site fully rebuilt). Found the current one by intercepting the bulk-download form's `fetch()`
  call in a browser: `GET api-govspending.data.go.th/api/get/api/bulkfile?user_key=...&type=GF&
  code=gf-summary&year=YYYY` returns a ZIP download URL; the ZIP contains one CSV with
  department-level rows, aggregated here up to ministry level (summed by `min_code`). The old API
  key still works — just needed the right param names/path.
- Output: **`budget_by_ministry.csv`** — `year, min_code, min_name, total_budget_million_baht,
  total_disbursed_million_baht`, all 11 years (2559-2569) x all 10 ministries, no gaps.
- Verified: FY2559 Finance = 199,174.0666 million baht matches **exactly** the pre-deletion value
  seen earlier this session; FY2564/2567 Finance figures are within ~0.1% of independently-sourced
  news figures the user cross-checked. FY2569 (the current, still-open fiscal year) differs more
  (~8%) from a preliminary news figure - expected, since in-year figures still get revised.

## Follow-up 4: joining the real budget data in — did the old correlation summary hold up?

- **`build_budget_features.py`** — computes `budget_yoy_pct` per ministry per (BE→CE-converted)
  year from `budget_by_ministry.csv`, joins it onto `training_panel_with_party.csv` →
  `training_panel_full.csv`. Every one of the 443 rows got a match (the only year with no prior-year
  budget to diff against, 2016, was already excluded from the panel by the lag requirement).
- **`analyze_budget_correlation.py`** — recomputes `correlation(budget_yoy_pct, avg mapped-stock
  return)` per ministry from the real 11-year series and compares it against the old hardcoded
  `simple_r` in `REAL_MINISTRY_INFO`.

### Result: most ministries match closely, two do not

| ministry | old simple_r | recomputed (real data) |
|---|---|---|
| Digital Economy | 0.418 | 0.422 |
| Transport | 0.239 | 0.235 |
| Public Health | 0.175 | 0.191 |
| **Finance** | **0.292** | **0.088** |
| Industry | -0.007 | -0.009 |
| Energy | 0.030 | -0.040 |
| Tourism & Sports | -0.061 | -0.079 |
| Interior | -0.196 | -0.177 |
| **Commerce** | **-0.602** | **-0.190** |
| Agriculture | -0.477 | -0.457 |

8 of 10 ministries match within a few hundredths. **Finance and Commerce are materially
different** — Finance's correlation with stock returns is much weaker than previously reported,
and Commerce's negative correlation is much less pronounced. This isn't a bug in this recomputation:
the underlying budget YoY% series was spot-checked and matches the pre-deletion data exactly (e.g.
FY2560/2017 Finance = 9.30821256827219%, identical to the value seen earlier this session before
the files were deleted). The discrepancy must come from either the exact stock-return data vintage
or the exact year range/methodology used for the original `simple_r` values, which can't be fully
reconstructed since the original computation itself is gone - flagging this as an open discrepancy
rather than silently trusting either number.

### Does the real budget series help predict returns?

| feature set | MAE | RMSE | R² |
|---|---|---|---|
| centrality + ministry + lag | 22.48 | 34.93 | 0.123 |
| + party_role | 22.98 | 35.33 | 0.103 |
| + budget_yoy_pct (real data) | 22.87 | 35.38 | 0.100 |

R² doesn't improve by adding the real budget series - but unlike `party_role` (which had almost no
standalone importance), `budget_yoy_pct` ranks **4th in feature importance** (0.112, behind only
`ministry_Industry`, `lagged_return`, and `weighted_degree`) - it carries real, distinct signal that
the model uses, it just doesn't translate into better held-out accuracy at this sample size. Same
pattern as the rest of this investigation: real signal exists, but 443 rows isn't enough to convert
it into a reliably better forecast.

## Streamlit page

`export_results_for_streamlit.py` consolidates the results that `analyze_party_effect.py` /
`analyze_budget_correlation.py` otherwise only print, into small CSVs
(`model_comparison.csv`, `party_effect_summary.csv`, `correlation_comparison.csv`) so the deployed
page never needs scikit-learn/scipy at page-load time - it only reads these plus
`cabinet_history.csv`, `budget_by_ministry.csv`, and `minister_network_centrality.csv`.

[pages/2_Ministry_Budget_Politics.py](../../pages/2_Ministry_Budget_Politics.py) renders all of the
above as five tabs: Cabinet & Party History, Ministry Budget Over Time, Ruling Party vs Returns,
Minister Career Network, and Prediction Model Results.

## Re-running

```bash
python fetch_prices.py                  # ~1 min, needs internet
python build_features.py
python train_baseline.py
python build_party_features.py          # needs cabinet_history.csv (already collected)
python analyze_party_effect.py
python analyze_by_ministry_cabinet.py
python build_minister_network.py
python visualize_minister_network.py
python fetch_budget_data.py             # ~30s, needs internet
python build_budget_features.py
python analyze_budget_correlation.py
python export_results_for_streamlit.py  # refresh the CSVs pages/2_Ministry_Budget_Politics.py reads
```

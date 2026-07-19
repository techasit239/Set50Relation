# -*- coding: utf-8 -*-
"""
Build the ministry<->stock bipartite graph, compute static centrality features per stock via
networkx, and join with annual_returns.csv into a training panel (one row per ticker x year).

Input:
    annual_returns.csv        (from fetch_prices.py)
    ministry_stock_data.py     REAL_MINISTRY_INFO / REAL_MINISTRY_STOCK_EDGES

Output:
    training_panel.csv         ticker, year, annual_return_pct, lagged_return, ministry,
                                degree, weighted_degree, betweenness, closeness, eigenvector, pagerank

Run:
    python build_features.py
"""
from pathlib import Path

import networkx as nx
import pandas as pd

from ministry_stock_data import REAL_MINISTRY_INFO, REAL_MINISTRY_STOCK_EDGES

HERE = Path(__file__).parent
RETURNS_CSV = HERE / "annual_returns.csv"
OUT_CSV = HERE / "training_panel.csv"


def build_ministry_stock_graph() -> nx.Graph:
    graph = nx.Graph()
    for th_name in REAL_MINISTRY_INFO:
        graph.add_node(th_name, node_type="ministry")
    for ministry, ticker, corr in REAL_MINISTRY_STOCK_EDGES:
        graph.add_node(ticker, node_type="stock")
        graph.add_edge(ministry, ticker, weight=abs(corr))
    return graph


def compute_centrality_features(graph: nx.Graph) -> pd.DataFrame:
    degree_centrality = nx.degree_centrality(graph)
    weighted_degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="weight")
    closeness = nx.closeness_centrality(graph)
    eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    pagerank = nx.pagerank(graph, weight="weight")

    ticker_to_ministry = {
        ticker: REAL_MINISTRY_INFO[ministry]["label_en"]
        for ministry, ticker, _ in REAL_MINISTRY_STOCK_EDGES
    }

    rows = []
    for node, data in graph.nodes(data=True):
        if data["node_type"] != "stock":
            continue
        rows.append({
            "ticker": node,
            "ministry": ticker_to_ministry[node],
            "degree": degree_centrality[node],
            "weighted_degree": weighted_degree[node],
            "betweenness": betweenness[node],
            "closeness": closeness[node],
            "eigenvector": eigenvector[node],
            "pagerank": pagerank[node],
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not RETURNS_CSV.exists():
        raise SystemExit(f"Missing {RETURNS_CSV} - run fetch_prices.py first")

    returns = pd.read_csv(RETURNS_CSV).sort_values(["ticker", "year"])
    returns["lagged_return"] = returns.groupby("ticker")["annual_return_pct"].shift(1)

    graph = build_ministry_stock_graph()
    centrality = compute_centrality_features(graph)

    panel = returns.merge(centrality, on="ticker", how="left")
    # drop rows with no lag (first year on record for that ticker) - can't predict without history
    panel = panel.dropna(subset=["lagged_return"]).reset_index(drop=True)

    panel = pd.get_dummies(panel, columns=["ministry"], prefix="ministry")

    panel.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"tickers with centrality features: {len(centrality)}")
    print(f"panel rows (after dropping first-year-per-ticker for lag): {len(panel)}")
    print(f"year range in panel: {panel['year'].min()}-{panel['year'].max()}")
    print(f"any NaN in centrality columns: "
          f"{panel[['degree','weighted_degree','betweenness','closeness','eigenvector','pagerank']].isna().any().any()}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

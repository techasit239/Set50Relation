# -*- coding: utf-8 -*-
"""
Build a tripartite Minister <-> Ministry <-> Stock network from cabinet_history.csv +
ministry_stock_data.py, and compute centrality. Recreates the "3-mode network" idea from the
original SNA report (Neo4j GDS/SNA__Report_6720422013.pdf, which found Suriya Juangroongruangkit
had unusually high betweenness for holding multiple ministries) using the real, verbatim-sourced
2014-2026 cabinet history collected this session.

Nodes:
    Minister  - one per distinct name in cabinet_history.csv
    Ministry  - 10, fixed
    Stock     - 50, fixed (from ministry_stock_data.py)

Edges:
    Minister -> Ministry   weight = total days held, summed across every stint/cabinet
    Ministry -> Stock      weight = |correlation| (reused as-is from ministry_stock_data.py)

Output:
    minister_network_centrality.csv   node, node_type, degree, weighted_degree, betweenness,
                                       eigenvector, pagerank

Run:
    python build_minister_network.py
"""
import datetime as dt
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from ministry_stock_data import REAL_MINISTRY_INFO, REAL_MINISTRY_STOCK_EDGES

HERE = Path(__file__).parent
CABINET_CSV = HERE / "cabinet_history.csv"
OUT_CSV = HERE / "minister_network_centrality.csv"

BE_OFFSET = 543


def parse_be_date(s: str) -> dt.date:
    s = s.strip()
    if s == "ปัจจุบัน":
        return dt.date.today()
    year_be, month, day = s.split("-")
    return dt.date(int(year_be) - BE_OFFSET, int(month), int(day))


def build_graph(cabinet: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()

    for ministry, info in REAL_MINISTRY_INFO.items():
        graph.add_node(ministry, node_type="ministry", label=info["label_en"])

    for ministry, ticker, corr in REAL_MINISTRY_STOCK_EDGES:
        graph.add_node(ticker, node_type="stock")
        graph.add_edge(ministry, ticker, weight=abs(corr))

    minister_ministry_days: dict[tuple[str, str], int] = {}
    minister_party: dict[str, str] = {}
    for row in cabinet.itertuples():
        days = (row.end_date - row.start_date).days
        key = (row.minister, row.ministry)
        minister_ministry_days[key] = minister_ministry_days.get(key, 0) + max(days, 1)
        minister_party[row.minister] = row.party  # last-seen party wins (most recent cabinet)

    for (minister, ministry), days in minister_ministry_days.items():
        graph.add_node(minister, node_type="minister", party=minister_party[minister])
        graph.add_edge(minister, ministry, weight=days / 365.0)  # years held, as edge weight

    return graph


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not CABINET_CSV.exists():
        raise SystemExit(f"Missing {CABINET_CSV}")

    cabinet = pd.read_csv(CABINET_CSV)
    cabinet["start_date"] = cabinet["start_date"].apply(parse_be_date)
    cabinet["end_date"] = cabinet["end_date"].apply(parse_be_date)

    graph = build_graph(cabinet)
    print(f"graph: {graph.number_of_nodes()} nodes "
          f"({sum(1 for _, d in graph.nodes(data=True) if d['node_type']=='minister')} ministers, "
          f"{sum(1 for _, d in graph.nodes(data=True) if d['node_type']=='ministry')} ministries, "
          f"{sum(1 for _, d in graph.nodes(data=True) if d['node_type']=='stock')} stocks), "
          f"{graph.number_of_edges()} edges")

    degree = nx.degree_centrality(graph)
    weighted_degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="weight")
    eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    pagerank = nx.pagerank(graph, weight="weight")

    rows = []
    for node, data in graph.nodes(data=True):
        rows.append({
            "node": node,
            "node_type": data["node_type"],
            "degree": degree[node],
            "weighted_degree": weighted_degree[node],
            "betweenness": betweenness[node],
            "eigenvector": eigenvector[node],
            "pagerank": pagerank[node],
        })
    result = pd.DataFrame(rows).sort_values("betweenness", ascending=False)
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {OUT_CSV}")

    print("\n=== Top 10 Minister nodes by betweenness (structural 'bridges') ===")
    top_ministers = result[result["node_type"] == "minister"].head(10)
    for row in top_ministers.itertuples():
        ministries_held = sorted(graph.neighbors(row.node))
        ministries_held = [n for n in ministries_held if graph.nodes[n]["node_type"] == "ministry"]
        print(f"{row.node:30s} betweenness={row.betweenness:.4f}  ministries held: {ministries_held}")


if __name__ == "__main__":
    main()

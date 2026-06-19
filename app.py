from __future__ import annotations

import math
import os
import re
import time
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── Visual constants ──────────────────────────────────────────────────────────
DARK_BG       = "#111827"
DARK_PANEL    = "#1f2937"
DARK_BORDER   = "#374151"
DARK_TEXT     = "#f1f5f9"
DARK_SUBTEXT  = "#94a3b8"

COMPANY_COLOR   = "#06b6d4"   # cyan  – listed companies
COMPANY_BORDER  = "#22d3ee"
SELECTED_COLOR  = "#a78bfa"   # violet – focused / selected node

# 12-colour palette for shareholder communities
COMMUNITY_PALETTE = [
    "#60a5fa",  # blue
    "#34d399",  # emerald
    "#fb923c",  # orange
    "#f472b6",  # pink
    "#fbbf24",  # amber
    "#818cf8",  # indigo
    "#22d3ee",  # cyan
    "#f87171",  # red
    "#4ade80",  # light-green
    "#e879f9",  # fuchsia
    "#a3e635",  # lime
    "#38bdf8",  # sky
]

def community_color(community_id: int) -> str:
    return COMMUNITY_PALETTE[community_id % len(COMMUNITY_PALETTE)]


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Assign a community index to every node using greedy modularity."""
    if graph.number_of_nodes() == 0:
        return {}
    try:
        communities = list(nx.community.greedy_modularity_communities(graph, weight=None))
        mapping: dict[str, int] = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                mapping[node] = idx
        return mapping
    except Exception:
        return {n: 0 for n in graph.nodes}

SET50_URL = "https://www.set.or.th/th/market/index/set50/overview"
SHAREHOLDER_URL = "https://www.set.or.th/th/market/product/stock/quote/{symbol}/major-shareholders"
CACHE_DIR    = Path("work") / "cache"
RAW_CACHE    = CACHE_DIR / "set50_shareholders.csv"
META_CACHE   = CACHE_DIR / "set50_shareholders_meta.csv"
PRICE_CACHE  = CACHE_DIR / "set50_prices.json"

NOMINEE_PATTERNS = [
    r"NVDR",
    r"NOMINEE",
    r"DEPOSITOR",
    r"STATE STREET",
    r"CITIBANK",
    r"HSBC",
    r"BNP PARIBAS",
    r"DBS",
    r"JPMORGAN",
    r"BANK OF NEW YORK",
    r"THAILAND SECURITIES DEPOSITORY",
    r"ศูนย์รับฝากหลักทรัพย์",
    r"ไทยเอ็นวีดีอาร์",
]


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def is_running_on_streamlit_cloud() -> bool:
    return bool(os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("STREAMLIT_CLOUD"))


def is_live_refresh_enabled() -> bool:
    return os.getenv("ENABLE_LIVE_REFRESH", "0").lower() in {"1", "true", "yes"}


def normalize_shareholder_name(name: str) -> str:
    cleaned = (name or "").strip().upper()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("&AMP;", "&")
    cleaned = cleaned.replace("PUBLIC COMPANY LIMITED", "PCL")
    cleaned = cleaned.replace("PUBLIC CO., LTD.", "PCL")
    cleaned = cleaned.replace("CO., LTD.", "CO LTD")
    cleaned = cleaned.replace("CO. LTD.", "CO LTD")
    return cleaned.strip(" .")


def is_nominee_holder(name: str) -> bool:
    upper_name = normalize_shareholder_name(name)
    return any(re.search(pattern, upper_name) for pattern in NOMINEE_PATTERNS)


def parse_number(value: str) -> float:
    value = (value or "").replace(",", "").strip()
    return float(value) if value else 0.0


def scrape_set50_symbols(page) -> list[str]:
    page.goto(SET50_URL, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(5000)
    page.wait_for_function(
        "() => document.querySelectorAll('table tbody tr td:first-child a').length >= 50",
        timeout=120000,
    )
    symbols = page.eval_on_selector_all(
        "table tbody tr td:first-child a",
        "nodes => nodes.map(node => node.textContent.trim())",
    )
    deduped: list[str] = []
    seen = set()
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if symbol and symbol not in seen:
            deduped.append(symbol)
            seen.add(symbol)
    return deduped[:50]


def scrape_major_shareholders(page, symbol: str) -> tuple[list[dict], dict]:
    url = SHAREHOLDER_URL.format(symbol=symbol)
    page.goto(url, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(4000)
    page.wait_for_function(
        """
        () => {
            const rows = Array.from(document.querySelectorAll('[role="tabpanel"] table tbody tr'));
            return rows.some(row => row.querySelectorAll('td').length >= 4);
        }
        """,
        timeout=120000,
    )

    overview_text = page.eval_on_selector('[role="tabpanel"]', "node => node.innerText")
    rows = page.eval_on_selector_all(
        '[role="tabpanel"] table tbody tr',
        """
        nodes => nodes
            .map(row => Array.from(row.querySelectorAll('td')).map(td => td.innerText.trim()))
            .filter(cols => cols.length >= 4)
        """,
    )

    match = re.search(r"ณ วันที่\s+(.*?)\s+ประเภท", overview_text)
    as_of_date = match.group(1).strip() if match else ""
    records: list[dict] = []
    for cols in rows:
        rank, shareholder_name, shares, pct = cols[:4]
        records.append(
            {
                "symbol": symbol,
                "rank": int(parse_number(rank)),
                "shareholder_name": shareholder_name,
                "shareholder_clean": normalize_shareholder_name(shareholder_name),
                "shares": parse_number(shares),
                "holding_pct": parse_number(pct),
                "is_nominee": is_nominee_holder(shareholder_name),
                "source_url": url,
                "as_of_date": as_of_date,
            }
        )
    meta = {"symbol": symbol, "as_of_date": as_of_date, "source_url": url}
    return records, meta


def refresh_data(limit: int | None = None, sleep_seconds: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Import Playwright only when a live refresh is explicitly requested.
    from playwright.sync_api import sync_playwright

    ensure_cache_dir()
    all_rows: list[dict] = []
    meta_rows: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        symbols = scrape_set50_symbols(page)
        if limit:
            symbols = symbols[:limit]

        for symbol in symbols:
            records, meta = scrape_major_shareholders(page, symbol)
            all_rows.extend(records)
            meta_rows.append(meta)
            time.sleep(sleep_seconds)
        browser.close()

    df = pd.DataFrame(all_rows)
    meta_df = pd.DataFrame(meta_rows)
    df.to_csv(RAW_CACHE, index=False, encoding="utf-8-sig")
    meta_df.to_csv(META_CACHE, index=False, encoding="utf-8-sig")
    return df, meta_df


def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_CACHE.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(RAW_CACHE)
    meta_df = pd.read_csv(META_CACHE) if META_CACHE.exists() else pd.DataFrame()
    return df, meta_df


def fetch_stock_prices(symbols: list[str]) -> dict[str, float]:
    """Download latest closing prices for SET symbols via yfinance (.BK suffix).
    Returns {symbol: price_thb}. Missing symbols are silently skipped."""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    tickers = [f"{s}.BK" for s in symbols]
    try:
        raw = yf.download(
            tickers,
            period="5d",          # grab a few days so weekends don't break it
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return {}
        close = raw["Close"] if "Close" in raw.columns else raw
        latest = close.ffill().iloc[-1]   # forward-fill then take most recent
        prices: dict[str, float] = {}
        for symbol in symbols:
            ticker = f"{symbol}.BK"
            if ticker in latest.index and not pd.isna(latest[ticker]):
                prices[symbol] = float(latest[ticker])
        return prices
    except Exception:
        return {}


def load_price_cache() -> dict[str, float]:
    if PRICE_CACHE.exists():
        try:
            return json.loads(PRICE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_price_cache(prices: dict[str, float]) -> None:
    ensure_cache_dir()
    PRICE_CACHE.write_text(json.dumps(prices, ensure_ascii=False, indent=2), encoding="utf-8")


def build_bipartite_graph(df: pd.DataFrame, prices: dict[str, float] | None = None) -> nx.Graph:
    prices = prices or {}
    graph = nx.Graph()
    for symbol in sorted(df["symbol"].unique()):
        graph.add_node(f"company::{symbol}", label=symbol, node_type="company")
    for holder in sorted(df["shareholder_clean"].unique()):
        display_name = (
            df.loc[df["shareholder_clean"] == holder, "shareholder_name"].value_counts().idxmax()
        )
        graph.add_node(f"holder::{holder}", label=display_name, node_type="shareholder")
    for row in df.itertuples(index=False):
        price = prices.get(row.symbol, 0.0)
        market_value = float(row.shares) * price   # THB; 0 if price unknown
        graph.add_edge(
            f"holder::{row.shareholder_clean}",
            f"company::{row.symbol}",
            weight=float(row.holding_pct),
            shares=float(row.shares),
            market_value=market_value,
        )
    return graph


def compute_centrality_scores(graph: nx.Graph, metric_name: str) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes)): 1.0}

    if metric_name == "Closeness":
        return nx.closeness_centrality(graph)
    if metric_name == "Betweenness":
        return nx.betweenness_centrality(graph, normalized=True)
    if metric_name == "Eigenvector":
        try:
            return nx.eigenvector_centrality(graph, max_iter=1000)
        except nx.NetworkXException:
            return nx.degree_centrality(graph)
    if metric_name == "Katz":
        try:
            return nx.katz_centrality(graph, alpha=0.03, beta=1.0, max_iter=1000)
        except nx.NetworkXException:
            return nx.degree_centrality(graph)
    if metric_name == "Notion of Centrality":
        degree_scores = nx.degree_centrality(graph)
        betweenness_scores = nx.betweenness_centrality(graph, normalized=True)
        return {
            node: (degree_scores.get(node, 0.0) * 0.6) + (betweenness_scores.get(node, 0.0) * 0.4)
            for node in graph.nodes
        }
    return nx.degree_centrality(graph)


def normalize_scores(raw_scores: dict[str, float], nodes: Iterable[str]) -> dict[str, float]:
    node_list = list(nodes)
    if not raw_scores:
        return {node: 0.5 for node in node_list}
    min_score = min(raw_scores.values(), default=0.0)
    max_score = max(raw_scores.values(), default=1.0)
    if math.isclose(max_score, min_score):
        return {node: 0.5 for node in node_list}
    return {
        node: (raw_scores.get(node, min_score) - min_score) / (max_score - min_score)
        for node in node_list
    }


def build_entity_options(graph: nx.Graph) -> tuple[dict[str, str], list[str]]:
    option_map: dict[str, str] = {}
    for node, attrs in sorted(
        graph.nodes(data=True),
        key=lambda item: (item[1]["node_type"], item[1]["label"]),
    ):
        prefix = "Company" if attrs["node_type"] == "company" else "Holder"
        label = f"[{prefix}] {attrs['label']}"
        option_map[label] = node
    return option_map, list(option_map.keys())


def edge_style_for_holding(holding_pct: float) -> tuple[str, int]:
    """Return (hex_color, pixel_width) for an edge based on holding percentage."""
    if holding_pct >= 10:
        return "#f87171", 4   # red    – major block (≥10 %)
    if holding_pct >= 5:
        return "#fb923c", 3   # orange – significant (5–9.99 %)
    if holding_pct >= 2:
        return "#60a5fa", 2   # blue   – moderate (2–4.99 %)
    return "#475569", 1       # slate  – minor (< 2 %)


def compute_market_value_edge_styles(
    graph: nx.Graph,
    min_width: float = 1.0,
    max_width: float = 12.0,
) -> dict[tuple[str, str], tuple[str, float]]:
    """Compute (color, width) for every edge using log10(market_value) scaling.

    Color tiers (THB):
      < 100M      → slate  #475569
      100M–1B     → blue   #60a5fa
      1B–10B      → orange #fb923c
      ≥ 10B       → red    #f87171
    Width is log-scaled across all edges with known market_value > 0.
    Edges without price data fall back to width=1, color=slate.
    """
    values = {
        (u, v): data["market_value"]
        for u, v, data in graph.edges(data=True)
        if data.get("market_value", 0) > 0
    }
    if not values:
        return {}

    log_vals = {k: math.log10(v) for k, v in values.items()}
    lo, hi = min(log_vals.values()), max(log_vals.values())
    span = hi - lo if hi > lo else 1.0

    result: dict[tuple[str, str], tuple[str, float]] = {}
    for (u, v), mv in values.items():
        # width: log-scale mapped to [min_width, max_width]
        w = min_width + (log_vals[(u, v)] - lo) / span * (max_width - min_width)
        # color tier
        if mv >= 10_000_000_000:       # ≥ 10B
            color = "#f87171"
        elif mv >= 1_000_000_000:      # 1B – 10B
            color = "#fb923c"
        elif mv >= 100_000_000:        # 100M – 1B
            color = "#60a5fa"
        else:                          # < 100M
            color = "#475569"
        result[(u, v)] = (color, round(w, 2))
        result[(v, u)] = (color, round(w, 2))   # undirected
    return result


def summarize_relationship_paths(
    graph: nx.Graph,
    selected_nodes: list[str],
) -> tuple[nx.Graph, list[dict], list[dict], set[tuple[str, str]]]:
    path_rows: list[dict] = []
    disconnected_rows: list[dict] = []
    path_nodes: set[str] = set(selected_nodes)
    path_edges: set[tuple[str, str]] = set()

    for source, target in combinations(selected_nodes, 2):
        try:
            path = nx.shortest_path(graph, source=source, target=target)
            labels = [graph.nodes[node]["label"] for node in path]
            node_types = [graph.nodes[node]["node_type"] for node in path]
            hop_count = max(len(path) - 1, 0)
            path_rows.append(
                {
                    "source": graph.nodes[source]["label"],
                    "target": graph.nodes[target]["label"],
                    "hop_count": hop_count,
                    "path": " -> ".join(labels),
                    "path_node_types": " -> ".join(node_types),
                }
            )
            path_nodes.update(path)
            for left, right in zip(path, path[1:]):
                path_edges.add(tuple(sorted((left, right))))
        except nx.NetworkXNoPath:
            disconnected_rows.append(
                {
                    "source": graph.nodes[source]["label"],
                    "target": graph.nodes[target]["label"],
                    "status": "No path under current filters",
                }
            )

    if not path_nodes:
        return nx.Graph(), path_rows, disconnected_rows, path_edges
    return graph.subgraph(path_nodes).copy(), path_rows, disconnected_rows, path_edges


def build_relationship_metrics(
    graph: nx.Graph,
    selected_nodes: list[str],
    include_nodes: Iterable[str] | None = None,
) -> pd.DataFrame:
    metric_names = [
        "Notion of Centrality",
        "Degree",
        "Closeness",
        "Betweenness",
        "Eigenvector",
        "Katz",
    ]
    score_maps = {metric: compute_centrality_scores(graph, metric) for metric in metric_names}
    try:
        pagerank_scores = nx.pagerank(graph)
    except (nx.NetworkXException, ModuleNotFoundError, ImportError):
        pagerank_scores = {node: 0.0 for node in graph.nodes}

    rows: list[dict] = []
    selected_set = set(selected_nodes)
    include_set = set(include_nodes) if include_nodes is not None else set(graph.nodes)
    for node, attrs in graph.nodes(data=True):
        if node not in include_set:
            continue
        rows.append(
            {
                "Node": attrs["label"],
                "Type": attrs["node_type"],
                "Selected": node in selected_set,
                "Connections": graph.degree(node),
                "Notion of Centrality": score_maps["Notion of Centrality"].get(node, 0.0),
                "Degree": score_maps["Degree"].get(node, 0.0),
                "Closeness": score_maps["Closeness"].get(node, 0.0),
                "Betweenness": score_maps["Betweenness"].get(node, 0.0),
                "Eigenvector": score_maps["Eigenvector"].get(node, 0.0),
                "Katz": score_maps["Katz"].get(node, 0.0),
                "PageRank": pagerank_scores.get(node, 0.0),
            }
        )

    return pd.DataFrame(rows)


def make_draggable_network_html(
    graph: nx.Graph,
    filtered_df: pd.DataFrame,
    focus_node: str | None = None,
    max_nodes: int = 180,
    layout_mode: str = "bipartite",
    centrality_metric: str = "Degree",
    selected_nodes: set[str] | None = None,
    emphasized_edges: set[tuple[str, str]] | None = None,
    allow_physics: bool = False,
    nx_position_scale: float = 1.0,
    edge_size_metric: str = "holding_pct",   # "holding_pct" | "market_value"
) -> str:
    if graph.number_of_nodes() == 0:
        return "<p style='color:#f1f5f9'>No graph data available.</p>"

    if graph.number_of_nodes() > max_nodes:
        company_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "company"]
        holder_nodes  = [n for n, d in graph.nodes(data=True) if d["node_type"] == "shareholder"]
        holder_nodes  = sorted(holder_nodes, key=lambda n: graph.degree(n), reverse=True)
        keep  = set(company_nodes) | set(holder_nodes[: max_nodes - len(company_nodes)])
        graph = graph.subgraph(keep).copy()

    highlighted_neighbors: set[str] = set()
    if focus_node and focus_node in graph:
        highlighted_neighbors = set(graph.neighbors(focus_node))
    selected_nodes  = selected_nodes  or set()
    emphasized_edges = emphasized_edges or set()

    # ── Community detection for shareholder colouring ────────────────────────
    comm_map = detect_communities(graph)

    company_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if d["node_type"] == "company"],
        key=lambda n: (-graph.degree(n), graph.nodes[n]["label"]),
    )
    holder_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if d["node_type"] == "shareholder"],
        key=lambda n: (-graph.degree(n), graph.nodes[n]["label"]),
    )

    net = Network(
        height="1150px",
        width="100%",
        bgcolor=DARK_BG,
        font_color=DARK_TEXT,
        directed=False,
        cdn_resources="in_line",
    )

    # ── Initial positions ────────────────────────────────────────────────────
    def y_positions(nds: list[str], top: int) -> dict[str, int]:
        total = len(nds)
        if total == 0:   return {}
        if total == 1:   return {nds[0]: 0}
        gap = (top * 2) / max(total - 1, 1)
        return {node: int(top - idx * gap) for idx, node in enumerate(nds)}

    initial_pos: dict[str, tuple[int, int]] = {}
    if layout_mode == "nx":
        scores = normalize_scores(compute_centrality_scores(graph, centrality_metric), graph.nodes)
        layout = nx.spring_layout(
            graph,
            seed=42,
            k=2.35 / math.sqrt(max(graph.number_of_nodes(), 2)),
            iterations=700,
            scale=1.55,
        )
        for node, (x, y) in layout.items():
            cs = 0.96 + ((1.0 - scores.get(node, 0.5)) * 0.62)
            initial_pos[node] = (
                int(x * 1520 * cs * nx_position_scale),
                int(y * 980  * cs * nx_position_scale),
            )
    else:
        company_y = y_positions(company_nodes, 900)
        holder_y  = y_positions(holder_nodes,  900)
        for n in company_nodes:
            initial_pos[n] = (-900, company_y[n])
        for n in holder_nodes:
            initial_pos[n] = ( 900, holder_y[n])

    # ── Add company nodes ────────────────────────────────────────────────────
    for node in company_nodes:
        attrs  = graph.nodes[node]
        degree = graph.degree(node)
        is_selected = node in selected_nodes or node == focus_node
        is_dimmed   = bool(focus_node) and node not in highlighted_neighbors and not is_selected

        if is_selected:
            fill_color   = SELECTED_COLOR
            border_color = SELECTED_COLOR
        elif is_dimmed:
            fill_color   = rgba(COMPANY_COLOR, 0.15)
            border_color = rgba(COMPANY_BORDER, 0.2)
        else:
            fill_color   = rgba(COMPANY_COLOR, 0.25)
            border_color = COMPANY_BORDER

        net.add_node(
            node,
            label=attrs["label"],
            title=(
                f"<b style='color:{COMPANY_BORDER}'>{attrs['label']}</b><br>"
                f"Type: Company<br>Connections: {degree}"
            ),
            color={"background": fill_color, "border": border_color,
                   "highlight": {"background": SELECTED_COLOR, "border": SELECTED_COLOR}},
            shape="box",
            size=min(36, 12 + degree * 1.4),
            font={"color": DARK_TEXT, "size": 14, "bold": True},
            borderWidth=2,
            x=initial_pos[node][0],
            y=initial_pos[node][1],
            physics=allow_physics,
        )

    # ── Add holder nodes ─────────────────────────────────────────────────────
    for node in holder_nodes:
        attrs    = graph.nodes[node]
        degree   = graph.degree(node)
        base_col = community_color(comm_map.get(node, 0))
        is_selected = node in selected_nodes or node == focus_node
        is_dimmed   = bool(focus_node) and node not in highlighted_neighbors and not is_selected

        if is_selected:
            fill_color   = SELECTED_COLOR
            border_color = SELECTED_COLOR
        elif is_dimmed:
            fill_color   = rgba(base_col, 0.12)
            border_color = rgba(base_col, 0.18)
        else:
            fill_color   = rgba(base_col, 0.22)
            border_color = base_col

        net.add_node(
            node,
            label=attrs["label"],
            title=(
                f"<b style='color:{base_col}'>{attrs['label']}</b><br>"
                f"Type: Shareholder<br>Connections: {degree}"
            ),
            color={"background": fill_color, "border": border_color,
                   "highlight": {"background": SELECTED_COLOR, "border": SELECTED_COLOR}},
            shape="dot",
            size=min(34, 8 + degree * 2.0),
            font={"color": DARK_TEXT, "size": 12},
            borderWidth=2,
            x=initial_pos[node][0],
            y=initial_pos[node][1],
            physics=allow_physics,
        )

    # ── Pre-compute market-value widths (once, across all edges) ─────────────
    mv_widths = compute_market_value_edge_styles(graph) if edge_size_metric == "market_value" else {}

    # ── Add edges ────────────────────────────────────────────────────────────
    for left, right, edge_attrs in graph.edges(data=True):
        edge_key   = tuple(sorted((left, right)))
        pct        = float(edge_attrs.get("weight", 0))
        mv         = float(edge_attrs.get("market_value", 0))
        shares     = float(edge_attrs.get("shares", 0))

        if edge_size_metric == "market_value" and (left, right) in mv_widths:
            base_color, base_width = mv_widths[(left, right)]
        else:
            base_color, base_width = edge_style_for_holding(pct)

        if edge_key in emphasized_edges:
            color, width = "#a78bfa", 5
        elif focus_node and focus_node in {left, right}:
            color, width = "#a78bfa", 4
        elif focus_node:
            color, width = rgba("#64748b", 0.1), 1
        else:
            color, width = base_color, base_width

        mv_label = f"฿{mv/1e9:.2f}B" if mv >= 1e9 else (f"฿{mv/1e6:.1f}M" if mv >= 1e6 else "ไม่มีราคา")
        net.add_edge(
            left, right,
            color=color,
            width=width,
            title=(
                f"<b>Holding: {pct:.2f}%</b><br>"
                f"Shares: {shares:,.0f}<br>"
                f"Market value: {mv_label}"
            ),
        )

    # ── Pyvis options (dark, improved physics) ───────────────────────────────
    options_js = f"""
        const options = {{
          "interaction": {{
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "hover": true,
            "navigationButtons": false,
            "tooltipDelay": 150
          }},
          "physics": {{
            "enabled": {"true" if allow_physics else "false"},
            "barnesHut": {{
              "gravitationalConstant": -5000,
              "centralGravity": 0.15,
              "springLength": 200,
              "springConstant": 0.025,
              "damping": 0.2,
              "avoidOverlap": 0.8
            }},
            "stabilization": {{
              "enabled": {"true" if allow_physics else "false"},
              "iterations": 250,
              "fit": true
            }},
            "minVelocity": 0.1,
            "solver": "barnesHut"
          }},
          "nodes": {{
            "font": {{ "size": 13, "face": "Inter, Arial, sans-serif" }},
            "borderWidthSelected": 3
          }},
          "edges": {{
            "smooth": {{ "enabled": true, "type": "continuous", "roundness": 0.2 }},
            "selectionWidth": 3
          }}
        }}
    """
    net.set_options(options_js)

    # ── Detail-panel data ────────────────────────────────────────────────────
    detail_rows: dict = {}
    for symbol in sorted(filtered_df["symbol"].unique()):
        cdf = filtered_df[filtered_df["symbol"] == symbol].sort_values("holding_pct", ascending=False)
        detail_rows[f"company::{symbol}"] = {
            "type": "company",
            "title": symbol,
            "color": COMPANY_BORDER,
            "subtitle": f"{cdf['shareholder_clean'].nunique()} shareholders (current filter)",
            "columns": ["shareholder_name", "holding_pct", "shares", "as_of_date"],
            "rows": cdf[["shareholder_name", "holding_pct", "shares", "as_of_date", "source_url"]].to_dict("records"),
        }

    holder_names = (
        filtered_df.groupby("shareholder_clean")["shareholder_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    for holder_clean, holder_name in holder_names.items():
        hdf = filtered_df[filtered_df["shareholder_clean"] == holder_clean].sort_values("holding_pct", ascending=False)
        node_key = f"holder::{holder_clean}"
        h_color  = community_color(comm_map.get(node_key, 0))
        detail_rows[node_key] = {
            "type": "shareholder",
            "title": holder_name,
            "color": h_color,
            "subtitle": f"{hdf['symbol'].nunique()} companies (current filter)",
            "columns": ["symbol", "holding_pct", "shares", "as_of_date"],
            "rows": hdf[["symbol", "holding_pct", "shares", "as_of_date", "source_url"]].to_dict("records"),
        }

    html = net.generate_html()
    details_json = json.dumps(detail_rows, ensure_ascii=False)

    # Build legend HTML separately (avoids nested triple-quotes in f-string)
    if edge_size_metric == "market_value":
        edge_legend_html = (
            '<div class="leg-item"><div class="leg-line" style="background:#475569"></div> &lt;100M฿</div>'
            '<div class="leg-item"><div class="leg-line" style="background:#60a5fa"></div> 100M–1B฿</div>'
            '<div class="leg-item"><div class="leg-line" style="background:#fb923c"></div> 1B–10B฿</div>'
            '<div class="leg-item"><div class="leg-line" style="background:#f87171"></div> ≥10B฿</div>'
            '<div class="leg-item" style="margin-left:8px;opacity:.6;font-size:11px">ขนาดเส้น = มูลค่าการถือ (log scale)</div>'
        )
    else:
        edge_legend_html = (
            '<div class="leg-item"><div class="leg-line" style="background:#475569"></div> &lt;2%</div>'
            '<div class="leg-item"><div class="leg-line" style="background:#60a5fa"></div> 2–5%</div>'
            '<div class="leg-item"><div class="leg-line" style="background:#fb923c"></div> 5–10%</div>'
            '<div class="leg-item"><div class="leg-line" style="background:#f87171"></div> ≥10%</div>'
            '<div class="leg-item" style="margin-left:8px;opacity:.6;font-size:11px">ขนาดเส้น = สัดส่วนการถือ</div>'
        )

    container_markup = f"""
    <style>
      *, *::before, *::after {{ box-sizing: border-box; }}
      body {{ margin: 0; font-family: Inter, Arial, sans-serif; background: {DARK_BG}; color: {DARK_TEXT}; }}
      .graph-shell {{
        display: grid;
        grid-template-columns: minmax(0, 3fr) minmax(300px, 1fr);
        gap: 14px;
        align-items: start;
      }}
      .graph-panel {{ min-width: 0; border-radius: 12px; overflow: hidden; }}
      #mynetwork {{ border-radius: 12px; }}

      /* ── Detail panel ── */
      .detail-panel {{
        border: 1px solid {DARK_BORDER};
        border-radius: 12px;
        padding: 16px;
        background: {DARK_PANEL};
        max-height: 1140px;
        overflow: auto;
        color: {DARK_TEXT};
      }}
      .detail-panel h3 {{ margin: 0 0 4px 0; font-size: 16px; font-weight: 700; }}
      .detail-panel .sub {{ margin: 0 0 12px 0; color: {DARK_SUBTEXT}; font-size: 12px; }}
      .detail-panel table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
      .detail-panel th {{
        position: sticky; top: 0;
        background: {DARK_PANEL};
        border-bottom: 1px solid {DARK_BORDER};
        padding: 6px 4px; text-align: left;
        color: {DARK_SUBTEXT}; font-weight: 600; font-size: 11px; text-transform: uppercase;
      }}
      .detail-panel td {{
        border-bottom: 1px solid {DARK_BORDER};
        padding: 6px 4px; vertical-align: top;
      }}
      .detail-panel tr:hover td {{ background: rgba(255,255,255,0.04); }}
      .detail-panel a {{ color: #60a5fa; text-decoration: none; }}
      .detail-empty {{ color: {DARK_SUBTEXT}; font-size: 13px; line-height: 1.6; }}

      /* ── Legend ── */
      .legend-bar {{
        display: flex; gap: 18px; flex-wrap: wrap;
        padding: 8px 14px;
        background: {DARK_PANEL};
        border: 1px solid {DARK_BORDER};
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 12px;
      }}
      .leg-item {{ display: flex; align-items: center; gap: 6px; }}
      .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
      .leg-box {{ width: 12px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
      .leg-line {{ width: 22px; height: 3px; border-radius: 2px; flex-shrink: 0; }}

      @media (max-width: 1100px) {{
        .graph-shell {{ grid-template-columns: 1fr; }}
        .detail-panel {{ max-height: 420px; }}
      }}
    </style>

    <div class="legend-bar">
      <div class="leg-item"><div class="leg-box" style="background:{rgba(COMPANY_COLOR,0.25)};border:2px solid {COMPANY_BORDER}"></div> Company (SET50)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#60a5fa"></div> Shareholders (coloured by community)</div>
      <div class="leg-item"><div class="leg-dot" style="background:{SELECTED_COLOR}"></div> Selected / Focus</div>
      {edge_legend_html}
    </div>

    <div class="graph-shell">
      <div class="graph-panel">
        <div id="mynetwork"></div>
      </div>
      <div class="detail-panel" id="detail-panel">
        <h3>Node Details</h3>
        <p class="detail-empty">Click any node to see its connections and shareholding data here.</p>
      </div>
    </div>
    """
    html = re.sub(
        r"<body>\s*<div class=\"card\"[^>]*>\s*<div id=\"mynetwork\"></div>\s*</div>",
        f"<body>{container_markup}",
        html, count=1, flags=re.S,
    )

    interaction_script = """
    <script type="text/javascript">
    (function() {
      if (typeof network === "undefined") return;
      const detailData  = __DETAILS_JSON__;
      const detailPanel = document.getElementById("detail-panel");
      const defaultNodeStyles = {};
      const defaultEdgeStyles = {};

      function snapshotDefaults() {
        nodes.get().forEach(n  => { defaultNodeStyles[n.id] = { color: n.color, size: n.size }; });
        edges.get().forEach(e  => { defaultEdgeStyles[e.id] = { color: e.color, width: e.width }; });
      }

      function resetStyles() {
        nodes.update(Object.entries(defaultNodeStyles).map(([id, s]) => ({ id, color: s.color, size: s.size })));
        edges.update(Object.entries(defaultEdgeStyles).map(([id, s]) => ({ id, color: s.color, width: s.width })));
        detailPanel.innerHTML = '<h3>Node Details</h3><p class="detail-empty">Click any node to see its connections and shareholding data here.</p>';
      }

      function fmt(key, val) {
        if (key === "holding_pct") return (+(val||0)).toFixed(2) + " %";
        if (key === "shares")      return Number(val||0).toLocaleString();
        return val ?? "";
      }

      function renderDetails(nodeId) {
        const data = detailData[nodeId];
        if (!data) { detailPanel.innerHTML = '<h3>Node Details</h3><p class="detail-empty">No detail found.</p>'; return; }
        const thead = data.columns.map(c => `<th>${c}</th>`).join("") + "<th></th>";
        const tbody = data.rows.map(row => {
          const cells = data.columns.map(c => `<td>${fmt(c, row[c])}</td>`).join("");
          const src   = row.source_url ? `<td><a href="${row.source_url}" target="_blank">↗</a></td>` : "<td></td>";
          return `<tr>${cells}${src}</tr>`;
        }).join("");
        detailPanel.innerHTML = `
          <h3 style="color:${data.color}">${data.title}</h3>
          <p class="sub">${data.subtitle}</p>
          <table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
      }

      function highlightNode(nodeId) {
        const connEdges = network.getConnectedEdges(nodeId);
        const connNodes = new Set(network.getConnectedNodes(nodeId));
        nodes.update(nodes.get().map(n => {
          if (n.id === nodeId)          return { id: n.id, color: { background: "#a78bfa", border: "#a78bfa" }, size: Math.max((n.size||10)*1.3, 28) };
          if (connNodes.has(n.id))      return { id: n.id, color: defaultNodeStyles[n.id].color, size: defaultNodeStyles[n.id].size };
          return { id: n.id, color: "rgba(100,116,139,0.15)", size: defaultNodeStyles[n.id].size };
        }));
        edges.update(edges.get().map(e => {
          if (connEdges.includes(e.id)) return { id: e.id, color: "#a78bfa", width: 4 };
          return { id: e.id, color: "rgba(100,116,139,0.08)", width: 1 };
        }));
        renderDetails(nodeId);
      }

      snapshotDefaults();
      network.on("click", params => {
        if (params.nodes?.length) highlightNode(params.nodes[0]);
        else resetStyles();
      });
    })();
    </script>
    """
    interaction_script = interaction_script.replace("__DETAILS_JSON__", details_json)
    return html.replace("</body>", interaction_script + "</body>")


def compute_holder_metrics(df: pd.DataFrame) -> pd.DataFrame:
    name_map = (
        df.groupby("shareholder_clean")["shareholder_name"]
        .agg(lambda series: series.value_counts().idxmax())
        .rename("shareholder_name")
    )
    summary = (
        df.groupby("shareholder_clean", as_index=False)
        .agg(
            company_count=("symbol", "nunique"),
            total_edges=("symbol", "count"),
            total_holding_pct=("holding_pct", "sum"),
            total_shares=("shares", "sum"),
        )
        .merge(name_map, on="shareholder_clean", how="left")
        .sort_values(["company_count", "total_holding_pct"], ascending=[False, False])
    )
    return summary[
        [
            "shareholder_clean",
            "shareholder_name",
            "company_count",
            "total_edges",
            "total_holding_pct",
            "total_shares",
        ]
    ]


def compute_company_projection(df: pd.DataFrame) -> pd.DataFrame:
    pairs = df.merge(df, on="shareholder_clean", suffixes=("_left", "_right"))
    pairs = pairs[pairs["symbol_left"] < pairs["symbol_right"]].copy()
    if pairs.empty:
        return pairs
    pairs["pair_strength"] = pairs[["holding_pct_left", "holding_pct_right"]].min(axis=1)
    projected = (
        pairs.groupby(["symbol_left", "symbol_right"], as_index=False)
        .agg(
            shared_holders=("shareholder_clean", "nunique"),
            min_shared_pct=("pair_strength", "min"),
            mean_shared_pct=("pair_strength", "mean"),
        )
        .sort_values(["shared_holders", "mean_shared_pct"], ascending=[False, False])
    )
    return projected


def make_network_figure_2d(
    graph: nx.Graph,
    max_nodes: int = 140,
    show_labels: bool = True,
    max_holder_labels: int = 20,
    focus_node: str | None = None,
) -> go.Figure:
    if graph.number_of_nodes() == 0:
        return go.Figure()

    if graph.number_of_nodes() > max_nodes:
        company_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "company"]
        holder_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == "shareholder"]
        holder_nodes = sorted(holder_nodes, key=lambda n: graph.degree(n), reverse=True)
        keep = set(company_nodes) | set(holder_nodes[: max_nodes - len(company_nodes)])
        graph = graph.subgraph(keep).copy()

    def spaced_positions(nodes: list[str], x_value: float) -> dict[str, tuple[float, float]]:
        total = len(nodes)
        if total == 0:
            return {}
        if total == 1:
            return {nodes[0]: (x_value, 0.0)}
        gap = 3.4 / (total - 1)
        start = 1.7
        return {
            node: (x_value, start - idx * gap)
            for idx, node in enumerate(nodes)
        }

    company_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if d["node_type"] == "company"],
        key=lambda n: (-graph.degree(n), graph.nodes[n]["label"]),
    )
    holder_nodes = sorted(
        [n for n, d in graph.nodes(data=True) if d["node_type"] == "shareholder"],
        key=lambda n: (-graph.degree(n), graph.nodes[n]["label"]),
    )

    pos: dict[str, tuple[float, float]] = {}
    pos.update(spaced_positions(company_nodes, -2.1))
    pos.update(spaced_positions(holder_nodes, 2.1))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for left, right in graph.edges():
        x0, y0 = pos[left]
        x1, y1 = pos[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    highlighted_neighbors: set[str] = set()
    if focus_node and focus_node in graph:
        highlighted_neighbors = set(graph.neighbors(focus_node))

    comm_map = detect_communities(graph)

    edge_traces: list[go.Scatter] = []
    if focus_node and focus_node in graph:
        dim_x: list[float] = []
        dim_y: list[float] = []
        hi_x: list[float] = []
        hi_y: list[float] = []
        for left, right in graph.edges():
            x0, y0 = pos[left]
            x1, y1 = pos[right]
            target = hi_x if focus_node in {left, right} else dim_x
            target.extend([x0, x1, None])
            (hi_y if focus_node in {left, right} else dim_y).extend([y0, y1, None])
        edge_traces.append(
            go.Scatter(
                x=dim_x,
                y=dim_y,
                mode="lines",
                line=dict(width=0.6, color="rgba(156,163,175,0.18)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        edge_traces.append(
            go.Scatter(
                x=hi_x,
                y=hi_y,
                mode="lines",
                line=dict(width=4, color="#2563EB"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        edge_traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.7, color="#9AA5B1"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_size: list[float] = []
    node_color: list[str] = []
    node_label_text: list[str] = []
    node_customdata: list[list[str]] = []

    holder_rank = sorted(
        [n for n, d in graph.nodes(data=True) if d["node_type"] == "shareholder"],
        key=lambda n: graph.degree(n),
        reverse=True,
    )
    labeled_holders = set(holder_rank[:max_holder_labels])

    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        degree = graph.degree(node)
        label = attrs["label"]
        node_x.append(x)
        node_y.append(y)
        node_size.append(min(38, 10 + degree * 1.8))
        node_text.append(f"{label}<br>type={attrs['node_type']}<br>degree={degree}")
        base_color = COMPANY_COLOR if attrs["node_type"] == "company" else community_color(comm_map.get(node, 0))
        if focus_node and node == focus_node:
            node_color.append(rgba(SELECTED_COLOR, 1.0))
        elif focus_node and node in highlighted_neighbors:
            node_color.append(rgba(base_color, 1.0))
        elif focus_node:
            node_color.append(rgba(base_color, 0.15))
        else:
            node_color.append(rgba(base_color, 0.85))
        should_label = attrs["node_type"] == "company" or node in labeled_holders
        node_label_text.append(label if show_labels and should_label else "")
        node_customdata.append([node, attrs["node_type"], label, str(degree)])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="white"),
        ),
        text=node_label_text,
        customdata=node_customdata,
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>"
            "type=%{customdata[1]}<br>"
            "degree=%{customdata[3]}<extra></extra>"
        ),
        textposition=[
            "middle left" if graph.nodes[node]["node_type"] == "company" else "middle right"
            for node in graph.nodes()
        ] if show_labels else None,
        textfont=dict(size=10, color=DARK_TEXT),
        texttemplate="%{text}",
        showlegend=False,
    )

    figure = go.Figure(data=[*edge_traces, node_trace])
    figure.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", range=[-2.8, 2.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="", range=[-1.95, 1.95]),
        height=1100,
        showlegend=False,
        font=dict(color=DARK_TEXT),
    )
    return figure


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def render_focus_details(node_id: str | None, filtered_df: pd.DataFrame) -> None:
    st.subheader("Selected Node Details")
    if not node_id:
        st.info("Use the detail panel next to the graph by clicking a node inside the network.")
        return

    node_type, raw_id = node_id.split("::", 1)
    if node_type == "company":
        company_df = filtered_df[filtered_df["symbol"] == raw_id].sort_values("holding_pct", ascending=False)
        st.markdown(f"**Company:** `{raw_id}`")
        st.write(
            f"This company is connected to **{company_df['shareholder_clean'].nunique()}** shareholders "
            f"under the current filters."
        )
        st.dataframe(company_df, use_container_width=True, hide_index=True)
        return

    holder_df = filtered_df[filtered_df["shareholder_clean"] == raw_id].sort_values("holding_pct", ascending=False)
    holder_name = holder_df["shareholder_name"].value_counts().idxmax()
    st.markdown(f"**Shareholder:** `{holder_name}`")
    st.write(
        f"This shareholder is connected to **{holder_df['symbol'].nunique()}** companies "
        f"under the current filters."
    )
    st.dataframe(
        holder_df[
            ["symbol", "holding_pct", "shares", "as_of_date", "is_nominee", "source_url"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def filter_dataframe(
    df: pd.DataFrame,
    min_pct: float,
    selected_companies: Iterable[str],
    excluded_companies: Iterable[str],
    excluded_holders_clean: Iterable[str],
    exclude_nominees: bool,
    only_cross_holders: bool,
) -> pd.DataFrame:
    filtered = df[df["holding_pct"] >= min_pct].copy()
    if selected_companies:
        filtered = filtered[filtered["symbol"].isin(selected_companies)]
    if excluded_companies:
        filtered = filtered[~filtered["symbol"].isin(excluded_companies)]
    if excluded_holders_clean:
        filtered = filtered[~filtered["shareholder_clean"].isin(excluded_holders_clean)]
    if exclude_nominees:
        filtered = filtered[~filtered["is_nominee"]]
    if only_cross_holders and not filtered.empty:
        counts = filtered.groupby("shareholder_clean")["symbol"].nunique()
        keep_holders = counts[counts > 1].index
        filtered = filtered[filtered["shareholder_clean"].isin(keep_holders)]
    return filtered


def render_macro_network_page() -> None:
    """Render the standalone macro-factor → SET Index network page."""
    st.title("ปัจจัยที่ส่งผลต่อตลาดหุ้นไทย")
    st.caption(
        "กราฟแสดงความสัมพันธ์ระหว่างปัจจัยมหภาคและตลาดหุ้นไทย (SET Index) "
        "ลากโหนดได้ · hover เพื่อดูคำอธิบาย · สีเส้น = ทิศทาง (+/−/mixed)"
    )

    macro_html = """<!DOCTYPE html>
<html>
<head>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#111827;color:#f1f5f9;font-family:system-ui,sans-serif;overflow:hidden}
  canvas{display:block}
  #tt{position:fixed;background:#1f2937;border:1px solid #374151;border-radius:8px;
      padding:10px 14px;font-size:13px;pointer-events:none;display:none;max-width:230px;
      z-index:100;line-height:1.5}
  #tt .tn{font-weight:700;font-size:14px;margin-bottom:3px}
  #tt .tc{font-size:11px;opacity:.6;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em}
  #leg{position:fixed;bottom:14px;left:14px;background:#1f2937;border:1px solid #374151;
       border-radius:8px;padding:10px 14px;font-size:12px;z-index:100}
  #leg .lt{font-weight:600;margin-bottom:7px;opacity:.6;font-size:11px;text-transform:uppercase}
  .li{display:flex;align-items:center;gap:7px;margin-bottom:4px}
  .ld{width:10px;height:10px;border-radius:50%;flex-shrink:0}
  .lb{width:12px;height:10px;border-radius:2px;flex-shrink:0}
  #leg2{position:fixed;bottom:14px;right:14px;background:#1f2937;border:1px solid #374151;
        border-radius:8px;padding:10px 14px;font-size:12px;z-index:100}
  .ll{width:26px;height:3px;border-radius:2px;flex-shrink:0}
  #hint{position:fixed;top:12px;left:50%;transform:translateX(-50%);
        background:#1f2937;border:1px solid #374151;border-radius:20px;
        padding:5px 14px;font-size:12px;opacity:.55;pointer-events:none;white-space:nowrap}
</style>
</head>
<body>
<div id="hint">hover บนโหนดเพื่อดูรายละเอียด · ลาก nodes ได้</div>
<canvas id="c"></canvas>
<div id="tt"></div>
<div id="leg">
  <div class="lt">หมวดหมู่</div>
  <div class="li"><div class="ld" style="background:#f59e42"></div>ตลาดต่างประเทศ</div>
  <div class="li"><div class="ld" style="background:#60a5fa"></div>มหภาคในประเทศ</div>
  <div class="li"><div class="ld" style="background:#34d399"></div>ปัจจัยเฉพาะไทย</div>
  <div class="li"><div class="ld" style="background:#a78bfa"></div>สินค้าโภคภัณฑ์</div>
  <div class="li"><div class="ld" style="background:#fb7185"></div>Sentiment</div>
  <div class="li"><div class="lb" style="background:#fbbf24;border:2px solid #fff"></div>SET Index</div>
</div>
<div id="leg2">
  <div class="lt">ทิศทาง</div>
  <div class="li"><div class="ll" style="background:#34d399"></div>บวก (+)</div>
  <div class="li"><div class="ll" style="background:#fb7185"></div>ลบ (−)</div>
  <div class="li"><div class="ll" style="background:#94a3b8"></div>Mixed (±)</div>
</div>
<script>
const C=document.getElementById('c'),ctx=C.getContext('2d');
let W,H;
function resize(){W=C.width=window.innerWidth;H=C.height=window.innerHeight}
resize();window.addEventListener('resize',resize);

const cats={foreign:{color:'#f59e42'},macro:{color:'#60a5fa'},thai:{color:'#34d399'},
            commodity:{color:'#a78bfa'},sentiment:{color:'#fb7185'},center:{color:'#fbbf24'}};

const nodes=[
  {id:'SET',label:'SET Index',cat:'center',r:32,desc:'ดัชนีตลาดหลักทรัพย์ไทย\\nเป้าหมายที่เราต้องการวิเคราะห์',x:0,y:0,vx:0,vy:0},
  {id:'SP500',label:'S&P 500',cat:'foreign',r:20,desc:'ตลาดสหรัฐ — correlation สูงช่วง risk-off\\nมักนำ SET ราว 1 วัน',x:0,y:0,vx:0,vy:0},
  {id:'HSI',label:'Hang Seng',cat:'foreign',r:18,desc:'ตลาดฮ่องกง/จีน\\nส่งผลผ่านท่องเที่ยวและส่งออก',x:0,y:0,vx:0,vy:0},
  {id:'CSI300',label:'CSI 300',cat:'foreign',r:16,desc:'ตลาดหุ้นจีน A-shares\\nความเชื่อมั่นต่อเศรษฐกิจจีน',x:0,y:0,vx:0,vy:0},
  {id:'DXY',label:'Dollar\\nIndex (DXY)',cat:'foreign',r:18,desc:'ความแข็งของดอลลาร์\\nDXY แข็ง → เงินไหลออก EM',x:0,y:0,vx:0,vy:0},
  {id:'VIX',label:'VIX\\n(Fear)',cat:'sentiment',r:19,desc:'ดัชนีความกลัวตลาดโลก\\nVIX สูง → เงินออกจาก EM',x:0,y:0,vx:0,vy:0},
  {id:'USDTHB',label:'USD/THB',cat:'macro',r:19,desc:'ค่าเงินบาทต่อดอลลาร์\\nบาทอ่อน → กดดัน import cost',x:0,y:0,vx:0,vy:0},
  {id:'BOT',label:'ดอกเบี้ย\\nBoT',cat:'macro',r:17,desc:'อัตราดอกเบี้ยนโยบาย ธปท.\\nขึ้นดอก → ต้นทุนทุนสูง',x:0,y:0,vx:0,vy:0},
  {id:'CPI',label:'เงินเฟ้อ\\n(CPI)',cat:'macro',r:16,desc:'ดัชนีราคาผู้บริโภค\\nเงินเฟ้อสูง → BoT ขึ้นดอก',x:0,y:0,vx:0,vy:0},
  {id:'GDP',label:'GDP Growth',cat:'macro',r:17,desc:'อัตราการเติบโต GDP ไทย\\nGDP ดี → กำไรบริษัทสูง',x:0,y:0,vx:0,vy:0},
  {id:'TOURIST',label:'นักท่องเที่ยว\\nต่างชาติ',cat:'thai',r:19,desc:'จำนวนนักท่องเที่ยวต่างชาติ\\nส่งผลตรงต่อ tourism stocks',x:0,y:0,vx:0,vy:0},
  {id:'FUND',label:'Fund Flow\\nต่างชาติ',cat:'thai',r:21,desc:'เงิน net buy/sell ต่างชาติ\\nตัวที่ lead/lag SET ได้ชัด',x:0,y:0,vx:0,vy:0},
  {id:'PTT',label:'PTT Weight',cat:'thai',r:16,desc:'PTT + กลุ่มมี weight สูงใน SET\\nราคาน้ำมันกระทบ SET ผ่าน PTT',x:0,y:0,vx:0,vy:0},
  {id:'EXPORT',label:'มูลค่า\\nส่งออก',cat:'thai',r:15,desc:'ปริมาณส่งออกรายเดือน\\nกระทบ manufacturing/agri stocks',x:0,y:0,vx:0,vy:0},
  {id:'OIL',label:'น้ำมัน\\n(Brent)',cat:'commodity',r:19,desc:'ราคาน้ำมันดิบ Brent\\nไทยนำเข้าสุทธิ — mixed กับ SET',x:0,y:0,vx:0,vy:0},
  {id:'GOLD',label:'ราคาทอง',cat:'commodity',r:17,desc:'ราคาทองคำโลก\\nไทยเป็นผู้ส่งออกทอง',x:0,y:0,vx:0,vy:0},
  {id:'RUBBER',label:'ราคายาง',cat:'commodity',r:14,desc:'ราคายางพาราโลก\\nกระทบ agri sector',x:0,y:0,vx:0,vy:0},
  {id:'CCI',label:'ความเชื่อมั่น\\nผู้บริโภค',cat:'sentiment',r:15,desc:'Consumer Confidence Index\\nสะท้อน sentiment ในประเทศ',x:0,y:0,vx:0,vy:0},
  {id:'GTREND',label:'Google\\nTrends',cat:'sentiment',r:13,desc:'ปริมาณการค้นหา "หุ้น"\\nproxy ของ retail investor sentiment',x:0,y:0,vx:0,vy:0},
];

const edges=[
  ['SP500','SET','pos',3],['HSI','SET','pos',2],['CSI300','SET','pos',2],
  ['DXY','SET','neg',3],['VIX','SET','neg',3],['USDTHB','SET','neg',2],
  ['BOT','SET','neg',2],['CPI','SET','neg',2],['GDP','SET','pos',2],
  ['TOURIST','SET','pos',2],['FUND','SET','pos',3],['PTT','SET','mixed',1],
  ['EXPORT','SET','pos',2],['OIL','SET','mixed',2],['GOLD','SET','pos',1],
  ['RUBBER','SET','pos',1],['CCI','SET','pos',2],['GTREND','SET','pos',1],
  ['DXY','USDTHB','pos',2],['VIX','DXY','pos',2],['SP500','VIX','neg',2],
  ['OIL','CPI','pos',2],['CPI','BOT','pos',2],['OIL','PTT','pos',2],
  ['HSI','CSI300','pos',2],['FUND','USDTHB','neg',1],['GDP','EXPORT','pos',1],
  ['GOLD','DXY','neg',2],['CCI','TOURIST','pos',1],
];

const nm={};nodes.forEach(n=>nm[n.id]=n);

function initPos(){
  const cx=0,cy=0;
  const foreign=['SP500','HSI','CSI300','DXY'];
  const macro=['USDTHB','BOT','CPI','GDP'];
  const thai=['TOURIST','FUND','PTT','EXPORT'];
  const commodity=['OIL','GOLD','RUBBER'];
  const sentiment=['VIX','CCI','GTREND'];
  function place(ids,sa,sp,r){
    ids.forEach((id,i)=>{
      const a=sa+(i-(ids.length-1)/2)*sp;
      nm[id].x=cx+Math.cos(a)*r;nm[id].y=cy+Math.sin(a)*r;
    });
  }
  place(foreign,Math.PI*1.1,.38,220);
  place(macro,Math.PI*.55,.4,215);
  place(thai,Math.PI*.0,.4,220);
  place(commodity,Math.PI*1.65,.38,215);
  place(sentiment,Math.PI*1.35,.38,215);
}
initPos();

function sim(){
  nodes.forEach(a=>{
    if(a.id==='SET')return;
    let fx=-a.x*.015,fy=-a.y*.015;
    nodes.forEach(b=>{
      if(a===b)return;
      const dx=a.x-b.x,dy=a.y-b.y,d=Math.sqrt(dx*dx+dy*dy)||1;
      const f=3200/(d*d);fx+=(dx/d)*f;fy+=(dy/d)*f;
    });
    edges.forEach(e=>{
      const other=e[0]===a.id?nm[e[1]]:e[1]===a.id?nm[e[0]]:null;
      if(!other)return;
      const dx=other.x-a.x,dy=other.y-a.y,d=Math.sqrt(dx*dx+dy*dy)||1;
      const ideal=(a.id==='SET'||other.id==='SET')?200:160;
      const f=(d-ideal)*.03;fx+=(dx/d)*f;fy+=(dy/d)*f;
    });
    a.vx=(a.vx+fx)*.8;a.vy=(a.vy+fy)*.8;a.x+=a.vx;a.y+=a.vy;
  });
}

let drag=null,ox=0,oy=0,hov=null,iter=0;

function nodeAt(mx,my){
  const wx=mx-W/2,wy=my-H/2;
  for(let i=nodes.length-1;i>=0;i--){
    const n=nodes[i];
    if(Math.hypot(n.x-wx,n.y-wy)<n.r+6)return n;
  }return null;
}

C.addEventListener('mousedown',e=>{const n=nodeAt(e.clientX,e.clientY);if(n){drag=n;ox=e.clientX-n.x-W/2;oy=e.clientY-n.y-H/2;}});
C.addEventListener('mousemove',e=>{
  if(drag){drag.x=e.clientX-ox-W/2;drag.y=e.clientY-oy-H/2;drag.vx=0;drag.vy=0;}
  hov=nodeAt(e.clientX,e.clientY);
  const tt=document.getElementById('tt');
  if(hov){
    tt.style.display='block';
    tt.style.left=(e.clientX+16)+'px';tt.style.top=(e.clientY-10)+'px';
    const cat=cats[hov.cat];
    tt.innerHTML='<div class="tn">'+hov.label.replace('\\n',' ')+'</div>'
      +'<div class="tc" style="color:'+cat.color+'">'+hov.cat+'</div>'
      +'<div>'+hov.desc.replace('\\n','<br>')+'</div>';
  }else{tt.style.display='none';}
});
C.addEventListener('mouseup',()=>drag=null);

function ecol(d){return d==='pos'?'#34d399':d==='neg'?'#fb7185':'#94a3b8';}

function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.save();ctx.translate(W/2,H/2);

  edges.forEach(([a,b,dir,str])=>{
    const na=nm[a],nb=nm[b];if(!na||!nb)return;
    const hi=hov&&(hov.id===a||hov.id===b);
    ctx.beginPath();ctx.moveTo(na.x,na.y);ctx.lineTo(nb.x,nb.y);
    ctx.strokeStyle=ecol(dir);ctx.globalAlpha=hi?.9:.2;ctx.lineWidth=hi?str*1.6:str*.8;ctx.stroke();ctx.globalAlpha=1;

    if(nb.id==='SET'||na.id==='SET'){
      const tgt=nb.id==='SET'?nb:na,src=nb.id==='SET'?na:nb;
      const dx=tgt.x-src.x,dy=tgt.y-src.y,d=Math.hypot(dx,dy);
      const ux=dx/d,uy=dy/d;
      const ax=tgt.x-ux*(tgt.r+4),ay=tgt.y-uy*(tgt.r+4);
      ctx.beginPath();ctx.moveTo(ax,ay);
      ctx.lineTo(ax-ux*10-uy*5,ay-uy*10+ux*5);
      ctx.lineTo(ax-ux*10+uy*5,ay-uy*10-ux*5);
      ctx.closePath();ctx.fillStyle=ecol(dir);ctx.globalAlpha=hi?.9:.35;ctx.fill();ctx.globalAlpha=1;
    }
  });

  nodes.forEach(n=>{
    const cat=cats[n.cat],hi=hov&&hov.id===n.id;
    if(hi){ctx.beginPath();ctx.arc(n.x,n.y,n.r+8,0,Math.PI*2);ctx.fillStyle=cat.color+'33';ctx.fill();}
    ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,Math.PI*2);
    ctx.fillStyle=cat.color+(n.id==='SET'?'ff':'22');
    ctx.strokeStyle=cat.color;ctx.lineWidth=n.id==='SET'?3:hi?2.5:1.5;
    ctx.fill();ctx.stroke();

    const lines=n.label.split('\\n');
    ctx.fillStyle=n.id==='SET'?'#0f1117':hi?cat.color:'#f1f5f9';
    ctx.textAlign='center';ctx.textBaseline='middle';
    const fs=n.id==='SET'?13:Math.min(11,n.r*.55);
    ctx.font=(n.id==='SET'?'700 ':'500 ')+fs+'px system-ui';
    if(lines.length===1){ctx.fillText(lines[0],n.x,n.y);}
    else{
      ctx.fillText(lines[0],n.x,n.y-fs*.6);
      ctx.font='400 '+(fs*.88)+'px system-ui';
      ctx.fillText(lines[1],n.x,n.y+fs*.7);
    }
  });
  ctx.restore();
}

function loop(){if(iter<200){sim();iter++;}draw();requestAnimationFrame(loop);}
loop();
</script>
</body>
</html>"""

    components.html(macro_html, height=720, scrolling=False)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ตัวแปรและทิศทางความสัมพันธ์")
        factor_data = [
            {"ตัวแปร": "S&P 500", "หมวด": "ตลาดต่างประเทศ", "ทิศทาง": "บวก (+)", "ความแรง": "สูง", "หมายเหตุ": "นำ SET ~1 วัน"},
            {"ตัวแปร": "Hang Seng / CSI 300", "หมวด": "ตลาดต่างประเทศ", "ทิศทาง": "บวก (+)", "ความแรง": "กลาง", "หมายเหตุ": "ผ่านท่องเที่ยว + ส่งออก"},
            {"ตัวแปร": "Dollar Index (DXY)", "หมวด": "ตลาดต่างประเทศ", "ทิศทาง": "ลบ (−)", "ความแรง": "สูง", "หมายเหตุ": "DXY แข็ง = เงินออก EM"},
            {"ตัวแปร": "VIX", "หมวด": "Sentiment", "ทิศทาง": "ลบ (−)", "ความแรง": "สูง", "หมายเหตุ": "Fear index กดดัน EM"},
            {"ตัวแปร": "USD/THB", "หมวด": "มหภาคในประเทศ", "ทิศทาง": "ลบ (−)", "ความแรง": "กลาง", "หมายเหตุ": "บาทอ่อน = ต้นทุนนำเข้าสูง"},
            {"ตัวแปร": "ดอกเบี้ย BoT", "หมวด": "มหภาคในประเทศ", "ทิศทาง": "ลบ (−)", "ความแรง": "กลาง", "หมายเหตุ": "ขึ้นดอก = กด valuation"},
            {"ตัวแปร": "CPI (เงินเฟ้อ)", "หมวด": "มหภาคในประเทศ", "ทิศทาง": "ลบ (−)", "ความแรง": "กลาง", "หมายเหตุ": "ผ่าน BoT"},
            {"ตัวแปร": "GDP Growth", "หมวด": "มหภาคในประเทศ", "ทิศทาง": "บวก (+)", "ความแรง": "กลาง", "หมายเหตุ": "กำไรบริษัทสูง"},
            {"ตัวแปร": "Fund Flow ต่างชาติ", "หมวด": "ปัจจัยเฉพาะไทย", "ทิศทาง": "บวก (+)", "ความแรง": "สูง", "หมายเหตุ": "Lead/lag SET ได้ชัด"},
            {"ตัวแปร": "นักท่องเที่ยวต่างชาติ", "หมวด": "ปัจจัยเฉพาะไทย", "ทิศทาง": "บวก (+)", "ความแรง": "กลาง", "หมายเหตุ": "กระทบ tourism stocks"},
            {"ตัวแปร": "น้ำมัน (Brent)", "หมวด": "สินค้าโภคภัณฑ์", "ทิศทาง": "Mixed (±)", "ความแรง": "กลาง", "หมายเหตุ": "ลบผ่าน import + บวกผ่าน PTT"},
            {"ตัวแปร": "ราคาทอง", "หมวด": "สินค้าโภคภัณฑ์", "ทิศทาง": "บวก (+)", "ความแรง": "ต่ำ", "หมายเหตุ": "ไทยส่งออกทอง"},
            {"ตัวแปร": "Google Trends 'หุ้น'", "หมวด": "Sentiment", "ทิศทาง": "บวก (+)", "ความแรง": "ต่ำ", "หมายเหตุ": "Retail sentiment proxy"},
        ]
        st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("แนวทางการวิเคราะห์ต่อ")
        st.markdown("""
**Time-series / Granger causality**
ทดสอบว่าตัวแปรไหน *นำ* SET จริง (lag 1–5 วัน)

**Correlation matrix**
คำนวณ Pearson / Spearman รายคู่ ดู rolling 90/180 วัน

**Event study**
เลือก events (ขึ้นดอก, เลือกตั้ง, วิกฤต) แล้วดู abnormal return

**ML feature importance**
ใช้ Random Forest / XGBoost rank ตัวแปรที่ทำนาย SET return ได้ดีที่สุด

**Regime analysis**
แบ่ง bull/bear market แล้วดูว่า correlation เปลี่ยนไปอย่างไร
        """)


def render_sidebar(df: pd.DataFrame, can_refresh: bool) -> dict:
    companies = sorted(df["symbol"].dropna().unique().tolist()) if not df.empty else []
    holder_options = []
    holder_label_to_clean: dict[str, str] = {}
    if not df.empty:
        holder_name_map = (
            df.groupby("shareholder_clean")["shareholder_name"]
            .agg(lambda series: series.value_counts().idxmax())
            .sort_values()
        )
        holder_options = holder_name_map.tolist()
        holder_label_to_clean = {label: clean for clean, label in holder_name_map.items()}
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "เลือกหน้า",
            options=["SET50 Shareholder Network", "Macro Factor Network"],
            index=0,
            label_visibility="collapsed",
        )
        st.divider()
        if page == "Macro Factor Network":
            return {
                "page": page,
                "min_pct": 1.0, "selected_companies": [], "excluded_companies": [],
                "excluded_holders_clean": [], "exclude_nominees": True,
                "only_cross_holders": True, "layout_mode": "NX Graph Layout",
                "centrality_metric": "Degree", "show_labels": True,
                "max_holder_labels": 20, "focus_holder_name": "None",
                "clear_selected_node": False, "force_refresh": False,
                "sample_limit": 50, "edge_size_metric": "holding_pct",
                "refresh_prices": False,
            }
        st.header("Filters")
        min_pct = st.slider("Min holding %", 0.0, 20.0, 1.0, 0.1)
        selected_companies = st.multiselect("Companies", companies, default=companies)
        excluded_companies = st.multiselect("Exclude companies", companies, default=[])
        excluded_holders = st.multiselect("Exclude shareholders", holder_options, default=[])
        exclude_nominees = st.toggle("Hide nominee / NVDR holders", value=True)
        only_cross_holders = st.toggle("Show only holders linked to >1 company", value=True)
        st.header("Graph")
        layout_mode = st.radio(
            "Initial layout",
            options=["Left/Right Bipartite", "NX Graph Layout"],
            index=1,
        )
        st.subheader("Edge size")
        edge_size_metric = st.radio(
            "ขนาดเส้นแทน",
            options=["สัดส่วนการถือ (%)", "มูลค่าการถือ (shares × ราคา)"],
            index=0,
        )
        refresh_prices = st.button(
            "🔄 อัปเดตราคาหุ้น (yfinance)",
            help="ดึงราคาปิดล่าสุดจาก yfinance (.BK) ใช้สำหรับคำนวณมูลค่าการถือ",
        )
        centrality_metric = "Degree"
        show_labels = st.toggle("Show labels", value=True)
        max_holder_labels = st.slider("Top holder labels", 5, 40, 20, 1)
        focus_holder_name = st.selectbox("Manual focus shareholder", ["None", *holder_options], index=0)
        clear_selected_node = st.button("Clear focus")
        sample_limit = st.number_input(
            "Refresh limit (local only)",
            min_value=5, max_value=50, value=50, step=5,
            disabled=not can_refresh,
        )
        force_refresh = st.button("Refresh from SET", type="primary", disabled=not can_refresh)
        if not can_refresh:
            st.caption("Live refresh is disabled on this deployment. The app reads cached CSV files only.")
    return {
        "min_pct": min_pct,
        "selected_companies": selected_companies,
        "excluded_companies": excluded_companies,
        "excluded_holders_clean": [
            holder_label_to_clean[label]
            for label in excluded_holders
            if label in holder_label_to_clean
        ],
        "exclude_nominees": exclude_nominees,
        "only_cross_holders": only_cross_holders,
        "layout_mode": layout_mode,
        "centrality_metric": centrality_metric,
        "show_labels": show_labels,
        "max_holder_labels": max_holder_labels,
        "focus_holder_name": focus_holder_name,
        "clear_selected_node": clear_selected_node,
        "force_refresh": force_refresh,
        "sample_limit": sample_limit,
        "edge_size_metric": "market_value" if "มูลค่า" in edge_size_metric else "holding_pct",
        "refresh_prices": refresh_prices,
        "page": page,
    }


def main() -> None:
    st.set_page_config(page_title="SET50 Network", layout="wide")

    ensure_cache_dir()
    cloud_mode = is_running_on_streamlit_cloud()
    can_refresh = (not cloud_mode) and is_live_refresh_enabled()

    cached_df, meta_df = load_cached_data()
    controls = render_sidebar(cached_df, can_refresh=can_refresh)

    # ── Route pages ───────────────────────────────────────────────────────────
    if controls["page"] == "Macro Factor Network":
        render_macro_network_page()
        return

    # ── SET50 Shareholder Network page ────────────────────────────────────────
    st.title("SET50 Shareholder Network Analysis")
    st.caption("Cache-first dashboard for major shareholders of SET50 companies.")

    if controls["force_refresh"]:
        with st.spinner("Scraping SET50 constituents and major shareholders from SET..."):
            cached_df, meta_df = refresh_data(limit=int(controls["sample_limit"]))

    # ── Stock prices ──────────────────────────────────────────────────────────
    prices = load_price_cache()
    if controls["refresh_prices"] or (not prices and controls["edge_size_metric"] == "market_value"):
        with st.spinner("กำลังดึงราคาหุ้นจาก yfinance..."):
            symbols = cached_df["symbol"].dropna().unique().tolist() if not cached_df.empty else []
            prices  = fetch_stock_prices(symbols)
            if prices:
                save_price_cache(prices)
                st.success(f"อัปเดตราคาสำเร็จ — {len(prices)} หุ้น")
            else:
                st.warning("ไม่สามารถดึงราคาได้ (ตรวจสอบ internet / yfinance)")
    if prices:
        price_date = time.strftime("%d %b %Y %H:%M", time.localtime(PRICE_CACHE.stat().st_mtime)) if PRICE_CACHE.exists() else "ล่าสุด"
        st.caption(f"ราคาหุ้น: {len(prices)} ตัว  •  อัปเดต {price_date}")

    if cached_df.empty:
        st.error("No cached dataset found. Upload CSV cache files into work/cache/ before deploying.")
        return

    filtered_df = filter_dataframe(
        cached_df,
        min_pct=controls["min_pct"],
        selected_companies=controls["selected_companies"],
        excluded_companies=controls["excluded_companies"],
        excluded_holders_clean=controls["excluded_holders_clean"],
        exclude_nominees=controls["exclude_nominees"],
        only_cross_holders=controls["only_cross_holders"],
    )

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    graph = build_bipartite_graph(filtered_df, prices=prices)
    holder_metrics = compute_holder_metrics(filtered_df)
    company_projection = compute_company_projection(filtered_df)
    top_holders_by_shares = holder_metrics.sort_values(
        ["total_shares", "total_holding_pct"], ascending=[False, False]
    )
    entity_map, entity_labels = build_entity_options(graph)

    focus_node = None
    if controls["focus_holder_name"] != "None":
        selected = holder_metrics.loc[
            holder_metrics["shareholder_name"] == controls["focus_holder_name"], "shareholder_clean"
        ]
        if not selected.empty:
            focus_node = f"holder::{selected.iloc[0]}"
    if controls["clear_selected_node"]:
        focus_node = None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", filtered_df["symbol"].nunique())
    col2.metric("Unique holders", filtered_df["shareholder_clean"].nunique())
    col3.metric("Edges", len(filtered_df))
    col4.metric("Components", nx.number_connected_components(graph))

    if not meta_df.empty:
        st.caption(
            "Cached shareholder dates: "
            + ", ".join(f"{r.symbol} ({r.as_of_date})" for r in meta_df.head(8).itertuples(index=False))
            + (" ..." if len(meta_df) > 8 else "")
        )

    st.subheader("2D Bipartite Network")
    st.caption(
        "Cyan box nodes = SET50 companies, coloured dots = shareholders (by community), "
        "violet = selected node. Edge width = chosen metric."
    )
    st.caption(
        "Edge colors (holding %): gray<2%, blue 2–5%, orange 5–10%, red≥10%. "
        "Edge colors (market value): gray<100M฿, blue 100M–1B฿, orange 1B–10B฿, red≥10B฿."
    )
    if focus_node:
        _, raw_id = focus_node.split("::", 1)
        try:
            holder_name = filtered_df.loc[filtered_df["shareholder_clean"] == raw_id, "shareholder_name"].value_counts().idxmax()
            st.caption(f"Focused shareholder: `{holder_name}`")
        except Exception:
            pass
    components.html(
        make_draggable_network_html(
            graph, filtered_df=filtered_df, focus_node=focus_node,
            layout_mode="nx" if controls["layout_mode"] == "NX Graph Layout" else "bipartite",
            centrality_metric=controls["centrality_metric"],
            allow_physics=controls["layout_mode"] == "NX Graph Layout",
            nx_position_scale=1.0,
            edge_size_metric=controls["edge_size_metric"],
        ),
        height=1180, scrolling=False,
    )

    st.subheader("Relationship Explorer")
    st.caption("Select two or more nodes to trace the shortest relationship paths between them.")
    rel_cols = st.columns([2.2, 1.2])
    with rel_cols[0]:
        relationship_entities = st.multiselect(
            "Choose companies / shareholders", options=entity_labels, default=[],
            placeholder="Pick at least 2 nodes",
        )
    with rel_cols[1]:
        relationship_metric = st.selectbox(
            "Sort bridge nodes by",
            options=["Notion of Centrality","Degree","Closeness","Betweenness","Eigenvector","Katz"],
            index=3,
        )

    if len(relationship_entities) >= 2:
        selected_rel_nodes = [entity_map[l] for l in relationship_entities]
        rel_graph, rel_paths, disconnected, path_edges = summarize_relationship_paths(graph, selected_rel_nodes)
        if rel_graph.number_of_nodes() == 0:
            st.warning("No relationship graph could be built from the selected nodes.")
        else:
            st.caption(f"Subgraph: {rel_graph.number_of_nodes()} nodes, {rel_graph.number_of_edges()} edges.")
            components.html(
                make_draggable_network_html(
                    rel_graph, filtered_df=filtered_df,
                    layout_mode="nx" if controls["layout_mode"] == "NX Graph Layout" else "bipartite",
                    centrality_metric=relationship_metric,
                    selected_nodes=set(selected_rel_nodes),
                    emphasized_edges=path_edges,
                    allow_physics=controls["layout_mode"] == "NX Graph Layout",
                    nx_position_scale=1.0,
                    edge_size_metric=controls["edge_size_metric"],
                ),
                height=860, scrolling=False,
            )
            if rel_paths:
                st.markdown("**Shortest paths found**")
                st.dataframe(pd.DataFrame(rel_paths), use_container_width=True, hide_index=True)
            if disconnected:
                st.markdown("**Pairs with no path**")
                st.dataframe(pd.DataFrame(disconnected), use_container_width=True, hide_index=True)
            rel_metrics = build_relationship_metrics(graph, selected_nodes=selected_rel_nodes, include_nodes=rel_graph.nodes)
            rel_metrics = rel_metrics.sort_values(by=[relationship_metric, "Connections"], ascending=[False, False])
            st.markdown("**Relationship Summary Table**")
            st.dataframe(
                rel_metrics[["Node","Type","Selected","Connections","Degree","Closeness",
                              "Betweenness","Eigenvector","Katz","PageRank","Notion of Centrality"]],
                use_container_width=True, hide_index=True,
            )
            bridge = rel_metrics[~rel_metrics["Selected"]].sort_values(by=[relationship_metric, "Connections"], ascending=[False, False])
            if not bridge.empty:
                st.markdown("**Top bridge nodes**")
                st.dataframe(bridge.head(15), use_container_width=True, hide_index=True)
    else:
        st.info("Pick at least 2 nodes to inspect how they are connected.")

    left, right = st.columns(2)
    with left:
        st.subheader("Top Holders by Total Shares")
        st.dataframe(top_holders_by_shares.head(30), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Top Cross-holders")
        st.dataframe(holder_metrics.head(30), use_container_width=True, hide_index=True)

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Company Overlap")
        if company_projection.empty:
            st.info("No company pairs share holders under the current filters.")
        else:
            st.dataframe(company_projection.head(30), use_container_width=True, hide_index=True)
    with right2:
        st.subheader("Interaction")
        st.info("Click nodes inside the graph to inspect details in the panel beside the network.")

    st.subheader("Raw Edges")
    st.dataframe(
        filtered_df.sort_values(["shareholder_clean", "holding_pct"], ascending=[True, False]),
        use_container_width=True, hide_index=True,
    )
    st.info(
        "Interpretation note: this graph is based on disclosed major shareholders on SET. "
        "It can include nominees, custodians, and NVDR holders, so it is not the same as "
        "ultimate beneficial ownership. Larger circles indicate nodes with more connections."
    )


if __name__ == "__main__":
    main()

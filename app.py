from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SET50_URL = "https://www.set.or.th/th/market/index/set50/overview"
SHAREHOLDER_URL = "https://www.set.or.th/th/market/product/stock/quote/{symbol}/major-shareholders"
CACHE_DIR = Path("work") / "cache"
RAW_CACHE = CACHE_DIR / "set50_shareholders.csv"
META_CACHE = CACHE_DIR / "set50_shareholders_meta.csv"

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


def live_refresh_enabled() -> bool:
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


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for symbol in sorted(df["symbol"].unique()):
        graph.add_node(f"company::{symbol}", label=symbol, node_type="company")
    for holder in sorted(df["shareholder_clean"].unique()):
        display_name = (
            df.loc[df["shareholder_clean"] == holder, "shareholder_name"].value_counts().idxmax()
        )
        graph.add_node(f"holder::{holder}", label=display_name, node_type="shareholder")
    for row in df.itertuples(index=False):
        graph.add_edge(
            f"holder::{row.shareholder_clean}",
            f"company::{row.symbol}",
            weight=float(row.holding_pct),
            shares=float(row.shares),
        )
    return graph


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


def make_network_figure_3d(
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

    pos = nx.spring_layout(
        graph,
        dim=3,
        seed=42,
        k=2.6 / math.sqrt(max(graph.number_of_nodes(), 2)),
        iterations=300,
    )

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_z: list[float] = []
    for left, right in graph.edges():
        x0, y0, z0 = pos[left]
        x1, y1, z1 = pos[right]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    highlighted_neighbors: set[str] = set()
    if focus_node and focus_node in graph:
        highlighted_neighbors = set(graph.neighbors(focus_node))

    edge_traces: list[go.Scatter3d] = []
    if focus_node and focus_node in graph:
        dim_x: list[float] = []
        dim_y: list[float] = []
        dim_z: list[float] = []
        hi_x: list[float] = []
        hi_y: list[float] = []
        hi_z: list[float] = []
        for left, right in graph.edges():
            x0, y0, z0 = pos[left]
            x1, y1, z1 = pos[right]
            target = hi_x if focus_node in {left, right} else dim_x
            target.extend([x0, x1, None])
            (hi_y if focus_node in {left, right} else dim_y).extend([y0, y1, None])
            (hi_z if focus_node in {left, right} else dim_z).extend([z0, z1, None])
        edge_traces.append(
            go.Scatter3d(
                x=dim_x,
                y=dim_y,
                z=dim_z,
                mode="lines",
                line=dict(width=0.6, color="rgba(156,163,175,0.18)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        edge_traces.append(
            go.Scatter3d(
                x=hi_x,
                y=hi_y,
                z=hi_z,
                mode="lines",
                line=dict(width=4, color="#2563EB"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        edge_traces.append(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(width=0.7, color="#9AA5B1"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_x: list[float] = []
    node_y: list[float] = []
    node_z: list[float] = []
    node_text: list[str] = []
    node_size: list[float] = []
    node_color: list[str] = []
    node_label_text: list[str] = []

    holder_rank = sorted(
        [n for n, d in graph.nodes(data=True) if d["node_type"] == "shareholder"],
        key=lambda n: graph.degree(n),
        reverse=True,
    )
    labeled_holders = set(holder_rank[:max_holder_labels])

    for node, attrs in graph.nodes(data=True):
        x, y, z = pos[node]
        degree = graph.degree(node)
        label = attrs["label"]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_size.append(16 + degree * 3)
        node_text.append(f"{label}<br>type={attrs['node_type']}<br>degree={degree}")
        base_color = "#0F766E" if attrs["node_type"] == "company" else "#C2410C"
        if focus_node and node == focus_node:
            node_color.append(rgba("#7C3AED", 1.0))
        elif focus_node and node in highlighted_neighbors:
            node_color.append(rgba(base_color, 1.0))
        elif focus_node:
            node_color.append(rgba(base_color, 0.18))
        else:
            node_color.append(rgba(base_color, 0.9))
        should_label = attrs["node_type"] == "company" or node in labeled_holders
        node_label_text.append(label if show_labels and should_label else "")

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="white"),
        ),
        text=node_text,
        hovertemplate="%{text}<extra></extra>",
        textposition="top center",
        textfont=dict(size=10, color="#1F2937"),
        customdata=node_label_text,
        texttemplate="%{customdata}",
        showlegend=False,
    )

    figure = go.Figure(data=[*edge_traces, node_trace])
    figure.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=1.9, y=1.9, z=1.35)),
            aspectmode="cube",
        ),
        height=920,
        showlegend=False,
    )
    return figure


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def filter_dataframe(
    df: pd.DataFrame,
    min_pct: float,
    selected_companies: Iterable[str],
    exclude_nominees: bool,
    only_cross_holders: bool,
) -> pd.DataFrame:
    filtered = df[df["holding_pct"] >= min_pct].copy()
    if selected_companies:
        filtered = filtered[filtered["symbol"].isin(selected_companies)]
    if exclude_nominees:
        filtered = filtered[~filtered["is_nominee"]]
    if only_cross_holders and not filtered.empty:
        counts = filtered.groupby("shareholder_clean")["symbol"].nunique()
        keep_holders = counts[counts > 1].index
        filtered = filtered[filtered["shareholder_clean"].isin(keep_holders)]
    return filtered


def render_sidebar(df: pd.DataFrame, can_refresh: bool) -> dict:
    companies = sorted(df["symbol"].dropna().unique().tolist()) if not df.empty else []
    holder_options = []
    if not df.empty:
        holder_options = (
            df.groupby("shareholder_clean")["shareholder_name"]
            .agg(lambda series: series.value_counts().idxmax())
            .sort_values()
            .tolist()
        )
    with st.sidebar:
        st.header("Filters")
        min_pct = st.slider("Min holding %", 0.0, 20.0, 0.5, 0.1)
        selected_companies = st.multiselect("Companies", companies, default=companies)
        exclude_nominees = st.toggle("Hide nominee / NVDR holders", value=False)
        only_cross_holders = st.toggle("Show only holders linked to >1 company", value=True)
        st.header("Graph")
        show_labels = st.toggle("Show labels", value=True)
        max_holder_labels = st.slider("Top holder labels", 5, 40, 20, 1)
        focus_holder_name = st.selectbox("Focus shareholder", ["None", *holder_options], index=0)
        sample_limit = st.number_input(
            "Refresh limit (local only)",
            min_value=5,
            max_value=50,
            value=50,
            step=5,
            disabled=not can_refresh,
        )
        force_refresh = st.button("Refresh from SET", type="primary", disabled=not can_refresh)
        if not can_refresh:
            st.caption("Live refresh is disabled on this deployment. The app reads cached CSV files only.")
    return {
        "min_pct": min_pct,
        "selected_companies": selected_companies,
        "exclude_nominees": exclude_nominees,
        "only_cross_holders": only_cross_holders,
        "show_labels": show_labels,
        "max_holder_labels": max_holder_labels,
        "focus_holder_name": focus_holder_name,
        "force_refresh": force_refresh,
        "sample_limit": sample_limit,
    }


def main() -> None:
    st.set_page_config(page_title="SET50 Shareholder Network", layout="wide")
    st.title("SET50 Shareholder Network Analysis")
    st.caption("Cache-first dashboard for major shareholders of SET50 companies.")

    ensure_cache_dir()
    cloud_mode = is_running_on_streamlit_cloud()
    can_refresh = (not cloud_mode) and live_refresh_enabled()

    cached_df, meta_df = load_cached_data()
    controls = render_sidebar(cached_df, can_refresh=can_refresh)

    if controls["force_refresh"]:
        with st.spinner("Scraping SET50 constituents and major shareholders from SET..."):
            cached_df, meta_df = refresh_data(limit=int(controls["sample_limit"]))

    if cached_df.empty:
        st.error(
            "No cached dataset found. Upload the CSV cache files into work/cache/ before deploying this app."
        )
        st.code(
            "work/cache/set50_shareholders.csv\nwork/cache/set50_shareholders_meta.csv",
            language="text",
        )
        if cloud_mode:
            st.info("This Streamlit Cloud deployment is running in cache-only mode.")
        else:
            st.info("For local refresh, set environment variable ENABLE_LIVE_REFRESH=1 before running.")
        return

    filtered_df = filter_dataframe(
        cached_df,
        min_pct=controls["min_pct"],
        selected_companies=controls["selected_companies"],
        exclude_nominees=controls["exclude_nominees"],
        only_cross_holders=controls["only_cross_holders"],
    )

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    graph = build_bipartite_graph(filtered_df)
    holder_metrics = compute_holder_metrics(filtered_df)
    company_projection = compute_company_projection(filtered_df)
    top_holders_by_shares = holder_metrics.sort_values(
        ["total_shares", "total_holding_pct"], ascending=[False, False]
    )

    focus_node = None
    focused_holder_clean = None
    if controls["focus_holder_name"] != "None":
        selected = holder_metrics.loc[
            holder_metrics["shareholder_name"] == controls["focus_holder_name"], "shareholder_clean"
        ]
        if not selected.empty:
            focused_holder_clean = selected.iloc[0]
            focus_node = f"holder::{focused_holder_clean}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", filtered_df["symbol"].nunique())
    col2.metric("Unique holders", filtered_df["shareholder_clean"].nunique())
    col3.metric("Edges", len(filtered_df))
    col4.metric("Components", nx.number_connected_components(graph))

    if not meta_df.empty:
        st.caption(
            "Cached shareholder dates: "
            + ", ".join(
                f"{row.symbol} ({row.as_of_date})" for row in meta_df.head(8).itertuples(index=False)
            )
            + (" ..." if len(meta_df) > 8 else "")
        )

    st.subheader("3D Bipartite Network")
    st.caption("Drag to rotate, zoom in/out, and hover nodes to inspect relationships.")
    st.markdown(
        "`Green/Teal nodes` = listed companies in SET50, "
        "`Orange nodes` = shareholders, "
        "`Gray lines` = shareholding relationships between a shareholder and a company, "
        "`Purple node` = focused shareholder, "
        "`Blue lines` = links from the focused shareholder to connected companies."
    )
    if focus_node:
        st.caption(f"Focused shareholder: `{controls['focus_holder_name']}`")
    st.plotly_chart(
        make_network_figure_3d(
            graph,
            show_labels=controls["show_labels"],
            max_holder_labels=int(controls["max_holder_labels"]),
            focus_node=focus_node,
        ),
        use_container_width=True,
    )

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
        if focus_node:
            st.subheader("Focused Holder Links")
            focused_edges = filtered_df[
                filtered_df["shareholder_clean"] == focused_holder_clean
            ].sort_values("holding_pct", ascending=False)
            st.dataframe(focused_edges, use_container_width=True, hide_index=True)
        else:
            st.subheader("Focused Holder Links")
            st.info("Select a shareholder in the sidebar to highlight its connections.")

    st.subheader("Raw Edges")
    st.dataframe(
        filtered_df.sort_values(["shareholder_clean", "holding_pct"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Interpretation note: this graph is based on disclosed major shareholders on SET. "
        "It can include nominees, custodians, and NVDR holders, so it is not the same as "
        "ultimate beneficial ownership. Larger circles indicate nodes with more connections."
    )


if __name__ == "__main__":
    main()

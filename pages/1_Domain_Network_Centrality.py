from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "Neo4j GDS" / "quotes_domain_network"
NODES_CSV = DATA_DIR / "web_export" / "domain_nodes_top200.csv"
EDGES_CSV = DATA_DIR / "web_export" / "domain_edges_top200.csv"
OVERVIEW_PNG = DATA_DIR / "centrality_visualization.png"

METRIC_LABELS = {
    "degree": "Degree",
    "betweenness": "Betweenness",
    "closeness": "Closeness",
    "eigenvector": "Eigenvector",
    "pagerank": "PageRank",
}

COMMUNITY_COLORWAY = [
    "#EC4899", "#22C55E", "#F97316", "#3B82F6",
    "#EF4444", "#92400E", "#9CA3AF", "#8B5CF6",
]


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(EDGES_CSV)
    return nodes, edges


@st.cache_data
def build_graph_and_layout(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[nx.Graph, dict]:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.domain, **row._asdict())
    for row in edges.itertuples(index=False):
        graph.add_edge(row.src_domain, row.tgt_domain, is_bridge=bool(row.is_bridge))
    pos = nx.spring_layout(graph, k=0.35, seed=42)
    return graph, pos


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def make_figure(
    graph: nx.Graph,
    pos: dict,
    metric: str,
    color_by_community: bool,
    n_labels: int,
    focus_node: str | None,
) -> go.Figure:
    edge_x, edge_y = [], []
    bridge_x, bridge_y = [], []
    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        target_x, target_y = (bridge_x, bridge_y) if data.get("is_bridge") else (edge_x, edge_y)
        target_x.extend([x0, x1, None])
        target_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="#9AA5B1"), hoverinfo="skip", showlegend=False,
    )
    bridge_trace = go.Scatter(
        x=bridge_x, y=bridge_y, mode="lines",
        line=dict(width=2.5, color="#DC2626"), hoverinfo="skip", showlegend=False,
    )

    values = [graph.nodes[n][metric] for n in graph.nodes]
    vmin, vmax = min(values), max(values)
    node_x, node_y, node_size, node_color, node_text, customdata = [], [], [], [], [], []

    top_labeled = set(sorted(graph.nodes, key=lambda n: graph.nodes[n][metric], reverse=True)[:n_labels])

    for n in graph.nodes:
        attrs = graph.nodes[n]
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        v = attrs[metric]
        size = 8 + 34 * ((v - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
        if focus_node and n == focus_node:
            size = max(size, 30)
        node_size.append(size)
        node_color.append(v)
        node_text.append(n if n in top_labeled or n == focus_node else "")
        customdata.append([n, attrs["degree"], attrs["betweenness"], attrs["closeness"],
                            attrs["eigenvector"], attrs["pagerank"], attrs["community"]])

    if color_by_community:
        communities = sorted(set(int(graph.nodes[n]["community"]) for n in graph.nodes))
        cmap = {c: COMMUNITY_COLORWAY[i % len(COMMUNITY_COLORWAY)] for i, c in enumerate(communities)}
        marker_color = [cmap[int(graph.nodes[n]["community"])] for n in graph.nodes]
        marker = dict(size=node_size, color=marker_color, line=dict(width=1, color="white"))
    else:
        marker = dict(
            size=node_size, color=node_color, colorscale="Viridis", showscale=True,
            colorbar=dict(title=METRIC_LABELS[metric]), line=dict(width=1, color="white"),
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=marker, text=node_text,
        textposition="top center", textfont=dict(size=9, color="#1F2937"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>degree=%{customdata[1]:.1f}<br>"
            "betweenness=%{customdata[2]:.1f}<br>closeness=%{customdata[3]:.3f}<br>"
            "eigenvector=%{customdata[4]:.4f}<br>pagerank=%{customdata[5]:.3f}<br>"
            "community=%{customdata[6]}<extra></extra>"
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, bridge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        height=760,
    )
    return fig


def get_selected_node(event) -> str | None:
    if not event:
        return None
    selection = event.get("selection") if hasattr(event, "get") else None
    if not selection:
        return None
    points = selection.get("points", [])
    if not points:
        return None
    customdata = points[-1].get("customdata")
    return customdata[0] if customdata else None


def main() -> None:
    st.set_page_config(page_title="Domain Hyperlink Network", layout="wide")
    st.title("Domain Hyperlink Network — Centrality (Cypher/GDS only)")
    st.caption(
        "Built from the MemeTracker/Spinn3r quotes corpus (quotes_2009-04.txt.gz), collapsed to a "
        "domain-to-domain hyperlink network. All 7 metrics below were computed with Neo4j GDS "
        "procedures called from Cypher — no NetworkX/Python centrality recomputation."
    )

    if not NODES_CSV.exists() or not EDGES_CSV.exists():
        st.error(f"Missing data export. Expected files under {NODES_CSV.parent}")
        return

    nodes, edges = load_data()
    graph, pos = build_graph_and_layout(nodes, edges)

    if "domain_focus_node" not in st.session_state:
        st.session_state["domain_focus_node"] = None

    with st.sidebar:
        st.header("Controls")
        metric_key = st.selectbox(
            "Size / color by metric", list(METRIC_LABELS.keys()),
            format_func=lambda k: METRIC_LABELS[k],
        )
        color_by_community = st.toggle("Color by Louvain community instead", value=True)
        n_labels = st.slider("Number of labeled top nodes", 0, 40, 10, 1)
        clear_focus = st.button("Clear selected node")
        if clear_focus:
            st.session_state["domain_focus_node"] = None

    st.subheader("All 7 metrics at a glance")
    if OVERVIEW_PNG.exists():
        st.image(str(OVERVIEW_PNG), use_container_width=True)
    st.caption(
        "Static overview (top 200 of 3,649 domains by degree). Node color = Louvain community "
        "(consistent across panels), red edges = bridges (none found in this graph — see note below)."
    )

    st.subheader(f"Interactive explorer — sized by {METRIC_LABELS[metric_key]}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Domains shown", graph.number_of_nodes())
    col2.metric("Edges shown", graph.number_of_edges())
    col3.metric("Louvain communities", nodes["community"].nunique())
    col4.metric("Bridges found", int(edges["is_bridge"].sum()))

    event = st.plotly_chart(
        make_figure(
            graph, pos, metric_key, color_by_community, n_labels,
            st.session_state.get("domain_focus_node"),
        ),
        use_container_width=True,
        key="domain_network_chart",
        on_select="rerun",
        selection_mode=("points",),
    )
    selected = get_selected_node(event)
    if selected and selected != st.session_state.get("domain_focus_node"):
        st.session_state["domain_focus_node"] = selected
        st.rerun()

    focus = st.session_state.get("domain_focus_node")
    with st.expander("Selected domain details", expanded=bool(focus)):
        if not focus:
            st.info("Click a node in the graph above to inspect its full metric breakdown.")
        else:
            row = nodes.loc[nodes["domain"] == focus].iloc[0]
            st.markdown(f"**{focus}**  (community {int(row['community'])})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Degree", f"{row['degree']:.1f}")
            c2.metric("Betweenness", f"{row['betweenness']:.1f}")
            c3.metric("Closeness", f"{row['closeness']:.3f}")
            c4.metric("Eigenvector", f"{row['eigenvector']:.4f}")
            c5.metric("PageRank", f"{row['pagerank']:.3f}")
            st.caption(f"Internal (same-domain) link count: {int(row['internal_link_count']):,}")

    st.markdown("## What each metric means, and what we found")
    st.markdown(
        """
- **Degree** — raw (log-weighted) connectivity. Top hubs: `en.wikipedia.org`, `youtube.com`,
  `nytimes.com`, `amazon.com`, `washingtonpost.com`.
- **Betweenness** — how often a domain sits on the shortest path between other domain pairs
  (a "broker"). Mostly the same hubs, **except `mahalo.com`**, which ranks 4th here despite not
  appearing in the top 5 of any other metric — it structurally bridges otherwise-separate
  clusters without being one of the most-linked domains itself.
- **Closeness** — average distance to every other domain. The spread is narrow (0.35–0.72)
  because the underlying graph was k-core-pruned (every domain kept has ≥30 neighbors), so the
  whole network is a small, densely-connected world — most domains are "close" to everything.
- **Eigenvector** — influence from being linked *by* other influential domains (computed on the
  directed graph). This is the one metric that diverges: `nytimes.com`, `topix.net` and
  `tinyurl.com` lead, while `en.wikipedia.org`/`youtube.com` drop to 6th–7th place — they're
  linked *a lot*, but not disproportionately by other high-eigenvector domains.
- **PageRank** — like Eigenvector but with a damping factor, which pulls the ranking back toward
  the Degree/Betweenness picture (`youtube.com`, `en.wikipedia.org`, `nytimes.com` on top again).
- **Louvain (community detection)** — found **8 communities** (sizes 772, 698, 652, 583, 376,
  326, 129, 113 domains; modularity ≈ 0.298 — a moderate split, expected on such a dense graph).
- **Bridges** — **0 found**. Not a bug: k-core pruning with `min_degree=30` guarantees every
  domain has at least 30 neighbors, making the network too densely/redundantly connected for any
  single edge to be a structural point of failure.

Full pipeline and data-quality decisions (internal-link filtering, log-weighting, why k-core
pruning was needed instead of a raw weight threshold) are documented in
`Neo4j GDS/quotes_domain_network/README.md`.
        """
    )

    st.subheader("Full data (top 200 domains)")
    st.dataframe(nodes.sort_values(metric_key, ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "Download this table as CSV",
        data=nodes.to_csv(index=False).encode("utf-8"),
        file_name="domain_nodes_top200.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

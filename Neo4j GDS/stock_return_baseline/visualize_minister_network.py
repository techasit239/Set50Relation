# -*- coding: utf-8 -*-
"""
Render the Minister <-> Ministry <-> Stock tripartite network built by build_minister_network.py.
Node shape/color by type, size proportional to betweenness (so multi-ministry "bridge" ministers
stand out visually), top-10 ministers by betweenness labeled directly.

Run:
    python visualize_minister_network.py
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

matplotlib.rcParams["font.family"] = "Tahoma"  # Thai glyph support, consistent with analyze_budget_stock_network.py

from build_minister_network import build_graph, parse_be_date
from ministry_stock_data import REAL_MINISTRY_INFO

HERE = Path(__file__).parent
CABINET_CSV = HERE / "cabinet_history.csv"
OUT_PNG = HERE / "minister_network.png"

TYPE_COLOR = {"ministry": "#F97316", "stock": "#9CA3AF", "minister": "#3B82F6"}
TYPE_SHAPE = {"ministry": "s", "stock": "o", "minister": "^"}


def main() -> None:
    cabinet = pd.read_csv(CABINET_CSV)
    cabinet["start_date"] = cabinet["start_date"].apply(parse_be_date)
    cabinet["end_date"] = cabinet["end_date"].apply(parse_be_date)
    graph = build_graph(cabinet)

    betweenness = nx.betweenness_centrality(graph, weight="weight")
    pos = nx.spring_layout(graph, k=0.5, seed=42, weight=None)  # edge weights mix years & correlation - wrong scale for layout physics

    fig, ax = plt.subplots(figsize=(18, 14))

    for node_type in ["stock", "ministry", "minister"]:  # draw ministers last (on top)
        nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == node_type]
        sizes = [80 + 3000 * betweenness[n] for n in nodes]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=TYPE_COLOR[node_type],
            node_shape=TYPE_SHAPE[node_type], node_size=sizes, alpha=0.85, ax=ax,
            label=node_type.capitalize(),
        )

    nx.draw_networkx_edges(graph, pos, width=0.4, alpha=0.3, ax=ax)

    # draw_networkx_edges' FancyArrowPatch autoscale miscomputes axis limits (blows up to ~1000x
    # the real data range) - reset explicitly from the actual node positions instead of trusting it
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin_x = (max(xs) - min(xs)) * 0.1
    margin_y = (max(ys) - min(ys)) * 0.1
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    ministry_labels = {n: REAL_MINISTRY_INFO[n]["label_en"] for n in graph.nodes if graph.nodes[n]["node_type"] == "ministry"}
    nx.draw_networkx_labels(graph, pos, labels=ministry_labels, font_size=9, font_weight="bold", font_family="Tahoma", ax=ax)

    top_ministers = sorted(
        (n for n, d in graph.nodes(data=True) if d["node_type"] == "minister"),
        key=lambda n: betweenness[n], reverse=True,
    )[:10]
    nx.draw_networkx_labels(graph, pos, labels={n: n for n in top_ministers}, font_size=8, font_family="Tahoma", ax=ax)

    ax.set_title(
        "Minister <-> Ministry <-> Stock network (2014-2026)\n"
        "Node size = betweenness centrality (bigger = bridges more of the network by holding "
        "multiple ministries) - top 10 ministers labeled",
        fontsize=13,
    )
    ax.legend(scatterpoints=1, loc="lower left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Render the Minister <-> Ministry <-> Stock tripartite network built by build_minister_network.py.

Uses a custom radial/cluster layout instead of plain force-directed spring_layout: ministries are
anchored evenly around a circle, each ministry's stocks fan out around it in a tidy arc, and each
minister is placed at the centroid of the ministries they served in - so a minister who bridged
multiple ministries visually sits *between* them, pulled toward the center, while a single-ministry
minister sits just outside their one ministry. Force-directed layout on a graph this lopsided
(a few high-degree ministry hubs vs hundreds of degree-1 stock leaves) just produces a messy
scatter with stray long edges - a structured layout reads far more clearly for a hub-and-spoke
graph like this one.

Run:
    python visualize_minister_network.py
"""
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Tahoma"  # Thai glyph support, consistent with analyze_budget_stock_network.py

from build_minister_network import build_graph, parse_be_date
from ministry_stock_data import REAL_MINISTRY_INFO

HERE = Path(__file__).parent
CABINET_CSV = HERE / "cabinet_history.csv"
OUT_PNG = HERE / "minister_network.png"

BG_COLOR = "#0F172A"
TYPE_COLOR = {"ministry": "#F59E0B", "stock": "#475569", "minister": "#38BDF8"}
TYPE_SHAPE = {"ministry": "s", "stock": "o", "minister": "^"}
LABEL_COLOR = "#F1F5F9"

MINISTRY_RADIUS = 4.0
STOCK_RADIUS = 6.4
STOCK_ARC_DEGREES = 34  # how wide each ministry's stock "petal" fans out
MINISTER_RADIUS = 2.2  # inner ring where single-ministry ministers fan out
MINISTER_ARC_DEGREES = 50  # wider than the stock arc - there are fewer ministers per ministry


def radial_layout(graph: nx.Graph) -> dict:
    ministries = sorted(
        (n for n, d in graph.nodes(data=True) if d["node_type"] == "ministry"),
        key=lambda n: REAL_MINISTRY_INFO[n]["label_en"],
    )
    n = len(ministries)
    ministry_angle = {m: 2 * math.pi * i / n for i, m in enumerate(ministries)}
    pos = {
        m: (MINISTRY_RADIUS * math.cos(a), MINISTRY_RADIUS * math.sin(a))
        for m, a in ministry_angle.items()
    }

    # stocks: fan out around their one ministry's angle, sorted for a tidy sweep
    for m in ministries:
        stocks = sorted(n for n in graph.neighbors(m) if graph.nodes[n]["node_type"] == "stock")
        base_angle = ministry_angle[m]
        spread = math.radians(STOCK_ARC_DEGREES)
        for i, s in enumerate(stocks):
            frac = (i / max(len(stocks) - 1, 1)) - 0.5  # -0.5..0.5
            angle = base_angle + frac * spread
            radius = STOCK_RADIUS + (0.5 if i % 2 else 0.0)  # slight zig-zag so labels/points don't collide
            pos[s] = (radius * math.cos(angle), radius * math.sin(angle))

    # ministers who served exactly one ministry: fan out on an inner ring around that ministry's
    # angle (mirrors the stock fan-out, just closer to the center) instead of all piling up in the
    # middle of the whole graph.
    minister_ministries: dict[str, list[str]] = {}
    for node, data in graph.nodes(data=True):
        if data["node_type"] != "minister":
            continue
        minister_ministries[node] = sorted(
            nb for nb in graph.neighbors(node) if graph.nodes[nb]["node_type"] == "ministry"
        )

    for m in ministries:
        single = sorted(name for name, served in minister_ministries.items() if served == [m])
        base_angle = ministry_angle[m]
        spread = math.radians(MINISTER_ARC_DEGREES)
        for i, name in enumerate(single):
            frac = (i / max(len(single) - 1, 1)) - 0.5
            angle = base_angle + frac * spread
            radius = MINISTER_RADIUS - (0.35 if i % 2 else 0.0)
            pos[name] = (radius * math.cos(angle), radius * math.sin(angle))

    # ministers who bridged multiple ministries: centroid of those ministries' positions, so they
    # visually sit *between* the ones they served (naturally pulled toward the center the more
    # spread-out their ministries are)
    rng = np.random.default_rng(42)
    for name, served in minister_ministries.items():
        if len(served) <= 1:
            continue
        pts = np.array([pos[m] for m in served])
        centroid = pts.mean(axis=0)
        jitter = rng.uniform(-0.2, 0.2, size=2)
        pos[name] = tuple(centroid + jitter)

    return pos


def main() -> None:
    cabinet = pd.read_csv(CABINET_CSV)
    cabinet["start_date"] = cabinet["start_date"].apply(parse_be_date)
    cabinet["end_date"] = cabinet["end_date"].apply(parse_be_date)
    graph = build_graph(cabinet)

    betweenness = nx.betweenness_centrality(graph, weight="weight")
    pos = radial_layout(graph)

    fig, ax = plt.subplots(figsize=(20, 16), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ministry_stock_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if {graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]} == {"ministry", "stock"}
    ]
    minister_ministry_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if {graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]} == {"minister", "ministry"}
    ]

    nx.draw_networkx_edges(graph, pos, edgelist=ministry_stock_edges, edge_color="#334155", width=0.6, alpha=0.55, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=minister_ministry_edges, edge_color="#7DD3FC", width=1.3, alpha=0.75, ax=ax)

    for node_type in ["stock", "ministry", "minister"]:  # draw ministers last (on top)
        nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == node_type]
        if node_type == "stock":
            sizes = 55
        else:
            sizes = [140 + 2600 * betweenness[n] for n in nodes]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=TYPE_COLOR[node_type],
            node_shape=TYPE_SHAPE[node_type], node_size=sizes, alpha=0.95, ax=ax,
            linewidths=0.6, edgecolors=BG_COLOR,
            label=node_type.capitalize(),
        )

    # axis limits from actual positions - draw_networkx_edges' autoscale can't be trusted (blows up
    # xlim/ylim by ~1000x on some matplotlib/networkx versions - a known FancyArrowPatch quirk)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 1.2
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    ministry_labels = {n: REAL_MINISTRY_INFO[n]["label_en"] for n in graph.nodes if graph.nodes[n]["node_type"] == "ministry"}
    for node, label in ministry_labels.items():
        x, y = pos[node]
        ax.annotate(
            label, (x, y), xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=12, fontweight="bold", color=LABEL_COLOR, font="Tahoma",
        )

    # Only label ministers who actually bridge >=2 ministries (betweenness > 0) - with only 8 such
    # ministers in this graph, padding out to a fixed top-N with zero-betweenness ministers just
    # adds meaningless, overlapping labels near the crowded center.
    top_ministers = sorted(
        (n for n, d in graph.nodes(data=True) if d["node_type"] == "minister" and betweenness[n] > 0),
        key=lambda n: betweenness[n], reverse=True,
    )
    for i, node in enumerate(top_ministers):
        x, y = pos[node]
        # alternate above/below (with a bigger gap than the ministry labels use) so labels whose
        # nodes land close together (multiple bridges pulled toward similar centroids) don't run
        # into each other
        dy = 16 if i % 2 == 0 else -28
        ax.annotate(
            node, (x, y), xytext=(0, dy), textcoords="offset points",
            ha="center", fontsize=9.5, color="#E2E8F0", font="Tahoma",
            bbox=dict(boxstyle="round,pad=0.15", fc=BG_COLOR, ec="none", alpha=0.85),
        )

    ax.set_title(
        "Minister <-> Ministry <-> Stock network (2014-2026)",
        fontsize=18, fontweight="bold", color=LABEL_COLOR, pad=18,
    )
    ax.text(
        0.5, 0.965,
        f"Node size = betweenness centrality (bigger = bridges more of the network by holding "
        f"multiple ministries) - the {len(top_ministers)} ministers who bridged 2+ ministries are labeled",
        transform=fig.transFigure, ha="center", fontsize=11, color="#94A3B8",
    )
    legend = ax.legend(scatterpoints=1, loc="lower left", frameon=False, fontsize=11, labelcolor=LABEL_COLOR)
    ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # tight_layout()'s internal draw pass re-triggers the same FancyArrowPatch autoscale bug and
    # silently blows xlim/ylim back up - reassert the correct limits as the very last step
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    fig.savefig(OUT_PNG, dpi=170, facecolor=BG_COLOR)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()

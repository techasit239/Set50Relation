# -*- coding: utf-8 -*-
"""
Pure-math radial layout for the Minister <-> Ministry <-> Stock tripartite graph, split out from
visualize_minister_network.py so consumers that don't render a matplotlib PNG (e.g. the pyvis
interactive version) don't need matplotlib installed - Streamlit Cloud's deployed environment
installs only requirements.txt, which doesn't include matplotlib.

Ministries are anchored evenly around a circle, each ministry's stocks fan out around it in a
tidy arc, and each minister is placed at the centroid of the ministries they served in - so a
minister who bridged multiple ministries visually sits *between* them, pulled toward the center,
while a single-ministry minister sits just outside their one ministry.
"""
import math

import networkx as nx
import numpy as np

from ministry_stock_data import REAL_MINISTRY_INFO

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

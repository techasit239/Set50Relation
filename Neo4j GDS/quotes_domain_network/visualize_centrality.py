# -*- coding: utf-8 -*-
"""
Read the 7 centrality/community metrics back out of Neo4j (already computed via
Cypher/GDS by run_centrality.cypher) and render them. This script never recomputes
anything - it only queries already-written node properties and draws the picture.

Requires: Neo4j container up, load_and_project.cypher + run_centrality.cypher already run.

Run:
    python visualize_centrality.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

HERE = Path(__file__).parent
ENV_FILE = HERE.parent / ".env"
OUT_PNG = HERE / "centrality_visualization.png"

TOP_N_FOR_DRAWING = 200  # keep the figure legible; GDS itself ran on the full 3,649-node graph

METRICS = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]


def read_password() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("NEO4J_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"NEO4J_PASSWORD not found in {ENV_FILE}")


def main() -> None:
    password = read_password()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))

    with driver.session() as session:
        top_nodes = session.run(
            f"""
            MATCH (d:Domain)
            RETURN d.name AS name, d.degree AS degree, d.betweenness AS betweenness,
                   d.closeness AS closeness, d.eigenvector AS eigenvector,
                   d.pagerank AS pagerank, d.community AS community
            ORDER BY d.degree DESC
            LIMIT {TOP_N_FOR_DRAWING}
            """
        ).data()

        names = {row["name"] for row in top_nodes}
        edges = session.run(
            """
            MATCH (a:Domain)-[r:LINKS_TO]-(b:Domain)
            WHERE a.name IN $names AND b.name IN $names AND a.name < b.name
            RETURN a.name AS src, b.name AS tgt
            """,
            names=list(names),
        ).data()

        bridge_rows = session.run(
            "CALL gds.bridges.stream('domainUndirected') YIELD from, to "
            "RETURN gds.util.asNode(from).name AS src, gds.util.asNode(to).name AS tgt"
        ).data()

    driver.close()

    bridge_pairs = {(r["src"], r["tgt"]) for r in bridge_rows} | {(r["tgt"], r["src"]) for r in bridge_rows}

    G = nx.Graph()
    for row in top_nodes:
        G.add_node(row["name"], **row)
    for e in edges:
        G.add_edge(e["src"], e["tgt"])

    print(f"drawing {G.number_of_nodes()} nodes / {G.number_of_edges()} edges "
          f"(full GDS graph was 3,649 nodes / 168,592 edges)")
    print(f"bridge edges found in full graph: {len(bridge_rows)}")

    pos = nx.spring_layout(G, k=0.4, seed=42, weight=None)

    communities = sorted({G.nodes[n]["community"] for n in G.nodes})
    palette = plt.get_cmap("tab10")
    community_color = {c: palette(i % 10) for i, c in enumerate(communities)}
    node_colors = [community_color[G.nodes[n]["community"]] for n in G.nodes]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        values = [G.nodes[n][metric] for n in G.nodes]
        vmin, vmax = min(values), max(values)
        sizes = [80 + 900 * ((v - vmin) / (vmax - vmin) if vmax > vmin else 0.5) for v in values]

        edge_colors = ["red" if (u, v) in bridge_pairs else "lightgray" for u, v in G.edges]
        edge_widths = [2.0 if (u, v) in bridge_pairs else 0.3 for u, v in G.edges]

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=node_colors, linewidths=0)

        top5 = sorted(G.nodes, key=lambda n: G.nodes[n][metric], reverse=True)[:5]
        nx.draw_networkx_labels(G, pos, labels={n: n for n in top5}, ax=ax, font_size=7)

        ax.set_title(metric.capitalize())
        ax.axis("off")

    # Louvain community panel
    ax = axes[5]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=0.3, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120, node_color=node_colors, linewidths=0)
    ax.set_title(f"Louvain communities ({len(communities)} found)")
    ax.axis("off")

    # Bridges panel
    ax = axes[6]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, node_color="lightgray", linewidths=0)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgray", width=0.3, alpha=0.3)
    bridge_edges_in_view = [(u, v) for u, v in G.edges if (u, v) in bridge_pairs]
    if bridge_edges_in_view:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=bridge_edges_in_view, edge_color="red", width=2.5)
        ax.set_title(f"Bridges ({len(bridge_rows)} found in full graph)")
    else:
        ax.set_title(f"Bridges: {len(bridge_rows)} found\n(k-core≥30 graph is densely connected - no single point of failure)")
    ax.axis("off")

    axes[7].axis("off")

    fig.suptitle(
        f"Domain hyperlink network - 7 Neo4j GDS / Cypher metrics "
        f"(top {G.number_of_nodes()} of 3,649 domains by degree shown)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()

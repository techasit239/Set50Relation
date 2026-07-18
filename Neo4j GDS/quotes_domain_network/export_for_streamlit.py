# -*- coding: utf-8 -*-
"""
Export a small, static snapshot of the domain hyperlink network + its 7 Cypher/GDS
metrics, for the deployed Streamlit app (which cannot reach this machine's Neo4j
container). Values are read back only - nothing is recomputed here.

Requires: Neo4j container up, load_and_project.cypher + run_centrality.cypher already run.

Run:
    python export_for_streamlit.py
"""
from pathlib import Path

from neo4j import GraphDatabase

HERE = Path(__file__).parent
ENV_FILE = HERE.parent / ".env"
WEB_EXPORT_DIR = HERE / "web_export"
NODES_OUT = WEB_EXPORT_DIR / "domain_nodes_top200.csv"
EDGES_OUT = WEB_EXPORT_DIR / "domain_edges_top200.csv"

TOP_N = 200


def read_password() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("NEO4J_PASSWORD="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"NEO4J_PASSWORD not found in {ENV_FILE}")


def main() -> None:
    password = read_password()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))

    with driver.session() as session:
        nodes = session.run(
            f"""
            MATCH (d:Domain)
            RETURN d.name AS domain, d.internalLinkCount AS internal_link_count,
                   d.degree AS degree, d.betweenness AS betweenness,
                   d.closeness AS closeness, d.eigenvector AS eigenvector,
                   d.pagerank AS pagerank, d.community AS community
            ORDER BY d.degree DESC
            LIMIT {TOP_N}
            """
        ).data()

        names = [row["domain"] for row in nodes]
        edges = session.run(
            """
            MATCH (a:Domain)-[r:LINKS_TO]-(b:Domain)
            WHERE a.name IN $names AND b.name IN $names AND a.name < b.name
            RETURN a.name AS src_domain, b.name AS tgt_domain,
                   max(r.weight) AS weight, max(r.logWeight) AS log_weight
            """,
            names=names,
        ).data()

        bridge_rows = session.run(
            "CALL gds.bridges.stream('domainUndirected') YIELD from, to "
            "RETURN gds.util.asNode(from).name AS src_domain, gds.util.asNode(to).name AS tgt_domain"
        ).data()

    driver.close()

    bridge_pairs = {(r["src_domain"], r["tgt_domain"]) for r in bridge_rows} | {
        (r["tgt_domain"], r["src_domain"]) for r in bridge_rows
    }

    WEB_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(NODES_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("domain,internal_link_count,degree,betweenness,closeness,eigenvector,pagerank,community\n")
        for row in nodes:
            f.write(
                f"{row['domain']},{row['internal_link_count']},{row['degree']},"
                f"{row['betweenness']},{row['closeness']},{row['eigenvector']},"
                f"{row['pagerank']},{row['community']}\n"
            )

    with open(EDGES_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("src_domain,tgt_domain,weight,log_weight,is_bridge\n")
        for e in edges:
            is_bridge = (e["src_domain"], e["tgt_domain"]) in bridge_pairs
            f.write(
                f"{e['src_domain']},{e['tgt_domain']},{e['weight']},{e['log_weight']},{is_bridge}\n"
            )

    print(f"wrote {len(nodes)} nodes -> {NODES_OUT}")
    print(f"wrote {len(edges)} edges -> {EDGES_OUT}")
    print(f"bridges found in full graph: {len(bridge_rows)}")


if __name__ == "__main__":
    main()

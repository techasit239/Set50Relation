# -*- coding: utf-8 -*-
"""
Filter + transform the raw aggregated domain hyperlink network for loading into Neo4j.

Input (produced by build_domain_edges.py, next to this script):
    domain_edges_raw.csv   src_domain,tgt_domain,weight
    domain_nodes_raw.csv   domain,internal_link_count

Output (written into ../neo4j-import/ so the Neo4j Docker container can LOAD CSV them):
    domain_nodes.csv        domain,internal_link_count
    domain_edges.csv        src_domain,tgt_domain,weight,log_weight

Filtering (two stages):
    1) drop (src,tgt) pairs with weight < MIN_WEIGHT (noise floor on one-off links)
    2) k-core prune: repeatedly drop domains whose undirected degree < MIN_DEGREE
       (and their edges), until stable. This is what actually controls final graph
       size - a raw weight threshold alone barely shrinks node count, since a domain
       only needs ONE surviving high-weight edge to stay in.
    log_weight = ln(1 + weight), so a few extreme repeat-link outliers don't dominate
    weighted centrality.

Run:
    python filter_domain_network.py [min_weight] [min_degree]
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
EDGES_RAW = HERE / "domain_edges_raw.csv"
NODES_RAW = HERE / "domain_nodes_raw.csv"

IMPORT_DIR = HERE.parent / "neo4j-import"
EDGES_OUT = IMPORT_DIR / "domain_edges.csv"
NODES_OUT = IMPORT_DIR / "domain_nodes.csv"


def load_internal_link_counts() -> dict:
    internal_link_count = {}
    with open(NODES_RAW, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            domain, count = line.rstrip("\n").rsplit(",", 1)
            internal_link_count[domain] = int(count)
    return internal_link_count


def load_edges_above_weight(min_weight: int):
    edges = []
    n_total = 0
    n_dropped = 0
    with open(EDGES_RAW, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            src, tgt, weight_s = line.rstrip("\n").rsplit(",", 2)
            if not src or not tgt:
                continue  # malformed row (embedded comma ate a field)
            try:
                weight = int(weight_s)
            except ValueError:
                continue  # malformed row (garbage "domain" from noisy source data)
            n_total += 1
            if weight < min_weight:
                n_dropped += 1
                continue
            edges.append((src, tgt, weight))
    return edges, n_total, n_dropped


def k_core_prune(edges, min_degree: int):
    """Repeatedly drop nodes with undirected degree < min_degree (and their edges)."""
    adj = defaultdict(dict)  # node -> {neighbor: weight}  (undirected, merges both directions)
    for s, t, w in edges:
        adj[s][t] = adj[s].get(t, 0) + w
        adj[t][s] = adj[t].get(s, 0) + w

    degree = {n: len(neighbors) for n, neighbors in adj.items()}
    to_remove = [n for n, d in degree.items() if d < min_degree]

    while to_remove:
        n = to_remove.pop()
        if n not in adj:
            continue
        for neighbor in list(adj[n].keys()):
            if neighbor in adj and n in adj[neighbor]:
                del adj[neighbor][n]
                degree[neighbor] -= 1
                if degree[neighbor] < min_degree and neighbor in adj:
                    to_remove.append(neighbor)
        del adj[n]
        del degree[n]

    kept_domains = set(adj.keys())
    kept_edges = [(s, t, w) for (s, t, w) in edges if s in kept_domains and t in kept_domains]
    return kept_edges, kept_domains


def main() -> None:
    min_weight = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    min_degree = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not EDGES_RAW.exists() or not NODES_RAW.exists():
        raise SystemExit(f"Missing raw input files in {HERE} — run build_domain_edges.py first")

    internal_link_count = load_internal_link_counts()
    edges, n_total, n_dropped_weight = load_edges_above_weight(min_weight)
    kept_edges, kept_domains = k_core_prune(edges, min_degree)

    IMPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EDGES_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("src_domain,tgt_domain,weight,log_weight\n")
        for s, t, w in kept_edges:
            f.write(f"{s},{t},{w},{math.log1p(w):.6f}\n")

    with open(NODES_OUT, "w", encoding="utf-8", newline="") as f:
        f.write("domain,internal_link_count\n")
        for d in kept_domains:
            f.write(f"{d},{internal_link_count.get(d, 0)}\n")

    print(f"min_weight: {min_weight}, min_degree (k-core): {min_degree}")
    print(f"raw distinct edges: {n_total:,} (dropped {n_dropped_weight:,} below min_weight)")
    print(f"edges after weight filter: {len(edges):,}")
    print(f"kept edges after k-core prune: {len(kept_edges):,}")
    print(f"kept domains (nodes): {len(kept_domains):,}")
    print(f"wrote {NODES_OUT}")
    print(f"wrote {EDGES_OUT}")


if __name__ == "__main__":
    main()

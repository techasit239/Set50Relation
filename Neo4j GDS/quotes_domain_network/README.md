# Domain hyperlink network — 7 centrality/community metrics via Neo4j Cypher/GDS

Builds a domain-to-domain hyperlink network from the MemeTracker/Spinn3r quotes corpus
(`quotes_2009-04.txt.gz`, P=permalink/T=timestamp/Q=quote/L=link blocks) and computes
Betweenness, Bridges, Closeness, Degree, Eigenvector, PageRank and Louvain **entirely via
Cypher/GDS procedures inside Neo4j** — no NetworkX/Python centrality computation.

## Pipeline

1. **`build_domain_edges.py`** — streams the 10.9GB source file once, resolves each `P`/`L`
   URL to a registered domain, and aggregates:
   - external link counts per `(src_domain, tgt_domain)` pair → `domain_edges_raw.csv`
   - internal (self-domain) link counts per domain → `domain_nodes_raw.csv`

   Full run: 15,312,737 permalinks, 28,096,121 external links, 51,795,969 internal links,
   3,584,339 distinct domain pairs. Took ~19 min.

2. **`filter_domain_network.py [min_weight] [min_degree]`** — two-stage filter:
   - drop pairs with `weight < min_weight` (default 2 — one-off links)
   - **k-core prune**: repeatedly drop domains with undirected degree `< min_degree`
     until stable (default 30). A weight-only filter barely shrinks node count — a
     domain only needs one heavy edge to survive — so degree-based pruning is what
     actually controls final graph size.
   - `log_weight = ln(1+weight)` is written alongside raw weight so a handful of
     extreme repeat-link pairs (max raw weight seen: 10,025) don't dominate weighted
     centrality.

   Final graph used: **3,649 domains / 168,592 edges** (`min_weight=2 min_degree=30`).
   Re-run with different args to resize the graph — no need to re-parse the 10.9GB source.

3. **Docker Compose (`../docker-compose.yml`)** — `neo4j:5-community` + GDS plugin,
   `./neo4j-import` bind-mounted as Neo4j's import dir. Password in `../.env`
   (gitignored).

4. **`load_and_project.cypher`** — `LOAD CSV` into `:Domain` nodes / `:LINKS_TO` rels,
   then two GDS graph projections: `domainDirected` (NATURAL orientation, for
   PageRank/Eigenvector's classical incoming-link-authority meaning) and
   `domainUndirected` (for connectivity-based Betweenness/Closeness/Degree/Bridges/
   Louvain — Bridges requires undirected).

5. **`run_centrality.cypher`** — `gds.degree/betweenness/closeness/louvain.write` on
   `domainUndirected`, `gds.eigenvector/pageRank.write` on `domainDirected`,
   `gds.bridges.stream` (no write mode) inline for inspection in Neo4j Browser too.

6. **`visualize_centrality.py`** — pulls the already-computed properties back out via
   the `neo4j` Python driver (never recomputes them) and draws a 2×4 subplot figure:
   one panel per metric (node size ∝ metric, color = Louvain community), a dedicated
   community panel, and a bridges panel. Only draws the top 200-by-degree domains for
   legibility — GDS itself computed over the full 3,649-node graph.

## Results

Top domains by PageRank/Betweenness/Degree are all plausible real-world hubs:
`youtube.com`, `en.wikipedia.org`, `nytimes.com`, `amazon.com`, `twitter.com`,
`google.com`, `guardian.co.uk`, `washingtonpost.com` — this validates the domain-collapse
+ k-core filtering approach (an earlier exploratory pass showed raw link-count-based
filtering was dominated by widget/boilerplate domains; distinct-domain degree fixed that).

**Bridges = 0.** Not a bug: `min_degree=30` k-core pruning guarantees every domain has
≥30 neighbors, which makes the resulting network extremely densely/redundantly
connected (avg degree ≈92) — a legitimate structural finding (no single point of
failure at the domain level), not an empty/failed computation.

Louvain found 8 communities, modularity ≈0.298.

## Re-running

```bash
# 1) one-time, ~19 min for the full 10.9GB source
python build_domain_edges.py

# 2) fast, re-run freely to resize the graph
python filter_domain_network.py 2 30

# 3) start Neo4j (from Neo4j GDS/)
docker compose up -d

# 4) load + project, then compute all 7 metrics (from inside the container or via cypher-shell)
docker cp load_and_project.cypher set50-quotes-neo4j:/tmp/
docker cp run_centrality.cypher set50-quotes-neo4j:/tmp/
docker exec set50-quotes-neo4j cypher-shell -u neo4j -p <password> -f /tmp/load_and_project.cypher
docker exec set50-quotes-neo4j cypher-shell -u neo4j -p <password> -f /tmp/run_centrality.cypher

# 5) render the figure
python visualize_centrality.py
```

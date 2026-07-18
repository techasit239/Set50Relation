// Run with: cat load_and_project.cypher | cypher-shell -u neo4j -p $NEO4J_PASSWORD
// (or paste into Neo4j Browser at http://localhost:7474)

// 1) constraint / index for fast MATCH during relationship load
CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE;

// 2) load nodes
LOAD CSV WITH HEADERS FROM 'file:///domain_nodes.csv' AS row
CREATE (:Domain {
  name: row.domain,
  internalLinkCount: toInteger(row.internal_link_count)
});

// 3) load relationships (weight = raw external link count, logWeight = ln(1+weight))
LOAD CSV WITH HEADERS FROM 'file:///domain_edges.csv' AS row
MATCH (a:Domain {name: row.src_domain})
MATCH (b:Domain {name: row.tgt_domain})
CREATE (a)-[:LINKS_TO {
  weight: toFloat(row.weight),
  logWeight: toFloat(row.log_weight)
}]->(b);

// 4) sanity check
MATCH (n:Domain) RETURN count(n) AS domainCount;
MATCH ()-[r:LINKS_TO]->() RETURN count(r) AS linkCount;

// 5) GDS graph projections
// directed: for PageRank / Eigenvector (classical incoming-link authority)
CALL gds.graph.project(
  'domainDirected',
  'Domain',
  {LINKS_TO: {orientation: 'NATURAL', properties: 'logWeight'}}
);

// undirected: for Betweenness / Closeness / Degree / Bridges / Louvain (connectivity)
CALL gds.graph.project(
  'domainUndirected',
  'Domain',
  {LINKS_TO: {orientation: 'UNDIRECTED', properties: 'logWeight'}}
);

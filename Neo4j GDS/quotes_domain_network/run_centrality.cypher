// Run with: cat run_centrality.cypher | cypher-shell -u neo4j -p $NEO4J_PASSWORD
// (or paste into Neo4j Browser at http://localhost:7474)
// Requires load_and_project.cypher to have been run first (projections domainDirected / domainUndirected).

// Degree
CALL gds.degree.write('domainUndirected', {
  writeProperty: 'degree',
  relationshipWeightProperty: 'logWeight'
});

// Betweenness
CALL gds.betweenness.write('domainUndirected', {
  writeProperty: 'betweenness'
});

// Closeness
CALL gds.closeness.write('domainUndirected', {
  writeProperty: 'closeness'
});

// Louvain community detection
CALL gds.louvain.write('domainUndirected', {
  writeProperty: 'community',
  relationshipWeightProperty: 'logWeight'
});

// Eigenvector (directed: incoming-link authority)
CALL gds.eigenvector.write('domainDirected', {
  writeProperty: 'eigenvector',
  relationshipWeightProperty: 'logWeight'
});

// PageRank (directed: incoming-link authority)
CALL gds.pageRank.write('domainDirected', {
  writeProperty: 'pagerank',
  relationshipWeightProperty: 'logWeight'
});

// Bridges: stream-only (no write mode) — captured live by visualize_centrality.py.
// Included here so it can be inspected directly in Neo4j Browser too:
CALL gds.bridges.stream('domainUndirected')
YIELD from, to
RETURN gds.util.asNode(from).name AS fromDomain, gds.util.asNode(to).name AS toDomain
ORDER BY fromDomain, toDomain;

// Spot-check: top 10 domains by each metric
MATCH (d:Domain) RETURN d.name, d.pagerank ORDER BY d.pagerank DESC LIMIT 10;
MATCH (d:Domain) RETURN d.name, d.betweenness ORDER BY d.betweenness DESC LIMIT 10;
MATCH (d:Domain) RETURN d.name, d.degree ORDER BY d.degree DESC LIMIT 10;

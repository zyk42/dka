"""
Export a DKA knowledge graph (NetworkX node-link JSON) into Neo4j.

Usage:
    dka-export-neo4j --graph data/knowledge_graph.json \
        --uri bolt://localhost:7687 --user neo4j --password secret

Environment variables NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD are used as
fallbacks for the corresponding flags.
"""

from __future__ import annotations

import argparse
import logging
import os

from dka.graph_builder import load_graph
from dka.neo4j_exporter import DEFAULT_REL_TYPE, export_to_neo4j

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a DKA knowledge graph to Neo4j.")
    parser.add_argument("--graph", required=True, help="Path to knowledge_graph.json")
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "password"))
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--rel-type", default=DEFAULT_REL_TYPE,
                        help="Relationship type for all edges (default: RELATED)")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--clear", action="store_true",
                        help="Delete all existing nodes/relationships before import")
    args = parser.parse_args()

    G = load_graph(args.graph)
    stats = export_to_neo4j(
        G,
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        rel_type=args.rel_type,
        batch_size=args.batch_size,
        clear=args.clear,
    )
    print(f"\nNeo4j import complete: {stats['nodes']} nodes, {stats['relationships']} relationships")


if __name__ == "__main__":
    main()

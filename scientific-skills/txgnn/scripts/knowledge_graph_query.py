"""TxGNN Knowledge Graph Query Tool

Explore the TxGNN biomedical knowledge graph: search for diseases, drugs,
and genes; find relationships between entities; trace multi-hop paths;
and export subgraphs for downstream analysis.

The TxGNN knowledge graph contains 17,080 diseases, 7,957 therapeutic
candidates, and associated genes, proteins, and biological relationships.

Usage:
    python knowledge_graph_query.py search --query "Alzheimer" --node-type disease
    python knowledge_graph_query.py neighbors --node-name "Alzheimer disease" --node-type disease
    python knowledge_graph_query.py path --source "metformin" --target "Alzheimer disease"
    python knowledge_graph_query.py subgraph --nodes "metformin,Alzheimer disease" --output subgraph.json
    python knowledge_graph_query.py stats

Requirements:
    pip install TxGNN pandas
"""

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = os.environ.get("TXGNN_DATA", "./data")


# ── Knowledge graph loader ───────────────────────────────────────────────────

class TxGNNKnowledgeGraph:
    """Interface to the TxGNN biomedical knowledge graph.

    Loads the knowledge graph data files that TxGNN downloads and processes,
    providing search, traversal, and export capabilities.

    The graph data is stored by TxGNN in its data folder as processed CSV
    and pickle files. This class reads those files directly for fast queries
    without needing to load the full GNN model.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        """Initialize the knowledge graph from TxGNN data files.

        Args:
            data_dir: Path to the TxGNN data folder. Must contain the
                processed knowledge graph files (downloaded by TxData).
        """
        self.data_dir = data_dir
        self.edges_df = None
        self.nodes = {}          # {node_id: {name, type, idx}}
        self.adj = defaultdict(list)  # {node_id: [(neighbor_id, relation)]}
        self._loaded = False

    def load(self) -> None:
        """Load the knowledge graph from TxGNN data files."""
        if self._loaded:
            return

        print(f"[TxGNN-KG] Loading knowledge graph from {self.data_dir} ...")

        # TxGNN stores its processed data in the data folder.
        # The primary file is the edge list used to build the DGL graph.
        kg_path = os.path.join(self.data_dir, "kg.csv")
        if not os.path.exists(kg_path):
            # Try alternative locations TxGNN may use
            alt_paths = [
                os.path.join(self.data_dir, "kg_directed.csv"),
                os.path.join(self.data_dir, "train.csv"),
                os.path.join(self.data_dir, "data", "kg.csv"),
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    kg_path = alt
                    break
            else:
                # If no pre-built KG file found, trigger TxData download
                print("[TxGNN-KG] Knowledge graph files not found. "
                      "Triggering download via TxData ...")
                from txgnn import TxData
                tx_data = TxData(data_folder_path=self.data_dir)
                # After TxData init, files should be available
                if os.path.exists(os.path.join(self.data_dir, "kg.csv")):
                    kg_path = os.path.join(self.data_dir, "kg.csv")
                else:
                    raise FileNotFoundError(
                        f"Could not find knowledge graph data in {self.data_dir}. "
                        f"Run TxData(data_folder_path='{self.data_dir}') first."
                    )

        # Load edges
        print(f"[TxGNN-KG] Reading edges from {kg_path} ...")
        self.edges_df = pd.read_csv(kg_path)

        # Normalize column names (TxGNN may use different conventions)
        col_map = {}
        for col in self.edges_df.columns:
            lower = col.lower()
            if "relation" in lower and "display" not in lower:
                col_map[col] = "relation"
            elif "x_id" in lower or "head" in lower or "source" == lower:
                col_map[col] = "x_id"
            elif "x_type" in lower or "head_type" in lower:
                col_map[col] = "x_type"
            elif "x_name" in lower or "head_name" in lower:
                col_map[col] = "x_name"
            elif "y_id" in lower or "tail" in lower or "target" == lower:
                col_map[col] = "y_id"
            elif "y_type" in lower or "tail_type" in lower:
                col_map[col] = "y_type"
            elif "y_name" in lower or "tail_name" in lower:
                col_map[col] = "y_name"
        if col_map:
            self.edges_df = self.edges_df.rename(columns=col_map)

        # Build node index and adjacency list
        required_cols = {"x_id", "y_id", "relation"}
        if not required_cols.issubset(set(self.edges_df.columns)):
            # Fallback: assume columns are positional (relation, x_id, x_type, x_name, y_id, y_type, y_name)
            if len(self.edges_df.columns) >= 7:
                self.edges_df.columns = [
                    "relation", "display_relation",
                    "x_index", "x_id", "x_type", "x_name", "x_source",
                    "y_index", "y_id", "y_type", "y_name", "y_source",
                ][:len(self.edges_df.columns)]

        print("[TxGNN-KG] Building node index and adjacency list ...")
        for _, row in self.edges_df.iterrows():
            x_id = str(row.get("x_id", ""))
            y_id = str(row.get("y_id", ""))
            relation = str(row.get("relation", ""))

            # Register nodes
            if x_id and x_id not in self.nodes:
                self.nodes[x_id] = {
                    "id": x_id,
                    "name": str(row.get("x_name", x_id)),
                    "type": str(row.get("x_type", "unknown")),
                }
            if y_id and y_id not in self.nodes:
                self.nodes[y_id] = {
                    "id": y_id,
                    "name": str(row.get("y_name", y_id)),
                    "type": str(row.get("y_type", "unknown")),
                }

            # Build adjacency (bidirectional for undirected KG)
            if x_id and y_id:
                self.adj[x_id].append((y_id, relation))
                self.adj[y_id].append((x_id, relation))

        self._loaded = True
        print(f"[TxGNN-KG] Loaded {len(self.nodes):,} nodes, "
              f"{len(self.edges_df):,} edges, "
              f"{self.edges_df['relation'].nunique()} relation types.")

    # ── Search ───────────────────────────────────────────────────────────

    def search_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Search nodes by name substring (case-insensitive).

        Exact matches rank first, then prefix matches, then substring matches.

        Args:
            query: Search string to match against node names.
            node_type: Optional filter by node type (e.g. 'disease', 'drug',
                'gene/protein').
            limit: Maximum number of results.

        Returns:
            List of node dicts with id, name, type, and match rank.
        """
        self.load()
        query_lower = query.lower()
        results = []

        for node_id, info in self.nodes.items():
            if node_type and info["type"] != node_type:
                continue

            name_lower = info["name"].lower()
            if query_lower == name_lower:
                rank = 0  # Exact match
            elif name_lower.startswith(query_lower):
                rank = 1  # Prefix match
            elif query_lower in name_lower:
                rank = 2  # Substring match
            else:
                continue

            results.append({
                "id": info["id"],
                "name": info["name"],
                "type": info["type"],
                "rank": rank,
            })

        results.sort(key=lambda x: (x["rank"], len(x["name"])))
        return results[:limit]

    # ── Neighbors ────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        node_type: Optional[str] = None,
        relation_filter: Optional[str] = None,
        neighbor_type_filter: Optional[str] = None,
        limit: int = 100,
    ) -> Dict:
        """Get direct neighbors of a node.

        Args:
            node_id: Node ID to look up directly.
            node_name: Node name to search for (uses first match).
            node_type: Node type hint for name search.
            relation_filter: Only return neighbors connected by this relation.
            neighbor_type_filter: Only return neighbors of this type.
            limit: Maximum neighbors to return.

        Returns:
            Dict with node info and categorized neighbor list.
        """
        self.load()

        # Resolve node
        resolved_id = node_id
        if resolved_id is None and node_name:
            matches = self.search_nodes(node_name, node_type=node_type, limit=1)
            if not matches:
                return {"error": f"Node not found: {node_name}"}
            resolved_id = matches[0]["id"]

        if resolved_id is None:
            return {"error": "Provide --node-id or --node-name"}

        resolved_id = str(resolved_id)
        node_info = self.nodes.get(resolved_id, {"id": resolved_id, "name": "unknown", "type": "unknown"})

        neighbors_by_relation = defaultdict(list)
        count = 0
        for neighbor_id, relation in self.adj.get(resolved_id, []):
            if relation_filter and relation != relation_filter:
                continue
            neighbor_info = self.nodes.get(neighbor_id, {"id": neighbor_id, "name": "unknown", "type": "unknown"})
            if neighbor_type_filter and neighbor_info["type"] != neighbor_type_filter:
                continue
            neighbors_by_relation[relation].append({
                "id": neighbor_info["id"],
                "name": neighbor_info["name"],
                "type": neighbor_info["type"],
            })
            count += 1
            if count >= limit:
                break

        return {
            "node": node_info,
            "total_neighbors": count,
            "neighbors_by_relation": dict(neighbors_by_relation),
            "relation_counts": {k: len(v) for k, v in neighbors_by_relation.items()},
        }

    # ── Path finding ─────────────────────────────────────────────────────

    def find_paths(
        self,
        source_id: Optional[str] = None,
        source_name: Optional[str] = None,
        target_id: Optional[str] = None,
        target_name: Optional[str] = None,
        max_depth: int = 3,
        max_paths: int = 10,
    ) -> Dict:
        """Find shortest paths between two nodes via BFS.

        Args:
            source_id: Source node ID.
            source_name: Source node name (searched if ID not given).
            target_id: Target node ID.
            target_name: Target node name (searched if ID not given).
            max_depth: Maximum path length in hops (default: 3).
            max_paths: Maximum number of paths to return (default: 10).

        Returns:
            Dict with list of paths. Each path is a list of
            (node_id, node_name, node_type, relation_to_next) tuples.
        """
        self.load()

        # Resolve source
        src = source_id
        if src is None and source_name:
            matches = self.search_nodes(source_name, limit=1)
            if not matches:
                return {"error": f"Source not found: {source_name}"}
            src = matches[0]["id"]

        # Resolve target
        tgt = target_id
        if tgt is None and target_name:
            matches = self.search_nodes(target_name, limit=1)
            if not matches:
                return {"error": f"Target not found: {target_name}"}
            tgt = matches[0]["id"]

        if src is None or tgt is None:
            return {"error": "Both source and target must be specified"}

        src, tgt = str(src), str(tgt)
        src_info = self.nodes.get(src, {"id": src, "name": src, "type": "unknown"})
        tgt_info = self.nodes.get(tgt, {"id": tgt, "name": tgt, "type": "unknown"})

        # BFS for shortest paths
        paths = []
        # Queue items: (current_node, path_so_far)
        # path_so_far: list of (node_id, relation_used_to_get_here)
        queue = deque([(src, [(src, None)])])
        visited_at_depth = {src: 0}

        while queue and len(paths) < max_paths:
            current, path = queue.popleft()
            depth = len(path) - 1

            if depth >= max_depth:
                continue

            for neighbor_id, relation in self.adj.get(current, []):
                new_depth = depth + 1

                # Allow revisiting at same depth for alternative paths
                if neighbor_id in visited_at_depth and visited_at_depth[neighbor_id] < new_depth:
                    continue

                new_path = path + [(neighbor_id, relation)]

                if neighbor_id == tgt:
                    # Found a path - format it
                    formatted = []
                    for i, (nid, rel) in enumerate(new_path):
                        node = self.nodes.get(nid, {"id": nid, "name": nid, "type": "unknown"})
                        formatted.append({
                            "node_id": node["id"],
                            "node_name": node["name"],
                            "node_type": node["type"],
                            "relation": rel,  # relation used to reach this node
                        })
                    paths.append(formatted)
                    if len(paths) >= max_paths:
                        break
                else:
                    visited_at_depth[neighbor_id] = new_depth
                    queue.append((neighbor_id, new_path))

        return {
            "source": src_info,
            "target": tgt_info,
            "paths_found": len(paths),
            "max_depth": max_depth,
            "paths": paths,
        }

    # ── Subgraph export ──────────────────────────────────────────────────

    def export_subgraph(
        self,
        node_names: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
        include_neighbors: bool = False,
        neighbor_depth: int = 1,
    ) -> Dict:
        """Export a subgraph induced by a set of nodes.

        Args:
            node_names: List of node names to include (searched by name).
            node_ids: List of node IDs to include directly.
            include_neighbors: If True, also include direct neighbors of
                the specified nodes.
            neighbor_depth: How many hops of neighbors to include (default: 1).

        Returns:
            Dict with nodes and edges of the subgraph.
        """
        self.load()

        # Resolve all node IDs
        resolved_ids: Set[str] = set()

        if node_ids:
            for nid in node_ids:
                resolved_ids.add(str(nid))

        if node_names:
            for name in node_names:
                matches = self.search_nodes(name, limit=1)
                if matches:
                    resolved_ids.add(matches[0]["id"])
                else:
                    print(f"[TxGNN-KG] Warning: node not found: {name}")

        if not resolved_ids:
            return {"error": "No valid nodes specified"}

        # Optionally expand to neighbors
        if include_neighbors:
            expansion = set(resolved_ids)
            for _ in range(neighbor_depth):
                new_nodes = set()
                for nid in expansion:
                    for neighbor_id, _ in self.adj.get(nid, []):
                        new_nodes.add(neighbor_id)
                expansion = new_nodes - resolved_ids
                resolved_ids.update(expansion)

        # Collect subgraph nodes and edges
        subgraph_nodes = []
        for nid in resolved_ids:
            if nid in self.nodes:
                subgraph_nodes.append(self.nodes[nid])

        subgraph_edges = []
        seen_edges = set()
        for nid in resolved_ids:
            for neighbor_id, relation in self.adj.get(nid, []):
                if neighbor_id in resolved_ids:
                    edge_key = tuple(sorted([nid, neighbor_id])) + (relation,)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        subgraph_edges.append({
                            "source": nid,
                            "source_name": self.nodes.get(nid, {}).get("name", nid),
                            "target": neighbor_id,
                            "target_name": self.nodes.get(neighbor_id, {}).get("name", neighbor_id),
                            "relation": relation,
                        })

        return {
            "node_count": len(subgraph_nodes),
            "edge_count": len(subgraph_edges),
            "nodes": subgraph_nodes,
            "edges": subgraph_edges,
        }

    # ── Statistics ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get summary statistics of the knowledge graph.

        Returns:
            Dict with node/edge counts, type distributions, and relation counts.
        """
        self.load()

        node_type_counts = defaultdict(int)
        for info in self.nodes.values():
            node_type_counts[info["type"]] += 1

        relation_counts = self.edges_df["relation"].value_counts().to_dict()

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges_df),
            "node_types": dict(sorted(node_type_counts.items(), key=lambda x: -x[1])),
            "relation_types": len(relation_counts),
            "relations": dict(sorted(relation_counts.items(), key=lambda x: -x[1])),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore the TxGNN biomedical knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to TxGNN data folder (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for saving results as JSON.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Query command")

    # search
    search_parser = subparsers.add_parser("search", help="Search for nodes by name")
    search_parser.add_argument("--query", type=str, required=True, help="Search query string.")
    search_parser.add_argument("--node-type", type=str, default=None,
                               help="Filter by node type (disease, drug, gene/protein, etc.).")
    search_parser.add_argument("--limit", type=int, default=20, help="Max results (default: 20).")

    # neighbors
    neighbors_parser = subparsers.add_parser("neighbors", help="Get neighbors of a node")
    neighbors_parser.add_argument("--node-name", type=str, default=None, help="Node name to search for.")
    neighbors_parser.add_argument("--node-id", type=str, default=None, help="Node ID.")
    neighbors_parser.add_argument("--node-type", type=str, default=None, help="Node type hint for name search.")
    neighbors_parser.add_argument("--relation", type=str, default=None, help="Filter by relation type.")
    neighbors_parser.add_argument("--neighbor-type", type=str, default=None, help="Filter by neighbor type.")
    neighbors_parser.add_argument("--limit", type=int, default=100, help="Max neighbors (default: 100).")

    # path
    path_parser = subparsers.add_parser("path", help="Find paths between two nodes")
    path_parser.add_argument("--source", type=str, required=True, help="Source node name.")
    path_parser.add_argument("--target", type=str, required=True, help="Target node name.")
    path_parser.add_argument("--source-id", type=str, default=None, help="Source node ID (overrides --source).")
    path_parser.add_argument("--target-id", type=str, default=None, help="Target node ID (overrides --target).")
    path_parser.add_argument("--max-depth", type=int, default=3, help="Max path length in hops (default: 3).")
    path_parser.add_argument("--max-paths", type=int, default=10, help="Max paths to return (default: 10).")

    # subgraph
    sub_parser = subparsers.add_parser("subgraph", help="Export a subgraph")
    sub_parser.add_argument("--nodes", type=str, required=True,
                            help="Comma-separated node names to include.")
    sub_parser.add_argument("--node-ids", type=str, default=None,
                            help="Comma-separated node IDs to include.")
    sub_parser.add_argument("--include-neighbors", action="store_true",
                            help="Include direct neighbors of specified nodes.")
    sub_parser.add_argument("--neighbor-depth", type=int, default=1,
                            help="Hops of neighbors to include (default: 1).")

    # stats
    subparsers.add_parser("stats", help="Show knowledge graph statistics")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.command:
        print("Error: specify a command (search, neighbors, path, subgraph, stats).")
        print("Run with --help for usage.")
        sys.exit(1)

    kg = TxGNNKnowledgeGraph(data_dir=args.data_dir)

    if args.command == "search":
        result = kg.search_nodes(
            query=args.query,
            node_type=args.node_type,
            limit=args.limit,
        )
        output = {"query": args.query, "results": result, "count": len(result)}

    elif args.command == "neighbors":
        result = kg.get_neighbors(
            node_id=args.node_id,
            node_name=args.node_name,
            node_type=args.node_type,
            relation_filter=args.relation,
            neighbor_type_filter=args.neighbor_type,
            limit=args.limit,
        )
        output = result

    elif args.command == "path":
        result = kg.find_paths(
            source_id=args.source_id,
            source_name=args.source,
            target_id=args.target_id,
            target_name=args.target,
            max_depth=args.max_depth,
            max_paths=args.max_paths,
        )
        output = result

    elif args.command == "subgraph":
        node_names = [n.strip() for n in args.nodes.split(",")]
        node_ids = [n.strip() for n in args.node_ids.split(",")] if args.node_ids else None
        result = kg.export_subgraph(
            node_names=node_names,
            node_ids=node_ids,
            include_neighbors=args.include_neighbors,
            neighbor_depth=args.neighbor_depth,
        )
        output = result

    elif args.command == "stats":
        output = kg.get_stats()

    # Display result
    formatted = json.dumps(output, indent=2, default=str)
    print(formatted)

    # Optionally save
    if args.output:
        with open(args.output, "w") as f:
            f.write(formatted)
        print(f"\n[TxGNN-KG] Results saved to {args.output}")


if __name__ == "__main__":
    main()

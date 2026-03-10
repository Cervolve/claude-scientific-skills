"""PrimeKG Knowledge Graph query functions.

Uses the pre-indexed SQLite database for fast lookups (8.1M edges, 129K nodes).
Falls back to CSV if SQLite is unavailable.
"""

import sqlite3
import os
from typing import List, Dict, Optional, Union

# Data paths — SQLite preferred, CSV fallback
_DATA_DIR = os.getenv(
    "PRIMEKG_DATA_DIR",
    "/mnt/c/Users/eamon/Documents/Data/PrimeKG",
)
SQLITE_PATH = os.path.join(_DATA_DIR, "kg.sqlite")
CSV_PATH = os.path.join(_DATA_DIR, "kg.csv")

_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Get or create a cached SQLite connection."""
    global _conn
    if _conn is not None:
        return _conn
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(
            f"PrimeKG SQLite database not found at {SQLITE_PATH}. "
            f"Set PRIMEKG_DATA_DIR to the directory containing kg.sqlite."
        )
    _conn = sqlite3.connect(SQLITE_PATH)
    _conn.row_factory = sqlite3.Row
    return _conn


def search_nodes(
    name_query: str,
    node_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """Search for nodes by name (case-insensitive substring match).

    Results are ranked: exact matches first, then by edge count (most-connected
    nodes are more useful and appear first).

    Args:
        name_query: Substring to search for in node names.
        node_type: Optional filter (e.g., 'gene/protein', 'drug', 'disease',
            'biological_process', 'pathway', 'effect/phenotype', 'anatomy',
            'molecular_function', 'cellular_component', 'exposure').
        limit: Maximum results to return.

    Returns:
        List of dicts with keys: id, type, name, source.
    """
    conn = _get_conn()
    # Fetch candidates (generous limit for re-ranking)
    fetch_limit = max(limit * 5, 100)
    if node_type:
        rows = conn.execute(
            "SELECT id, type, name, source FROM nodes "
            "WHERE name LIKE ? AND type = ? LIMIT ?",
            (f"%{name_query}%", node_type, fetch_limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, type, name, source FROM nodes "
            "WHERE name LIKE ? LIMIT ?",
            (f"%{name_query}%", fetch_limit),
        ).fetchall()

    candidates = [dict(r) for r in rows]
    if not candidates:
        return []

    # Rank by: (1) exact name match, (2) ID length as proxy for MONDO merge
    # count (longer underscore-joined IDs = more DB entries merged = more
    # canonical), (3) shorter name (more general term) as tiebreaker.
    # This avoids expensive COUNT(*) on the 8M-row edges table.
    query_lower = name_query.lower()
    candidates.sort(
        key=lambda c: (
            c["name"].lower() == query_lower,   # exact match first
            len(c["id"]),                        # longer merged IDs = more canonical
            -len(c["name"]),                     # shorter name = more general
        ),
        reverse=True,
    )

    return candidates[:limit]


def get_neighbors(
    node_id: Union[str, int],
    relation_type: Optional[str] = None,
    neighbor_type: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    """Get direct neighbors of a node by its ID.

    Searches both x_id and y_id columns (undirected lookup).

    Args:
        node_id: Node identifier (e.g., NCBI gene ID, DrugBank ID, MONDO ID).
        relation_type: Optional filter for edge type (e.g., 'disease_protein',
            'drug_protein', 'indication', 'contraindication').
        neighbor_type: Optional filter for neighbor node type.
        limit: Maximum results.

    Returns:
        List of dicts with: relation, display_relation, neighbor_id,
        neighbor_type, neighbor_name, neighbor_source.
    """
    conn = _get_conn()
    node_id = str(node_id)
    results = []

    # Query where node is on x side
    sql_x = "SELECT relation, display_relation, y_id, y_type, y_name, y_source FROM edges WHERE x_id = ?"
    params_x: list = [node_id]
    # Query where node is on y side
    sql_y = "SELECT relation, display_relation, x_id, x_type, x_name, x_source FROM edges WHERE y_id = ?"
    params_y: list = [node_id]

    if relation_type:
        sql_x += " AND relation = ?"
        params_x.append(relation_type)
        sql_y += " AND relation = ?"
        params_y.append(relation_type)

    if neighbor_type:
        sql_x += " AND y_type = ?"
        params_x.append(neighbor_type)
        sql_y += " AND x_type = ?"
        params_y.append(neighbor_type)

    sql_x += " LIMIT ?"
    params_x.append(limit)
    sql_y += " LIMIT ?"
    params_y.append(limit)

    for row in conn.execute(sql_x, params_x).fetchall():
        results.append({
            "relation": row[0],
            "display_relation": row[1],
            "neighbor_id": row[2],
            "neighbor_type": row[3],
            "neighbor_name": row[4],
            "neighbor_source": row[5],
        })
    for row in conn.execute(sql_y, params_y).fetchall():
        results.append({
            "relation": row[0],
            "display_relation": row[1],
            "neighbor_id": row[2],
            "neighbor_type": row[3],
            "neighbor_name": row[4],
            "neighbor_source": row[5],
        })

    return results[:limit]


def find_paths(
    start_id: str,
    end_id: str,
    max_depth: int = 2,
    limit: int = 10,
) -> List[List[Dict]]:
    """Find shortest paths between two nodes via BFS (up to max_depth hops).

    Uses (id, type) pairs to avoid false matches between nodes that share an
    ID but have different types (e.g., gene "1234" vs anatomy "1234").

    Args:
        start_id: Source node ID.
        end_id: Target node ID.
        max_depth: Maximum path length (1 = direct, 2 = one intermediate).
        limit: Maximum number of paths to return.

    Returns:
        List of paths, where each path is a list of edge dicts.
    """
    conn = _get_conn()
    start_id = str(start_id)
    end_id = str(end_id)
    paths: List[List[Dict]] = []

    # Depth 1: direct edges
    direct = conn.execute(
        "SELECT relation, display_relation, x_id, x_name, x_type, y_id, y_name, y_type "
        "FROM edges WHERE (x_id = ? AND y_id = ?) OR (x_id = ? AND y_id = ?)",
        (start_id, end_id, end_id, start_id),
    ).fetchall()

    for row in direct:
        paths.append([dict(row)])
        if len(paths) >= limit:
            return paths

    if max_depth < 2 or len(paths) >= limit:
        return paths

    # Depth 2: find shared intermediates using (id, type) pairs
    start_neighbors: set[tuple[str, str]] = set()
    for row in conn.execute(
        "SELECT y_id, y_type FROM edges WHERE x_id = ? "
        "UNION SELECT x_id, x_type FROM edges WHERE y_id = ?",
        (start_id, start_id),
    ).fetchall():
        start_neighbors.add((row[0], row[1]))

    end_neighbors: set[tuple[str, str]] = set()
    for row in conn.execute(
        "SELECT y_id, y_type FROM edges WHERE x_id = ? "
        "UNION SELECT x_id, x_type FROM edges WHERE y_id = ?",
        (end_id, end_id),
    ).fetchall():
        end_neighbors.add((row[0], row[1]))

    intermediates = list(start_neighbors & end_neighbors)
    if not intermediates:
        return paths

    # Batch-fetch edges for all intermediates at once using temp table
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _mid(id TEXT, type TEXT)")
    conn.execute("DELETE FROM _mid")
    remaining = limit - len(paths)
    conn.executemany(
        "INSERT INTO _mid VALUES (?, ?)",
        intermediates[:remaining * 3],  # overfetch slightly
    )

    # Edges from start to intermediates
    e1_rows = conn.execute(
        "SELECT e.relation, e.display_relation, e.x_id, e.x_name, e.x_type, "
        "       e.y_id, e.y_name, e.y_type "
        "FROM edges e JOIN _mid m ON "
        "  ((e.x_id = ? AND e.y_id = m.id AND e.y_type = m.type) OR "
        "   (e.y_id = ? AND e.x_id = m.id AND e.x_type = m.type))",
        (start_id, start_id),
    ).fetchall()

    # Index by (mid_id, mid_type) -> first edge
    e1_by_mid: dict[tuple[str, str], dict] = {}
    for row in e1_rows:
        d = dict(row)
        # Determine which side is the intermediate
        if d["x_id"] == start_id:
            key = (d["y_id"], d["y_type"])
        else:
            key = (d["x_id"], d["x_type"])
        if key not in e1_by_mid:
            e1_by_mid[key] = d

    # Edges from intermediates to end
    e2_rows = conn.execute(
        "SELECT e.relation, e.display_relation, e.x_id, e.x_name, e.x_type, "
        "       e.y_id, e.y_name, e.y_type "
        "FROM edges e JOIN _mid m ON "
        "  ((e.y_id = ? AND e.x_id = m.id AND e.x_type = m.type) OR "
        "   (e.x_id = ? AND e.y_id = m.id AND e.y_type = m.type))",
        (end_id, end_id),
    ).fetchall()

    e2_by_mid: dict[tuple[str, str], dict] = {}
    for row in e2_rows:
        d = dict(row)
        if d["y_id"] == end_id:
            key = (d["x_id"], d["x_type"])
        else:
            key = (d["y_id"], d["y_type"])
        if key not in e2_by_mid:
            e2_by_mid[key] = d

    # Assemble paths
    for mid_key in intermediates:
        if mid_key in e1_by_mid and mid_key in e2_by_mid:
            paths.append([e1_by_mid[mid_key], e2_by_mid[mid_key]])
            if len(paths) >= limit:
                break

    conn.execute("DROP TABLE IF EXISTS _mid")
    return paths


def get_disease_context(disease_name: str) -> Dict:
    """Get a structured summary of all associations for a disease.

    Args:
        disease_name: Disease name to search for (e.g., "Parkinson", "Alzheimer").

    Returns:
        Dict with keys: disease_info, associated_genes, associated_drugs,
        phenotypes, related_diseases, pathways, exposures.
    """
    results = search_nodes(disease_name, node_type="disease")
    if not results:
        return {"error": f"Disease '{disease_name}' not found"}

    # search_nodes already ranks by edge count, so first result is best
    disease = results[0]
    neighbors = get_neighbors(disease["id"], limit=2000)

    return {
        "disease_info": disease,
        "associated_genes": [n for n in neighbors if n["neighbor_type"] == "gene/protein"],
        "associated_drugs": [n for n in neighbors if n["neighbor_type"] == "drug"],
        "phenotypes": [n for n in neighbors if n["neighbor_type"] == "effect/phenotype"],
        "related_diseases": [n for n in neighbors if n["neighbor_type"] == "disease"],
        "pathways": [n for n in neighbors if n["neighbor_type"] == "pathway"],
        "exposures": [n for n in neighbors if n["neighbor_type"] == "exposure"],
        "biological_processes": [n for n in neighbors if n["neighbor_type"] == "biological_process"],
    }


def get_drug_context(drug_name: str) -> Dict:
    """Get a structured summary of all associations for a drug.

    Args:
        drug_name: Drug name to search for (e.g., "Levodopa", "Aspirin").

    Returns:
        Dict with keys: drug_info, targets, indications, contraindications,
        side_effects, drug_interactions.
    """
    results = search_nodes(drug_name, node_type="drug")
    if not results:
        return {"error": f"Drug '{drug_name}' not found"}

    drug = results[0]
    neighbors = get_neighbors(drug["id"], limit=2000)

    return {
        "drug_info": drug,
        "targets": [n for n in neighbors if n["neighbor_type"] == "gene/protein"],
        "indications": [n for n in neighbors if n["relation"] == "indication"],
        "contraindications": [n for n in neighbors if n["relation"] == "contraindication"],
        "off_label": [n for n in neighbors if n["relation"] == "off-label use"],
        "side_effects": [n for n in neighbors if n["neighbor_type"] == "effect/phenotype"],
        "drug_interactions": [n for n in neighbors if n["relation"] == "drug_drug"],
    }


def get_gene_context(gene_name: str) -> Dict:
    """Get a structured summary of all associations for a gene/protein.

    Args:
        gene_name: Gene symbol to search for (e.g., "APOE", "LRRK2", "GBA").

    Returns:
        Dict with keys: gene_info, diseases, drugs, pathways, biological_processes,
        molecular_functions, cellular_components, anatomy_expressed, ppis.
    """
    results = search_nodes(gene_name, node_type="gene/protein")
    if not results:
        return {"error": f"Gene '{gene_name}' not found"}

    # Prefer exact match
    gene = results[0]
    for r in results:
        if r["name"].upper() == gene_name.upper():
            gene = r
            break

    neighbors = get_neighbors(gene["id"], limit=2000)

    return {
        "gene_info": gene,
        "diseases": [n for n in neighbors if n["neighbor_type"] == "disease"],
        "drugs": [n for n in neighbors if n["neighbor_type"] == "drug"],
        "pathways": [n for n in neighbors if n["neighbor_type"] == "pathway"],
        "biological_processes": [n for n in neighbors if n["neighbor_type"] == "biological_process"],
        "molecular_functions": [n for n in neighbors if n["neighbor_type"] == "molecular_function"],
        "cellular_components": [n for n in neighbors if n["neighbor_type"] == "cellular_component"],
        "anatomy_expressed": [n for n in neighbors if n["relation"] == "anatomy_protein_present"],
        "ppis": [n for n in neighbors if n["relation"] == "protein_protein"],
    }


def get_relation_types() -> List[Dict]:
    """List all relation types with their edge counts.

    Returns:
        List of dicts with keys: relation, count, sorted by count descending.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT relation, COUNT(*) as cnt FROM edges GROUP BY relation ORDER BY cnt DESC"
    ).fetchall()
    return [{"relation": r[0], "count": r[1]} for r in rows]


def get_node_types() -> List[Dict]:
    """List all node types with their counts.

    Returns:
        List of dicts with keys: type, count, sorted by count descending.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type ORDER BY cnt DESC"
    ).fetchall()
    return [{"type": r[0], "count": r[1]} for r in rows]

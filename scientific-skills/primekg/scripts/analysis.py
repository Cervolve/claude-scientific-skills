"""PrimeKG network analysis functions.

Reproducible network-based analyses on the Precision Medicine Knowledge Graph:

- Drug repurposing scoring via metapath evidence
  (Himmelstein et al., eLife 2017; Guney et al., Nat Commun 2016)
- Functional enrichment via hypergeometric ORA with BH FDR
  (Boyle et al., Bioinformatics 2004; Benjamini & Hochberg, JRSS-B 1995)
- Hub gene identification via disease module centrality
  (Menche et al., Science 2015; Barabási et al., Nat Rev Genet 2011)
- Disease similarity via multi-layer Jaccard
  (Goh et al., PNAS 2007; Zhou et al., Nat Commun 2014)
- Subgraph extraction for downstream network analysis

All functions use lazy in-memory caching (~3s first load per relation type,
sub-millisecond thereafter). No external dependencies beyond Python stdlib.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from scripts.query_primekg import _get_conn, search_nodes


# ---------------------------------------------------------------------------
# Relation cache: lazy-load entire relation into memory for fast analysis
# ---------------------------------------------------------------------------

class _RelationCache:
    """Lazily loads a PrimeKG relation into bidirectional gene↔term mappings.

    On first access for a given relation, loads all edges into memory (~3s
    for 85K pathway_protein edges). Subsequent accesses are O(1) dict lookups.
    """

    _caches: dict[str, "_RelationCache"] = {}

    def __init__(self, relation: str):
        self.relation = relation
        self.gene_to_terms: dict[str, set[str]] = defaultdict(set)
        self.term_to_genes: dict[str, set[str]] = defaultdict(set)
        self.term_names: dict[str, str] = {}
        self.gene_names: dict[str, str] = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        conn = _get_conn()
        rows = conn.execute(
            "SELECT x_id, x_name, x_type, y_id, y_name, y_type "
            "FROM edges WHERE relation = ?",
            (self.relation,),
        ).fetchall()

        for x_id, x_name, x_type, y_id, y_name, y_type in rows:
            if x_type == "gene/protein":
                gene_id, gene_name = x_id, x_name
                term_id, term_name = y_id, y_name
            elif y_type == "gene/protein":
                gene_id, gene_name = y_id, y_name
                term_id, term_name = x_id, x_name
            else:
                continue  # skip non-gene edges

            self.gene_to_terms[gene_id].add(term_id)
            self.term_to_genes[term_id].add(gene_id)
            self.term_names[term_id] = term_name
            self.gene_names[gene_id] = gene_name

        self._loaded = True

    @classmethod
    def get(cls, relation: str) -> "_RelationCache":
        if relation not in cls._caches:
            cache = cls(relation)
            cache._load()
            cls._caches[relation] = cache
        return cls._caches[relation]


# Mapping from user-friendly category names to PrimeKG relation types
_ENRICHMENT_CATEGORIES = {
    "pathway": "pathway_protein",
    "biological_process": "bioprocess_protein",
    "molecular_function": "molfunc_protein",
    "cellular_component": "cellcomp_protein",
    "disease": "disease_protein",
    "phenotype": "phenotype_protein",
}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(M, n, N).

    Parameters:
        k: observed overlap (successes in sample)
        M: population size (total genes in background)
        n: successes in population (genes annotated to term)
        N: sample size (user's gene list size)

    Pure Python using math.comb. No scipy required.
    Numerically stable for PrimeKG's population sizes (M ≈ 27,600).
    """
    if k <= 0:
        return 1.0
    upper = min(n, N)
    if k > upper:
        return 0.0
    # Compute in log-space to avoid overflow for large populations
    log_denom = _log_comb(M, N)
    total = 0.0
    for i in range(k, upper + 1):
        log_num = _log_comb(n, i) + _log_comb(M - n, N - i)
        total += math.exp(log_num - log_denom)
    return min(total, 1.0)


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient C(n, k) using lgamma."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _benjamini_hochberg(pvalues: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values in the same order as input.

    Reference: Benjamini Y, Hochberg Y. Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. JRSS-B. 1995;57:289-300.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvalues[i])
    fdr = [0.0] * n
    for rank_minus_1, idx in enumerate(order):
        rank = rank_minus_1 + 1
        fdr[idx] = pvalues[idx] * n / rank
    # Enforce monotonicity (step-up procedure)
    cummin = 1.0
    for idx in reversed(order):
        cummin = min(cummin, fdr[idx])
        fdr[idx] = min(cummin, 1.0)
    return fdr


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Gene resolution helpers
# ---------------------------------------------------------------------------

def _resolve_gene_ids(gene_names: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map gene symbols to PrimeKG node IDs.

    Returns:
        (name_to_id mapping, list of unmapped names)
    """
    conn = _get_conn()
    name_to_id: dict[str, str] = {}
    unmapped: list[str] = []

    for name in gene_names:
        row = conn.execute(
            "SELECT id, name FROM nodes WHERE type = 'gene/protein' AND name = ?",
            (name,),
        ).fetchone()
        if row:
            name_to_id[name] = row[0]
        else:
            # Try case-insensitive
            row = conn.execute(
                "SELECT id, name FROM nodes WHERE type = 'gene/protein' "
                "AND name LIKE ? LIMIT 1",
                (name,),
            ).fetchone()
            if row:
                name_to_id[name] = row[0]
            else:
                unmapped.append(name)

    return name_to_id, unmapped


def _resolve_best_disease(name: str) -> Optional[Dict]:
    """Resolve a disease name to its best PrimeKG node."""
    results = search_nodes(name, node_type="disease")
    return results[0] if results else None


def _resolve_best_drug(name: str) -> Optional[Dict]:
    """Resolve a drug name to its best PrimeKG node."""
    results = search_nodes(name, node_type="drug")
    return results[0] if results else None


def _get_associated_gene_ids(node_id: str, relation: str) -> Set[str]:
    """Get gene IDs connected to a node by a specific relation."""
    conn = _get_conn()
    node_id = str(node_id)
    ids = set()
    for row in conn.execute(
        "SELECT y_id FROM edges WHERE x_id = ? AND relation = ? AND y_type = 'gene/protein'",
        (node_id, relation),
    ).fetchall():
        ids.add(row[0])
    for row in conn.execute(
        "SELECT x_id FROM edges WHERE y_id = ? AND relation = ? AND x_type = 'gene/protein'",
        (node_id, relation),
    ).fetchall():
        ids.add(row[0])
    return ids


def _get_associated_ids(node_id: str, relation: str, neighbor_type: str) -> Set[str]:
    """Get IDs of a specific type connected to a node by a relation."""
    conn = _get_conn()
    node_id = str(node_id)
    ids = set()
    for row in conn.execute(
        "SELECT y_id FROM edges WHERE x_id = ? AND relation = ? AND y_type = ?",
        (node_id, relation, neighbor_type),
    ).fetchall():
        ids.add(row[0])
    for row in conn.execute(
        "SELECT x_id FROM edges WHERE y_id = ? AND relation = ? AND x_type = ?",
        (node_id, relation, neighbor_type),
    ).fetchall():
        ids.add(row[0])
    return ids


def _id_to_name(node_ids: Set[str], node_type: str) -> Dict[str, str]:
    """Batch-resolve node IDs to names."""
    if not node_ids:
        return {}
    conn = _get_conn()
    id_list = list(node_ids)
    placeholders = ",".join("?" * len(id_list))
    rows = conn.execute(
        f"SELECT id, name FROM nodes WHERE id IN ({placeholders}) AND type = ?",
        id_list + [node_type],
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# ═══════════════════════════════════════════════════════════════════════════
# 1. DRUG REPURPOSING SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_drug_repurposing(drug_name: str, disease_name: str) -> Dict:
    """Score drug repurposing potential via multi-evidence network analysis.

    Computes metapath-based evidence inspired by Himmelstein et al. (eLife
    2017) and network proximity from Guney et al. (Nat Commun 2016).

    Evidence components:
    1. **Shared targets** (CbGaD metapath): genes that are both drug targets
       and disease-associated. Quantified by count and Jaccard index.
    2. **PPI bridges** (CbGiGaD metapath): drug targets connected to disease
       genes via one protein-protein interaction hop. These suggest indirect
       mechanistic links even when the drug doesn't directly target a
       disease gene.
    3. **Shared pathways**: Reactome pathways containing both drug targets
       and disease genes, indicating functional convergence.
    4. **Indication analogs**: other diseases this drug already treats that
       share genes with the query disease (mechanistic plausibility).
    5. **Known status**: existing indication, contraindication, or off-label
       use in PrimeKG.

    The function returns all evidence components separately for transparent
    interpretation rather than collapsing into an opaque composite score.

    Args:
        drug_name: Drug name (e.g., "Metformin", "Ibuprofen").
        disease_name: Disease name (e.g., "Parkinson", "Alzheimer").

    Returns:
        Dict with:
        - drug_info, disease_info: resolved node metadata
        - shared_targets: list of {id, name} dicts
        - shared_target_jaccard: float
        - ppi_bridges: list of {drug_target, bridge_gene, disease_gene} paths
        - ppi_bridge_fraction: fraction of drug targets with a PPI bridge
        - shared_pathways: list of {id, name, drug_genes, disease_genes}
        - indication_analogs: diseases this drug treats sharing genes
        - known_status: 'indication', 'contraindication', 'off-label', or None
        - evidence_lines: count of independent positive evidence types (0-5)

    References:
        - Himmelstein DS et al. Systematic integration of biomedical knowledge
          prioritizes drugs for repurposing. eLife. 2017;6:e26726.
          doi:10.7554/eLife.26726
        - Guney E et al. Network-based in silico drug efficacy screening.
          Nat Commun. 2016;7:10331. doi:10.1038/ncomms10331
        - Chandak P et al. Building a knowledge graph to enable precision
          medicine. Sci Data. 2023;10:67. doi:10.1038/s41597-023-01960-3
    """
    drug = _resolve_best_drug(drug_name)
    disease = _resolve_best_disease(disease_name)
    if not drug:
        return {"error": f"Drug '{drug_name}' not found"}
    if not disease:
        return {"error": f"Disease '{disease_name}' not found"}

    drug_id = drug["id"]
    disease_id = disease["id"]

    # --- 1. Shared targets ---
    drug_targets = _get_associated_gene_ids(drug_id, "drug_protein")
    disease_genes = _get_associated_gene_ids(disease_id, "disease_protein")

    shared_ids = drug_targets & disease_genes
    shared_names = _id_to_name(shared_ids, "gene/protein")
    shared_targets = [{"id": gid, "name": shared_names.get(gid, gid)} for gid in shared_ids]
    shared_jaccard = _jaccard(drug_targets, disease_genes)

    # --- 2. PPI bridges (CbGiGaD) ---
    ppi_cache = _RelationCache.get("protein_protein")
    ppi_bridges = []
    bridge_drug_targets = set()

    for dt_id in drug_targets:
        if dt_id in shared_ids:
            continue  # skip direct shared targets
        ppi_neighbors = ppi_cache.gene_to_terms.get(dt_id, set())
        # For PPI, "terms" are also genes — gene_to_terms maps gene→PPI_partners
        # But _RelationCache treats non-gene side as "term". For protein_protein,
        # both sides are genes, so we need both directions.
        ppi_partners = set()
        if dt_id in ppi_cache.gene_to_terms:
            ppi_partners |= ppi_cache.gene_to_terms[dt_id]
        if dt_id in ppi_cache.term_to_genes:
            ppi_partners |= ppi_cache.term_to_genes[dt_id]

        bridge_genes = ppi_partners & disease_genes
        for bg_id in bridge_genes:
            dt_name = ppi_cache.gene_names.get(dt_id, dt_id)
            bg_name = ppi_cache.gene_names.get(bg_id, ppi_cache.term_names.get(bg_id, bg_id))
            ppi_bridges.append({
                "drug_target": dt_name,
                "drug_target_id": dt_id,
                "bridge_gene": bg_name,
                "bridge_gene_id": bg_id,
            })
            bridge_drug_targets.add(dt_id)

    ppi_bridge_fraction = len(bridge_drug_targets) / max(len(drug_targets - shared_ids), 1)

    # --- 3. Shared pathways ---
    pathway_cache = _RelationCache.get("pathway_protein")
    drug_pathways: dict[str, set[str]] = defaultdict(set)
    for dt_id in drug_targets:
        for pw_id in pathway_cache.gene_to_terms.get(dt_id, set()):
            drug_pathways[pw_id].add(dt_id)

    disease_pathways: dict[str, set[str]] = defaultdict(set)
    for dg_id in disease_genes:
        for pw_id in pathway_cache.gene_to_terms.get(dg_id, set()):
            disease_pathways[pw_id].add(dg_id)

    shared_pw_ids = set(drug_pathways) & set(disease_pathways)
    shared_pathways = []
    for pw_id in shared_pw_ids:
        dt_names = _id_to_name(drug_pathways[pw_id], "gene/protein")
        dg_names = _id_to_name(disease_pathways[pw_id], "gene/protein")
        shared_pathways.append({
            "id": pw_id,
            "name": pathway_cache.term_names.get(pw_id, pw_id),
            "drug_genes": [dt_names.get(g, g) for g in drug_pathways[pw_id]],
            "disease_genes": [dg_names.get(g, g) for g in disease_pathways[pw_id]],
        })
    shared_pathways.sort(key=lambda x: len(x["drug_genes"]) + len(x["disease_genes"]), reverse=True)

    # --- 4. Indication analogs ---
    conn = _get_conn()
    drug_indications = _get_associated_ids(drug_id, "indication", "disease")
    drug_offlabel = _get_associated_ids(drug_id, "off-label use", "disease")
    treated_diseases = drug_indications | drug_offlabel

    indication_analogs = []
    for td_id in treated_diseases:
        if td_id == disease_id:
            continue
        td_genes = _get_associated_gene_ids(td_id, "disease_protein")
        overlap = td_genes & disease_genes
        if overlap:
            td_name_row = conn.execute(
                "SELECT name FROM nodes WHERE id = ? AND type = 'disease' LIMIT 1",
                (td_id,),
            ).fetchone()
            td_name = td_name_row[0] if td_name_row else td_id
            overlap_names = _id_to_name(overlap, "gene/protein")
            indication_analogs.append({
                "disease_id": td_id,
                "disease_name": td_name,
                "shared_genes_with_query": len(overlap),
                "shared_gene_names": [overlap_names.get(g, g) for g in list(overlap)[:10]],
            })
    indication_analogs.sort(key=lambda x: x["shared_genes_with_query"], reverse=True)

    # --- 5. Known status ---
    known_status = None
    if disease_id in _get_associated_ids(drug_id, "indication", "disease"):
        known_status = "indication"
    elif disease_id in _get_associated_ids(drug_id, "contraindication", "disease"):
        known_status = "contraindication"
    elif disease_id in _get_associated_ids(drug_id, "off-label use", "disease"):
        known_status = "off-label use"

    # --- Evidence lines count ---
    evidence_lines = sum([
        len(shared_targets) > 0,
        len(ppi_bridges) > 0,
        len(shared_pathways) > 0,
        len(indication_analogs) > 0,
        known_status == "indication" or known_status == "off-label use",
    ])

    return {
        "drug_info": drug,
        "disease_info": disease,
        "shared_targets": shared_targets,
        "shared_target_count": len(shared_targets),
        "shared_target_jaccard": round(shared_jaccard, 6),
        "drug_target_count": len(drug_targets),
        "disease_gene_count": len(disease_genes),
        "ppi_bridges": ppi_bridges[:50],  # cap output size
        "ppi_bridge_count": len(ppi_bridges),
        "ppi_bridge_fraction": round(ppi_bridge_fraction, 4),
        "shared_pathways": shared_pathways[:20],
        "shared_pathway_count": len(shared_pathways),
        "indication_analogs": indication_analogs[:10],
        "known_status": known_status,
        "evidence_lines": evidence_lines,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. FUNCTIONAL ENRICHMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def enrichment_analysis(
    gene_names: List[str],
    category: str = "pathway",
    fdr_threshold: float = 0.05,
    min_overlap: int = 2,
    min_term_size: int = 5,
    max_term_size: int = 500,
) -> Dict:
    """Over-representation analysis of a gene list against PrimeKG annotations.

    Performs a standard one-sided hypergeometric test for each annotation
    term, with Benjamini-Hochberg FDR correction for multiple testing.
    Terms are filtered by size (min/max) to focus on biologically
    interpretable results and reduce multiple testing burden.

    Args:
        gene_names: Gene symbols (e.g., ["LRRK2", "SNCA", "PRKN", "GBA"]).
        category: One of 'pathway', 'biological_process', 'molecular_function',
            'cellular_component', 'disease', 'phenotype'.
        fdr_threshold: Maximum FDR q-value for reporting (default 0.05).
        min_overlap: Minimum overlap count to report (default 2).
        min_term_size: Minimum genes per term (default 5, excludes very
            specific terms with low statistical power).
        max_term_size: Maximum genes per term (default 500, excludes very
            broad terms like "metabolic process").

    Returns:
        Dict with:
        - results: list of enriched terms sorted by p-value, each with
          term_id, term_name, overlap_count, overlap_genes, term_size,
          p_value, q_value (BH-adjusted), fold_enrichment
        - parameters: analysis metadata (background size, gene list size, etc.)
        - gene_mapping: {input_name: primekg_id} for resolved genes
        - unmapped_genes: list of gene names that couldn't be resolved

    Statistical method:
        For each term T with |T| annotated genes in a background of M genes,
        given a query list of N genes with k overlapping T:

            P(X >= k) = Σ_{i=k}^{min(|T|,N)} C(|T|,i)·C(M-|T|,N-i) / C(M,N)

        Multiple testing corrected via Benjamini-Hochberg step-up procedure
        to control FDR at the specified threshold.

    References:
        - Boyle EI et al. GO::TermFinder - open source software for accessing
          Gene Ontology information. Bioinformatics. 2004;20:3710-3715.
          doi:10.1093/bioinformatics/bth456
        - Benjamini Y, Hochberg Y. Controlling the false discovery rate.
          JRSS-B. 1995;57:289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
        - Rivals I et al. Enrichment or depletion of a GO category within a
          class of genes: which test? Bioinformatics. 2007;23:401-407.
          doi:10.1093/bioinformatics/btl633
    """
    if category not in _ENRICHMENT_CATEGORIES:
        return {
            "error": f"Unknown category '{category}'. "
            f"Valid: {list(_ENRICHMENT_CATEGORIES.keys())}",
        }

    relation = _ENRICHMENT_CATEGORIES[category]
    cache = _RelationCache.get(relation)

    # Resolve gene names to IDs
    name_to_id, unmapped = _resolve_gene_ids(gene_names)
    query_ids = set(name_to_id.values())

    if len(query_ids) < 2:
        return {
            "error": f"Only {len(query_ids)} genes resolved. Need at least 2.",
            "gene_mapping": name_to_id,
            "unmapped_genes": unmapped,
        }

    # Background: all genes annotated in this category
    M = len(cache.gene_to_terms)
    N = len(query_ids & set(cache.gene_to_terms.keys()))  # query genes in background

    if N < 2:
        return {
            "error": f"Only {N} query genes found in {category} annotations.",
            "gene_mapping": name_to_id,
            "unmapped_genes": unmapped,
        }

    # Test each term
    raw_results = []
    for term_id, term_genes in cache.term_to_genes.items():
        n = len(term_genes)  # term size
        if n < min_term_size or n > max_term_size:
            continue

        overlap = query_ids & term_genes
        k = len(overlap)
        if k < min_overlap:
            continue

        p_val = _hypergeom_sf(k, M, n, N)
        expected = n * N / M if M > 0 else 0
        fold_enrichment = k / expected if expected > 0 else float("inf")

        overlap_gene_names = {
            gid: name for name, gid in name_to_id.items() if gid in overlap
        }

        raw_results.append({
            "term_id": term_id,
            "term_name": cache.term_names.get(term_id, term_id),
            "overlap_count": k,
            "overlap_genes": [overlap_gene_names.get(g, cache.gene_names.get(g, g)) for g in overlap],
            "term_size": n,
            "p_value": p_val,
            "fold_enrichment": round(fold_enrichment, 2),
        })

    # BH FDR correction
    if raw_results:
        pvals = [r["p_value"] for r in raw_results]
        qvals = _benjamini_hochberg(pvals)
        for r, q in zip(raw_results, qvals):
            r["q_value"] = q

        # Filter by FDR and sort
        results = [r for r in raw_results if r["q_value"] <= fdr_threshold]
        results.sort(key=lambda r: r["p_value"])
    else:
        results = []

    return {
        "results": results,
        "total_terms_tested": len(raw_results),
        "significant_terms": len(results),
        "parameters": {
            "category": category,
            "background_genes": M,
            "query_genes_in_background": N,
            "query_genes_total": len(query_ids),
            "fdr_threshold": fdr_threshold,
            "min_overlap": min_overlap,
            "min_term_size": min_term_size,
            "max_term_size": max_term_size,
            "correction_method": "Benjamini-Hochberg",
        },
        "gene_mapping": name_to_id,
        "unmapped_genes": unmapped,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. HUB GENE IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def rank_hub_genes(disease_name: str, top_n: int = 20) -> Dict:
    """Identify hub genes in a disease-specific PPI subnetwork.

    Builds the "disease module" — the PPI subgraph among disease-associated
    genes — and ranks genes by multi-metric centrality:

    1. **Module degree**: PPI connections within the disease module (genes
       with more intra-module connections are functionally central).
    2. **Total disease associations**: number of distinct diseases linked
       to each gene (broadly pleiotropic genes may be key regulators).
    3. **Druggability**: whether the gene is a known drug target, indicating
       therapeutic tractability.
    4. **Composite hub score**: normalized sum of module degree and disease
       breadth, with a druggability bonus.

    The disease module concept follows Menche et al. (Science 2015), who
    showed that disease genes cluster in PPI networks and that the density
    of this clustering predicts biological and clinical similarity.

    Args:
        disease_name: Disease name (e.g., "Parkinson", "breast cancer").
        top_n: Number of top hub genes to return (default 20).

    Returns:
        Dict with:
        - hub_genes: ranked list of {gene_id, gene_name, module_degree,
          disease_count, is_drug_target, hub_score}
        - module_stats: {n_genes, n_edges, density, avg_degree}
        - disease_info: resolved disease node

    References:
        - Menche J et al. Uncovering disease-disease relationships through
          the incomplete interactome. Science. 2015;347:1257601.
          doi:10.1126/science.1257601
        - Barabási AL et al. Network medicine: a network-based approach to
          human disease. Nat Rev Genet. 2011;12:56-68. doi:10.1038/nrg2918
        - Ghiassian SD et al. A DIseAse MOdule Detection (DIAMOnD) algorithm
          derived from a systematic analysis of connectivity patterns of
          disease proteins in the human interactome. PLoS Comput Biol.
          2015;11:e1004120. doi:10.1371/journal.pcbi.1004120
    """
    disease = _resolve_best_disease(disease_name)
    if not disease:
        return {"error": f"Disease '{disease_name}' not found"}

    disease_id = disease["id"]
    disease_genes = _get_associated_gene_ids(disease_id, "disease_protein")

    if len(disease_genes) < 3:
        return {
            "error": f"Only {len(disease_genes)} genes found for '{disease['name']}'. Need >= 3.",
            "disease_info": disease,
        }

    # Build disease module: PPI edges within disease genes
    ppi_cache = _RelationCache.get("protein_protein")
    module_edges: list[tuple[str, str]] = []
    module_degree: dict[str, int] = defaultdict(int)

    for gene_id in disease_genes:
        # PPI partners (both directions in cache)
        partners = set()
        if gene_id in ppi_cache.gene_to_terms:
            partners |= ppi_cache.gene_to_terms[gene_id]
        if gene_id in ppi_cache.term_to_genes:
            partners |= ppi_cache.term_to_genes[gene_id]

        intra_module = partners & disease_genes
        for partner_id in intra_module:
            if gene_id < partner_id:  # avoid double-counting
                module_edges.append((gene_id, partner_id))
            module_degree[gene_id] += 1

    # Disease breadth: how many diseases is each gene linked to?
    disease_cache = _RelationCache.get("disease_protein")
    disease_count: dict[str, int] = {}
    for gene_id in disease_genes:
        disease_count[gene_id] = len(disease_cache.gene_to_terms.get(gene_id, set()))

    # Druggability: is this gene a known drug target?
    drug_cache = _RelationCache.get("drug_protein")
    is_drug_target: dict[str, bool] = {}
    for gene_id in disease_genes:
        has_drug = bool(drug_cache.gene_to_terms.get(gene_id, set()))
        # Also check reverse mapping (gene as "term" for drug_protein)
        if not has_drug:
            has_drug = bool(drug_cache.term_to_genes.get(gene_id, set()))
        is_drug_target[gene_id] = has_drug

    # Gene names
    gene_names = _id_to_name(disease_genes, "gene/protein")

    # Composite hub score: normalized degree + disease breadth + druggability
    max_degree = max(module_degree.values()) if module_degree else 1
    max_disease = max(disease_count.values()) if disease_count else 1

    hub_genes = []
    for gene_id in disease_genes:
        deg = module_degree.get(gene_id, 0)
        dc = disease_count.get(gene_id, 0)
        druggable = is_drug_target.get(gene_id, False)

        # Score: 50% module degree + 30% disease breadth + 20% druggability
        score = (
            0.5 * (deg / max_degree)
            + 0.3 * (dc / max_disease)
            + 0.2 * (1.0 if druggable else 0.0)
        )

        hub_genes.append({
            "gene_id": gene_id,
            "gene_name": gene_names.get(gene_id, gene_id),
            "module_degree": deg,
            "disease_count": dc,
            "is_drug_target": druggable,
            "hub_score": round(score, 4),
        })

    hub_genes.sort(key=lambda x: x["hub_score"], reverse=True)

    # Module statistics
    n_genes = len(disease_genes)
    n_edges = len(module_edges)
    max_edges = n_genes * (n_genes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0
    avg_degree = 2 * n_edges / n_genes if n_genes > 0 else 0

    return {
        "hub_genes": hub_genes[:top_n],
        "total_disease_genes": n_genes,
        "module_stats": {
            "n_genes": n_genes,
            "n_edges": n_edges,
            "density": round(density, 6),
            "avg_degree": round(avg_degree, 2),
        },
        "disease_info": disease,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. DISEASE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def disease_similarity(disease_a_name: str, disease_b_name: str) -> Dict:
    """Compute multi-layer similarity between two diseases.

    Measures overlap across five evidence layers, extending the human disease
    network framework (Goh et al., PNAS 2007):

    1. **Gene overlap**: Jaccard on disease-associated gene sets, with
       hypergeometric p-value for statistical significance.
    2. **Phenotype overlap**: Jaccard on associated phenotype/symptom sets.
    3. **Drug overlap**: Jaccard on drug association sets (shared treatments
       suggest mechanistic similarity).
    4. **Exposure overlap**: Jaccard on environmental/chemical exposure sets.
    5. **Pathway overlap**: Jaccard on pathways implicated via disease genes.

    Combined similarity is the mean of per-layer Jaccard scores, weighted
    by evidence availability (layers with zero entities in either disease
    are excluded from the mean).

    Args:
        disease_a_name: First disease name (e.g., "Parkinson").
        disease_b_name: Second disease name (e.g., "Alzheimer").

    Returns:
        Dict with per-layer {jaccard, shared_count, shared_names, a_count,
        b_count}, gene_overlap_pvalue, combined_similarity, and metadata.

    References:
        - Goh KI et al. The human disease network. PNAS. 2007;104:8685-8690.
          doi:10.1073/pnas.0701361104
        - Zhou X et al. Human symptoms-disease network. Nat Commun.
          2014;5:4212. doi:10.1038/ncomms5212
        - Menche J et al. Uncovering disease-disease relationships through
          the incomplete interactome. Science. 2015;347:1257601.
          doi:10.1126/science.1257601
    """
    da = _resolve_best_disease(disease_a_name)
    db = _resolve_best_disease(disease_b_name)
    if not da:
        return {"error": f"Disease '{disease_a_name}' not found"}
    if not db:
        return {"error": f"Disease '{disease_b_name}' not found"}

    layers = {
        "genes": ("disease_protein", "gene/protein"),
        "phenotypes": ("disease_phenotype_positive", "effect/phenotype"),
        "drugs_indication": ("indication", "drug"),
        "drugs_contraindication": ("contraindication", "drug"),
        "exposures": ("exposure_disease", "exposure"),
    }

    result_layers: dict[str, dict] = {}
    jaccards_for_mean: list[float] = []

    # Background gene count for hypergeometric test
    conn = _get_conn()
    total_genes = conn.execute(
        "SELECT COUNT(DISTINCT id) FROM nodes WHERE type = 'gene/protein'"
    ).fetchone()[0]

    for layer_name, (relation, neighbor_type) in layers.items():
        set_a = _get_associated_ids(da["id"], relation, neighbor_type)
        set_b = _get_associated_ids(db["id"], relation, neighbor_type)

        shared = set_a & set_b
        jacc = _jaccard(set_a, set_b)

        shared_names_map = _id_to_name(shared, neighbor_type) if shared else {}
        shared_name_list = list(shared_names_map.values())[:20]

        layer_result: dict = {
            "jaccard": round(jacc, 6),
            "shared_count": len(shared),
            "shared_names": shared_name_list,
            "a_count": len(set_a),
            "b_count": len(set_b),
        }

        # Hypergeometric p-value for gene overlap
        if layer_name == "genes" and set_a and set_b:
            p_val = _hypergeom_sf(
                len(shared), total_genes, len(set_a), len(set_b),
            )
            layer_result["hypergeom_pvalue"] = p_val

        result_layers[layer_name] = layer_result

        # Include in combined mean only if both diseases have entities
        if set_a and set_b:
            jaccards_for_mean.append(jacc)

    # Pathway overlap (derived: genes → pathways)
    pathway_cache = _RelationCache.get("pathway_protein")
    pw_a: set[str] = set()
    pw_b: set[str] = set()
    genes_a = _get_associated_gene_ids(da["id"], "disease_protein")
    genes_b = _get_associated_gene_ids(db["id"], "disease_protein")
    for gid in genes_a:
        pw_a |= pathway_cache.gene_to_terms.get(gid, set())
    for gid in genes_b:
        pw_b |= pathway_cache.gene_to_terms.get(gid, set())

    shared_pw = pw_a & pw_b
    pw_names = {pid: pathway_cache.term_names.get(pid, pid) for pid in shared_pw}
    pw_jacc = _jaccard(pw_a, pw_b)

    result_layers["pathways"] = {
        "jaccard": round(pw_jacc, 6),
        "shared_count": len(shared_pw),
        "shared_names": list(pw_names.values())[:20],
        "a_count": len(pw_a),
        "b_count": len(pw_b),
    }
    if pw_a and pw_b:
        jaccards_for_mean.append(pw_jacc)

    combined = sum(jaccards_for_mean) / len(jaccards_for_mean) if jaccards_for_mean else 0.0

    return {
        "disease_a": da,
        "disease_b": db,
        "layers": result_layers,
        "combined_similarity": round(combined, 6),
        "n_layers_with_data": len(jaccards_for_mean),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. SUBGRAPH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_subgraph(
    node_ids: List[str],
    expand_hops: int = 0,
    relation_types: Optional[List[str]] = None,
) -> Dict:
    """Extract a subgraph induced by a set of nodes.

    Returns all edges where both endpoints are in the specified node set
    (or their 1-hop neighbors if expand_hops=1). Output is structured for
    direct use with NetworkX, igraph, or Cytoscape.

    Args:
        node_ids: List of PrimeKG node IDs.
        expand_hops: 0 = induced subgraph only, 1 = include immediate
            neighbors of all input nodes. Max 1.
        relation_types: Optional list of relation types to include
            (e.g., ["protein_protein", "disease_protein"]). If None,
            all relation types are included.

    Returns:
        Dict with:
        - edges: list of {source_id, source_name, source_type, target_id,
          target_name, target_type, relation, display_relation}
        - nodes: list of {id, name, type, is_seed} (is_seed=True for
          input nodes, False for expanded neighbors)
        - stats: {n_nodes, n_edges, relation_counts}

    Example (NetworkX):
        >>> sg = extract_subgraph(["120892", "6622"], expand_hops=1,
        ...                       relation_types=["protein_protein"])
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> for e in sg["edges"]:
        ...     G.add_edge(e["source_id"], e["target_id"],
        ...                relation=e["relation"])
    """
    conn = _get_conn()
    seed_ids = set(str(nid) for nid in node_ids)

    # Optionally expand by 1 hop
    active_ids = set(seed_ids)
    if expand_hops >= 1:
        for nid in seed_ids:
            rel_filter = ""
            params: list = [nid, nid]
            if relation_types:
                placeholders = ",".join("?" * len(relation_types))
                rel_filter = f" AND relation IN ({placeholders})"
                params += relation_types + relation_types

            for row in conn.execute(
                f"SELECT y_id FROM edges WHERE x_id = ?{rel_filter} "
                f"UNION SELECT x_id FROM edges WHERE y_id = ?{rel_filter}",
                params,
            ).fetchall():
                active_ids.add(row[0])

    # Fetch edges between active nodes
    id_list = list(active_ids)
    if not id_list:
        return {"edges": [], "nodes": [], "stats": {"n_nodes": 0, "n_edges": 0}}

    # Use temp table for efficient IN queries with large node sets
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _sg_nodes(id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _sg_nodes")
    conn.executemany("INSERT OR IGNORE INTO _sg_nodes VALUES (?)", [(i,) for i in id_list])

    rel_filter = ""
    params_edge: list = []
    if relation_types:
        placeholders = ",".join("?" * len(relation_types))
        rel_filter = f" AND e.relation IN ({placeholders})"
        params_edge = list(relation_types)

    edge_rows = conn.execute(
        f"SELECT e.x_id, e.x_name, e.x_type, e.y_id, e.y_name, e.y_type, "
        f"       e.relation, e.display_relation "
        f"FROM edges e "
        f"JOIN _sg_nodes n1 ON e.x_id = n1.id "
        f"JOIN _sg_nodes n2 ON e.y_id = n2.id"
        f"{rel_filter}",
        params_edge,
    ).fetchall()

    edges = []
    seen_edges: set[tuple[str, str, str]] = set()  # deduplicate bidirectional
    node_info: dict[str, dict] = {}
    relation_counts: dict[str, int] = defaultdict(int)

    for x_id, x_name, x_type, y_id, y_name, y_type, rel, disp_rel in edge_rows:
        # PrimeKG stores edges bidirectionally; deduplicate by canonical order
        edge_key = (min(x_id, y_id), max(x_id, y_id), rel)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        edges.append({
            "source_id": x_id,
            "source_name": x_name,
            "source_type": x_type,
            "target_id": y_id,
            "target_name": y_name,
            "target_type": y_type,
            "relation": rel,
            "display_relation": disp_rel,
        })
        relation_counts[rel] += 1
        if x_id not in node_info:
            node_info[x_id] = {"id": x_id, "name": x_name, "type": x_type, "is_seed": x_id in seed_ids}
        if y_id not in node_info:
            node_info[y_id] = {"id": y_id, "name": y_name, "type": y_type, "is_seed": y_id in seed_ids}

    conn.execute("DROP TABLE IF EXISTS _sg_nodes")

    return {
        "edges": edges,
        "nodes": list(node_info.values()),
        "stats": {
            "n_nodes": len(node_info),
            "n_edges": len(edges),
            "n_seed_nodes": len(seed_ids & set(node_info.keys())),
            "relation_counts": dict(relation_counts),
        },
    }

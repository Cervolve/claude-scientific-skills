---
name: primekg
description: Query the Precision Medicine Knowledge Graph (PrimeKG) for multiscale biological data including genes, drugs, diseases, phenotypes, pathways, and exposures.
license: MIT license
metadata:
    skill-author: K-Dense Inc. (PrimeKG original from Harvard MIMS)
---

# PrimeKG Knowledge Graph Skill

## Overview

PrimeKG is a precision medicine knowledge graph integrating 20+ biomedical databases into a single resource. It contains **129,375 nodes** across 10 entity types and **8.1 million edges** across 29 relationship types, covering drug-target, disease-gene, phenotype-disease, pathway, and exposure associations.

**Data sources:** DrugBank, NCBI, MONDO, HPO, UBERON, GO, REACTOME, CTD, STRING, DisGeNET, and more.

**Key capabilities:**
- Search for nodes (genes, drugs, diseases, phenotypes, pathways, anatomical structures)
- Retrieve direct neighbors with relationship type filtering
- Get structured disease/drug/gene context summaries
- Find paths between entities (e.g., drug → intermediate → disease for repurposing)
- **Drug repurposing scoring** via metapath evidence (Himmelstein et al. 2017)
- **Functional enrichment analysis** with hypergeometric ORA + BH FDR correction
- **Hub gene identification** in disease-specific PPI modules (Menche et al. 2015)
- **Disease-disease similarity** via multi-layer Jaccard (Goh et al. 2007)
- **Subgraph extraction** for NetworkX/Cytoscape downstream analysis

**Data access:** SQLite database at `$PRIMEKG_DATA_DIR/kg.sqlite` (default: `/mnt/c/Users/eamon/Documents/Data/PrimeKG/`). All queries use indexed SQL for fast lookups.

## When to Use This Skill

- **Drug repurposing:** Find existing drugs that target genes associated with a disease of interest.
- **Target identification:** Discover gene/protein associations for a disease and prioritise by evidence type.
- **Phenotype analysis:** Understand how symptoms and phenotypes relate to diseases, genes, and exposures.
- **Network pharmacology:** Investigate multi-target drug effects and off-target interactions.
- **Pathway enrichment context:** Annotate gene lists with pathway, biological process, and cellular component associations.
- **Exposure-disease links:** Identify environmental and chemical exposures linked to diseases.

## Node Types

| Type | Count | Source | Example |
|------|-------|--------|---------|
| gene/protein | 27,671 | NCBI | APOE, LRRK2, GBA |
| drug | 7,957 | DrugBank | Levodopa, Aspirin |
| disease | 17,080 | MONDO | Parkinson disease, Alzheimer disease |
| effect/phenotype | 15,311 | HPO | Tremor, Bradykinesia |
| biological_process | 28,642 | GO | Autophagy, Apoptotic process |
| molecular_function | 11,169 | GO | Kinase activity |
| cellular_component | 4,176 | GO | Mitochondrion, Lysosome |
| pathway | 2,516 | REACTOME | Dopamine metabolism |
| anatomy | 14,035 | UBERON | Substantia nigra, Hippocampus |
| exposure | 818 | CTD | Rotenone, MPTP |

## Relationship Types

Top relationships by edge count:

| Relation | Edges | Description |
|----------|-------|-------------|
| anatomy_protein_present | 3,036,406 | Protein expressed in anatomical region |
| drug_drug | 2,672,628 | Drug-drug synergistic interaction |
| protein_protein | 642,150 | Physical protein-protein interaction |
| disease_phenotype_positive | 300,634 | Disease associated with phenotype |
| bioprocess_protein | 289,610 | Protein in biological process |
| cellcomp_protein | 166,804 | Protein in cellular component |
| disease_protein | 160,822 | Gene/protein associated with disease |
| molfunc_protein | 139,060 | Protein molecular function |
| drug_effect | 129,568 | Drug side effect |
| pathway_protein | 85,292 | Protein in pathway |
| indication | 18,776 | Drug approved for disease |
| contraindication | 61,350 | Drug contraindicated for disease |
| off-label use | 5,136 | Drug used off-label for disease |
| exposure_disease | 4,608 | Exposure linked to disease |

## Core Workflow

### 1. Search for Entities

```python
from scripts.query_primekg import search_nodes

# Search for Parkinson's disease
results = search_nodes("Parkinson", node_type="disease")

# Search for a gene
genes = search_nodes("LRRK2", node_type="gene/protein")

# Search across all types
all_matches = search_nodes("dopamine")
```

### 2. Get Neighbors

```python
from scripts.query_primekg import get_neighbors

# Get all neighbors of LRRK2 (use the ID from search_nodes)
neighbors = get_neighbors("120892")

# Filter by relationship type
targets = get_neighbors("DB01235", relation_type="drug_protein")

# Filter by neighbor type
gene_assoc = get_neighbors("12345", neighbor_type="gene/protein")
```

### 3. Structured Context Summaries

```python
from scripts.query_primekg import get_disease_context, get_drug_context, get_gene_context

# Disease context: genes, drugs, phenotypes, pathways, exposures
pd_context = get_disease_context("Parkinson")
print(f"Associated genes: {len(pd_context['associated_genes'])}")
print(f"Associated drugs: {len(pd_context['associated_drugs'])}")

# Drug context: targets, indications, contraindications, side effects
drug_ctx = get_drug_context("Levodopa")

# Gene context: diseases, drugs, pathways, PPIs, expression
gene_ctx = get_gene_context("GBA")
```

### 4. Find Paths Between Entities

```python
from scripts.query_primekg import find_paths

# Find drug-disease paths (direct and 2-hop via shared gene targets)
paths = find_paths(start_id="DB01235", end_id="12345", max_depth=2)
for path in paths:
    print(" -> ".join(f"{e['x_name']} --[{e['relation']}]--> {e['y_name']}" for e in path))
```

### 5. Explore Graph Metadata

```python
from scripts.query_primekg import get_relation_types, get_node_types

# List all relation types with counts
for rt in get_relation_types():
    print(f"{rt['relation']}: {rt['count']:,} edges")

# List all node types with counts
for nt in get_node_types():
    print(f"{nt['type']}: {nt['count']:,} nodes")
```

## Network Analysis

### 6. Drug Repurposing Scoring

Computes metapath-based evidence for drug repurposing potential, inspired by
Himmelstein et al. (eLife 2017) and Guney et al. (Nat Commun 2016). Returns
shared targets, PPI bridges, shared pathways, indication analogs, and known
status — all as separate transparent evidence components.

```python
from scripts.analysis import score_drug_repurposing

result = score_drug_repurposing("Ibuprofen", "Parkinson")
print(f"Shared targets: {result['shared_target_count']} (Jaccard={result['shared_target_jaccard']:.4f})")
print(f"PPI bridges: {result['ppi_bridge_count']}")
print(f"Shared pathways: {result['shared_pathway_count']}")
print(f"Evidence lines: {result['evidence_lines']}/5")
for st in result['shared_targets']:
    print(f"  Target: {st['name']}")
```

### 7. Functional Enrichment Analysis

Standard over-representation analysis (hypergeometric test + Benjamini-Hochberg
FDR). Supports pathway, biological_process, molecular_function, cellular_component,
disease, and phenotype categories.

```python
from scripts.analysis import enrichment_analysis

# Pathway enrichment for PD genes
result = enrichment_analysis(
    gene_names=["LRRK2", "SNCA", "PRKN", "GBA", "PINK1", "VPS35"],
    category="pathway",
    fdr_threshold=0.05,
)
for term in result['results']:
    print(f"{term['term_name']}: {term['overlap_count']} genes, "
          f"FE={term['fold_enrichment']}x, q={term['q_value']:.2e}")

# Biological process enrichment
bp = enrichment_analysis(gene_names=["BRCA1", "TP53", "ATM"], category="biological_process")
```

### 8. Hub Gene Identification

Builds the disease module (PPI subgraph among disease genes) and ranks by
composite centrality: module degree, disease breadth, and druggability.

```python
from scripts.analysis import rank_hub_genes

result = rank_hub_genes("Parkinson", top_n=10)
print(f"Module: {result['module_stats']['n_genes']} genes, "
      f"{result['module_stats']['n_edges']} PPI edges")
for g in result['hub_genes']:
    drug = " [DRUGGABLE]" if g['is_drug_target'] else ""
    print(f"  {g['gene_name']}: deg={g['module_degree']}, "
          f"diseases={g['disease_count']}, score={g['hub_score']:.3f}{drug}")
```

### 9. Disease Similarity

Multi-layer Jaccard similarity across genes, phenotypes, drugs, exposures,
and pathways. Includes hypergeometric p-value for gene overlap significance.

```python
from scripts.analysis import disease_similarity

result = disease_similarity("Parkinson", "Alzheimer")
print(f"Combined similarity: {result['combined_similarity']:.4f}")
for layer, data in result['layers'].items():
    print(f"  {layer}: Jaccard={data['jaccard']:.4f}, shared={data['shared_count']}")
```

### 10. Subgraph Extraction

Extract induced subgraphs for downstream analysis in NetworkX, igraph, or
Cytoscape. Supports 1-hop expansion and relation type filtering.

```python
from scripts.analysis import extract_subgraph

# PPI subgraph for a gene set
sg = extract_subgraph(
    node_ids=["120892", "6622", "5071"],  # LRRK2, SNCA, PRKN
    expand_hops=0,
    relation_types=["protein_protein"],
)
print(f"Nodes: {sg['stats']['n_nodes']}, Edges: {sg['stats']['n_edges']}")

# Convert to NetworkX
import networkx as nx
G = nx.Graph()
for e in sg['edges']:
    G.add_edge(e['source_name'], e['target_name'], relation=e['relation'])
```

## Best Practices

1. **Use `search_nodes` first** to get the correct node ID before calling `get_neighbors` or `find_paths`.
2. **Prefer context functions** (`get_disease_context`, `get_drug_context`, `get_gene_context`) for broad overviews — they structure neighbors by type automatically.
3. **Filter by relation_type** when you only need specific evidence (e.g., `indication` for approved drugs, `disease_protein` for genetic associations).
4. **Combine with other skills:** Use with OpenTargets for genetic evidence scoring, UniProt for protein function, or Semantic Scholar for literature context.
5. **Node IDs vary by source:** Gene IDs are NCBI Entrez, drug IDs are DrugBank (DBxxxxx), disease IDs are MONDO numeric.

## Resources

### Scripts
- `scripts/query_primekg.py`: Search, neighbor lookup, context summaries, path finding.
- `scripts/analysis.py`: Drug repurposing, enrichment, hub genes, disease similarity, subgraph extraction.

### References
- [PrimeKG paper](https://doi.org/10.1038/s41597-023-01960-3) — Chandak et al., Scientific Data 2023
- [GitHub](https://github.com/mims-harvard/PrimeKG) — Harvard MIMS
- [Himmelstein et al. 2017](https://doi.org/10.7554/eLife.26726) — Metapath-based drug repurposing (eLife)
- [Guney et al. 2016](https://doi.org/10.1038/ncomms10331) — Network proximity for drug efficacy (Nat Commun)
- [Menche et al. 2015](https://doi.org/10.1126/science.1257601) — Disease modules in the interactome (Science)
- [Goh et al. 2007](https://doi.org/10.1073/pnas.0701361104) — Human disease network (PNAS)
- [Benjamini & Hochberg 1995](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) — FDR correction (JRSS-B)
- [Boyle et al. 2004](https://doi.org/10.1093/bioinformatics/bth456) — GO::TermFinder enrichment (Bioinformatics)

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
- Enumerate all relationship and node types with counts

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

## Best Practices

1. **Use `search_nodes` first** to get the correct node ID before calling `get_neighbors` or `find_paths`.
2. **Prefer context functions** (`get_disease_context`, `get_drug_context`, `get_gene_context`) for broad overviews — they structure neighbors by type automatically.
3. **Filter by relation_type** when you only need specific evidence (e.g., `indication` for approved drugs, `disease_protein` for genetic associations).
4. **Combine with other skills:** Use with OpenTargets for genetic evidence scoring, UniProt for protein function, or Semantic Scholar for literature context.
5. **Node IDs vary by source:** Gene IDs are NCBI Entrez, drug IDs are DrugBank (DBxxxxx), disease IDs are MONDO numeric.

## Resources

### Scripts
- `scripts/query_primekg.py`: All query functions (SQLite-backed, indexed).

### References
- [PrimeKG paper](https://doi.org/10.1038/s41597-023-01960-3) — Chandak et al., Scientific Data 2023
- [GitHub](https://github.com/mims-harvard/PrimeKG) — Harvard MIMS

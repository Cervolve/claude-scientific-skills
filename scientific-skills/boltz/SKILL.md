---
name: boltz
description: Open-source biomolecular foundation model (MIT license) for structure prediction and binding affinity estimation. Use this skill for protein structure prediction, protein-ligand co-folding, binding affinity prediction, and modeling RNA/DNA-protein complexes. Boltz-2 matches AlphaFold3 accuracy on most targets while also predicting binding affinities at 1000x the speed of physics-based FEP methods.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Boltz: Biomolecular Structure & Affinity Prediction

## Overview

Boltz is an open-source (MIT licensed) biomolecular foundation model for predicting the 3D structure of biomolecular complexes and estimating binding affinities. Boltz-2 jointly models structures and binding affinities, achieving accuracy comparable to AlphaFold3 on most structure prediction benchmarks while additionally providing binding affinity estimates that match physics-based free energy perturbation (FEP) calculations at roughly 1000x the speed (~20 seconds per complex vs. 6-12 hours for FEP).

**Key references:**
- Wohlwend et al. (2025) "Boltz-2: Jointly Modeling Structure and Binding Affinities" (bioRxiv, doi: 10.1101/2025.06.14.659707)
- Wohlwend et al. (2024) "Boltz-1: Democratizing Biomolecular Interaction Modeling" (bioRxiv, doi: 10.1101/2024.11.19.624167)
- FoldBench benchmark: Nature Communications (2025)

## Core Capabilities

### 1. Protein Structure Prediction

Predict 3D structures of proteins and protein complexes from amino acid sequences.

**When to use:**
- Predicting monomeric protein structures
- Modeling protein-protein complex assemblies
- Generating structural hypotheses for experimental design

**Basic CLI usage:**

```bash
boltz predict input.yaml --use_msa_server
```

**YAML input for a single protein:**

```yaml
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
```

**Python API usage:**

```python
import boltz

# Predict from a YAML or FASTA input
boltz.predict(
    "input.yaml",
    out_dir="./predictions",
    use_msa_server=True,
    recycling_steps=3,
    diffusion_samples=1,
)
```

### 2. Protein-Ligand Co-Folding

Model protein-ligand complexes by co-folding protein sequences with small molecule ligands specified as SMILES strings or CCD codes.

**When to use:**
- Predicting binding poses for drug candidates
- Virtual screening of compound libraries
- Understanding protein-ligand interactions

**YAML input for protein-ligand complex:**

```yaml
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
  - ligand:
      id: B
      smiles: "CC(=O)Oc1ccccc1C(=O)O"
```

**Ligand specification options:**

```yaml
# Option 1: SMILES string
- ligand:
    id: B
    smiles: "CC(=O)Oc1ccccc1C(=O)O"

# Option 2: CCD code (Chemical Component Dictionary)
- ligand:
    id: B
    ccd: ATP
```

### 3. Binding Affinity Prediction

Boltz-2's hallmark feature: jointly predict complex structure and estimate binding affinity. This produces two affinity metrics without additional computational cost.

**When to use:**
- Ranking compounds by predicted binding strength
- Hit discovery screening (binder vs. non-binder classification)
- Lead optimization (relative affinity ranking)
- Replacing or supplementing expensive FEP calculations

**Affinity output fields:**
- `affinity_pred_value`: Predicted binding affinity as log10(IC50) in micromolar units. Use for ligand optimization and relative ranking.
- `affinity_probability_binary`: Probability (0-1) that the ligand is a binder. Use for hit-discovery screening.

**YAML input with affinity prediction:**

```yaml
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
  - ligand:
      id: B
      smiles: "CC(=O)Oc1ccccc1C(=O)O"
properties:
  - affinity:
      binder: [A, B]
```

### 4. RNA and DNA-Protein Complex Modeling

Model nucleic acid structures and their complexes with proteins. Boltz-2 shows particularly strong improvements over prior methods on RNA and DNA-protein targets.

**When to use:**
- Predicting RNA secondary and tertiary structures
- Modeling DNA-protein transcription factor complexes
- Studying ribonucleoprotein assemblies

**YAML input for protein-RNA complex:**

```yaml
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
  - rna:
      id: B
      sequence: GGGAAACCC
```

**YAML input for protein-DNA complex:**

```yaml
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
  - dna:
      id: B
      sequence: ATCGATCGATCG
```

## Installation

**Recommended (with CUDA support):**

```bash
pip install boltz[cuda] -U
```

**CPU only (significantly slower):**

```bash
pip install boltz -U
```

**From source (for latest updates):**

```bash
git clone https://github.com/jwohlwend/boltz.git
cd boltz && pip install -e .[cuda]
```

## CLI Reference

```bash
# Basic prediction
boltz predict input.yaml --use_msa_server

# Specify output directory
boltz predict input.yaml --out_dir ./results --use_msa_server

# Batch prediction from a directory of YAML files
boltz predict ./inputs/ --use_msa_server

# All options
boltz predict --help
```

**Key CLI options:**
- `input_path`: Path to YAML file or directory of YAML files
- `--out_dir`: Output directory (default: `./boltz_results`)
- `--use_msa_server`: Use remote MSA server for multiple sequence alignments
- `--recycling_steps`: Number of recycling iterations (default: 3)
- `--diffusion_samples`: Number of diffusion samples to generate (default: 1)
- `--device`: Compute device (`cuda`, `cpu`)

## Output Format

Predictions are written to the output directory with the following structure:

```
boltz_results/
  predictions/
    <input_name>/
      <input_name>_model_0.cif        # Predicted structure (mmCIF)
      confidence_<input_name>_model_0.json  # Confidence metrics
      affinity_<input_name>_model_0.json    # Affinity predictions (if requested)
```

**Confidence metrics include:**
- **pLDDT**: Per-residue confidence (0-1, higher is better; >0.7 is generally reliable)
- **pDE**: Predicted distance error
- **pAE**: Predicted aligned error (inter-chain confidence)

## Model Selection Guide

- **Boltz-2** (default, latest): Joint structure and affinity prediction. Best overall accuracy.
- **Boltz-1**: Structure prediction only. First-generation model.

## Benchmark Context

Based on FoldBench (Nature Communications, 2025) and Boltz-2 preprint evaluations:

- **Structure accuracy**: Boltz-2 matches AlphaFold3 on most protein and complex structure prediction benchmarks.
- **RNA/DNA complexes**: Boltz-2 shows strongest improvements over prior methods on RNA and DNA-protein targets.
- **Antibody-antigen**: AlphaFold3 retains a slight edge on antibody-antigen complexes specifically.
- **Binding affinity**: Boltz-2 achieves ~0.6 Pearson correlation with experimental binding data, comparable to FEP calculations that take 1000x longer.
- **Speed**: ~20 seconds per complex on a single GPU, vs. 6-12 hours for physics-based FEP methods.

## Comparison with AlphaFold3

| Feature | Boltz-2 | AlphaFold3 |
|---|---|---|
| License | MIT (fully open) | Restricted |
| Structure prediction | Comparable accuracy | Slightly better on Ab-Ag |
| Binding affinity | Yes (joint prediction) | No |
| RNA/DNA complexes | Strong (best improvements) | Good |
| Speed (affinity) | ~20s/complex | N/A |
| Commercial use | Allowed | Restricted |

## Known Limitations

- **Newer model**: Less literature validation compared to AlphaFold3 which has been available longer.
- **Affinity calibration**: Binding affinity predictions are best used for relative ranking rather than absolute pKd values.
- **MSA dependency**: Best results require MSA generation, which adds to runtime. Use `--use_msa_server` for convenience.
- **GPU memory**: Large complexes may require significant GPU memory. Consider CPU fallback for very large systems (with patience).
- **Antibody-antigen**: AlphaFold3 currently outperforms Boltz-2 on antibody-antigen interface prediction.

## Common Workflows

### Virtual screening workflow:

1. Prepare protein target YAML with sequence
2. Generate YAML files for each ligand candidate (SMILES)
3. Run batch prediction with affinity: `boltz predict ./ligands_dir/ --use_msa_server`
4. Rank compounds by `affinity_pred_value` or filter by `affinity_probability_binary`

### Structure-guided drug design:

1. Predict protein-ligand complex structure
2. Analyze predicted binding pose and contacts
3. Design modified ligands based on structural insights
4. Re-predict to estimate affinity changes

## Resources and Documentation

- **GitHub Repository:** https://github.com/jwohlwend/boltz
- **Boltz-2 Paper:** doi: 10.1101/2025.06.14.659707
- **Boltz-1 Paper:** doi: 10.1101/2024.11.19.624167
- **License:** MIT - free for academic and commercial use

## Responsible Use

Boltz is designed for scientific research and drug discovery applications. When using binding affinity predictions for drug development, always validate computational predictions with experimental assays. Predicted structures and affinities should inform, not replace, experimental decision-making.

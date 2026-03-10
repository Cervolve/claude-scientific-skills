---
name: structure-based-drug-design
description: State-of-the-art structure-based drug design models NOT available on HuggingFace. Includes PocketXMol (atom-level generative foundation model for pocket-interacting molecules, Cell 2026) and DrugCLIP (contrastive learning for ultra-fast virtual screening, NeurIPS 2023). Use this skill for de novo molecule generation in protein pockets, fragment linking/growing, PROTAC design, peptide design, and large-scale virtual screening. These models require manual installation from GitHub with custom conda environments.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Structure-Based Drug Design

## Overview

This skill provides access to state-of-the-art structure-based drug design models that are **not available on HuggingFace** and require manual installation from their respective GitHub repositories. These models cover two complementary workflows: **de novo molecular generation** (PocketXMol) and **ultra-fast virtual screening** (DrugCLIP).

## Models

### 1. PocketXMol -- Generative Foundation Model for Pocket-Interacting Molecules

PocketXMol is an atom-level generative foundation model that learns atomic interactions within molecular pockets. Published in **Cell (2026)**, it achieves **SOTA on 11 out of 13 tasks** against 55 baselines across structure prediction, molecular design, and complex manipulation.

**Repository:** https://github.com/pengxingang/PocketXMol
**Paper:** Peng et al., "Unified modeling of 3D molecular generation via atomic interactions with PocketXMol", Cell (2026). DOI: 10.1016/j.cell.2026.01.003
**License:** MIT

#### Capabilities

| Category | Tasks |
|---|---|
| Structure Prediction | Small-molecule docking, peptide docking, conformation generation |
| Molecular Design | Structure-based drug design (SBDD), fragment linking, fragment growing, PROTAC design, de novo peptide design, inverse folding |
| Complex Manipulation | Hybrid prediction-design tasks, partial structure generation |

#### Installation

```bash
# Clone repository
git clone https://github.com/pengxingang/PocketXMol.git
cd PocketXMol

# Create conda environment (requires CUDA 11.7)
conda env create -f environment.yml
conda activate pxm

# Download model weights from Zenodo
# https://zenodo.org/records/17801271
wget https://zenodo.org/records/17801271/files/model_weights.tar.gz
tar -zxvf model_weights.tar.gz
```

An alternative environment file `environment_cu128_base.yml` is provided for CUDA 12.8.

#### Usage

```bash
# Basic inference (structure-based drug design)
python scripts/sample_use.py \
  --config_task configs/sample/examples/dock_smallmol.yml \
  --outdir outputs_examples \
  --device cuda:0

# Reduce batch size if GPU OOM
python scripts/sample_use.py \
  --config_task configs/sample/examples/sbdd.yml \
  --outdir outputs_examples \
  --device cuda:0 \
  --batch_size 50
```

Task-specific config files are in `configs/sample/examples/`:
- `dock_smallmol.yml` -- small-molecule docking
- `sbdd.yml` -- structure-based drug design
- `fragment_linking.yml` -- fragment linking
- `fragment_growing.yml` -- fragment growing
- `protac.yml` -- PROTAC design
- `peptide_design.yml` -- de novo peptide design

#### Output Format

Outputs are written to `{exp_name}_{timestamp}/`:
- `{exp_name}_{timestamp}_SDF/` -- Final molecules in SDF or PDB format
- `SDF/` -- Sampling trajectories (if enabled)
- `gen_info.csv` -- Metadata and confidence scores (use `cfd_traj` field for ranking)
- `log.txt` -- Execution logs

#### Limitations

- Requires **CUDA 11.7** (primary environment) or CUDA 12.8 (alternative)
- Complex conda environment with PyTorch Geometric, RDKit, BioPython dependencies
- Model weights hosted on Zenodo (not pip-installable)
- One-person maintained project

---

### 2. DrugCLIP -- Contrastive Learning for Virtual Screening

DrugCLIP uses contrastive protein-molecule representation learning to perform virtual screening **10 million times faster** than physics-based docking methods. Published at **NeurIPS 2023**, evaluated on DUD-E and PCBA benchmarks.

**Repository:** https://github.com/bowen-gao/DrugCLIP
**Paper:** Gao et al., "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening", NeurIPS 2023
**License:** MIT

#### How It Works

DrugCLIP learns joint embeddings of protein pockets and small molecules via contrastive learning (similar to CLIP for images/text). At inference time, it encodes both the target protein and a library of candidate molecules, then ranks molecules by cosine similarity in the learned embedding space. This avoids expensive physics-based docking calculations entirely.

#### Installation

```bash
# Clone repository
git clone https://github.com/bowen-gao/DrugCLIP.git
cd DrugCLIP

# Install dependencies (same as Uni-Mol)
# CRITICAL: rdkit version must be 2022.9.5
pip install rdkit==2022.9.5

# Download data and checkpoints from Google Drive
# (link provided in repository README)
# Extract training data
unzip train_no_test_af.zip
```

#### Data Format

DrugCLIP uses **LMDB databases** containing:
- `atoms` -- ligand atom type identifiers
- `coordinates` -- 3D conformations (up to 10 per ligand)
- `pocket_atoms` -- protein pocket atom types
- `pocket_coordinates` -- protein spatial coordinates
- `mol` -- RDKit molecule objects
- `smi` -- SMILES strings
- `pocket` -- PDB identifiers

#### Usage

```bash
# Training
bash drugclip.sh

# Testing on DUD-E/PCBA
bash test.sh

# Molecular retrieval / virtual screening
bash retrieval.sh
```

Data preprocessing reference: `py_scripts/write_dude_multi.py`
LMDB utilities: `py_scripts/lmdb_utils.py`

#### Limitations

- Authors note: "Currently the code is a raw version, will be updated ASAP"
- Model weights distributed via **Google Drive** (not a stable hosting solution)
- Requires exact `rdkit==2022.9.5`
- Same dependency stack as Uni-Mol (complex setup)

---

## Model Comparison

| Feature | PocketXMol | DrugCLIP | DiffDock | Boltz-2 |
|---|---|---|---|---|
| **Primary Use** | De novo molecule generation | Virtual screening | Blind docking | Structure prediction + docking |
| **Speed** | Minutes per molecule | 10M x faster than docking | Minutes per complex | Minutes per complex |
| **Input** | Protein pocket + task config | Protein + SMILES library | Protein + ligand | Protein + ligand |
| **Output** | Novel 3D molecules (SDF) | Ranked molecule list | Docked poses | Predicted complex structure |
| **On HuggingFace** | No | No | No | Yes |
| **Best For** | Novel drug design, PROTACs, peptides | Screening millions of compounds | Binding pose prediction (unknown site) | Co-folding, covalent docking |
| **Published** | Cell 2026 | NeurIPS 2023 | ICLR 2023 | 2025 |

## Important Warning: DiffDock vs Traditional Docking

> **When the binding site is known**, traditional docking methods (AutoDock Vina, Glide) **outperform DiffDock by 20-25 percentage points** on pose prediction accuracy. DiffDock's advantage is primarily for **blind docking** where the binding site is unknown. If you know the binding pocket, use Vina/Glide for pose prediction or Boltz-2 for co-folding, not DiffDock.

## When to Use Each Model

**Use PocketXMol when:**
- You need to **generate novel molecules** for a known protein pocket
- Designing PROTACs, fragment linkers, or peptide binders
- You want a single model covering multiple drug design tasks
- You need confidence scores for generated molecules

**Use DrugCLIP when:**
- You have a **large existing library** of compounds to screen
- Speed is critical (millions of compounds in seconds)
- You need a fast pre-filter before expensive docking/MD simulations
- Working with DUD-E or PCBA-style screening benchmarks

**Use Boltz-2 (HuggingFace) when:**
- You need accurate **binding pose prediction** for a known ligand
- Co-folding protein-ligand complexes
- Covalent docking
- You want a pip-installable solution

**Use AutoDock Vina when:**
- Binding site is known and you need **accurate pose ranking**
- You need physics-based binding energy estimates
- Established, well-validated workflow is required

## Scripts

- `scripts/pocket_molecule_generation.py` -- Generate novel molecules for a protein pocket using PocketXMol
- `scripts/virtual_screening.py` -- Run ultra-fast virtual screening using DrugCLIP

## References

- Peng et al., "Unified modeling of 3D molecular generation via atomic interactions with PocketXMol", Cell (2026). DOI: 10.1016/j.cell.2026.01.003
- Gao et al., "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening", NeurIPS 2023
- Corso et al., "DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking", ICLR 2023
- Wohlwend et al., "Boltz-2: Towards Accurate and Efficient Biomolecular Structure Prediction", 2025

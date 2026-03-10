---
name: txgnn
description: Zero-shot drug repurposing with graph neural networks across 17,080 diseases and 7,957 therapeutic candidates. Use this skill for predicting novel drug-disease indications and contraindications, especially for rare diseases with limited treatment options. Provides multi-hop explainability for predicted drug-disease relationships. Based on TxGNN (Nature Medicine 2024).
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# TxGNN: Zero-Shot Drug Repurposing with Graph Neural Networks

## Overview

TxGNN is a graph neural network framework for zero-shot therapeutic use prediction. It operates on a comprehensive biomedical knowledge graph spanning 17,080 clinically-recognized diseases and 7,957 therapeutic candidates, enabling drug repurposing predictions even for diseases with no existing treatments. Published in Nature Medicine 2024.

Key result: TxGNN improves indication prediction by 49.2% and contraindication prediction by 35.1% over 8 baselines including BioBERT, HGT, HAN, and network medicine approaches. Predictions align with real-world off-label prescriptions and expert evaluations.

## Core Capabilities

### 1. Drug Repurposing Prediction

Predict candidate therapeutics for any disease in the knowledge graph, including diseases with zero known treatments.

**When to use:**
- Screening drug candidates for rare or understudied diseases
- Identifying repurposing opportunities for existing drugs
- Prioritizing candidates for experimental validation

**Basic usage:**

```python
from txgnn import TxData, TxGNN, TxEval

# Load data and prepare split
TxData = TxData(data_folder_path='./data')
TxData.prepare_split(split='complex_disease', seed=42)

# Initialize and load pretrained model
model = TxGNN(data=TxData,
              weight_bias_track=False,
              proj_name='TxGNN',
              exp_name='TxGNN',
              device='cuda:0')

model.model_initialize(n_hid=100, n_inp=100, n_out=100,
                       proto=True, proto_num=3,
                       attention=False,
                       sim_measure='all_nodes_profile',
                       agg_measure='rarity',
                       num_walks=200, path_length=2)

model.load_pretrained('./model_ckpt')

# Evaluate drug candidates for specific diseases
evaluator = TxEval(model=model)
result = evaluator.eval_disease_centric(
    disease_idxs=[9907.0, 12787.0],
    relation='indication',
    save_result=False
)
```

Use `scripts/drug_repurposing.py` for a ready-made CLI to run predictions.

### 2. Contraindication Prediction

Predict drugs that should NOT be used for a given disease, improving patient safety.

```python
result = evaluator.eval_disease_centric(
    disease_idxs=[9907.0],
    relation='contraindication',
    save_result=False
)
```

### 3. Multi-Hop Explainability (GraphMask)

Generate interpretable reasoning paths explaining why a drug is predicted for a disease.

```python
# Train GraphMask explainer
model.train_graphmask(relation='indication',
                      learning_rate=3e-4,
                      allowance=0.005,
                      epochs_per_layer=3,
                      penalty_scaling=1,
                      valid_per_n=20)

# Retrieve explanation gates
gates = model.retrieve_save_gates('./explanation_output')
# Output: ./explanation_output/graphmask_output_indication.pkl
```

Use `scripts/drug_repurposing.py --explain` to run explainability from the CLI.

### 4. Knowledge Graph Exploration

Explore the underlying biomedical knowledge graph containing diseases, drugs, genes, and their relationships.

Use `scripts/knowledge_graph_query.py` for searching entities, finding paths, and exporting subgraphs.

## Model Architecture

TxGNN uses:
- **Heterogeneous graph neural network** operating on a multi-relational biomedical knowledge graph
- **Metric learning module** (`proto=True`) that retrieves similar diseases for zero-shot augmentation
- **Disease similarity measures**: `all_nodes_profile`, `protein_profile`, or `protein_random_walk`
- **Rarity-weighted aggregation** (`agg_measure='rarity'`) to handle imbalanced disease frequencies

Key hyperparameters:
- `n_hid`, `n_inp`, `n_out`: Embedding dimensions (default 100)
- `proto_num`: Number of similar diseases retrieved (default 3)
- `num_walks`, `path_length`: Random walk parameters for similarity computation

## Data Split Strategies

TxGNN supports multiple evaluation paradigms via `TxData.prepare_split()`:

- **`complex_disease`**: Moves all treatments for selected diseases to test set
- **Disease-area splits**: `cell_proliferation`, `mental_health`, `cardiovascular`, `anemia`, `adrenal_gland`, `autoimmune`, `metabolic_disorder`, `diabetes`, `neurodegenerative`
- **`random`**: Random drug-disease pair shuffling
- **`disease_eval`**: Single disease evaluation (specify `disease_eval_idx`)
- **`full_graph`**: No masking; 95% train, 5% validation

## Installation

```bash
# Create dedicated environment (Python 3.8 required)
conda create --name txgnn_env python=3.8
conda activate txgnn_env

# Install PyTorch (match your CUDA version)
# See https://pytorch.org/ for the correct command

# Install DGL (match your CUDA version)
conda install -c dglteam dgl-cuda11.3==0.5.2

# Install TxGNN
pip install TxGNN
```

For disease-area splits, also install PyTorch Geometric:
```bash
pip install torch-geometric
```

Pretrained model weights are available from the TxGNN GitHub repository (Google Drive link).

## Benchmark Performance

Compared against 8 baselines on drug repurposing tasks:

| Method | Indication AUPRC Improvement | Contraindication AUPRC Improvement |
|--------|-----------------------------|------------------------------------|
| TxGNN vs BioBERT | +49.2% | +35.1% |
| TxGNN vs HGT | significant | significant |
| TxGNN vs HAN | significant | significant |
| TxGNN vs Network Medicine | significant | significant |

- Predictions align with real-world off-label prescription patterns
- Strong zero-shot performance on diseases with no training examples
- Clinician evaluations confirm clinical plausibility of top predictions

## When to Use This Skill

**Use TxGNN when:**
- Screening drug repurposing candidates, especially for rare diseases
- Predicting both indications and contraindications for drug-disease pairs
- Needing explainable predictions with multi-hop reasoning paths
- Working with diseases that have few or no known treatments (zero-shot)
- Prioritizing candidates before expensive experimental validation

**Do NOT use TxGNN for:**
- Clinical decision-making without independent validation
- Replacing clinical trials or regulatory processes
- Diseases or drugs not represented in the knowledge graph
- Real-time clinical dosing or treatment planning
- Predictions requiring temporal or longitudinal modeling

## Limitations

- **Predictions are hypotheses**: All outputs require clinical validation before any therapeutic application
- **Knowledge graph snapshot**: The model is trained on a specific snapshot of biomedical knowledge; newly discovered drugs, diseases, or relationships may not be represented
- **No dosage or pharmacokinetics**: TxGNN predicts therapeutic relevance, not dosing, timing, or pharmacokinetic properties
- **Graph coverage**: Performance depends on the connectivity of entities in the knowledge graph; poorly connected entities yield less reliable predictions

## Resources

- **GitHub Repository:** https://github.com/mims-harvard/TxGNN
- **Publication:** Huang et al., Nature Medicine (2024) - Zero-shot prediction of therapeutic use with geometric deep learning and clinician centered design
- **Demo Notebook:** `TxGNN_Demo.ipynb` in the repository
- **Pretrained Weights:** Available via Google Drive (linked in repository README)

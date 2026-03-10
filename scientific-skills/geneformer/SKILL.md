---
name: geneformer
description: Foundation transformer model pretrained on ~104M single-cell transcriptomes for context-aware predictions in network biology. Use this skill for cell type classification, gene network analysis, disease classification, chromatin dynamics prediction, and in silico perturbation. Requires fine-tuning for most production tasks — zero-shot performance has known limitations. For quick cell type annotation without fine-tuning, prefer scVI or Harmony+scANVI instead.
license: Apache-2.0
metadata:
    skill-author: K-Dense Inc.
---

# Geneformer: Single-Cell Foundation Model

## Overview

Geneformer is a self-supervised transformer model pretrained on large-scale single-cell transcriptomes to enable context-aware predictions in network biology. It uses a novel **rank-value encoding** scheme that ranks genes by expression within each cell and scales by expression across the pretraining corpus, prioritizing biologically informative genes over housekeeping genes.

**Key facts:**
- **Architecture**: Transformer encoder with masked learning objective (15% gene masking)
- **Pretraining data**: ~104 million human single-cell transcriptomes (V2, Dec 2024); V1 used ~30M
- **Tokenization**: Rank-value encoding (non-parametric, reduces batch effects vs. raw counts)
- **HuggingFace model**: `ctheodoris/Geneformer`
- **Publication**: Theodoris et al., *Nature* (2023); Chen et al., *bioRxiv* (2024) for V2

## Model Variants

| Model | Parameters | Input Size | Training Data |
|-------|-----------|-----------|--------------|
| `geneformer-v1-10M` | 10M | 2048 tokens | ~30M transcriptomes |
| `geneformer-v2-104M` | 104M | 4096 tokens | ~104M transcriptomes |
| `geneformer-v2-316M` | 316M | 4096 tokens | ~104M transcriptomes (default) |
| `geneformer-v2-104M_CLcancer` | 104M | 4096 tokens | Cancer domain-tuned |

## Installation

```bash
# Requires git-lfs for model weights
git lfs install

# Clone and install from HuggingFace
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

GPU resources are strongly recommended. See https://geneformer.readthedocs.io/ for full documentation.

## CRITICAL BENCHMARK WARNINGS

**You MUST read this section before using Geneformer in any analysis.**

### Zero-Shot Performance Limitations

- **"Zero-shot evaluation reveals limitations"** (Genome Biology 2025, Microsoft Research): Geneformer consistently ranks lowest on integration metrics among foundation models, retains batch effects in zero-shot mode.
- **"Gene perturbation prediction does not outperform linear baselines"** (Nature Methods 2025): In silico perturbation predictions from foundation models including Geneformer do not outperform simple linear models.
- **Cell type annotation**: Geneformer achieves higher accuracy but lower macro-F1 than scGPT, indicating poor performance on rare cell types.
- **Dataset bias**: Relative strength on blood/immune datasets, underperforms on other tissues.
- **Median gene ranking Pearson R = 0.56** for perturbation prediction tasks.

### Mandatory Guidance

- **ALWAYS fine-tune** -- do NOT use zero-shot for production tasks.
- **Apply batch correction** (e.g., Harmony) on Geneformer embeddings before downstream analysis.
- **Validate against simple baselines** (logistic regression on HVGs, scVI, Harmony+scANVI).

## Core Capabilities

### 1. Cell Type Classification (Fine-Tuned)

Fine-tune Geneformer on labeled single-cell data for cell type annotation.

```python
from geneformer import Classifier

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_type", "states": "all"},
    filter_data={"cell_type": ["unknown"]},  # exclude unlabeled
    training_args={"num_train_epochs": 10, "learning_rate": 5e-5},
    freeze_layers=2,
    num_crossval_splits=5,
    forward_batch_size=100,
    nproc=8,
)

# Run cross-validation with hyperparameter tuning
cc.validate(
    model_directory="path/to/geneformer-v2-104M",
    prepared_input_data_file="tokenized_data.dataset",
    id_class_dict_file="cell_type_dict.pkl",
    output_directory="./results/",
    output_prefix="cell_classification",
    n_hyperopt_trials=20,
)
```

**Use the bundled script for a streamlined workflow:**
```bash
python scripts/cell_type_classification.py \
    --input data.h5ad \
    --label-column cell_type \
    --model-size 104M \
    --epochs 10 \
    --output ./results/
```

### 2. Cell Embedding Extraction

Extract contextual cell embeddings for downstream analysis.

```python
from geneformer import EmbExtractor

embex = EmbExtractor(
    model_type="Pretrained",
    num_classes=0,
    filter_data=None,
    max_ncells=None,
    emb_layer=-1,           # last layer
    emb_label=["cell_type"],
    labels_to_plot=["cell_type"],
    forward_batch_size=100,
    nproc=8,
)

embs = embex.extract_embs(
    model_directory="path/to/geneformer-v2-104M",
    input_data_file="tokenized_data.dataset",
    output_directory="./embeddings/",
    output_prefix="geneformer_embs",
)

# Plot UMAP of embeddings
embex.plot_embs(
    embs=embs,
    plot_style="umap",
    output_directory="./embeddings/",
    output_prefix="geneformer_umap",
)
```

**Use the bundled script:**
```bash
python scripts/extract_embeddings.py \
    --input data.h5ad \
    --model-size 104M \
    --output ./embeddings/
```

### 3. Tokenization (Rank-Value Encoding)

Convert AnnData to Geneformer's tokenized format:

```python
from geneformer import TranscriptomeTokenizer

tk = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_type": "cell_type"},
    nproc=4,
    model_input_size=4096,  # V2 context length
)

tk.tokenize_data(
    data_directory="./loom_files/",
    output_directory="./tokenized/",
    output_prefix="my_dataset",
    file_format="loom",
)
```

**Note**: The tokenizer expects `.loom` files by default. For `.h5ad` input, convert first:
```python
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")
adata.write_loom("data.loom")
```

### 4. In Silico Perturbation

Predict effects of gene perturbations on cell state (zero-shot or fine-tuned):

```python
from geneformer import InSilicoPerturber, InSilicoPerturberStats

isp = InSilicoPerturber(
    perturb_type="delete",
    perturb_rank_shift=None,
    genes_to_perturb="all",
    combos=0,
    anchor_gene=None,
    model_type="Pretrained",
    num_classes=0,
    emb_mode="cell",
    cell_emb_style="mean_pool",
    filter_data={"cell_type": ["cardiomyocyte"]},
    cell_states_to_model=None,
    max_ncells=2000,
    emb_layer=-1,
    forward_batch_size=100,
    nproc=8,
)

isp.perturb_data(
    model_directory="path/to/geneformer-v2-104M",
    input_data_file="tokenized_data.dataset",
    output_directory="./perturbation/",
    output_prefix="gene_perturbation",
)
```

**Warning**: Gene perturbation predictions from foundation models do not consistently outperform linear baselines (Nature Methods 2025). Validate predictions experimentally.

## When to Use Geneformer

**Good use cases (WITH fine-tuning):**
- Cell type classification on well-labeled training data
- Disease state classification
- Gene network / centrality analysis
- Chromatin dynamics (bivalent promoter prediction)
- Therapeutic target identification

**When NOT to use Geneformer:**
- Zero-shot cell type annotation (use Harmony+scANVI or scVI instead)
- Batch integration (Geneformer retains batch effects; use Harmony, scVI, or scANVI)
- Quick exploratory analysis (use scanpy standard workflows)
- Gene perturbation prediction without experimental validation
- Small datasets where simple baselines (logistic regression on HVGs) suffice

## Hyperparameter Tuning

There are **no universally good default hyperparameters** for Geneformer fine-tuning. Always tune:

- `max_learning_rate`: Try 1e-5, 5e-5, 1e-4
- `learning_schedule`: Linear warmup + decay
- `freeze_layers`: 0 (none), 2, or 4 layers frozen
- `num_train_epochs`: 5-20 depending on dataset size

Use `n_hyperopt_trials` in the Classifier to run automated hyperparameter search.

## Bundled Scripts

### scripts/cell_type_classification.py
End-to-end fine-tuning pipeline for cell type classification from an AnnData file:
```bash
python scripts/cell_type_classification.py \
    --input data.h5ad \
    --label-column cell_type \
    --model-size 104M \
    --epochs 10 \
    --output ./results/
```

### scripts/extract_embeddings.py
Extract Geneformer cell embeddings and optionally visualize with UMAP:
```bash
python scripts/extract_embeddings.py \
    --input data.h5ad \
    --model-size 104M \
    --emb-layer -1 \
    --output ./embeddings/ \
    --umap
```

## Resources and Documentation

- **HuggingFace Model**: https://huggingface.co/ctheodoris/Geneformer
- **Documentation**: https://geneformer.readthedocs.io/en/latest/
- **Tutorial**: https://tinyurl.com/geneformertutorial
- **Original Paper (V1)**: Theodoris et al., *Nature* (2023) -- "Transfer learning enables predictions in network biology"
- **V2 Paper**: Chen et al., *bioRxiv* (2024) -- "Quantized multi-task learning for context-specific representations"
- **Benchmark (zero-shot)**: Heimberg et al., *Genome Biology* (2025) -- Microsoft Research zero-shot evaluation
- **Perturbation benchmark**: Squires et al., *Nature Methods* (2025) -- Linear baselines comparison

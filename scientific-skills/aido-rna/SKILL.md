---
name: aido-rna
description: AIDO.RNA is the largest RNA foundation model (1.6B parameters) trained on 42M non-coding RNA sequences from RNAcentral. Use this skill for RNA structure prediction, RNA function prediction, genetic regulation analysis, sequence design, mRNA vaccine design, and RNA embedding extraction. Achieves SOTA on 24/26 RNA understanding tasks across a 26-dataset benchmark spanning 9 task categories.
license: Apache-2.0
metadata:
    skill-author: K-Dense Inc.
---

# AIDO.RNA: RNA Foundation Model

## Overview

AIDO.RNA is the largest RNA foundation model, with 1.6 billion parameters, developed by GenBio AI as part of the AIDO (AI-Driven Digital Organism) platform. It is an encoder-only Transformer trained with masked language modeling (MLM) on 42 million unique non-coding RNA sequences from RNAcentral v24.0 at single-nucleotide resolution.

**HuggingFace model:** `genbio-ai/AIDO.RNA-1.6B`

| Parameter        | Value                    |
|------------------|--------------------------|
| Architecture     | Encoder-only Transformer |
| Parameters       | 1.6B                     |
| Layers           | 32                       |
| Hidden Size      | 2,048                    |
| FFN Hidden Size  | 5,440                    |
| Attention Heads  | 32                       |
| Vocab Size       | 16                       |
| Training Data    | 42M ncRNA sequences (RNAcentral v24.0) |
| Training Objective | Masked Language Modeling (MLM) |

## Installation

**ModelGenerator framework (recommended):**

```bash
pip install modelgenerator
pip install git+https://github.com/genbio-ai/openfold.git@c4aa2fd0d920c06d3fd80b177284a22573528442
```

**From source:**

```bash
git clone https://github.com/genbio-ai/ModelGenerator.git
cd ModelGenerator
pip install -e .
```

A coding-sequence-optimized variant is also available: `genbio-ai/AIDO.RNA-1.6B-CDS`.

## Core Capabilities

### 1. RNA Embedding Extraction

Generate per-nucleotide and mean-pooled embeddings for downstream tasks such as clustering, classification, or similarity analysis.

**When to use:**
- Extracting RNA representations for machine learning pipelines
- Computing sequence similarities across RNA families
- Feature extraction for RNA classification or regression
- Transfer learning for RNA-related tasks

**Basic usage:**

```python
from modelgenerator.tasks import Embed

model = Embed.from_config({"model.backbone": "aido_rna_1b600m"}).eval()
transformed_batch = model.transform({"sequences": ["ACGU", "AGCU"]})
embedding = model(transformed_batch)
print(embedding.shape)
```

**CLI usage:**

```bash
mgen predict --model Embed --model.backbone aido_rna_1b600m \
  --data SequencesDataModule --data.path my_rna_sequences.csv \
  --data.x_col sequence --data.id_col id \
  --config configs/examples/save_predictions.yaml
```

See `scripts/rna_embeddings.py` for a complete FASTA-based embedding pipeline with optional UMAP visualization.

### 2. RNA Secondary Structure Prediction

Predict RNA secondary structure in dot-bracket notation using token-level classification.

**When to use:**
- Predicting base-pairing patterns in RNA sequences
- Analyzing structural motifs (stems, loops, bulges)
- Guiding mRNA vaccine or therapeutic RNA design

**Basic usage:**

```python
import torch
from modelgenerator.tasks import TokenClassification

# 3 classes: unpaired '.', paired '(', paired ')'
model = TokenClassification.from_config({
    "model.backbone": "aido_rna_1b600m",
    "model.n_classes": 3,
}).eval()

transformed_batch = model.transform({"sequences": ["GGGAAACCC"]})
logits = model(transformed_batch)
predictions = torch.argmax(logits, dim=-1)
```

See `scripts/secondary_structure_prediction.py` for a complete pipeline that outputs dot-bracket notation.

### 3. Sequence-level Classification

Classify RNA sequences by function, family, or other categorical properties.

```python
import torch
from modelgenerator.tasks import SequenceClassification

model = SequenceClassification.from_config({
    "model.backbone": "aido_rna_1b600m",
    "model.n_classes": 13,  # e.g., 13 ncRNA families
}).eval()

transformed_batch = model.transform({"sequences": ["ACGU", "AGCU"]})
logits = model(transformed_batch)
predictions = torch.argmax(logits, dim=-1)
```

### 4. Sequence-level Regression

Predict continuous properties of RNA sequences (e.g., stability, expression level).

```python
from modelgenerator.tasks import SequenceRegression

model = SequenceRegression.from_config({
    "model.backbone": "aido_rna_1b600m",
}).eval()

transformed_batch = model.transform({"sequences": ["ACGU", "AGCU"]})
predictions = model(transformed_batch)
```

## Benchmark Performance

AIDO.RNA achieves state-of-the-art results on **24 out of 26 RNA understanding tasks** across a 26-dataset benchmark spanning 9 task categories:

**Secondary Structure Prediction (bpRNA-TS0):**
- AIDO.RNA: **F1 = 0.787** (SOTA)
- RNAErnie: F1 = 0.720
- RiNALMo: F1 = 0.770
- RNA-FM: F1 = 0.695

**Task categories covered:**
1. Secondary structure prediction
2. RNA function classification
3. RNA modification prediction
4. RNA stability prediction
5. mRNA expression prediction
6. Translation efficiency prediction
7. RNA-protein interaction prediction
8. Subcellular localization
9. Splice site prediction

## Model Comparison

| Model | Params | Architecture | Training Data | Key Strengths |
|-------|--------|-------------|---------------|---------------|
| **AIDO.RNA** | 1.6B | Encoder-only Transformer | 42M ncRNA (RNAcentral) | Best overall performance, largest scale |
| **RiNALMo** | 650M | Encoder-only Transformer | ncRNA + mRNA | Better generalization to unseen RNA families |
| **RNA-FM** | ~100M | BERT-style | ncRNA | Smaller, faster, good for basic tasks |
| **RNAErnie** | ~100M | BERT-style | ncRNA | Multi-task pretraining |
| **RNAGenesis** | ~600M | Encoder | ncRNA | Comparable on secondary structure |

## When to Use AIDO.RNA

**Use AIDO.RNA when:**
- You need the best overall RNA structure or function prediction accuracy
- Working with RNA secondary structure prediction
- Designing mRNA sequences (vaccines, therapeutics)
- Extracting general-purpose RNA embeddings for downstream ML tasks
- Predicting RNA modifications, stability, or expression levels
- Working within the broader AIDO ecosystem (DNA, Protein, Cell models)

**Consider alternatives when:**
- You specifically need generalization to novel/unseen RNA families -- consider **RiNALMo** (650M), which shows better performance on RNA families not seen during training
- You need a lightweight model for rapid prototyping -- consider **RNA-FM** or **RNAErnie**
- You are working exclusively with protein sequences -- use **ESM** instead
- You need DNA-level analysis -- use **AIDO.DNA** or **Nucleotide Transformer**

## Resources and Documentation

- **HuggingFace Model:** https://huggingface.co/genbio-ai/AIDO.RNA-1.6B
- **HuggingFace (CDS variant):** https://huggingface.co/genbio-ai/AIDO.RNA-1.6B-CDS
- **GitHub (AIDO):** https://github.com/genbio-ai/AIDO
- **GitHub (ModelGenerator):** https://github.com/genbio-ai/ModelGenerator
- **Scientific Paper:** https://doi.org/10.1101/2024.11.28.625345
- **RNAcentral:** https://rnacentral.org/

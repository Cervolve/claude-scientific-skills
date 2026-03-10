---
name: nucleotide-transformer
description: DNA foundation models (up to 2.5B parameters) for genomic sequence understanding. Use this skill for regulatory element classification (promoter prediction, splice site detection, enhancer prediction, histone modification prediction), DNA sequence embedding extraction, and fine-tuning classification heads on genomic tasks. Supports local/offline inference via HuggingFace Transformers. Best suited for sequences up to ~6kb with 6-mer tokenization.
license: cc-by-nc-sa-4.0
metadata:
    skill-author: K-Dense Inc.
---

# Nucleotide Transformer: DNA Foundation Models

## Overview

Nucleotide Transformer (NT) is a family of transformer-based DNA language models developed by InstaDeep, NVIDIA, and TUM. The flagship model contains 2.5 billion parameters and was trained on 174 billion nucleotides (~29 billion tokens) spanning 850 genomes from diverse species. These models use masked language modeling (BERT-style) on DNA sequences to learn representations useful for a wide range of genomic downstream tasks.

## Available Models on HuggingFace

| Model | Parameters | Training Data | HuggingFace ID |
|-------|-----------|---------------|----------------|
| Multi-species 2.5B | 2.5B | 850 genomes, 174B nucleotides | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` |
| 1000G 2.5B | 2.5B | 3,200+ human genomes | `InstaDeepAI/nucleotide-transformer-2.5b-1000g` |
| Multi-species 500M | 500M | 850 genomes | `InstaDeepAI/nucleotide-transformer-500m-multi-species` |
| 1000G 500M | 500M | 3,200+ human genomes | `InstaDeepAI/nucleotide-transformer-500m-1000g` |
| Human reference 500M | 500M | Human reference genome | `InstaDeepAI/nucleotide-transformer-500m-human-ref` |

All models use 6-mer tokenization (vocabulary size: 4,105 tokens) with a maximum context of 1,000 tokens (~6,000 nucleotides).

## Installation

```bash
pip install transformers torch
```

For optimal performance with large models:

```bash
pip install transformers torch accelerate
```

## Key Capabilities

- **Promoter prediction** -- Classify sequences as core/non-core promoters
- **Splice site detection** -- Identify donor and acceptor splice sites
- **Regulatory element classification** -- Predict enhancers, silencers, and other regulatory regions
- **Histone modification prediction** -- Predict histone marks from sequence
- **Enhancer prediction** -- Identify enhancer regions across species
- **DNA sequence embeddings** -- Extract learned representations for custom downstream tasks

## Benchmark Context (Nature Communications 2025)

**Nucleotide Transformer 2.5B multi-species achieves the highest overall performance** across the benchmark suite of 18 genomic tasks, with particular strength on:

- Promoter detection tasks (SOTA)
- Splice site identification (SOTA)

**Comparative landscape:**

- **DNABERT-2** is more consistent across diverse task types, making it a safer default for heterogeneous workloads
- **HyenaDNA** excels at long-range genomic dependencies beyond 6kb where NT's context window is a limitation
- **Caduceus-Ph** achieves the best overall classification accuracy when considering all task categories together
- For **variant effect prediction**, specialized models such as AlphaGenome and Enformer substantially outperform general-purpose DNA foundation models including NT

**Mean token embedding** (averaging over all token positions) consistently improves performance over CLS-token or other pooling strategies across all models tested.

## Comparison with AlphaGenome

| Aspect | Nucleotide Transformer | AlphaGenome |
|--------|----------------------|-------------|
| Access | Downloadable, fully offline | API-only (Google) |
| Variant effect prediction | Limited accuracy | State-of-the-art |
| Regulatory element classification | Strong (SOTA on promoters/splicing) | Strong |
| Sequence length | ~6kb (1,000 tokens) | Up to 1Mb |
| Fine-tuning | Full fine-tuning supported | Not available |
| Cost | Free (compute only) | API usage fees |

## When to Use

- **Regulatory element classification** when you need a local/offline model that you can fine-tune
- **Promoter and splice site prediction** where NT achieves state-of-the-art results
- **DNA embedding extraction** for custom downstream ML pipelines
- **Multi-species genomic analysis** leveraging the diverse training data
- When you need **full control** over the model (fine-tuning, custom heads, embedding access)

## When NOT to Use

- **Variant effect prediction** -- Use AlphaGenome (API) or Enformer instead; general DNA FMs perform poorly on VEP tasks
- **Long-range genomic dependencies >6kb** -- Use HyenaDNA, which supports much longer contexts
- **Tasks requiring consistent cross-category performance** -- Consider DNABERT-2 for more uniform accuracy across diverse task types
- **Plant or viral genomes** -- The multi-species training data excludes these; results may be unreliable

## Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
tokens = tokenizer.batch_encode_plus(
    sequences,
    return_tensors="pt",
    padding="max_length",
    max_length=tokenizer.model_max_length,
)["input_ids"]

attention_mask = tokens != tokenizer.pad_token_id
with torch.no_grad():
    outputs = model(tokens, attention_mask=attention_mask, output_hidden_states=True)

# Mean pooling over token embeddings (recommended)
hidden_states = outputs.hidden_states[-1]
mask = attention_mask.unsqueeze(-1).float()
mean_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
```

## Scripts

- `scripts/sequence_classification.py` -- Fine-tune a classification head on DNA sequences for tasks like promoter detection
- `scripts/extract_embeddings.py` -- Extract and save DNA sequence embeddings with optional UMAP visualization

## References

- Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics" (bioRxiv 2023)
- Nature Communications 2025 benchmark of DNA foundation models
- HuggingFace: https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
- Downstream task datasets: https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks

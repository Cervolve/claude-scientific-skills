---
name: alphagenome
description: Google DeepMind's AlphaGenome model for deciphering the regulatory code within DNA sequences. Use this skill for variant effect prediction, gene expression prediction, chromatin accessibility, transcription factor binding, splice site prediction, and chromatin contact maps. Supports human and mouse genomes via API. SOTA on 25/26 variant effect prediction tasks (Nature 2026).
license: Non-commercial use only (DeepMind Terms of Service)
metadata:
    skill-author: K-Dense Inc.
---

# AlphaGenome: DNA Regulatory Code Prediction

## Overview

AlphaGenome is Google DeepMind's foundation model for predicting regulatory genomic outputs from DNA sequence. It processes up to 1 million base pairs of input sequence and produces multimodal predictions at single base-pair resolution, including gene expression (RNA-seq), chromatin accessibility (ATAC-seq/DNase-seq), transcription factor binding (ChIP-seq), histone modifications, splice site usage, and chromatin contact maps (Hi-C/Micro-C).

AlphaGenome achieves state-of-the-art performance on 25 out of 26 variant effect prediction benchmark tasks (Nature 2026), significantly outperforming Enformer and other prior models.

## Core Capabilities

### 1. Variant Effect Prediction

Predict the functional impact of genetic variants by comparing reference and alternate allele predictions across multiple regulatory tracks.

**When to use:**
- Interpreting GWAS hits and candidate causal variants
- Prioritizing variants in regulatory regions
- Understanding non-coding variant mechanisms
- Scoring variants for pathogenicity in regulatory elements

**Basic usage:**

```python
from alphagenome.data import genome
from alphagenome.models import dna_client

API_KEY = 'MyAPIKey'
model = dna_client.create(API_KEY)

interval = genome.Interval(chromosome='chr22', start=35677410, end=36725986)
variant = genome.Variant(
    chromosome='chr22',
    position=36201698,
    reference_bases='A',
    alternate_bases='C',
)

outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=['UBERON:0001157'],
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
)

# Access reference and alternate predictions
ref_expression = outputs.reference.rna_seq
alt_expression = outputs.alternate.rna_seq
```

See `scripts/variant_effect_prediction.py` for a complete CLI workflow that processes VCF files and computes variant effect scores.

### 2. Gene Expression Prediction

Predict gene expression levels and regulatory track profiles for arbitrary genomic regions.

**When to use:**
- Predicting tissue-specific gene expression patterns
- Understanding regulatory landscapes around genes
- Comparing predicted vs observed expression
- Exploring cis-regulatory architecture

**Basic usage:**

```python
from alphagenome.data import genome
from alphagenome.models import dna_client

model = dna_client.create(API_KEY)

interval = genome.Interval(chromosome='chr7', start=5500000, end=6548576)

outputs = model.predict(
    interval=interval,
    ontology_terms=['UBERON:0002048'],  # lung
    requested_outputs=[
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.ATAC_SEQ,
        dna_client.OutputType.CAGE,
    ],
)

expression = outputs.rna_seq
accessibility = outputs.atac_seq
cage = outputs.cage
```

See `scripts/gene_expression_prediction.py` for a complete CLI workflow with visualization.

### 3. Chromatin Accessibility and TF Binding

Predict ATAC-seq, DNase-seq, and ChIP-seq profiles to understand chromatin state and transcription factor occupancy.

```python
outputs = model.predict(
    interval=interval,
    ontology_terms=['CL:0000746'],  # cardiac muscle cell
    requested_outputs=[
        dna_client.OutputType.ATAC_SEQ,
        dna_client.OutputType.DNASE_SEQ,
        dna_client.OutputType.CHIP_SEQ,
    ],
)
```

### 4. Splice Site Prediction

Predict splice site usage and splicing patterns from sequence.

```python
outputs = model.predict(
    interval=interval,
    requested_outputs=[dna_client.OutputType.SPLICE_SITE],
)
splice_sites = outputs.splice_site
```

### 5. Chromatin Contact Maps

Predict 3D chromatin organization (Hi-C / Micro-C contact frequencies).

```python
outputs = model.predict(
    interval=interval,
    ontology_terms=['CL:0000127'],  # astrocyte
    requested_outputs=[dna_client.OutputType.CONTACT_MAP],
)
contact_map = outputs.contact_map
```

### 6. Visualization

AlphaGenome includes built-in plotting utilities for overlaid track comparison.

```python
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt

plot_components.plot(
    [
        plot_components.OverlaidTracks(
            tdata={
                'REF': outputs.reference.rna_seq,
                'ALT': outputs.alternate.rna_seq,
            },
            colors={'REF': 'dimgrey', 'ALT': 'red'},
        ),
    ],
    interval=outputs.reference.rna_seq.interval.resize(2**15),
    annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
)
plt.savefig('variant_effect.png', dpi=150, bbox_inches='tight')
```

## Installation

**Clone and install from source:**

```bash
git clone https://github.com/google-deepmind/alphagenome.git
cd alphagenome
pip install ./alphagenome
```

**Dependencies:** The package installs its own dependencies. Requires Python 3.10+.

## API Authentication

AlphaGenome requires an API key for all predictions (the model runs on DeepMind servers).

1. Visit https://deepmind.google.com/science/alphagenome to request access
2. Generate an API key from the developer console
3. Set the key in your environment:

```bash
export ALPHAGENOME_API_KEY='your-api-key-here'
```

Or pass it directly when creating the client:

```python
from alphagenome.models import dna_client
model = dna_client.create('your-api-key-here')
```

## Key API Reference

**Client creation:**
- `dna_client.create(api_key)` -- Initialize the prediction client

**Prediction methods:**
- `model.predict(interval, ontology_terms, requested_outputs)` -- Predict tracks for a genomic region
- `model.predict_variant(interval, variant, ontology_terms, requested_outputs)` -- Compare ref vs alt predictions

**Data classes:**
- `genome.Interval(chromosome, start, end)` -- Genomic interval (0-based coordinates)
- `genome.Variant(chromosome, position, reference_bases, alternate_bases)` -- Genetic variant (1-based position)

**Output types (`dna_client.OutputType`):**
- `RNA_SEQ` -- Gene expression
- `CAGE` -- Transcription start site activity
- `ATAC_SEQ` -- Chromatin accessibility (ATAC)
- `DNASE_SEQ` -- Chromatin accessibility (DNase)
- `CHIP_SEQ` -- Transcription factor / histone ChIP-seq
- `SPLICE_SITE` -- Splice donor/acceptor predictions
- `CONTACT_MAP` -- Hi-C / Micro-C chromatin contacts

**Ontology terms:** Use UBERON (tissue/organ) or CL (cell type) ontology identifiers to specify the biological context. Examples:
- `UBERON:0002048` -- lung
- `UBERON:0001157` -- colon
- `UBERON:0000955` -- brain
- `CL:0000746` -- cardiac muscle cell
- `CL:0000127` -- astrocyte

## When to Use AlphaGenome

**Use AlphaGenome when:**
- You need variant effect predictions for regulatory variants (SOTA on 25/26 tasks)
- Predicting gene expression from sequence in specific tissues/cell types
- Analyzing chromatin accessibility, TF binding, or 3D genome organization
- Working with human or mouse genomes
- You need single base-pair resolution predictions over large (up to 1 Mbp) regions
- Interpreting non-coding GWAS variants or eQTLs

**Do NOT use AlphaGenome when:**
- Working with non-human, non-mouse species (not supported)
- You need protein-level predictions (use ESM or AlphaFold instead)
- You need to run millions of predictions (API rate limits; consider Enformer locally)
- Commercial use is required (non-commercial license only)
- You need predictions for structural variants or large indels (designed for SNVs and small indels)
- You need personal/individual-level gene expression prediction (retains limitations per bioRxiv evaluation)

**AlphaGenome vs Enformer:**
- AlphaGenome significantly outperforms Enformer on variant effect prediction (SOTA on 25/26 tasks vs Enformer's baseline)
- AlphaGenome processes 1 Mbp input (vs Enformer's ~200 kbp), capturing longer-range regulation
- AlphaGenome improves over Enformer on GTEx eQTL enrichment (OR=3.0 vs Enformer baseline)
- Enformer is open-weight and can be run locally; AlphaGenome is API-only
- For high-throughput screening of millions of variants, Enformer may be more practical

**AlphaGenome vs Nucleotide Transformer:**
- AlphaGenome is specialized for regulatory genomics; Nucleotide Transformer is a general DNA foundation model
- AlphaGenome provides direct functional predictions (expression, accessibility, etc.); Nucleotide Transformer provides embeddings that require fine-tuning
- Use Nucleotide Transformer for non-human species or when you need DNA embeddings for custom tasks

## Known Limitations

- **API-only:** No local inference; all predictions require API calls to DeepMind servers
- **Non-commercial:** Free for academic/research use only; commercial use requires separate agreement
- **Species:** Human (hg38) and mouse (mm10) genomes only
- **Rate limits:** Query rates vary; best suited for analyses requiring hundreds to thousands of predictions, not millions
- **Personal expression:** While AlphaGenome improves over Enformer (OR=3.0 on GTEx), it retains key limitations in personal gene expression prediction (bioRxiv evaluation)
- **Variant types:** Optimized for SNVs and small indels; not designed for structural variants or large CNVs
- **Sequence length:** Maximum input interval of ~1 Mbp

## Benchmark Performance

- **Variant effect prediction:** SOTA on 25 out of 26 benchmark tasks (Nature 2026)
- **GTEx eQTL enrichment:** OR=3.0, significantly above Enformer baseline
- **Chromatin accessibility:** Strong performance on ENCODE ATAC-seq and DNase-seq benchmarks
- **Contact maps:** Improved Micro-C prediction over prior sequence-based methods

## Resources and Documentation

- **GitHub Repository:** https://github.com/google-deepmind/alphagenome
- **Official Documentation:** https://www.alphagenomedocs.com/
- **API Key Registration:** https://deepmind.google.com/science/alphagenome
- **Scientific Paper:** Nature (2026)
- **Support Email:** alphagenome@google.com
- **Community Forum:** https://www.alphagenomedocs.com/ (community section)

## Responsible Use

AlphaGenome predictions are computational estimates and should not be used as the sole basis for clinical decisions. Variant effect predictions should be validated experimentally (e.g., MPRA, CRISPR screens) before clinical interpretation. Follow applicable guidelines for genomic data handling and variant interpretation (ACMG/AMP).

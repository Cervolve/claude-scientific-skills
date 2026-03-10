# Biological & Biomedical AI Models Catalog

> A curated catalog of SOTA models for the AI Scientist, organized by domain.
> Each entry includes: what it does, where to get it, known limitations, benchmark references, and when to use (or avoid) it.
>
> Last updated: 2026-03-10

---

## 1. Protein Structure Prediction

> **Key benchmark papers:**
> - [FoldBench: Benchmarking all-atom biomolecular structure prediction](https://www.nature.com/articles/s41467-025-67127-3) — Nature Comms 2025, 1522 assemblies across 9 tasks
> - [Benchmarking AlphaFold3-like methods for protein-peptide complexes](https://www.biorxiv.org/content/10.1101/2025.03.09.642277v1) — bioRxiv 2025
> - [Why do some predicted structures fold poorly? Benchmarking AF, ESMFold, and Boltz](https://www.biorxiv.org/content/10.1101/2025.07.05.663230v1) — bioRxiv 2025
> - [Protein Engineering with AI: OpenFold3 vs Boltz 2 vs AlphaFold 3](https://blackthorn.ai/blog/protein-engineering-with-ai/) — Blackthorn 2025

### AlphaFold3
- **What:** Predicts 3D structures of proteins, nucleic acids, small molecules, and their complexes at atomic accuracy
- **Where:** [GitHub](https://github.com/google-deepmind/alphafold3) — restricted academic license, weights require access request
- **HuggingFace:** No
- **Benchmarks:** ~88% correct monomeric structures, ~77% dimeric; best-in-class for protein-molecule complexes (DNA, RNA, ligands); outperforms Boltz-1 and Chai-1 on antibody-antigen complexes; >60% success in self-docking but plateaus ~60% in cross-docking
- **Limitations:** Closed license for commercial use; large compute requirements; does not predict binding affinity; performance declines for complexes lacking structural similarity to training set
- **When to use:** Gold-standard structure prediction for complexes; when you need protein-ligand/protein-DNA co-folding
- **When NOT to use:** When you need open-source/commercial license; when speed matters more than accuracy (use ESMFold)

### Boltz-2
- **What:** Open-source biomolecular foundation model — co-folds protein-ligand pairs AND predicts binding affinity (~0.6 correlation with experiment)
- **Where:** [GitHub](https://github.com/jwohlwend/boltz) — MIT license
- **HuggingFace:** [boltz-community/boltz-1](https://huggingface.co/boltz-community/boltz-1) (Boltz-1 weights; Boltz-2 via GitHub)
- **Benchmarks:** Accuracy on par with FEP calculations; ~20 seconds per complex on single GPU vs 6-12 hours for FEP; matches or moderately improves over Boltz-1 across modalities; strongest improvements on RNA and DNA-protein complexes
- **Benchmark paper:** [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/) — 2025
- **Limitations:** Newer model, less extensively validated than AlphaFold3 in literature
- **When to use:** When you need BOTH structure AND binding affinity; open-source drug discovery pipelines
- **When NOT to use:** N/A — currently best open-source option

### Chai-1
- **What:** Protein structure predictor for proteins, nucleic acids, small molecules, glycans, and modified residues
- **Where:** [GitHub](https://github.com/chaidiscovery/chai-lab) | [HuggingFace](https://huggingface.co/chaidiscovery/chai-1)
- **HuggingFace:** Yes — `chaidiscovery/chai-1`
- **Benchmarks:** Competitive with AlphaFold3 on protein-peptide complexes; over 80% success on standard benchmarks
- **Limitations:** Publicly available weights but not fully open-source license
- **When to use:** Ensemble predictions alongside AlphaFold3/Boltz-2; peptide-protein complexes

### ESMFold
- **What:** Single-sequence protein structure prediction (no MSA needed) — much faster than AlphaFold
- **Where:** [GitHub](https://github.com/facebookresearch/esm) | [HuggingFace](https://huggingface.co/facebook/esmfold_v1)
- **HuggingFace:** Yes — `facebook/esmfold_v1`
- **Benchmarks:** 76% correct monomeric structures (vs AlphaFold's 88%); poor on dimers (41% vs 77%)
- **Limitations:** Significantly less accurate than AlphaFold, especially for multimers and complexes
- **When to use:** Rapid screening of many sequences; when MSA computation is too slow; single-chain proteins
- **When NOT to use:** When accuracy is critical; protein complexes; dimers

### OpenFold
- **What:** Open-source reimplementation of AlphaFold2 with training code
- **Where:** [GitHub](https://github.com/aqlaboratory/openfold)
- **HuggingFace:** No
- **When to use:** When you need to retrain/fine-tune AlphaFold-like architecture on custom data

---

## 2. Protein Language Models & Design

> **Key benchmark papers:**
> - [PFMBench: Protein Foundation Model Benchmark](https://arxiv.org/html/2506.14796v1) — 2025, 22 tasks across 5 categories
> - [ProteinBench: A Holistic Evaluation of Protein Foundation Models](https://proteinbench.github.io/) — 2024
> - [Medium-sized protein language models perform well at transfer learning](https://www.nature.com/articles/s41598-025-05674-x) — Scientific Reports 2025

### ESM-2 / ESM3
- **What:** ESM-2 is a protein language model (up to 15B params) for embeddings; ESM3 is a multimodal generative model for protein sequence, structure, and function (up to 98B params)
- **Where:** [GitHub](https://github.com/facebookresearch/esm) | [HuggingFace](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- **HuggingFace:** Yes — multiple sizes (8M to 15B parameters)
- **Benchmarks:** ESM-2 systematically outperforms ProtTrans, ProteinBERT, and MSA Transformer; ESM-2 650M and ESM C 600M offer best performance/size tradeoff; ESM3 demonstrated novel GFP design; ProTrek (multimodal) achieves 75% winning rate on representative tasks in PFMBench
- **When to use:** Protein embeddings, variant effect prediction, zero-shot fitness prediction, protein generation

### RFdiffusion
- **What:** Protein backbone generation via diffusion; creates novel protein structures from scratch or with constraints
- **Where:** [GitHub](https://github.com/RosettaCommons/RFdiffusion)
- **HuggingFace:** No
- **Benchmarks:** Designs experimentally validated in wet lab; high success rate for binder design
- **When to use:** De novo protein design, binder design, symmetric assemblies
- **Typical workflow:** RFdiffusion → ProteinMPNN → AlphaFold2 validation

### ProteinMPNN
- **What:** Inverse folding — designs amino acid sequences that fold into a target backbone structure
- **Where:** [GitHub](https://github.com/dauparas/ProteinMPNN) | [HuggingFace](https://huggingface.co/spaces/simonduerr/ProteinMPNN)
- **HuggingFace:** Yes (Spaces demo)
- **Benchmarks:** 52.4% native sequence recovery; widely experimentally validated
- **When to use:** Sequence design for a given backbone; used downstream of RFdiffusion/Chroma

### Chroma
- **What:** Generative model for proteins/complexes with programmable conditioning (symmetry, shape, semantics)
- **Where:** [GitHub](https://github.com/generatebio/chroma)
- **HuggingFace:** No
- **When to use:** When you need to condition protein generation on natural language descriptions or geometric constraints

### ProGen / ProGen2
- **What:** Autoregressive protein language model (1.2B params) that generates functional protein sequences conditioned on protein family tags
- **Where:** [GitHub](https://github.com/salesforce/progen) | [HuggingFace](https://huggingface.co/hugohrban/progen2-large)
- **HuggingFace:** Yes — `hugohrban/progen2-large` (community)
- **When to use:** Generating novel proteins within a specific family/function

---

## 3. Antibody & Immunology Models

> **Key benchmark papers:**
> - [Fast, accurate antibody structure prediction from deep learning (IgFold)](https://www.nature.com/articles/s41467-023-38063-x) — Nature Comms 2023
> - [ImmuneBuilder: Deep-Learning models for immune proteins](https://www.nature.com/articles/s42003-023-04927-7) — Comms Biology 2023
> - [Review of Antibody Structure Prediction Based on AI](https://link.springer.com/article/10.1007/s11831-025-10404-7) — Archives of Computational Methods 2025

### IgFold
- **What:** Fast antibody structure prediction from sequence alone using language model embeddings
- **Where:** [GitHub](https://github.com/Graylab/IgFold)
- **HuggingFace:** No
- **Benchmarks:** Matches AlphaFold2 accuracy at ~1000x faster inference (<25s); compared against RepertoireBuilder, DeepAb, ABlooper, NanoNet, AlphaFold
- **When to use:** High-throughput antibody structure screening

### ABodyBuilder3
- **What:** Improved antibody structure prediction leveraging language model embeddings
- **Where:** [GitHub](https://github.com/oxpig/ABodyBuilder3)
- **HuggingFace:** No
- **Benchmarks:** SOTA accuracy on CDR loop modeling
- **When to use:** When CDR loop accuracy is critical (e.g., paratope prediction)

### ImmuneBuilder (ABodyBuilder2 / NanoBodyBuilder2 / TCRBuilder2)
- **What:** Predicts structures of antibodies, nanobodies, and TCRs
- **Where:** [GitHub](https://github.com/oxpig/ImmuneBuilder)
- **HuggingFace:** No
- **Benchmarks:** CDR-H3 RMSD of 2.81Å (0.09Å better than AlphaFold-Multimer); nanobody CDR-H3 RMSD of 2.89Å (0.55Å better than AF2); >100x faster than AlphaFold
- **When to use:** Antibody, nanobody, or TCR structure prediction

### AbLang / AbLang2
- **What:** Antibody-specific language model trained on OAS antibody sequences
- **Where:** [GitHub](https://github.com/oxpig/AbLang) | [HuggingFace](https://huggingface.co/qyzhang/AbLang)
- **HuggingFace:** Yes
- **When to use:** Antibody sequence analysis, restoring missing residues, humanness scoring

---

## 4. Small Molecule & Drug Discovery

> **Key benchmark papers:**
> - [Assessing the potential of deep learning for protein-ligand docking](https://www.nature.com/articles/s42256-025-01160-1) — Nature Machine Intelligence 2025
> - [Deep-Learning Based Docking Methods: Fair Comparisons](https://arxiv.org/html/2412.02889v1) — 2024
> - [PoseX: AI Defeats Physics-based Methods on Cross-Docking](https://arxiv.org/html/2505.01700v2) — 2025
> - [Decoding the limits of deep learning in molecular docking](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc05395a) — Chemical Science 2025

### DrugCLIP ⚠️ NOT ON HUGGINGFACE
- **What:** Contrastive learning framework for virtual screening — learns joint protein-molecule representations; 10M× faster than physics-based docking
- **Where:** [GitHub](https://github.com/bowen-gao/DrugCLIP) — weights via Google Drive
- **Benchmarks:** Published at NeurIPS 2023; evaluated on DUD-E and PCBA datasets; published in Science (Tsinghua University)
- **Limitations:** "Currently a raw version" per authors; requires exact rdkit==2022.9.5; data via Google Drive
- **When to use:** Large-scale virtual screening where speed is critical
- **When NOT to use:** When you need precise binding pose information

### PocketXMol ⚠️ NOT ON HUGGINGFACE
- **What:** Atom-level generative foundation model for pocket-interacting molecules — SOTA on 11/13 tasks across small molecules AND peptides
- **Where:** [GitHub](https://github.com/pengxingang/PocketXMol) — weights via Zenodo
- **Benchmark paper:** [Unified modeling of 3D molecular generation via atomic interactions](https://www.biorxiv.org/content/10.1101/2024.10.17.618827v2) — Cell 2026; compared against 55 baselines across 13 tasks (PoseBusters, PepBDB, GEOM, CSD, MOAD, PROTAC-DB)
- **Limitations:** Complex conda environment (CUDA 11.7); not on HuggingFace
- **When to use:** Structure-based drug design, fragment linking/growing, PROTAC design, peptide design, molecular docking
- **When NOT to use:** If you only need screening (use DrugCLIP); if environment setup is a constraint

### DiffDock ⚠️ USE WITH CAUTION
- **What:** Diffusion-based molecular docking — predicts binding poses
- **Where:** [GitHub](https://github.com/gcorso/DiffDock) | [HuggingFace Spaces](https://huggingface.co/spaces/reginabarzilaygroup/DiffDock-Web)
- **Benchmarks:** 38% top-1 success (RMSD<2Å) on PDBBind; DiffDock-L reaches 43%
- **Limitations:**
  - ⚠️ **Known accuracy issues:** Surflex-Dock 68%, Glide 67%, Vina ~63% vs DiffDock 45% when binding site is known ([Fair Comparisons study](https://arxiv.org/html/2412.02889v1))
  - Rigid docking only — fails when conformational changes occur
  - No binding affinity prediction
  - Struggles with novel targets outside training set
  - Primarily for small drug-like molecules + 1-2 chain proteins
  - SurfDock's PB-valid scores (physicochemical validity) only 40-64% despite high RMSD success
- **When to use:** Blind docking (unknown binding site); as part of an ensemble with physics-based methods
- **When NOT to use:** Known binding sites (use Glide/Vina instead); large ligands; large protein complexes; when you need binding affinity

### Uni-Mol / SurfDock
- **What:** 3D molecular pretraining framework (Uni-Mol) and surface-aware docking (SurfDock)
- **Where:** [GitHub](https://github.com/deepmodeling/Uni-Mol) | [HuggingFace](https://huggingface.co/dptech/Uni-Mol)
- **HuggingFace:** Yes (Uni-Mol) — `dptech/Uni-Mol`
- **Benchmarks:** Uni-Mol + SurfDock achieve 94.1% docking success with relaxation on ASTEX; SurfDock: 91.76% (Astex), 77.34% (PoseBusters), 75.66% (DockGen) — but PB-valid scores only 40-64%
- **When to use:** Molecular property prediction, docking with relaxation

### ChemBERTa
- **What:** BERT model pretrained on 77M molecules from ZINC
- **Where:** [HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)
- **HuggingFace:** Yes — `seyonec/ChemBERTa-zinc-base-v1`
- **When to use:** Molecular property prediction, SMILES-based representations, transfer learning

### MoLFormer
- **What:** Large-scale molecular transformer pretrained on 1.1B molecules with linear attention
- **Where:** [HuggingFace](https://huggingface.co/ibm/MoLFormer-XL-both-10pct)
- **HuggingFace:** Yes — `ibm/MoLFormer-XL-both-10pct`
- **When to use:** Molecular property prediction at scale; when you need efficient attention for long SMILES

---

## 5. Drug Repurposing & Clinical Knowledge Graphs

> **Key benchmark papers:**
> - [A foundation model for clinician-centered drug repurposing (TxGNN)](https://www.nature.com/articles/s41591-024-03233-x) — Nature Medicine 2024
> - [Learning the natural history of human disease (Delphi-2M)](https://www.nature.com/articles/s41586-025-09529-3) — Nature 2025

### TxGNN
- **What:** Graph neural network for zero-shot drug repurposing across 17,080 diseases and 7,957 therapeutic candidates
- **Where:** [GitHub](https://github.com/mims-harvard/TxGNN)
- **HuggingFace:** No
- **Benchmarks:** +49.2% improvement for indication prediction; +35.1% for contraindication prediction vs 8 baselines (BioBERT, HGT, HAN, network medicine approaches); predictions align with real-world off-label prescriptions in large healthcare system
- **When to use:** Identifying drug repurposing candidates; contraindication prediction; when interpretable reasoning paths are needed (TxGNN Explainer provides multi-hop knowledge paths)
- **Limitations:** Predictions are hypotheses requiring clinical validation

### Delphi-2M ⚠️ NOT ON HUGGINGFACE
- **What:** GPT-2-based model trained on disease trajectories — predicts rates of 1,000+ diseases up to 20 years ahead
- **Where:** [GitHub](https://github.com/gerstung-lab/delphi) — requires UK Biobank data access for full training
- **Benchmark paper:** [Learning the natural history of human disease with generative transformers](https://www.nature.com/articles/s41586-025-09529-3) — Nature 2025; trained on 400K UK Biobank patients, validated on 1.9M Danish individuals with no parameter changes
- **Limitations:**
  - ⚠️ Not ready for clinical use (per authors)
  - Full model requires UK Biobank institutional data access
  - Only demo/synthetic data provided in repo
  - Based on ICD codes — quality depends on coding accuracy
- **When to use:** Research on disease progression modeling; synthetic patient trajectory generation; epidemiological simulation
- **When NOT to use:** Clinical decision-making; individual patient predictions

---

## 6. Genomics / DNA Foundation Models

> **Key benchmark papers:**
> - [Advancing regulatory variant effect prediction with AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) — Nature 2026
> - [AlphaGenome enhances personal gene expression prediction but retains key limitations](https://www.biorxiv.org/content/10.1101/2025.08.05.668750v1) — bioRxiv 2025
> - [Benchmarking DNA foundation models for genomic and genetic tasks](https://www.nature.com/articles/s41467-025-65823-8) — Nature Comms 2025, evaluates DNABERT-2, NT V2, HyenaDNA, Caduceus-Ph, GROVER across sequence classification, gene expression, variant effects, TAD recognition
> - [DNALONGBENCH: benchmark suite for long-range DNA prediction](https://www.nature.com/articles/s41467-025-65077-4) — Nature Comms 2025
> - [Frontiers: Gene-LLMs comprehensive survey](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2025.1634882/full) — Frontiers in Genetics 2025

### AlphaGenome ⭐ HIGHEST PRIORITY
- **What:** DeepMind's DNA foundation model — takes 1Mb DNA input and predicts thousands of functional genomic tracks at single-base-pair resolution: gene expression, transcription initiation, chromatin accessibility, histone modifications, TF binding, chromatin contact maps, splice site usage
- **Where:** [API](https://deepmind.google.com/science/alphagenome/) | [GitHub SDK](https://github.com/google-deepmind/alphagenome) | [Research code](https://github.com/google-deepmind/alphagenome_research)
- **HuggingFace:** No — API-only (free for non-commercial research)
- **License:** Apache 2.0 (code); model accessed via API
- **Benchmark paper:** [Advancing regulatory variant effect prediction with AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) — Nature 2026
- **Benchmarks:**
  - Matches or exceeds strongest models in **25 of 26** evaluations of variant effect prediction
  - AUC = 0.80 for sQTLs, AUC = 0.86 for ipaQTLs (best among all models including Enformer and Sei)
  - Improves expression direction prediction over Enformer (odds ratio 3.0 using GTEx data)
  - ~1 million API requests per day from ~3,000 scientists in 160 countries
- **Limitations:**
  - ⚠️ API-only — no downloadable weights, requires internet
  - Non-commercial use only (commercial in "early stage testing")
  - [Recent evaluation](https://www.biorxiv.org/content/10.1101/2025.08.05.668750v1) shows it "retains key limitations" in personal gene expression prediction
  - Trained on human + mouse only
- **When to use:** Regulatory variant effect prediction; gene expression prediction from sequence; chromatin state prediction; any DNA→function task
- **When NOT to use:** Offline/air-gapped environments; commercial use; non-human/non-mouse species; when you need downloadable weights

### Evo
- **What:** Long-context genomic foundation model (7B params) trained at single-nucleotide resolution on prokaryotic + phage genomes
- **Where:** [GitHub](https://github.com/evo-design/evo) | [HuggingFace](https://huggingface.co/togethercomputer/evo-1-131k-base)
- **HuggingFace:** Yes — `togethercomputer/evo-1-131k-base`
- **Benchmark paper:** [Sequence modeling and design from molecular to genome scale with Evo](https://www.science.org/doi/10.1126/science.ado9336) — Science 2024
- **Benchmarks:** Generates functional CRISPR systems and transposons; molecular to genome-scale generation
- **Limitations:** Trained primarily on prokaryotic genomes — less suited for eukaryotic/human tasks
- **When to use:** DNA sequence generation, gene-level and genome-level prediction tasks, prokaryotic genome design
- **When NOT to use:** Human regulatory genomics (use AlphaGenome instead)

### Nucleotide Transformer
- **What:** DNA language models (up to 2.5B params) for genomic sequence understanding
- **Where:** [HuggingFace](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)
- **HuggingFace:** Yes — `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` (+ other variants)
- **Benchmark paper:** [Nucleotide Transformer: building and evaluating robust foundation models for human genomics](https://www.nature.com/articles/s41592-024-02523-z) — Nature Methods 2024
- **Benchmarks:** Multispecies 2.5B achieves highest overall performance in [Nature Comms benchmark](https://www.nature.com/articles/s41467-025-65823-8); SOTA on promoter and splicing tasks
- **When to use:** Promoter prediction, splice site detection, regulatory element classification; when you need downloadable weights (vs AlphaGenome API)

### HyenaDNA
- **What:** Long-range genomic model using sub-quadratic Hyena architecture; handles up to 1M nucleotide context
- **Where:** [GitHub](https://github.com/HazyResearch/hyena-dna) | [HuggingFace](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf)
- **HuggingFace:** Yes — `LongSafari/hyenadna-large-1m-seqlen-hf`
- **Benchmarks:** HyenaDNA-450K significantly outperforms Enformer in [Nature Comms benchmark](https://www.nature.com/articles/s41467-025-65823-8); 160x faster training than Transformer; scales sub-quadratically
- **When to use:** Long-range genomic dependencies; when context length >10kb matters; when compute efficiency matters

### Enformer
- **What:** Predicts gene expression from DNA sequence using attention over 200kb context
- **Where:** [GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/enformer) | [HuggingFace](https://huggingface.co/EleutherAI/enformer-official-rough)
- **HuggingFace:** Yes (community ports)
- **Benchmarks:** Best-in-class for enhancer and chromatin accessibility prediction in [Nature Comms benchmark](https://www.nature.com/articles/s41467-025-65823-8); now superseded by AlphaGenome on variant effect prediction
- **When to use:** Gene expression prediction from sequence (if AlphaGenome API not available); enhancer prediction
- **When NOT to use:** If AlphaGenome API is available — AlphaGenome outperforms Enformer on nearly all tasks

### DNABERT-2
- **What:** Multi-species DNA language model with improved tokenization (BPE instead of k-mer)
- **Where:** [GitHub](https://github.com/MAGICS-LAB/DNABERT_2) | [HuggingFace](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- **HuggingFace:** Yes — `zhihan1996/DNABERT-2-117M`
- **Benchmarks:** Most consistent performance across human genome tasks in [Nature Comms benchmark](https://www.nature.com/articles/s41467-025-65823-8); superior on promoter identification alongside Caduceus-Ph
- **When to use:** DNA sequence classification, regulatory element prediction; when consistency across tasks matters

### Caduceus
- **What:** Bi-directional DNA language model using Mamba architecture (sub-quadratic)
- **Where:** [GitHub](https://github.com/kuleshov-group/caduceus) | [HuggingFace](https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16)
- **HuggingFace:** Yes
- **Benchmarks:** Superior overall performance across multiple human genome classification tasks in [Nature Comms benchmark](https://www.nature.com/articles/s41467-025-65823-8)
- **When to use:** When you need bi-directional context and sub-quadratic scaling

---

## 7. RNA Foundation Models

> **Key benchmark papers:**
> - [Comprehensive benchmarking of LLMs for RNA secondary structure prediction](https://academic.oup.com/bib/article/26/2/bbaf137/8109668) — Briefings in Bioinformatics 2025
> - [A Comparative Review of RNA Language Models](https://arxiv.org/pdf/2505.09087) — 2025
> - [mRNABench: curated benchmark for mRNA prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC12265608/) — 2025

### AIDO.RNA
- **What:** Largest RNA foundation model (1.6B params) trained on 42M ncRNA sequences from RNAcentral
- **Where:** [GitHub](https://github.com/genbio-ai/AIDO) | [HuggingFace](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B)
- **HuggingFace:** Yes — `genbio-ai/AIDO.RNA-1.6B`
- **Benchmarks:** SOTA on 24/26 RNA understanding tasks; F1 = 0.787 on bpRNA-TS0 (outperforms RNAErnie and RiNALMo by large margins); 26-dataset benchmark across 9 task categories
- **When to use:** RNA structure prediction, function prediction, sequence design, mRNA vaccine design

### RiNALMo
- **What:** RNA language model (650M params) trained on 36M ncRNA sequences
- **Where:** [GitHub](https://github.com/lbcb-sci/RiNALMo) | [HuggingFace](https://huggingface.co/anonymous8/RiNALMo)
- **HuggingFace:** Yes
- **Benchmark paper:** [RiNALMo: general-purpose RNA language models can generalize well on structure prediction](https://www.nature.com/articles/s41467-025-60872-5) — Nature Comms 2025
- **Benchmarks:** SOTA on structure prediction; generalizes to unseen RNA families
- **When to use:** RNA secondary structure prediction; when generalization to novel families matters

### RNA-FM
- **What:** BERT-style RNA foundation model (published with RhoFold for structure prediction)
- **Where:** [GitHub](https://github.com/ml4bio/RNA-FM)
- **HuggingFace:** No (GitHub only)
- **When to use:** RNA structure prediction, embedding extraction

### RNAGenesis
- **What:** Generalist foundation model for functional RNA therapeutics
- **Where:** [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.12.30.630826v2)
- **Benchmarks:** Comparable or better than RiNALMo and AIDO.RNA on secondary structure prediction
- **When to use:** RNA therapeutic design; when generative capabilities matter

---

## 8. Single-Cell & Spatial Omics

> **Key benchmark papers:**
> - [Zero-shot evaluation reveals limitations of single-cell foundation models](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x) — Genome Biology 2025 (Microsoft Research)
> - [Deep-learning-based gene perturbation prediction does not outperform linear baselines](https://www.nature.com/articles/s41592-025-02772-6) — Nature Methods 2025
> - [BioLLM: standardized framework for benchmarking single-cell FMs](https://www.sciencedirect.com/science/article/pii/S2666389925001746) — 2025
> - [Single-cell foundation models: bringing AI into cell biology](https://www.nature.com/articles/s12276-025-01547-5) — Exp & Mol Medicine 2025

### scGPT
- **What:** Foundation model for single-cell multi-omics, pretrained on 33M cells
- **Where:** [GitHub](https://github.com/bowang-lab/scGPT) | [HuggingFace](https://huggingface.co/tdc/scGPT)
- **HuggingFace:** Yes — `tdc/scGPT`
- **Benchmark paper:** [scGPT: toward building a foundation model for single-cell multi-omics](https://www.nature.com/articles/s41592-024-02201-0) — Nature Methods 2024
- **Benchmarks / Limitations:**
  - ⚠️ [Zero-shot evaluation](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x) shows: does not outperform averaged bin prediction; performs worse than HVG + Harmony/scVI for cell type clustering; often regresses to mean expression values
  - ⚠️ [Perturbation prediction](https://www.nature.com/articles/s41592-025-02772-6) does not outperform simple linear baselines
  - Larger pretraining datasets do not always increase performance
  - Better at predicting cell types with limited training data vs Geneformer
- **When to use:** Cell type annotation (with fine-tuning), gene perturbation response prediction (as starting point), multi-batch integration
- **When NOT to use:** Zero-shot without fine-tuning; when simpler methods (PCA + kNN, Harmony, scVI) suffice

### Geneformer
- **What:** Transformer pretrained on ~104M single-cell transcriptomes from Genecorpus
- **Where:** [HuggingFace](https://huggingface.co/ctheodoris/Geneformer)
- **HuggingFace:** Yes — `ctheodoris/Geneformer` (V2: 104M and 316M param variants)
- **Benchmarks / Limitations:**
  - ⚠️ [Zero-shot evaluation](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x): consistently ranks lowest on integration metrics; retains or amplifies batch-specific variation; gene ranking correlation modest (median Pearson R = 0.56)
  - ⚠️ Relative strength in blood datasets but underperforms in others
  - Higher accuracy but lower macro-F1 than scGPT on some datasets
  - Fine-tuning substantially improves performance
- **When to use:** Gene network analysis, chromatin dynamics, disease state classification — **always with fine-tuning**
- **When NOT to use:** Zero-shot tasks; batch integration

### Nicheformer
- **What:** First foundation model integrating single-cell AND spatial transcriptomics (110M cells from SpatialCorpus-110M)
- **Where:** [GitHub](https://github.com/theislab/nicheformer)
- **HuggingFace:** Likely (check theislab)
- **Benchmark paper:** [Nicheformer: a foundation model for single-cell and spatial omics](https://www.nature.com/articles/s41592-025-02814-z) — Nature Methods 2025
- **Benchmarks:** Excels in spatial composition prediction, spatial label prediction via fine-tuning; trained on 57M dissociated + 53M spatially resolved cells across 73 tissues
- **When to use:** Spatial composition prediction, cell niche identification, integrating dissociated + spatial data

### scFoundation / scPRINT / UCE
- **What:** Alternative single-cell foundation models with different architectures
- **Where:** Various GitHub repos
- **When to use:** When benchmarking against scGPT/Geneformer; ensemble approaches

---

## 9. Biomedical Vision-Language & Pathology

> **Key benchmark papers:**
> - [Pathology Foundation Models review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11799676/) — JMA Journal 2025
> - [Foundation Models in Medical Imaging review](https://arxiv.org/html/2506.09095v1) — 2025
> - [Benchmarking pathology FMs for MSI prediction in colorectal cancer](https://www.sciencedirect.com/science/article/pii/S0895611125001892) — 2025

### BiomedCLIP
- **What:** Vision-language foundation model pretrained on 15M biomedical figure-caption pairs from PMC
- **Where:** [HuggingFace](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **HuggingFace:** Yes — `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Benchmark paper:** [BiomedCLIP: a multimodal biomedical foundation model](https://ai.nejm.org/doi/full/10.1056/AIoa2400640) — NEJM AI 2024
- **Benchmarks:** Outperforms radiology-specific models (BioViL) even on radiology tasks; SOTA on multiple VLP benchmarks
- **When to use:** Biomedical image classification, image-text retrieval, visual question answering

### UNI / UNI2
- **What:** Pathology foundation model trained on 200M+ H&E and IHC images from 350K+ whole-slide images
- **Where:** [GitHub](https://github.com/mahmoodlab/UNI) | [HuggingFace](https://huggingface.co/MahmoodLab/UNI) (requires access request)
- **HuggingFace:** Yes — gated access
- **Benchmarks:** SOTA across disease detection, biomarker prediction, treatment outcome prediction; UNI2 released Jan 2025 with 25K+ pre-extracted TCGA/CPTAC/PANDA WSI embeddings
- **When to use:** Computational pathology, cancer subtyping, biomarker prediction from histology

### CONCH
- **What:** Vision-language pathology foundation model
- **Where:** [GitHub](https://github.com/mahmoodlab/CONCH) | [HuggingFace](https://huggingface.co/MahmoodLab/CONCH) (requires registration)
- **HuggingFace:** Yes — gated access
- **Benchmark paper:** [A visual-language foundation model for computational pathology](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384335/) — Nature Medicine 2024
- **When to use:** Pathology image-text tasks, zero-shot classification of histology images

### Virchow
- **What:** Foundation model for computational pathology (by Paige/Microsoft)
- **Where:** [HuggingFace](https://huggingface.co/paige-ai/Virchow) (gated)
- **HuggingFace:** Yes — gated access
- **Benchmark paper:** [A foundation model for clinical-grade computational pathology and rare cancers detection](https://www.nature.com/articles/s41591-024-03141-0) — Nature Medicine 2024
- **Benchmarks:** Best cancer detection AUC across all types; AUC 0.937 on rare cancers
- **When to use:** Cancer detection, especially rare cancers

---

## 10. Clinical / EHR Models

> **Key benchmark papers:**
> - [Extending BEHRT to UK Biobank: assessing transformer model performance](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1715506/full) — Frontiers 2026
> - [BEHRT: Transformer for Electronic Health Records](https://www.nature.com/articles/s41598-020-62922-y) — Scientific Reports 2020

### BEHRT
- **What:** BERT for Electronic Health Records — predicts 301 conditions from visit sequences
- **Where:** [GitHub](https://github.com/deepmedicine/BEHRT)
- **HuggingFace:** No
- **Benchmarks:** 8-13% improvement over prior deep EHR models; large models consistently outperform smaller ones for long-term prediction (up to 5 years)
- **Limitations:** Requires structured EHR data with ICD codes; institution-specific training typically needed
- **When to use:** Multi-disease prediction from longitudinal health records

### Med-BERT
- **What:** Contextualized embeddings pretrained on 28.5M patient EHR records
- **Where:** [GitHub](https://github.com/ZhiGroup/Med-BERT)
- **HuggingFace:** No
- **Benchmark paper:** [Med-BERT: pretrained contextualized embeddings on large-scale structured EHR](https://www.nature.com/articles/s41746-021-00455-y) — npj Digital Medicine 2021
- **When to use:** Disease prediction from structured EHR; transfer learning for clinical tasks

### ClinicalBERT / Bio_ClinicalBERT
- **What:** BERT models fine-tuned on clinical notes (MIMIC-III)
- **Where:** [HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **HuggingFace:** Yes — `emilyalsentzer/Bio_ClinicalBERT`
- **When to use:** Clinical NLP tasks, discharge summary analysis, medical NER

---

## 11. CRISPR / Gene Editing

> **Key benchmark papers:**
> - [Deep Learning Based Models for CRISPR/Cas Off-Target Prediction](https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/smtd.202500122) — Small Methods 2025, evaluates CRISPR-Net, CRISPR-IP, R-CRISPR, CRISPR-M, CrisprDNT, Crispr-SGRU across 6 datasets
> - [CCLMoff: versatile CRISPR/Cas9 off-target prediction using language model](https://www.nature.com/articles/s42003-025-08275-6) — Comms Biology 2025

### CRISPR-Net / CCLMoff
- **What:** Deep learning models for CRISPR off-target prediction
- **Where:** Various GitHub repos
- **HuggingFace:** No
- **Benchmarks:** CRISPR-Net, R-CRISPR, and Crispr-SGRU show strongest overall performance across 6 datasets; CCLMoff uses RNA language model embeddings for strong generalization
- **When to use:** Predicting off-target effects of guide RNAs; CRISPR experiment design

---

## 12. Protein-Protein Interaction

> **Key benchmark papers:**
> - [Recent advances in deep learning for protein-protein interaction: a review](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-025-00457-6) — BioData Mining 2025
> - [Advances in PPI prediction: a deep learning perspective](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1710937/full) — Frontiers in Bioinformatics 2025

### Struct2Graph
- **What:** Multi-layer mutual graph attention network for structure-based PPI prediction
- **Where:** GitHub
- **HuggingFace:** No
- **Benchmarks:** 98.89% accuracy (balanced) / 99.42% (unbalanced 1:10)
- **Limitations:** Performance may vary on unseen protein pairs; accuracy figures may reflect dataset-specific evaluation

### ScanNet
- **What:** Interpretable deep learning model for protein binding site prediction
- **Where:** [GitHub](https://github.com/jertubiana/ScanNet)
- **HuggingFace:** No
- **When to use:** When interpretability of binding predictions matters

---

## 13. Multi-Modal / Cross-Domain

### LucaOne
- **What:** Unified foundation model pretrained on nucleic acid AND protein sequences from 169,861 species; captures central dogma relationships
- **Where:** GitHub
- **Benchmark paper:** [Generalized biological foundation model with unified nucleic acid and protein language](https://www.nature.com/articles/s42256-025-01044-4) — Nature Machine Intelligence 2025
- **When to use:** Cross-modal transfer between DNA/RNA/protein; when you need a single model spanning the central dogma

---

## Summary: HuggingFace Availability

### ✅ On HuggingFace (ready to use)
| Model | HuggingFace ID | Domain |
|-------|----------------|--------|
| ESM-2 | `facebook/esm2_t33_650M_UR50D` | Protein embeddings |
| ESMFold | `facebook/esmfold_v1` | Protein structure |
| Geneformer V2 | `ctheodoris/Geneformer` | Single-cell |
| scGPT | `tdc/scGPT` | Single-cell |
| Nucleotide Transformer | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` | DNA |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` | DNA |
| HyenaDNA | `LongSafari/hyenadna-large-1m-seqlen-hf` | DNA |
| Caduceus | `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` | DNA |
| Evo | `togethercomputer/evo-1-131k-base` | DNA generation |
| AIDO.RNA | `genbio-ai/AIDO.RNA-1.6B` | RNA |
| RiNALMo | `anonymous8/RiNALMo` | RNA |
| Chai-1 | `chaidiscovery/chai-1` | Structure prediction |
| Boltz-1 | `boltz-community/boltz-1` | Structure prediction |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Biomedical vision-language |
| UNI/CONCH/Virchow | MahmoodLab / paige-ai (gated) | Pathology |
| ChemBERTa | `seyonec/ChemBERTa-zinc-base-v1` | Molecular |
| MoLFormer | `ibm/MoLFormer-XL-both-10pct` | Molecular |
| Uni-Mol | `dptech/Uni-Mol` | Molecular 3D |
| ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | Clinical NLP |
| AbLang | `qyzhang/AbLang` | Antibody |
| ProGen2 | `hugohrban/progen2-large` | Protein generation |

### ⚠️ NOT on HuggingFace (GitHub/API only)
| Model | Source | Domain | Why it matters |
|-------|--------|--------|----------------|
| **AlphaGenome** | [API](https://deepmind.google.com/science/alphagenome/) | DNA/regulatory | SOTA on 25/26 variant effect tasks; Nature 2026 |
| **Delphi-2M** | [GitHub](https://github.com/gerstung-lab/delphi) | Disease prediction | Only model predicting 1000+ diseases 20 years ahead |
| **DrugCLIP** | [GitHub](https://github.com/bowen-gao/DrugCLIP) | Virtual screening | 10M× faster than docking |
| **PocketXMol** | [GitHub](https://github.com/pengxingang/PocketXMol) | Molecular generation | SOTA on 11/13 tasks; Cell 2026 |
| **RFdiffusion** | [GitHub](https://github.com/RosettaCommons/RFdiffusion) | Protein design | De facto standard for backbone generation |
| **AlphaFold3** | [GitHub](https://github.com/google-deepmind/alphafold3) | Structure prediction | Gold standard |
| **TxGNN** | [GitHub](https://github.com/mims-harvard/TxGNN) | Drug repurposing | Zero-shot across 17K diseases |
| **IgFold** | [GitHub](https://github.com/Graylab/IgFold) | Antibody structure | 1000x faster than AF2 for antibodies |
| **BEHRT** | [GitHub](https://github.com/deepmedicine/BEHRT) | EHR prediction | Predicts 301 conditions |
| **Chroma** | [GitHub](https://github.com/generatebio/chroma) | Protein design | Programmable protein generation |
| **Boltz-2** | [GitHub](https://github.com/jwohlwend/boltz) | Structure + affinity | Best open-source option (MIT) |

---

## Recommended Priority for Script Development

Based on impact, SOTA status, and current skill coverage gaps:

### High Priority (no existing scripts, high impact)
1. **AlphaGenome** — SOTA DNA/regulatory model; API-based; Nature 2026 (⭐ highest priority)
2. **ESM-2/ESM3** — protein embeddings + generation (HF available)
3. **Boltz-2** — structure prediction + binding affinity (open source, MIT)
4. **Geneformer** — single-cell analysis with fine-tuning (HF available)
5. **TxGNN** — drug repurposing (GitHub)

### Medium Priority (valuable but more niche or complex setup)
6. **PocketXMol** — molecular generation (Cell 2026, GitHub)
7. **DrugCLIP** — virtual screening (GitHub)
8. **RFdiffusion + ProteinMPNN** — protein design pipeline (GitHub)
9. **BiomedCLIP** — biomedical image analysis (HF available)
10. **AIDO.RNA** — RNA analysis (HF available)
11. **Nucleotide Transformer** — DNA analysis, downloadable alternative to AlphaGenome (HF available)

### Lower Priority (existing scripts or less mature)
12. **DiffDock** — already has scripts; known accuracy issues (see benchmark warnings above)
13. **Delphi-2M** — requires institutional data access
14. **BEHRT/Med-BERT** — requires EHR data access
15. **CRISPR models** — fragmented landscape

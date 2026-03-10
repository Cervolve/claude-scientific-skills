#!/usr/bin/env python3
"""
Extract Geneformer Cell Embeddings

This script takes an AnnData (.h5ad) file, tokenizes gene expression using
Geneformer's rank-value encoding, runs the data through the pretrained (or
fine-tuned) model, extracts cell embeddings, saves them to the AnnData obsm
slot, and optionally produces a UMAP visualization.

Requires:
    pip install geneformer scanpy

IMPORTANT: Geneformer embeddings can amplify batch effects (Genome Biology
2025, Microsoft Research). Apply batch correction (e.g., Harmony) on
embeddings before downstream analysis. For batch integration tasks, prefer
scVI or Harmony directly over Geneformer embeddings.

Usage:
    python extract_embeddings.py \
        --input data.h5ad \
        --model-size 104M \
        --output ./embeddings/ \
        --umap
"""

# ============================================================================
# IMPORTANT: Geneformer embeddings can amplify batch effects (Genome Biology
# 2025). Apply batch correction (e.g., Harmony) on embeddings before
# downstream analysis.
# ============================================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import scanpy as sc
except ImportError:
    raise ImportError(
        "scanpy is required. Install with:\n  pip install scanpy"
    )

try:
    from geneformer import TranscriptomeTokenizer, EmbExtractor
except ImportError:
    raise ImportError(
        "geneformer is not installed. Install with:\n"
        "  git lfs install\n"
        "  git clone https://huggingface.co/ctheodoris/Geneformer\n"
        "  cd Geneformer && pip install ."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

_MODEL_SIZE_MAP = {
    "10M": "geneformer-v1-10M",
    "104M": "geneformer-v2-104M",
    "316M": "geneformer-v2-316M",
    "104M_cancer": "geneformer-v2-104M_CLcancer",
}


def resolve_model_directory(model_size: str) -> str:
    """Resolve a model size shorthand to a HuggingFace model directory."""
    if os.path.isdir(model_size):
        return model_size
    variant = _MODEL_SIZE_MAP.get(model_size, model_size)
    return f"ctheodoris/Geneformer/{variant}"


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_anndata(
    adata: sc.AnnData,
    work_dir: str,
    label_columns: Optional[list[str]] = None,
    model_input_size: int = 4096,
) -> str:
    """Convert AnnData to Geneformer tokenized dataset via .loom intermediate.

    Args:
        adata: AnnData object with gene expression counts.
        work_dir: Working directory for intermediate files.
        label_columns: Optional list of obs columns to carry through tokenization.
        model_input_size: Context window (2048 for V1, 4096 for V2).

    Returns:
        Path to the tokenized dataset directory.
    """
    loom_dir = os.path.join(work_dir, "loom_input")
    os.makedirs(loom_dir, exist_ok=True)
    loom_path = os.path.join(loom_dir, "data.loom")

    # Write loom
    adata.write_loom(loom_path, write_obsm_varm=False)
    logger.info(f"Wrote loom file: {loom_path} ({adata.n_obs} cells, {adata.n_vars} genes)")

    # Build custom attribute dict for label columns
    custom_attrs = {}
    if label_columns:
        for col in label_columns:
            if col in adata.obs.columns:
                custom_attrs[col] = col
            else:
                logger.warning(f"Column '{col}' not found in adata.obs, skipping")

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attrs if custom_attrs else None,
        nproc=4,
        model_input_size=model_input_size,
    )

    tokenized_dir = os.path.join(work_dir, "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)

    tk.tokenize_data(
        data_directory=loom_dir,
        output_directory=tokenized_dir,
        output_prefix="geneformer_tokenized",
        file_format="loom",
    )

    tokenized_path = os.path.join(tokenized_dir, "geneformer_tokenized.dataset")
    if not os.path.exists(tokenized_path):
        candidates = list(Path(tokenized_dir).glob("geneformer_tokenized*"))
        if candidates:
            tokenized_path = str(candidates[0])
        else:
            raise FileNotFoundError(
                f"Tokenization produced no output in {tokenized_dir}"
            )

    logger.info(f"Tokenized dataset: {tokenized_path}")
    return tokenized_path


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model_directory: str,
    tokenized_path: str,
    output_dir: str,
    emb_layer: int = -1,
    label_columns: Optional[list[str]] = None,
    max_cells: Optional[int] = None,
    batch_size: int = 100,
) -> np.ndarray:
    """Extract cell embeddings using Geneformer's EmbExtractor.

    Args:
        model_directory: Path to Geneformer model (pretrained or fine-tuned).
        tokenized_path: Path to tokenized dataset.
        output_dir: Directory for embedding output files.
        emb_layer: Which transformer layer to extract (-1 = last layer).
        label_columns: Optional obs columns for labeling embeddings.
        max_cells: Maximum number of cells to process (None = all).
        batch_size: Forward pass batch size.

    Returns:
        Numpy array of cell embeddings (n_cells x embedding_dim).
    """
    os.makedirs(output_dir, exist_ok=True)

    emb_label = label_columns if label_columns else []
    labels_to_plot = label_columns if label_columns else []

    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        filter_data=None,
        max_ncells=max_cells,
        emb_layer=emb_layer,
        emb_label=emb_label if emb_label else None,
        labels_to_plot=labels_to_plot if labels_to_plot else None,
        forward_batch_size=batch_size,
        nproc=4,
    )

    logger.info(f"Extracting embeddings from layer {emb_layer}...")
    embs = embex.extract_embs(
        model_directory=model_directory,
        input_data_file=tokenized_path,
        output_directory=output_dir,
        output_prefix="geneformer_embs",
    )

    logger.info(f"Extracted embeddings shape: {embs.shape}")
    return embs


def save_embeddings_to_anndata(
    adata: sc.AnnData,
    embeddings: np.ndarray,
    output_path: str,
    obsm_key: str = "X_geneformer",
) -> sc.AnnData:
    """Save embeddings to AnnData obsm slot.

    Args:
        adata: Original AnnData object.
        embeddings: Embedding matrix (n_cells x embedding_dim).
        output_path: Path to save the updated AnnData file.
        obsm_key: Key in adata.obsm for storing embeddings.

    Returns:
        Updated AnnData object.
    """
    if embeddings.shape[0] != adata.n_obs:
        logger.warning(
            f"Embedding count ({embeddings.shape[0]}) differs from "
            f"AnnData cell count ({adata.n_obs}). Truncating to minimum."
        )
        n = min(embeddings.shape[0], adata.n_obs)
        adata = adata[:n].copy()
        embeddings = embeddings[:n]

    adata.obsm[obsm_key] = embeddings
    logger.info(
        f"Saved embeddings to adata.obsm['{obsm_key}'] "
        f"({embeddings.shape[0]} cells x {embeddings.shape[1]} dims)"
    )

    adata.write_h5ad(output_path)
    logger.info(f"Saved AnnData with embeddings to: {output_path}")
    return adata


# ---------------------------------------------------------------------------
# UMAP visualization
# ---------------------------------------------------------------------------

def compute_and_plot_umap(
    adata: sc.AnnData,
    obsm_key: str = "X_geneformer",
    color_by: Optional[list[str]] = None,
    output_dir: str = ".",
    n_neighbors: int = 15,
    min_dist: float = 0.5,
) -> sc.AnnData:
    """Compute UMAP from Geneformer embeddings and generate plots.

    NOTE: If your data contains multiple batches, apply batch correction
    (e.g., Harmony) on the embeddings BEFORE computing UMAP. Geneformer
    embeddings can amplify batch effects.

    Args:
        adata: AnnData with embeddings in obsm[obsm_key].
        obsm_key: Key in obsm containing Geneformer embeddings.
        color_by: List of obs columns to color UMAP by.
        output_dir: Directory for saving plots.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.

    Returns:
        AnnData with UMAP coordinates added.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Computing neighbors from {obsm_key}...")
    sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors)

    logger.info("Computing UMAP...")
    sc.tl.umap(adata, min_dist=min_dist)

    # Set figure output
    sc.settings.figdir = output_dir
    sc.settings.set_figure_params(dpi=150, frameon=False, figsize=(6, 6))

    if color_by:
        for col in color_by:
            if col in adata.obs.columns:
                logger.info(f"Plotting UMAP colored by '{col}'...")
                sc.pl.umap(
                    adata,
                    color=col,
                    legend_loc="on data" if adata.obs[col].nunique() < 30 else "right margin",
                    legend_fontoutline=2,
                    frameon=False,
                    save=f"_geneformer_{col}.png",
                    show=False,
                )
            else:
                logger.warning(f"Column '{col}' not in adata.obs, skipping plot")
    else:
        # Plot without color
        sc.pl.umap(
            adata,
            frameon=False,
            save="_geneformer.png",
            show=False,
        )

    logger.info(f"UMAP plots saved to: {output_dir}")
    return adata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Geneformer cell embeddings from single-cell data.\n\n"
            "IMPORTANT: Geneformer embeddings can amplify batch effects "
            "(Genome Biology 2025). Apply batch correction (e.g., Harmony) "
            "on embeddings before downstream analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to AnnData .h5ad file.",
    )
    parser.add_argument(
        "--model-size", default="104M",
        choices=list(_MODEL_SIZE_MAP.keys()),
        help="Geneformer model variant (default: 104M).",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Direct path to a local Geneformer model directory (overrides --model-size).",
    )
    parser.add_argument(
        "--emb-layer", type=int, default=-1,
        help="Transformer layer to extract embeddings from (-1 = last, default: -1).",
    )
    parser.add_argument(
        "--label-columns", nargs="*", default=None,
        help="Obs columns to carry through and use for labeling (e.g., cell_type batch).",
    )
    parser.add_argument(
        "--max-cells", type=int, default=None,
        help="Maximum number of cells to process (default: all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Forward pass batch size (default: 100).",
    )
    parser.add_argument(
        "--umap", action="store_true",
        help="Compute and plot UMAP from embeddings.",
    )
    parser.add_argument(
        "--umap-color", nargs="*", default=None,
        help="Obs columns to color UMAP by (e.g., cell_type batch). Implies --umap.",
    )
    parser.add_argument(
        "--obsm-key", default="X_geneformer",
        help="Key in adata.obsm for storing embeddings (default: X_geneformer).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for embeddings and plots.",
    )

    args = parser.parse_args()

    # Resolve model
    if args.model_path:
        model_dir = args.model_path
    else:
        model_dir = resolve_model_directory(args.model_size)

    model_input_size = 2048 if args.model_size == "10M" else 4096

    logger.info(f"Model: {model_dir}")
    logger.info(f"Input: {args.input}")

    # Load AnnData
    logger.info(f"Loading AnnData from {args.input}...")
    adata = sc.read_h5ad(args.input)
    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Working directory
    work_dir = os.path.join(args.output, "_work")
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Tokenize
    logger.info("Step 1/3: Tokenizing with rank-value encoding...")
    tokenized_path = tokenize_anndata(
        adata,
        work_dir,
        label_columns=args.label_columns,
        model_input_size=model_input_size,
    )

    # Step 2: Extract embeddings
    logger.info("Step 2/3: Extracting embeddings...")
    embeddings = extract_embeddings(
        model_directory=model_dir,
        tokenized_path=tokenized_path,
        output_dir=args.output,
        emb_layer=args.emb_layer,
        label_columns=args.label_columns,
        max_cells=args.max_cells,
        batch_size=args.batch_size,
    )

    # Step 3: Save to AnnData
    logger.info("Step 3/3: Saving embeddings to AnnData...")
    output_h5ad = os.path.join(args.output, "adata_with_embeddings.h5ad")
    adata = save_embeddings_to_anndata(
        adata, embeddings, output_h5ad, obsm_key=args.obsm_key
    )

    # Optional: UMAP
    if args.umap or args.umap_color:
        logger.info("Computing UMAP from Geneformer embeddings...")
        color_by = args.umap_color or args.label_columns
        adata = compute_and_plot_umap(
            adata,
            obsm_key=args.obsm_key,
            color_by=color_by,
            output_dir=args.output,
        )
        # Re-save with UMAP coordinates
        adata.write_h5ad(output_h5ad)

    logger.info("Done. Output saved to: %s", args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Cells:          {adata.n_obs}")
    print(f"  Embedding dim:  {embeddings.shape[1]}")
    print(f"  obsm key:       {args.obsm_key}")
    print(f"  Output file:    {output_h5ad}")
    if args.umap or args.umap_color:
        print(f"  UMAP:           computed and plotted")
    print("=" * 60)
    print(
        "\nREMINDER: Geneformer embeddings can amplify batch effects.\n"
        "Apply Harmony or similar correction before downstream analysis."
    )


if __name__ == "__main__":
    main()

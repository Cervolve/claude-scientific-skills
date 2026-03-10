#!/usr/bin/env python3
"""
Extract DNA Sequence Embeddings with Nucleotide Transformer

Loads a Nucleotide Transformer model from HuggingFace, tokenizes input DNA
sequences (FASTA format), extracts embeddings from the last hidden layer using
mean pooling, and saves them as NumPy (.npy) or HDF5 (.h5) files. Optionally
generates a UMAP visualization.

Nucleotide Transformer 2.5B multi-species achieves highest overall performance
across 18 genomic tasks. Use 6-mer tokenization (handled automatically by the
tokenizer). For sequences >6kb, consider HyenaDNA instead.

Usage:
    # Save embeddings as NumPy:
    python extract_embeddings.py \
        --input sequences.fasta \
        --model InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
        --output embeddings.npy

    # Save as HDF5 with UMAP plot:
    python extract_embeddings.py \
        --input sequences.fasta \
        --output embeddings.h5 \
        --format hdf5 \
        --umap umap_plot.png

    # Use labels for colored UMAP:
    python extract_embeddings.py \
        --input sequences.fasta \
        --output embeddings.npy \
        --umap umap_plot.png \
        --labels labels.csv
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_fasta(filepath: str) -> list[tuple[str, str]]:
    """Parse a FASTA file and return list of (header, sequence) tuples.
    Also accepts plain-text files with one sequence per line."""
    records: list[tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: list[str] = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None or seq_lines:
                    records.append((header or f"seq_{len(records)}", "".join(seq_lines)))
                    seq_lines = []
                header = line[1:].strip()
            else:
                seq_lines.append(line.upper())
        if header is not None or seq_lines:
            records.append((header or f"seq_{len(records)}", "".join(seq_lines)))

    return records


def load_labels(filepath: str) -> dict[str, str]:
    """Load labels from CSV with columns: sequence_id, label."""
    labels: dict[str, str] = {}
    with open(filepath, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels[row["sequence_id"]] = row["label"]
    return labels


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    sequences: list[tuple[str, str]],
    model_name: str,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract mean-pooled embeddings from the last hidden state.

    Mean token embedding consistently improves performance over other pooling
    strategies (Nature Comms 2025 benchmark).

    Args:
        sequences: List of (header, sequence) tuples.
        model_name: HuggingFace model ID or local path.
        batch_size: Number of sequences per forward pass.
        device: Torch device. Auto-detected if None.

    Returns:
        Tuple of (embeddings array [N, hidden_dim], list of headers).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    max_length = tokenizer.model_max_length
    headers: list[str] = []
    all_embeddings: list[np.ndarray] = []

    num_batches = (len(sequences) + batch_size - 1) // batch_size
    logger.info(f"Extracting embeddings for {len(sequences)} sequences in {num_batches} batches")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(sequences))
        batch_headers = [s[0] for s in sequences[start:end]]
        batch_seqs = [s[1] for s in sequences[start:end]]

        encoding = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Mean pooling over non-padding tokens from last hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        mean_emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_embeddings.append(mean_emb.cpu().numpy())
        headers.extend(batch_headers)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            logger.info(f"  Batch {batch_idx + 1}/{num_batches} complete")

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings, headers


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_numpy(embeddings: np.ndarray, headers: list[str], output_path: str):
    """Save embeddings as .npy and headers as companion .txt."""
    np.save(output_path, embeddings)
    headers_path = output_path.replace(".npy", "_headers.txt")
    with open(headers_path, "w") as fh:
        fh.write("\n".join(headers))
    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"Saved headers to {headers_path}")


def save_hdf5(embeddings: np.ndarray, headers: list[str], output_path: str):
    """Save embeddings as HDF5 with sequence IDs."""
    try:
        import h5py
    except ImportError:
        logger.error("h5py is required for HDF5 output. Install with: pip install h5py")
        sys.exit(1)

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings)
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset("headers", data=np.array(headers, dtype=object), dtype=dt)
    logger.info(f"Saved embeddings to {output_path}")


# ---------------------------------------------------------------------------
# UMAP visualization
# ---------------------------------------------------------------------------

def generate_umap(
    embeddings: np.ndarray,
    headers: list[str],
    output_path: str,
    labels: Optional[dict[str, str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """Generate a UMAP visualization of the embeddings."""
    try:
        import umap
    except ImportError:
        logger.error("umap-learn is required for UMAP visualization. Install with: pip install umap-learn")
        sys.exit(1)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for UMAP visualization. Install with: pip install matplotlib")
        sys.exit(1)

    logger.info(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    projection = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels:
        # Color by label
        label_list = [labels.get(h, "unknown") for h in headers]
        unique_labels = sorted(set(label_list))
        color_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        colors = [color_map[lbl] for lbl in label_list]

        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=colors,
            cmap="tab10",
            s=10,
            alpha=0.7,
        )
        # Legend
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab10(color_map[lbl] / max(len(unique_labels) - 1, 1)),
                markersize=8,
                label=lbl,
            )
            for lbl in unique_labels
        ]
        ax.legend(handles=handles, title="Label", loc="best")
    else:
        ax.scatter(projection[:, 0], projection[:, 1], s=10, alpha=0.7)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Nucleotide Transformer Embeddings -- UMAP Projection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"UMAP visualization saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract DNA sequence embeddings with Nucleotide Transformer"
    )
    parser.add_argument(
        "--input", required=True, help="Path to FASTA file or plain-text file with DNA sequences"
    )
    parser.add_argument(
        "--model",
        default="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for embeddings (.npy or .h5)",
    )
    parser.add_argument(
        "--format",
        choices=["numpy", "hdf5"],
        default=None,
        help="Output format. Auto-detected from extension if omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument(
        "--umap",
        default=None,
        help="If provided, save a UMAP visualization to this path (e.g. umap.png)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional CSV with columns: sequence_id, label. Used to color UMAP plot.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors parameter")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist parameter")
    parser.add_argument(
        "--device", default=None, help="Device (cuda/cpu). Auto-detected if omitted."
    )

    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    # Parse sequences
    sequences = parse_fasta(args.input)
    logger.info(f"Loaded {len(sequences)} sequences from {args.input}")

    if not sequences:
        logger.error("No sequences found in input file.")
        sys.exit(1)

    # Extract embeddings
    embeddings, headers = extract_embeddings(
        sequences, args.model, batch_size=args.batch_size, device=device
    )

    # Determine output format
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".h5") or args.output.endswith(".hdf5"):
            fmt = "hdf5"
        else:
            fmt = "numpy"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if fmt == "hdf5":
        save_hdf5(embeddings, headers, args.output)
    else:
        save_numpy(embeddings, headers, args.output)

    # Optional UMAP visualization
    if args.umap:
        labels = load_labels(args.labels) if args.labels else None
        os.makedirs(os.path.dirname(args.umap) or ".", exist_ok=True)
        generate_umap(
            embeddings,
            headers,
            args.umap,
            labels=labels,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )


if __name__ == "__main__":
    main()

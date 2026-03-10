#!/usr/bin/env python3
"""
RNA Embedding Extraction with AIDO.RNA-1.6B

Extract per-nucleotide and mean-pooled RNA embeddings using the AIDO.RNA
foundation model (1.6B parameters) via the ModelGenerator framework.
Supports FASTA input, batch processing, and optional dimensionality
reduction visualization.

Usage:
    # Extract embeddings from a FASTA file
    python rna_embeddings.py --input rna_sequences.fasta --output embeddings.h5

    # Extract with UMAP visualization
    python rna_embeddings.py --input rna_sequences.fasta --output embeddings.h5 --visualize umap

    # Pass sequences directly on the command line
    python rna_embeddings.py --sequences ACGUACGU UUAGCCGA --output embeddings.npz

    # Save per-nucleotide embeddings (not just mean-pooled)
    python rna_embeddings.py --input rna_sequences.fasta --output embeddings.h5 --per-nucleotide

    # Use the CDS-optimized variant for coding sequences
    python rna_embeddings.py --input cds.fasta --output embeddings.npz --backbone aido_rna_1b600m_cds

AIDO.RNA (1.6B params) is the largest RNA foundation model, trained on 42M
non-coding RNA sequences from RNAcentral v24.0. It achieves SOTA on 24/26
RNA understanding tasks across a 26-dataset benchmark.

Requirements:
    pip install modelgenerator numpy
    pip install h5py          # optional, for HDF5 output
    pip install umap-learn    # optional, for UMAP visualization
    pip install matplotlib    # optional, for visualization
    pip install scikit-learn  # optional, for PCA visualization
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """
    Parse a FASTA file into a list of (header, sequence) tuples.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        List of (header, sequence) pairs.
    """
    records: list[tuple[str, str]] = []
    header = ""
    seq_lines: list[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_lines)))
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line.upper())
        if header:
            records.append((header, "".join(seq_lines)))

    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BACKBONE_MAP = {
    "aido_rna_1b600m": "aido_rna_1b600m",
    "aido_rna_1b600m_cds": "aido_rna_1b600m_cds",
}


def load_model(backbone: str):
    """
    Load an AIDO.RNA embedding model via ModelGenerator.

    Args:
        backbone: ModelGenerator backbone identifier
                  (e.g. "aido_rna_1b600m" or "aido_rna_1b600m_cds").

    Returns:
        ModelGenerator Embed task model in eval mode.
    """
    from modelgenerator.tasks import Embed

    print(f"Loading AIDO.RNA backbone '{backbone}' ...")
    model = Embed.from_config({"model.backbone": backbone}).eval()
    print("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model,
    sequences: list[str],
    batch_size: int = 8,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Extract per-nucleotide and mean-pooled embeddings for a list of RNA sequences.

    Args:
        model: ModelGenerator Embed task model.
        sequences: List of RNA nucleotide strings (A, C, G, U).
        batch_size: Number of sequences per forward pass.

    Returns:
        (per_nucleotide, mean_pooled) where per_nucleotide is a list of arrays
        with shape (seq_len, hidden_dim) and mean_pooled is an array with
        shape (n_sequences, hidden_dim).
    """
    per_nucleotide: list[np.ndarray] = []
    mean_pooled_list: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        print(
            f"  Processing batch {start // batch_size + 1} "
            f"({len(batch_seqs)} sequences) ..."
        )

        transformed_batch = model.transform({"sequences": batch_seqs})

        with torch.no_grad():
            # ModelGenerator Embed returns tensor of shape
            # (batch, seq_len, hidden_dim) for per-token embeddings
            # or (batch, hidden_dim) for pooled embeddings depending on config.
            output = model(transformed_batch)

        if isinstance(output, torch.Tensor):
            output_np = output.cpu().numpy()
        else:
            output_np = np.array(output)

        # Handle both pooled (2D) and per-token (3D) outputs
        if output_np.ndim == 3:
            # Per-token embeddings: (batch, seq_len, hidden_dim)
            for i, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                # Extract only the nucleotide positions (skip special tokens)
                nucleotide_emb = output_np[i, 1 : seq_len + 1, :]
                per_nucleotide.append(nucleotide_emb)

                # Mean pooling over nucleotide positions
                pooled = nucleotide_emb.mean(axis=0)
                mean_pooled_list.append(pooled)
        elif output_np.ndim == 2:
            # Already pooled: (batch, hidden_dim)
            for i, seq in enumerate(batch_seqs):
                mean_pooled_list.append(output_np[i])
                # No per-nucleotide embeddings available in pooled mode
                per_nucleotide.append(output_np[i].reshape(1, -1))

    mean_pooled = np.stack(mean_pooled_list, axis=0)
    return per_nucleotide, mean_pooled


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_embeddings(
    output_path: str,
    headers: list[str],
    per_nucleotide: list[np.ndarray],
    mean_pooled: np.ndarray,
    save_per_nucleotide: bool,
):
    """
    Save embeddings to .npz or .h5 depending on output_path extension.

    Args:
        output_path: Destination file (.npz or .h5/.hdf5).
        headers: Sequence identifiers.
        per_nucleotide: List of per-nucleotide embedding arrays.
        mean_pooled: Mean-pooled embeddings array.
        save_per_nucleotide: Whether to include per-nucleotide embeddings.
    """
    ext = Path(output_path).suffix.lower()

    if ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(output_path, "w") as hf:
            hf.create_dataset("mean_pooled", data=mean_pooled)
            hf.create_dataset("headers", data=np.array(headers, dtype="S"))
            if save_per_nucleotide:
                grp = hf.create_group("per_nucleotide")
                for hdr, emb in zip(headers, per_nucleotide):
                    grp.create_dataset(hdr, data=emb)
        print(f"Saved HDF5 embeddings to {output_path}")

    else:
        # Default to numpy compressed archive
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        save_dict = {
            "mean_pooled": mean_pooled,
            "headers": np.array(headers),
        }
        if save_per_nucleotide:
            for hdr, emb in zip(headers, per_nucleotide):
                save_dict[f"per_nucleotide_{hdr}"] = emb
        np.savez_compressed(output_path, **save_dict)
        print(f"Saved numpy embeddings to {output_path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_embeddings(
    mean_pooled: np.ndarray,
    headers: list[str],
    method: str = "pca",
    output_path: str | None = None,
):
    """
    Reduce embeddings to 2D and produce a scatter plot.

    Args:
        mean_pooled: (n_sequences, hidden_dim) array.
        headers: Labels for each point.
        method: 'pca' or 'umap'.
        output_path: If given, derive plot filename from this path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mean_pooled.shape[0] < 2:
        print("Need at least 2 sequences to visualize. Skipping.")
        return

    print(f"Running {method.upper()} dimensionality reduction ...")

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(mean_pooled)
        x_label = f"PC1 ({reducer.explained_variance_ratio_[0]:.1%} var)"
        y_label = f"PC2 ({reducer.explained_variance_ratio_[1]:.1%} var)"
    elif method == "umap":
        try:
            import umap
        except ImportError:
            print("umap-learn not installed. Install with: pip install umap-learn")
            print("Falling back to PCA.")
            return visualize_embeddings(mean_pooled, headers, "pca", output_path)

        n_neighbors = min(15, mean_pooled.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        coords = reducer.fit_transform(mean_pooled)
        x_label = "UMAP1"
        y_label = "UMAP2"
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca' or 'umap'.")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=40)

    # Annotate points (limit labels to avoid clutter)
    max_labels = 50
    for i, hdr in enumerate(headers[:max_labels]):
        ax.annotate(
            hdr[:20],
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.6,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"AIDO.RNA Embeddings ({method.upper()})")
    plt.tight_layout()

    if output_path:
        fig_path = str(Path(output_path).with_suffix("")) + f"_{method}.png"
    else:
        fig_path = f"rna_embeddings_{method}.png"

    plt.savefig(fig_path, dpi=150)
    print(f"Visualization saved to {fig_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract RNA embeddings using AIDO.RNA-1.6B via ModelGenerator. "
            "AIDO.RNA is the largest RNA foundation model (1.6B params), "
            "trained on 42M ncRNA sequences from RNAcentral v24.0."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to FASTA file containing RNA sequences.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="RNA sequences passed directly on the command line.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="aido_rna_1b600m",
        choices=list(BACKBONE_MAP.keys()),
        help=(
            "ModelGenerator backbone identifier (default: aido_rna_1b600m). "
            "Use aido_rna_1b600m_cds for coding sequences."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rna_embeddings.npz",
        help="Output file path (.npz or .h5/.hdf5).",
    )
    parser.add_argument(
        "--per-nucleotide",
        action="store_true",
        help="Also save per-nucleotide embeddings (increases file size).",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["pca", "umap"],
        default=None,
        help="Produce a 2D scatter plot of mean-pooled embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8).",
    )

    args = parser.parse_args()

    # ---- Gather sequences ------------------------------------------------
    if args.input and args.sequences:
        print(
            "Error: provide either --input or --sequences, not both.",
            file=sys.stderr,
        )
        return 1

    if args.input:
        records = parse_fasta(args.input)
        if not records:
            print(f"Error: no sequences found in {args.input}", file=sys.stderr)
            return 1
        headers = [r[0] for r in records]
        sequences = [r[1] for r in records]
    elif args.sequences:
        headers = [f"seq_{i}" for i in range(len(args.sequences))]
        sequences = [s.upper().replace("T", "U") for s in args.sequences]
    else:
        print("Error: provide --input (FASTA) or --sequences.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("AIDO.RNA Embedding Extraction")
    print("=" * 60)
    print(f"Sequences:  {len(sequences)}")
    print(f"Backbone:   {args.backbone}")
    print(f"Output:     {args.output}")

    # ---- Load model ------------------------------------------------------
    model = load_model(args.backbone)

    # ---- Extract embeddings ----------------------------------------------
    print("\nExtracting embeddings ...")
    per_nucleotide, mean_pooled = extract_embeddings(
        model, sequences, batch_size=args.batch_size
    )
    print(f"Mean-pooled shape: {mean_pooled.shape}")

    # ---- Save ------------------------------------------------------------
    save_embeddings(
        args.output, headers, per_nucleotide, mean_pooled, args.per_nucleotide
    )

    # ---- Visualize -------------------------------------------------------
    if args.visualize:
        visualize_embeddings(mean_pooled, headers, args.visualize, args.output)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

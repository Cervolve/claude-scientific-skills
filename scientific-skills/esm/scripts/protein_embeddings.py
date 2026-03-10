#!/usr/bin/env python3
"""
Protein Embedding Extraction with ESM-2

Extract per-residue and mean-pooled protein embeddings from ESM-2 models
via HuggingFace Transformers. Supports FASTA input, batch processing,
and optional dimensionality reduction visualization.

Usage:
    # Extract embeddings from a FASTA file
    python protein_embeddings.py --input proteins.fasta --output embeddings.h5

    # Use a specific model size and visualize with UMAP
    python protein_embeddings.py --input proteins.fasta --model-size 150M --visualize umap

    # Extract embeddings for sequences passed directly
    python protein_embeddings.py --sequences MKFLILLFNILCL MGTRELPSALLL --output embeddings.npz

    # Save per-residue embeddings (not just mean-pooled)
    python protein_embeddings.py --input proteins.fasta --output embeddings.h5 --per-residue

ESM-2 650M offers best performance/size tradeoff per PFMBench 2025.
Larger models (3B, 15B) show only marginal gains.
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
                seq_lines.append(line)
        if header:
            records.append((header, "".join(seq_lines)))

    return records


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "15B": "facebook/esm2_t48_15B_UR50D",
}


def load_model(model_size: str, device: str):
    """
    Load an ESM-2 model and tokenizer from HuggingFace.

    Args:
        model_size: One of 8M, 35M, 150M, 650M, 3B, 15B.
        device: Torch device string (e.g. "cuda", "cpu").

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import AutoModel, AutoTokenizer

    model_name = MODEL_MAP.get(model_size)
    if model_name is None:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Choose from: {', '.join(MODEL_MAP.keys())}"
        )

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"Model loaded on {device}.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    model,
    tokenizer,
    sequences: list[str],
    device: str,
    batch_size: int = 8,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Extract per-residue and mean-pooled embeddings for a list of sequences.

    Args:
        model: ESM-2 HuggingFace model.
        tokenizer: Corresponding tokenizer.
        sequences: List of amino-acid strings.
        device: Torch device.
        batch_size: Number of sequences per forward pass.

    Returns:
        (per_residue, mean_pooled) where per_residue is a list of arrays
        with shape (seq_len, hidden_dim) and mean_pooled is an array with
        shape (n_sequences, hidden_dim).
    """
    per_residue: list[np.ndarray] = []
    mean_pooled_list: list[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        print(
            f"  Processing batch {start // batch_size + 1} "
            f"({len(batch_seqs)} sequences) ..."
        )

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
        hidden_states = outputs.last_hidden_state.cpu().numpy()
        attention_mask = inputs["attention_mask"].cpu().numpy()

        for i, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Strip special tokens: [CLS] residues... [EOS] [PAD]...
            residue_emb = hidden_states[i, 1 : seq_len + 1, :]
            per_residue.append(residue_emb)

            # Mean pooling over actual residue positions
            mask = attention_mask[i, 1 : seq_len + 1].astype(np.float32)
            pooled = (residue_emb * mask[:, None]).sum(axis=0) / mask.sum()
            mean_pooled_list.append(pooled)

    mean_pooled = np.stack(mean_pooled_list, axis=0)
    return per_residue, mean_pooled


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_embeddings(
    output_path: str,
    headers: list[str],
    per_residue: list[np.ndarray],
    mean_pooled: np.ndarray,
    save_per_residue: bool,
):
    """
    Save embeddings to .npz or .h5 depending on output_path extension.

    Args:
        output_path: Destination file (.npz or .h5).
        headers: Sequence identifiers.
        per_residue: List of per-residue embedding arrays.
        mean_pooled: Mean-pooled embeddings array.
        save_per_residue: Whether to include per-residue embeddings.
    """
    ext = Path(output_path).suffix.lower()

    if ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(output_path, "w") as hf:
            hf.create_dataset("mean_pooled", data=mean_pooled)
            hf.create_dataset("headers", data=np.array(headers, dtype="S"))
            if save_per_residue:
                grp = hf.create_group("per_residue")
                for i, (hdr, emb) in enumerate(zip(headers, per_residue)):
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
        if save_per_residue:
            for i, (hdr, emb) in enumerate(zip(headers, per_residue)):
                save_dict[f"per_residue_{hdr}"] = emb
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
        output_path: If given, save plot to file instead of showing.
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
    ax.set_title(f"ESM-2 Protein Embeddings ({method.upper()})")
    plt.tight_layout()

    if output_path:
        fig_path = str(Path(output_path).with_suffix("")) + f"_{method}.png"
    else:
        fig_path = f"embeddings_{method}.png"

    plt.savefig(fig_path, dpi=150)
    print(f"Visualization saved to {fig_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract protein embeddings using ESM-2 from HuggingFace."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Protein sequences passed directly on the command line.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="650M",
        choices=list(MODEL_MAP.keys()),
        help="ESM-2 model size (default: 650M).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings.npz",
        help="Output file path (.npz or .h5/.hdf5).",
    )
    parser.add_argument(
        "--per-residue",
        action="store_true",
        help="Also save per-residue embeddings (increases file size).",
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/cpu).",
    )

    args = parser.parse_args()

    # ---- Gather sequences ------------------------------------------------
    if args.input and args.sequences:
        print("Error: provide either --input or --sequences, not both.", file=sys.stderr)
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
        sequences = args.sequences
    else:
        print("Error: provide --input (FASTA) or --sequences.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("ESM-2 Protein Embedding Extraction")
    print("=" * 60)
    print(f"Sequences: {len(sequences)}")
    print(f"Model:     {MODEL_MAP[args.model_size]}")
    print(f"Output:    {args.output}")

    # ---- Device ----------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ------------------------------------------------------
    model, tokenizer = load_model(args.model_size, device)

    # ---- Extract embeddings ----------------------------------------------
    print("\nExtracting embeddings ...")
    per_residue, mean_pooled = extract_embeddings(
        model, tokenizer, sequences, device, batch_size=args.batch_size
    )
    print(f"Mean-pooled shape: {mean_pooled.shape}")

    # ---- Save ------------------------------------------------------------
    save_embeddings(args.output, headers, per_residue, mean_pooled, args.per_residue)

    # ---- Visualize -------------------------------------------------------
    if args.visualize:
        visualize_embeddings(mean_pooled, headers, args.visualize, args.output)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

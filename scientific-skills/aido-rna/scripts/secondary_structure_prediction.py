#!/usr/bin/env python3
"""
RNA Secondary Structure Prediction with AIDO.RNA-1.6B

Predict RNA secondary structure in dot-bracket notation using the AIDO.RNA
foundation model via ModelGenerator's token-level classification task.

AIDO.RNA achieves F1=0.787 on bpRNA-TS0 (SOTA). For unseen RNA families,
RiNALMo may generalize better.

Usage:
    # Predict structure for a single sequence
    python secondary_structure_prediction.py --sequence GGGAAACCC

    # Predict structures from a FASTA file
    python secondary_structure_prediction.py --input rna_sequences.fasta

    # Save predictions to a file
    python secondary_structure_prediction.py --input rna_sequences.fasta --output structures.tsv

    # Use a fine-tuned checkpoint
    python secondary_structure_prediction.py --sequence GGGAAACCC --checkpoint path/to/checkpoint.ckpt

The model uses 3-class token classification:
    Class 0: '.' (unpaired)
    Class 1: '(' (paired, opening)
    Class 2: ')' (paired, closing)

Requirements:
    pip install modelgenerator torch numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# Mapping from class index to dot-bracket character
CLASS_TO_BRACKET = {0: ".", 1: "(", 2: ")"}

# Number of structure classes (unpaired, left-paired, right-paired)
N_STRUCTURE_CLASSES = 3


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

def load_model(checkpoint: str | None = None):
    """
    Load AIDO.RNA for token-level secondary structure classification.

    If a fine-tuned checkpoint is provided, loads from that checkpoint.
    Otherwise, loads the pretrained backbone with a randomly initialized
    classification head (suitable for inference only if you have fine-tuned
    weights, or for demonstration purposes).

    Args:
        checkpoint: Optional path to a fine-tuned ModelGenerator checkpoint.

    Returns:
        TokenClassification model in eval mode.
    """
    from modelgenerator.tasks import TokenClassification

    if checkpoint:
        print(f"Loading fine-tuned checkpoint from {checkpoint} ...")
        model = TokenClassification.load_from_checkpoint(checkpoint).eval()
    else:
        print("Loading AIDO.RNA backbone with structure prediction head ...")
        print(
            "NOTE: Without a fine-tuned checkpoint, the classification head "
            "is randomly initialized. For accurate predictions, provide a "
            "checkpoint trained on secondary structure data (e.g., bpRNA)."
        )
        model = TokenClassification.from_config({
            "model.backbone": "aido_rna_1b600m",
            "model.n_classes": N_STRUCTURE_CLASSES,
        }).eval()

    print("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Structure prediction
# ---------------------------------------------------------------------------

def predict_structure(
    model,
    sequences: list[str],
    batch_size: int = 8,
) -> list[str]:
    """
    Predict RNA secondary structure in dot-bracket notation.

    Args:
        model: ModelGenerator TokenClassification model.
        sequences: List of RNA sequences (A, C, G, U).
        batch_size: Number of sequences per forward pass.

    Returns:
        List of dot-bracket strings, one per input sequence.
    """
    all_structures: list[str] = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
        print(
            f"  Predicting batch {start // batch_size + 1} "
            f"({len(batch_seqs)} sequences) ..."
        )

        transformed_batch = model.transform({"sequences": batch_seqs})

        with torch.no_grad():
            # logits shape: (batch, seq_len, n_classes)
            logits = model(transformed_batch)

        if isinstance(logits, torch.Tensor):
            logits_np = logits.cpu().numpy()
        else:
            logits_np = np.array(logits)

        # Convert per-token predictions to dot-bracket notation
        for i, seq in enumerate(batch_seqs):
            seq_len = len(seq)

            # Extract logits for nucleotide positions (skip special tokens)
            if logits_np.ndim == 3:
                token_logits = logits_np[i, 1 : seq_len + 1, :]
            else:
                # Fallback if logits are already trimmed
                token_logits = logits_np[i, :seq_len, :]

            predicted_classes = np.argmax(token_logits, axis=-1)

            # Build dot-bracket string
            dot_bracket = "".join(
                CLASS_TO_BRACKET.get(cls, ".") for cls in predicted_classes
            )

            # Post-process: ensure bracket balance
            dot_bracket = _balance_brackets(dot_bracket)

            all_structures.append(dot_bracket)

    return all_structures


def _balance_brackets(dot_bracket: str) -> str:
    """
    Post-process dot-bracket notation to ensure balanced parentheses.

    Unmatched closing brackets are converted to unpaired '.', and
    unmatched opening brackets are converted to unpaired '.'.

    Args:
        dot_bracket: Raw dot-bracket string from model predictions.

    Returns:
        Balanced dot-bracket string.
    """
    result = list(dot_bracket)
    stack: list[int] = []

    for i, char in enumerate(result):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                stack.pop()
            else:
                # Unmatched closing bracket
                result[i] = "."

    # Remaining unmatched opening brackets
    for i in stack:
        result[i] = "."

    return "".join(result)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results(
    headers: list[str],
    sequences: list[str],
    structures: list[str],
) -> str:
    """
    Format prediction results for display.

    Args:
        headers: Sequence identifiers.
        sequences: RNA sequences.
        structures: Predicted dot-bracket strings.

    Returns:
        Formatted string for printing.
    """
    lines: list[str] = []

    for header, seq, struct in zip(headers, sequences, structures):
        # Count paired and unpaired nucleotides
        n_paired = struct.count("(") + struct.count(")")
        n_unpaired = struct.count(".")
        pair_fraction = n_paired / len(struct) if struct else 0.0

        lines.append(f">{header}")
        lines.append(f"  Sequence:  {seq}")
        lines.append(f"  Structure: {struct}")
        lines.append(
            f"  Length: {len(seq)} nt | "
            f"Paired: {n_paired} ({pair_fraction:.1%}) | "
            f"Unpaired: {n_unpaired}"
        )
        lines.append("")

    return "\n".join(lines)


def save_results(
    output_path: str,
    headers: list[str],
    sequences: list[str],
    structures: list[str],
):
    """
    Save predictions to a TSV file.

    Args:
        output_path: Destination file path.
        headers: Sequence identifiers.
        sequences: RNA sequences.
        structures: Predicted dot-bracket strings.
    """
    with open(output_path, "w") as fh:
        fh.write("id\tsequence\tstructure\tlength\tpaired_fraction\n")
        for header, seq, struct in zip(headers, sequences, structures):
            n_paired = struct.count("(") + struct.count(")")
            pair_fraction = n_paired / len(struct) if struct else 0.0
            fh.write(
                f"{header}\t{seq}\t{struct}\t{len(seq)}\t{pair_fraction:.4f}\n"
            )
    print(f"Predictions saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict RNA secondary structure using AIDO.RNA-1.6B. "
            "AIDO.RNA achieves F1=0.787 on bpRNA-TS0 (SOTA). "
            "For unseen RNA families, RiNALMo may generalize better."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to FASTA file containing RNA sequences.",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Single RNA sequence passed directly on the command line.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a fine-tuned ModelGenerator checkpoint (.ckpt). "
            "Without this, the classification head is randomly initialized."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TSV file for predictions. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8).",
    )

    args = parser.parse_args()

    # ---- Gather sequences ------------------------------------------------
    if args.input and args.sequence:
        print(
            "Error: provide either --input or --sequence, not both.",
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
    elif args.sequence:
        seq = args.sequence.upper().replace("T", "U")
        headers = ["query"]
        sequences = [seq]
    else:
        print("Error: provide --input (FASTA) or --sequence.", file=sys.stderr)
        return 1

    # ---- Validate sequences ----------------------------------------------
    valid_chars = set("ACGUNRYSWKMBDHV")
    for i, seq in enumerate(sequences):
        invalid = set(seq) - valid_chars
        if invalid:
            print(
                f"Warning: sequence '{headers[i]}' contains non-standard "
                f"characters: {invalid}. These may affect predictions.",
                file=sys.stderr,
            )

    print("=" * 60)
    print("AIDO.RNA Secondary Structure Prediction")
    print("=" * 60)
    print(f"Sequences:  {len(sequences)}")
    print(f"Checkpoint: {args.checkpoint or '(pretrained backbone, random head)'}")

    # ---- Load model ------------------------------------------------------
    model = load_model(args.checkpoint)

    # ---- Predict structures ----------------------------------------------
    print("\nPredicting secondary structures ...")
    structures = predict_structure(model, sequences, batch_size=args.batch_size)

    # ---- Output ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(format_results(headers, sequences, structures))

    if args.output:
        save_results(args.output, headers, sequences, structures)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

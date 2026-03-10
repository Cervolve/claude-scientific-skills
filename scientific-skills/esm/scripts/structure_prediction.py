#!/usr/bin/env python3
"""
Rapid Structure Prediction with ESMFold

Predict 3D protein structures from amino-acid sequences using ESMFold,
a single-sequence structure prediction model. Outputs PDB files with
per-residue pLDDT confidence scores and flags low-confidence regions.

Usage:
    # Predict structure for sequences in a FASTA file
    python structure_prediction.py --input proteins.fasta --output-dir ./structures

    # Predict a single sequence
    python structure_prediction.py --sequence MKFLILLFNILCLFPVLAADNHGVSMRV --output-dir ./structures

    # Set a custom pLDDT threshold for low-confidence warnings
    python structure_prediction.py --input proteins.fasta --output-dir ./structures --plddt-threshold 70

ESMFold: 76% accuracy on monomers (vs AF2's 88%). Best for rapid
screening. For accuracy-critical work, use AlphaFold3 or Boltz-2.
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

def load_esmfold(device: str):
    """
    Load the ESMFold model from HuggingFace.

    Args:
        device: Torch device string.

    Returns:
        ESMFold model and tokenizer.
    """
    from transformers import AutoTokenizer, EsmForProteinFolding

    model_name = "facebook/esmfold_v1"
    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name)

    # ESMFold benefits from float16 on GPU
    if device != "cpu":
        model = model.half()

    model = model.to(device).eval()
    print(f"Model loaded on {device}.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Structure prediction
# ---------------------------------------------------------------------------

def predict_structure(
    model,
    tokenizer,
    sequence: str,
    device: str,
) -> tuple[str, np.ndarray]:
    """
    Predict 3D structure for a single protein sequence.

    Args:
        model: ESMFold model.
        tokenizer: Corresponding tokenizer.
        sequence: Amino-acid sequence string.
        device: Torch device.

    Returns:
        (pdb_string, plddt_per_residue) tuple.
    """
    inputs = tokenizer(
        [sequence],
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert atom positions to PDB
    # outputs.positions: (batch, num_residues, num_atoms, 3)
    # outputs.plddt: (batch, num_residues, num_atoms)
    pdb_string = convert_outputs_to_pdb(outputs)[0]

    # Extract per-residue pLDDT (mean over atoms per residue)
    plddt = outputs["plddt"].cpu().numpy()[0]  # (num_residues, num_atoms)
    # Take the CA atom pLDDT (atom index 1) or mean across atoms
    if plddt.ndim == 2:
        plddt_per_residue = plddt.mean(axis=-1)
    else:
        plddt_per_residue = plddt

    return pdb_string, plddt_per_residue


def convert_outputs_to_pdb(outputs) -> list[str]:
    """
    Convert ESMFold model outputs to PDB format strings.

    Uses the built-in conversion from the transformers library.
    """
    from transformers.models.esm.openfold_utils.protein import Protein, to_pdb

    final_atom_positions = outputs["positions"][-1].cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"].cpu().numpy()
    pdbs = []

    for i in range(final_atom_positions.shape[0]):
        aa_seq = outputs.get("aatype", None)
        if aa_seq is not None:
            aatype = aa_seq[i].cpu().numpy()
        else:
            aatype = np.zeros(final_atom_positions.shape[1], dtype=np.int64)

        # pLDDT as B-factor
        plddt = outputs["plddt"][i].cpu().numpy()
        if plddt.ndim == 2:
            # Expand to atom37 format: replicate per-residue pLDDT to all atoms
            b_factors = np.repeat(
                plddt.mean(axis=-1, keepdims=True),
                final_atom_mask.shape[-1],
                axis=-1,
            )
        else:
            b_factors = np.repeat(
                plddt[:, None],
                final_atom_mask.shape[-1],
                axis=-1,
            )

        protein = Protein(
            aatype=aatype,
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=np.arange(final_atom_positions.shape[1]),
            chain_index=np.zeros(final_atom_positions.shape[1], dtype=np.int64),
            b_factors=b_factors[i] if b_factors.ndim == 3 else b_factors,
        )
        pdbs.append(to_pdb(protein))

    return pdbs


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

def analyze_confidence(
    header: str,
    sequence: str,
    plddt: np.ndarray,
    threshold: float = 50.0,
) -> dict:
    """
    Analyze per-residue pLDDT scores and flag low-confidence regions.

    Args:
        header: Sequence identifier.
        sequence: Amino-acid sequence.
        plddt: Per-residue pLDDT array.
        threshold: pLDDT threshold below which residues are flagged.

    Returns:
        Dict with summary statistics and flagged residues.
    """
    plddt_clipped = plddt[: len(sequence)]  # ensure alignment
    mean_plddt = float(np.mean(plddt_clipped))
    median_plddt = float(np.median(plddt_clipped))
    low_conf_mask = plddt_clipped < threshold
    low_conf_positions = np.where(low_conf_mask)[0]

    # Group consecutive low-confidence positions into regions
    regions: list[tuple[int, int]] = []
    if len(low_conf_positions) > 0:
        start = int(low_conf_positions[0])
        prev = start
        for pos in low_conf_positions[1:]:
            if pos == prev + 1:
                prev = int(pos)
            else:
                regions.append((start + 1, prev + 1))  # 1-based
                start = int(pos)
                prev = start
        regions.append((start + 1, prev + 1))

    return {
        "header": header,
        "length": len(sequence),
        "mean_plddt": mean_plddt,
        "median_plddt": median_plddt,
        "n_low_confidence": int(low_conf_mask.sum()),
        "frac_low_confidence": float(low_conf_mask.mean()),
        "low_confidence_regions": regions,
    }


def print_confidence_report(analysis: dict, threshold: float):
    """Print a human-readable confidence report for one sequence."""
    a = analysis
    print(f"\n  Sequence:    {a['header']} ({a['length']} residues)")
    print(f"  Mean pLDDT:  {a['mean_plddt']:.1f}")
    print(f"  Median pLDDT: {a['median_plddt']:.1f}")

    quality = (
        "Very high" if a["mean_plddt"] >= 90
        else "Confident" if a["mean_plddt"] >= 70
        else "Low" if a["mean_plddt"] >= 50
        else "Very low"
    )
    print(f"  Quality:     {quality}")

    n_low = a["n_low_confidence"]
    if n_low > 0:
        pct = a["frac_low_confidence"] * 100
        print(f"  WARNING: {n_low} residue(s) ({pct:.1f}%) below pLDDT {threshold}")
        print(f"  Low-confidence regions (1-based):")
        for start, end in a["low_confidence_regions"]:
            if start == end:
                print(f"    - Position {start}")
            else:
                print(f"    - Positions {start}-{end}")
    else:
        print(f"  All residues above pLDDT {threshold} threshold.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Predict protein structures with ESMFold and assess confidence."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to FASTA file with protein sequences.",
    )
    input_group.add_argument(
        "--sequence",
        type=str,
        help="Single protein sequence.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./esmfold_output",
        help="Directory for output PDB files (default: ./esmfold_output).",
    )
    parser.add_argument(
        "--plddt-threshold",
        type=float,
        default=50.0,
        help="pLDDT threshold for flagging low-confidence residues (default: 50).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=800,
        help="Maximum sequence length to predict (default: 800). "
             "Longer sequences require significant GPU memory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/cpu).",
    )

    args = parser.parse_args()

    # ---- Gather sequences ------------------------------------------------
    if args.input:
        records = parse_fasta(args.input)
        if not records:
            print(f"Error: no sequences found in {args.input}", file=sys.stderr)
            return 1
    else:
        records = [("query", args.sequence.strip().upper())]

    print("=" * 60)
    print("ESMFold Structure Prediction")
    print("=" * 60)
    print(f"Sequences: {len(records)}")
    print(f"pLDDT threshold: {args.plddt_threshold}")

    # ---- Device ----------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: Running on CPU. Structure prediction will be very slow.")
        print("         GPU with >= 8 GB VRAM is strongly recommended.")

    # ---- Output directory ------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ------------------------------------------------------
    model, tokenizer = load_esmfold(device)

    # ---- Predict structures ----------------------------------------------
    print(f"\nPredicting structures ...")
    analyses = []
    skipped = 0

    for i, (header, sequence) in enumerate(records):
        print(f"\n[{i + 1}/{len(records)}] {header} ({len(sequence)} residues)")

        if len(sequence) > args.max_length:
            print(
                f"  SKIPPED: sequence length {len(sequence)} exceeds "
                f"--max-length {args.max_length}. Increase limit or truncate."
            )
            skipped += 1
            continue

        try:
            pdb_string, plddt = predict_structure(model, tokenizer, sequence, device)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Save PDB
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in header)
        pdb_path = output_dir / f"{safe_name}.pdb"
        with open(pdb_path, "w") as fh:
            fh.write(pdb_string)
        print(f"  PDB saved: {pdb_path}")

        # Save pLDDT scores
        plddt_path = output_dir / f"{safe_name}_plddt.npy"
        np.save(plddt_path, plddt[: len(sequence)])

        # Confidence analysis
        analysis = analyze_confidence(header, sequence, plddt, args.plddt_threshold)
        analyses.append(analysis)
        print_confidence_report(analysis, args.plddt_threshold)

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Predicted: {len(analyses)}")
    print(f"  Skipped:   {skipped}")

    if analyses:
        mean_plddt_all = np.mean([a["mean_plddt"] for a in analyses])
        print(f"  Average pLDDT across all predictions: {mean_plddt_all:.1f}")

        flagged = [a for a in analyses if a["n_low_confidence"] > 0]
        if flagged:
            print(f"\n  Sequences with low-confidence regions ({len(flagged)}):")
            for a in flagged:
                pct = a["frac_low_confidence"] * 100
                print(f"    - {a['header']}: {a['n_low_confidence']} residues ({pct:.1f}%)")

    print(f"\n  Output directory: {output_dir.resolve()}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

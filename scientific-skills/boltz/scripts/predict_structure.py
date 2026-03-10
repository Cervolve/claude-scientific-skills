#!/usr/bin/env python3
"""
Protein Structure Prediction with Boltz-2

Predict 3D structures of proteins and biomolecular complexes using Boltz-2,
an open-source (MIT licensed) biomolecular foundation model. Supports proteins,
RNA, DNA, and their complexes.

Boltz-2 matches AlphaFold3 accuracy on most targets (FoldBench 2025). MIT
licensed. Strongest improvements on RNA and DNA-protein complexes.

Usage:
    # Predict structure from a FASTA file
    python predict_structure.py --input protein.fasta --output ./results

    # Predict from a YAML input (supports ligands, RNA, DNA, multi-chain)
    python predict_structure.py --input complex.yaml --output ./results

    # Predict with MSA server and multiple diffusion samples
    python predict_structure.py --input protein.fasta --output ./results --use_msa_server --diffusion_samples 5

    # Run on CPU (slower but no GPU required)
    python predict_structure.py --input protein.fasta --output ./results --device cpu

Input formats:
    - FASTA: Single or multi-sequence FASTA files (protein only)
    - YAML: Rich input format supporting proteins, ligands (SMILES/CCD),
      RNA, DNA, and property predictions (affinity, etc.)

Output:
    - mmCIF structure file(s) in the output directory
    - Confidence JSON with pLDDT, pDE, and pAE metrics
    - Summary printed to stdout

Examples:
    # Simple protein structure prediction
    $ echo ">chain_A\\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH" > input.fasta
    $ python predict_structure.py --input input.fasta --output ./results

    # Multi-chain complex from YAML
    $ cat complex.yaml
    sequences:
      - protein:
          id: A
          sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
      - protein:
          id: B
          sequence: MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST
    $ python predict_structure.py --input complex.yaml --output ./results --use_msa_server
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


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
# YAML generation from FASTA
# ---------------------------------------------------------------------------

def fasta_to_yaml(records: list[tuple[str, str]], yaml_path: str) -> None:
    """
    Convert FASTA records to a Boltz-compatible YAML input file.

    Each sequence becomes a protein entity with a chain ID derived from
    the FASTA header.

    Args:
        records: List of (header, sequence) tuples from parse_fasta.
        yaml_path: Path to write the YAML file.
    """
    import yaml

    chains = []
    for header, sequence in records:
        # Use the first token of the header as the chain ID
        chain_id = header.split()[0][:10]  # Truncate long IDs
        chains.append({
            "protein": {
                "id": chain_id,
                "sequence": sequence,
            }
        })

    data = {"sequences": chains}

    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    print(f"Generated YAML input: {yaml_path}")


# ---------------------------------------------------------------------------
# Structure prediction
# ---------------------------------------------------------------------------

def predict_structure(
    input_path: str,
    out_dir: str,
    use_msa_server: bool = True,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    device: str | None = None,
) -> dict:
    """
    Run Boltz-2 structure prediction on an input YAML or FASTA file.

    Boltz-2 matches AlphaFold3 accuracy on most targets (FoldBench 2025).
    MIT licensed. Strongest improvements on RNA and DNA-protein complexes.

    Args:
        input_path: Path to a YAML file (or FASTA, which will be converted).
        out_dir: Directory for output files.
        use_msa_server: Whether to use the remote MSA server for alignments.
        recycling_steps: Number of recycling iterations for refinement.
        diffusion_samples: Number of diffusion samples to generate.
        device: Compute device ('cuda', 'cpu', or None for auto-detect).

    Returns:
        Dictionary with prediction results including output paths and
        confidence metrics.
    """
    import boltz

    input_path = str(Path(input_path).resolve())
    out_dir = str(Path(out_dir).resolve())
    os.makedirs(out_dir, exist_ok=True)

    # Auto-detect device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Boltz-2 structure prediction on {device}...")
    print(f"  Input:             {input_path}")
    print(f"  Output directory:  {out_dir}")
    print(f"  MSA server:        {use_msa_server}")
    print(f"  Recycling steps:   {recycling_steps}")
    print(f"  Diffusion samples: {diffusion_samples}")

    # Run prediction via the boltz Python API
    boltz.predict(
        input_path,
        out_dir=out_dir,
        use_msa_server=use_msa_server,
        recycling_steps=recycling_steps,
        diffusion_samples=diffusion_samples,
        accelerator="gpu" if device.startswith("cuda") else "cpu",
    )

    # Collect results
    results = _collect_results(out_dir, input_path)
    return results


def _collect_results(out_dir: str, input_path: str) -> dict:
    """
    Collect and summarize prediction results from the output directory.

    Args:
        out_dir: The Boltz output directory.
        input_path: Original input path (used to derive prediction name).

    Returns:
        Dictionary summarizing the prediction outputs.
    """
    predictions_dir = Path(out_dir) / "predictions"
    results = {
        "output_dir": out_dir,
        "structures": [],
        "confidence": [],
    }

    if not predictions_dir.exists():
        # Boltz may place outputs directly in out_dir
        predictions_dir = Path(out_dir)

    # Find all structure files (mmCIF format)
    for cif_file in sorted(predictions_dir.rglob("*.cif")):
        results["structures"].append(str(cif_file))
        print(f"  Structure: {cif_file}")

    # Find all confidence files
    for conf_file in sorted(predictions_dir.rglob("confidence_*.json")):
        results["confidence"].append(str(conf_file))
        try:
            with open(conf_file) as fh:
                conf_data = json.load(fh)
            # Print summary confidence metrics
            if "plddt" in conf_data:
                mean_plddt = (
                    sum(conf_data["plddt"]) / len(conf_data["plddt"])
                    if isinstance(conf_data["plddt"], list)
                    else conf_data["plddt"]
                )
                print(f"  Mean pLDDT: {mean_plddt:.3f}")
                results["mean_plddt"] = mean_plddt
        except (json.JSONDecodeError, KeyError):
            pass

    if not results["structures"]:
        print("  WARNING: No structure files found in output directory.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict biomolecular structures with Boltz-2. "
            "Supports proteins, RNA, DNA, ligands, and their complexes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict_structure.py --input protein.fasta --output ./results\n"
            "  python predict_structure.py --input complex.yaml --output ./results --use_msa_server\n"
            "  python predict_structure.py --input complex.yaml --output ./results --diffusion_samples 5\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file (FASTA or YAML format).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./boltz_results",
        help="Output directory for predictions (default: ./boltz_results).",
    )
    parser.add_argument(
        "--use_msa_server",
        action="store_true",
        default=False,
        help="Use remote MSA server for multiple sequence alignments.",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=3,
        help="Number of recycling iterations (default: 3).",
    )
    parser.add_argument(
        "--diffusion_samples",
        type=int,
        default=1,
        help="Number of diffusion samples to generate (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device (default: auto-detect).",
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Boltz-2 Structure Prediction")
    print("=" * 60)

    # If input is FASTA, convert to YAML first
    yaml_input = args.input
    tmp_yaml = None

    if input_path.suffix.lower() in (".fasta", ".fa", ".faa", ".fas"):
        records = parse_fasta(args.input)
        if not records:
            print(f"Error: no sequences found in {args.input}", file=sys.stderr)
            return 1

        print(f"Parsed {len(records)} sequence(s) from FASTA file.")
        for header, seq in records:
            print(f"  {header}: {len(seq)} residues")

        # Write temporary YAML
        tmp_yaml = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, prefix="boltz_"
        )
        tmp_yaml.close()
        fasta_to_yaml(records, tmp_yaml.name)
        yaml_input = tmp_yaml.name

    try:
        results = predict_structure(
            input_path=yaml_input,
            out_dir=args.output,
            use_msa_server=args.use_msa_server,
            recycling_steps=args.recycling_steps,
            diffusion_samples=args.diffusion_samples,
            device=args.device,
        )
    finally:
        # Clean up temporary YAML if created
        if tmp_yaml is not None:
            try:
                os.unlink(tmp_yaml.name)
            except OSError:
                pass

    # Print summary
    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)
    print(f"  Structures generated: {len(results['structures'])}")
    if "mean_plddt" in results:
        plddt = results["mean_plddt"]
        quality = (
            "High confidence"
            if plddt > 0.7
            else "Medium confidence"
            if plddt > 0.5
            else "Low confidence"
        )
        print(f"  Mean pLDDT: {plddt:.3f} ({quality})")
    print(f"  Output directory: {results['output_dir']}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Protein-Ligand Binding Affinity Prediction with Boltz-2

Predict protein-ligand complex structures and estimate binding affinities
using Boltz-2's joint structure-affinity model. Boltz-2 achieves ~0.6
correlation with experimental binding data, on par with FEP calculations
at 1000x speed (~20s/complex vs. 6-12h for FEP).

Usage:
    # Predict binding affinity for a protein-ligand pair
    python binding_affinity.py --protein protein.fasta --ligand "CC(=O)Oc1ccccc1C(=O)O" --output ./results

    # Use a CCD code instead of SMILES
    python binding_affinity.py --protein protein.fasta --ligand ATP --ligand_type ccd --output ./results

    # Provide protein sequence directly
    python binding_affinity.py --protein_seq MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH \\
        --ligand "CC(=O)Oc1ccccc1C(=O)O" --output ./results

    # Compare against known experimental affinity
    python binding_affinity.py --protein protein.fasta --ligand "CC(=O)Oc1ccccc1C(=O)O" \\
        --output ./results --known_affinity -6.5

    # Batch screen multiple ligands
    python binding_affinity.py --protein protein.fasta \\
        --ligand "CC(=O)Oc1ccccc1C(=O)O" "c1ccccc1" "CC(=O)O" \\
        --output ./results

Output:
    - Predicted complex structure (mmCIF)
    - Binding affinity as log10(IC50) in micromolar
    - Binary binder probability (0-1)
    - Confidence metrics (pLDDT, pDE, pAE)
    - Comparison with known affinity if provided
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
# YAML generation for protein-ligand affinity prediction
# ---------------------------------------------------------------------------

def build_affinity_yaml(
    protein_id: str,
    protein_sequence: str,
    ligand_spec: str,
    ligand_type: str = "smiles",
) -> dict:
    """
    Build a Boltz-compatible YAML dict for protein-ligand affinity prediction.

    Args:
        protein_id: Identifier for the protein chain.
        protein_sequence: Amino acid sequence.
        ligand_spec: SMILES string or CCD code for the ligand.
        ligand_type: Either 'smiles' or 'ccd'.

    Returns:
        Dictionary representing the Boltz YAML input.
    """
    protein_entry = {
        "protein": {
            "id": protein_id,
            "sequence": protein_sequence,
        }
    }

    if ligand_type == "smiles":
        ligand_entry = {
            "ligand": {
                "id": "LIG",
                "smiles": ligand_spec,
            }
        }
    elif ligand_type == "ccd":
        ligand_entry = {
            "ligand": {
                "id": "LIG",
                "ccd": ligand_spec,
            }
        }
    else:
        raise ValueError(f"Unknown ligand_type: {ligand_type}. Use 'smiles' or 'ccd'.")

    data = {
        "sequences": [protein_entry, ligand_entry],
        "properties": [
            {
                "affinity": {
                    "binder": [protein_id, "LIG"],
                }
            }
        ],
    }

    return data


def write_yaml(data: dict, yaml_path: str) -> None:
    """Write a dictionary as a YAML file."""
    import yaml

    with open(yaml_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Binding affinity prediction
# ---------------------------------------------------------------------------

def predict_binding_affinity(
    protein_sequence: str,
    ligand_spec: str,
    out_dir: str,
    protein_id: str = "A",
    ligand_type: str = "smiles",
    use_msa_server: bool = True,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    device: str | None = None,
) -> dict:
    """
    Predict protein-ligand binding affinity using Boltz-2.

    Boltz-2 achieves ~0.6 correlation with experimental binding data,
    on par with FEP calculations at 1000x speed.

    Args:
        protein_sequence: Amino acid sequence of the target protein.
        ligand_spec: SMILES string or CCD code for the ligand.
        out_dir: Directory for output files.
        protein_id: Chain ID for the protein (default: 'A').
        ligand_type: 'smiles' or 'ccd' (default: 'smiles').
        use_msa_server: Use remote MSA server for alignments.
        recycling_steps: Number of recycling iterations.
        diffusion_samples: Number of diffusion samples to generate.
        device: Compute device ('cuda', 'cpu', or None for auto).

    Returns:
        Dictionary with:
            - affinity_pred_value: Predicted log10(IC50) in micromolar
            - affinity_probability_binary: Binder probability (0-1)
            - structure_path: Path to predicted complex structure
            - confidence: Confidence metrics dict
    """
    import boltz

    os.makedirs(out_dir, exist_ok=True)

    # Auto-detect device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build and write YAML input
    yaml_data = build_affinity_yaml(protein_id, protein_sequence, ligand_spec, ligand_type)

    tmp_yaml = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, prefix="boltz_affinity_"
    )
    tmp_yaml.close()
    write_yaml(yaml_data, tmp_yaml.name)

    print(f"Running Boltz-2 binding affinity prediction on {device}...")
    print(f"  Protein:  {protein_id} ({len(protein_sequence)} residues)")
    print(f"  Ligand:   {ligand_spec} ({ligand_type})")
    print(f"  Output:   {out_dir}")

    try:
        # Run Boltz-2 prediction with affinity
        boltz.predict(
            tmp_yaml.name,
            out_dir=out_dir,
            use_msa_server=use_msa_server,
            recycling_steps=recycling_steps,
            diffusion_samples=diffusion_samples,
            accelerator="gpu" if device.startswith("cuda") else "cpu",
        )
    finally:
        try:
            os.unlink(tmp_yaml.name)
        except OSError:
            pass

    # Collect results
    results = _collect_affinity_results(out_dir)
    return results


def _collect_affinity_results(out_dir: str) -> dict:
    """
    Collect binding affinity prediction results from Boltz output.

    Args:
        out_dir: The Boltz output directory.

    Returns:
        Dictionary with affinity predictions and structure paths.
    """
    results = {
        "affinity_pred_value": None,
        "affinity_probability_binary": None,
        "structure_path": None,
        "confidence": {},
    }

    out_path = Path(out_dir)

    # Find structure file
    for cif_file in sorted(out_path.rglob("*.cif")):
        results["structure_path"] = str(cif_file)
        break

    # Find and parse affinity results
    for affinity_file in sorted(out_path.rglob("affinity_*.json")):
        try:
            with open(affinity_file) as fh:
                affinity_data = json.load(fh)
            if "affinity_pred_value" in affinity_data:
                results["affinity_pred_value"] = affinity_data["affinity_pred_value"]
            if "affinity_probability_binary" in affinity_data:
                results["affinity_probability_binary"] = affinity_data["affinity_probability_binary"]
            break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: could not parse affinity file {affinity_file}: {e}")

    # Find and parse confidence metrics
    for conf_file in sorted(out_path.rglob("confidence_*.json")):
        try:
            with open(conf_file) as fh:
                conf_data = json.load(fh)
            results["confidence"] = conf_data
            if "plddt" in conf_data:
                if isinstance(conf_data["plddt"], list):
                    results["confidence"]["mean_plddt"] = (
                        sum(conf_data["plddt"]) / len(conf_data["plddt"])
                    )
                else:
                    results["confidence"]["mean_plddt"] = conf_data["plddt"]
            break
        except (json.JSONDecodeError, KeyError):
            pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict protein-ligand binding affinity with Boltz-2. "
            "Jointly predicts complex structure and binding affinity."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python binding_affinity.py --protein protein.fasta --ligand "CCO" --output ./results\n'
            '  python binding_affinity.py --protein_seq MVLSPAD... --ligand ATP --ligand_type ccd\n'
            '  python binding_affinity.py --protein protein.fasta --ligand "CCO" --known_affinity -6.5\n'
        ),
    )

    # Protein input (mutually exclusive: file or sequence)
    protein_group = parser.add_mutually_exclusive_group(required=True)
    protein_group.add_argument(
        "--protein",
        type=str,
        help="Path to protein FASTA file.",
    )
    protein_group.add_argument(
        "--protein_seq",
        type=str,
        help="Protein amino acid sequence (passed directly).",
    )

    # Ligand input
    parser.add_argument(
        "--ligand",
        type=str,
        nargs="+",
        required=True,
        help="Ligand SMILES string(s) or CCD code(s). Multiple ligands for batch screening.",
    )
    parser.add_argument(
        "--ligand_type",
        type=str,
        default="smiles",
        choices=["smiles", "ccd"],
        help="Type of ligand specification (default: smiles).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="./boltz_affinity_results",
        help="Output directory for predictions (default: ./boltz_affinity_results).",
    )

    # Optional known affinity for comparison
    parser.add_argument(
        "--known_affinity",
        type=float,
        default=None,
        help=(
            "Known experimental binding affinity as log10(IC50) in micromolar. "
            "If provided, predicted and experimental values will be compared."
        ),
    )

    # Prediction parameters
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

    # ---- Resolve protein sequence -----------------------------------------
    if args.protein:
        protein_path = Path(args.protein)
        if not protein_path.exists():
            print(f"Error: protein file not found: {args.protein}", file=sys.stderr)
            return 1
        records = parse_fasta(args.protein)
        if not records:
            print(f"Error: no sequences found in {args.protein}", file=sys.stderr)
            return 1
        protein_id = records[0][0].split()[0][:10]
        protein_sequence = records[0][1]
        if len(records) > 1:
            print(
                f"Warning: FASTA contains {len(records)} sequences. "
                f"Using first sequence: {protein_id}"
            )
    else:
        protein_id = "A"
        protein_sequence = args.protein_seq

    print("=" * 60)
    print("Boltz-2 Binding Affinity Prediction")
    print("=" * 60)
    print(f"Protein: {protein_id} ({len(protein_sequence)} residues)")
    print(f"Ligands: {len(args.ligand)} compound(s)")
    if args.known_affinity is not None:
        print(f"Known affinity: {args.known_affinity:.2f} log10(IC50 uM)")
    print()

    # ---- Run predictions for each ligand ----------------------------------
    all_results = []

    for i, ligand in enumerate(args.ligand):
        ligand_label = ligand if len(ligand) <= 50 else ligand[:47] + "..."
        print(f"\n--- Ligand {i + 1}/{len(args.ligand)}: {ligand_label} ---")

        # Create per-ligand output directory for batch runs
        if len(args.ligand) > 1:
            ligand_out = os.path.join(args.output, f"ligand_{i:04d}")
        else:
            ligand_out = args.output

        results = predict_binding_affinity(
            protein_sequence=protein_sequence,
            ligand_spec=ligand,
            out_dir=ligand_out,
            protein_id=protein_id,
            ligand_type=args.ligand_type,
            use_msa_server=args.use_msa_server,
            recycling_steps=args.recycling_steps,
            diffusion_samples=args.diffusion_samples,
            device=args.device,
        )
        results["ligand"] = ligand
        all_results.append(results)

    # ---- Print summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("Binding Affinity Results Summary")
    print("=" * 60)
    print(f"{'Ligand':<50} {'Affinity':>10} {'P(binder)':>10}")
    print("-" * 72)

    for res in all_results:
        ligand_label = res["ligand"]
        if len(ligand_label) > 48:
            ligand_label = ligand_label[:45] + "..."

        aff = res.get("affinity_pred_value")
        prob = res.get("affinity_probability_binary")

        aff_str = f"{aff:.2f}" if aff is not None else "N/A"
        prob_str = f"{prob:.3f}" if prob is not None else "N/A"

        print(f"{ligand_label:<50} {aff_str:>10} {prob_str:>10}")

    # Compare with known affinity if provided
    if args.known_affinity is not None and len(all_results) == 1:
        pred = all_results[0].get("affinity_pred_value")
        if pred is not None:
            delta = pred - args.known_affinity
            print(f"\nExperimental affinity:  {args.known_affinity:.2f} log10(IC50 uM)")
            print(f"Predicted affinity:    {pred:.2f} log10(IC50 uM)")
            print(f"Difference:            {delta:+.2f} log units")
            if abs(delta) <= 1.0:
                print("Assessment: Prediction within 1 log unit of experimental (good agreement).")
            elif abs(delta) <= 2.0:
                print("Assessment: Prediction within 2 log units (moderate agreement).")
            else:
                print("Assessment: Prediction differs by >2 log units (poor agreement).")

    # Confidence summary
    for res in all_results:
        mean_plddt = res.get("confidence", {}).get("mean_plddt")
        if mean_plddt is not None:
            quality = (
                "High"
                if mean_plddt > 0.7
                else "Medium"
                if mean_plddt > 0.5
                else "Low"
            )
            print(f"\nStructural confidence (mean pLDDT): {mean_plddt:.3f} ({quality})")
            break

    # Save consolidated results JSON
    summary_path = os.path.join(args.output, "affinity_summary.json")
    os.makedirs(args.output, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nFull results saved to: {summary_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

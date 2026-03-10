#!/usr/bin/env python3
"""Design a protein binder for a target using RFdiffusion + ProteinMPNN.

Standard protein design pipeline: RFdiffusion -> ProteinMPNN -> AF2/ESMFold validation.
Experimentally validated in multiple wet lab studies.

Pipeline:
  1. RFdiffusion generates binder backbones targeting specified hotspot residues
  2. ProteinMPNN designs sequences for each generated backbone
  3. (Optional) ESMFold predicts structure of designed sequences for validation

Requirements:
  - RFdiffusion installed and on PATH (or set RFDIFFUSION_DIR env var)
  - ProteinMPNN installed and on PATH (or set PROTEINMPNN_DIR env var)
  - (Optional) ESM for ESMFold validation (pip install esm)
"""

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_tool_dir(env_var: str, repo_name: str) -> Path:
    """Locate a tool directory from environment variable or common locations."""
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    # Check common locations
    candidates = [
        Path.home() / repo_name,
        Path.home() / "software" / repo_name,
        Path("/opt") / repo_name,
        Path.cwd() / repo_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {repo_name}. Set the {env_var} environment variable "
        f"to the installation directory."
    )


def run_rfdiffusion(
    target_pdb: str,
    hotspot_residues: list[str],
    binder_length: int,
    num_designs: int,
    output_dir: str,
    rfdiffusion_dir: Path,
    ckpt_override: str | None = None,
    noise_scale_ca: float = 1.0,
    noise_scale_frame: float = 1.0,
    diffusion_steps: int = 50,
) -> list[str]:
    """Run RFdiffusion to generate binder backbones.

    Args:
        target_pdb: Path to the target protein PDB file.
        hotspot_residues: List of hotspot residues (e.g., ["A30", "A33", "A34"]).
        binder_length: Length of the binder chain to generate.
        num_designs: Number of binder backbones to generate.
        output_dir: Directory for output PDB files.
        rfdiffusion_dir: Path to the RFdiffusion installation.
        ckpt_override: Optional path to a specific model checkpoint.
        noise_scale_ca: Translation noise scale (lower = closer to input).
        noise_scale_frame: Rotation noise scale (lower = closer to input).
        diffusion_steps: Number of diffusion timesteps.

    Returns:
        List of paths to generated backbone PDB files.
    """
    output_prefix = os.path.join(output_dir, "rfdiff_binder")
    inference_script = rfdiffusion_dir / "scripts" / "run_inference.py"

    if not inference_script.exists():
        raise FileNotFoundError(f"RFdiffusion inference script not found at {inference_script}")

    # Build the contig string: keep entire target chain, generate binder of specified length
    # Format: [target_chain/0 binder_length-binder_length]
    # The target chain is inferred from the PDB; /0 denotes a chain break
    contig_str = f"[B1-1000/0 {binder_length}-{binder_length}]"

    # Format hotspot residues for RFdiffusion
    hotspot_str = "[" + ",".join(hotspot_residues) + "]"

    cmd = [
        sys.executable, str(inference_script),
        f"contigmap.contigs={contig_str}",
        f"inference.input_pdb={target_pdb}",
        f"ppi.hotspot_res={hotspot_str}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
        f"diffuser.T={diffusion_steps}",
        f"denoiser.noise_scale_ca={noise_scale_ca}",
        f"denoiser.noise_scale_frame={noise_scale_frame}",
    ]

    if ckpt_override:
        cmd.append(f"inference.ckpt_override_path={ckpt_override}")
    else:
        # Use the Complex model for binder design by default
        complex_ckpt = rfdiffusion_dir / "models" / "Complex_base_ckpt.pt"
        if complex_ckpt.exists():
            cmd.append(f"inference.ckpt_override_path={complex_ckpt}")

    logger.info("Running RFdiffusion for binder design...")
    logger.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("RFdiffusion stderr:\n%s", result.stderr)
        raise RuntimeError(f"RFdiffusion failed with return code {result.returncode}")

    logger.info("RFdiffusion stdout:\n%s", result.stdout)

    # Collect generated PDB files
    generated_pdbs = sorted(glob.glob(f"{output_prefix}_*.pdb"))
    if not generated_pdbs:
        raise RuntimeError(f"No PDB files generated at {output_prefix}_*.pdb")

    logger.info("Generated %d binder backbones.", len(generated_pdbs))
    return generated_pdbs


def run_proteinmpnn(
    pdb_paths: list[str],
    output_dir: str,
    proteinmpnn_dir: Path,
    num_seqs_per_target: int = 8,
    sampling_temp: float = 0.1,
    seed: int = 42,
    use_soluble_model: bool = False,
    batch_size: int = 1,
) -> dict[str, list[str]]:
    """Run ProteinMPNN to design sequences for backbone structures.

    Args:
        pdb_paths: List of PDB files to design sequences for.
        output_dir: Output directory for designed sequences.
        proteinmpnn_dir: Path to the ProteinMPNN installation.
        num_seqs_per_target: Number of sequences to design per backbone.
        sampling_temp: Sampling temperature (0.1-0.3 recommended).
        seed: Random seed.
        use_soluble_model: Whether to use the soluble-protein-trained model.
        batch_size: Batch size for sequence design.

    Returns:
        Dictionary mapping PDB path to list of designed sequences (FASTA strings).
    """
    parse_script = proteinmpnn_dir / "helper_scripts" / "parse_multiple_chains.py"
    mpnn_script = proteinmpnn_dir / "protein_mpnn_run.py"

    if not mpnn_script.exists():
        raise FileNotFoundError(f"ProteinMPNN script not found at {mpnn_script}")

    mpnn_output_dir = os.path.join(output_dir, "proteinmpnn_output")
    os.makedirs(mpnn_output_dir, exist_ok=True)

    all_sequences: dict[str, list[str]] = {}

    for pdb_path in pdb_paths:
        pdb_name = Path(pdb_path).stem
        logger.info("Designing sequences for %s...", pdb_name)

        # Step 1: Parse the PDB chains
        parsed_jsonl = os.path.join(mpnn_output_dir, f"{pdb_name}_parsed.jsonl")

        parse_cmd = [
            sys.executable, str(parse_script),
            "--input_path", pdb_path,
            "--output_path", parsed_jsonl,
        ]

        result = subprocess.run(parse_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Chain parsing failed for %s: %s", pdb_path, result.stderr)
            continue

        # Step 2: Run ProteinMPNN
        seq_output_dir = os.path.join(mpnn_output_dir, pdb_name)
        os.makedirs(seq_output_dir, exist_ok=True)

        mpnn_cmd = [
            sys.executable, str(mpnn_script),
            "--jsonl_path", parsed_jsonl,
            "--out_folder", seq_output_dir,
            "--num_seq_per_target", str(num_seqs_per_target),
            "--sampling_temp", str(sampling_temp),
            "--seed", str(seed),
            "--batch_size", str(batch_size),
        ]

        if use_soluble_model:
            mpnn_cmd.append("--use_soluble_model")
            mpnn_cmd.extend(["--path_to_model_weights",
                             str(proteinmpnn_dir / "soluble_model_weights")])
        else:
            mpnn_cmd.extend(["--path_to_model_weights",
                             str(proteinmpnn_dir / "vanilla_model_weights")])

        result = subprocess.run(mpnn_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ProteinMPNN failed for %s: %s", pdb_path, result.stderr)
            continue

        logger.info("ProteinMPNN stdout:\n%s", result.stdout)

        # Collect designed sequences from FASTA output
        fasta_dir = os.path.join(seq_output_dir, "seqs")
        sequences = []
        fasta_files = sorted(glob.glob(os.path.join(fasta_dir, "*.fa")))
        for fasta_file in fasta_files:
            with open(fasta_file) as f:
                lines = f.readlines()
            # Parse FASTA: every other line starting from line 1 is a sequence
            for i, line in enumerate(lines):
                if not line.startswith(">"):
                    sequences.append(line.strip())

        all_sequences[pdb_path] = sequences
        logger.info("Designed %d sequences for %s.", len(sequences), pdb_name)

    return all_sequences


def validate_with_esmfold(
    sequences: dict[str, list[str]],
    output_dir: str,
    plddt_threshold: float = 70.0,
) -> dict[str, list[dict]]:
    """Validate designed sequences using ESMFold structure prediction.

    Checks that designed sequences are predicted to fold into high-confidence
    structures (pLDDT > threshold).

    Args:
        sequences: Dictionary mapping backbone PDB to list of designed sequences.
        output_dir: Output directory for predicted structures.
        plddt_threshold: Minimum mean pLDDT to consider a design successful.

    Returns:
        Dictionary mapping backbone PDB to list of validation results.
    """
    try:
        import torch
        import esm
    except ImportError:
        logger.warning(
            "ESM not installed. Skipping ESMFold validation. "
            "Install with: pip install esm"
        )
        return {}

    logger.info("Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Using GPU for ESMFold.")
    else:
        logger.info("Using CPU for ESMFold (this will be slow).")

    validation_dir = os.path.join(output_dir, "esmfold_validation")
    os.makedirs(validation_dir, exist_ok=True)

    results: dict[str, list[dict]] = {}

    for pdb_path, seqs in sequences.items():
        pdb_name = Path(pdb_path).stem
        pdb_results = []

        for i, seq in enumerate(seqs):
            logger.info("Validating %s sequence %d/%d...", pdb_name, i + 1, len(seqs))

            try:
                with torch.no_grad():
                    output = model.infer_pdb(seq)

                # Extract pLDDT from the B-factor column of the output PDB
                plddt_values = []
                for line in output.split("\n"):
                    if line.startswith("ATOM"):
                        try:
                            bfactor = float(line[60:66].strip())
                            plddt_values.append(bfactor)
                        except (ValueError, IndexError):
                            continue

                mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0.0
                passed = mean_plddt >= plddt_threshold

                # Save predicted structure
                pred_pdb_path = os.path.join(
                    validation_dir, f"{pdb_name}_seq{i:03d}_plddt{mean_plddt:.1f}.pdb"
                )
                with open(pred_pdb_path, "w") as f:
                    f.write(output)

                result = {
                    "sequence_index": i,
                    "sequence": seq,
                    "mean_plddt": mean_plddt,
                    "passed_threshold": passed,
                    "predicted_pdb": pred_pdb_path,
                }
                pdb_results.append(result)

                status = "PASS" if passed else "FAIL"
                logger.info(
                    "  %s: mean pLDDT = %.1f (%s)", pdb_name, mean_plddt, status
                )

            except Exception as e:
                logger.error("ESMFold failed for %s seq %d: %s", pdb_name, i, e)
                pdb_results.append({
                    "sequence_index": i,
                    "sequence": seq,
                    "error": str(e),
                })

        results[pdb_path] = pdb_results

    return results


def write_summary(
    backbone_pdbs: list[str],
    sequences: dict[str, list[str]],
    validation_results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Write a summary of the design campaign."""
    summary = {
        "num_backbones": len(backbone_pdbs),
        "backbone_pdbs": backbone_pdbs,
        "designs": {},
    }

    total_designed = 0
    total_passed = 0

    for pdb_path in backbone_pdbs:
        pdb_name = Path(pdb_path).stem
        seqs = sequences.get(pdb_path, [])
        val = validation_results.get(pdb_path, [])

        num_passed = sum(1 for v in val if v.get("passed_threshold", False))
        total_designed += len(seqs)
        total_passed += num_passed

        summary["designs"][pdb_name] = {
            "backbone_pdb": pdb_path,
            "num_sequences": len(seqs),
            "num_validated": len(val),
            "num_passed": num_passed,
            "sequences": seqs,
            "validation": val,
        }

    summary["total_sequences_designed"] = total_designed
    summary["total_passed_validation"] = total_passed

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary written to %s", output_path)
    logger.info(
        "Total: %d backbones, %d sequences designed, %d passed validation.",
        len(backbone_pdbs), total_designed, total_passed,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Design protein binders using RFdiffusion + ProteinMPNN pipeline. "
            "Generates binder backbones with RFdiffusion, designs sequences with "
            "ProteinMPNN, and optionally validates with ESMFold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic binder design\n"
            "  python design_binder.py --target-pdb target.pdb \\\n"
            "      --hotspot-residues A30 A33 A34 --num-designs 10\n"
            "\n"
            "  # With ESMFold validation and custom binder length\n"
            "  python design_binder.py --target-pdb target.pdb \\\n"
            "      --hotspot-residues A30 A33 A34 --num-designs 50 \\\n"
            "      --binder-length 80 --validate --output results/\n"
        ),
    )

    # Required arguments
    parser.add_argument(
        "--target-pdb", required=True,
        help="Path to the target protein PDB file.",
    )
    parser.add_argument(
        "--hotspot-residues", nargs="+", required=True,
        help="Target hotspot residues for binding (e.g., A30 A33 A34).",
    )

    # Design parameters
    parser.add_argument(
        "--num-designs", type=int, default=10,
        help="Number of binder backbones to generate (default: 10).",
    )
    parser.add_argument(
        "--binder-length", type=int, default=100,
        help="Length of the binder protein in residues (default: 100).",
    )
    parser.add_argument(
        "--seqs-per-backbone", type=int, default=8,
        help="Number of sequences to design per backbone (default: 8).",
    )
    parser.add_argument(
        "--sampling-temp", type=float, default=0.1,
        help="ProteinMPNN sampling temperature; 0.1-0.3 recommended (default: 0.1).",
    )
    parser.add_argument(
        "--use-soluble-model", action="store_true",
        help="Use ProteinMPNN's soluble-protein-trained model weights.",
    )

    # RFdiffusion parameters
    parser.add_argument(
        "--ckpt-override", default=None,
        help="Path to a specific RFdiffusion model checkpoint.",
    )
    parser.add_argument(
        "--noise-scale-ca", type=float, default=1.0,
        help="RFdiffusion translation noise scale (default: 1.0).",
    )
    parser.add_argument(
        "--noise-scale-frame", type=float, default=1.0,
        help="RFdiffusion rotation noise scale (default: 1.0).",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=50,
        help="Number of diffusion timesteps (default: 50).",
    )

    # Validation
    parser.add_argument(
        "--validate", action="store_true",
        help="Run ESMFold validation on designed sequences.",
    )
    parser.add_argument(
        "--plddt-threshold", type=float, default=70.0,
        help="Minimum mean pLDDT for validation pass (default: 70.0).",
    )

    # Output
    parser.add_argument(
        "--output", default="binder_design_output",
        help="Output directory (default: binder_design_output).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # Validate inputs
    target_pdb = os.path.abspath(args.target_pdb)
    if not os.path.isfile(target_pdb):
        parser.error(f"Target PDB file not found: {target_pdb}")

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Locate tool directories
    rfdiffusion_dir = find_tool_dir("RFDIFFUSION_DIR", "RFdiffusion")
    proteinmpnn_dir = find_tool_dir("PROTEINMPNN_DIR", "ProteinMPNN")

    logger.info("RFdiffusion directory: %s", rfdiffusion_dir)
    logger.info("ProteinMPNN directory: %s", proteinmpnn_dir)
    logger.info("Target PDB: %s", target_pdb)
    logger.info("Hotspot residues: %s", args.hotspot_residues)
    logger.info("Number of designs: %d", args.num_designs)
    logger.info("Binder length: %d", args.binder_length)
    logger.info("Output directory: %s", output_dir)

    # Step 1: Generate binder backbones with RFdiffusion
    logger.info("=" * 60)
    logger.info("STEP 1: RFdiffusion - Generating binder backbones")
    logger.info("=" * 60)

    backbone_pdbs = run_rfdiffusion(
        target_pdb=target_pdb,
        hotspot_residues=args.hotspot_residues,
        binder_length=args.binder_length,
        num_designs=args.num_designs,
        output_dir=output_dir,
        rfdiffusion_dir=rfdiffusion_dir,
        ckpt_override=args.ckpt_override,
        noise_scale_ca=args.noise_scale_ca,
        noise_scale_frame=args.noise_scale_frame,
        diffusion_steps=args.diffusion_steps,
    )

    # Step 2: Design sequences with ProteinMPNN
    logger.info("=" * 60)
    logger.info("STEP 2: ProteinMPNN - Designing sequences")
    logger.info("=" * 60)

    sequences = run_proteinmpnn(
        pdb_paths=backbone_pdbs,
        output_dir=output_dir,
        proteinmpnn_dir=proteinmpnn_dir,
        num_seqs_per_target=args.seqs_per_backbone,
        sampling_temp=args.sampling_temp,
        seed=args.seed,
        use_soluble_model=args.use_soluble_model,
    )

    # Step 3: Optional ESMFold validation
    validation_results: dict[str, list[dict]] = {}
    if args.validate:
        logger.info("=" * 60)
        logger.info("STEP 3: ESMFold - Validating designed sequences")
        logger.info("=" * 60)

        validation_results = validate_with_esmfold(
            sequences=sequences,
            output_dir=output_dir,
            plddt_threshold=args.plddt_threshold,
        )
    else:
        logger.info("Skipping ESMFold validation (use --validate to enable).")

    # Write summary
    summary_path = os.path.join(output_dir, "design_summary.json")
    write_summary(backbone_pdbs, sequences, validation_results, summary_path)

    logger.info("Binder design pipeline complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()

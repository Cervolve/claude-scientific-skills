#!/usr/bin/env python3
"""Scaffold a functional motif into a new protein using RFdiffusion + ProteinMPNN.

Standard protein design pipeline: RFdiffusion -> ProteinMPNN -> AF2/ESMFold validation.
Experimentally validated in multiple wet lab studies.

This script takes a PDB containing a functional motif (e.g., enzyme active site,
binding epitope, catalytic residues) and generates new protein scaffolds that
incorporate the motif while designing flanking structure around it.

Pipeline:
  1. RFdiffusion generates scaffolds that embed the motif with new flanking structure
  2. ProteinMPNN designs sequences for the scaffolded backbones
  3. (Optional) ESMFold validates that designed sequences fold correctly

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
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_tool_dir(env_var: str, repo_name: str) -> Path:
    """Locate a tool directory from environment variable or common locations."""
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
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


def build_contig_string(
    motif_chain: str,
    motif_start: int,
    motif_end: int,
    flank_n_min: int,
    flank_n_max: int,
    flank_c_min: int,
    flank_c_max: int,
    extra_motif_segments: list[str] | None = None,
) -> str:
    """Build the RFdiffusion contig map string for motif scaffolding.

    The contig string tells RFdiffusion which residues to keep from the input PDB
    (the motif) and which regions to generate de novo (the flanking scaffold).

    Args:
        motif_chain: Chain ID of the motif in the input PDB (e.g., "A").
        motif_start: First residue number of the motif.
        motif_end: Last residue number of the motif.
        flank_n_min: Minimum length of N-terminal flanking region.
        flank_n_max: Maximum length of N-terminal flanking region.
        flank_c_min: Minimum length of C-terminal flanking region.
        flank_c_max: Maximum length of C-terminal flanking region.
        extra_motif_segments: Additional motif segments (e.g., ["A50-60", "A80-90"])
            for discontinuous motifs. Each segment will be separated by generated
            linker regions.

    Returns:
        Contig string for RFdiffusion (e.g., "[10-40/A10-25/10-40]").
    """
    parts = []

    # N-terminal flank
    parts.append(f"{flank_n_min}-{flank_n_max}")

    # Primary motif segment
    parts.append(f"{motif_chain}{motif_start}-{motif_end}")

    # Handle discontinuous motifs with linkers between segments
    if extra_motif_segments:
        for segment in extra_motif_segments:
            # Add a linker region between motif segments
            parts.append(f"{flank_n_min}-{flank_c_max}")
            parts.append(segment)

    # C-terminal flank
    parts.append(f"{flank_c_min}-{flank_c_max}")

    contig = "[" + "/".join(parts) + "]"
    return contig


def run_rfdiffusion_scaffold(
    motif_pdb: str,
    contig_string: str,
    num_designs: int,
    output_dir: str,
    rfdiffusion_dir: Path,
    ckpt_override: str | None = None,
    inpaint_seq_ranges: list[str] | None = None,
    diffusion_steps: int = 50,
    noise_scale_ca: float = 1.0,
    noise_scale_frame: float = 1.0,
) -> list[str]:
    """Run RFdiffusion for motif scaffolding.

    Args:
        motif_pdb: Path to the PDB file containing the motif.
        contig_string: RFdiffusion contig map string.
        num_designs: Number of scaffold designs to generate.
        output_dir: Directory for output PDB files.
        rfdiffusion_dir: Path to the RFdiffusion installation.
        ckpt_override: Optional path to a specific model checkpoint.
        inpaint_seq_ranges: Optional list of residue ranges for sequence inpainting
            (e.g., ["A1", "A30-40"]). These motif residues will have their
            sequence masked, allowing RFdiffusion to redesign them.
        diffusion_steps: Number of diffusion timesteps.
        noise_scale_ca: Translation noise scale.
        noise_scale_frame: Rotation noise scale.

    Returns:
        List of paths to generated scaffold PDB files.
    """
    output_prefix = os.path.join(output_dir, "scaffold")
    inference_script = rfdiffusion_dir / "scripts" / "run_inference.py"

    if not inference_script.exists():
        raise FileNotFoundError(f"RFdiffusion inference script not found at {inference_script}")

    cmd = [
        sys.executable, str(inference_script),
        f"contigmap.contigs={contig_string}",
        f"inference.input_pdb={motif_pdb}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
        f"diffuser.T={diffusion_steps}",
        f"denoiser.noise_scale_ca={noise_scale_ca}",
        f"denoiser.noise_scale_frame={noise_scale_frame}",
    ]

    if ckpt_override:
        cmd.append(f"inference.ckpt_override_path={ckpt_override}")

    if inpaint_seq_ranges:
        inpaint_str = "[" + "/".join(inpaint_seq_ranges) + "]"
        cmd.append(f"contigmap.inpaint_seq={inpaint_str}")

    logger.info("Running RFdiffusion for motif scaffolding...")
    logger.info("Contig string: %s", contig_string)
    logger.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("RFdiffusion stderr:\n%s", result.stderr)
        raise RuntimeError(f"RFdiffusion failed with return code {result.returncode}")

    logger.info("RFdiffusion stdout:\n%s", result.stdout)

    generated_pdbs = sorted(glob.glob(f"{output_prefix}_*.pdb"))
    if not generated_pdbs:
        raise RuntimeError(f"No PDB files generated at {output_prefix}_*.pdb")

    logger.info("Generated %d scaffold backbones.", len(generated_pdbs))
    return generated_pdbs


def run_proteinmpnn(
    pdb_paths: list[str],
    output_dir: str,
    proteinmpnn_dir: Path,
    num_seqs_per_target: int = 8,
    sampling_temp: float = 0.1,
    seed: int = 42,
    fixed_positions: dict | None = None,
    use_soluble_model: bool = False,
    batch_size: int = 1,
) -> dict[str, list[str]]:
    """Run ProteinMPNN to design sequences for scaffold backbones.

    Args:
        pdb_paths: List of scaffold PDB files.
        output_dir: Output directory for designed sequences.
        proteinmpnn_dir: Path to the ProteinMPNN installation.
        num_seqs_per_target: Number of sequences to design per backbone.
        sampling_temp: Sampling temperature (0.1-0.3 recommended).
        seed: Random seed.
        fixed_positions: Optional dict specifying residue positions to keep fixed.
            Format: {pdb_name: {chain: [positions]}}. Useful for preserving
            catalytic residues in the motif.
        use_soluble_model: Whether to use the soluble-protein-trained model.
        batch_size: Batch size for sequence design.

    Returns:
        Dictionary mapping PDB path to list of designed sequences.
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

        # Step 1: Parse PDB chains
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

        # Step 2: Write fixed positions JSONL if specified
        fixed_pos_jsonl = None
        if fixed_positions and pdb_name in fixed_positions:
            fixed_pos_jsonl = os.path.join(mpnn_output_dir, f"{pdb_name}_fixed.jsonl")
            fixed_data = {pdb_name: fixed_positions[pdb_name]}
            with open(fixed_pos_jsonl, "w") as f:
                f.write(json.dumps(fixed_data) + "\n")

        # Step 3: Run ProteinMPNN
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

        if fixed_pos_jsonl:
            mpnn_cmd.extend(["--fixed_positions_jsonl", fixed_pos_jsonl])

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
            for line in lines:
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
                logger.info("  %s seq %d: mean pLDDT = %.1f (%s)", pdb_name, i, mean_plddt, status)

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
    scaffold_pdbs: list[str],
    sequences: dict[str, list[str]],
    validation_results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Write a JSON summary of the scaffolding campaign."""
    summary = {
        "num_scaffolds": len(scaffold_pdbs),
        "scaffold_pdbs": scaffold_pdbs,
        "designs": {},
    }

    total_designed = 0
    total_passed = 0

    for pdb_path in scaffold_pdbs:
        pdb_name = Path(pdb_path).stem
        seqs = sequences.get(pdb_path, [])
        val = validation_results.get(pdb_path, [])

        num_passed = sum(1 for v in val if v.get("passed_threshold", False))
        total_designed += len(seqs)
        total_passed += num_passed

        summary["designs"][pdb_name] = {
            "scaffold_pdb": pdb_path,
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
        "Total: %d scaffolds, %d sequences designed, %d passed validation.",
        len(scaffold_pdbs), total_designed, total_passed,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold a functional motif into a new protein using "
            "RFdiffusion + ProteinMPNN. Takes a PDB containing the motif "
            "and generates new protein scaffolds that incorporate it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Scaffold residues 10-25 of chain A with flanking regions\n"
            "  python scaffold_motif.py --motif-pdb motif.pdb \\\n"
            "      --motif-chain A --motif-start 10 --motif-end 25 \\\n"
            "      --num-designs 10\n"
            "\n"
            "  # Discontinuous motif with sequence inpainting\n"
            "  python scaffold_motif.py --motif-pdb enzyme.pdb \\\n"
            "      --motif-chain A --motif-start 10 --motif-end 25 \\\n"
            "      --extra-motif-segments A50-60 A80-85 \\\n"
            "      --inpaint-seq A10-15 \\\n"
            "      --num-designs 20 --validate\n"
            "\n"
            "  # Use ActiveSite checkpoint for small motifs\n"
            "  python scaffold_motif.py --motif-pdb active_site.pdb \\\n"
            "      --motif-chain A --motif-start 100 --motif-end 110 \\\n"
            "      --ckpt-override models/ActiveSite_ckpt.pt \\\n"
            "      --num-designs 50\n"
        ),
    )

    # Motif specification
    parser.add_argument(
        "--motif-pdb", required=True,
        help="Path to the PDB file containing the functional motif.",
    )
    parser.add_argument(
        "--motif-chain", default="A",
        help="Chain ID of the motif in the input PDB (default: A).",
    )
    parser.add_argument(
        "--motif-start", type=int, required=True,
        help="First residue number of the motif to scaffold.",
    )
    parser.add_argument(
        "--motif-end", type=int, required=True,
        help="Last residue number of the motif to scaffold.",
    )
    parser.add_argument(
        "--extra-motif-segments", nargs="*", default=None,
        help=(
            "Additional motif segments for discontinuous motifs "
            "(e.g., A50-60 A80-90). Linker regions will be generated "
            "between segments."
        ),
    )

    # Scaffold geometry
    parser.add_argument(
        "--flank-n-min", type=int, default=10,
        help="Minimum N-terminal flanking region length (default: 10).",
    )
    parser.add_argument(
        "--flank-n-max", type=int, default=40,
        help="Maximum N-terminal flanking region length (default: 40).",
    )
    parser.add_argument(
        "--flank-c-min", type=int, default=10,
        help="Minimum C-terminal flanking region length (default: 10).",
    )
    parser.add_argument(
        "--flank-c-max", type=int, default=40,
        help="Maximum C-terminal flanking region length (default: 40).",
    )

    # Design parameters
    parser.add_argument(
        "--num-designs", type=int, default=10,
        help="Number of scaffold backbones to generate (default: 10).",
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
        help=(
            "Path to a specific RFdiffusion checkpoint. Use "
            "models/ActiveSite_ckpt.pt for small functional motifs."
        ),
    )
    parser.add_argument(
        "--inpaint-seq", nargs="*", default=None,
        help=(
            "Motif residue ranges for sequence inpainting (e.g., A10-15 A20). "
            "These residues will have their sequence masked during diffusion, "
            "allowing partial sequence redesign of the motif."
        ),
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=50,
        help="Number of diffusion timesteps (default: 50).",
    )
    parser.add_argument(
        "--noise-scale-ca", type=float, default=1.0,
        help="Translation noise scale (default: 1.0).",
    )
    parser.add_argument(
        "--noise-scale-frame", type=float, default=1.0,
        help="Rotation noise scale (default: 1.0).",
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
        "--output", default="scaffold_motif_output",
        help="Output directory (default: scaffold_motif_output).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # Validate inputs
    motif_pdb = os.path.abspath(args.motif_pdb)
    if not os.path.isfile(motif_pdb):
        parser.error(f"Motif PDB file not found: {motif_pdb}")

    if args.motif_start > args.motif_end:
        parser.error("--motif-start must be <= --motif-end")

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Locate tool directories
    rfdiffusion_dir = find_tool_dir("RFDIFFUSION_DIR", "RFdiffusion")
    proteinmpnn_dir = find_tool_dir("PROTEINMPNN_DIR", "ProteinMPNN")

    logger.info("RFdiffusion directory: %s", rfdiffusion_dir)
    logger.info("ProteinMPNN directory: %s", proteinmpnn_dir)
    logger.info("Motif PDB: %s", motif_pdb)
    logger.info(
        "Motif: chain %s, residues %d-%d",
        args.motif_chain, args.motif_start, args.motif_end,
    )
    if args.extra_motif_segments:
        logger.info("Extra motif segments: %s", args.extra_motif_segments)
    logger.info("Output directory: %s", output_dir)

    # Build contig string
    contig_string = build_contig_string(
        motif_chain=args.motif_chain,
        motif_start=args.motif_start,
        motif_end=args.motif_end,
        flank_n_min=args.flank_n_min,
        flank_n_max=args.flank_n_max,
        flank_c_min=args.flank_c_min,
        flank_c_max=args.flank_c_max,
        extra_motif_segments=args.extra_motif_segments,
    )

    # Step 1: Generate scaffolds with RFdiffusion
    logger.info("=" * 60)
    logger.info("STEP 1: RFdiffusion - Generating motif scaffolds")
    logger.info("=" * 60)

    scaffold_pdbs = run_rfdiffusion_scaffold(
        motif_pdb=motif_pdb,
        contig_string=contig_string,
        num_designs=args.num_designs,
        output_dir=output_dir,
        rfdiffusion_dir=rfdiffusion_dir,
        ckpt_override=args.ckpt_override,
        inpaint_seq_ranges=args.inpaint_seq,
        diffusion_steps=args.diffusion_steps,
        noise_scale_ca=args.noise_scale_ca,
        noise_scale_frame=args.noise_scale_frame,
    )

    # Step 2: Design sequences with ProteinMPNN
    logger.info("=" * 60)
    logger.info("STEP 2: ProteinMPNN - Designing sequences")
    logger.info("=" * 60)

    sequences = run_proteinmpnn(
        pdb_paths=scaffold_pdbs,
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
    summary_path = os.path.join(output_dir, "scaffold_summary.json")
    write_summary(scaffold_pdbs, sequences, validation_results, summary_path)

    logger.info("Motif scaffolding pipeline complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()

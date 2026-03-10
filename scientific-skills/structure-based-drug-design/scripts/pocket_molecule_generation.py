"""PocketXMol: SOTA on 11/13 molecular generation tasks (Cell 2026).
For virtual screening of existing libraries, use DrugCLIP instead.

Generate novel molecules for a protein binding pocket using PocketXMol's
atom-level generative foundation model. Supports structure-based drug design,
fragment linking/growing, and PROTAC design.

Requirements:
    - PocketXMol repository cloned (https://github.com/pengxingang/PocketXMol)
    - Conda environment created from environment.yml (CUDA 11.7)
    - Model weights downloaded from Zenodo (https://zenodo.org/records/17801271)

Environment variable:
    POCKETXMOL_DIR: Path to cloned PocketXMol repository root.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Configuration ────────────────────────────────────────────────────────────

POCKETXMOL_DIR = os.environ.get("POCKETXMOL_DIR", "")

# Task name -> config file mapping (relative to PocketXMol/configs/sample/examples/)
TASK_CONFIGS = {
    "sbdd": "sbdd.yml",
    "dock": "dock_smallmol.yml",
    "fragment-linking": "fragment_linking.yml",
    "fragment-growing": "fragment_growing.yml",
    "protac": "protac.yml",
    "peptide-design": "peptide_design.yml",
}


# ── Pocket extraction ────────────────────────────────────────────────────────

def extract_pocket_from_pdb(
    pdb_path: str,
    residue_ids: Optional[List[str]] = None,
    distance_cutoff: float = 10.0,
    ligand_chain: Optional[str] = None,
    ligand_resname: Optional[str] = None,
) -> Tuple[str, List[dict]]:
    """Extract binding pocket residues from a PDB file.

    If residue_ids are provided, extracts those specific residues. Otherwise,
    attempts to auto-detect the pocket by finding residues within distance_cutoff
    of a co-crystallized ligand (HETATM records).

    Args:
        pdb_path: Path to input PDB file.
        residue_ids: Optional list of residue identifiers as "chain:resnum"
                     (e.g., ["A:42", "A:45", "A:100"]).
        distance_cutoff: Angstrom cutoff for auto-detection (default 10.0).
        ligand_chain: Chain ID for ligand (auto-detection).
        ligand_resname: Residue name for ligand (auto-detection).

    Returns:
        Tuple of (pocket_pdb_string, list_of_residue_dicts).
    """
    try:
        from Bio.PDB import PDBParser, NeighborSearch, Selection
    except ImportError:
        raise ImportError(
            "BioPython is required for pocket extraction. "
            "Install with: pip install biopython"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    pocket_residues = []

    if residue_ids:
        # User-specified residues
        for res_id in residue_ids:
            parts = res_id.strip().split(":")
            if len(parts) == 2:
                chain_id, resnum = parts[0], int(parts[1])
            else:
                # Assume chain A if not specified
                chain_id, resnum = "A", int(parts[0])
            try:
                chain = model[chain_id]
                residue = chain[(" ", resnum, " ")]
                pocket_residues.append(residue)
            except KeyError:
                print(f"[PocketXMol] Warning: residue {chain_id}:{resnum} not found, skipping.")
    else:
        # Auto-detect: find HETATM ligand atoms, then nearby protein residues
        ligand_atoms = []
        protein_atoms = []
        standard_residues = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "HOH", "WAT",
        }

        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if ligand_resname and resname != ligand_resname:
                    if resname not in standard_residues:
                        continue
                if ligand_chain and chain.get_id() != ligand_chain:
                    pass

                if resname not in standard_residues:
                    # This is a ligand/HETATM
                    if ligand_resname is None or resname == ligand_resname:
                        for atom in residue:
                            ligand_atoms.append(atom)
                else:
                    for atom in residue:
                        protein_atoms.append(atom)

        if not ligand_atoms:
            raise ValueError(
                "No ligand (HETATM) found in PDB for auto-detection. "
                "Please specify --pocket-residues manually."
            )

        if not protein_atoms:
            raise ValueError("No protein atoms found in PDB.")

        # Find protein residues within cutoff of any ligand atom
        ns = NeighborSearch(protein_atoms)
        nearby_residues = set()
        for atom in ligand_atoms:
            neighbors = ns.search(atom.get_vector().get_array(), distance_cutoff, level="R")
            for res in neighbors:
                nearby_residues.add(res)

        pocket_residues = sorted(nearby_residues, key=lambda r: r.get_id()[1])

    if not pocket_residues:
        raise ValueError("No pocket residues identified. Check inputs.")

    # Build pocket PDB string
    from Bio.PDB import PDBIO
    from io import StringIO

    class PocketSelect:
        """Selection class to write only pocket residues."""
        def __init__(self, residues):
            self._residue_set = set(id(r) for r in residues)

        def accept_model(self, model):
            return True

        def accept_chain(self, chain):
            return True

        def accept_residue(self, residue):
            return id(residue) in self._residue_set

        def accept_atom(self, atom):
            return True

    io = PDBIO()
    io.set_structure(structure)
    pocket_stream = StringIO()
    io.save(pocket_stream, PocketSelect(pocket_residues))
    pocket_pdb = pocket_stream.getvalue()

    residue_info = []
    for res in pocket_residues:
        residue_info.append({
            "chain": res.get_parent().get_id(),
            "resnum": res.get_id()[1],
            "resname": res.get_resname().strip(),
        })

    print(f"[PocketXMol] Identified {len(pocket_residues)} pocket residues.")
    return pocket_pdb, residue_info


# ── Config generation ────────────────────────────────────────────────────────

def generate_task_config(
    task: str,
    protein_pdb: str,
    pocket_pdb_path: str,
    num_molecules: int,
    output_dir: str,
    batch_size: int = 100,
) -> str:
    """Generate a YAML config file for PocketXMol sampling.

    Args:
        task: Task name (sbdd, fragment-linking, protac, etc.).
        protein_pdb: Path to full protein PDB.
        pocket_pdb_path: Path to extracted pocket PDB.
        num_molecules: Number of molecules to generate.
        output_dir: Output directory.
        batch_size: Batch size for generation.

    Returns:
        Path to generated config YAML file.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    pxm_dir = Path(POCKETXMOL_DIR)
    if not pxm_dir.exists():
        raise FileNotFoundError(
            f"PocketXMol directory not found at {POCKETXMOL_DIR}. "
            f"Set POCKETXMOL_DIR environment variable."
        )

    # Load the base config for the task
    base_config_name = TASK_CONFIGS.get(task)
    if not base_config_name:
        raise ValueError(
            f"Unknown task: {task}. Available tasks: {list(TASK_CONFIGS.keys())}"
        )

    base_config_path = pxm_dir / "configs" / "sample" / "examples" / base_config_name
    if not base_config_path.exists():
        raise FileNotFoundError(
            f"Base config not found: {base_config_path}. "
            f"Ensure PocketXMol is properly set up."
        )

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with user settings
    config["data"] = config.get("data", {})
    config["data"]["protein_path"] = str(protein_pdb)
    config["data"]["pocket_path"] = str(pocket_pdb_path)

    config["sample"] = config.get("sample", {})
    config["sample"]["num_samples"] = num_molecules
    config["sample"]["batch_size"] = batch_size

    # Write custom config
    config_path = os.path.join(output_dir, f"config_{task}.yml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


# ── Main generation pipeline ─────────────────────────────────────────────────

def run_pocketxmol(
    protein_pdb: str,
    pocket_residues: Optional[List[str]] = None,
    num_molecules: int = 100,
    task: str = "sbdd",
    output_dir: str = "pocketxmol_output",
    device: str = "cuda:0",
    batch_size: int = 100,
    distance_cutoff: float = 10.0,
) -> Dict:
    """Run PocketXMol molecule generation for a protein pocket.

    Args:
        protein_pdb: Path to protein PDB file.
        pocket_residues: Optional list of pocket residues as "chain:resnum".
                         If None, auto-detects from co-crystallized ligand.
        num_molecules: Number of molecules to generate.
        task: Generation task (sbdd, fragment-linking, fragment-growing,
              protac, peptide-design, dock).
        output_dir: Directory for output files.
        device: CUDA device (e.g., "cuda:0").
        batch_size: Batch size for generation.
        distance_cutoff: Angstrom cutoff for pocket auto-detection.

    Returns:
        Dict with generation results including output paths and summary.
    """
    pxm_dir = Path(POCKETXMOL_DIR)
    if not pxm_dir.exists():
        raise FileNotFoundError(
            f"PocketXMol directory not found at {POCKETXMOL_DIR}. "
            f"Clone from https://github.com/pengxingang/PocketXMol and set "
            f"POCKETXMOL_DIR environment variable."
        )

    protein_pdb = str(Path(protein_pdb).resolve())
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract pocket
    print(f"[PocketXMol] Extracting pocket from {protein_pdb} ...")
    pocket_pdb_str, residue_info = extract_pocket_from_pdb(
        protein_pdb,
        residue_ids=pocket_residues,
        distance_cutoff=distance_cutoff,
    )

    pocket_pdb_path = os.path.join(output_dir, "pocket.pdb")
    with open(pocket_pdb_path, "w") as f:
        f.write(pocket_pdb_str)

    print(f"[PocketXMol] Pocket saved to {pocket_pdb_path}")
    print(f"[PocketXMol] Pocket residues: {len(residue_info)}")

    # Step 2: Generate task config
    print(f"[PocketXMol] Generating config for task '{task}' ...")
    config_path = generate_task_config(
        task=task,
        protein_pdb=protein_pdb,
        pocket_pdb_path=pocket_pdb_path,
        num_molecules=num_molecules,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    # Step 3: Run PocketXMol inference
    sample_script = pxm_dir / "scripts" / "sample_use.py"
    if not sample_script.exists():
        raise FileNotFoundError(f"PocketXMol sample script not found: {sample_script}")

    cmd = [
        sys.executable, str(sample_script),
        "--config_task", config_path,
        "--outdir", output_dir,
        "--device", device,
        "--batch_size", str(batch_size),
    ]

    print(f"[PocketXMol] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(pxm_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[PocketXMol] STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"PocketXMol exited with code {result.returncode}. "
            f"Check stderr output above."
        )

    print(f"[PocketXMol] Generation complete.")

    # Step 4: Collect results
    sdf_files = list(Path(output_dir).rglob("*.sdf"))
    pdb_files = list(Path(output_dir).rglob("*.pdb"))
    gen_info_files = list(Path(output_dir).rglob("gen_info.csv"))

    # Parse gen_info.csv for summary statistics
    molecules_summary = []
    if gen_info_files:
        with open(gen_info_files[0], "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {"index": row.get("index", "")}
                if "cfd_traj" in row:
                    entry["confidence"] = float(row["cfd_traj"])
                if "smiles" in row:
                    entry["smiles"] = row["smiles"]
                molecules_summary.append(entry)

    # Extract SMILES from SDF files if not in gen_info
    smiles_list = [m.get("smiles") for m in molecules_summary if m.get("smiles")]
    if not smiles_list and sdf_files:
        try:
            from rdkit import Chem
            for sdf_path in sdf_files:
                suppl = Chem.SDMolSupplier(str(sdf_path))
                for mol in suppl:
                    if mol is not None:
                        smiles_list.append(Chem.MolToSmiles(mol))
        except ImportError:
            print("[PocketXMol] RDKit not available; SMILES extraction skipped.")

    # Write SMILES output
    smiles_output_path = os.path.join(output_dir, "generated_molecules.smi")
    if smiles_list:
        with open(smiles_output_path, "w") as f:
            for i, smi in enumerate(smiles_list):
                f.write(f"{smi}\tgen_{i}\n")
        print(f"[PocketXMol] SMILES written to {smiles_output_path}")

    results = {
        "task": task,
        "protein_pdb": protein_pdb,
        "pocket_residues": len(residue_info),
        "pocket_pdb": pocket_pdb_path,
        "config": config_path,
        "num_requested": num_molecules,
        "num_generated": len(smiles_list) if smiles_list else len(sdf_files),
        "sdf_files": [str(p) for p in sdf_files],
        "smiles_file": smiles_output_path if smiles_list else None,
        "smiles": smiles_list[:20],  # First 20 for preview
        "gen_info": str(gen_info_files[0]) if gen_info_files else None,
        "output_dir": output_dir,
    }

    # Write summary JSON
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[PocketXMol] Summary written to {summary_path}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate molecules for a protein binding pocket using PocketXMol. "
            "PocketXMol: SOTA on 11/13 molecular generation tasks (Cell 2026). "
            "For virtual screening of existing libraries, use DrugCLIP instead."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect pocket from co-crystallized ligand
  python pocket_molecule_generation.py \\
    --protein-pdb 1a2b.pdb \\
    --num-molecules 100 \\
    --task sbdd \\
    --output results/

  # Specify pocket residues manually
  python pocket_molecule_generation.py \\
    --protein-pdb target.pdb \\
    --pocket-residues A:42 A:45 A:100 A:103 A:150 \\
    --num-molecules 200 \\
    --task sbdd \\
    --output results/

  # Fragment linking
  python pocket_molecule_generation.py \\
    --protein-pdb target.pdb \\
    --pocket-residues A:42 A:45 \\
    --task fragment-linking \\
    --output results/

  # PROTAC design
  python pocket_molecule_generation.py \\
    --protein-pdb target.pdb \\
    --task protac \\
    --output results/

Environment:
  POCKETXMOL_DIR   Path to cloned PocketXMol repository
        """,
    )

    parser.add_argument(
        "--protein-pdb",
        required=True,
        help="Path to protein PDB file.",
    )
    parser.add_argument(
        "--pocket-residues",
        nargs="+",
        default=None,
        help=(
            "Pocket residue identifiers as chain:resnum (e.g., A:42 A:45 A:100). "
            "If not specified, auto-detects from co-crystallized ligand."
        ),
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=100,
        help="Number of molecules to generate (default: 100).",
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS.keys()),
        default="sbdd",
        help=(
            "Generation task. Options: sbdd (structure-based drug design), "
            "dock (small-molecule docking), fragment-linking, fragment-growing, "
            "protac (PROTAC design), peptide-design. Default: sbdd."
        ),
    )
    parser.add_argument(
        "--output",
        default="pocketxmol_output",
        help="Output directory (default: pocketxmol_output).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (default: cuda:0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for generation; reduce if GPU OOM (default: 100).",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=10.0,
        help="Angstrom cutoff for pocket auto-detection (default: 10.0).",
    )

    args = parser.parse_args()

    if not POCKETXMOL_DIR:
        parser.error(
            "POCKETXMOL_DIR environment variable not set. "
            "Set it to the path of your cloned PocketXMol repository."
        )

    results = run_pocketxmol(
        protein_pdb=args.protein_pdb,
        pocket_residues=args.pocket_residues,
        num_molecules=args.num_molecules,
        task=args.task,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        distance_cutoff=args.distance_cutoff,
    )

    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"  Task:             {results['task']}")
    print(f"  Protein:          {results['protein_pdb']}")
    print(f"  Pocket residues:  {results['pocket_residues']}")
    print(f"  Requested:        {results['num_requested']}")
    print(f"  Generated:        {results['num_generated']}")
    print(f"  Output:           {results['output_dir']}")
    if results.get("smiles"):
        print(f"\n  First 5 SMILES:")
        for smi in results["smiles"][:5]:
            print(f"    {smi}")
    print()


if __name__ == "__main__":
    main()

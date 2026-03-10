#!/usr/bin/env python3
"""
AlphaGenome Gene Expression and Regulatory Track Prediction

Query AlphaGenome for gene expression (RNA-seq), chromatin accessibility (ATAC-seq),
CAGE, and other regulatory track predictions for a genomic region or gene. Produces
both tabular output and matplotlib visualizations of predicted tracks.

Reliability note: While AlphaGenome improves over Enformer (OR=3.0 on GTEx), it
retains limitations in personal gene expression prediction. Predictions represent
population-level regulatory potential from reference sequence, not individual-level
expression. See the AlphaGenome bioRxiv evaluation for details.

Usage examples:
    # Predict expression for a genomic region
    python gene_expression_prediction.py \\
        --region chr7:5500000-6548576 \\
        --ontology UBERON:0002048 \\
        --output expression_chr7.tsv

    # Predict for a gene by name (requires gget for coordinate lookup)
    python gene_expression_prediction.py \\
        --gene BRCA1 \\
        --ontology UBERON:0000310 \\
        --tracks RNA_SEQ ATAC_SEQ CAGE \\
        --plot brca1_tracks.png

    # Multi-tissue comparison
    python gene_expression_prediction.py \\
        --region chr22:35677410-36725986 \\
        --ontology UBERON:0001157 UBERON:0002048 UBERON:0000955 \\
        --plot multi_tissue.png

    # All available 1D tracks
    python gene_expression_prediction.py \\
        --region chr11:5200000-5350000 \\
        --tracks RNA_SEQ ATAC_SEQ CAGE DNASE_SEQ SPLICE_SITE \\
        --output hbb_locus.tsv --plot hbb_tracks.png
"""

import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Track configuration
# ---------------------------------------------------------------------------

AVAILABLE_TRACKS = ["RNA_SEQ", "ATAC_SEQ", "CAGE", "DNASE_SEQ", "CHIP_SEQ", "SPLICE_SITE"]
DEFAULT_TRACKS = ["RNA_SEQ", "ATAC_SEQ", "CAGE"]

TRACK_DISPLAY_NAMES = {
    "RNA_SEQ": "Gene Expression (RNA-seq)",
    "ATAC_SEQ": "Chromatin Accessibility (ATAC-seq)",
    "CAGE": "TSS Activity (CAGE)",
    "DNASE_SEQ": "Chromatin Accessibility (DNase-seq)",
    "CHIP_SEQ": "TF / Histone Binding (ChIP-seq)",
    "SPLICE_SITE": "Splice Site Usage",
}

TRACK_COLORS = {
    "RNA_SEQ": "#1f77b4",
    "ATAC_SEQ": "#ff7f0e",
    "CAGE": "#2ca02c",
    "DNASE_SEQ": "#d62728",
    "CHIP_SEQ": "#9467bd",
    "SPLICE_SITE": "#8c564b",
}


# ---------------------------------------------------------------------------
# Coordinate resolution
# ---------------------------------------------------------------------------

def parse_region(region_str: str) -> tuple[str, int, int]:
    """
    Parse a genomic region string.

    Args:
        region_str: Region in format 'chr7:5500000-6548576'.

    Returns:
        Tuple of (chromosome, start, end).
    """
    try:
        chrom, coords = region_str.split(":")
        start_str, end_str = coords.split("-")
        return chrom, int(start_str), int(end_str)
    except (ValueError, AttributeError):
        raise ValueError(
            f"Invalid region format '{region_str}'. "
            "Expected 'chr:start-end' (e.g., 'chr7:5500000-6548576')"
        )


def resolve_gene_coordinates(
    gene_name: str,
    flank: int = 100_000,
    species: str = "homo_sapiens",
) -> tuple[str, int, int]:
    """
    Look up genomic coordinates for a gene name using gget.

    Requires the gget package: pip install gget

    Args:
        gene_name: Gene symbol (e.g., 'BRCA1', 'TP53').
        flank: Extra bases to include upstream/downstream of gene body.
        species: Species name for Ensembl lookup.

    Returns:
        Tuple of (chromosome, start, end) with flanking region.
    """
    try:
        import gget
    except ImportError:
        raise RuntimeError(
            "The gget package is required for gene name lookup. "
            "Install with: pip install gget"
        )

    results = gget.search([gene_name], species=species, limit=1)
    if results is None or len(results) == 0:
        raise ValueError(f"Gene '{gene_name}' not found in {species}")

    row = results.iloc[0]
    chrom = row.get("chromosome_name", row.get("chrom", ""))
    if not str(chrom).startswith("chr"):
        chrom = f"chr{chrom}"
    start = int(row.get("start_position", row.get("start", 0)))
    end = int(row.get("end_position", row.get("end", 0)))

    # Add flanking region
    start = max(0, start - flank)
    end = end + flank

    print(f"  Gene '{gene_name}' resolved to {chrom}:{start}-{end}")
    return chrom, start, end


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_tracks(
    chromosome: str,
    start: int,
    end: int,
    track_names: list[str],
    ontology_terms: Optional[list[str]] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Query AlphaGenome for track predictions on a genomic region.

    Args:
        chromosome: Chromosome name (e.g., 'chr7').
        start: Start coordinate (0-based).
        end: End coordinate.
        track_names: List of track names to predict.
        ontology_terms: UBERON/CL ontology terms for tissue/cell context.
        api_key: API key (reads ALPHAGENOME_API_KEY env var if None).

    Returns:
        Dictionary mapping track names to numpy arrays of predictions,
        plus 'interval' key with the prediction interval metadata.
    """
    from alphagenome.data import genome
    from alphagenome.models import dna_client

    if api_key is None:
        api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AlphaGenome API key not found. Set ALPHAGENOME_API_KEY environment "
            "variable or pass --api-key argument."
        )

    model = dna_client.create(api_key)

    interval = genome.Interval(chromosome=chromosome, start=start, end=end)

    # Build requested output types
    output_types = []
    for name in track_names:
        output_types.append(getattr(dna_client.OutputType, name))

    predict_kwargs = dict(
        interval=interval,
        requested_outputs=output_types,
    )
    if ontology_terms:
        predict_kwargs["ontology_terms"] = ontology_terms

    # While AlphaGenome improves over Enformer (OR=3.0 on GTEx), it retains
    # limitations in personal gene expression prediction. These predictions
    # represent population-level regulatory potential from reference sequence.
    outputs = model.predict(**predict_kwargs)

    result = {"interval": interval}
    for track_name in track_names:
        track_attr = track_name.lower()
        track_data = getattr(outputs, track_attr, None)
        if track_data is not None:
            result[track_name] = np.array(track_data)
        else:
            print(f"  Warning: Track '{track_name}' not found in output.")
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_track_table(
    predictions: dict,
    chromosome: str,
    start: int,
    output_path: str,
) -> None:
    """
    Write predicted track values to a TSV file.

    Each row corresponds to a genomic position, with columns for each track.

    Args:
        predictions: Dict of track_name -> numpy array.
        chromosome: Chromosome name.
        start: Start coordinate of the interval.
        output_path: Path to output TSV file.
    """
    track_names = [k for k in predictions if k != "interval"]
    if not track_names:
        print("No tracks to write.")
        return

    # Determine length from first available track
    first_track = predictions[track_names[0]]
    n_positions = len(first_track)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["chromosome", "position"] + track_names)
        for i in range(n_positions):
            row = [chromosome, start + i]
            for tn in track_names:
                arr = predictions.get(tn)
                if arr is not None and i < len(arr):
                    row.append(f"{arr[i]:.8f}")
                else:
                    row.append("NA")
            writer.writerow(row)

    print(f"Track predictions written to: {output_path}")
    print(f"  Positions: {n_positions:,}")
    print(f"  Tracks: {', '.join(track_names)}")


def plot_tracks(
    predictions: dict,
    chromosome: str,
    start: int,
    end: int,
    ontology_labels: Optional[list[str]] = None,
    output_path: str = "tracks.png",
    gene_name: Optional[str] = None,
) -> None:
    """
    Plot predicted regulatory tracks using matplotlib.

    Creates a multi-panel figure with one subplot per track.

    Args:
        predictions: Dict of track_name -> numpy array.
        chromosome: Chromosome name.
        start: Start coordinate.
        end: End coordinate.
        ontology_labels: List of ontology term labels for title.
        output_path: Path to save the figure.
        gene_name: Optional gene name for title.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print(
            "Warning: matplotlib not available, skipping plot. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    track_names = [k for k in predictions if k != "interval"]
    if not track_names:
        print("No tracks to plot.")
        return

    n_tracks = len(track_names)
    fig, axes = plt.subplots(
        n_tracks, 1,
        figsize=(14, 3 * n_tracks),
        sharex=True,
        squeeze=False,
    )

    for idx, track_name in enumerate(track_names):
        ax = axes[idx, 0]
        data = predictions[track_name]
        n_positions = len(data)
        positions = np.linspace(start, end, n_positions)

        color = TRACK_COLORS.get(track_name, "#333333")
        display_name = TRACK_DISPLAY_NAMES.get(track_name, track_name)

        ax.fill_between(positions, data, alpha=0.4, color=color)
        ax.plot(positions, data, linewidth=0.5, color=color)
        ax.set_ylabel(display_name, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Format x-axis with Mbp labels
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.2f} Mb")
        )

    # Title
    title_parts = [f"{chromosome}:{start:,}-{end:,}"]
    if gene_name:
        title_parts.insert(0, gene_name)
    if ontology_labels:
        title_parts.append(f"[{', '.join(ontology_labels)}]")
    axes[0, 0].set_title(
        "AlphaGenome Predictions: " + " | ".join(title_parts),
        fontsize=11,
        fontweight="bold",
    )

    axes[-1, 0].set_xlabel(f"Genomic position ({chromosome})", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Track plot saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict gene expression and regulatory tracks using AlphaGenome. "
            "Note: While AlphaGenome improves over Enformer (OR=3.0 on GTEx), "
            "it retains limitations in personal gene expression prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --region chr7:5500000-6548576 --ontology UBERON:0002048\n"
            "  %(prog)s --gene BRCA1 --ontology UBERON:0000310 --plot brca1.png\n"
            "  %(prog)s --region chr11:5200000-5350000 --tracks RNA_SEQ ATAC_SEQ CAGE\n"
        ),
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--region",
        metavar="CHR:START-END",
        help="Genomic region (e.g., chr7:5500000-6548576)",
    )
    input_group.add_argument(
        "--gene",
        metavar="SYMBOL",
        help="Gene symbol (e.g., BRCA1). Requires gget for coordinate lookup.",
    )

    # Track selection
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=DEFAULT_TRACKS,
        choices=AVAILABLE_TRACKS,
        help=f"Prediction tracks to query (default: {' '.join(DEFAULT_TRACKS)})",
    )

    # Tissue/cell context
    parser.add_argument(
        "--ontology",
        nargs="+",
        metavar="TERM",
        help=(
            "UBERON or CL ontology terms for tissue/cell type context "
            "(e.g., UBERON:0002048 for lung). Multiple terms produce separate queries."
        ),
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output TSV file for track values (optional)",
    )
    parser.add_argument(
        "--plot",
        "-p",
        metavar="FILE",
        help="Output plot file path (e.g., tracks.png). Requires matplotlib.",
    )

    # Configuration
    parser.add_argument(
        "--flank",
        type=int,
        default=100_000,
        metavar="BP",
        help="Extra flanking bases when using --gene (default: 100000)",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="AlphaGenome API key (default: ALPHAGENOME_API_KEY env var)",
    )
    parser.add_argument(
        "--species",
        default="homo_sapiens",
        help="Species for gene lookup (default: homo_sapiens)",
    )

    args = parser.parse_args()

    if not args.output and not args.plot:
        print(
            "Warning: Neither --output nor --plot specified. "
            "Results will only be printed to stdout.",
            file=sys.stderr,
        )

    # Resolve coordinates
    gene_name = None
    if args.gene:
        gene_name = args.gene
        chromosome, start, end = resolve_gene_coordinates(
            args.gene, flank=args.flank, species=args.species
        )
    else:
        chromosome, start, end = parse_region(args.region)

    region_size = end - start
    print(f"AlphaGenome Gene Expression / Regulatory Track Prediction")
    print(f"  Region: {chromosome}:{start:,}-{end:,} ({region_size:,} bp)")
    print(f"  Tracks: {', '.join(args.tracks)}")
    print(f"  Ontology: {', '.join(args.ontology) if args.ontology else 'default'}")
    print()

    # Warn if region is very large
    if region_size > 1_048_576:
        print(
            f"  Warning: Region is {region_size:,} bp. AlphaGenome supports up to ~1 Mbp. "
            "Results may be truncated or the API may reject the request.",
            file=sys.stderr,
        )

    # Run prediction
    print("Querying AlphaGenome API...", flush=True)
    predictions = predict_tracks(
        chromosome=chromosome,
        start=start,
        end=end,
        track_names=args.tracks,
        ontology_terms=args.ontology,
        api_key=args.api_key,
    )

    # Summary statistics
    track_names = [k for k in predictions if k != "interval"]
    print(f"\nPrediction Summary:")
    for tn in track_names:
        data = predictions[tn]
        print(
            f"  {tn:15s}  "
            f"min={np.min(data):.6f}  max={np.max(data):.6f}  "
            f"mean={np.mean(data):.6f}  std={np.std(data):.6f}"
        )

    # Write table
    if args.output:
        write_track_table(predictions, chromosome, start, args.output)

    # Generate plot
    if args.plot:
        plot_tracks(
            predictions=predictions,
            chromosome=chromosome,
            start=start,
            end=end,
            ontology_labels=args.ontology,
            output_path=args.plot,
            gene_name=gene_name,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

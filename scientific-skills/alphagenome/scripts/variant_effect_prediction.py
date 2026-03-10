#!/usr/bin/env python3
"""
AlphaGenome Variant Effect Prediction

Predict functional effects of genetic variants using Google DeepMind's AlphaGenome
model. Processes variants from a VCF file or command-line list, queries the
AlphaGenome API for reference and alternate allele predictions across multiple
regulatory tracks, computes variant effect scores, and outputs a summary table.

AlphaGenome achieves SOTA on 25/26 variant effect prediction tasks (Nature 2026),
outperforming Enformer and other prior sequence-based models.

Usage examples:
    # Single variant from command line
    python variant_effect_prediction.py \\
        --variants "chr22:36201698:A:C" \\
        --tracks RNA_SEQ ATAC_SEQ \\
        --output results.tsv

    # Multiple variants
    python variant_effect_prediction.py \\
        --variants "chr22:36201698:A:C" "chr7:5530601:G:T" \\
        --ontology UBERON:0001157 \\
        --output results.tsv

    # From VCF file
    python variant_effect_prediction.py \\
        --vcf variants.vcf \\
        --tracks RNA_SEQ ATAC_SEQ CAGE SPLICE_SITE \\
        --output results.tsv

    # With custom flanking region size
    python variant_effect_prediction.py \\
        --vcf variants.vcf \\
        --flank 524288 \\
        --output results.tsv
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VariantRecord:
    """A genetic variant with genomic coordinates."""
    chromosome: str
    position: int
    reference_bases: str
    alternate_bases: str
    variant_id: Optional[str] = None

    def __str__(self) -> str:
        vid = f" ({self.variant_id})" if self.variant_id else ""
        return f"{self.chromosome}:{self.position}:{self.reference_bases}:{self.alternate_bases}{vid}"


def parse_variant_string(variant_str: str) -> VariantRecord:
    """
    Parse a variant string in the format chr:pos:ref:alt.

    Args:
        variant_str: Variant in format 'chr22:36201698:A:C'

    Returns:
        VariantRecord with parsed fields.

    Raises:
        ValueError: If the format is invalid.
    """
    parts = variant_str.strip().split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid variant format '{variant_str}'. "
            "Expected 'chr:pos:ref:alt' (e.g., 'chr22:36201698:A:C')"
        )
    chrom, pos_str, ref, alt = parts
    try:
        pos = int(pos_str)
    except ValueError:
        raise ValueError(f"Invalid position '{pos_str}' in variant '{variant_str}'")
    return VariantRecord(
        chromosome=chrom,
        position=pos,
        reference_bases=ref,
        alternate_bases=alt,
    )


def parse_vcf(vcf_path: str, max_variants: Optional[int] = None) -> list[VariantRecord]:
    """
    Parse variants from a VCF file.

    Reads standard VCF format (v4.x). Skips header lines starting with '#'.
    For multi-allelic sites, creates separate records for each ALT allele.

    Args:
        vcf_path: Path to the VCF file.
        max_variants: Maximum number of variants to read (None for all).

    Returns:
        List of VariantRecord objects.
    """
    variants = []
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            variant_id = fields[2] if fields[2] != "." else None
            ref = fields[3]
            alt_alleles = fields[4].split(",")

            for alt in alt_alleles:
                alt = alt.strip()
                if alt == "." or alt == "*":
                    continue
                variants.append(VariantRecord(
                    chromosome=chrom,
                    position=pos,
                    reference_bases=ref,
                    alternate_bases=alt,
                    variant_id=variant_id,
                ))
                if max_variants and len(variants) >= max_variants:
                    return variants
    return variants


# Mapping from string track names to OutputType enum values
TRACK_NAME_MAP = {
    "RNA_SEQ": "RNA_SEQ",
    "CAGE": "CAGE",
    "ATAC_SEQ": "ATAC_SEQ",
    "DNASE_SEQ": "DNASE_SEQ",
    "CHIP_SEQ": "CHIP_SEQ",
    "SPLICE_SITE": "SPLICE_SITE",
    "CONTACT_MAP": "CONTACT_MAP",
}

# Default tracks for variant effect scoring (contact maps excluded --
# they produce 2D outputs that require separate handling)
DEFAULT_TRACKS = ["RNA_SEQ", "ATAC_SEQ", "CAGE"]


def resolve_output_types(track_names: list[str]):
    """
    Convert track name strings to dna_client.OutputType enum values.

    Args:
        track_names: List of track name strings (e.g., ['RNA_SEQ', 'ATAC_SEQ']).

    Returns:
        List of dna_client.OutputType values.
    """
    from alphagenome.models import dna_client

    output_types = []
    for name in track_names:
        name_upper = name.upper()
        if name_upper not in TRACK_NAME_MAP:
            raise ValueError(
                f"Unknown track '{name}'. Available tracks: {list(TRACK_NAME_MAP.keys())}"
            )
        output_types.append(getattr(dna_client.OutputType, TRACK_NAME_MAP[name_upper]))
    return output_types


def compute_variant_effect_score(ref_track, alt_track) -> dict:
    """
    Compute variant effect scores comparing reference and alternate predictions.

    Computes multiple summary statistics:
    - max_abs_diff: Maximum absolute difference across the track
    - mean_abs_diff: Mean absolute difference
    - sum_abs_diff: Sum of absolute differences (total effect magnitude)
    - max_log2fc: Maximum absolute log2 fold change (where ref > 0)
    - correlation: Pearson correlation between ref and alt

    Args:
        ref_track: Reference allele prediction (numpy array or array-like).
        alt_track: Alternate allele prediction (numpy array or array-like).

    Returns:
        Dictionary of score names to values.
    """
    ref_vals = np.array(ref_track, dtype=np.float64).flatten()
    alt_vals = np.array(alt_track, dtype=np.float64).flatten()

    diff = alt_vals - ref_vals
    abs_diff = np.abs(diff)

    scores = {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "sum_abs_diff": float(np.sum(abs_diff)),
    }

    # Log2 fold change where reference signal is non-negligible
    # Add pseudocount to avoid division by zero
    pseudocount = 1e-6
    ref_nonzero = ref_vals + pseudocount
    alt_nonzero = alt_vals + pseudocount
    log2fc = np.log2(alt_nonzero / ref_nonzero)
    scores["max_abs_log2fc"] = float(np.max(np.abs(log2fc)))

    # Pearson correlation
    if np.std(ref_vals) > 0 and np.std(alt_vals) > 0:
        scores["pearson_r"] = float(np.corrcoef(ref_vals, alt_vals)[0, 1])
    else:
        scores["pearson_r"] = float("nan")

    return scores


def predict_variant_effects(
    variants: list[VariantRecord],
    track_names: list[str],
    ontology_terms: Optional[list[str]] = None,
    flank_size: int = 524_288,
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0,
) -> list[dict]:
    """
    Query AlphaGenome API for variant effect predictions.

    For each variant, constructs a genomic interval centered on the variant
    position, queries the model for reference and alternate predictions, and
    computes effect scores for each requested track.

    Args:
        variants: List of VariantRecord objects.
        track_names: List of track name strings to predict.
        ontology_terms: UBERON/CL ontology terms for tissue/cell context.
        flank_size: Bases to include on each side of variant (default 524288 = 2^19).
        api_key: AlphaGenome API key. If None, reads from ALPHAGENOME_API_KEY env var.
        rate_limit_delay: Seconds to wait between API calls (default 1.0).

    Returns:
        List of result dicts, one per variant, with scores for each track.
    """
    from alphagenome.data import genome
    from alphagenome.models import dna_client

    # Resolve API key
    if api_key is None:
        api_key = os.environ.get("ALPHAGENOME_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AlphaGenome API key not found. Set ALPHAGENOME_API_KEY environment "
            "variable or pass --api-key argument."
        )

    # Initialize client
    model = dna_client.create(api_key)

    # Resolve output types
    output_types = resolve_output_types(track_names)

    results = []
    total = len(variants)

    for i, var in enumerate(variants):
        print(f"  [{i + 1}/{total}] Processing {var}...", flush=True)

        # Build interval centered on variant
        center = var.position
        start = max(0, center - flank_size)
        end = center + flank_size

        interval = genome.Interval(
            chromosome=var.chromosome,
            start=start,
            end=end,
        )

        variant_obj = genome.Variant(
            chromosome=var.chromosome,
            position=var.position,
            reference_bases=var.reference_bases,
            alternate_bases=var.alternate_bases,
        )

        try:
            # Query AlphaGenome API
            # AlphaGenome SOTA on 25/26 variant effect tasks (Nature 2026),
            # outperforms Enformer
            predict_kwargs = dict(
                interval=interval,
                variant=variant_obj,
                requested_outputs=output_types,
            )
            if ontology_terms:
                predict_kwargs["ontology_terms"] = ontology_terms

            outputs = model.predict_variant(**predict_kwargs)

            # Compute effect scores for each track
            result = {
                "variant_id": var.variant_id or ".",
                "chromosome": var.chromosome,
                "position": var.position,
                "ref": var.reference_bases,
                "alt": var.alternate_bases,
            }

            for track_name in track_names:
                track_attr = track_name.lower()
                ref_track_data = getattr(outputs.reference, track_attr, None)
                alt_track_data = getattr(outputs.alternate, track_attr, None)

                if ref_track_data is None or alt_track_data is None:
                    print(f"    Warning: Track '{track_name}' not available in output, skipping.")
                    continue

                scores = compute_variant_effect_score(ref_track_data, alt_track_data)
                for score_name, score_val in scores.items():
                    result[f"{track_name}_{score_name}"] = score_val

            result["status"] = "success"
            results.append(result)

        except Exception as e:
            print(f"    Error processing {var}: {e}", file=sys.stderr)
            result = {
                "variant_id": var.variant_id or ".",
                "chromosome": var.chromosome,
                "position": var.position,
                "ref": var.reference_bases,
                "alt": var.alternate_bases,
                "status": f"error: {e}",
            }
            results.append(result)

        # Rate limiting between API calls
        if i < total - 1 and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

    return results


def write_results(results: list[dict], output_path: str) -> None:
    """
    Write variant effect prediction results to a TSV file.

    Args:
        results: List of result dictionaries.
        output_path: Path to output TSV file.
    """
    if not results:
        print("No results to write.")
        return

    # Collect all fieldnames across all results
    fieldnames = []
    seen = set()
    for r in results:
        for key in r.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults written to: {output_path}")
    print(f"  Variants processed: {len(results)}")
    n_success = sum(1 for r in results if r.get("status") == "success")
    n_error = len(results) - n_success
    print(f"  Successful: {n_success}")
    if n_error:
        print(f"  Errors: {n_error}")


def print_summary(results: list[dict], track_names: list[str]) -> None:
    """Print a human-readable summary of variant effect predictions."""
    print("\n" + "=" * 80)
    print("VARIANT EFFECT PREDICTION SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        print("No successful predictions.")
        return

    for r in successful:
        vid = r.get("variant_id", ".")
        pos_str = f"{r['chromosome']}:{r['position']}:{r['ref']}>{r['alt']}"
        label = f"{vid} ({pos_str})" if vid != "." else pos_str
        print(f"\n  {label}")

        for track in track_names:
            max_diff_key = f"{track}_max_abs_diff"
            mean_diff_key = f"{track}_mean_abs_diff"
            log2fc_key = f"{track}_max_abs_log2fc"
            if max_diff_key in r:
                print(
                    f"    {track:15s}  max_abs_diff={r[max_diff_key]:.6f}  "
                    f"mean_abs_diff={r[mean_diff_key]:.6f}  "
                    f"max_log2fc={r[log2fc_key]:.4f}"
                )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict variant effects using AlphaGenome. "
            "SOTA on 25/26 variant effect prediction tasks (Nature 2026)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --variants chr22:36201698:A:C --tracks RNA_SEQ ATAC_SEQ\n"
            "  %(prog)s --vcf variants.vcf --output results.tsv\n"
            "  %(prog)s --variants chr7:5530601:G:T --ontology UBERON:0002048\n"
        ),
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--variants",
        nargs="+",
        metavar="CHR:POS:REF:ALT",
        help="One or more variants in chr:pos:ref:alt format (e.g., chr22:36201698:A:C)",
    )
    input_group.add_argument(
        "--vcf",
        metavar="FILE",
        help="Path to a VCF file containing variants",
    )

    # Track selection
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=DEFAULT_TRACKS,
        choices=list(TRACK_NAME_MAP.keys()),
        help=f"Prediction tracks to query (default: {' '.join(DEFAULT_TRACKS)})",
    )

    # Tissue/cell context
    parser.add_argument(
        "--ontology",
        nargs="+",
        metavar="TERM",
        help=(
            "UBERON or CL ontology terms for tissue/cell type context "
            "(e.g., UBERON:0001157 for colon, CL:0000746 for cardiac muscle cell)"
        ),
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        default="variant_effects.tsv",
        metavar="FILE",
        help="Output TSV file path (default: variant_effects.tsv)",
    )

    # Configuration
    parser.add_argument(
        "--flank",
        type=int,
        default=524_288,
        metavar="BP",
        help="Flanking bases on each side of variant (default: 524288 = 2^19)",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="AlphaGenome API key (default: ALPHAGENOME_API_KEY env var)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Seconds to wait between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of variants to process from VCF",
    )

    args = parser.parse_args()

    # Parse variants
    if args.variants:
        variants = [parse_variant_string(v) for v in args.variants]
    else:
        if not os.path.isfile(args.vcf):
            print(f"Error: VCF file not found: {args.vcf}", file=sys.stderr)
            sys.exit(1)
        variants = parse_vcf(args.vcf, max_variants=args.max_variants)

    if not variants:
        print("Error: No variants to process.", file=sys.stderr)
        sys.exit(1)

    print(f"AlphaGenome Variant Effect Prediction")
    print(f"  Variants: {len(variants)}")
    print(f"  Tracks: {', '.join(args.tracks)}")
    print(f"  Ontology: {', '.join(args.ontology) if args.ontology else 'default'}")
    print(f"  Flank: {args.flank:,} bp")
    print(f"  Output: {args.output}")
    print()

    # Run predictions
    results = predict_variant_effects(
        variants=variants,
        track_names=args.tracks,
        ontology_terms=args.ontology,
        flank_size=args.flank,
        api_key=args.api_key,
        rate_limit_delay=args.rate_limit,
    )

    # Write output
    write_results(results, args.output)

    # Print summary
    print_summary(results, args.tracks)


if __name__ == "__main__":
    main()

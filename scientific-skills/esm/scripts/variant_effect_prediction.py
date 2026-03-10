#!/usr/bin/env python3
"""
Zero-Shot Variant Effect Prediction with ESM-2

Predict the functional effect of amino-acid substitutions using ESM-2
masked marginal scoring. No training data required -- the pretrained
language model's log-likelihood captures evolutionary constraints.

Negative scores indicate deleterious variants; positive scores indicate
benign or potentially beneficial variants.

Usage:
    # Score a list of mutations on a given sequence
    python variant_effect_prediction.py \
        --sequence MKFLILLFNILCLFPVLAADNHGVSMRV \
        --mutations A5G,L8P,F3A

    # Read the wild-type sequence from a FASTA file
    python variant_effect_prediction.py \
        --fasta wild_type.fasta \
        --mutations A5G,L8P \
        --output scores.csv

    # Score all possible single-residue substitutions at a position
    python variant_effect_prediction.py \
        --sequence MKFLILLFNILCLFPVLAADNHGVSMRV \
        --scan-positions 5,8,12

ESM-2 zero-shot variant effect prediction correlates well with
experimental DMS data. See ProteinBench (2024) for comprehensive
evaluation.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
    "15B": "facebook/esm2_t48_15B_UR50D",
}

MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_fasta_first(fasta_path: str) -> str:
    """Return the first sequence from a FASTA file."""
    seq_lines: list[str] = []
    started = False
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if started:
                    break
                started = True
                continue
            if started:
                seq_lines.append(line)
    if not seq_lines:
        raise ValueError(f"No sequence found in {fasta_path}")
    return "".join(seq_lines)


def parse_mutation(mutation_str: str, sequence: str) -> tuple[str, int, str]:
    """
    Parse a mutation string like 'A5G' -> (wt_aa, position_0indexed, mut_aa).

    Validates that the wild-type residue matches the sequence.
    """
    m = MUTATION_RE.match(mutation_str.strip().upper())
    if not m:
        raise ValueError(
            f"Invalid mutation format '{mutation_str}'. "
            "Expected e.g. 'A123G' (wild-type AA, 1-based position, mutant AA)."
        )
    wt_aa, pos_str, mut_aa = m.group(1), m.group(2), m.group(3)
    pos_1 = int(pos_str)
    pos_0 = pos_1 - 1

    if pos_0 < 0 or pos_0 >= len(sequence):
        raise ValueError(
            f"Position {pos_1} is out of range for sequence of length {len(sequence)}."
        )
    if sequence[pos_0] != wt_aa:
        raise ValueError(
            f"Mutation {mutation_str}: expected '{wt_aa}' at position {pos_1}, "
            f"but sequence has '{sequence[pos_0]}'."
        )
    return wt_aa, pos_0, mut_aa


# ---------------------------------------------------------------------------
# Masked marginal scoring
# ---------------------------------------------------------------------------

def load_model(model_size: str, device: str):
    """Load ESM-2 model for masked language modeling."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name = MODELS.get(model_size)
    if model_name is None:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Choose from: {', '.join(MODELS.keys())}"
        )

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    print(f"Model loaded on {device}.")
    return model, tokenizer


def masked_marginal_score(
    model,
    tokenizer,
    sequence: str,
    positions: list[int],
    device: str,
) -> np.ndarray:
    """
    Compute masked marginal log-probabilities at given positions.

    For each position, mask it, run a forward pass, and extract the
    log-softmax distribution over the vocabulary.

    Args:
        model: ESM-2 masked LM.
        tokenizer: Corresponding tokenizer.
        sequence: Wild-type protein sequence.
        positions: 0-based residue positions to score.
        device: Torch device.

    Returns:
        Array of shape (len(positions), vocab_size) with log-probs.
    """
    # Build token-id lookup for standard amino acids
    aa_token_ids = []
    for aa in STANDARD_AAS:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer cannot encode amino acid '{aa}'.")
        aa_token_ids.append(tid)
    aa_token_ids = np.array(aa_token_ids)

    # Tokenize wild-type once to get the base input_ids
    encoding = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    base_ids = encoding["input_ids"].squeeze(0).clone()  # (seq_len_with_special,)

    mask_token_id = tokenizer.mask_token_id
    log_probs_all = []

    for pos_0 in positions:
        # Token index = pos_0 + 1 because of leading [CLS]
        token_idx = pos_0 + 1

        masked_ids = base_ids.clone()
        masked_ids[token_idx] = mask_token_id
        masked_ids = masked_ids.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(masked_ids).logits  # (1, seq_len, vocab_size)

        log_probs = torch.log_softmax(logits[0, token_idx], dim=-1).cpu().numpy()
        log_probs_all.append(log_probs)

    return np.stack(log_probs_all, axis=0), aa_token_ids


def score_mutations(
    model,
    tokenizer,
    sequence: str,
    mutations: list[tuple[str, int, str]],
    device: str,
) -> list[dict]:
    """
    Score a list of mutations using masked marginal log-likelihood ratio.

    Returns a list of dicts with keys: mutation, wt_aa, mut_aa, position,
    wt_logprob, mut_logprob, score (= mut_logprob - wt_logprob).
    """
    positions = sorted(set(pos for _, pos, _ in mutations))
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    print(f"Scoring {len(mutations)} mutation(s) at {len(positions)} position(s) ...")
    log_probs, aa_token_ids = masked_marginal_score(
        model, tokenizer, sequence, positions, device
    )

    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}

    results = []
    for wt_aa, pos_0, mut_aa in mutations:
        lp = log_probs[pos_to_idx[pos_0]]
        wt_tid = aa_token_ids[aa_to_idx[wt_aa]]
        mut_tid = aa_token_ids[aa_to_idx[mut_aa]]
        wt_lp = float(lp[wt_tid])
        mut_lp = float(lp[mut_tid])
        score = mut_lp - wt_lp  # log-likelihood ratio

        results.append({
            "mutation": f"{wt_aa}{pos_0 + 1}{mut_aa}",
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "position": pos_0 + 1,
            "wt_logprob": round(wt_lp, 4),
            "mut_logprob": round(mut_lp, 4),
            "score": round(score, 4),
        })

    return results


def scan_positions(
    model,
    tokenizer,
    sequence: str,
    positions_0: list[int],
    device: str,
) -> list[dict]:
    """
    Exhaustive single-residue substitution scan at given positions.
    """
    print(f"Scanning all substitutions at {len(positions_0)} position(s) ...")
    log_probs, aa_token_ids = masked_marginal_score(
        model, tokenizer, sequence, positions_0, device
    )

    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}
    results = []

    for i, pos_0 in enumerate(positions_0):
        wt_aa = sequence[pos_0]
        wt_tid = aa_token_ids[aa_to_idx[wt_aa]]
        wt_lp = float(log_probs[i][wt_tid])

        for mut_aa in STANDARD_AAS:
            if mut_aa == wt_aa:
                continue
            mut_tid = aa_token_ids[aa_to_idx[mut_aa]]
            mut_lp = float(log_probs[i][mut_tid])
            score = mut_lp - wt_lp

            results.append({
                "mutation": f"{wt_aa}{pos_0 + 1}{mut_aa}",
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "position": pos_0 + 1,
                "wt_logprob": round(wt_lp, 4),
                "mut_logprob": round(mut_lp, 4),
                "score": round(score, 4),
            })

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: list[dict]):
    """Pretty-print variant effect scores to stdout."""
    print("\n" + "=" * 65)
    print(f"{'Mutation':<12} {'Score':>8}  {'WT logP':>9}  {'Mut logP':>9}  Interpretation")
    print("-" * 65)

    for r in sorted(results, key=lambda x: x["score"]):
        interp = (
            "deleterious" if r["score"] < -2.0
            else "likely deleterious" if r["score"] < -1.0
            else "neutral" if r["score"] < 1.0
            else "likely benign" if r["score"] < 2.0
            else "benign"
        )
        print(
            f"{r['mutation']:<12} {r['score']:>8.4f}  "
            f"{r['wt_logprob']:>9.4f}  {r['mut_logprob']:>9.4f}  {interp}"
        )
    print("=" * 65)


def save_results(results: list[dict], output_path: str):
    """Save results to CSV."""
    fieldnames = ["mutation", "position", "wt_aa", "mut_aa",
                  "wt_logprob", "mut_logprob", "score"]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["score"]):
            writer.writerow(r)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot variant effect prediction with ESM-2 masked marginal scoring."
    )

    seq_group = parser.add_mutually_exclusive_group(required=True)
    seq_group.add_argument(
        "--sequence",
        type=str,
        help="Wild-type protein sequence.",
    )
    seq_group.add_argument(
        "--fasta",
        type=str,
        help="FASTA file with wild-type sequence (uses the first record).",
    )

    mut_group = parser.add_mutually_exclusive_group(required=True)
    mut_group.add_argument(
        "--mutations",
        type=str,
        help="Comma-separated list of mutations, e.g. 'A5G,L8P,F3A'.",
    )
    mut_group.add_argument(
        "--scan-positions",
        type=str,
        help="Comma-separated 1-based positions for exhaustive substitution scan.",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="650M",
        choices=list(MODELS.keys()),
        help="ESM-2 model size (default: 650M).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional; results always printed to stdout).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/cpu).",
    )

    args = parser.parse_args()

    # ---- Sequence --------------------------------------------------------
    if args.fasta:
        sequence = parse_fasta_first(args.fasta)
    else:
        sequence = args.sequence.strip().upper()

    print("=" * 60)
    print("ESM-2 Zero-Shot Variant Effect Prediction")
    print("=" * 60)
    print(f"Sequence length: {len(sequence)}")

    # ---- Device ----------------------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ------------------------------------------------------
    model, tokenizer = load_model(args.model_size, device)

    # ---- Score -----------------------------------------------------------
    if args.mutations:
        mutation_strs = [m.strip() for m in args.mutations.split(",") if m.strip()]
        parsed = [parse_mutation(m, sequence) for m in mutation_strs]
        results = score_mutations(model, tokenizer, sequence, parsed, device)
    else:
        positions_1 = [int(p.strip()) for p in args.scan_positions.split(",")]
        positions_0 = [p - 1 for p in positions_1]
        for p0 in positions_0:
            if p0 < 0 or p0 >= len(sequence):
                print(
                    f"Error: position {p0 + 1} out of range for "
                    f"sequence of length {len(sequence)}.",
                    file=sys.stderr,
                )
                return 1
        results = scan_positions(model, tokenizer, sequence, positions_0, device)

    # ---- Output ----------------------------------------------------------
    print_results(results)

    if args.output:
        save_results(results, args.output)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Zero-Shot Biomedical Image Classification with BiomedCLIP

Classify biomedical images into arbitrary categories without fine-tuning
by computing image-text similarity using BiomedCLIP (PubMedBERT + ViT-B/16).

Usage:
    # Classify a single image
    python image_classification.py \
        --images chest_xray.jpg \
        --labels "normal chest X-ray" "pneumonia" "pleural effusion"

    # Classify multiple images with JSON output
    python image_classification.py \
        --images img1.jpg img2.png img3.jpg \
        --labels "adenocarcinoma" "squamous cell carcinoma" "normal tissue" \
        --output results.json

    # Use a custom prompt template
    python image_classification.py \
        --images slide.png \
        --labels "benign" "malignant" \
        --template "a histopathology image showing {label}"

BiomedCLIP outperforms radiology-specific BioViL on RSNA pneumonia detection
(NEJM AI 2024). For pathology-specific tasks, consider UNI or Virchow instead.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_model(device: str):
    """
    Load BiomedCLIP model, preprocessing transform, and tokenizer.

    Args:
        device: Torch device string (e.g. "cuda", "cpu").

    Returns:
        (model, preprocess, tokenizer) tuple.
    """
    from open_clip import create_model_from_pretrained, get_tokenizer

    print(f"Loading BiomedCLIP from {MODEL_NAME} ...")
    model, preprocess = create_model_from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()
    print(f"Model loaded on {device}.")
    return model, preprocess, tokenizer


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_images(
    model,
    preprocess,
    tokenizer,
    image_paths: list[str],
    labels: list[str],
    template: str,
    device: str,
) -> list[dict]:
    """
    Perform zero-shot classification on a list of images.

    Args:
        model: BiomedCLIP model.
        preprocess: Image preprocessing transform.
        tokenizer: Text tokenizer.
        image_paths: Paths to input images.
        labels: Candidate label strings.
        template: Prompt template with {label} placeholder.
        device: Torch device.

    Returns:
        List of dicts, one per image, each containing:
            - image: filename
            - predictions: list of {label, probability} sorted by probability
            - top_label: highest-probability label
            - top_probability: highest probability value
    """
    # Prepare text prompts from labels
    prompts = [template.format(label=label) for label in labels]
    text_tokens = tokenizer(prompts, context_length=256).to(device)

    # Encode text features once (shared across all images)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results = []

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"  WARNING: Image not found: {img_path}", file=sys.stderr)
            results.append({
                "image": str(img_path),
                "error": "File not found",
            })
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  WARNING: Could not open {img_path}: {e}", file=sys.stderr)
            results.append({
                "image": str(img_path),
                "error": str(e),
            })
            continue

        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity scaled by learned logit_scale
            logit_scale = model.logit_scale.exp()
            logits = (logit_scale * image_features @ text_features.t()).squeeze(0)
            probs = logits.softmax(dim=-1).cpu().numpy()

        # Sort by probability descending
        sorted_indices = np.argsort(probs)[::-1]
        predictions = [
            {"label": labels[idx], "probability": float(probs[idx])}
            for idx in sorted_indices
        ]

        result = {
            "image": str(img_path.name),
            "predictions": predictions,
            "top_label": predictions[0]["label"],
            "top_probability": predictions[0]["probability"],
        }
        results.append(result)

        # Print to stdout
        print(f"\n  {img_path.name}:")
        for pred in predictions:
            bar = "#" * int(pred["probability"] * 40)
            print(f"    {pred['label']:40s} {pred['probability']:.4f}  {bar}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot biomedical image classification with BiomedCLIP. "
            "BiomedCLIP outperforms radiology-specific BioViL on RSNA pneumonia "
            "detection (NEJM AI 2024). For pathology-specific tasks, consider "
            "UNI or Virchow instead."
        )
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Path(s) to input image file(s).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Candidate classification labels.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="this is a photo of {label}",
        help=(
            "Prompt template with {label} placeholder "
            "(default: 'this is a photo of {label}')."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If omitted, results are printed to stdout only.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/cpu).",
    )

    args = parser.parse_args()

    # Validate template
    if "{label}" not in args.template:
        print(
            "Error: --template must contain '{label}' placeholder.",
            file=sys.stderr,
        )
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("BiomedCLIP Zero-Shot Image Classification")
    print("=" * 60)
    print(f"Images:   {len(args.images)}")
    print(f"Labels:   {args.labels}")
    print(f"Template: {args.template}")
    print(f"Device:   {device}")

    # Load model
    model, preprocess, tokenizer = load_model(device)

    # Classify
    print("\nClassifying images ...")
    results = classify_images(
        model, preprocess, tokenizer,
        args.images, args.labels, args.template, device,
    )

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

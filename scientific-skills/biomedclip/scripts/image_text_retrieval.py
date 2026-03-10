#!/usr/bin/env python3
"""
Cross-Modal Biomedical Image-Text Retrieval with BiomedCLIP

Given a text query, retrieve the most relevant images from a collection,
or given an image query, retrieve the most relevant text descriptions.
Builds an embedding index of images for efficient similarity search.

Usage:
    # Text-to-image retrieval: find images matching a query
    python image_text_retrieval.py \
        --query "chest X-ray showing bilateral pneumonia" \
        --image-dir ./radiology_images/ \
        --top-k 5 \
        --output results.json

    # Image-to-text retrieval: find best-matching descriptions for an image
    python image_text_retrieval.py \
        --query-image pathology_slide.jpg \
        --captions captions.txt \
        --top-k 3

    # Save and reuse image embeddings for faster repeated queries
    python image_text_retrieval.py \
        --query "histopathology adenocarcinoma" \
        --image-dir ./slides/ \
        --save-embeddings embeddings.npz \
        --top-k 10

    # Load pre-computed embeddings
    python image_text_retrieval.py \
        --query "brain MRI with tumor" \
        --load-embeddings embeddings.npz \
        --top-k 5

BiomedCLIP is trained on 15M biomedical figure-caption pairs from PubMed Central,
providing strong cross-modal retrieval across diverse biomedical imaging modalities.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: str):
    """
    Load BiomedCLIP model, preprocessing transform, and tokenizer.

    Args:
        device: Torch device string.

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
# Embedding computation
# ---------------------------------------------------------------------------

def encode_images(
    model,
    preprocess,
    image_paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute normalized image embeddings for a list of image files.

    Args:
        model: BiomedCLIP model.
        preprocess: Image preprocessing transform.
        image_paths: List of paths to image files.
        device: Torch device.
        batch_size: Number of images per forward pass.

    Returns:
        (embeddings, filenames) where embeddings has shape (n_images, embed_dim)
        and filenames is a list of corresponding file names.
    """
    all_embeddings = []
    valid_filenames = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_tensors = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(preprocess(img))
                valid_filenames.append(str(p))
            except Exception as e:
                print(f"  WARNING: Skipping {p.name}: {e}", file=sys.stderr)

        if not batch_tensors:
            continue

        images_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            features = model.encode_image(images_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        n_done = min(start + batch_size, len(image_paths))
        print(f"  Encoded {n_done}/{len(image_paths)} images ...")

    if not all_embeddings:
        return np.array([]).reshape(0, 0), []

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings, valid_filenames


def encode_text_query(
    model,
    tokenizer,
    query: str,
    device: str,
) -> np.ndarray:
    """
    Compute a normalized text embedding for a query string.

    Args:
        model: BiomedCLIP model.
        tokenizer: Text tokenizer.
        query: Text query string.
        device: Torch device.

    Returns:
        Normalized embedding array of shape (1, embed_dim).
    """
    tokens = tokenizer([query], context_length=256).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()


def encode_image_query(
    model,
    preprocess,
    image_path: str,
    device: str,
) -> np.ndarray:
    """
    Compute a normalized image embedding for a single query image.

    Args:
        model: BiomedCLIP model.
        preprocess: Image preprocessing transform.
        image_path: Path to the query image.
        device: Torch device.

    Returns:
        Normalized embedding array of shape (1, embed_dim).
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy()


def encode_captions(
    model,
    tokenizer,
    captions: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute normalized text embeddings for a list of captions.

    Args:
        model: BiomedCLIP model.
        tokenizer: Text tokenizer.
        captions: List of caption strings.
        device: Torch device.
        batch_size: Number of captions per forward pass.

    Returns:
        Normalized embeddings array of shape (n_captions, embed_dim).
    """
    all_embeddings = []

    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        tokens = tokenizer(batch, context_length=256).to(device)

        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    query_embedding: np.ndarray,
    index_embeddings: np.ndarray,
    index_labels: list[str],
    top_k: int,
) -> list[dict]:
    """
    Retrieve the top-k most similar items from an embedding index.

    Args:
        query_embedding: Query embedding of shape (1, embed_dim).
        index_embeddings: Index embeddings of shape (n_items, embed_dim).
        index_labels: Labels/names for each item in the index.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: rank, item, similarity.
    """
    # Cosine similarity (embeddings are already normalized)
    similarities = (query_embedding @ index_embeddings.T).squeeze(0)
    top_k = min(top_k, len(index_labels))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "item": index_labels[idx],
            "similarity": float(similarities[idx]),
        })

    return results


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def save_embeddings(path: str, embeddings: np.ndarray, filenames: list[str]):
    """Save image embeddings and filenames to a .npz file."""
    np.savez_compressed(
        path,
        embeddings=embeddings,
        filenames=np.array(filenames),
    )
    print(f"Saved embeddings ({embeddings.shape[0]} images) to {path}")


def load_embeddings(path: str) -> tuple[np.ndarray, list[str]]:
    """Load image embeddings and filenames from a .npz file."""
    data = np.load(path, allow_pickle=False)
    embeddings = data["embeddings"]
    filenames = list(data["filenames"])
    print(f"Loaded embeddings ({embeddings.shape[0]} images) from {path}")
    return embeddings, filenames


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(image_dir: str) -> list[Path]:
    """Find all image files in a directory (non-recursive)."""
    d = Path(image_dir)
    if not d.is_dir():
        print(f"Error: {image_dir} is not a directory.", file=sys.stderr)
        return []

    paths = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    print(f"Found {len(paths)} images in {image_dir}")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-modal biomedical image-text retrieval with BiomedCLIP. "
            "Supports text-to-image and image-to-text retrieval."
        )
    )

    # Query input (one of these is required)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        type=str,
        help="Text query for text-to-image retrieval.",
    )
    query_group.add_argument(
        "--query-image",
        type=str,
        help="Image path for image-to-text retrieval.",
    )

    # Index input (one of these is required)
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument(
        "--image-dir",
        type=str,
        help="Directory of images to search (for text-to-image retrieval).",
    )
    index_group.add_argument(
        "--captions",
        type=str,
        help="Text file with one caption per line (for image-to-text retrieval).",
    )
    index_group.add_argument(
        "--load-embeddings",
        type=str,
        help="Path to pre-computed image embeddings (.npz) to load.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If omitted, results are printed to stdout only.",
    )
    parser.add_argument(
        "--save-embeddings",
        type=str,
        default=None,
        help="Save computed image embeddings to this .npz file for reuse.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding images (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect cuda/cpu).",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("BiomedCLIP Cross-Modal Retrieval")
    print("=" * 60)
    print(f"Device: {device}")

    # Load model
    model, preprocess, tokenizer = load_model(device)

    # ---- Text-to-image retrieval ------------------------------------------
    if args.query:
        print(f"\nQuery: \"{args.query}\"")
        query_embedding = encode_text_query(model, tokenizer, args.query, device)

        if args.load_embeddings:
            index_embeddings, index_labels = load_embeddings(args.load_embeddings)
        elif args.image_dir:
            image_paths = discover_images(args.image_dir)
            if not image_paths:
                print("Error: No images found.", file=sys.stderr)
                return 1
            print(f"\nEncoding {len(image_paths)} images ...")
            index_embeddings, index_labels = encode_images(
                model, preprocess, image_paths, device, args.batch_size
            )
            if args.save_embeddings:
                save_embeddings(args.save_embeddings, index_embeddings, index_labels)
        else:
            print(
                "Error: --image-dir or --load-embeddings required for text queries.",
                file=sys.stderr,
            )
            return 1

    # ---- Image-to-text retrieval ------------------------------------------
    elif args.query_image:
        print(f"\nQuery image: {args.query_image}")
        if not Path(args.query_image).exists():
            print(f"Error: Image not found: {args.query_image}", file=sys.stderr)
            return 1

        query_embedding = encode_image_query(model, preprocess, args.query_image, device)

        if args.captions:
            captions_path = Path(args.captions)
            if not captions_path.exists():
                print(f"Error: Captions file not found: {args.captions}", file=sys.stderr)
                return 1
            captions = [
                line.strip() for line in captions_path.read_text().splitlines()
                if line.strip()
            ]
            print(f"Loaded {len(captions)} captions from {args.captions}")
            print(f"\nEncoding {len(captions)} captions ...")
            index_embeddings = encode_captions(
                model, tokenizer, captions, device, args.batch_size
            )
            index_labels = captions
        elif args.load_embeddings:
            # Image-to-image retrieval using pre-computed embeddings
            index_embeddings, index_labels = load_embeddings(args.load_embeddings)
        elif args.image_dir:
            # Image-to-image retrieval
            image_paths = discover_images(args.image_dir)
            if not image_paths:
                print("Error: No images found.", file=sys.stderr)
                return 1
            print(f"\nEncoding {len(image_paths)} images ...")
            index_embeddings, index_labels = encode_images(
                model, preprocess, image_paths, device, args.batch_size
            )
            if args.save_embeddings:
                save_embeddings(args.save_embeddings, index_embeddings, index_labels)
        else:
            print(
                "Error: --captions, --image-dir, or --load-embeddings required.",
                file=sys.stderr,
            )
            return 1

    # ---- Retrieve ---------------------------------------------------------
    if index_embeddings.size == 0:
        print("Error: No valid items in index.", file=sys.stderr)
        return 1

    results = retrieve_top_k(
        query_embedding, index_embeddings, index_labels, args.top_k
    )

    print(f"\nTop-{args.top_k} results:")
    for r in results:
        print(f"  [{r['rank']}] {r['item']}  (similarity: {r['similarity']:.4f})")

    # ---- Output -----------------------------------------------------------
    output_data = {
        "query": args.query or args.query_image,
        "query_type": "text" if args.query else "image",
        "top_k": args.top_k,
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

---
name: biomedclip
description: BiomedCLIP is a vision-language foundation model pretrained on 15M biomedical figure-caption pairs from PubMed Central. Use this skill for zero-shot biomedical image classification, cross-modal image-text retrieval, and visual question answering across diverse biomedical imaging modalities (radiology, histopathology, microscopy, etc.). Based on PubMedBERT text encoder and ViT-B/16 image encoder.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# BiomedCLIP: Biomedical Vision-Language Foundation Model

## Overview

BiomedCLIP is a multimodal biomedical foundation model that aligns biomedical images and text in a shared embedding space using contrastive learning. It was pretrained on **PMC-15M**, a dataset of 15 million figure-caption pairs extracted from PubMed Central open-access research articles, covering diverse biomedical image types including radiology, histopathology, microscopy, clinical photography, and scientific charts.

- **HuggingFace model:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Text encoder:** PubMedBERT (256 token context length)
- **Image encoder:** Vision Transformer (ViT-B/16, 224x224 resolution)
- **Paper:** Zhang et al., "A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image-Text Pairs", *NEJM AI* (2024)

## Core Capabilities

### 1. Zero-Shot Biomedical Image Classification

Classify biomedical images into arbitrary categories without fine-tuning by computing image-text similarity scores.

**When to use:**
- Classifying biomedical images across modalities (X-ray, MRI, histology, microscopy)
- Rapid prototyping of image classifiers without labeled training data
- Screening or triaging images by category

**Basic usage:**

```python
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image

model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Load and preprocess image
image = preprocess(Image.open("chest_xray.jpg")).unsqueeze(0).to(device)

# Define candidate labels
labels = ["normal chest X-ray", "pneumonia chest X-ray", "pleural effusion"]
texts = tokenizer(["this is a photo of " + l for l in labels], context_length=256).to(device)

# Compute similarities
with torch.no_grad():
    image_features, text_features, logit_scale = model(image, texts)
    probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)

for label, prob in zip(labels, probs[0]):
    print(f"  {label}: {prob:.4f}")
```

**CLI script:**

```bash
python scripts/image_classification.py \
    --images chest_xray.jpg brain_mri.jpg \
    --labels "normal chest X-ray" "pneumonia" "brain tumor" "healthy brain" \
    --output results.json
```

### 2. Cross-Modal Image-Text Retrieval

Retrieve the most relevant images given a text query, or find the best-matching text descriptions for an image.

**When to use:**
- Searching a biomedical image collection with natural language queries
- Finding images similar to a text description
- Ranking text descriptions for a given image

**Basic usage:**

```python
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from pathlib import Path

model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Encode a collection of images
image_dir = Path("biomedical_images/")
image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
images = torch.stack([preprocess(Image.open(p)) for p in image_paths]).to(device)

with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Query with text
query = "histopathology showing adenocarcinoma"
text_tokens = tokenizer([query], context_length=256).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Rank images by similarity
similarities = (text_features @ image_features.t()).squeeze(0)
top_indices = similarities.argsort(descending=True)[:5]

for idx in top_indices:
    print(f"  {image_paths[idx].name}: {similarities[idx]:.4f}")
```

**CLI script:**

```bash
python scripts/image_text_retrieval.py \
    --query "chest X-ray showing pneumonia" \
    --image-dir ./radiology_images/ \
    --top-k 5 \
    --output retrieval_results.json
```

### 3. Visual Question Answering

Frame VQA as a classification task by encoding candidate answers as text and selecting the best match.

```python
# Frame VQA as zero-shot classification
image = preprocess(Image.open("pathology_slide.jpg")).unsqueeze(0).to(device)
answers = ["benign tissue", "malignant tumor", "inflammatory infiltrate", "necrosis"]
texts = tokenizer(answers, context_length=256).to(device)

with torch.no_grad():
    image_features, text_features, logit_scale = model(image, texts)
    probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)

best = probs[0].argmax()
print(f"Answer: {answers[best]} ({probs[0][best]:.4f})")
```

## Model Comparison

| Model | Training Data | Scope | Key Strengths |
|-------|--------------|-------|---------------|
| **BiomedCLIP** | 15M PMC figure-caption pairs | General biomedical | Broadest coverage; SOTA across multiple VLP benchmarks |
| PubMedCLIP | 80K radiology image-text pairs | Radiology | Small-scale; superseded by BiomedCLIP |
| PLIP | 208K pathology images from Twitter/X | Pathology | Social media data; noisy labels |
| BioViL | Radiology reports + images | Radiology | Outperformed by BiomedCLIP even on radiology tasks |
| UNI | 100M+ pathology tiles | Pathology | Superior for pathology-specific tasks; self-supervised |
| Virchow | 1.5B pathology tiles | Cancer pathology | Largest pathology model; best for cancer detection |

**Key benchmark results:**
- Outperforms BioViL (radiology-specific) even on radiology benchmarks such as RSNA pneumonia detection
- State-of-the-art on multiple vision-language processing benchmarks at time of publication
- Strong zero-shot performance across diverse biomedical imaging modalities

## When to Use BiomedCLIP

**Use BiomedCLIP when:**
- You need a general-purpose biomedical image-text model spanning multiple modalities
- Performing zero-shot classification without labeled training data
- Building cross-modal retrieval systems over biomedical image collections
- Working with diverse image types (radiology, histopathology, microscopy, charts)
- Prototyping before investing in domain-specific fine-tuning

**Do NOT use BiomedCLIP when:**
- You have a domain-specific model that fits your task (UNI or Virchow for pathology, CheXNet for chest X-rays)
- You need pixel-level segmentation (BiomedCLIP provides image-level representations only)
- You are working with non-biomedical images (use standard CLIP or domain-specific models)
- You need clinical-grade diagnostic outputs (BiomedCLIP is a research tool, not approved for clinical use)
- Your images are very different from PMC figures (e.g., raw DICOM volumes without 2D rendering)

## Installation

```bash
pip install open_clip_torch transformers
```

For GPU acceleration:

```bash
pip install open_clip_torch transformers torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Best Practices

**For zero-shot classification:**
- Use descriptive label prompts (e.g., "a chest X-ray showing pneumonia" rather than just "pneumonia")
- Experiment with prompt templates: "this is a photo of {label}", "a medical image of {label}", "{label} histopathology"
- Normalize features before computing similarities for stable results

**For retrieval:**
- Pre-compute and cache image embeddings for large collections
- Normalize embeddings to unit length before computing cosine similarity
- Use batch processing for large image sets to maximize GPU utilization

**For production use:**
- BiomedCLIP is intended for research purposes only, not for clinical deployment
- Always validate results against domain-expert ground truth
- Consider fine-tuning on your specific domain if zero-shot performance is insufficient

## Resources and Documentation

- **HuggingFace:** https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Paper:** Zhang et al., NEJM AI (2024) - https://ai.nejm.org/doi/full/10.1056/AIoa2400640
- **Data Pipeline:** https://github.com/microsoft/BiomedCLIP_data_pipeline
- **Example Notebook:** https://aka.ms/biomedclip-example-notebook

## Responsible Use

BiomedCLIP is designed for biomedical research and is NOT approved for clinical decision-making. Always have qualified medical professionals validate outputs. The model may reflect biases present in PubMed Central literature and should not be used as a sole diagnostic tool.

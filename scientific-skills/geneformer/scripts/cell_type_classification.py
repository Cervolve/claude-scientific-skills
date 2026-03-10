#!/usr/bin/env python3
"""
Fine-tune Geneformer for Cell Type Classification

This script takes an AnnData (.h5ad) file with cell type labels, tokenizes
the gene expression data using Geneformer's rank-value encoding, fine-tunes
the pretrained model, and evaluates performance with train/test split.

Requires:
    pip install geneformer scanpy scikit-learn datasets

WARNING: Zero-shot Geneformer underperforms simple baselines (Genome Biology
2025, Microsoft Research). ALWAYS fine-tune. For quick cell type annotation
without fine-tuning, consider scVI or Harmony+scANVI.

Usage:
    python cell_type_classification.py \
        --input data.h5ad \
        --label-column cell_type \
        --model-size 104M \
        --epochs 10 \
        --output ./results/
"""

# ============================================================================
# IMPORTANT: Zero-shot Geneformer underperforms simple baselines
# (Genome Biology 2025). Always fine-tune. For quick cell type annotation
# without fine-tuning, consider scVI or Harmony+scANVI.
# ============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import scanpy as sc
except ImportError:
    raise ImportError(
        "scanpy is required. Install with:\n  pip install scanpy"
    )

try:
    from geneformer import TranscriptomeTokenizer, Classifier
except ImportError:
    raise ImportError(
        "geneformer is not installed. Install with:\n"
        "  git lfs install\n"
        "  git clone https://huggingface.co/ctheodoris/Geneformer\n"
        "  cd Geneformer && pip install ."
    )

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError(
        "scikit-learn is required. Install with:\n  pip install scikit-learn"
    )

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    raise ImportError(
        "datasets (HuggingFace) is required. Install with:\n  pip install datasets"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

_MODEL_SIZE_MAP = {
    "10M": "geneformer-v1-10M",
    "104M": "geneformer-v2-104M",
    "316M": "geneformer-v2-316M",
    "104M_cancer": "geneformer-v2-104M_CLcancer",
}


def resolve_model_directory(model_size: str) -> str:
    """Resolve a model size shorthand to a HuggingFace model directory.

    If model_size is a path that exists on disk, use it directly.
    Otherwise, construct the HuggingFace model identifier.
    """
    if os.path.isdir(model_size):
        return model_size

    variant = _MODEL_SIZE_MAP.get(model_size, model_size)
    hf_path = f"ctheodoris/Geneformer/{variant}"
    return hf_path


# ---------------------------------------------------------------------------
# Data preparation: AnnData -> tokenized Geneformer Dataset
# ---------------------------------------------------------------------------

def prepare_anndata_for_tokenization(
    adata: sc.AnnData,
    label_column: str,
    work_dir: str,
) -> tuple[str, Dict[str, int]]:
    """Convert AnnData to .loom format and prepare label dictionaries.

    Geneformer's TranscriptomeTokenizer expects .loom files. This function:
    1. Validates that label_column exists in adata.obs
    2. Filters out cells with missing labels
    3. Ensures gene names are in var_names (Ensembl IDs preferred)
    4. Writes a .loom file for tokenization
    5. Creates a label-to-integer mapping dictionary

    Args:
        adata: AnnData object with raw or normalized counts.
        label_column: Column in adata.obs containing cell type labels.
        work_dir: Working directory for intermediate files.

    Returns:
        Tuple of (path to loom directory, label-to-int dict).
    """
    if label_column not in adata.obs.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Filter cells with missing labels
    mask = adata.obs[label_column].notna() & (adata.obs[label_column] != "")
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.warning(
            f"Removing {n_removed} cells with missing labels in '{label_column}'"
        )
        adata = adata[mask].copy()

    logger.info(
        f"Dataset: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{adata.obs[label_column].nunique()} cell types"
    )

    # Build label dictionary
    unique_labels = sorted(adata.obs[label_column].unique())
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    logger.info(f"Label mapping: {label_to_int}")

    # Write loom file for tokenization
    loom_dir = os.path.join(work_dir, "loom_input")
    os.makedirs(loom_dir, exist_ok=True)
    loom_path = os.path.join(loom_dir, "data.loom")

    # Geneformer expects the label in the loom file
    adata.obs["cell_type_label"] = adata.obs[label_column].astype(str)
    adata.write_loom(loom_path, write_obsm_varm=False)
    logger.info(f"Wrote loom file: {loom_path}")

    return loom_dir, label_to_int


def tokenize_data(
    loom_dir: str,
    work_dir: str,
    label_column: str,
    model_input_size: int = 4096,
) -> str:
    """Tokenize single-cell data using Geneformer's rank-value encoding.

    Args:
        loom_dir: Directory containing .loom file(s).
        work_dir: Working directory for output.
        label_column: Name of the label attribute in the loom file.
        model_input_size: Context length (2048 for V1, 4096 for V2).

    Returns:
        Path to the tokenized dataset directory.
    """
    tokenized_dir = os.path.join(work_dir, "tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={"cell_type_label": "cell_type_label"},
        nproc=4,
        model_input_size=model_input_size,
    )

    tk.tokenize_data(
        data_directory=loom_dir,
        output_directory=tokenized_dir,
        output_prefix="geneformer_tokenized",
        file_format="loom",
    )

    tokenized_path = os.path.join(tokenized_dir, "geneformer_tokenized.dataset")
    if not os.path.exists(tokenized_path):
        # Some versions output without .dataset extension
        candidates = list(Path(tokenized_dir).glob("geneformer_tokenized*"))
        if candidates:
            tokenized_path = str(candidates[0])
        else:
            raise FileNotFoundError(
                f"Tokenization produced no output in {tokenized_dir}"
            )

    logger.info(f"Tokenized dataset saved to: {tokenized_path}")
    return tokenized_path


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_dataset(
    tokenized_path: str,
    label_to_int: Dict[str, int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[str, str, Dict[int, str]]:
    """Split tokenized dataset into train and test sets.

    Args:
        tokenized_path: Path to tokenized .dataset directory.
        label_to_int: Mapping from label string to integer.
        test_size: Fraction of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset_path, test_dataset_path, int_to_label dict).
    """
    dataset = Dataset.load_from_disk(tokenized_path)
    int_to_label = {v: k for k, v in label_to_int.items()}

    # Map string labels to integers
    if "cell_type_label" in dataset.column_names:
        dataset = dataset.map(
            lambda x: {"label": label_to_int.get(x["cell_type_label"], -1)},
            num_proc=4,
        )
        # Remove cells with unknown labels
        dataset = dataset.filter(lambda x: x["label"] >= 0, num_proc=4)

    # Split
    split = dataset.train_test_split(
        test_size=test_size,
        seed=random_state,
        stratify_by_column="label",
    )

    parent_dir = str(Path(tokenized_path).parent)
    train_path = os.path.join(parent_dir, "train.dataset")
    test_path = os.path.join(parent_dir, "test.dataset")

    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(test_path)

    logger.info(
        f"Split: {len(split['train'])} train, {len(split['test'])} test cells"
    )

    return train_path, test_path, int_to_label


# ---------------------------------------------------------------------------
# Fine-tuning and evaluation
# ---------------------------------------------------------------------------

def fine_tune_and_evaluate(
    model_directory: str,
    train_path: str,
    test_path: str,
    label_to_int: Dict[str, int],
    int_to_label: Dict[int, str],
    output_dir: str,
    epochs: int = 10,
    learning_rate: float = 5e-5,
    freeze_layers: int = 2,
    batch_size: int = 32,
    n_hyperopt_trials: int = 0,
) -> Dict:
    """Fine-tune Geneformer and evaluate on held-out test set.

    Args:
        model_directory: Path to pretrained Geneformer model.
        train_path: Path to training dataset.
        test_path: Path to test dataset.
        label_to_int: Label string to integer mapping.
        int_to_label: Integer to label string mapping.
        output_dir: Directory for results.
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        freeze_layers: Number of transformer layers to freeze (from input).
        batch_size: Training batch size.
        n_hyperopt_trials: Number of hyperparameter optimization trials (0=none).

    Returns:
        Dictionary with evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save label dictionary for the classifier
    id_class_dict = {i: label for i, label in int_to_label.items()}
    id_class_dict_path = os.path.join(output_dir, "id_class_dict.pkl")
    with open(id_class_dict_path, "wb") as f:
        pickle.dump(id_class_dict, f)

    num_classes = len(label_to_int)
    logger.info(f"Fine-tuning with {num_classes} classes, {epochs} epochs")

    # -----------------------------------------------------------------------
    # Strategy 1: Use Geneformer's built-in Classifier (preferred)
    # -----------------------------------------------------------------------
    try:
        cc = Classifier(
            classifier="cell",
            cell_state_dict={"state_key": "cell_type_label", "states": "all"},
            training_args={
                "num_train_epochs": epochs,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
                "warmup_steps": 100,
                "weight_decay": 0.01,
            },
            freeze_layers=freeze_layers,
            num_crossval_splits=1,  # Single train/test split
            forward_batch_size=batch_size,
            nproc=4,
        )

        # Run validation (which includes training + evaluation)
        cc.validate(
            model_directory=model_directory,
            prepared_input_data_file=train_path,
            id_class_dict_file=id_class_dict_path,
            output_directory=output_dir,
            output_prefix="cell_classifier",
            n_hyperopt_trials=n_hyperopt_trials,
            split_sizes={"train": 0.8, "valid": 0.1, "test": 0.1},
        )

        logger.info("Geneformer Classifier training complete.")

    except Exception as e:
        logger.warning(
            f"Geneformer built-in Classifier failed ({e}). "
            "Falling back to manual HuggingFace Trainer approach."
        )
        return _fallback_train_evaluate(
            model_directory=model_directory,
            train_path=train_path,
            test_path=test_path,
            int_to_label=int_to_label,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            freeze_layers=freeze_layers,
            batch_size=batch_size,
            num_classes=num_classes,
        )

    # -----------------------------------------------------------------------
    # Evaluate on held-out test set
    # -----------------------------------------------------------------------
    return _evaluate_test_set(
        model_directory=output_dir,
        test_path=test_path,
        int_to_label=int_to_label,
        output_dir=output_dir,
    )


def _fallback_train_evaluate(
    model_directory: str,
    train_path: str,
    test_path: str,
    int_to_label: Dict[int, str],
    output_dir: str,
    epochs: int,
    learning_rate: float,
    freeze_layers: int,
    batch_size: int,
    num_classes: int,
) -> Dict:
    """Fallback: fine-tune using HuggingFace Trainer directly.

    This approach is used when the built-in Geneformer Classifier raises
    an error (e.g., due to API changes between versions).
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    logger.info("Loading model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_directory,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    )

    # Freeze layers if requested
    if freeze_layers > 0:
        modules_to_freeze = []
        if hasattr(model, "bert"):
            encoder = model.bert.encoder
        elif hasattr(model, "model"):
            encoder = model.model.encoder
        else:
            encoder = None

        if encoder is not None:
            for i, layer in enumerate(encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Froze first {freeze_layers} transformer layers")

    # Load datasets
    train_dataset = Dataset.load_from_disk(train_path)
    test_dataset = Dataset.load_from_disk(test_path)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save fine-tuned model
    finetuned_dir = os.path.join(output_dir, "finetuned_model")
    trainer.save_model(finetuned_dir)
    logger.info(f"Fine-tuned model saved to: {finetuned_dir}")

    # Evaluate
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    return _compute_and_save_metrics(preds, labels, int_to_label, output_dir)


def _evaluate_test_set(
    model_directory: str,
    test_path: str,
    int_to_label: Dict[int, str],
    output_dir: str,
) -> Dict:
    """Evaluate a fine-tuned model on the test dataset."""
    from transformers import AutoModelForSequenceClassification, Trainer

    # Look for the fine-tuned model in the output directory
    finetuned_candidates = list(Path(output_dir).glob("**/pytorch_model.bin")) + \
                           list(Path(output_dir).glob("**/model.safetensors"))

    if finetuned_candidates:
        model_path = str(finetuned_candidates[0].parent)
    else:
        model_path = model_directory

    test_dataset = Dataset.load_from_disk(test_path)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids
        return _compute_and_save_metrics(preds, labels, int_to_label, output_dir)
    except Exception as e:
        logger.warning(f"Could not evaluate test set: {e}")
        return {"error": str(e)}


def _compute_and_save_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    int_to_label: Dict[int, str],
    output_dir: str,
) -> Dict:
    """Compute classification metrics and save results."""
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    # Classification report
    target_names = [int_to_label.get(i, str(i)) for i in sorted(set(labels))]
    report = classification_report(
        labels, preds, target_names=target_names, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    results = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_test_cells": int(len(labels)),
        "n_classes": int(len(set(labels))),
    }

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:        {acc:.4f}")
    logger.info(f"Macro F1:        {f1_macro:.4f}")
    logger.info(f"Weighted F1:     {f1_weighted:.4f}")
    logger.info(f"Test cells:      {len(labels)}")
    logger.info(f"Classes:         {len(set(labels))}")
    logger.info("-" * 60)
    logger.info(f"\n{report}")

    # Save results
    results_path = os.path.join(output_dir, "classification_results.json")
    saveable = {k: v for k, v in results.items() if k != "classification_report"}
    saveable["classification_report_text"] = report
    with open(results_path, "w") as f:
        json.dump(saveable, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Geneformer for cell type classification.\n\n"
            "WARNING: Zero-shot Geneformer underperforms simple baselines "
            "(Genome Biology 2025). This script ALWAYS fine-tunes. For quick "
            "cell type annotation without fine-tuning, use scVI or Harmony+scANVI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to AnnData .h5ad file with cell type labels.",
    )
    parser.add_argument(
        "--label-column", default="cell_type",
        help="Column in adata.obs containing cell type labels (default: cell_type).",
    )
    parser.add_argument(
        "--model-size", default="104M",
        choices=list(_MODEL_SIZE_MAP.keys()),
        help="Geneformer model variant (default: 104M). Use 316M for best quality.",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Direct path to a local Geneformer model directory (overrides --model-size).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of fine-tuning epochs (default: 10).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="Learning rate (default: 5e-5).",
    )
    parser.add_argument(
        "--freeze-layers", type=int, default=2,
        help="Number of transformer layers to freeze from input side (default: 2).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for testing (default: 0.2).",
    )
    parser.add_argument(
        "--hyperopt-trials", type=int, default=0,
        help="Number of hyperparameter optimization trials (default: 0 = disabled).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for results and fine-tuned model.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    # Resolve model
    if args.model_path:
        model_dir = args.model_path
    else:
        model_dir = resolve_model_directory(args.model_size)

    # Determine model input size from variant
    model_input_size = 2048 if args.model_size == "10M" else 4096

    logger.info(f"Model: {model_dir}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Label column: {args.label_column}")

    # Load AnnData
    logger.info(f"Loading AnnData from {args.input}...")
    adata = sc.read_h5ad(args.input)

    # Set up working directory
    work_dir = os.path.join(args.output, "_work")
    os.makedirs(work_dir, exist_ok=True)

    # Step 1: Prepare data
    logger.info("Step 1/4: Preparing data for tokenization...")
    loom_dir, label_to_int = prepare_anndata_for_tokenization(
        adata, args.label_column, work_dir
    )

    # Step 2: Tokenize
    logger.info("Step 2/4: Tokenizing with rank-value encoding...")
    tokenized_path = tokenize_data(
        loom_dir, work_dir, args.label_column, model_input_size
    )

    # Step 3: Train/test split
    logger.info("Step 3/4: Splitting into train/test...")
    train_path, test_path, int_to_label = split_dataset(
        tokenized_path, label_to_int, args.test_size, args.seed
    )

    # Step 4: Fine-tune and evaluate
    logger.info("Step 4/4: Fine-tuning and evaluating...")
    results = fine_tune_and_evaluate(
        model_directory=model_dir,
        train_path=train_path,
        test_path=test_path,
        label_to_int=label_to_int,
        int_to_label=int_to_label,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_layers=args.freeze_layers,
        batch_size=args.batch_size,
        n_hyperopt_trials=args.hyperopt_trials,
    )

    logger.info("Done. Results saved to: %s", args.output)
    return results


if __name__ == "__main__":
    main()

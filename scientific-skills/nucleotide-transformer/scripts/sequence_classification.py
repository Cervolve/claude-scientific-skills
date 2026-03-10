#!/usr/bin/env python3
"""
DNA Sequence Classification with Nucleotide Transformer

Fine-tunes a classification head on top of Nucleotide Transformer embeddings
for genomic tasks such as promoter detection, splice site classification,
enhancer prediction, and histone modification prediction.

Mean token embedding consistently improves performance over other pooling
strategies (Nature Comms 2025 benchmark).

Usage:
    python sequence_classification.py \
        --input sequences.fasta \
        --labels labels.csv \
        --task promoter_detection \
        --model InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
        --output results/

    # Inference only (with trained head):
    python sequence_classification.py \
        --input test_sequences.fasta \
        --model InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
        --task promoter_detection \
        --checkpoint results/best_model.pt \
        --output predictions.csv
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_fasta(filepath: str) -> list[tuple[str, str]]:
    """Parse a FASTA file and return list of (header, sequence) tuples.
    Also accepts plain-text files with one sequence per line (no headers)."""
    records: list[tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: list[str] = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None or seq_lines:
                    records.append((header or f"seq_{len(records)}", "".join(seq_lines)))
                    seq_lines = []
                header = line[1:].strip()
            else:
                seq_lines.append(line.upper())
        if header is not None or seq_lines:
            records.append((header or f"seq_{len(records)}", "".join(seq_lines)))

    return records


def load_labels(filepath: str) -> dict[str, int]:
    """Load labels from a CSV with columns: sequence_id, label."""
    labels: dict[str, int] = {}
    with open(filepath, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels[row["sequence_id"]] = int(row["label"])
    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DNAClassificationDataset(Dataset):
    """Dataset for DNA sequence classification."""

    def __init__(
        self,
        sequences: list[tuple[str, str]],
        labels: Optional[dict[str, int]],
        tokenizer,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[tuple[str, str, Optional[int]]] = []

        for header, seq in sequences:
            label = labels.get(header) if labels else None
            self.items.append((header, seq, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        header, seq, label = self.items[idx]
        encoding = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "header": header,
        }
        if label is not None:
            item["label"] = torch.tensor(label, dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------

class NucleotideClassifier(nn.Module):
    """Classification head on top of Nucleotide Transformer embeddings.

    Uses mean token embedding pooling, which consistently improves performance
    over CLS-token or other pooling strategies (Nature Comms 2025 benchmark).
    """

    def __init__(self, backbone, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # Mean pooling over non-padding tokens from last hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: NucleotideClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    output_dir: str = "results",
) -> NucleotideClassifier:
    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # -- Train --
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        # -- Validate --
        val_acc = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} -- "
            f"loss: {avg_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.classifier.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info(f"  Saved best model (val_acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    return model


def evaluate(model: NucleotideClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: NucleotideClassifier,
    loader: DataLoader,
    device: torch.device,
    output_path: str,
):
    model.eval()
    headers_all: list[str] = []
    preds_all: list[int] = []
    probs_all: list[list[float]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            headers_all.extend(batch["header"])
            preds_all.extend(predictions.cpu().tolist())
            probs_all.extend(probabilities.cpu().tolist())

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        num_classes = len(probs_all[0]) if probs_all else 0
        header_row = ["sequence_id", "predicted_label"] + [f"prob_class_{i}" for i in range(num_classes)]
        writer.writerow(header_row)
        for h, pred, probs in zip(headers_all, preds_all, probs_all):
            writer.writerow([h, pred] + [f"{p:.6f}" for p in probs])

    logger.info(f"Predictions saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DNA sequence classification with Nucleotide Transformer"
    )
    parser.add_argument(
        "--input", required=True, help="Path to FASTA file or plain-text file with DNA sequences"
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to CSV with columns: sequence_id, label. Required for training.",
    )
    parser.add_argument(
        "--model",
        default="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--task",
        default="classification",
        help="Task name (for logging and output organization)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory (training) or output CSV path (inference)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained classifier head (.pt) for inference-only mode",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for classification head")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for classification head")
    parser.add_argument(
        "--device", default=None, help="Device (cuda/cpu). Auto-detected if omitted."
    )

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")
    logger.info(f"Task: {args.task}")

    # Load tokenizer and backbone
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModelForMaskedLM.from_pretrained(args.model)
    backbone.eval()
    backbone.to(device)

    max_length = tokenizer.model_max_length
    # Determine hidden size from backbone config
    hidden_size = backbone.config.hidden_size

    # Parse input sequences
    sequences = parse_fasta(args.input)
    logger.info(f"Loaded {len(sequences)} sequences from {args.input}")

    # Build classifier
    classifier = NucleotideClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint:
        # --- Inference mode ---
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        classifier.classifier.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
        dataset = DNAClassificationDataset(sequences, labels=None, tokenizer=tokenizer, max_length=max_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        output_path = args.output if args.output.endswith(".csv") else os.path.join(args.output, "predictions.csv")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        predict(classifier, loader, device, output_path)
    else:
        # --- Training mode ---
        if not args.labels:
            logger.error("--labels is required for training. Provide a CSV with sequence_id,label columns.")
            sys.exit(1)

        labels = load_labels(args.labels)
        logger.info(f"Loaded {len(labels)} labels from {args.labels}")

        dataset = DNAClassificationDataset(sequences, labels=labels, tokenizer=tokenizer, max_length=max_length)

        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        logger.info(f"Train: {train_size}, Val: {val_size}, Classes: {args.num_classes}")

        train_model(
            classifier,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            output_dir=args.output,
        )

        # Final evaluation
        final_acc = evaluate(classifier, val_loader, device)
        logger.info(f"Final validation accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()

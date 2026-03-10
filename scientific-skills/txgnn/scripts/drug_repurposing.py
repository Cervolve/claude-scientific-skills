"""TxGNN Drug Repurposing Prediction

TxGNN improves indication prediction by 49.2% over baselines (Nature Medicine 2024).
Predictions are hypotheses -- always validate clinically.

Predicts top-K drug candidates (indications and contraindications) for a given
disease using the TxGNN pretrained graph neural network model. Optionally runs
the TxGNN GraphMask explainer to produce multi-hop interpretable reasoning paths.

Usage:
    python drug_repurposing.py --disease "Alzheimer disease" --top-k 20
    python drug_repurposing.py --disease "Alzheimer disease" --relation contraindication
    python drug_repurposing.py --disease "Alzheimer disease" --explain --output results/
    python drug_repurposing.py --disease-id 9907.0 --top-k 50 --output results/

Requirements:
    conda create --name txgnn_env python=3.8
    conda activate txgnn_env
    pip install TxGNN
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = os.environ.get("TXGNN_DATA", "./data")
DEFAULT_MODEL_CKPT = os.environ.get("TXGNN_MODEL_CKPT", "./model_ckpt")
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ── Model loading ────────────────────────────────────────────────────────────

def load_txgnn_model(
    data_dir: str = DEFAULT_DATA_DIR,
    model_ckpt: str = DEFAULT_MODEL_CKPT,
    device: str = DEFAULT_DEVICE,
    split: str = "full_graph",
    seed: int = 42,
) -> Tuple:
    """Load TxGNN data, model, and evaluator.

    Args:
        data_dir: Path to the TxGNN data folder. Downloaded automatically
            on first use by the TxGNN library.
        model_ckpt: Path to pretrained model checkpoint directory.
        device: Torch device string (e.g. 'cuda:0' or 'cpu').
        split: Data split strategy. Use 'full_graph' for inference with
            pretrained weights (95% train, 5% val, no test masking).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (TxData, TxGNN model, TxEval evaluator).
    """
    from txgnn import TxData, TxGNN, TxEval

    print(f"[TxGNN] Loading data from {data_dir} ...")
    tx_data = TxData(data_folder_path=data_dir)
    tx_data.prepare_split(split=split, seed=seed)

    print(f"[TxGNN] Initializing model on {device} ...")
    model = TxGNN(
        data=tx_data,
        weight_bias_track=False,
        proj_name="TxGNN",
        exp_name="TxGNN_inference",
        device=device,
    )

    model.model_initialize(
        n_hid=100,
        n_inp=100,
        n_out=100,
        proto=True,
        proto_num=3,
        attention=False,
        sim_measure="all_nodes_profile",
        agg_measure="rarity",
        num_walks=200,
        path_length=2,
    )

    print(f"[TxGNN] Loading pretrained weights from {model_ckpt} ...")
    model.load_pretrained(model_ckpt)

    evaluator = TxEval(model=model)

    print("[TxGNN] Model loaded successfully.")
    return tx_data, model, evaluator


# ── Disease lookup ───────────────────────────────────────────────────────────

def resolve_disease(
    tx_data,
    disease_name: Optional[str] = None,
    disease_id: Optional[float] = None,
) -> Tuple[float, str]:
    """Resolve a disease name or ID to (disease_idx, disease_name).

    The TxGNN knowledge graph stores disease nodes with float indices.
    This function performs a fuzzy name search across the graph's disease
    nodes when a name is provided, or validates the ID if given directly.

    Args:
        tx_data: Loaded TxData instance.
        disease_name: Human-readable disease name to search for.
        disease_id: Numeric disease index in the TxGNN graph.

    Returns:
        Tuple of (disease_idx, resolved_disease_name).

    Raises:
        ValueError: If the disease cannot be found.
    """
    # Access the knowledge graph data
    # TxData stores node information in its internal dataframes
    df = tx_data.df  # The edge dataframe underlying the knowledge graph

    # Collect all disease nodes from edge endpoints
    disease_nodes = {}

    # Disease nodes appear as x_id/x_name or y_id/y_name with type 'disease'
    if hasattr(tx_data, "node_info"):
        # Newer TxGNN versions may expose node info directly
        node_info = tx_data.node_info
        for idx, row in node_info.iterrows():
            if row.get("node_type", "") == "disease":
                disease_nodes[float(row["node_index"])] = row.get("node_name", str(idx))
    else:
        # Fall back to extracting from the graph edges
        # TxGNN internally maps disease names to indices; we reconstruct this
        # from the processed data
        if hasattr(tx_data, "disease_idx_to_name"):
            disease_nodes = tx_data.disease_idx_to_name
        elif hasattr(tx_data, "G"):
            # Extract from the DGL heterogeneous graph
            g = tx_data.G
            if "disease" in g.ntypes:
                n_diseases = g.number_of_nodes("disease")
                for i in range(n_diseases):
                    disease_nodes[float(i)] = f"disease_{i}"
            # Try to get actual names from node features or separate mapping
            if hasattr(tx_data, "idx2id_disease"):
                for idx, name in tx_data.idx2id_disease.items():
                    disease_nodes[float(idx)] = name

    if disease_id is not None:
        disease_idx = float(disease_id)
        name = disease_nodes.get(disease_idx, f"disease_{disease_idx}")
        return disease_idx, name

    if disease_name is None:
        raise ValueError("Provide either --disease or --disease-id")

    # Fuzzy match by name
    query_lower = disease_name.lower()
    best_match = None
    best_score = -1

    for idx, name in disease_nodes.items():
        name_lower = name.lower()
        if name_lower == query_lower:
            return idx, name  # Exact match
        if query_lower in name_lower:
            # Prefer shorter names (more specific matches)
            score = len(query_lower) / len(name_lower)
            if score > best_score:
                best_score = score
                best_match = (idx, name)

    if best_match is not None:
        print(f"[TxGNN] Matched disease: '{best_match[1]}' (idx={best_match[0]})")
        return best_match

    # If internal lookup fails, search the raw data files
    data_path = os.path.join(tx_data.data_folder_path, "nodes_diseases.csv")
    if os.path.exists(data_path):
        disease_df = pd.read_csv(data_path)
        name_col = [c for c in disease_df.columns if "name" in c.lower()]
        idx_col = [c for c in disease_df.columns if "idx" in c.lower() or "index" in c.lower()]
        if name_col and idx_col:
            for _, row in disease_df.iterrows():
                if query_lower in str(row[name_col[0]]).lower():
                    return float(row[idx_col[0]]), str(row[name_col[0]])

    raise ValueError(
        f"Disease not found: '{disease_name}'. "
        f"Try a different name or use --disease-id with a known index. "
        f"Found {len(disease_nodes)} diseases in the knowledge graph."
    )


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_drug_candidates(
    evaluator,
    disease_idx: float,
    disease_name: str,
    relation: str = "indication",
    top_k: int = 20,
) -> pd.DataFrame:
    """Predict top-K drug candidates for a disease.

    Uses TxGNN's disease-centric evaluation to score all candidate
    therapeutics against the target disease.

    Args:
        evaluator: Loaded TxEval instance.
        disease_idx: Numeric disease index in the TxGNN graph.
        disease_name: Disease name (for display).
        relation: 'indication' or 'contraindication'.
        top_k: Number of top candidates to return.

    Returns:
        DataFrame with columns: rank, drug_name, drug_idx, score.
    """
    print(f"[TxGNN] Predicting {relation}s for: {disease_name} (idx={disease_idx})")
    print(f"[TxGNN] Scoring all candidate therapeutics ...")

    result = evaluator.eval_disease_centric(
        disease_idxs=[disease_idx],
        relation=relation,
        show_plot=False,
        verbose=True,
        save_result=False,
        return_raw=True,
    )

    # Parse the raw evaluation output
    # TxEval.eval_disease_centric returns a dict with scores per disease
    if isinstance(result, dict):
        # Extract scores for our disease
        disease_key = disease_idx
        if disease_key in result:
            scores = result[disease_key]
        elif str(disease_key) in result:
            scores = result[str(disease_key)]
        else:
            # Result may be structured differently; try the first key
            scores = list(result.values())[0] if result else {}
    elif isinstance(result, pd.DataFrame):
        scores = result
    else:
        scores = result

    # Build ranked output
    if isinstance(scores, pd.DataFrame):
        df = scores.copy()
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False).head(top_k)
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        return df

    if isinstance(scores, dict):
        rows = []
        for drug_key, score_val in scores.items():
            rows.append({
                "drug_idx": drug_key,
                "score": float(score_val) if not isinstance(score_val, (list, dict)) else score_val,
            })
        df = pd.DataFrame(rows)
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False).head(top_k)
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        return df

    # If scores is a numpy array or tensor, build DataFrame from it
    if isinstance(scores, (np.ndarray, torch.Tensor)):
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        rows = []
        for rank, idx in enumerate(top_indices, 1):
            rows.append({
                "rank": rank,
                "drug_idx": int(idx),
                "score": float(scores[idx]),
            })
        return pd.DataFrame(rows).set_index("rank")

    print(f"[TxGNN] Warning: unexpected result type {type(result)}. Returning raw.")
    return pd.DataFrame({"raw_result": [str(result)]})


# ── Explainability ───────────────────────────────────────────────────────────

def run_explainer(
    model,
    relation: str = "indication",
    output_dir: str = "./explanation_output",
    learning_rate: float = 3e-4,
    allowance: float = 0.005,
    epochs_per_layer: int = 3,
    penalty_scaling: float = 1.0,
    valid_per_n: int = 20,
) -> Dict:
    """Train GraphMask explainer and retrieve explanation gates.

    GraphMask identifies which edges in the knowledge graph are most
    important for a given prediction, producing multi-hop reasoning paths.

    Note: Training the explainer can take significant time. If a pretrained
    GraphMask checkpoint exists, use model.load_pretrained_graphmask() instead.

    Args:
        model: Loaded TxGNN model instance.
        relation: 'indication' or 'contraindication'.
        output_dir: Directory to save explanation output.
        learning_rate: GraphMask learning rate.
        allowance: Faithfulness allowance parameter.
        epochs_per_layer: Training epochs per GNN layer.
        penalty_scaling: Sparsity penalty scaling factor.
        valid_per_n: Validation frequency (every N epochs).

    Returns:
        Dict with gate values and explanation paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing GraphMask checkpoint
    graphmask_ckpt = os.path.join(output_dir, "graphmask_model_ckpt")
    if os.path.exists(graphmask_ckpt):
        print(f"[TxGNN] Loading existing GraphMask checkpoint from {graphmask_ckpt}")
        model.load_pretrained_graphmask(graphmask_ckpt)
    else:
        print(f"[TxGNN] Training GraphMask explainer for '{relation}' ...")
        print(f"[TxGNN] This may take a while. Consider using a pretrained checkpoint.")

        model.train_graphmask(
            relation=relation,
            learning_rate=learning_rate,
            allowance=allowance,
            epochs_per_layer=epochs_per_layer,
            penalty_scaling=penalty_scaling,
            valid_per_n=valid_per_n,
        )

        # Save the trained GraphMask model for reuse
        model.save_graphmask_model(graphmask_ckpt)
        print(f"[TxGNN] GraphMask model saved to {graphmask_ckpt}")

    # Retrieve and save explanation gates
    print(f"[TxGNN] Retrieving explanation gates ...")
    gates = model.retrieve_save_gates(output_dir)

    # The gates are saved to: output_dir/graphmask_output_{relation}.pkl
    gate_file = os.path.join(output_dir, f"graphmask_output_{relation}.pkl")
    if os.path.exists(gate_file):
        with open(gate_file, "rb") as f:
            gate_data = pickle.load(f)
        print(f"[TxGNN] Explanation gates saved to {gate_file}")
        return {"gate_file": gate_file, "gates": gate_data}

    return {"gate_file": None, "gates": gates}


# ── Output formatting ────────────────────────────────────────────────────────

def format_results(
    disease_name: str,
    disease_idx: float,
    relation: str,
    predictions: pd.DataFrame,
    explanation: Optional[Dict] = None,
) -> str:
    """Format prediction results as a human-readable report.

    Args:
        disease_name: Name of the queried disease.
        disease_idx: Disease index in the knowledge graph.
        relation: 'indication' or 'contraindication'.
        predictions: DataFrame of ranked predictions.
        explanation: Optional explanation data from GraphMask.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 70,
        f"TxGNN Drug Repurposing Predictions",
        f"Disease: {disease_name} (idx={disease_idx})",
        f"Relation: {relation}",
        f"Top {len(predictions)} candidates",
        "=" * 70,
        "",
    ]

    lines.append(predictions.to_string())
    lines.append("")

    if explanation and explanation.get("gate_file"):
        lines.append(f"GraphMask explanation saved to: {explanation['gate_file']}")
        lines.append("")

    lines.extend([
        "-" * 70,
        "DISCLAIMER: These predictions are computational hypotheses generated",
        "by TxGNN (Nature Medicine 2024). They require independent clinical",
        "validation before any therapeutic application.",
        "-" * 70,
    ])

    return "\n".join(lines)


def save_results(
    output_dir: str,
    disease_name: str,
    relation: str,
    predictions: pd.DataFrame,
    report: str,
) -> None:
    """Save prediction results to files.

    Args:
        output_dir: Directory to save output files.
        disease_name: Disease name (used in filenames).
        relation: 'indication' or 'contraindication'.
        predictions: DataFrame of ranked predictions.
        report: Formatted text report.
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_name = disease_name.replace(" ", "_").replace("/", "-").lower()

    # Save CSV
    csv_path = os.path.join(output_dir, f"txgnn_{safe_name}_{relation}.csv")
    predictions.to_csv(csv_path)
    print(f"[TxGNN] Predictions saved to {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, f"txgnn_{safe_name}_{relation}.json")
    result_dict = {
        "disease": disease_name,
        "relation": relation,
        "predictions": predictions.reset_index().to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"[TxGNN] JSON results saved to {json_path}")

    # Save report
    report_path = os.path.join(output_dir, f"txgnn_{safe_name}_{relation}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[TxGNN] Report saved to {report_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TxGNN Drug Repurposing Prediction. "
            "Predicts top-K drug candidates for a disease using pretrained "
            "graph neural networks. TxGNN improves indication prediction by "
            "49.2%% over baselines (Nature Medicine 2024). "
            "Predictions are hypotheses -- always validate clinically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Disease specification (one required)
    disease_group = parser.add_mutually_exclusive_group(required=True)
    disease_group.add_argument(
        "--disease",
        type=str,
        help="Disease name to search for (e.g. 'Alzheimer disease', 'cystic fibrosis').",
    )
    disease_group.add_argument(
        "--disease-id",
        type=float,
        help="Numeric disease index in the TxGNN knowledge graph.",
    )

    # Prediction options
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top drug candidates to return (default: 20).",
    )
    parser.add_argument(
        "--relation",
        type=str,
        choices=["indication", "contraindication"],
        default="indication",
        help="Prediction type: 'indication' or 'contraindication' (default: indication).",
    )

    # Explainability
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run GraphMask explainer for multi-hop interpretable reasoning paths.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for saving results (CSV, JSON, report). "
             "If not specified, results are printed to stdout only.",
    )

    # Model configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to TxGNN data folder (default: {DEFAULT_DATA_DIR}). "
             "Data is downloaded automatically on first use.",
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=DEFAULT_MODEL_CKPT,
        help=f"Path to pretrained model checkpoint (default: {DEFAULT_MODEL_CKPT}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Torch device (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="full_graph",
        help="Data split strategy (default: full_graph).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    tx_data, model, evaluator = load_txgnn_model(
        data_dir=args.data_dir,
        model_ckpt=args.model_ckpt,
        device=args.device,
        split=args.split,
        seed=args.seed,
    )

    # Resolve disease
    disease_idx, disease_name = resolve_disease(
        tx_data,
        disease_name=args.disease,
        disease_id=args.disease_id,
    )

    # Run predictions
    predictions = predict_drug_candidates(
        evaluator=evaluator,
        disease_idx=disease_idx,
        disease_name=disease_name,
        relation=args.relation,
        top_k=args.top_k,
    )

    # Optionally run explainer
    explanation = None
    if args.explain:
        explain_dir = args.output if args.output else "./explanation_output"
        explanation = run_explainer(
            model=model,
            relation=args.relation,
            output_dir=explain_dir,
        )

    # Format and display results
    report = format_results(
        disease_name=disease_name,
        disease_idx=disease_idx,
        relation=args.relation,
        predictions=predictions,
        explanation=explanation,
    )
    print(report)

    # Save if output directory specified
    if args.output:
        save_results(
            output_dir=args.output,
            disease_name=disease_name,
            relation=args.relation,
            predictions=predictions,
            report=report,
        )


if __name__ == "__main__":
    main()

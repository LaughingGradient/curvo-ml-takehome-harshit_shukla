"""
Training script for Sentence-Transformer + MLP multi-label classifier.

Usage:
    python -m approaches.task1.sbert_mlp_train \
        --train_path data/labeled_train.jsonl \
        --dev_path   data/labeled_dev.jsonl

Outputs saved to approaches/task1/sbert_mlp_artifacts/:
    model.pt              - trained MLP weights
    thresholds.json       - per-class thresholds tuned on dev
    config.json           - model config (embedding dim, hidden dim, labels)
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LABELS = ["OBJECTION", "NEXT_STEP", "UNCERTAINTY", "POSITIVE_SIGNAL", "QUESTION"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "sbert_mlp_artifacts")
ENCODER_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _encode(
    examples: List[Dict[str, Any]],
    encoder: SentenceTransformer,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, D], labels [N, C])."""
    texts = [ex["text"] for ex in examples]
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=64)

    label_matrix = np.zeros((len(examples), len(LABELS)), dtype=np.float32)
    for i, ex in enumerate(examples):
        for lab in ex.get("labels", []):
            if lab in LABEL_TO_IDX:
                label_matrix[i, LABEL_TO_IDX[lab]] = 1.0

    return np.array(embeddings, dtype=np.float32), label_matrix


# ---------------------------------------------------------------------------
# MLP definition
# ---------------------------------------------------------------------------

class EventMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_labels: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    dev_X: np.ndarray,
    dev_Y: np.ndarray,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
) -> EventMLP:
    input_dim = train_X.shape[1]
    model = EventMLP(input_dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    dev_Xt = torch.from_numpy(dev_X)
    dev_Yt = torch.from_numpy(dev_Y)

    best_dev_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            dev_logits = model(dev_Xt)
            dev_loss = criterion(dev_logits, dev_Yt).item()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d  train_loss=%.4f  dev_loss=%.4f", epoch, epoch_loss, dev_loss
            )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    model.load_state_dict(best_state)
    logger.info("Best dev loss: %.4f", best_dev_loss)
    return model


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def tune_thresholds(
    model: EventMLP,
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()

    thresholds: Dict[str, float] = {}
    candidates = [i / 20.0 for i in range(1, 20)]

    for idx, label in enumerate(LABELS):
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            tp = fp = fn = 0
            for i in range(len(Y)):
                predicted = probs[i, idx] >= t
                actual = Y[i, idx] > 0.5
                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and actual:
                    fn += 1
            f1 = _f1(tp, fp, fn)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[label] = best_t
        logger.info("  %-18s threshold=%.2f  dev_F1=%.4f", label, best_t, best_f1)

    return thresholds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train SBERT+MLP event classifier")
    ap.add_argument("--train_path", required=True, help="Path to labeled_train.jsonl")
    ap.add_argument("--dev_path", required=True, help="Path to labeled_dev.jsonl")
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--encoder", default=ENCODER_NAME)
    args = ap.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # --- Encode ---
    logger.info("Loading encoder: %s", args.encoder)
    encoder = SentenceTransformer(args.encoder)
    emb_dim = encoder.get_sentence_embedding_dimension()

    logger.info("Encoding train set...")
    train_data = _load_jsonl(args.train_path)
    train_X, train_Y = _encode(train_data, encoder)
    logger.info("Train: %d samples, %d-dim embeddings", *train_X.shape)

    logger.info("Encoding dev set...")
    dev_data = _load_jsonl(args.dev_path)
    dev_X, dev_Y = _encode(dev_data, encoder)
    logger.info("Dev: %d samples, %d-dim embeddings", *dev_X.shape)

    # --- Train ---
    logger.info("Training MLP (hidden=%d, lr=%s, patience=%d)...", args.hidden_dim, args.lr, args.patience)
    model = train_model(
        train_X, train_Y, dev_X, dev_Y,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # --- Tune thresholds on dev ---
    logger.info("Tuning per-class thresholds on dev set...")
    thresholds = tune_thresholds(model, dev_X, dev_Y)

    # --- Save artifacts ---
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "model.pt"))
    logger.info("Saved model → %s/model.pt", ARTIFACT_DIR)

    with open(os.path.join(ARTIFACT_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Saved thresholds → %s/thresholds.json", ARTIFACT_DIR)

    config = {
        "encoder": args.encoder,
        "embedding_dim": emb_dim,
        "hidden_dim": args.hidden_dim,
        "num_labels": len(LABELS),
        "labels": LABELS,
    }
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config → %s/config.json", ARTIFACT_DIR)

    logger.info("Done. Artifacts in %s", ARTIFACT_DIR)


if __name__ == "__main__":
    main()

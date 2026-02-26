"""
Inference module for Sentence-Transformer + MLP multi-label classifier.

Loads pre-trained artifacts from approaches/task1/sbert_mlp_artifacts/
and exposes tag() matching the standard Task 1 interface.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from approaches.task1.sbert_mlp_train import ARTIFACT_DIR, EventMLP, LABELS

logger = logging.getLogger(__name__)


def _load_artifacts():
    """Load config, model weights, and thresholds from disk."""
    config_path = os.path.join(ARTIFACT_DIR, "config.json")
    model_path = os.path.join(ARTIFACT_DIR, "model.pt")
    thresh_path = os.path.join(ARTIFACT_DIR, "thresholds.json")

    for p in (config_path, model_path, thresh_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Missing artifact: {p}. Run sbert_mlp_train.py first."
            )

    with open(config_path) as f:
        config = json.load(f)

    model = EventMLP(
        input_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_labels=config["num_labels"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with open(thresh_path) as f:
        thresholds = json.load(f)

    encoder = SentenceTransformer(config["encoder"])

    logger.info(
        "Loaded SBERT+MLP artifacts (encoder=%s, hidden=%d, thresholds=%s)",
        config["encoder"], config["hidden_dim"], thresholds,
    )
    return encoder, model, thresholds


def tag(examples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Classify utterances using pre-trained SBERT+MLP.

    Returns conversation-level predictions (no retrieved_cards).
    """
    encoder, model, thresholds = _load_artifacts()

    texts = [ex["text"] for ex in examples]
    embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(embeddings))
        probs = torch.sigmoid(logits).numpy()

    convs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for i, ex in enumerate(examples):
        convs[str(ex["conversation_id"])].append((ex, probs[i]))

    results: List[Dict[str, Any]] = []
    for conv_id in sorted(convs.keys()):
        items = sorted(convs[conv_id], key=lambda x: int(x[0]["utterance_index"]))
        events: List[Dict[str, Any]] = []

        for ex, prob_vec in items:
            idx = int(ex["utterance_index"])
            text = ex["text"]
            for label_idx, label in enumerate(LABELS):
                if prob_vec[label_idx] >= thresholds.get(label, 0.5):
                    events.append({
                        "event_type": label,
                        "utterance_index": idx,
                        "text": text,
                    })

        events.sort(key=lambda e: (e["utterance_index"], e["event_type"]))
        results.append({"conversation_id": conv_id, "events": events})

    return results

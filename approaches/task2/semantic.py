"""Semantic retrieval for Task 2 using sentence-transformer cosine similarity."""

import logging
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from approaches.task2 import build_query
from src.baseline import _read_playbook_cards

logger = logging.getLogger(__name__)

_ENCODER_NAME = "all-MiniLM-L6-v2"


def _build_semantic_retriever(cards: List[Dict[str, str]]):
    """
    Build a semantic retriever: encode card texts once, return a callable
    that encodes a query and ranks cards by cosine similarity.
    """
    if not cards:
        return lambda query, k: []

    logger.info("Loading sentence-transformer: %s", _ENCODER_NAME)
    encoder = SentenceTransformer(_ENCODER_NAME)

    card_ids = [c["card_id"] for c in cards]
    card_texts = [c["text"] for c in cards]

    logger.info("Encoding %d playbook cards", len(cards))
    card_embeddings = encoder.encode(card_texts, normalize_embeddings=True)

    def _retrieve(query: str, k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        q_emb = encoder.encode([query], normalize_embeddings=True)[0]
        scores = card_embeddings @ q_emb
        ranked_idx = np.argsort(-scores)[:k]
        return [
            {"card_id": card_ids[i], "score": round(float(scores[i]), 4)}
            for i in ranked_idx
        ]

    return _retrieve


def retrieve(
    predictions: List[Dict[str, Any]],
    playbook_dir: str,
    k: int = 3,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Attach top-k playbook cards to every event using semantic similarity.

    Mutates and returns *predictions* in-place.
    """
    cards = _read_playbook_cards(playbook_dir)
    retriever = _build_semantic_retriever(cards)

    for conv in predictions:
        for event in conv.get("events", []):
            query = build_query(event["event_type"], event["text"])
            event["retrieved_cards"] = retriever(query, k)

    return predictions

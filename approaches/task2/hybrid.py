"""Hybrid retrieval for Task 2: BM25 + Semantic via Reciprocal Rank Fusion."""

import logging
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from approaches.task2 import build_query
from src.baseline import _build_retriever, _read_playbook_cards

logger = logging.getLogger(__name__)

_ENCODER_NAME = "all-MiniLM-L6-v2"
_RRF_K = 60


def _build_hybrid_retriever(cards: List[Dict[str, str]]):
    """
    Build a hybrid retriever that fuses BM25 and semantic rankings via RRF.
    """
    if not cards:
        return lambda query, k: []

    bm25_retrieve = _build_retriever(cards)

    logger.info("Loading sentence-transformer: %s", _ENCODER_NAME)
    encoder = SentenceTransformer(_ENCODER_NAME)

    card_ids = [c["card_id"] for c in cards]
    card_texts = [c["text"] for c in cards]
    n_cards = len(cards)

    logger.info("Encoding %d playbook cards", n_cards)
    card_embeddings = encoder.encode(card_texts, normalize_embeddings=True)

    def _retrieve(query: str, k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        # BM25 ranked list (retrieve all cards so we have full ranking)
        bm25_results = bm25_retrieve(query, n_cards)
        bm25_rank = {r["card_id"]: i + 1 for i, r in enumerate(bm25_results)}

        # Semantic ranked list
        q_emb = encoder.encode([query], normalize_embeddings=True)[0]
        scores = card_embeddings @ q_emb
        sem_order = np.argsort(-scores)
        sem_rank = {card_ids[idx]: i + 1 for i, idx in enumerate(sem_order)}

        # RRF fusion
        rrf_scores = {}
        for cid in card_ids:
            r_bm25 = bm25_rank.get(cid, n_cards + 1)
            r_sem = sem_rank.get(cid, n_cards + 1)
            rrf_scores[cid] = 1.0 / (r_bm25 + _RRF_K) + 1.0 / (r_sem + _RRF_K)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        return [
            {"card_id": cid, "score": round(score, 6)}
            for cid, score in ranked
        ]

    return _retrieve


def retrieve(
    predictions: List[Dict[str, Any]],
    playbook_dir: str,
    k: int = 3,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Attach top-k playbook cards to every event using hybrid BM25+semantic RRF.

    Mutates and returns *predictions* in-place.
    """
    cards = _read_playbook_cards(playbook_dir)
    retriever = _build_hybrid_retriever(cards)

    for conv in predictions:
        for event in conv.get("events", []):
            query = build_query(event["event_type"], event["text"])
            event["retrieved_cards"] = retriever(query, k)

    return predictions

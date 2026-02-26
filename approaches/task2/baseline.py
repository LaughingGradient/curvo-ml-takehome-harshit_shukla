"""Baseline BM25 retrieval for Task 2 (strategy card retrieval)."""

from typing import Any, Dict, List

from approaches.task2 import build_query
from src.baseline import _build_retriever, _read_playbook_cards


def retrieve(
    predictions: List[Dict[str, Any]],
    playbook_dir: str,
    k: int = 3,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Attach top-k playbook cards to every event using BM25 scoring.

    Mutates and returns *predictions* in-place.
    """
    cards = _read_playbook_cards(playbook_dir)
    retriever = _build_retriever(cards)

    for conv in predictions:
        for event in conv.get("events", []):
            query = build_query(event["event_type"], event["text"])
            event["retrieved_cards"] = retriever(query, k)

    return predictions

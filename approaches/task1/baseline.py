"""Baseline heuristic tagger for Task 1 (event detection)."""

from collections import defaultdict
from typing import Any, Dict, List

from src.baseline import _predict_labels


def tag(examples: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    """
    Classify utterances using regex heuristics.

    Returns conversation-level predictions (no retrieved_cards).
    """
    convs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        convs[str(ex["conversation_id"])].append(ex)

    results: List[Dict[str, Any]] = []
    for conv_id in sorted(convs.keys()):
        utts = sorted(convs[conv_id], key=lambda x: int(x["utterance_index"]))
        events: List[Dict[str, Any]] = []
        for utt in utts:
            idx = int(utt["utterance_index"])
            text = utt["text"]
            for label in _predict_labels(text):
                events.append({
                    "event_type": label,
                    "utterance_index": idx,
                    "text": text,
                })
        events.sort(key=lambda e: (e["utterance_index"], e["event_type"]))
        results.append({"conversation_id": conv_id, "events": events})

    return results

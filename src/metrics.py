from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set


LABELS = ["OBJECTION", "NEXT_STEP", "UNCERTAINTY", "POSITIVE_SIGNAL", "QUESTION"]


def _gold_map(gold: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Set[str]]:
    """
    Convert gold JSONL rows into a mapping:
      (conversation_id, utterance_index) -> set(labels)
    """
    out: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
    for g in gold:
        cid = str(g.get("conversation_id", ""))
        idx = int(g.get("utterance_index", 0))
        labs = g.get("labels", []) or []
        for lab in labs:
            out[(cid, idx)].add(lab)
    return out


def _pred_map(preds: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Set[str]]:
    """
    Convert prediction events into the same mapping shape as gold.

    preds is expected to be a list of conversation objects:
      {
        "conversation_id": "...",
        "events": [
            {"event_type": "...", "utterance_index": i, ...}
        ]
      }
    """
    out: Dict[Tuple[str, int], Set[str]] = defaultdict(set)

    for conv in preds:
        cid = str(conv.get("conversation_id", ""))
        for ev in conv.get("events", []) or []:
            idx = int(ev.get("utterance_index", 0))
            lab = ev.get("event_type")
            if lab in LABELS:
                out[(cid, idx)].add(lab)
    return out


def _micro_f1(gold_map: Dict[Tuple[str, int], Set[str]],
              pred_map: Dict[Tuple[str, int], Set[str]]) -> Dict[str, float]:
    """
    Compute micro precision, recall, F1 over multi-label utterances.
    """
    tp = 0
    fp = 0
    fn = 0

    keys = set(gold_map.keys()) | set(pred_map.keys())

    for k in keys:
        g = gold_map.get(k, set())
        p = pred_map.get(k, set())

        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics(gold: List[Dict[str, Any]], preds: List[Dict[str, Any]], k: int = 3) -> Dict[str, Any]:
    """
    Main entry point used by cli.py.

    Returns:
      - event_micro_f1
      - event_precision
      - event_recall
      - recall_at_k (placeholder for now)
      - counts for debugging
    """

    gmap = _gold_map(gold)
    pmap = _pred_map(preds)

    f = _micro_f1(gmap, pmap)

    # Retrieval recall@k cannot be computed yet because gold does not
    # contain target playbook ids. Leave as placeholder.
    return {
        "event_micro_f1": f["f1"],
        "event_precision": f["precision"],
        "event_recall": f["recall"],
        "counts": {
            "tp": f["tp"],
            "fp": f["fp"],
            "fn": f["fn"],
            "gold_utterances": len(gmap),
            "pred_utterances": len(pmap),
        },
        "recall_at_k": None,
    }
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple


# ----------------------------
# Helpers: playbook loading
# ----------------------------

def _read_playbook_cards(playbook_dir: str) -> List[Dict[str, str]]:
    """
    Load strategy/playbook cards from a directory.

    Each card is expected to be a .md/.txt file.
    card_id is derived from the filename (without extension).
    """
    cards: List[Dict[str, str]] = []
    if not playbook_dir or not os.path.isdir(playbook_dir):
        return cards

    for name in sorted(os.listdir(playbook_dir)):
        if not (name.endswith(".md") or name.endswith(".txt")):
            continue
        path = os.path.join(playbook_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
        except Exception:
            continue
        card_id = os.path.splitext(name)[0]
        if txt:
            cards.append({"card_id": card_id, "text": txt})
    return cards


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _score_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """
    Simple lexical overlap score as a dependency-free fallback.
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    q = set(query_tokens)
    d = set(doc_tokens)
    return float(len(q & d)) / float(len(q) + 1e-9)


def _build_retriever(cards: List[Dict[str, str]]):
    """
    Build a retriever index once from the playbook cards.
    Returns a callable (query, k) -> List[Dict] for top-k retrieval.
    """
    if not cards:
        return lambda query, k: []

    card_ids = [c["card_id"] for c in cards]
    card_tokens = [_tokenize(c["text"]) for c in cards]

    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        bm25 = BM25Okapi(card_tokens)

        def _retrieve(query: str, k: int) -> List[Dict[str, Any]]:
            if k <= 0:
                return []
            scores = bm25.get_scores(_tokenize(query))
            ranked = sorted(
                [(card_ids[i], float(scores[i])) for i in range(len(cards))],
                key=lambda x: x[1],
                reverse=True,
            )[:k]
            return [{"card_id": cid, "score": round(score, 4)} for cid, score in ranked]

    except ImportError:
        def _retrieve(query: str, k: int) -> List[Dict[str, Any]]:
            if k <= 0:
                return []
            qt = _tokenize(query)
            scored = [(cid, _score_overlap(qt, ct)) for cid, ct in zip(card_ids, card_tokens)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [{"card_id": cid, "score": round(score, 4)} for cid, score in scored[:k]]

    return _retrieve


# ----------------------------
# Baseline tagger
# ----------------------------

# Keep label strings consistent with README.
LABELS = ["OBJECTION", "NEXT_STEP", "UNCERTAINTY", "POSITIVE_SIGNAL", "QUESTION"]


def _predict_labels(text: str) -> List[str]:
    """
    Very simple heuristic baseline for multi-label event detection.

    The goal is not high accuracy; it is to provide a runnable, reproducible baseline.
    """
    t = (text or "").strip()
    tl = t.lower()
    labels: List[str] = []

    # QUESTION
    if t.endswith("?") or re.search(r"\b(what|why|how|when|where|who|which)\b", tl):
        labels.append("QUESTION")

    # NEXT_STEP
    if re.search(r"\b(let's|lets|how about|shall we|we should|i suggest|why don't we|we can|we could)\b", tl):
        labels.append("NEXT_STEP")

    # UNCERTAINTY
    if re.search(r"\b(not sure|maybe|perhaps|i guess|i think|might|could be|unsure|depends)\b", tl):
        labels.append("UNCERTAINTY")

    # OBJECTION
    if re.search(r"\b(no|not really|don't|do not|can't|cannot|won't|wouldn't|but|however|rather not|i disagree)\b", tl):
        labels.append("OBJECTION")

    # POSITIVE_SIGNAL
    if re.search(r"\b(yes|yeah|yep|sure|sounds good|good idea|great|awesome|love to|works for me|okay|ok)\b", tl):
        labels.append("POSITIVE_SIGNAL")

    # De-dup while preserving order
    seen = set()
    out: List[str] = []
    for lab in labels:
        if lab in LABELS and lab not in seen:
            out.append(lab)
            seen.add(lab)
    return out


# ----------------------------
# Public API used by cli.py
# ----------------------------

def run_pipeline(examples: List[Dict[str, Any]], playbook_dir: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Turn utterance-level examples into conversation-level predictions.

    Input examples (from labeled_*.jsonl) are expected to have:
      - conversation_id
      - utterance_index
      - text

    Output is a list of conversation objects:
      - conversation_id
      - events: list of {event_type, utterance_index, text, retrieval}
      - timing_ms: {tagging, retrieval, total}
    """
    cards = _read_playbook_cards(playbook_dir)
    retrieve = _build_retriever(cards)

    by_conv: Dict[str, Dict[str, Any]] = {}

    for ex in examples:
        conv_id = str(ex.get("conversation_id", ""))
        utt_idx = int(ex.get("utterance_index", 0))
        text = str(ex.get("text", ""))

        if conv_id not in by_conv:
            by_conv[conv_id] = {
                "conversation_id": conv_id,
                "events": [],
                "timing_ms": {"tagging": 0.0, "retrieval": 0.0, "total": 0.0},
            }

        t_tag0 = time.perf_counter()
        pred_labels = _predict_labels(text)
        tag_ms = (time.perf_counter() - t_tag0) * 1000

        t_ret0 = time.perf_counter()
        for lab in pred_labels:
            retrieved_cards = retrieve(f"{lab}: {text}", k)
            by_conv[conv_id]["events"].append(
                {
                    "event_type": lab,
                    "utterance_index": utt_idx,
                    "text": text,
                    "retrieved_cards": retrieved_cards,
                }
            )
        ret_ms = (time.perf_counter() - t_ret0) * 1000

        by_conv[conv_id]["timing_ms"]["tagging"] += tag_ms
        by_conv[conv_id]["timing_ms"]["retrieval"] += ret_ms
        by_conv[conv_id]["timing_ms"]["total"] += (tag_ms + ret_ms)

    preds: List[Dict[str, Any]] = []
    for conv_id in sorted(by_conv.keys()):
        obj = by_conv[conv_id]
        obj["events"].sort(key=lambda e: (e["utterance_index"], e["event_type"]))
        obj["timing_ms"] = {k: round(v, 2) for k, v in obj["timing_ms"].items()}
        preds.append(obj)

    return preds
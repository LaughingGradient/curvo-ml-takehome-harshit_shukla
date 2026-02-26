"""
NLI zero-shot classification for Task 1 (event detection).

Uses facebook/bart-large-mnli via HuggingFace Inference API.
Each utterance is independently scored against descriptive hypotheses
for all five event types. Per-class thresholds are tuned on labeled data.
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

LABELS = ["OBJECTION", "NEXT_STEP", "UNCERTAINTY", "POSITIVE_SIGNAL", "QUESTION"]

DEFAULT_MODEL = "facebook/bart-large-mnli"

LABEL_HYPOTHESES = {
    "QUESTION": "asking a question or seeking information",
    "NEXT_STEP": "proposing, suggesting, or committing to a specific action or plan",
    "UNCERTAINTY": "expressing doubt, hesitation, or uncertainty",
    "OBJECTION": "expressing disagreement, refusal, or resistance",
    "POSITIVE_SIGNAL": "expressing agreement, acceptance, or enthusiasm",
}

HYPOTHESIS_TEMPLATE = "This utterance is {}."

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "nli_thresholds.json")

DEFAULT_THRESHOLDS = {label: 0.5 for label in LABELS}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_utterance(
    client: InferenceClient,
    text: str,
    max_retries: int = 3,
) -> Dict[str, float]:
    """Return {label_name: entailment_score} for a single utterance."""
    hypotheses = list(LABEL_HYPOTHESES.values())
    hypothesis_to_label = {v: k for k, v in LABEL_HYPOTHESES.items()}

    for attempt in range(max_retries):
        try:
            result = client.zero_shot_classification(
                text,
                candidate_labels=hypotheses,
                multi_label=True,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
            )
            return {
                hypothesis_to_label[item.label]: item.score
                for item in result
                if item.label in hypothesis_to_label
            }
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("NLI API attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                logger.error("NLI API failed after %d attempts: %s", max_retries, exc)
                raise
    return {}


def score_all(
    examples: List[Dict[str, Any]],
    hf_token: str,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Score every utterance and return raw scores alongside gold labels.

    Returns list of {"text": ..., "gold": set(...), "scores": {label: float}}.
    """
    client = InferenceClient(model=model or DEFAULT_MODEL, token=hf_token)
    results = []
    for i, ex in enumerate(examples):
        scores = _score_utterance(client, ex["text"])
        results.append({
            "text": ex["text"],
            "gold": set(ex.get("labels", [])),
            "scores": scores,
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(examples):
            logger.info("Scored %d / %d utterances", i + 1, len(examples))
    return results


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def _f1(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def tune_thresholds(
    scored: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Sweep thresholds per class on pre-scored data and pick the one
    that maximises F1. Saves result to THRESHOLDS_PATH.
    """
    thresholds: Dict[str, float] = {}
    candidates = [i / 20.0 for i in range(1, 20)]  # 0.05 … 0.95

    for label in LABELS:
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            tp = fp = fn = 0
            for item in scored:
                predicted = item["scores"].get(label, 0.0) >= t
                actual = label in item["gold"]
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
        logger.info("  %-18s threshold=%.2f  F1=%.4f", label, best_t, best_f1)

    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Saved tuned thresholds → %s", THRESHOLDS_PATH)
    return thresholds


def _load_thresholds() -> Dict[str, float]:
    if os.path.isfile(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH) as f:
            loaded = json.load(f)
        logger.info("Loaded thresholds from %s", THRESHOLDS_PATH)
        return loaded
    logger.warning("No thresholds file found — using defaults (0.5)")
    return DEFAULT_THRESHOLDS.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag(
    examples: List[Dict[str, Any]],
    *,
    hf_token: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Classify utterances via NLI zero-shot with tuned thresholds.

    Returns conversation-level predictions (no retrieved_cards).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF token required. Pass --hf_token or set HF_TOKEN env var.")

    model_id = model or DEFAULT_MODEL
    client = InferenceClient(model=model_id, token=token)
    thresholds = _load_thresholds()
    logger.info("Thresholds: %s", json.dumps(thresholds))

    convs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        convs[str(ex["conversation_id"])].append(ex)

    results: List[Dict[str, Any]] = []
    scored_count = 0

    for conv_id in sorted(convs.keys()):
        utts = sorted(convs[conv_id], key=lambda x: int(x["utterance_index"]))
        events: List[Dict[str, Any]] = []

        for utt in utts:
            idx = int(utt["utterance_index"])
            text = utt["text"]

            try:
                scores = _score_utterance(client, text)
            except Exception:
                logger.exception("Scoring failed for utterance %s:%d", conv_id, idx)
                scores = {}

            scored_count += 1
            if scored_count % 25 == 0:
                logger.info("Scored %d utterances so far", scored_count)

            for label in LABELS:
                if scores.get(label, 0.0) >= thresholds.get(label, 0.5):
                    events.append({
                        "event_type": label,
                        "utterance_index": idx,
                        "text": text,
                    })

        events.sort(key=lambda e: (e["utterance_index"], e["event_type"]))
        results.append({"conversation_id": conv_id, "events": events})

    logger.info("Scored %d utterances total", scored_count)
    return results

from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set


LABELS = ["OBJECTION", "NEXT_STEP", "UNCERTAINTY", "POSITIVE_SIGNAL", "QUESTION"]

EVENT_TO_CARD: Dict[str, str] = {
    "OBJECTION": "handle_objection",
    "NEXT_STEP": "propose_next_step",
    "UNCERTAINTY": "handle_uncertainty",
    "POSITIVE_SIGNAL": "reinforce_positive",
    "QUESTION": "clarify_question",
}


def _gold_map(gold: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Set[str]]:
    """
    (conversation_id, utterance_index) -> set(labels).
    Includes utterances with empty labels as empty sets.
    """
    out: Dict[Tuple[str, int], Set[str]] = {}
    for g in gold:
        cid = str(g.get("conversation_id", ""))
        idx = int(g.get("utterance_index", 0))
        key = (cid, idx)
        if key not in out:
            out[key] = set()
        for lab in (g.get("labels", []) or []):
            out[key].add(lab)
    return out


def _pred_map(preds: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Set[str]]:
    """
    Convert prediction objects into the same mapping shape as gold.
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


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


# ── Micro P / R / F1 ────────────────────────────────────────────────

def _micro_prf(
    gold_map: Dict[Tuple[str, int], Set[str]],
    pred_map: Dict[Tuple[str, int], Set[str]],
) -> Dict[str, Any]:
    tp = fp = fn = 0
    keys = set(gold_map.keys()) | set(pred_map.keys())
    for k in keys:
        g = gold_map.get(k, set())
        p = pred_map.get(k, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Per-class P / R / F1 + support ──────────────────────────────────

def _per_class_prf(
    gold_map: Dict[Tuple[str, int], Set[str]],
    pred_map: Dict[Tuple[str, int], Set[str]],
) -> Dict[str, Dict[str, Any]]:
    keys = set(gold_map.keys()) | set(pred_map.keys())

    per_class: Dict[str, Dict[str, int]] = {
        lab: {"tp": 0, "fp": 0, "fn": 0} for lab in LABELS
    }

    for k in keys:
        g = gold_map.get(k, set())
        p = pred_map.get(k, set())
        for lab in LABELS:
            in_g = lab in g
            in_p = lab in p
            if in_g and in_p:
                per_class[lab]["tp"] += 1
            elif in_p and not in_g:
                per_class[lab]["fp"] += 1
            elif in_g and not in_p:
                per_class[lab]["fn"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for lab in LABELS:
        c = per_class[lab]
        prec = _safe_div(c["tp"], c["tp"] + c["fp"])
        rec = _safe_div(c["tp"], c["tp"] + c["fn"])
        support = c["tp"] + c["fn"]
        out[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(_f1(prec, rec), 4),
            "support": support,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }
    return out


# ── Macro F1 ─────────────────────────────────────────────────────────

def _macro_f1(per_class: Dict[str, Dict[str, Any]]) -> float:
    f1s = [v["f1"] for v in per_class.values()]
    return round(_safe_div(sum(f1s), len(f1s)), 4) if f1s else 0.0


# ── Exact-match ratio ────────────────────────────────────────────────

def _exact_match(
    gold_map: Dict[Tuple[str, int], Set[str]],
    pred_map: Dict[Tuple[str, int], Set[str]],
    all_utterance_keys: Set[Tuple[str, int]],
) -> float:
    """Fraction of utterances where the predicted label set == gold label set."""
    match = 0
    for k in all_utterance_keys:
        g = gold_map.get(k, set())
        p = pred_map.get(k, set())
        if g == p:
            match += 1
    return round(_safe_div(match, len(all_utterance_keys)), 4)


# ── Confusion pairs ──────────────────────────────────────────────────

def _confusion_pairs(
    gold_map: Dict[Tuple[str, int], Set[str]],
    pred_map: Dict[Tuple[str, int], Set[str]],
) -> Dict[str, int]:
    """
    Count the most common (gold_label -> predicted_label) mismatches.
    FP: predicted_label present but not in gold  (appears as "NONE->label")
    FN: gold_label present but not predicted     (appears as "label->NONE")
    """
    pairs: Dict[str, int] = defaultdict(int)
    keys = set(gold_map.keys()) | set(pred_map.keys())
    for k in keys:
        g = gold_map.get(k, set())
        p = pred_map.get(k, set())
        for lab in p - g:
            pairs[f"NONE->{lab}"] += 1
        for lab in g - p:
            pairs[f"{lab}->NONE"] += 1
    return dict(sorted(pairs.items(), key=lambda x: -x[1]))


# ── Retrieval: Recall@k and MRR@k (TP-only) ────────────────────────

def _pred_retrieval_map(
    preds: List[Dict[str, Any]],
) -> Dict[Tuple[str, int, str], List[str]]:
    """
    (conversation_id, utterance_index, event_type) -> [card_id, ...] in rank order.
    """
    out: Dict[Tuple[str, int, str], List[str]] = {}
    for conv in preds:
        cid = str(conv.get("conversation_id", ""))
        for ev in conv.get("events", []) or []:
            idx = int(ev.get("utterance_index", 0))
            lab = ev.get("event_type", "")
            cards = [c["card_id"] for c in (ev.get("retrieved_cards") or [])]
            out[(cid, idx, lab)] = cards
    return out


def _retrieval_metrics(
    gold_map: Dict[Tuple[str, int], Set[str]],
    pred_map: Dict[Tuple[str, int], Set[str]],
    retrieval_map: Dict[Tuple[str, int, str], List[str]],
    k: int,
) -> Dict[str, Any]:
    """
    Compute Recall@k and MRR@k on True Positive events only.

    A TP event is a (conversation_id, utterance_index, label) triple where
    the label appears in both gold and predicted sets.
    """
    hits = 0
    rr_sum = 0.0
    tp_total = 0
    per_class_stats: Dict[str, Dict[str, Any]] = {
        lab: {"hits": 0, "rr_sum": 0.0, "total": 0} for lab in LABELS
    }

    keys = set(gold_map.keys()) | set(pred_map.keys())
    for key in keys:
        g = gold_map.get(key, set())
        p = pred_map.get(key, set())
        tp_labels = g & p
        cid, idx = key

        for lab in tp_labels:
            tp_total += 1
            per_class_stats[lab]["total"] += 1

            correct_card = EVENT_TO_CARD.get(lab)
            if correct_card is None:
                continue

            retrieved = retrieval_map.get((cid, idx, lab), [])[:k]
            if correct_card in retrieved:
                hits += 1
                rank = retrieved.index(correct_card) + 1
                rr_sum += 1.0 / rank
                per_class_stats[lab]["hits"] += 1
                per_class_stats[lab]["rr_sum"] += 1.0 / rank

    recall_at_k = round(_safe_div(hits, tp_total), 4)
    mrr_at_k = round(_safe_div(rr_sum, tp_total), 4)

    per_class_retrieval: Dict[str, Dict[str, Any]] = {}
    for lab in LABELS:
        s = per_class_stats[lab]
        per_class_retrieval[lab] = {
            "recall_at_k": round(_safe_div(s["hits"], s["total"]), 4),
            "mrr_at_k": round(_safe_div(s["rr_sum"], s["total"]), 4),
            "tp_events": s["total"],
            "hits": s["hits"],
        }

    return {
        "recall_at_k": recall_at_k,
        "mrr_at_k": mrr_at_k,
        "tp_events_evaluated": tp_total,
        "hits": hits,
        "per_class": per_class_retrieval,
    }


# ── Public API ───────────────────────────────────────────────────────

def compute_metrics(
    gold: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    k: int = 3,
) -> Dict[str, Any]:
    gmap = _gold_map(gold)
    pmap = _pred_map(preds)
    rmap = _pred_retrieval_map(preds)

    all_keys = set(gmap.keys())
    no_event_keys = {k for k, v in gmap.items() if len(v) == 0}

    micro = _micro_prf(gmap, pmap)
    per_class = _per_class_prf(gmap, pmap)
    macro = _macro_f1(per_class)
    exact = _exact_match(gmap, pmap, all_keys)
    confusion = _confusion_pairs(gmap, pmap)
    retrieval = _retrieval_metrics(gmap, pmap, rmap, k)

    return {
        "micro": micro,
        "macro_f1": macro,
        "per_class": per_class,
        "exact_match_ratio": exact,
        "confusion_pairs": confusion,
        "counts": {
            "total_utterances": len(all_keys),
            "no_event_utterances": len(no_event_keys),
            "gold_utterances_with_labels": len(all_keys) - len(no_event_keys),
            "pred_utterances_with_labels": len(pmap),
        },
        "retrieval": retrieval,
    }

"""Shared utilities for Task 2 retrieval approaches."""

EVENT_DESCRIPTIONS = {
    "OBJECTION": "disagreement, refusal, pushback, resistance",
    "NEXT_STEP": "proposal, suggestion, plan, action",
    "UNCERTAINTY": "hesitation, doubt, unsure, risk",
    "POSITIVE_SIGNAL": "agreement, enthusiasm, alignment, approval",
    "QUESTION": "question, asking, information seeking, clarification",
}


def build_query(event_type: str, text: str, enrich: bool = True) -> str:
    if enrich:
        desc = EVENT_DESCRIPTIONS.get(event_type, "")
        if desc:
            return f"{event_type} ({desc}): {text}"
    return f"{event_type}: {text}"

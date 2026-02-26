"""
LLM few-shot approach for multi-label conversational event detection.

Uses the HuggingFace Inference API to classify utterances into event types
via a carefully crafted system prompt and curated few-shot examples.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

VALID_LABELS: Set[str] = {
    "OBJECTION", "NEXT_STEP", "UNCERTAINTY", "POSITIVE_SIGNAL", "QUESTION",
}

DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert conversational analyst. Your task is to detect communicative \
events in each utterance of a multi-turn dialogue.

For every utterance assign **zero or more** of the five labels below.

─────────────────────────────────────────────
LABEL DEFINITIONS
─────────────────────────────────────────────

1. QUESTION
   The utterance seeks information, clarification, or confirmation.
   • Direct questions ending with "?"
   • Indirect requests for information ("Tell me what happened")
   • A question that ALSO proposes an action → label QUESTION + NEXT_STEP

2. NEXT_STEP
   The utterance proposes, suggests, or commits to a concrete action, plan,
   or timeline. This is the broadest label — watch for it carefully.
   • Explicit proposals: "Let's …", "How about …", "Shall we …"
   • Requests phrased as questions: "Can you review this?", "Could you pick
     up the package?" → these are QUESTION + NEXT_STEP
   • Commitments to act: "I will send it now", "Okay, I'll start on that"
   • Scheduling / timing suggestions: "Maybe tomorrow morning",
     "In half an hour" (proposing a when = proposing a step)
   • Agreeing AND committing: "Okay, let's do that" → POSITIVE_SIGNAL + NEXT_STEP
   ⚠ If an utterance moves the conversation toward a specific action it is
     NEXT_STEP, even without words like "let's" or "shall we".

3. UNCERTAINTY
   The utterance expresses doubt, hesitation, hedging, or perceived risk.
   • Hedging words: "maybe", "perhaps", "I think", "I guess", "might"
   • Doubt: "I'm not sure", "not completely sure", "depends"
   • Risk: "seems too risky", "worried about delays"
   • "I think we should X" → UNCERTAINTY (hedge) + NEXT_STEP (proposal)
   • "That could work" → UNCERTAINTY (hedge) + sometimes POSITIVE_SIGNAL

4. OBJECTION
   The utterance expresses disagreement, refusal, resistance, or raises a
   blocker against the current direction.
   • Refusals: "No", "I can't", "Not today", "I would rather not"
   • Disagreement: "I disagree", "I don't think so"
   • Blockers / concerns: "We still have failing tests",
     "We are missing a few details", "The timeline is too aggressive"
   ⚠ Raising a concrete problem or concern IS an objection even when the
     speaker does not say "no" explicitly.

5. POSITIVE_SIGNAL
   The utterance expresses agreement, acceptance, enthusiasm, or alignment.
   • Agreement: "Yes", "Sure", "Okay", "Sounds good", "That works"
   • Enthusiasm: "Great!", "Awesome", "I'd love to"
   • Acceptance + action: "Alright, I will do X" → POSITIVE_SIGNAL + NEXT_STEP

─────────────────────────────────────────────
MULTI-LABEL GUIDELINES
─────────────────────────────────────────────
• An utterance can carry 0, 1, 2, or even 3 labels at once.
• Neutral or purely informational statements (e.g. "I tried calling
  earlier but got no answer.") get an empty list [].
• Common combos:
  – Request-as-question  → QUESTION + NEXT_STEP
  – Hedged proposal      → UNCERTAINTY + NEXT_STEP
  – Reluctant refusal    → OBJECTION + UNCERTAINTY
  – Agreeing to act      → POSITIVE_SIGNAL + NEXT_STEP
  – Tentative acceptance → POSITIVE_SIGNAL + UNCERTAINTY

─────────────────────────────────────────────
OUTPUT FORMAT  (strict)
─────────────────────────────────────────────
Return **only** a JSON array — one object per utterance:

[
  {"utterance_index": 0, "labels": ["LABEL_A"]},
  {"utterance_index": 1, "labels": []},
  ...
]

• Use ONLY these strings: OBJECTION, NEXT_STEP, UNCERTAINTY, POSITIVE_SIGNAL, QUESTION
• Do NOT add any text, explanation, or markdown outside the JSON array.\
"""

# ---------------------------------------------------------------------------
# Few-shot examples (curated from training data)
# ---------------------------------------------------------------------------

_FEW_SHOT_CONVERSATIONS: List[Tuple[str, str]] = [
    # ── Example 1: train_001 ──────────────────────────────────────────
    # Covers: empty labels, single label, QUESTION+NEXT_STEP,
    #         OBJECTION+UNCERTAINTY, UNCERTAINTY+NEXT_STEP, POSITIVE_SIGNAL+NEXT_STEP
    (
        'Conversation:\n'
        '[0] Speaker A: "Hey, do you have a minute to chat?"\n'
        '[1] Speaker B: "Sure, what\'s up?"\n'
        '[2] Speaker A: "Could we move our meeting to tomorrow morning?"\n'
        '[3] Speaker B: "I am not sure, my morning is packed."\n'
        '[4] Speaker A: "What time works better for you?"\n'
        '[5] Speaker B: "Maybe late afternoon, around four."\n'
        '[6] Speaker A: "Great, let\'s do four then."',
        # ---
        '[\n'
        '  {"utterance_index": 0, "labels": []},\n'
        '  {"utterance_index": 1, "labels": ["POSITIVE_SIGNAL"]},\n'
        '  {"utterance_index": 2, "labels": ["NEXT_STEP", "QUESTION"]},\n'
        '  {"utterance_index": 3, "labels": ["OBJECTION", "UNCERTAINTY"]},\n'
        '  {"utterance_index": 4, "labels": ["QUESTION"]},\n'
        '  {"utterance_index": 5, "labels": ["UNCERTAINTY", "NEXT_STEP"]},\n'
        '  {"utterance_index": 6, "labels": ["POSITIVE_SIGNAL", "NEXT_STEP"]}\n'
        ']'
    ),
    # ── Example 2: train_006 ──────────────────────────────────────────
    # Covers: UNCERTAINTY+NEXT_STEP without "let's", plain OBJECTION,
    #         UNCERTAINTY+OBJECTION (risk-based), QUESTION+NEXT_STEP (proposal)
    (
        'Conversation:\n'
        '[0] Speaker A: "I think we should change the approach."\n'
        '[1] Speaker B: "Why do you think that?"\n'
        '[2] Speaker A: "The current plan might be too risky."\n'
        '[3] Speaker B: "I don\'t agree, it is manageable."\n'
        '[4] Speaker A: "What if we do a smaller pilot first?"\n'
        '[5] Speaker B: "That sounds reasonable."\n'
        '[6] Speaker A: "Great, let\'s set up a pilot next week."',
        # ---
        '[\n'
        '  {"utterance_index": 0, "labels": ["UNCERTAINTY", "NEXT_STEP"]},\n'
        '  {"utterance_index": 1, "labels": ["QUESTION"]},\n'
        '  {"utterance_index": 2, "labels": ["UNCERTAINTY", "OBJECTION"]},\n'
        '  {"utterance_index": 3, "labels": ["OBJECTION"]},\n'
        '  {"utterance_index": 4, "labels": ["QUESTION", "NEXT_STEP"]},\n'
        '  {"utterance_index": 5, "labels": ["POSITIVE_SIGNAL"]},\n'
        '  {"utterance_index": 6, "labels": ["POSITIVE_SIGNAL", "NEXT_STEP"]}\n'
        ']'
    ),
    # ── Example 3: train_009 ──────────────────────────────────────────
    # Covers: QUESTION+NEXT_STEP (shipping request), OBJECTION via blocker,
    #         plain QUESTION, plain UNCERTAINTY, plain NEXT_STEP,
    #         POSITIVE_SIGNAL+NEXT_STEP (committing)
    (
        'Conversation:\n'
        '[0] Speaker A: "Can we ship this today?"\n'
        '[1] Speaker B: "Not really, we still have failing tests."\n'
        '[2] Speaker A: "How many tests are failing?"\n'
        '[3] Speaker B: "Maybe five or six, I need to confirm."\n'
        '[4] Speaker A: "Let\'s fix the top two first."\n'
        '[5] Speaker B: "Okay, I will start now."',
        # ---
        '[\n'
        '  {"utterance_index": 0, "labels": ["QUESTION", "NEXT_STEP"]},\n'
        '  {"utterance_index": 1, "labels": ["OBJECTION"]},\n'
        '  {"utterance_index": 2, "labels": ["QUESTION"]},\n'
        '  {"utterance_index": 3, "labels": ["UNCERTAINTY"]},\n'
        '  {"utterance_index": 4, "labels": ["NEXT_STEP"]},\n'
        '  {"utterance_index": 5, "labels": ["POSITIVE_SIGNAL", "NEXT_STEP"]}\n'
        ']'
    ),
]


def _build_messages_prefix() -> List[Dict[str, str]]:
    """System prompt + few-shot turns."""
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, assistant_text in _FEW_SHOT_CONVERSATIONS:
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": assistant_text})
    return msgs


# ---------------------------------------------------------------------------
# Conversation grouping & formatting
# ---------------------------------------------------------------------------

def _group_by_conversation(
    examples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    convs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        convs[str(ex["conversation_id"])].append(ex)
    for cid in convs:
        convs[cid].sort(key=lambda x: int(x["utterance_index"]))
    return dict(convs)


def _format_conversation(utterances: List[Dict[str, Any]]) -> str:
    lines = ["Conversation:"]
    for utt in utterances:
        idx = int(utt["utterance_index"])
        speaker = "Speaker A" if idx % 2 == 0 else "Speaker B"
        lines.append(f'[{idx}] {speaker}: "{utt["text"]}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(
    text: str,
    utterances: List[Dict[str, Any]],
) -> Dict[int, List[str]]:
    """Extract per-utterance labels from the LLM JSON response."""
    text = text.strip()

    # Locate the JSON array in the response (ignore any surrounding text)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in LLM response; defaulting to empty.")
        return {int(u["utterance_index"]): [] for u in utterances}

    raw = match.group()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raw = raw.replace("'", '"')
        raw = re.sub(r",\s*]", "]", raw)
        raw = re.sub(r",\s*}", "}", raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed after cleanup; defaulting to empty.")
            return {int(u["utterance_index"]): [] for u in utterances}

    result: Dict[int, List[str]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("utterance_index", -1))
            labels = [l for l in item.get("labels", []) if l in VALID_LABELS]
            result[idx] = labels

    for utt in utterances:
        idx = int(utt["utterance_index"])
        if idx not in result:
            result[idx] = []

    return result


# ---------------------------------------------------------------------------
# HF Inference API call
# ---------------------------------------------------------------------------

def _call_hf_api(
    client: InferenceClient,
    messages: List[Dict[str, str]],
    max_retries: int = 3,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "HF API attempt %d failed: %s — retrying in %ds",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("HF API failed after %d attempts: %s", max_retries, exc)
                raise
    return ""


# ---------------------------------------------------------------------------
# Per-conversation prediction
# ---------------------------------------------------------------------------

def _predict_conversation(
    client: InferenceClient,
    utterances: List[Dict[str, Any]],
    prefix: List[Dict[str, str]],
) -> Dict[int, List[str]]:
    user_prompt = _format_conversation(utterances)
    messages = prefix + [{"role": "user", "content": user_prompt}]
    response = _call_hf_api(client, messages)
    return _parse_llm_response(response, utterances)


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
    Classify utterances via LLM few-shot prompting.

    Returns conversation-level predictions (no retrieved_cards).
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF token required. Pass --hf_token or set the HF_TOKEN env var."
        )

    model_id = model or DEFAULT_MODEL
    client = InferenceClient(model=model_id, token=token)
    prefix = _build_messages_prefix()

    convs = _group_by_conversation(examples)
    results: List[Dict[str, Any]] = []

    for conv_id in sorted(convs.keys()):
        utts = convs[conv_id]
        logger.info("Tagging conversation %s (%d utterances)", conv_id, len(utts))

        try:
            preds = _predict_conversation(client, utts, prefix)
        except Exception:
            logger.exception("LLM failed on %s — falling back to empty labels", conv_id)
            preds = {int(u["utterance_index"]): [] for u in utts}

        events: List[Dict[str, Any]] = []
        for utt in utts:
            idx = int(utt["utterance_index"])
            text = utt["text"]
            for label in preds.get(idx, []):
                events.append({
                    "event_type": label,
                    "utterance_index": idx,
                    "text": text,
                })

        events.sort(key=lambda e: (e["utterance_index"], e["event_type"]))
        results.append({"conversation_id": conv_id, "events": events})

    return results

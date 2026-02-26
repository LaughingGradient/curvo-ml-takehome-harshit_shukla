# Report: Conversation Event Detection and Strategy Retrieval

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Overview](#1-overview) | Problem definition, event types, playbook cards, pipeline summary |
| 2 | [Evaluation Metrics](#2-evaluation-metrics) | How we measure Task 1 (classification) and Task 2 (retrieval), changes to original scaffold |
| 3 | [Task 1: Event Detection](#3-task-1-event-detection) | Four approaches (baseline → NLI → SBERT+MLP → LLM), results, what improved |
| 4 | [Task 2: Strategy Card Retrieval](#4-task-2-strategy-card-retrieval) | Evaluation design, three retrievers (BM25 → semantic → hybrid), query expansion, results |
| 5 | [Best Configurations & Runtime](#5-best-configurations--runtime) | Recommended setups, speed vs accuracy tradeoffs |
| 6 | [Failure Cases & Limitations](#6-failure-cases--limitations) | Error analysis for both tasks, dataset limitations |
| 7 | [Code Organization](#7-code-organization) | Repository structure, file descriptions |
| 8 | [Reproducibility](#8-reproducibility) | Setup, CLI usage, how to run each configuration, output format |

---

## 1. Overview

### Problem

Given multi-turn conversations, the pipeline must:
1. **Task 1 — Event Detection:** Classify each utterance with zero or more event labels (multi-label classification).
2. **Task 2 — Strategy Retrieval:** For each detected event, retrieve the most relevant playbook card from a set of coaching strategies.

### Event Types

| Event Type | Meaning | Example |
|---|---|---|
| OBJECTION | Disagreement, refusal, pushback | *"I would rather not, I still have doubts."* |
| NEXT_STEP | Proposal, suggestion, plan | *"Let's schedule a follow-up tomorrow."* |
| UNCERTAINTY | Hesitation, doubt, risk | *"Maybe, I need to think about it."* |
| POSITIVE_SIGNAL | Agreement, enthusiasm, alignment | *"Yes, that sounds great."* |
| QUESTION | Information seeking | *"What time works better?"* |

A single utterance can carry multiple labels (e.g., *"I'm not sure we can do that"* → OBJECTION + UNCERTAINTY).

### Playbook Cards

There are five strategy cards, one per event type:

| Card | Purpose |
|---|---|
| `handle_objection` | Acknowledge concerns, reduce perceived risk |
| `propose_next_step` | Convert discussion into concrete action |
| `handle_uncertainty` | Surface missing information, suggest low-risk experiments |
| `reinforce_positive` | Strengthen commitment, maintain momentum |
| `clarify_question` | Provide clear answers, link to progress |

### Pipeline

The system runs as a single command. Task 1 and Task 2 approaches are selected independently via `--task1` and `--task2` flags, allowing any combination:

```bash
python -m src.cli \
  --data_dir data --playbook_dir playbooks --split test \
  --task1 sbert_mlp --task2 semantic
```

This loads labeled data, runs tagging, runs retrieval, writes predictions + metrics, and prints a summary to stdout.

---

## 2. Evaluation Metrics

### What we changed

The original `metrics.py` scaffold computed only Task 1 classification metrics and had a placeholder for retrieval:

```python
# Retrieval recall@k cannot be computed yet because gold does not
# contain target playbook ids. Leave as placeholder.
```

We rewrote the evaluation module (`src/metrics_v2.py`) to cover both tasks comprehensively. The gold data does not include ground-truth playbook card IDs, so we defined an event-to-card mapping for retrieval evaluation (explained in the Task 2 section).

### Task 1 — Event Detection Metrics

Multi-label classification at the utterance level. Each utterance can have zero or more gold labels and zero or more predicted labels.

| Metric | What it measures | How it's computed |
|---|---|---|
| **Micro Precision** | Of all predicted labels, how many are correct | TP / (TP + FP) across all labels |
| **Micro Recall** | Of all gold labels, how many are found | TP / (TP + FN) across all labels |
| **Micro F1** | Harmonic mean of micro precision and recall | 2 × P × R / (P + R) |
| **Macro F1** | Average quality across label types | Mean of per-class F1 scores |
| **Per-class P / R / F1** | Performance on each event type individually | TP/FP/FN counted per class |
| **Exact Match Ratio** | Strictest — full label set must match exactly | Fraction of utterances where predicted set == gold set |
| **Confusion Pairs** | Most common error patterns | Counts of `LABEL→NONE` (FN) and `NONE→LABEL` (FP) |

**Why both Micro and Macro F1:** Micro F1 weights each label instance equally (favors high-support classes like QUESTION with 25 samples). Macro F1 weights each class equally (gives OBJECTION with only 8 samples the same importance as QUESTION). A model that scores well on Micro but poorly on Macro is failing on rare classes.

**Why Exact Match:** Micro F1 gives partial credit — if gold is `{OBJECTION, UNCERTAINTY}` and prediction is `{OBJECTION}`, that counts as 1 TP + 1 FN. Exact match requires the entire label set to match, which is much stricter and better reflects real-world correctness.

### Task 2 — Retrieval Metrics

| Metric | What it measures | How it's computed |
|---|---|---|
| **Recall@k** | Is the correct card found in the top-k list? | Fraction of events where the primary card appears in top-k |
| **MRR@k** | How high is the correct card ranked? | Mean of 1/rank across events (0 if not in top-k) |
| **Per-class R@k / MRR@k** | Retrieval quality per event type | Same metrics broken down by label |

**Evaluation scope — True Positive events only:** We evaluate retrieval only on events where the tagger correctly identified the label (TP). This isolates retrieval quality from tagging errors:
- Tagger misses an event (FN) → no retrieval to evaluate → skipped
- Tagger hallucinates an event (FP) → no gold card to compare → skipped
- Tagger is correct (TP) → check if the correct card is in top-k

This means the retrieval metrics answer: *"when the tagger gets it right, does the retriever find the right card?"*

---

## 3. Task 1: Event Detection

We explored four approaches, progressively improving from a simple heuristic baseline to a state-of-the-art LLM.

### Approach 1 — Baseline (Heuristic Regex)

Hand-crafted regular expressions matching keywords per label (e.g., `?` or wh-words for QUESTION, `"let's"` / `"shall we"` for NEXT_STEP). Simple and fast (<0.1s), but limited to surface-level patterns.

**Key weakness:** NEXT_STEP recall of only 0.26 — misses implicit proposals like *"Can you review this?"* that lack explicit keywords.

### Approach 2 — NLI Zero-Shot (facebook/bart-large-mnli via HF Inference API)

Repurposes a Natural Language Inference model for zero-shot classification. For each utterance, hypotheses like *"This utterance is expressing disagreement, refusal, or resistance"* are scored via entailment probability.

- `multi_label=True` for independent per-label scoring.
- Per-class thresholds tuned on combined train+dev data via F1-maximizing sweep (0.05–0.95).

**Result:** All thresholds converged to 0.95 — the model assigns high entailment scores to nearly every utterance. Even at 0.95, 107 false positives on 58 utterances. This is a **meaningful negative result**: NLI zero-shot is poorly calibrated for nuanced, multi-label conversational event detection.

### Approach 3 — SBERT + MLP (all-MiniLM-L6-v2 + trained classifier)

A supervised approach combining pre-trained sentence embeddings with a lightweight neural classifier.

- **Encoder:** `all-MiniLM-L6-v2` (384-dim embeddings, 80MB, fast on CPU).
- **Classifier:** `Linear(384→128) → ReLU → Dropout(0.3) → Linear(128→5)`.
- **Activation:** Sigmoid (not softmax) — multi-label requires independent per-class probabilities.
- **Loss:** `BCEWithLogitsLoss`. **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4).
- **Training:** Batch size 32, early stopping on dev loss (patience=20).
- **Threshold tuning:** Per-class thresholds tuned on dev set. Final: OBJECTION=0.15, NEXT_STEP=0.25, UNCERTAINTY=0.20, POSITIVE_SIGNAL=0.10, QUESTION=0.75.
- Training script is separate from inference module. Artifacts saved to `sbert_mlp_artifacts/`.

**Why it works:** Semantic embeddings capture that *"Can you review this?"* is a proposal even without explicit keywords. Best local-only approach with no API dependency.

**Potential improvement — data augmentation via DailyDialog:** The current MLP is trained on only ~200 labeled utterances (train split). The DailyDialog dataset (provided for experimentation) contains thousands of unlabeled conversations. To further improve this approach, we could use the LLM few-shot tagger (which achieves 0.95 F1) to automatically label DailyDialog utterances, then retrain the MLP on this much larger silver-labeled dataset. This is a form of **knowledge distillation** — transferring the LLM's labeling ability into the lightweight local model. The tradeoff is cost: labeling thousands of utterances via the HF Inference API requires significant API credits and time, but the resulting MLP would be trained on 10–50x more data and could potentially close the gap with the LLM while remaining fully local and fast at inference.

### Approach 4 — LLM Few-Shot (Qwen/Qwen2.5-72B-Instruct via HF Inference API)

Few-shot prompting with a 72B instruction-tuned LLM.

- **System prompt:** Detailed label definitions with emphasis on NEXT_STEP (baseline's weakest label), multi-label guidelines, and common label combinations.
- **Few-shot examples:** Three curated conversations from training data covering: empty labels, single labels, all five multi-label combinations, and subtle edge cases.
- **Conversation-level batching:** Each conversation sent as one prompt so the LLM has full turn-taking context (~11 API calls for the entire test set).
- **Structured JSON output** with regex-based fallback parsing and graceful degradation on API failures.

**Why it works:** Conversation context helps disambiguate — the LLM understands that *"I am not sure, my morning is packed"* is both OBJECTION (refusing) and UNCERTAINTY (hedging).

**Note:** Results are non-deterministic across runs due to LLM sampling.

### Task 1 Results (Test Split)

#### Overall Metrics

| Approach | Micro F1 | Macro F1 | Exact Match | Precision | Recall | Time |
|----------|----------|----------|-------------|-----------|--------|------|
| baseline | 0.774 | 0.759 | 0.431 | 0.878 | 0.691 | <0.1s |
| nli_zeroshot | 0.614 | 0.600 | 0.035 | 0.454 | 0.947 | ~38s |
| sbert_mlp | 0.865 | 0.844 | 0.724 | 0.847 | 0.883 | ~4s |
| **llm_fewshot** | **0.951** | **0.953** | **0.845** | **0.967** | **0.936** | ~45s |

#### Per-Class F1

| Class | Support | baseline | nli_zeroshot | sbert_mlp | llm_fewshot |
|-------|---------|----------|-------------|-----------|-------------|
| OBJECTION | 8 | 0.667 | 0.552 | 0.632 | **0.941** |
| NEXT_STEP | 31 | 0.400 | 0.727 | 0.857 | **0.915** |
| UNCERTAINTY | 15 | **0.966** | 0.476 | 0.849 | **1.000** |
| POSITIVE_SIGNAL | 15 | 0.800 | 0.605 | **0.968** | 0.929 |
| QUESTION | 25 | **0.962** | 0.641 | 0.913 | **0.980** |

#### What Improved Results

1. **Baseline → SBERT+MLP (+9.1 micro F1):** Semantic embeddings replaced keyword matching. Biggest gain on NEXT_STEP (0.40 → 0.86) — the encoder captures that *"Can you review this?"* is a proposal even without keywords. POSITIVE_SIGNAL jumped to 0.97 because the embedding space clusters agreement phrases together.

2. **SBERT+MLP → LLM Few-Shot (+8.7 micro F1):** Conversation context and few-shot reasoning pushed the final leap. The LLM understands multi-label nuance — *"I am not sure, my morning is packed"* is both OBJECTION and UNCERTAINTY, something the per-utterance MLP cannot learn without context. UNCERTAINTY hit perfect 1.0 F1.

3. **NLI Zero-Shot underperformed baseline (−16 micro F1):** The entailment model's scores are poorly calibrated for this task. Broad hypotheses fire on nearly everything. Documented as a meaningful negative result.

---

## 4. Task 2: Strategy Card Retrieval

### Evaluation Design

The labeled data contains event-type labels but no ground-truth playbook card annotations. To evaluate retrieval, we define a **deterministic 1-to-1 mapping** between event types and their primary card:

| Event Type | Primary Card |
|---|---|
| OBJECTION | `handle_objection` |
| NEXT_STEP | `propose_next_step` |
| UNCERTAINTY | `handle_uncertainty` |
| POSITIVE_SIGNAL | `reinforce_positive` |
| QUESTION | `clarify_question` |

This mapping is natural — the card names and content directly correspond to the event type semantics. Retrieval is evaluated on TP events only using Recall@k and MRR@k (see Evaluation Metrics section).

### Approach 1 — BM25 Baseline (Lexical Overlap)

Tokenizes the query and playbook card text, ranks by BM25 (term frequency overlap).

**Weakness:** Fails when there is no word overlap between the utterance and the correct card. For example, *"Okay, that sounds good"* (POSITIVE_SIGNAL) shares zero words with `reinforce_positive`'s content ("strengthen commitment," "acknowledge the agreement").

### Approach 2 — Semantic Retrieval (Sentence-Transformer Cosine Similarity)

Encodes both the query and each playbook card using `all-MiniLM-L6-v2`, ranks by cosine similarity.

**Why it helps:** The encoder understands that "sounds good" ≈ "agreement" ≈ "enthusiasm," even with zero word overlap.

**Initial weakness:** Without enriched queries, QUESTION Recall@3 dropped to 0.44 because the encoder mapped question utterances closer to action-oriented cards than to `clarify_question`.

### Approach 3 — Hybrid (BM25 + Semantic via Reciprocal Rank Fusion)

Combines BM25 and semantic rankings using RRF:

```
RRF_score(card) = 1/(rank_bm25 + 60) + 1/(rank_semantic + 60)
```

RRF ignores raw scores (which live on different scales) and only uses ranks. K=60 is the standard constant from the original RRF paper.

**Why it works:** BM25 nails QUESTION (keyword match), semantic nails POSITIVE_SIGNAL (meaning match). RRF merges both signals.

### Query Expansion (Applied to All Retrievers)

A key improvement applied to all three retrievers: instead of querying with just `"{event_type}: {text}"`, we enrich the query with descriptive keywords:

```
Before: "POSITIVE_SIGNAL: Okay, that sounds good."
After:  "POSITIVE_SIGNAL (agreement, enthusiasm, alignment, approval): Okay, that sounds good."
```

The description mapping:
```
OBJECTION       → "disagreement, refusal, pushback, resistance"
NEXT_STEP       → "proposal, suggestion, plan, action"
UNCERTAINTY     → "hesitation, doubt, unsure, risk"
POSITIVE_SIGNAL → "agreement, enthusiasm, alignment, approval"
QUESTION        → "question, asking, information seeking, clarification"
```

This gives BM25 more matching keywords and anchors the semantic encoder toward the correct card. Applied via a shared `build_query()` utility in `approaches/task2/__init__.py`.

### Task 2 Results (Test Split, with Enriched Queries)

Results shown using the two deterministic taggers (baseline and sbert_mlp) for clean comparison.

#### Overall Retrieval Metrics

| Retriever | Recall@3 | MRR@3 | Recall@3 | MRR@3 |
|-----------|----------|-------|----------|-------|
| | *baseline tagger (65 TP)* | | *sbert_mlp tagger (83 TP)* | |
| BM25 | 0.985 | 0.821 | 0.952 | 0.757 |
| Semantic | **1.000** | **0.939** | **1.000** | **0.964** |
| Hybrid (RRF) | **1.000** | 0.908 | **1.000** | 0.888 |

#### Per-Class Recall@3 (sbert_mlp tagger, enriched queries)

| Class | BM25 | Semantic | Hybrid |
|-------|------|----------|--------|
| OBJECTION | 0.83 | **1.00** | **1.00** |
| NEXT_STEP | 0.89 | **1.00** | **1.00** |
| UNCERTAINTY | **1.00** | **1.00** | **1.00** |
| POSITIVE_SIGNAL | **1.00** | **1.00** | **1.00** |
| QUESTION | **1.00** | **1.00** | **1.00** |

#### What Improved Results

The retrieval improvements came from two independent axes:

**1. BM25 → Semantic (+18 points MRR@3):** Embedding-based similarity fixed the weakness on classes with low keyword overlap (POSITIVE_SIGNAL, OBJECTION). However, pure semantic initially failed on QUESTION (Recall@3 = 0.44) because the encoder mapped question utterances closer to action-oriented cards.

**2. Plain queries → Enriched queries:** Adding descriptive keywords to the query gave BM25 the lexical signal it was missing and anchored the semantic encoder toward the correct card. This single change:
- Lifted BM25 Recall@3 from ~0.82 to ~0.97
- Fixed semantic QUESTION from 0.44 to **1.00**
- Made semantic the best standalone retriever (no fusion needed)

**3. Hybrid RRF (intermediate step):** Before enriched queries, hybrid was needed to combine BM25's QUESTION strength with semantic's POSITIVE_SIGNAL strength (R@3 = 0.99). After enriched queries, semantic alone achieves R@3 = 1.00 with higher MRR, making fusion unnecessary.

**Progression summary:**

| Configuration | Recall@3 | MRR@3 |
|---|---|---|
| BM25 (plain queries) | 0.82 | 0.59 |
| BM25 (enriched queries) | 0.95 | 0.76 |
| Hybrid RRF (plain queries) | 0.99 | 0.78 |
| Semantic (enriched queries) | **1.00** | **0.96** |

---

## 5. Best Configurations & Runtime

### Recommended configurations

| Configuration | Micro F1 | R@3 | MRR@3 | Runtime | API? |
|---|---|---|---|---|---|
| **sbert_mlp + semantic** | 0.865 | 1.000 | 0.964 | ~8s | No |
| **llm_fewshot + semantic** | 0.951 | 1.000 | 0.955 | ~50s | Yes |
| baseline + baseline | 0.774 | 0.985 | 0.821 | <0.1s | No |

**For production:** `sbert_mlp + semantic` — 0.865 micro F1 and perfect retrieval in ~8 seconds with zero API dependency. The sentence-transformer model (`all-MiniLM-L6-v2`) is shared between the Task 1 classifier and Task 2 retriever.

**For maximum accuracy:** `llm_fewshot + semantic` — 0.951 micro F1 with perfect retrieval, but requires a HuggingFace API token and ~50 seconds of API latency.

### Runtime breakdown

| Configuration | Task 1 | Task 2 | Total | Notes |
|---|---|---|---|---|
| baseline + baseline | <0.1s | <0.1s | **<0.1s** | No model loading |
| sbert_mlp + semantic | ~4s | ~4s | **~8s** | Model loads on first import |
| llm_fewshot + semantic | ~45s | ~4s | **~50s** | Dominated by API latency |
| nli_zeroshot + baseline | ~38s | <0.1s | **~38s** | API latency, poor results |

---

## 6. Failure Cases & Limitations

### Task 1

- **OBJECTION** has the smallest support (8 test samples) and is the hardest class across all approaches. Implicit objections like *"He tends to challenge every decision"* are subtle — even the LLM misses some.
- **NEXT_STEP** false negatives persist for the MLP: implicit proposals buried in polite phrases (*"Sounds good, let me know the details"*) require conversation context the per-utterance model cannot access.
- **NLI zero-shot** is fundamentally broken for this task: 107 false positives on 58 utterances. The model's entailment scores do not separate well for conversational event labels.

### Task 2

- **BM25** struggles on POSITIVE_SIGNAL and OBJECTION due to vocabulary mismatch between casual utterances and formal card content. Enriched queries largely mitigate this but MRR is still below 1.0 (correct card not always at rank 1).
- With only **5 playbook cards**, retrieval is a relatively easy ranking problem. In production with hundreds of cards, the approaches that matter would shift: hybrid retrieval and cross-encoder re-ranking would become critical, and the current near-perfect scores would not hold.

### General

- **LLM non-determinism:** The `llm_fewshot` approach produces slightly different results across runs due to sampling. Micro F1 typically falls in the 0.93–0.96 range.
- **Small test set:** 58 utterances across 11 conversations. Per-class metrics (especially OBJECTION with only 8 samples) have high variance — a single additional error can swing F1 significantly.

---

## 7. Code Organization

```
.
├── REPORT.md
├── requirements.txt
├── data/
│   ├── labeled_train.jsonl
│   ├── labeled_dev.jsonl
│   └── labeled_test.jsonl
├── playbooks/
│   ├── handle_objection.md
│   ├── propose_next_step.md
│   ├── handle_uncertainty.md
│   ├── reinforce_positive.md
│   └── clarify_question.md
├── src/
│   ├── cli.py                         ← main entry point
│   ├── baseline.py                    ← original baseline (reused by approaches)
│   └── metrics_v2.py                  ← evaluation: Task 1 + Task 2 metrics
├── approaches/
│   ├── task1/
│   │   ├── baseline.py                ← heuristic regex tagger
│   │   ├── llm_fewshot.py             ← LLM few-shot tagger
│   │   ├── nli_zeroshot.py            ← NLI zero-shot tagger
│   │   ├── sbert_mlp.py               ← SBERT+MLP inference
│   │   ├── sbert_mlp_train.py         ← SBERT+MLP training script
│   │   └── sbert_mlp_artifacts/       ← saved model, thresholds, config
│   └── task2/
│       ├── __init__.py                ← shared query builder (enriched queries)
│       ├── baseline.py                ← BM25 retriever
│       ├── semantic.py                ← sentence-transformer retriever
│       └── hybrid.py                  ← BM25 + semantic RRF retriever
└── outputs/                           ← generated predictions and reports
    ├── sbert_mlp+semantic/
    ├── llm_fewshot+semantic/
    ├── baseline+baseline/
    └── ...
```

Each Task 1 module exposes a `tag()` function; each Task 2 module exposes a `retrieve()` function. The CLI composes them, measures timing, and writes outputs.

---

## 8. Reproducibility

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. All dependencies are in `requirements.txt`. The sentence-transformer model (`all-MiniLM-L6-v2`, ~80MB) is downloaded automatically on first use.

### How to Run

The entire pipeline is driven by a single command:

```bash
python -m src.cli \
  --data_dir data \
  --playbook_dir playbooks \
  --split test \
  --out_dir outputs \
  --k 3 \
  --task1 <TASK1_APPROACH> \
  --task2 <TASK2_APPROACH>
```

This single command:
1. Loads the labeled data from `data/labeled_{split}.jsonl`
2. Runs event detection (Task 1) using the chosen `--task1` approach
3. Runs strategy card retrieval (Task 2) using the chosen `--task2` approach
4. Writes predictions JSONL to `outputs/{task1}+{task2}/preds_{split}.jsonl`
5. Writes metrics report JSON to `outputs/{task1}+{task2}/report_{split}.json`
6. Prints the full metrics summary to stdout

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes | — | Path to directory containing `labeled_{split}.jsonl` files |
| `--playbook_dir` | Yes | — | Path to directory containing playbook `.md` files |
| `--split` | No | `test` | Data split to run: `train`, `dev`, or `test` |
| `--out_dir` | No | `outputs` | Root output directory (`{task1}+{task2}` subfolder created automatically) |
| `--k` | No | `3` | Number of playbook cards to retrieve per event |
| `--task1` | No | `baseline` | `baseline`, `sbert_mlp`, `llm_fewshot`, or `nli_zeroshot` |
| `--task2` | No | `baseline` | `baseline`, `semantic`, or `hybrid` |
| `--hf_token` | No | — | HuggingFace API token (required for `llm_fewshot` and `nli_zeroshot`) |
| `--model` | No | — | Override HF model ID for LLM approaches |

### Recommended: SBERT+MLP + Semantic (local, no API, ~8s)

Best fully-local configuration. No API key needed, fully deterministic. Pre-trained model artifacts are included in the repository.

```bash
python -m src.cli \
  --data_dir data \
  --playbook_dir playbooks \
  --split test \
  --out_dir outputs \
  --k 3 \
  --task1 sbert_mlp \
  --task2 semantic
```

Output: `outputs/sbert_mlp+semantic/preds_test.jsonl` and `outputs/sbert_mlp+semantic/report_test.json`.

**Re-training (optional):** To retrain the MLP classifier from scratch:

```bash
python -m approaches.task1.sbert_mlp_train \
  --train_path data/labeled_train.jsonl \
  --dev_path data/labeled_dev.jsonl
```

Saves `model.pt`, `thresholds.json`, and `config.json` to `approaches/task1/sbert_mlp_artifacts/`.

### Best Accuracy: LLM Few-Shot + Semantic (requires HF API, ~50s)

```bash
python -m src.cli \
  --data_dir data \
  --playbook_dir playbooks \
  --split test \
  --out_dir outputs \
  --k 3 \
  --task1 llm_fewshot \
  --task2 semantic \
  --hf_token <YOUR_HF_TOKEN>
```

Output: `outputs/llm_fewshot+semantic/preds_test.jsonl` and `outputs/llm_fewshot+semantic/report_test.json`.

**Note:** LLM results may vary slightly across runs due to non-deterministic sampling.

### Baseline (instant, no setup)

```bash
python -m src.cli \
  --data_dir data \
  --playbook_dir playbooks \
  --split test \
  --out_dir outputs \
  --k 3 \
  --task1 baseline \
  --task2 baseline
```

Output: `outputs/baseline+baseline/preds_test.jsonl` and `outputs/baseline+baseline/report_test.json`.

### NLI Zero-Shot (documented negative result, requires HF API)

```bash
python -m src.cli \
  --data_dir data \
  --playbook_dir playbooks \
  --split test \
  --out_dir outputs \
  --k 3 \
  --task1 nli_zeroshot \
  --task2 baseline \
  --hf_token <YOUR_HF_TOKEN>
```

Output: `outputs/nli_zeroshot+baseline/preds_test.jsonl` and `outputs/nli_zeroshot+baseline/report_test.json`.

### Mix and Match

Any `--task1` and `--task2` approach can be combined freely:

```bash
python -m src.cli \
  --data_dir data --playbook_dir playbooks --split test \
  --task1 sbert_mlp --task2 hybrid

python -m src.cli \
  --data_dir data --playbook_dir playbooks --split test \
  --task1 baseline --task2 semantic

python -m src.cli \
  --data_dir data --playbook_dir playbooks --split dev \
  --task1 sbert_mlp --task2 semantic
```

Each combination produces its own output folder under `outputs/`.

### Output Format

Each `preds_{split}.jsonl` contains one JSON object per conversation:

```json
{
  "conversation_id": "test_001",
  "events": [
    {
      "event_type": "QUESTION",
      "utterance_index": 0,
      "text": "Can we schedule the review for today?",
      "retrieved_cards": [
        {"card_id": "clarify_question", "score": 0.82},
        {"card_id": "propose_next_step", "score": 0.71},
        {"card_id": "handle_objection", "score": 0.55}
      ]
    }
  ],
  "timing_ms": {"tagging": 3200.0, "retrieval": 450.0, "total": 3650.0}
}
```

Each `report_{split}.json` contains Task 1 metrics (micro/macro F1, per-class P/R/F1, exact match, confusion pairs), Task 2 metrics (Recall@k, MRR@k overall and per-class), timing, and which approaches were used.

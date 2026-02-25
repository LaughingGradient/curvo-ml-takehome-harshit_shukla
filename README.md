# Curvo Take Home: Conversation Event Detection and Strategy Retrieval

This take home simulates a simplified version of Curvo style work: turning multi turn conversations into structured signals and retrieving the best response strategy.

You will build a reproducible offline pipeline that:

1. Detects conversational events in dialogue turns  
2. Extracts evidence (which utterance triggered the event)  
3. Retrieves the most relevant strategy/playbook card for each event  
4. Produces metrics and timing numbers with a single command  

This is intentionally domain agnostic. The goal is to evaluate modeling skill, retrieval intuition, evaluation design, and practical system thinking.

Estimated time: about 10 to 12 hours. You may spread this across a few days.

---

## Dataset

We use the DailyDialog dataset:

https://huggingface.co/datasets/roskoN/dailydialog

Each example contains a list of utterances forming a conversation. Treat these as generic human interactions.

### Downloading the dataset

A helper script is included:

```
python src/scripts/download_public_dataset.py
```

This downloads the dataset locally using Hugging Face.

### How you may use the dataset

You are free to use DailyDialog for:

- experimentation
- feature extraction
- embedding index building
- weak supervision
- prompt development

However:

**Final evaluation must run only on the labeled splits in `data/`.**

Typical workflow:

- Use DailyDialog train and validation for development
- Ignore the original DailyDialog test split
- Run final evaluation only on `data/labeled_test.jsonl`

This ensures results are reproducible and comparable.

---

## Task

Your pipeline should:

1. Read ordered utterances forming a conversation  
2. Detect conversational events at the utterance level  
3. Attach one or more labels to each relevant utterance  
4. Retrieve the top k strategy cards for each event  
5. Output structured predictions and timing metrics  

---

## Event types (core)

Your model must support at least:

- OBJECTION: disagreement, refusal, pushback  
- NEXT_STEP: proposal, suggestion, plan  
- UNCERTAINTY: hesitation, doubt, perceived risk  
- POSITIVE_SIGNAL: agreement, enthusiasm, alignment  
- QUESTION: information seeking  

A single utterance may contain multiple event types.

You may add additional labels if useful.

---

## Input format

Each conversation is an ordered list of utterances.

Example:

```
[
  "Say, Jim, how about going for a few beers after dinner?",
  "That sounds tempting but it's really not good for our fitness.",
  "What do you mean?",
  "It will just make us fat and act silly."
]
```

Your system should operate on sequences like this.

---

## Output format

Your pipeline must produce structured JSON predictions.

Example:

```json
{
  "conversation_id": "dlg_001",
  "events": [
    {
      "event_type": "OBJECTION",
      "utterance_index": 1,
      "text": "That sounds tempting but it's really not good for our fitness.",
      "retrieved_cards": [
        {"card_id": "handle_objection.md", "score": 0.83},
        {"card_id": "handle_uncertainty.md", "score": 0.76}
      ]
    }
  ],
  "timing_ms": {
    "tagging": 210,
    "retrieval": 95,
    "total": 305
  }
}
```

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── labeled_train.jsonl
│   ├── labeled_dev.jsonl
│   └── labeled_test.jsonl
├── playbooks/
├── src/
│   ├── cli.py
│   ├── baseline.py
│   ├── metrics.py
```

You may reorganize if needed, but the project must remain easy to run.

---

## Setup

Python 3.10 or higher recommended.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use additional services or databases, they must be reproducible locally. Provide Docker Compose if required.

---

## How we will run your solution

Your repository must support a single command that:

1. Loads labeled data  
2. Runs tagging and retrieval  
3. Writes predictions  
4. Prints metrics and timing  

Example:

```
python -m src.cli \
  --data_dir data \
  --playbook_dir playbooks \
  --split test \
  --out_dir outputs \
  --k 3
```

Expected behavior:

- Writes predictions JSONL
- Writes metrics report JSON
- Prints summary to stdout

---

## Evaluation metrics

We evaluate three dimensions.

### Event detection quality
Multi label classification at utterance level.

Required:

- Micro F1  
- Per class precision, recall, F1  

### Retrieval quality

Given an event, retrieve relevant strategy cards.

Required:

- Recall@k (default k = 3)  

Optional:

- MRR@k  

### Runtime

Report:

- Total time per conversation  
- Optional breakdown by stage  

We care about practical tradeoffs between speed and accuracy.

---

## Constraints

- Any modeling approach is allowed  
- Classical ML, embeddings, LLMs, or hybrids are all acceptable  
- Everything must run locally and be reproducible  

If you use external APIs, document clearly how to enable them.

---

## Deliverables

Your fork must include:

1. Working code  
2. Prediction file for the test split  
3. Metrics report  
4. `REPORT.md` (1 to 2 pages) describing:
   - approach
   - design choices
   - what improved results
   - runtime considerations
   - failure cases  

---

## Bonus ideas

Optional extensions:

- Confidence calibration  
- Better evidence extraction  
- Hybrid retrieval strategies  
- Lightweight fine tuning  
- Error analysis notebook  
- Fast mode vs accurate mode  

---

## What we look for

- Reproducibility  
- Modeling intuition  
- Retrieval reasoning  
- Practical system thinking  
- Code clarity and evaluation rigor  

Good luck and enjoy the problem.
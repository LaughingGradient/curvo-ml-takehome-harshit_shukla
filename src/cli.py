import argparse
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List

from src.metrics_v2 import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Approach registry
# ---------------------------------------------------------------------------

TASK1_APPROACHES = {"baseline", "llm_fewshot", "nli_zeroshot", "sbert_mlp"}
TASK2_APPROACHES = {"baseline", "semantic", "hybrid"}


def _get_tagger(name: str) -> Callable:
    """Return the tag() callable for the chosen Task 1 approach."""
    if name == "baseline":
        from approaches.task1.baseline import tag
        return tag
    if name == "llm_fewshot":
        from approaches.task1.llm_fewshot import tag
        return tag
    if name == "nli_zeroshot":
        from approaches.task1.nli_zeroshot import tag
        return tag
    if name == "sbert_mlp":
        from approaches.task1.sbert_mlp import tag
        return tag
    raise ValueError(f"Unknown task1 approach: {name!r}")


def _get_retriever(name: str) -> Callable:
    """Return the retrieve() callable for the chosen Task 2 approach."""
    if name == "baseline":
        from approaches.task2.baseline import retrieve
        return retrieve
    if name == "semantic":
        from approaches.task2.semantic import retrieve
        return retrieve
    if name == "hybrid":
        from approaches.task2.hybrid import retrieve
        return retrieve
    raise ValueError(f"Unknown task2 approach: {name!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run the event-detection + retrieval pipeline.",
    )

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--playbook_dir", required=True)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--k", type=int, default=3)

    ap.add_argument(
        "--task1",
        choices=sorted(TASK1_APPROACHES),
        default="baseline",
        help="Task 1 approach: event tagging (default: baseline)",
    )
    ap.add_argument(
        "--task2",
        choices=sorted(TASK2_APPROACHES),
        default="baseline",
        help="Task 2 approach: card retrieval (default: baseline)",
    )

    ap.add_argument("--hf_token", default=None, help="HuggingFace API token")
    ap.add_argument(
        "--model", default=None,
        help="HF model ID for LLM approaches (default: Qwen/Qwen2.5-72B-Instruct)",
    )

    args = ap.parse_args()

    out_dir = os.path.join(args.out_dir, f"{args.task1}+{args.task2}")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(args.data_dir, f"labeled_{args.split}.jsonl")
    examples = list(read_jsonl(data_path))

    tag = _get_tagger(args.task1)
    retrieve = _get_retriever(args.task2)

    # -- Task 1: event tagging ------------------------------------------------
    task1_kwargs: Dict[str, Any] = {}
    if args.task1 in ("llm_fewshot", "nli_zeroshot"):
        if args.hf_token:
            task1_kwargs["hf_token"] = args.hf_token
        if args.model:
            task1_kwargs["model"] = args.model

    logger.info("Task 1 (%s): tagging %d utterances", args.task1, len(examples))
    t_tag = time.perf_counter()
    preds = tag(examples, **task1_kwargs)
    tag_ms = (time.perf_counter() - t_tag) * 1000

    # -- Task 2: card retrieval -----------------------------------------------
    task2_kwargs: Dict[str, Any] = {}

    logger.info("Task 2 (%s): retrieving top-%d cards", args.task2, args.k)
    t_ret = time.perf_counter()
    preds = retrieve(preds, playbook_dir=args.playbook_dir, k=args.k, **task2_kwargs)
    ret_ms = (time.perf_counter() - t_ret) * 1000

    # -- Inject timing into each conversation ----------------------------------
    for conv in preds:
        conv.setdefault("timing_ms", {})
        conv["timing_ms"].update({
            "tagging": round(tag_ms, 2),
            "retrieval": round(ret_ms, 2),
            "total": round(tag_ms + ret_ms, 2),
        })

    total_ms = int(tag_ms + ret_ms)

    # -- Write outputs ---------------------------------------------------------
    pred_path = os.path.join(out_dir, f"preds_{args.split}.jsonl")
    write_jsonl(pred_path, preds)

    report = compute_metrics(examples, preds, k=args.k)
    report["timing_ms"] = {"total": total_ms}
    report["approaches"] = {"task1": args.task1, "task2": args.task2}

    report_path = os.path.join(out_dir, f"report_{args.split}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

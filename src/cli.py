import argparse
import json
import os
import time

from src.baseline import run_pipeline
from src.metrics import compute_metrics

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--playbook_dir", required=True)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_path = os.path.join(args.data_dir, f"labeled_{args.split}.jsonl")

    examples = list(read_jsonl(data_path))

    t0 = time.perf_counter()
    preds = run_pipeline(examples, playbook_dir=args.playbook_dir, k=args.k)
    total_ms = int((time.perf_counter() - t0) * 1000)

    pred_path = os.path.join(args.out_dir, f"preds_{args.split}.jsonl")
    write_jsonl(pred_path, preds)

    report = compute_metrics(examples, preds, k=args.k)
    report["timing_ms"] = {"total": total_ms}

    report_path = os.path.join(args.out_dir, f"report_{args.split}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
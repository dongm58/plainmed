"""
scripts/02_score.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Step 2. Scores one system on one dataset split with
plainmed/metrics.py and records the result.

Usage:
  python scripts/02_score.py --system extractive --dataset cochrane --split val
  python scripts/02_score.py --system bart_base --dataset cochrane --split test --bertscore real

Input:
  outputs/<system>/<dataset>_<split>.jsonl
  data/processed/<dataset>/<split>.jsonl
Output:
  results/scores/<system>_<dataset>_<split>.csv   one row per example
  results/summary.csv                              one row per system/dataset/split, means with 95% CIs
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plainmed.io import load_config, read_jsonl, repo_path
from plainmed.metrics import score_system, summarize, upsert_summary


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=config["systems"])
    parser.add_argument("--dataset", required=True, choices=list(config["datasets"]))
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--bertscore", default="off", choices=["off", "stub", "real"])
    args = parser.parse_args()

    examples = read_jsonl(repo_path(config["paths"]["processed"]) / args.dataset / f"{args.split}.jsonl")
    outputs = read_jsonl(repo_path(config["paths"]["outputs"]) / args.system / f"{args.dataset}_{args.split}.jsonl")

    df = score_system(outputs, examples, bertscore=args.bertscore, bertscore_model=config["models"]["bertscore"],
                      is_reference=args.system == "reference")
    results = repo_path(config["paths"]["results"])
    (results / "scores").mkdir(parents=True, exist_ok=True)
    df.to_csv(results / "scores" / f"{args.system}_{args.dataset}_{args.split}.csv", index=False)

    ev = config["evaluation"]
    row = {"system": args.system, "dataset": args.dataset, "split": args.split, "bertscore": args.bertscore,
           **summarize(df, ev["bootstrap_resamples"], ev["ci_level"], config["seed"]),
           "scored_at": datetime.now().isoformat(timespec="seconds")}
    upsert_summary(results / "summary.csv", row)

    shown = {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items() if not k.endswith(("_low", "_high"))}
    print(shown)


if __name__ == "__main__":
    main()

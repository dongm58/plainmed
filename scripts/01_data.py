"""
scripts/01_data.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Step 1. Downloads the raw data if needed, writes the processed
datasets, and saves the data checks and summary statistics.

Usage: python scripts/01_data.py

Input: the Cochrane and CELLS repositories (cloned into data/raw/ if missing).
Output:
  data/processed/cochrane/{train,val,test}.jsonl
  data/processed/nihr/test.jsonl
  results/data_checks.json
  results/data_stats.csv
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plainmed.data import dataset_stats, ensure_raw, load_cochrane, load_nihr, run_checks
from plainmed.io import load_config, repo_path, write_jsonl


def main() -> None:
    config = load_config()
    ensure_raw(config)

    cochrane = load_cochrane(config)
    nihr, nihr_stats = load_nihr(config)

    processed = repo_path(config["paths"]["processed"])
    for split, records in cochrane.items():
        write_jsonl(processed / "cochrane" / f"{split}.jsonl", records)
    write_jsonl(processed / "nihr" / "test.jsonl", nihr)

    checks = run_checks(cochrane, nihr, nihr_stats, config)
    results = repo_path(config["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "data_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    stats = dataset_stats(cochrane, nihr)
    stats.to_csv(results / "data_stats.csv", index=False)

    print("sizes:", checks["sizes"])
    print("problems:", checks["problems"] or "none")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()

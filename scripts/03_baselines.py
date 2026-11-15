"""
scripts/03_baselines.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Step 3. Writes the outputs of the two baselines and of the
`reference` system (the human summary) for one dataset split, then scores all
three with plainmed/metrics.py. By default it runs only on Cochrane val:
test and NIHR are run once, in Step 9 (CONTRACT.md rule 1).

Usage: python scripts/03_baselines.py [--dataset cochrane --split val]

Input: data/processed/<dataset>/<split>.jsonl
Output:
  outputs/{copy_source,extractive,reference}/<dataset>_<split>.jsonl
  results/scores/<system>_<dataset>_<split>.csv and rows in results/summary.csv
  results/extractive_examples_<dataset>_<split>.md   3 seeded examples, source next to output
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plainmed.baselines import copy_source, extractive
from plainmed.io import load_config, read_jsonl, repo_path, write_jsonl


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cochrane", choices=list(config["datasets"]))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    if args.split == "test":
        print("Note: test split. CONTRACT.md rule 1 says test is only run in Step 9.")

    examples = read_jsonl(repo_path(config["paths"]["processed"]) / args.dataset / f"{args.split}.jsonl")
    k = config["baselines"]["extractive_k"]
    outputs = {"copy_source": [], "extractive": [], "reference": []}
    for e in examples:
        outputs["copy_source"].append({"id": e["id"], "system": "copy_source", "output": copy_source(e["source"]), "meta": {}})
        text, kept = extractive(e["source"], k)
        outputs["extractive"].append({"id": e["id"], "system": "extractive", "output": text, "meta": {"kept_sentences": kept}})
        outputs["reference"].append({"id": e["id"], "system": "reference", "output": e["target"], "meta": {}})

    out_dir = repo_path(config["paths"]["outputs"])
    for system, records in outputs.items():
        write_jsonl(out_dir / system / f"{args.dataset}_{args.split}.jsonl", records)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "02_score.py"), "--system", system,
                        "--dataset", args.dataset, "--split", args.split], check=True)

    rng = random.Random(config["seed"])
    shown = rng.sample(range(len(examples)), 3)
    lines = [f"# Extractive baseline: 3 seeded examples ({args.dataset} {args.split}, seed {config['seed']})", ""]
    for i in shown:
        e, o = examples[i], outputs["extractive"][i]
        lines += [f"## {e['id']}", "", f"**Source ({len(e['source'].split())} words):** {e['source']}", "",
                  f"**Extractive output (kept sentences {o['meta']['kept_sentences']}):** {o['output']}", "",
                  f"**Human summary:** {e['target']}", ""]
    path = repo_path(config["paths"]["results"]) / f"extractive_examples_{args.dataset}_{args.split}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

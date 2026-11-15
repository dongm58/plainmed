"""
scripts/07_check.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Step 7. Runs the faithfulness checker (plainmed/checker.py) on
one or more system output files and summarizes how often sentences are
flagged.

Usage:
  python scripts/07_check.py --inputs outputs/llm_zeroshot/cochrane_val.jsonl outputs/extractive/cochrane_val.jsonl
  python scripts/07_check.py --inputs ... --out-root /content/drive/MyDrive/plainmed/step7 --threshold 0.5
  python scripts/07_check.py --inputs ... --stub        (fake NLI model, for tests)

Input: system output files; the sources come from data/processed/.
Output:
  <out-root>/checked/<system>/<file>.checked.jsonl   each output record plus per-sentence checker results
  <out-root>/results/checker_summary.csv             one row per input file

Each file's results are written as soon as that file is done, and a file whose
checked output already covers every id is skipped, so a run cut off by a
Colab disconnect continues where it stopped (point --out-root at Google Drive).
"""

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from plainmed.checker import HFNLI, StubNLI, check_output
from plainmed.io import load_config, read_jsonl, repo_path, write_jsonl
from plainmed.metrics import bootstrap_ci


def load_sources(config) -> dict[str, str]:
    """Every processed example's source text, by id."""
    sources = {}
    for dataset, info in config["datasets"].items():
        for split in info["splits"]:
            path = repo_path(config["paths"]["processed"]) / dataset / f"{split}.jsonl"
            if path.exists():
                sources.update({e["id"]: e["source"] for e in read_jsonl(path)})
    return sources


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-root", default=str(ROOT))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None:
        frozen = config["checker"]["threshold"]
        threshold = frozen if isinstance(frozen, (int, float)) else config["checker"]["default_threshold"]
    nli = None
    sources = load_sources(config)
    out_root = Path(args.out_root)
    for input_path in args.inputs:
        input_path = Path(input_path)
        records = read_jsonl(input_path)
        out_path = out_root / "checked" / input_path.parent.name / f"{input_path.stem}.checked.jsonl"
        label = f"{input_path.parent.name}/{input_path.stem}"
        if out_path.exists() and sorted(c["id"] for c in read_jsonl(out_path)) == sorted(r["id"] for r in records):
            checked = read_jsonl(out_path)
            print(f"{label}: already checked, reusing {out_path}", flush=True)
        else:
            if nli is None:
                nli = StubNLI() if args.stub else HFNLI(config["models"]["nli"], batch_size=args.batch_size)
            checked = []
            for k, r in enumerate(records):
                result = check_output(r["output"], sources[r["id"]], nli, threshold)
                checked.append({**r, **result, "checker": {"model": nli.name, "threshold": threshold}})
                if (k + 1) % 100 == 0:
                    print(f"{input_path.name}: {k + 1}/{len(records)}", flush=True)
            write_jsonl(out_path, checked)
        system = records[0].get("system", input_path.parent.name) if records else input_path.parent.name
        sentences = [s for c in checked for s in c["sentences"]]
        mean, low, high = bootstrap_ci([c["flag_rate"] for c in checked], seed=config["seed"])
        row = {
            "file": label, "system": system, "n_outputs": len(checked), "n_sentences": len(sentences),
            "flag_rate_mean": mean, "flag_rate_ci_low": low, "flag_rate_ci_high": high,
            "share_sentences_flagged": statistics.mean(s["flag"] for s in sentences) if sentences else 0.0,
            "share_sentences_low_nli": statistics.mean(s["low_nli"] for s in sentences) if sentences else 0.0,
            "share_sentences_number_mismatch": statistics.mean(s["number_mismatch"] for s in sentences) if sentences else 0.0,
            "share_outputs_with_any_flag": statistics.mean(c["n_flagged"] > 0 for c in checked) if checked else 0.0,
            "mean_nli_score": statistics.mean(s["nli_score"] for s in sentences) if sentences else 0.0,
            "threshold": threshold, "nli_model": checked[0]["checker"]["model"] if checked else "",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"{label}: {row['share_sentences_flagged']:.1%} of sentences flagged "
              f"({row['share_sentences_number_mismatch']:.1%} number mismatch)", flush=True)
        results = out_root / "results"
        results.mkdir(parents=True, exist_ok=True)
        path = results / "checker_summary.csv"
        new = pd.DataFrame([row])
        if path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old[old["file"] != label], new], ignore_index=True)
        new.to_csv(path, index=False)
    if nli is not None and isinstance(getattr(nli, "truncated", None), int):
        print(f"NLI inputs cut at {nli.max_length} tokens: {nli.truncated}")


if __name__ == "__main__":
    main()

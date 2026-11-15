"""
scripts/11_demo_data.py

Author: Michael Dong
Date: November 15, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Step 11. Packs the precomputed outputs (and checker results, when
they exist) into one JSON file that the static demo page (docs/index.html)
reads. Because everything is precomputed, the demo needs no GPU, no model
download and no internet beyond loading the page, so it cannot fail during
the presentation.

For each example it stores the source split into sentences (with the same
splitter the checker uses), the human summary, and for each system its output
split into sentences, each sentence's checker result (support score, which
source sentences support it, unsupported numbers, flag), and readability.

Usage: python scripts/11_demo_data.py [--splits val test]

Input:
  data/processed/cochrane/<split>.jsonl
  outputs/<system>/cochrane_<split>.jsonl
  checked/<system>/cochrane_<split>.checked.jsonl   (optional; without it, sentences show as not yet checked)
  demo/cases.json                                    (optional; featured examples with notes)
Output: docs/data.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plainmed.io import load_config, read_jsonl, repo_path
from plainmed.metrics import readability
from plainmed.text import split_sentences

SYSTEMS = {"llm_fewshot": "LLM, 3-shot", "llm_zeroshot": "LLM, zero-shot", "extractive": "Extractive baseline"}


def sentence_entries(output: str, checked: dict | None, source_sentences: list[str]) -> list[dict]:
    """One entry per output sentence with its checker result, or unchecked placeholders."""
    if checked is None:
        return [{"t": s, "checked": False} for s in split_sentences(output)]
    entries = []
    for s in checked["sentences"]:
        span = 1
        if s["best_source_idx"] is not None and s["best_source_idx"] + 1 < len(source_sentences):
            pair = source_sentences[s["best_source_idx"]] + " " + source_sentences[s["best_source_idx"] + 1]
            span = 2 if s["best_source_text"] == pair else 1
        entries.append({"t": s["text"], "checked": True, "score": s["nli_score"], "src": s["best_source_idx"], "span": span,
                        "nums": s["unsupported_numbers"], "low": s["low_nli"], "flag": s["flag"]})
    return entries


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    examples_out = []
    threshold = None
    for split in args.splits:
        path = repo_path(config["paths"]["processed"]) / "cochrane" / f"{split}.jsonl"
        outputs = {}
        checked = {}
        for system in SYSTEMS:
            out_path = repo_path(config["paths"]["outputs"]) / system / f"cochrane_{split}.jsonl"
            if out_path.exists():
                outputs[system] = {r["id"]: r for r in read_jsonl(out_path)}
            chk_path = ROOT / "checked" / system / f"cochrane_{split}.checked.jsonl"
            if chk_path.exists():
                checked[system] = {r["id"]: r for r in read_jsonl(chk_path)}
                threshold = next(iter(checked[system].values()))["checker"]["threshold"]
        if not outputs.get("llm_fewshot") and not outputs.get("llm_zeroshot"):
            continue
        for e in read_jsonl(path):
            source_sentences = split_sentences(e["source"])
            systems = {}
            for system in SYSTEMS:
                if system in outputs and e["id"] in outputs[system]:
                    out = outputs[system][e["id"]]["output"]
                    chk = checked.get(system, {}).get(e["id"])
                    r = readability(out)
                    systems[system] = {"sentences": sentence_entries(out, chk, source_sentences),
                                       "fkgl": round(r["fkgl"], 1), "words": r["n_words"], "digits": round(r["digits_per_100"], 1)}
            src = readability(e["source"])
            ref = readability(e["target"])
            examples_out.append({"id": e["id"], "split": split, "source": source_sentences,
                                 "source_fkgl": round(src["fkgl"], 1), "source_words": src["n_words"],
                                 "reference": e["target"], "reference_fkgl": round(ref["fkgl"], 1), "systems": systems})

    cases_path = ROOT / "demo" / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8")) if cases_path.exists() else []
    data = {"systems": SYSTEMS, "threshold": threshold, "nli_model": config["models"]["nli"],
            "llm_model": config["models"]["llm"], "cases": cases, "examples": examples_out}
    out = ROOT / "docs" / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(examples_out)} examples, checker results: {threshold is not None})")


if __name__ == "__main__":
    main()

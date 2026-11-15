"""
scripts/04_llm.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Step 4, the prompted LLM. Three subcommands, run in this order
(notebooks/04_llm.ipynb runs them on a Colab T4):

  smoke        load the model, generate for 2 validation examples, print the
               outputs and the time per example, and estimate how long the
               rest of the notebook will take
  sensitivity  prompt variants A, B, C x {zero-shot, 3-shot} on the same 100
               seeded validation examples, then score them
               -> results/prompt_sensitivity.csv
  choose       apply the selection rule fixed in DECISIONS.md before any
               results were seen -> results/prompt_choice.json
  run          one system (zero-shot or 3-shot) with one variant on a full split
               -> outputs/llm_zeroshot/<dataset>_<split>.jsonl (or llm_fewshot)

--out-root sends outputs/ and results/ somewhere other than the repo (the
notebook points it at Google Drive so nothing is lost if Colab disconnects).
--stub swaps in a fake model so everything can be tested without a GPU.

Usage examples:
  python scripts/04_llm.py smoke
  python scripts/04_llm.py sensitivity --out-root /content/drive/MyDrive/plainmed/step4
  python scripts/04_llm.py choose --out-root ...
  python scripts/04_llm.py run --shots zero --variant B --out-root ...

Input: data/processed/cochrane/{train,val}.jsonl and config.yaml.
Output: the files listed above.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from plainmed.io import load_config, read_jsonl, repo_path
from plainmed.llm import HFGenerator, StubGenerator, build_messages, render_prompt, run_system, select_fewshot
from plainmed.metrics import bootstrap_ci, readability, score_system

VARIANTS = ["A", "B", "C"]
SHOT_SETTINGS = {"zeroshot": 0, "fewshot": 3}


def load_generator(args, config):
    """The real model, or the stub with --stub."""
    if args.stub:
        return StubGenerator(config["llm"]["max_new_tokens"])
    return HFGenerator(args.model or config["models"]["llm"], config["llm"]["max_new_tokens"], dtype=args.dtype)


def get_shots(config, n_shots: int) -> list[dict]:
    """The few-shot examples: chosen from train with the seed, and checked against the ids recorded in config."""
    if n_shots == 0:
        return []
    train = read_jsonl(repo_path(config["paths"]["processed"]) / "cochrane" / "train.jsonl")
    shots = select_fewshot(train, n_shots, config["seed"])
    recorded = config["llm"]["fewshot_ids"]
    if recorded != "TBD_ON_VAL" and [s["id"] for s in shots] != recorded:
        raise ValueError(f"few-shot selection {[s['id'] for s in shots]} does not match config {recorded}")
    return shots


def sensitivity_sample(config) -> list[dict]:
    """The same seeded sample of validation examples for every prompt configuration."""
    val = read_jsonl(repo_path(config["paths"]["processed"]) / "cochrane" / "val.jsonl")
    rng = random.Random(config["seed"])
    return sorted(rng.sample(val, config["llm"]["sensitivity_sample_size"]), key=lambda e: e["id"])


def cmd_smoke(args, config) -> None:
    val = read_jsonl(repo_path(config["paths"]["processed"]) / "cochrane" / "val.jsonl")[:2]
    generator = load_generator(args, config)
    shots = get_shots(config, 3)
    for label, s in [("zero-shot", []), ("3-shot", shots)]:
        prompts = [render_prompt(build_messages(e["source"], "A", s), generator.tokenizer) for e in val]
        t0 = time.time()
        results = generator.generate(prompts)
        per = (time.time() - t0) / len(val)
        print(f"\n=== {label}: {per:.1f} s per example (batch of {len(val)}) ===")
        for e, r in zip(val, results):
            print(f"[{e['id']}] input {r['input_tokens']} tokens, output {r['output_tokens']} tokens, "
                  f"hit max: {r['hit_max_new_tokens']}\n{r['text']}\n")
    print("Rough time estimate (batching usually makes it 3-6x faster than this unbatched-size test):")
    print(f"  sensitivity: 600 generations; full validation runs: 822 generations; at {per:.1f} s each that is "
          f"{(600 + 822) * per / 3600:.1f} h before batching speed-ups.")


def score_rows(outputs: list[dict], examples: list[dict]) -> dict:
    """Mean and CI for the metrics used to choose a prompt, plus format, truncation and timing rates."""
    df = score_system(outputs, examples)
    row = {}
    for metric in ["sari", "fkgl", "n_words", "digits_per_100", "rouge_l"]:
        mean, low, high = bootstrap_ci(df[metric])
        row[metric], row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = mean, low, high
    row["one_paragraph_rate"] = statistics.mean(o["meta"]["one_paragraph"] for o in outputs)
    row["truncated_rate"] = statistics.mean(o["meta"]["hit_max_new_tokens"] for o in outputs)
    row["seconds_per_example"] = statistics.mean(o["meta"]["seconds"] for o in outputs)
    return row


def cmd_sensitivity(args, config) -> None:
    out_root = Path(args.out_root)
    examples = sensitivity_sample(config)
    generator = load_generator(args, config)
    rows = []
    for variant in VARIANTS:
        for setting, n_shots in SHOT_SETTINGS.items():
            path = out_root / "outputs" / "prompt_sensitivity" / f"{variant}_{setting}.jsonl"
            run_system(examples, f"llm_{setting}", variant, get_shots(config, n_shots), generator, path, args.batch_size)
            rows.append({"variant": variant, "shots": setting, "n": len(examples), "model": generator.name,
                         **score_rows(read_jsonl(path), examples)})
    reference_fkgl = statistics.mean(readability(e["target"])["fkgl"] for e in examples)
    table = pd.DataFrame(rows)
    table["reference_fkgl_same_sample"] = reference_fkgl
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    table.to_csv(out_root / "results" / "prompt_sensitivity.csv", index=False)
    print(table[["variant", "shots", "sari", "fkgl", "n_words", "digits_per_100", "one_paragraph_rate",
                 "truncated_rate", "seconds_per_example"]].round(3).to_string(index=False))


def choose_variant(table: pd.DataFrame) -> dict:
    """The selection rule written in DECISIONS.md before any results existed.

    1. Drop a variant if, in either shot setting, fewer than 80% of outputs are one
       paragraph, more than 5% hit the token limit, or its mean FKGL is above the
       human summaries' FKGL on the same 100 examples.
    2. Among the rest, pick the highest SARI averaged over the two shot settings.
    3. If the top two are within 0.5 SARI, pick the one with the lower average FKGL.
    If every variant is dropped in step 1, step 2 is applied to all of them and the
    result says so.
    """
    reference = float(table["reference_fkgl_same_sample"].iloc[0])
    summary = []
    for variant, group in table.groupby("variant"):
        reasons = []
        if (group["one_paragraph_rate"] < 0.8).any():
            reasons.append("format")
        if (group["truncated_rate"] > 0.05).any():
            reasons.append("truncation")
        if (group["fkgl"] > reference).any():
            reasons.append("fkgl above human summaries")
        summary.append({"variant": variant, "mean_sari": float(group["sari"].mean()), "mean_fkgl": float(group["fkgl"].mean()),
                        "dropped_for": reasons})
    eligible = [s for s in summary if not s["dropped_for"]]
    fallback = not eligible
    pool = sorted(eligible or summary, key=lambda s: -s["mean_sari"])
    best = pool[0]
    tie_break = False
    if len(pool) > 1 and pool[0]["mean_sari"] - pool[1]["mean_sari"] < 0.5:
        best = min(pool[:2], key=lambda s: s["mean_fkgl"])
        tie_break = True
    return {"chosen_variant": best["variant"], "reference_fkgl": reference, "used_tie_break": tie_break,
            "all_variants_dropped_so_rule_relaxed": fallback, "variants": summary}


def cmd_choose(args, config) -> None:
    out_root = Path(args.out_root)
    table = pd.read_csv(out_root / "results" / "prompt_sensitivity.csv")
    choice = choose_variant(table)
    (out_root / "results" / "prompt_choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(json.dumps(choice, indent=2))


def cmd_run(args, config) -> None:
    variant = args.variant or config["llm"]["prompt_variant"]
    if variant not in VARIANTS:
        raise SystemExit("Pass --variant, or freeze llm.prompt_variant in config.yaml after the sensitivity run.")
    setting = f"{args.shots}shot"
    examples = read_jsonl(repo_path(config["paths"]["processed"]) / args.dataset / f"{args.split}.jsonl")
    if args.split == "test":
        print("Note: test split. CONTRACT.md rule 1 says test is only run in Step 9.")
    generator = load_generator(args, config)
    path = Path(args.out_root) / "outputs" / f"llm_{setting}" / f"{args.dataset}_{args.split}.jsonl"
    run_system(examples, f"llm_{setting}", variant, get_shots(config, SHOT_SETTINGS[setting]), generator, path, args.batch_size)
    print(f"wrote {path}")


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "sensitivity", "choose", "run"])
    parser.add_argument("--out-root", default=str(ROOT))
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--model", default=None, help="override config models.llm, e.g. the 1.5B fallback")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shots", choices=["zero", "few"], default="zero")
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--dataset", default="cochrane", choices=list(config["datasets"]))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    {"smoke": cmd_smoke, "sensitivity": cmd_sensitivity, "choose": cmd_choose, "run": cmd_run}[args.command](args, config)


if __name__ == "__main__":
    main()

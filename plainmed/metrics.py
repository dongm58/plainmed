"""
plainmed/metrics.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Scores every system the same way, one row per example, then
summarizes each metric with a mean and a 95% bootstrap confidence interval.

Per-example metrics
  fkgl            Flesch-Kincaid grade of the output with every number replaced
                  by NUM (textstat treats decimal points as sentence ends)
  fkgl_raw        the same without masking, kept only for the appendix
  n_words         output length in words
  digits_per_100  digit characters per 100 words (how statistic-heavy it is)
  sari            SARI against the source and the reference (plainmed/sari.py)
  rouge_l         ROUGE-L F1 against the reference, with stemming (rouge_score)
  bertscore_f1    BERTScore F1 against the reference with roberta-large,
                  rescaled with the package's baseline so scores spread out;
                  needs a GPU-sized model, so it only runs for real in Colab

The `reference` system is the human summary itself, so its SARI, ROUGE-L and
BERTScore are left empty (comparing a text with itself says nothing). Its
readability numbers are the human ceiling the systems are compared against.

BERTScore modes: "off" (column left out), "stub" (a word-overlap stand-in
used only by tests and dry runs, written to a column named
bertscore_f1_STUB so it can never be mistaken for the real thing), and
"real" (bert_score with the model named in config.yaml).

Input: system outputs and processed examples (CONTRACT.md Section 3).
Output: a per-example DataFrame, summary rows with confidence intervals, and
paired-bootstrap comparisons between two systems.
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import textstat
from rouge_score import rouge_scorer

from plainmed.sari import sari
from plainmed.text import digits_per_100, mask_numbers, word_count

READABILITY_METRICS = ["fkgl", "fkgl_raw", "n_words", "digits_per_100"]
REFERENCE_METRICS = ["sari", "rouge_l"]
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def readability(text: str) -> dict:
    """Readability and length metrics for one text."""
    return {
        "fkgl": textstat.flesch_kincaid_grade(mask_numbers(text)) if text.strip() else math.nan,
        "fkgl_raw": textstat.flesch_kincaid_grade(text) if text.strip() else math.nan,
        "n_words": word_count(text),
        "digits_per_100": digits_per_100(text),
    }


def rouge_l(output: str, reference: str) -> float:
    """ROUGE-L F1 on a 0-100 scale."""
    return 100 * _ROUGE.score(reference, output)["rougeL"].fmeasure


def bertscore_stub(outputs: list[str], references: list[str]) -> list[float]:
    """Word-overlap F1 standing in for BERTScore in tests. Never reported."""
    scores = []
    for out, ref in zip(outputs, references):
        o, r = set(out.lower().split()), set(ref.lower().split())
        common = len(o & r)
        scores.append(0.0 if not common else 100 * 2 * common / (len(o) + len(r)))
    return scores


def bertscore_real(outputs: list[str], references: list[str], model: str, batch_size: int = 16) -> list[float]:
    """BERTScore F1 (rescaled with the bert_score baseline) on a 0-100 scale."""
    from bert_score import score

    _, _, f1 = score(outputs, references, model_type=model, lang="en", rescale_with_baseline=True,
                     batch_size=batch_size, verbose=False)
    return [100 * float(x) for x in f1]


def score_system(outputs: list[dict], examples: list[dict], bertscore: str = "off",
                 bertscore_model: str = "roberta-large", is_reference: bool = False,
                 bertscore_fn: Callable | None = None) -> pd.DataFrame:
    """Score a system's outputs against their examples. Outputs and examples must cover the same ids."""
    by_id = {e["id"]: e for e in examples}
    out_ids = [o["id"] for o in outputs]
    if set(out_ids) != set(by_id) or len(out_ids) != len(set(out_ids)):
        missing = len(set(by_id) - set(out_ids))
        extra = len(set(out_ids) - set(by_id))
        raise ValueError(f"outputs and examples differ: {missing} missing, {extra} extra, {len(out_ids) - len(set(out_ids))} duplicated")

    rows = []
    for o in outputs:
        e = by_id[o["id"]]
        row = {"id": o["id"], **readability(o["output"])}
        if is_reference:
            row.update({"sari": math.nan, "rouge_l": math.nan})
        else:
            row["sari"] = sari(e["source"], o["output"], [e["target"]])
            row["rouge_l"] = rouge_l(o["output"], e["target"])
        rows.append(row)
    df = pd.DataFrame(rows)

    if bertscore != "off":
        column = "bertscore_f1" if bertscore == "real" else "bertscore_f1_STUB"
        if is_reference:
            df[column] = math.nan
        else:
            texts = [o["output"] for o in outputs]
            refs = [by_id[o["id"]]["target"] for o in outputs]
            if bertscore_fn is not None:
                df[column] = bertscore_fn(texts, refs)
            elif bertscore == "real":
                df[column] = bertscore_real(texts, refs, bertscore_model)
            else:
                df[column] = bertscore_stub(texts, refs)
    return df


def bootstrap_ci(values, resamples: int = 1000, level: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Mean and percentile bootstrap confidence interval, resampling examples with replacement."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(resamples, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarize(df: pd.DataFrame, resamples: int = 1000, level: float = 0.95, seed: int = 42) -> dict:
    """One summary row: for every metric column, its mean and confidence interval bounds."""
    row = {"n": len(df)}
    for column in [c for c in df.columns if c != "id"]:
        mean, low, high = bootstrap_ci(df[column], resamples, level, seed)
        row[column] = mean
        row[f"{column}_ci_low"] = low
        row[f"{column}_ci_high"] = high
    return row


def paired_bootstrap(a, b, resamples: int = 1000, level: float = 0.95, seed: int = 42) -> dict:
    """Compare two systems on the same examples: mean of (a - b), its CI, and how often a beats b across resamples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs scores for the same examples in the same order")
    keep = ~(np.isnan(a) | np.isnan(b))
    diff = a[keep] - b[keep]
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    alpha = (1 - level) / 2
    return {
        "n": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1 - alpha)),
        "share_a_better": float((means > 0).mean()),
    }


def upsert_summary(path, row: dict) -> pd.DataFrame:
    """Add a summary row to results/summary.csv, replacing any earlier row for the same system, dataset and split."""
    import os

    key = ["system", "dataset", "split"]
    new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~(old[key] == new[key].iloc[0]).all(axis=1)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(key).reset_index(drop=True)
    new.to_csv(path, index=False)
    return new

"""
tests/test_metrics.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks the metrics.
- SARI reproduces the value documented in the evaluate library, and matches
  values computed with evaluate's original SARI code on a real Cochrane example.
- Copying the source gives exactly the source's readability.
- Number masking keeps decimals from lowering the grade level.
- The bootstrap interval contains the mean and gets narrower with more data.
- The paired bootstrap, the reference-system rule, the BERTScore stub column,
  id checking, and the summary upsert all behave as documented.

Input: data/processed/cochrane/val.jsonl for the real-example tests (skipped if Step 1 has not been run).
Output: pass/fail results from pytest.
"""

import math

import numpy as np
import pandas as pd
import pytest

from plainmed.io import read_jsonl, repo_path
from plainmed.metrics import (bootstrap_ci, paired_bootstrap, readability, rouge_l, score_system, summarize,
                              upsert_summary)
from plainmed.sari import sari
from plainmed.text import split_sentences

VAL = repo_path("data/processed/cochrane/val.jsonl")
needs_data = pytest.mark.skipif(not VAL.exists(), reason="run scripts/01_data.py first")


def test_sari_matches_evaluate_documentation():
    value = sari(
        "About 95 species are currently accepted .",
        "About 95 you now get in .",
        ["About 95 species are currently known .", "About 95 species are now accepted .", "95 species are now accepted ."],
    )
    assert value == pytest.approx(26.953601953601954, abs=1e-12)


@needs_data
def test_sari_matches_original_code_on_real_example():
    example = read_jsonl(VAL)[0]
    first_three = " ".join(split_sentences(example["source"])[:3])
    assert sari(example["source"], example["source"], [example["target"]]) == pytest.approx(39.144218413951634, abs=1e-9)
    assert sari(example["source"], first_three, [example["target"]]) == pytest.approx(32.81920045067944, abs=1e-9)
    assert sari(example["source"], example["target"], [example["target"]]) == pytest.approx(97.539193063459, abs=1e-9)


def test_copy_source_has_source_readability():
    source = "We included 25 RCTs involving 4788 participants. Risk ratio 0.82 (95% CI 0.70 to 0.96)."
    examples = [{"id": "x", "source": source, "target": "Fewer people got sick."}]
    df = score_system([{"id": "x", "output": source}], examples)
    expected = readability(source)
    for column in ["fkgl", "fkgl_raw", "n_words", "digits_per_100"]:
        assert df.loc[0, column] == expected[column]


def test_masking_stops_decimals_counting_as_sentences():
    text = "The risk ratio was 0.82 with a confidence interval from 0.70 to 0.96 in the pooled analysis of trials."
    masked, raw = readability(text)["fkgl"], readability(text)["fkgl_raw"]
    assert masked > raw


def test_rouge_identical_is_100():
    assert rouge_l("Fewer people got sick.", "Fewer people got sick.") == pytest.approx(100.0)


def test_bootstrap_ci_contains_mean_and_narrows():
    rng = np.random.default_rng(0)
    small, large = rng.normal(10, 3, 30), rng.normal(10, 3, 3000)
    mean, low, high = bootstrap_ci(small)
    assert low <= mean <= high
    _, low_l, high_l = bootstrap_ci(large)
    assert (high_l - low_l) < (high - low)
    assert all(math.isnan(v) for v in bootstrap_ci([math.nan, math.nan]))


def test_paired_bootstrap():
    a = np.array([5.0, 6.0, 7.0, 8.0, 9.0] * 20)
    same = paired_bootstrap(a, a)
    assert same["mean_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = paired_bootstrap(a + 1, a)
    assert better["mean_diff"] == pytest.approx(1.0) and better["share_a_better"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap(a, a[:-1])


def test_reference_system_has_no_reference_metrics():
    examples = [{"id": "x", "source": "Long technical source text here.", "target": "Short summary."}]
    df = score_system([{"id": "x", "output": "Short summary."}], examples, bertscore="stub", is_reference=True)
    assert math.isnan(df.loc[0, "sari"]) and math.isnan(df.loc[0, "rouge_l"]) and math.isnan(df.loc[0, "bertscore_f1_STUB"])
    assert df.loc[0, "n_words"] == 2


def test_bertscore_stub_uses_its_own_column():
    examples = [{"id": "x", "source": "Source text.", "target": "Fewer people got sick."}]
    df = score_system([{"id": "x", "output": "Fewer people got sick."}], examples, bertscore="stub")
    assert "bertscore_f1_STUB" in df.columns and "bertscore_f1" not in df.columns
    off = score_system([{"id": "x", "output": "Fewer people got sick."}], examples)
    assert not any(c.startswith("bertscore") for c in off.columns)


def test_score_system_rejects_mismatched_ids():
    examples = [{"id": "a", "source": "s", "target": "t"}, {"id": "b", "source": "s", "target": "t"}]
    with pytest.raises(ValueError):
        score_system([{"id": "a", "output": "t"}], examples)
    with pytest.raises(ValueError):
        score_system([{"id": "a", "output": "t"}, {"id": "a", "output": "t"}], examples)


def test_summarize_and_upsert(tmp_path):
    df = pd.DataFrame({"id": ["a", "b", "c"], "sari": [10.0, 20.0, 30.0], "rouge_l": [math.nan, 5.0, 7.0]})
    row = summarize(df)
    assert row["n"] == 3 and row["sari"] == pytest.approx(20.0) and row["rouge_l"] == pytest.approx(6.0)
    path = tmp_path / "summary.csv"
    upsert_summary(path, {"system": "x", "dataset": "cochrane", "split": "val", "sari": 1.0})
    upsert_summary(path, {"system": "y", "dataset": "cochrane", "split": "val", "sari": 2.0})
    out = upsert_summary(path, {"system": "x", "dataset": "cochrane", "split": "val", "sari": 3.0})
    assert len(out) == 2 and out.set_index("system").loc["x", "sari"] == 3.0

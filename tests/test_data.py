"""
tests/test_data.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks the NIHR filter and deduplication on a tiny made-up table,
the n-gram near-duplicate finder, and (if Step 1 has been run) that the
processed files have the right sizes, fields and unique ids.

Input: data/processed/ if it exists; otherwise only the unit tests run.
Output: pass/fail results from pytest.
"""

import pandas as pd
import pytest

from plainmed.data import NgramIndex, select_nihr
from plainmed.io import load_config, read_jsonl, repo_path

FIELDS = {"id", "dataset", "split", "source", "target", "doi"}


def test_select_nihr_filters_and_dedups():
    meta = pd.DataFrame({
        "title": ["a", "b", "c", "d"],
        "journal": ["nhir_Public_Health_Research", "pnas", "nhir_Public_Health_Research", "nhir_Health_Technology_Assessment"],
        "abs_text": ["Same abstract.", "Other.", "Same  abstract.", "Different abstract."],
        "pls_text": ["Same summary.", "x", "Same summary.", "Another summary."],
        "cells_split": ["train", "train", "val", "test"],
    })
    kept, stats = select_nihr(meta, "nhir_")
    assert list(kept["title"]) == ["a", "d"]
    assert stats["nihr_rows_before_dedup"] == 3
    assert stats["nihr_duplicate_rows_removed"] == 1


def test_ngram_index_finds_contained_text():
    full_abstract = "Background word " * 5 + "We included 25 RCTs involving 4788 participants and found that written information improved knowledge about medicines in six of twelve trials."
    results_only = "We included 25 RCTs involving 4788 participants and found that written information improved knowledge about medicines in six of twelve trials."
    index = NgramIndex([full_abstract, "an unrelated abstract about something else entirely with enough words to index"])
    best, share = index.best_match(results_only)
    assert best == 0 and share == 1.0
    assert index.best_match("too short")[1] == 0.0


processed = repo_path("data/processed")
needs_data = pytest.mark.skipif(not (processed / "cochrane" / "train.jsonl").exists(), reason="run scripts/01_data.py first")


@needs_data
def test_processed_files():
    config = load_config()
    all_ids = set()
    for dataset, info in config["datasets"].items():
        for split, size in info["expected_sizes"].items():
            records = read_jsonl(processed / dataset / f"{split}.jsonl")
            assert len(records) == size
            for i, r in enumerate(records):
                assert FIELDS <= set(r)
                assert r["id"] == f"{dataset}-{split}-{i:04d}"
                assert r["dataset"] == dataset and r["split"] == split
                assert r["source"] and r["target"]
            all_ids |= {r["id"] for r in records}
    assert len(all_ids) == 3568 + 411 + 480 + 1036

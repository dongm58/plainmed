"""
tests/test_baselines.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks the baselines: copy_source returns its input; extractive
picks the fewest-digit sentences, breaks ties by length, keeps the original
order, and copes with short sources; and (if Step 3 has been run) the saved
outputs cover exactly the validation ids.

Input: outputs/ and data/processed/ if they exist.
Output: pass/fail results from pytest.
"""

import pytest

from plainmed.baselines import copy_source, extractive
from plainmed.io import read_jsonl, repo_path

SOURCE = (
    "We included 25 RCTs involving 4788 participants. "
    "The evidence was of low certainty. "
    "Risk ratio 0.82 (95% CI 0.70 to 0.96). "
    "Most trials were small and short. "
    "Written information may improve knowledge about medicines compared with no information. "
    "Six of twelve trials found a benefit."
)


def test_copy_source():
    assert copy_source(SOURCE) == SOURCE


def test_extractive_picks_fewest_digits_in_order():
    text, kept = extractive(SOURCE, k=3)
    assert kept == [1, 3, 5]
    assert text == "The evidence was of low certainty. Most trials were small and short. Six of twelve trials found a benefit."


def test_extractive_breaks_ties_by_length():
    source = "This sentence has no digits but is fairly long overall. Short one. Another short."
    text, kept = extractive(source, k=2)
    assert kept == [1, 2]


def test_extractive_short_source():
    text, kept = extractive("Only one sentence here.", k=3)
    assert text == "Only one sentence here." and kept == [0]


outputs_dir = repo_path("outputs")
needs_outputs = pytest.mark.skipif(not (outputs_dir / "extractive" / "cochrane_val.jsonl").exists(), reason="run scripts/03_baselines.py first")


@needs_outputs
def test_saved_outputs_cover_val():
    val_ids = [r["id"] for r in read_jsonl(repo_path("data/processed/cochrane/val.jsonl"))]
    for system in ["copy_source", "extractive", "reference"]:
        records = read_jsonl(outputs_dir / system / "cochrane_val.jsonl")
        assert [r["id"] for r in records] == val_ids
        assert all(r["system"] == system and r["output"] for r in records)

"""
tests/test_io.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks that JSONL files round-trip, that appending and resuming
work, and that config.yaml loads with the expected settings.

Input: a temporary folder from pytest and the repo's config.yaml.
Output: pass/fail results from pytest.
"""

from plainmed.io import append_jsonl, done_ids, load_config, read_jsonl, repo_path, write_jsonl

EXPECTED_SYSTEMS = ["copy_source", "extractive", "llm_zeroshot", "llm_fewshot", "bart_base", "reference"]


def test_jsonl_round_trip(tmp_path):
    records = [
        {"id": "cochrane-test-0000", "source": "Risk ratio 0.82 (95% CI 0.70 to 0.96).", "target": "Fewer people got sick."},
        {"id": "cochrane-test-0001", "source": "Unicode check: ≤ 5 mg", "target": "café"},
    ]
    path = tmp_path / "nested" / "folder" / "test.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path) == records


def test_append_and_resume(tmp_path):
    path = tmp_path / "outputs.jsonl"
    assert done_ids(path) == set()
    append_jsonl(path, {"id": "a", "output": "first"})
    append_jsonl(path, {"id": "b", "output": "second"})
    assert done_ids(path) == {"a", "b"}
    assert [r["output"] for r in read_jsonl(path)] == ["first", "second"]


def test_write_replaces_file(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [{"id": "old"}])
    write_jsonl(path, [{"id": "new"}])
    assert done_ids(path) == {"new"}


def test_config_loads_with_expected_settings():
    config = load_config()
    assert config["seed"] == 42
    assert config["systems"] == EXPECTED_SYSTEMS
    assert config["datasets"]["cochrane"]["expected_sizes"] == {"train": 3568, "val": 411, "test": 480}
    assert config["datasets"]["nihr"]["expected_sizes"] == {"test": 1036}
    assert config["models"]["bart"] == "facebook/bart-base"
    assert config["models"]["nli"].startswith("MoritzLaurer/DeBERTa-v3")
    assert repo_path(config["paths"]["processed"]).name == "processed"

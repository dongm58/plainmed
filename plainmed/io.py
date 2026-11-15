"""
plainmed/io.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Loads config.yaml and reads/writes JSONL files, the one file
format every step of the project uses.
append_jsonl and done_ids exist so long Colab runs can be resumed after a
disconnect without redoing finished examples.

Input: config.yaml and JSONL files on disk.
Output: Python dicts and lists, and JSONL files written to disk.
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml (from the repo root unless a path is given) into a dict."""
    config_path = Path(path) if path else ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(relative: str | Path) -> Path:
    """Turn a path from config.yaml (relative to the repo root) into an absolute path."""
    return ROOT / relative


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write records to a JSONL file, replacing it if it exists and creating folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    """Add one record to the end of a JSONL file and flush it to disk right away."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def done_ids(path: str | Path) -> set[str]:
    """Return the ids already written to a JSONL file, or an empty set if the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return set()
    return {record["id"] for record in read_jsonl(path)}

"""
plainmed/data.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Gets the raw data, turns it into the processed JSONL format from
CONTRACT.md Section 3, and runs the data checks.
- ensure_raw(): clones the Cochrane (Devaraj et al. 2021) and CELLS (Guo et al.
  2024) repositories and unzips CELLS if they are not already in data/raw/.
- load_cochrane(): one record per line of data-1024/{split}.source/.target/.doi.
- load_nihr(): the NIHR rows of CELLS (journal starts with "nhir_", CELLS's own
  spelling) from all three CELLS splits, with exact duplicate pairs removed.
- NgramIndex: finds near-copies by shared 8-word sequences. Exact matching
  misses Cochrane abstracts inside CELLS because Devaraj kept only the results
  section while CELLS stores the full abstract.
- run_checks() and dataset_stats(): the numbers saved to results/.

Input: the raw repositories under data/raw/ (cloned automatically if missing).
Output: lists of processed records, a checks dict, and a stats DataFrame.
"""

import re
import statistics
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from plainmed.io import repo_path
from plainmed.text import digits_per_100, mask_numbers, normalize, word_count

CI_PATTERN = re.compile(r"\bCI\b")


def ensure_raw(config: dict) -> None:
    """Clone both repositories and unzip CELLS into data/raw/ unless they are already there."""
    paths, sources = config["paths"], config["sources"]
    for key, url in [("cochrane_repo", sources["cochrane_git"]), ("cells_repo", sources["cells_git"])]:
        target = repo_path(paths[key])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    unzipped = repo_path(paths["cells_unzipped"])
    if not (unzipped / "train_meta.csv").exists() or not (unzipped / "CELLS" / "train.source").exists():
        unzipped.mkdir(parents=True, exist_ok=True)
        for name in ["CELLS.zip", "CELLS_metadata.zip"]:
            with zipfile.ZipFile(repo_path(paths["cells_repo"]) / "data" / name) as archive:
                members = [m for m in archive.namelist() if not m.startswith("__MACOSX")]
                archive.extractall(unzipped, members=members)


def _read_lines(path: Path) -> list[str]:
    """Read a one-example-per-line file, keeping empty lines so line numbers stay aligned."""
    return path.read_text(encoding="utf-8").rstrip("\n").split("\n")


def load_cochrane(config: dict) -> dict[str, list[dict]]:
    """Build processed Cochrane records for train, val and test."""
    base = repo_path(config["paths"]["cochrane_data"])
    out = {}
    for split in config["datasets"]["cochrane"]["splits"]:
        sources = _read_lines(base / f"{split}.source")
        targets = _read_lines(base / f"{split}.target")
        dois = _read_lines(base / f"{split}.doi")
        if not len(sources) == len(targets) == len(dois):
            raise ValueError(f"cochrane {split}: {len(sources)} sources, {len(targets)} targets, {len(dois)} DOIs")
        out[split] = [
            {"id": f"cochrane-{split}-{i:04d}", "dataset": "cochrane", "split": split,
             "source": normalize(s), "target": normalize(t), "doi": d.strip() or None}
            for i, (s, t, d) in enumerate(zip(sources, targets, dois))
        ]
    return out


def read_cells_meta(config: dict) -> pd.DataFrame:
    """All three CELLS metadata files in one table, with the CELLS split in a column."""
    base = repo_path(config["paths"]["cells_unzipped"])
    frames = [pd.read_csv(base / f"{split}_meta.csv").assign(cells_split=split) for split in ["train", "val", "test"]]
    return pd.concat(frames, ignore_index=True)


def select_nihr(meta: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, dict]:
    """Keep NIHR rows and drop exact duplicate (abstract, summary) pairs, keeping the first copy."""
    nihr = meta[meta["journal"].astype(str).str.startswith(prefix)].copy()
    nihr["source"] = nihr["abs_text"].astype(str).map(normalize)
    nihr["target"] = nihr["pls_text"].astype(str).map(normalize)
    key = nihr["source"].str.lower() + "\n" + nihr["target"].str.lower()
    duplicated = key.duplicated(keep="first")
    stats = {
        "nihr_rows_before_dedup": int(len(nihr)),
        "nihr_rows_by_cells_split": {k: int(v) for k, v in nihr["cells_split"].value_counts().items()},
        "nihr_duplicate_rows_removed": int(duplicated.sum()),
        "nihr_rows_after_dedup": int((~duplicated).sum()),
        "nihr_series": {k: int(v) for k, v in nihr.loc[~duplicated, "journal"].value_counts().items()},
    }
    return nihr.loc[~duplicated].reset_index(drop=True), stats


def load_nihr(config: dict) -> tuple[list[dict], dict]:
    """Build processed NIHR test records (deduplicated) and return them with the dedup counts."""
    nihr, stats = select_nihr(read_cells_meta(config), config["datasets"]["nihr"]["journal_prefix"])
    records = [
        {"id": f"nihr-test-{i:04d}", "dataset": "nihr", "split": "test",
         "source": row.source, "target": row.target, "doi": None,
         "title": normalize(str(row.title)), "series": row.journal, "cells_split": row.cells_split}
        for i, row in enumerate(nihr.itertuples())
    ]
    return records, stats


def _overlap_key(text: str) -> list[str]:
    """Lowercase words with punctuation removed, used only for near-duplicate matching."""
    return re.sub(r"[^a-z0-9 ]", "", normalize(text).lower()).split()


class NgramIndex:
    """Index of 8-word sequences over a pool of texts, to find which pool text best covers a query."""

    def __init__(self, texts: list[str], n: int = 8):
        self.n = n
        self.index: dict[str, set[int]] = {}
        for i, text in enumerate(texts):
            for gram in self._grams(text):
                self.index.setdefault(gram, set()).add(i)

    def _grams(self, text: str) -> set[str]:
        words = _overlap_key(text)
        return {" ".join(words[i:i + self.n]) for i in range(len(words) - self.n + 1)}

    def best_match(self, query: str) -> tuple[int | None, float]:
        """Return (pool index, share of the query's n-grams found in that pool text)."""
        grams = self._grams(query)
        if not grams:
            return None, 0.0
        counts: dict[int, int] = {}
        for gram in grams:
            for i in self.index.get(gram, ()):
                counts[i] = counts.get(i, 0) + 1
        if not counts:
            return None, 0.0
        best = max(counts, key=counts.get)
        return best, counts[best] / len(grams)


def run_checks(cochrane: dict[str, list[dict]], nihr: list[dict], nihr_stats: dict, config: dict) -> dict:
    """Sizes, empty fields, duplicates and cross-split overlap for the processed data."""
    checks: dict = {"expected_sizes": config["datasets"], "sizes": {}, "problems": []}
    seen_sources: dict[str, set[str]] = {}
    for split, records in cochrane.items():
        sources = [r["source"].lower() for r in records]
        targets = [r["target"].lower() for r in records]
        dois = [r["doi"] for r in records]
        checks["sizes"][f"cochrane_{split}"] = len(records)
        checks[f"cochrane_{split}"] = {
            "empty_source": sum(not s for s in sources), "empty_target": sum(not t for t in targets),
            "duplicate_source": len(sources) - len(set(sources)), "duplicate_target": len(targets) - len(set(targets)),
            "duplicate_doi": len(dois) - len(set(dois)), "missing_doi": sum(d is None for d in dois),
        }
        seen_sources[split] = set(sources)
    checks["cochrane_cross_split_source_overlap"] = {
        f"{a}-{b}": len(seen_sources[a] & seen_sources[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    }
    nihr_sources = {r["source"].lower() for r in nihr}
    checks["sizes"]["nihr_test"] = len(nihr)
    checks["nihr"] = dict(nihr_stats)
    checks["nihr"]["empty_source"] = sum(not r["source"] for r in nihr)
    checks["nihr"]["empty_target"] = sum(not r["target"] for r in nihr)
    checks["nihr_exact_overlap_with_cochrane"] = {split: len(nihr_sources & s) for split, s in seen_sources.items()}
    index = NgramIndex([r["source"] for r in nihr])
    checks["nihr_ngram_overlap_with_cochrane_test_val"] = sum(
        index.best_match(r["source"])[1] >= 0.5 for split in ["val", "test"] for r in cochrane[split]
    )
    expected = config["datasets"]
    for split, size in expected["cochrane"]["expected_sizes"].items():
        if checks["sizes"][f"cochrane_{split}"] != size:
            checks["problems"].append(f"cochrane {split} has {checks['sizes'][f'cochrane_{split}']} pairs, expected {size}")
    if len(nihr) != expected["nihr"]["expected_sizes"]["test"]:
        checks["problems"].append(f"nihr has {len(nihr)} pairs, expected {expected['nihr']['expected_sizes']['test']}")
    for split in cochrane:
        for name, value in checks[f"cochrane_{split}"].items():
            if value:
                checks["problems"].append(f"cochrane {split}: {name} = {value}")
    for name, value in checks["cochrane_cross_split_source_overlap"].items():
        if value:
            checks["problems"].append(f"cochrane source overlap {name} = {value}")
    if any(checks["nihr_exact_overlap_with_cochrane"].values()) or checks["nihr_ngram_overlap_with_cochrane_test_val"]:
        checks["problems"].append("nihr overlaps cochrane")
    return checks


def _describe(texts: list[str]) -> dict:
    """Length, readability and number-density summary for a list of texts."""
    import textstat

    words = [word_count(t) for t in texts]
    return {
        "n": len(texts),
        "mean_words": round(statistics.mean(words), 1),
        "median_words": statistics.median(words),
        "min_words": min(words),
        "max_words": max(words),
        "fkgl_numbers_masked": round(statistics.mean(textstat.flesch_kincaid_grade(mask_numbers(t)) for t in texts), 2),
        "fkgl_raw": round(statistics.mean(textstat.flesch_kincaid_grade(t) for t in texts), 2),
        "digits_per_100_words": round(statistics.mean(digits_per_100(t) for t in texts), 2),
        "n_mentioning_CI": sum(bool(CI_PATTERN.search(t)) for t in texts),
        "pct_mentioning_CI": round(100 * sum(bool(CI_PATTERN.search(t)) for t in texts) / len(texts), 1),
    }


def dataset_stats(cochrane: dict[str, list[dict]], nihr: list[dict]) -> pd.DataFrame:
    """One row per dataset, split and side (source or target)."""
    rows = []
    groups = [("cochrane", split, records) for split, records in cochrane.items()] + [("nihr", "test", nihr)]
    for dataset, split, records in groups:
        for side in ["source", "target"]:
            rows.append({"dataset": dataset, "split": split, "side": side, **_describe([r[side] for r in records])})
        shorter = sum(word_count(r["target"]) < word_count(r["source"]) for r in records)
        rows[-1]["pct_target_shorter_than_source"] = round(100 * shorter / len(records), 1)
    return pd.DataFrame(rows)

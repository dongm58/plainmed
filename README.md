# PlainMed: plain-language medical evidence, with checks

COSC 72/272 (Dartmouth, Fall 2026) final project · Michael Dong

PlainMed rewrites the results sections of Cochrane systematic-review abstracts into plain English with a prompted open LLM (Qwen2.5-3B-Instruct), then runs a **faithfulness checker** on every sentence of the rewrite: an NLI model (DeBERTa-v3) scores how well the source supports the sentence, and a rule flags numbers that are not in the source. Flagged sentences are shown to the reader as "check this."

- **Live demo:** https://dongm58.github.io/plainmed-demo/
- **Design decisions and every change from the proposal:** [`DECISIONS.md`](DECISIONS.md)
- **Project rules (formats, names, evaluation rules):** [`CONTRACT.md`](CONTRACT.md)

**Reading aid only, not medical advice.**

## Datasets

| Dataset | Use | Source | License |
|---|---|---|---|
| Cochrane paragraph-level simplification (Devaraj et al., 2021) | train / validation / test (3,568 / 411 / 480) | https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts | CC BY 4.0 (text belongs to Cochrane; non-commercial research use) |
| CELLS, NIHR subset (Guo et al., 2024) | second test set (1,036 pairs after removing duplicates) | https://github.com/LinguisticAnomalies/pls_retrieval | MIT |

The raw data is not copied into this repo. `python scripts/01_data.py` downloads both repositories and rebuilds the processed files (`data/processed/`) with fixed ids; data checks are in `results/data_checks.json` and statistics in `results/data_stats.csv`.

## Results so far (Cochrane test set, n = 480)

| System | Grade level (FKGL, numbers masked) | Words | SARI | BERTScore F1 | Sentences flagged by checker |
|---|---|---|---|---|---|
| Copy the source | 14.1 | 350 | 41.8 | 15.5 | — |
| Extractive baseline | 10.6 | 38 | 33.3 | 11.7 | 0% |
| Qwen2.5-3B, zero-shot | 7.7 | 200 | 36.6 | 12.6 | 36% |
| Qwen2.5-3B, 3-shot | 8.3 | 156 | 37.4 | 18.0 | 33% |
| Human plain-language summary | 12.5 | 215 | — | — | 50% |

Hand-labeled validation of the checker is in progress (see `DECISIONS.md`).

## How to reproduce

Python 3.11.

```
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/01_data.py                                    # data (about 30 s)
python scripts/03_baselines.py --split val                   # baselines + scores
python -m pytest -q                                          # tests
```

Model steps run on a Colab T4 GPU with the notebooks in `notebooks/`:
1. `04_llm.ipynb`: prompt-sensitivity study (3 wordings × zero/3-shot on 100 validation examples), prompt choice by a rule fixed in advance, full validation runs.
2. `09_check_and_test.ipynb` (or `09b_resume_checker.ipynb` after a disconnect): test-set generation, faithfulness checker, all metrics with 95% bootstrap confidence intervals.

## Code layout

- `plainmed/`: `data.py` (loading, deduplication, checks), `text.py` (normalization, sentence splitting), `metrics.py` + `sari.py` (FKGL, SARI, ROUGE-L, BERTScore, bootstrap), `baselines.py`, `llm.py` (prompts, batched generation), `checker.py` (NLI + number rule)
- `scripts/`: one script per step
- `tests/`: 61 tests (`pytest -q`)

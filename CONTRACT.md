# PlainMed Project Contract

Every build prompt says "follow CONTRACT.md". This file is the single source of truth for names, paths, formats and rules, so pieces built on different days still fit together. If a piece needs to change something here, change this file first and log why in DECISIONS.md.

## 1. Where things run

| Where | What runs there | Why |
|---|---|---|
| Claude's workspace | Writing all code, data download (git clone works), baselines, metrics that need no model, tests with stub models | Hugging Face model downloads are blocked here |
| Google Colab (T4 GPU) | Anything that loads a Hugging Face model: BART fine-tuning, LLM prompting, NLI checker, BERTScore | Free GPU, models download fine |
| My laptop | The Gradio demo for the presentation | Has to run live in class |

Every script that needs a model also has a `--stub` mode that swaps in a fake model, so it can be tested end to end in Claude's workspace before it is run for real in Colab.

## 2. Folder layout

```
plainmed/
  CONTRACT.md            this file
  DECISIONS.md           every choice and every change from the proposal, with the reason
  BUILD_PROMPTS.md       the step-by-step prompts
  README.md              how to reproduce everything (written in Step 12)
  config.yaml            all paths, seeds, model names and generation settings
  requirements.txt
  data/
    raw/                 untouched downloads (git-ignored)
    processed/           cleaned JSONL (format in Section 3)
  plainmed/              the Python package
    io.py                read/write JSONL, load config
    text.py              normalization, sentence splitting, number extraction
    metrics.py           readability + reference metrics + bootstrap
    baselines.py
    llm.py               prompt templates + LLM runner
    bart.py              fine-tuning + generation
    checker.py           NLI + number-mismatch faithfulness checker
  scripts/               one runnable script per step (01_data.py, 02_metrics_check.py, ...)
  notebooks/             Colab notebooks that call the package (no logic lives only in a notebook)
  outputs/<system>/<dataset>_<split>.jsonl
  results/               every table (CSV) and figure (PNG) used in the report or slides
  labels/                hand-labeling sheets and finished labels
  demo/app.py
  tests/
```

## 3. Data formats

**Processed example** (`data/processed/<dataset>/<split>.jsonl`), one JSON object per line:

```json
{"id": "cochrane-test-0042", "dataset": "cochrane", "split": "test",
 "source": "...technical abstract...", "target": "...plain-language summary...",
 "doi": "10.1002/14651858.CD000000"}
```

- `dataset` is `cochrane` or `nihr`. `split` is `train`, `val` or `test`. NIHR has only `test` (1,036 deduplicated pairs, pulled from all three CELLS splits). NIHR records also carry `series` (e.g. `nhir_Health_Technology_Assessment`) and `cells_split`; NIHR has no DOI, so `doi` is null.
- `id` is `<dataset>-<split>-<4-digit index>` and never changes once written. Every later file joins on `id`.

**System output** (`outputs/<system>/<dataset>_<split>.jsonl`):

```json
{"id": "cochrane-test-0042", "system": "bart_base", "output": "...", "meta": {"seconds": 1.8}}
```

**Checker output** (`outputs/<system>/<dataset>_<split>.checked.jsonl`): the output record plus

```json
{"sentences": [{"text": "...", "nli_score": 0.91, "best_source_idx": 3,
                "best_source_text": "...", "number_mismatch": false, "flag": false}],
 "flag_rate": 0.25}
```

## 4. System names (use exactly these strings)

| Name | What it is |
|---|---|
| `copy_source` | Baseline 1: the source, unchanged |
| `extractive` | Baseline 2: the 3 source sentences with the fewest digits, ties broken by length |
| `llm_zeroshot` | Prompted LLM, zero-shot |
| `llm_fewshot` | Prompted LLM, 3-shot |
| `bart_base` | BART-base fine-tuned on Cochrane train |
| `bart_cells` | Optional: BART-base fine-tuned on Cochrane + deduplicated CELLS |
| `reference` | The human-written plain-language summary, scored like a system so the reader has a ceiling |

## 5. Rules that protect the evaluation

1. **Test sets are touched once.** All choices are made on Cochrane `val`: prompt wording, decoding settings, checker threshold and BART epochs. Test and NIHR are run only in Step 9, after those choices are frozen in `config.yaml`.
2. **Seeds.** `seed: 42` in config, used for every sample, split, shuffle and training run.
3. **No leakage.** Few-shot examples come only from `train`. Any extra training data is deduplicated against Cochrane `val` and `test` before use, and the overlap counts are logged.
4. **Same inputs for every system.** All systems see the same normalized `source`. Each output is scored by the same `metrics.py` functions.
5. **Uncertainty.** Every main-table number gets a 95% bootstrap confidence interval (1,000 resamples over examples). System-vs-system claims use a paired bootstrap.
6. **Honest reporting.** If a simpler system wins, or the checker is weak, report it as is.

## 6. Code style

- Every file starts with a docstring header with the file name, author (Michael Dong), date, what it does, its input and its output.
- No inline `#` comments and no `====` banners. Explanations go in docstrings and DECISIONS.md.
- Simple, readable Python. Functions that do one thing. Type hints on public functions.
- Paths, model names and settings come from `config.yaml`, never hard-coded.
- Each step has a test file in `tests/` that runs in under a minute with stubs (`pytest -q`).

## 7. Definition of done (every step)

1. Its tests pass.
2. It writes exactly the files the prompt lists, in the formats above.
3. DECISIONS.md has a dated entry with what was chosen, the alternatives, and why.
4. There is a short "defend it" note (5 likely oral-defense questions with answers) at the bottom of that DECISIONS.md entry.

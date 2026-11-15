# PlainMed Decisions Log

Every choice, and every change from the proposal, with the reason. Newest entries at the bottom.

## Proposal to final

| Proposal said | Final does | Why |
|---|---|---|
| "An instruction-tuned model" (unnamed) | Qwen2.5-3B-Instruct, with Qwen2.5-1.5B-Instruct as the fallback | Reviewer asked us to name the models. Open weights, fits a free T4, runs the same way every time, nothing is sent to an outside API |
| NLI checker "in the style of SummaC" (model unnamed) | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | Reviewer asked us to name the models. Trained on three NLI datasets including adversarial ANLI; the base size also runs on a laptop CPU for the live demo |
| 100 hand-labeled sentences, one set for both tuning and reporting | 120 sentences: 60 from val to tune the threshold, 60 from test to report precision and recall | Reviewer feedback: keep tuning and reporting labels separate, otherwise precision and recall are inflated |
| FKGL 15.5 (source) vs 14.2 (target), hand-written syllable counter | FKGL computed with numbers masked: 14.2 vs 12.8 | textstat treats every decimal point as a sentence end, which makes number-heavy abstracts look easy (10.3 raw). See 2026-11-14 entry |
| NIHR test set: 1,287 pairs | 1,036 pairs after removing exact duplicates | 251 NIHR pairs appear twice in CELLS (same abstract and same summary) |
| 227 of 480 Cochrane test examples appear in CELLS train | 229, found by 8-gram overlap, since an exact match finds 0 | Devaraj's sources are only the results section, while CELLS stores the full abstract, so exact matching misses every overlap |

## 2026-11-14: Pre-build data checks

Scripts are in `scripts/precheck/`; the full output is in `results/precheck_2026-11-14.txt`.

**Cochrane (Devaraj et al. 2021, `data/data-1024/`)**
- 3,568 / 411 / 480 pairs; each has a DOI file of the same length.
- No empty fields, no duplicate sources, targets or DOIs, and no overlap between splits (by source text or by DOI).
- 92.0% of targets are shorter than their sources.
- License: CC BY 4.0 (LICENSE.md in the repo). Cite the NAACL 2021 paper as the README asks.

**CELLS (Guo et al. 2024)**
- The metadata CSVs (`title, journal, abs_text, pls_text`) line up row for row with the `.source` files (45,280 / 11,295 / 6,311, all rows match).
- NIHR rows are labeled `nhir_<series>` in the `journal` column (CELLS's own misspelling), across 5 series. Totals: 926 train + 252 val + 109 test = 1,287.
- 251 of those are exact duplicate pairs, some spread across CELLS splits. After deduplication: **1,036 unique NIHR pairs**. NIHR doesn't overlap Cochrane at all.
- CELLS train has 1,769 duplicate pairs inside it (this matches the proposal).
- Overlap with Cochrane: 0 exact matches, but by 8-gram overlap (at least 50% of a Cochrane source's 8-grams found in one CELLS abstract), CELLS contains 229 of the 480 Cochrane test sources and 185 of the 411 val sources in its train split, plus more in its val and test splits. **Any use of CELLS for training must deduplicate by n-gram overlap against all CELLS splits, not by exact match.**
- License: MIT.

**Readability (textstat 0.7.3, pinned)**
- The current textstat (0.7.4+) needs the NLTK cmudict download, so 0.7.3 is pinned, since it works offline.
- Raw FKGL says sources are *easier* than targets (10.3 vs 12.6). That's an artifact: textstat counts decimal points as sentence ends, so a source averages 26.7 "sentences" instead of about 14.
- With numbers masked as `NUM` first, it's 14.2 vs 12.8 (a 1.4-grade gap, close to the proposal's 1.3).
- **Decision:** all FKGL in this project is computed on number-masked text, with digits per 100 words reported as a separate metric.

**Defend it**
1. *Why mask numbers before FKGL?* FKGL relies on words per sentence, and the sentence counter breaks on decimals. Without masking, the copy-source baseline would look more readable than human summaries, which is wrong.
2. *Doesn't masking hide the numbers problem?* No, it moves it to its own metric (digits per 100 words), so readability and numeric density are measured separately.
3. *Why deduplicate NIHR?* Duplicates count the same example twice, which narrows the confidence intervals for no real reason and overweights those reviews.
4. *Why does exact matching find 0 Cochrane–CELLS overlaps?* Devaraj kept only the results section; CELLS kept the whole abstract. The text overlaps, but the strings are never identical.
5. *Is NIHR still independent of Cochrane?* Yes: zero shared abstracts, and it's a different publisher with different writing guidelines.

## 2026-11-14: Step 0, skeleton and config

**What was built:** the folder layout from CONTRACT.md, `config.yaml`, `plainmed/io.py` (`load_config`, `read_jsonl`, `write_jsonl`, `append_jsonl`, `done_ids`), `pyproject.toml` (so Colab can run `pip install -e .` and tests can import the package), two requirements files, and `tests/test_io.py` (4 tests, all pass).

**Choices**
- **Two requirements files.** `requirements.txt` pins everything for my laptop (Python 3.11). `requirements-colab.txt` leaves out torch, numpy, scipy, pandas, scikit-learn and matplotlib, because Colab already ships GPU builds of those. Reinstalling them wastes several minutes per session and can break Colab's CUDA setup.
- **Pinned for Python 3.11.** The newest numpy (2.5) and scipy (1.18) need Python 3.12, so they're pinned to numpy 2.4.6 and scipy 1.17.1. The whole file was resolved with `uv pip compile` for Python 3.11 on both Windows and Linux, with no conflicts. The Colab file resolves for Python 3.12.
- **textstat pinned to 0.7.3** (see the data-check entry: newer versions need an NLTK download).
- **Values that must be chosen on validation are written as `TBD_ON_VAL`** in config.yaml: prompt variant, few-shot ids, BART decoding settings and the checker threshold. Step 9 refuses to run on test while any `TBD_ON_VAL` is left, which enforces CONTRACT rule 1 (test is touched once).
- **Resumable writes.** `append_jsonl` flushes every record and `done_ids` lists what's finished, so a Colab disconnect in the middle of a 480-example run only loses the example in progress.

**Defend it**
1. *Why JSONL instead of CSV?* Abstracts contain commas, quotes and newlines; JSONL stores one record per line with no escaping problems, and each line can be appended on its own, which makes resuming possible.
2. *Why put every setting in config.yaml?* So the report, the code and the demo read the same values, and a change can't happen in one place and not another.
3. *Why not pin torch on Colab?* Colab's torch is built for its GPU drivers; a different pinned version could install a CPU-only build or a mismatched CUDA build.
4. *What stops you from tuning on the test set by accident?* The `TBD_ON_VAL` placeholders plus the check at the start of Step 9.
5. *Why Python 3.11?* It's supported by every pinned package, and brand-new Python versions often lack PyTorch wheels.

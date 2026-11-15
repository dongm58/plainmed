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

## 2026-11-14: Step 1, data

**What was built:** `plainmed/text.py` (normalize, sentence splitter, number masking, counts), `plainmed/data.py` (download, loaders, NIHR filter and dedup, n-gram near-duplicate index, checks, stats), `scripts/01_data.py`, and tests (`tests/test_text.py`, `tests/test_data.py`). All 14 tests pass.

**Outputs:** `data/processed/cochrane/{train,val,test}.jsonl` (3,568 / 411 / 480), `data/processed/nihr/test.jsonl` (1,036), `results/data_checks.json` (no problems found), and `results/data_stats.csv`. The processed data isn't committed to git; `python scripts/01_data.py` rebuilds it in about 20 seconds, cloning the repositories first if needed.

**Choices**
- **Normalization:** Unicode NFKC plus whitespace collapsing, applied once to every source and target so all systems get byte-identical input. NFKC also turns ligatures and full-width characters into plain ones (for example "ﬁ" becomes "fi").
- **A rule-based sentence splitter instead of spaCy (decided by comparison, see the audit entry below).** It splits after `.`, `!` or `?` only when the next word starts with a capital letter, digit or quote, and never after a known abbreviation (e.g., vs., et al.), a single initial, or a numbered-list marker ("1."). Decimals ("0.82") and confidence intervals are safe because there's no space after the decimal point. It passes a test of 10 real Cochrane sentences with CIs, fractions and quotes.
- **NIHR:** taken from all three CELLS splits because NIHR is only a test set for us. Exact duplicate (abstract, summary) pairs are removed, keeping the first copy, which leaves 1,036 pairs across 5 NIHR series. Each record keeps `title`, `series` and `cells_split`.
- **IDs** are `<dataset>-<split>-<4-digit line number>`, so any record can be traced back to its line in the original files.

**Numbers compared with the proposal**
- The sizes match (3,568 / 411 / 480). The word counts match (source mean 351 and median 320, target mean 212 and median 198). 92.0% of targets are shorter than their sources (same as the proposal). Digits per 100 words: 21.3 vs 5.2 (same).
- FKGL changed from 15.5 / 14.2 (hand-written syllable counter) to **14.2 / 12.8** (textstat with numbers masked). The gap is about the same.
- "Mentions CI" changed from 2,716 / 53 to **2,707 / 26**. The final count matches "CI" only as a whole, case-sensitive word, so it no longer counts words that merely contain those letters.

**Something found that matters later: NIHR inputs are a different format.** Cochrane sources are only the *results section* (14 sentences on average). NIHR sources are *full structured abstracts* with section names run into the text ("Background ... Objectives ... Design ... Results ... Conclusions ... Limitations ... Future work"), averaging 31 sentences and 551 words. So NIHR tests a change in source *and* in input format. This stays as in the proposal (full abstract as input), and the report has to say so when interpreting NIHR results. A follow-up, if time allows, is to feed only the Results and Conclusions sections of NIHR abstracts and see whether the gap shrinks. The longest NIHR abstracts (up to 860 words) may also go over BART's 1,024-token limit; Step 5 measures how many with the real tokenizer.

**Dataset paragraph for the report (draft)**
> We use the Cochrane paragraph-level simplification dataset of Devaraj et al. (2021): 4,459 pairs of a systematic-review abstract's results section and the review's author-written plain-language summary, split 3,568 / 411 / 480 for train / validation / test as released. The repository is licensed CC BY 4.0, and the underlying text belongs to Cochrane and is used here for non-commercial research. Sources average 351 words and a grade level of 14.2 (Flesch–Kincaid computed with numbers masked), and targets average 212 words at grade 12.8, with four times fewer digits (21.3 vs 5.2 per 100 words). As a second, independent test set we use the NIHR subset of CELLS (Guo et al., 2024; MIT License): plain-language summaries of UK National Institute for Health and Care Research reports. After removing 251 exact duplicate pairs, it has 1,036 pairs, taken from all CELLS splits, since we never train on it. NIHR inputs are full structured abstracts (551 words on average) and share no abstracts with Cochrane. Neither dataset contains patient records.

**Defend it**
1. *Why not use spaCy for sentences?* Same splitter everywhere with no model download; it handles the specific traps in this data (decimals, CIs, "et al."), and the test proves it on real sentences.
2. *Why include NIHR from CELLS's train split in a test set?* We never train on CELLS in the main systems, so its own split labels don't matter to us. What matters is that NIHR shares nothing with Cochrane, which the checks confirm.
3. *What does the NIHR format difference mean for conclusions?* A drop on NIHR can't be blamed only on the new source: the inputs are also longer and include background and methods. The report says this, and the Results-only follow-up would separate the two.
4. *Why did FKGL change from the proposal?* The proposal used a rough syllable counter; the final number uses textstat on number-masked text, because unmasked decimals break sentence counting.
5. *How would someone rebuild the data?* `python scripts/01_data.py`. It clones both repositories, unzips CELLS, and writes the same files with the same ids.

## 2026-11-14: Audit of Steps 0 and 1

Checked both steps against their prompts and rebuilt everything from a fresh clone of the GitHub repo on **Python 3.11.15** in a clean virtual environment (the laptop's version), not on the workspace's own Python.

**Problems found and fixed**
1. **textstat 0.7.3 crashed on a fresh install.** It imports `pkg_resources`, which setuptools removed in version 82. My workspace hid this because it had an old setuptools. Fix: `setuptools==81.0.0` pinned in both requirements files. Verified: it fails with 82.0.1 and 83.0.0 and works with 81.0.0.
2. **The Step 1 prompt said to use spaCy if it installs, and I hadn't tried.** It does install (spaCy 3.8.18 + en_core_web_sm 3.8.0). Both splitters pass the 10-real-sentence test, so I compared them on 100 seeded validation abstracts. Before any fixes they gave identical splits on 86 of them. On the 14 disagreements:
   - The rule-based splitter was right on 10. spaCy split mid-sentence after units ("156/101 mm" | "Hg."), inside lists of abbreviations, before a parenthesized CI, and in plain sentences ("quality of life and" | "immunological function.").
   - spaCy was right on 2: a missing space in the source ("each study.The Nerve…") and a source typo ("(RR 0.62. 95% CI…)").
   - Two were rule-based bugs that this comparison exposed and I fixed. One: a unit ending in a single letter ("3.5 g/d. No trials…") was treated like an initial, so the sentences weren't split.
   - Two: a numbered list ("1. Chlorpromazine versus…") was split into a lone "1." sentence.
   After the two fixes, the splitters agree on 88 of 100, and the remaining 12 split 10 to 2 in favor of the rule-based splitter.
   **Decision: keep the rule-based splitter.** It's more accurate on this data, it's deterministic, and it needs no model download for the demo. Both fixes have tests.
3. **Unmeasured claim removed.** The first version of the Step 1 entry gave a percentage of NIHR abstracts over 1,024 tokens that had never been measured. It now says Step 5 will measure it.

**Verified**
- On Python 3.11: every pinned package except torch installs and imports together, including transformers, gradio, evaluate, sacrebleu, rouge-score, bert-score, datasets and accelerate. Torch isn't installed or tested yet; that happens in the first model step.
- From an empty folder, `python scripts/01_data.py` clones both repositories, builds the data in about 25 seconds, and the processed files are **byte-identical** to the ones in my working copy. The data stats are identical too.
- All 16 tests pass.

**Deviations from the prompts (on purpose)**
- Step 0 said to leave the "Proposal to final" table empty; it was filled in early from the data pre-checks and the reviewer's feedback, because those changes were already known.
- Step 1 said to reuse `scripts/precheck/`; that logic now lives in `plainmed/data.py` instead (so Step 6 can import it), and the results match the pre-check output exactly. The precheck scripts stay in the repo as the record of what was checked before building.

**Defend it**
1. *Why trust a hand-written splitter over spaCy?* Because I measured it: on 100 real abstracts it was right in 10 of the 12 disagreements left after fixing two bugs, and spaCy's errors (splitting "mm | Hg.") would create fake "unsupported sentences" for the checker.
2. *What would a fresh-install bug have cost?* The demo or a grader's reproduction would crash on the first readability score. Testing from a clean clone on the same Python as the laptop is what caught it.
3. *Is the regex splitter perfect?* No: it can't split when the source is missing a space after a period, and it splits on typos like "(RR 0.62. 95% CI". Both are rare (2 of 100 abstracts).

## 2026-11-14: Step 2, metrics

**What was built:** `plainmed/sari.py` (SARI), `plainmed/metrics.py` (per-example scoring, bootstrap CIs, paired bootstrap, summary upsert), `scripts/02_score.py`, and `tests/test_metrics.py` (11 tests). The project now has its own Python 3.11 environment (`.venv`, not committed), and all 27 tests pass in it.

**Each metric, why it's there, and what it can't tell us**

| Metric | Why | Can't tell us |
|---|---|---|
| FKGL, numbers masked | The standard readability formula; the task is about readability | It only counts sentence length and syllables, so it can't see jargon that has few syllables ("RCT") or tell whether the text makes sense. Readability formulas were also built for general English, not health text |
| Raw FKGL (appendix only) | Shows why masking was needed | Unmasked decimals make number-heavy text look easy |
| Words | Plain-language summaries condense; length shows whether a system just copies | Shorter isn't automatically clearer |
| Digits per 100 words | Our data exploration found that simplification here mostly means dropping statistics (21.3 to 5.2) | It can't say whether the *right* numbers were kept |
| SARI | The standard simplification metric: rewards keeping, deleting and adding the same n-grams the human did | There's only one reference per example, and the references are weak (only 1.4 grades simpler than the source), so SARI partly measures "wrote like this particular author" |
| ROUGE-L | Content overlap with the reference | Rewards copying; says nothing about faithfulness to the *source* |
| BERTScore F1 | Overlap of meaning, not exact words, so it gives credit for paraphrases | Still compared with the weak reference; roberta-large reads at most 512 tokens, so long texts are cut off |

Faithfulness to the source isn't measured by any of these. That's the checker's job (Step 7) and the hand labels' job (Step 8).

**Choices**
- **SARI is copied from `evaluate` instead of called through it.** `evaluate.load("sari")` downloads the metric's code from the Hugging Face Hub every time it runs, which fails offline (including in my workspace). The copy keeps the same logic (Apache 2.0 license, credited in the file). Verified two ways: it reproduces the value in evaluate's documentation (26.9536…), and it gives **exactly the same score as evaluate's original code (maximum difference 0) on 2,466 real cases**: 411 validation examples × 6 kinds of output (copy the source, the reference itself, the first 3 sentences, a random half of the sentences, an empty string, and a one-sentence summary).
- **SARI is scored per example and averaged.** That's how evaluate defines the corpus score anyway, and per-example scores are needed for the bootstrap.
- A quirk worth knowing for the defense: the reference compared with itself gets a SARI of **97.5, not 100**, because of how the keep and delete parts are defined. So the `reference` system's SARI, ROUGE-L and BERTScore are left empty rather than reported.
- **ROUGE-L comes from the `rouge_score` package** (with stemming), for the same reason as SARI: evaluate's version downloads from the Hub.
- **BERTScore** uses roberta-large rescaled with the baseline file that ships with bert_score, so scores spread over a readable range instead of all sitting near 0.85–0.90. It only runs for real in Colab. The test stand-in writes to a column named `bertscore_f1_STUB`, so a fake score can never end up in a results table by mistake.
- **Bootstrap:** 1,000 resamples of examples, percentile 95% CI, seed 42. The paired bootstrap resamples the *same* examples for both systems, which is the right test when two systems are compared on one test set.
- **`results/summary.csv` replaces a row instead of appending a duplicate** when a system is re-scored, so re-running a step can't leave two different numbers for the same system.

**End-to-end check (fresh clone, Python 3.11):** scoring copy-the-source and the reference on Cochrane val took about 10 seconds per system, and the readability numbers match `results/data_stats.csv` exactly (FKGL 14.28 vs 12.85, digits 20.84 vs 5.15), so Step 1 and Step 2 agree. **Copying the source scores SARI 41.5 (95% CI 41.1–41.9).** That is the bar every real system has to clear, and it shows how much SARI gives for simply keeping the source's words when the references are close to the source. Worth saying in the report.

**Not tested yet:** real BERTScore, because it needs to download roberta-large, which is blocked here. The first Colab notebook will run it on a few examples as a check.

**Defend it**
1. *Why not just use evaluate.load?* It fetches code from the internet at run time, so a result could change if the hosted script changed, and it can't run offline. The copied version is checked to give identical scores on 2,466 cases.
2. *Why mask numbers for FKGL instead of using a different formula?* Every common formula uses words per sentence, and the sentence counting is what breaks. Masking fixes the cause, and digits per 100 words still tracks the numbers.
3. *If SARI rewards matching a weak reference, why report it?* It's the field's standard, so readers expect it. But the conclusions weight readability and faithfulness more, as the proposal's risk table said they would.
4. *What does a confidence interval mean here?* If we drew new test sets of the same size from the same population, about 95% of intervals built this way would contain the true mean score. It shows how much a number could move just from which examples happened to be in the test set.
5. *Why a paired bootstrap for comparing systems?* Some abstracts are harder than others for every system. Pairing compares systems on the same abstracts, which removes that shared difficulty and gives a fairer and tighter comparison.

## 2026-11-14: Step 3, baselines

**What was built:** `plainmed/baselines.py` (`copy_source`, `extractive`), `scripts/03_baselines.py`, and `tests/test_baselines.py` (5 tests). All 32 tests pass. It ran on Cochrane val only; test and NIHR wait for Step 9.

**Outputs:** `outputs/{copy_source,extractive,reference}/cochrane_val.jsonl`, per-example scores in `results/scores/`, rows in `results/summary.csv`, and `results/extractive_examples_cochrane_val.md` (3 seeded examples, each source next to its outputs). System outputs are committed to git: they're small, and the demo's cached mode and the report tables must use exactly these files.

**Validation results (n = 411, 95% bootstrap CIs)**

| System | FKGL | Words | Digits/100 | SARI | ROUGE-L |
|---|---|---|---|---|---|
| copy_source | 14.3 (14.1–14.5) | 347 | 20.8 | 41.5 (41.0–41.9) | 25.2 (24.3–26.3) |
| extractive | 11.0 (10.8–11.3) | 39 | 0.3 | 34.2 (33.7–34.6) | 14.9 (14.2–15.7) |
| reference (human) | 12.9 (12.6–13.1) | 204 | 5.2 | n/a | n/a |

**What these show**
- The numbers behave as expected: copy_source has exactly the source's readability, and extractive is much shorter, with almost no digits.
- **Extractive "beats" the human summaries on FKGL (11.0 vs 12.9) while being clearly worse.** Picking the fewest-digit sentences with ties broken by length favors short sentences, so the output is only 39 words, about a fifth of the human summary. This is a concrete example of why FKGL can't be the only readability measure: it rewards short sentences, not clear content. Length and faithfulness have to be read alongside it.
- **Extractive scores lower SARI than copying the source (34.2 vs 41.5).** Deleting most of the source loses many n-grams the reference kept. So a neural system's SARI should be compared with *both* baselines.
- **The examples show the extractive baseline's typical failure: it drops the main finding.** In `cochrane-val-0012`, the key result (text-message reminders improved appointment attendance, RR 1.14) has numbers in it, so it's dropped, and the output keeps "Four of these studies were newly identified during this update" instead. That's an omission error (category 3 in the proposal) and a good failure example for the report.
- Ties are broken by fewest words, as the proposal specified. Reversing it (prefer longer sentences) would give longer output that's more comparable to the reference, but that would be a design change made after seeing results, so it's left alone. It could appear as a variant in the analysis.

**Defend it**
1. *Why have a baseline that obviously loses?* Copy-source sets the floor for readability and shows how much SARI and ROUGE reward doing nothing (41.5 SARI). Without it, a model's SARI of 42 would look like progress.
2. *What is the extractive baseline testing?* Whether "just drop the statistic-heavy sentences" is enough. It isn't: it loses the main findings and is far too short.
3. *Why does extractive have a better grade level than the human summary?* FKGL rewards short sentences and short words, and the baseline picks exactly those. Grade level measures the surface of the text, not whether it says the right things.
4. *Why not tune k or the tie-break to look better?* The design was fixed in the proposal. Changing it after seeing validation results would be fitting the baseline to the metric.
5. *Why commit outputs to git?* So the tables, the demo and anyone reproducing the work use the exact same outputs.

# PlainMed Build Prompts

How to use this: paste **one** prompt at a time. Don't start the next one until the current step's tests pass and you've read its DECISIONS.md entry. Every prompt starts with the same line so the conventions carry over even in a fresh chat.

Each step lists the rubric points it earns, so nothing in the rubric is left uncovered:

| Rubric category (points) | Steps that earn it |
|---|---|
| 1. Technical implementation and evolution (10) | 0–7, plus the DECISIONS.md log |
| 2. Evaluation and results (10) | 2, 3, 4, 5, 8, 9 |
| 3. Interpretation and future work (15) | 10, 12 |
| 4. Interactive demo (15) | 11 |
| 5. Oral defense (20) | the "defend it" notes in every step, plus 13 |
| 6. Report and reproducibility (10) | 1, 12, 14 |

---

## Step 0: Skeleton and config

```
Follow CONTRACT.md in /home/claude/plainmed exactly.

Step 0: set up the project skeleton.
- Create the folder layout from CONTRACT.md Section 2 (empty modules with their docstring headers).
- Write config.yaml with: seed 42; data paths; dataset names; system names from Section 4; model names (bart: facebook/bart-base; llm: Qwen/Qwen2.5-3B-Instruct, fallback Qwen/Qwen2.5-1.5B-Instruct; nli: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli; bertscore: roberta-large); max_source_tokens 1024; max_target_tokens 512.
- Write plainmed/io.py: load_config(), read_jsonl(), write_jsonl(), append_jsonl() (append is for resumable runs in Colab).
- Write requirements.txt with pinned versions (textstat==0.7.3: newer versions need an NLTK download that's blocked here; see DECISIONS.md).
- Start DECISIONS.md with a "Proposal to final" table (columns: proposal said / final does / why). Leave it empty for now.
- tests/test_io.py: round-trip a JSONL file and load the config.
Done when pytest passes. Don't download data or write any model code yet.
```

Earns: reproducibility (6).

---

## Step 1: Data

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 1: build the datasets.
- scripts/01_data.py: git clone the Devaraj et al. repo (AshOlogn/Paragraph-level-Simplification-of-Medical-Texts) into data/raw/, read data/data-1024/{train,val,test}.{source,target} plus the DOI files, and write data/processed/cochrane/{train,val,test}.jsonl in the Section 3 format.
- CELLS is already unzipped in data/raw/cells_x/ (CELLS/{split}.source/.target and {split}_meta.csv with columns title, journal, abs_text, pls_text; the rows line up with the .source files). NIHR rows have journal starting with "nhir_" (CELLS's spelling). Take them from all three CELLS splits (1,287 rows), drop the 251 exact duplicate pairs (keep the first), and write the 1,036 unique pairs to data/processed/nihr/test.jsonl. Keep the series name and the original CELLS split in each record.
- plainmed/text.py: normalize() (Unicode NFKC, whitespace) and split_sentences(). Use spaCy en_core_web_sm if it installs, otherwise a regex splitter that doesn't break on "e.g.", "vs.", "CI 0.8 to 1.2" or decimals. Test it on 10 real abstract sentences.
- Checks, saved to results/data_checks.json:
  split sizes (expect 3568/411/480 and NIHR 1,036), empty fields, duplicate sources within and across splits, and NIHR overlap with Cochrane (expect 0). The numbers must match results/precheck_2026-11-14.txt; reuse the logic in scripts/precheck/ rather than writing it again.
- results/data_stats.csv: for each dataset and split, the mean/median/min/max words, FKGL (textstat on number-masked text, per DECISIONS.md), digits per 100 words and the share mentioning "CI", for source and target. Compare with proposal Section 5 and log any number that changed.
- Write the dataset paragraph for the report (source, size, license: Cochrane repo CC BY 4.0 with Cochrane text for non-commercial research; CELLS MIT) into DECISIONS.md.
Done when the tests pass and data_checks.json shows no unexplained problems.
```

Earns: data description and licensing (6), data issues and leakage (2).

---

## Step 2: Metrics

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 2: plainmed/metrics.py, one place that scores every system the same way.
- Per example: fkgl (textstat 0.7.3 on text with every number replaced by NUM, because textstat treats decimal points as sentence ends; keep raw_fkgl too, for the appendix), n_words, digits_per_100, and sari (needs source, output and reference; use the evaluate library's sari), rouge_l and bertscore_f1 (model name from config; stub in tests).
- score_system(outputs, examples) -> per-example DataFrame, and summarize() -> mean plus 95% bootstrap CI for each metric (1,000 resamples, seed from config).
- paired_bootstrap(a, b) -> mean difference, CI and the share of resamples where a beats b.
- BERTScore is the only part that needs a model, so it can be turned off with a flag. In my workspace it runs as a stub; in Colab it runs for real.
- scripts/02_score.py --system NAME --dataset D --split S: writes results/scores/<system>_<dataset>_<split>.csv (per example) and appends a row to results/summary.csv.
- Tests: SARI on the example in the evaluate library's SARI docs gives the documented value; copying the source gives the same FKGL as the source; the bootstrap CI contains the mean.
Explain in DECISIONS.md why each metric is there and what it can't tell us (SARI against weak references, FKGL being a surface formula).
```

Earns: metrics that fit the task, uncertainty (2).

---

## Step 3: Baselines

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 3: the two baselines plus the reference as a "system".
- plainmed/baselines.py: copy_source(); extractive(source, k=3): the k sentences with the fewest digits, ties broken by fewest words, kept in their original order.
- scripts/03_baselines.py writes outputs/{copy_source,extractive,reference}/cochrane_val.jsonl, then scores them with metrics.py (BERTScore off for now).
- Show me the val summary table and 3 extractive outputs next to their sources.
- Don't run test or NIHR yet (CONTRACT rule 1). Step 9 runs everything on test at once.
Done when the tests pass and the val numbers look sensible (copy_source should have the source's FKGL; extractive should be shorter with fewer digits).
```

Earns: meaningful baselines (2), implementation (1).

---

## Step 4: Prompted LLM

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 4: the prompted LLM system.
- Model: Qwen/Qwen2.5-3B-Instruct in fp16 on a Colab T4; fall back to Qwen/Qwen2.5-1.5B-Instruct if a full val run takes more than about 2 hours. Names are already in config. Check the license on each model card and record it in DECISIONS.md. Log why an open model instead of an API (cost, reproducibility, nothing leaves the machine). The reviewer asked us to name the models, so this has to be explicit in the report.
- plainmed/llm.py:
  - PROMPTS: a dict of 3 instruction wordings (A, B, C) that all ask for one plain-language paragraph at about an 8th-grade level, keeping the main findings and how certain they are, with no new facts.
  - build_prompt(source, variant, shots) using the model's chat template. Few-shot uses 3 training pairs chosen with the config seed, short enough to fit the context; store their ids in config.
  - generate(): greedy decoding, max_new_tokens from config, records seconds per example.
  - The runner appends to the output JSONL and skips ids already done, so a Colab disconnect doesn't lose work. Has a --stub mode.
- notebooks/04_llm.ipynb: installs the package from a zip or Drive, runs on Colab.
- Prompt sensitivity (this is a diagnostic the rubric asks for): run variants A/B/C x {zero-shot, 3-shot} on the same 100 val examples (seeded sample) and score them. Write results/prompt_sensitivity.csv. Pick the final variant on val only and freeze it in config.
- Then run the chosen zero-shot and 3-shot prompts on all of val.
- Build and test everything here with --stub; I will run the notebook in Colab and send back the outputs folder.
```

Earns: prompting strategy (1), prompt sensitivity (2).

---

## Step 5: Fine-tuned BART-base

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 5: fine-tune BART-base on Cochrane train.
- plainmed/bart.py with Hugging Face Seq2SeqTrainer: source truncated to 1024 tokens, target to 512, lr 3e-5, batch size that fits a T4 (use gradient accumulation and fp16), at most 10 epochs, evaluate on val loss every epoch, early stopping with patience 2, load the best checkpoint at the end. All settings live in config.
- Log the train and val loss per epoch to results/bart_history.csv and plot results/fig_bart_curves.png.
- Decoding: compare 2-3 settings on val (beam 4 vs beam 4 + no_repeat_ngram_size 3 + length_penalty) and freeze the best in config, decided on val SARI and FKGL together. Log the choice.
- Save the model to Drive. Write outputs/bart_base/cochrane_val.jsonl.
- Test the data collation, truncation and the generate loop with a tiny random model in my workspace, then I run notebooks/05_bart.ipynb in Colab.
- Log anything that changes from the proposal (for example, if the unlikelihood loss is dropped, say why).
```

Earns: correct training, validation and early stopping (2), justified model choices (1).

---

## Step 6 (optional): Cochrane + CELLS (SKIPPED on 2026-11-14, see DECISIONS.md)

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 6 (optional): the data experiment from the proposal.
- Build the extra training set from CELLS train: drop every pair that overlaps any Cochrane val or test source by n-gram overlap (exact matching finds 0 overlaps, because Devaraj kept only the results section; reuse scripts/precheck/overlap.py, which finds 229 test and 185 val overlaps), drop the 1,769 internal duplicates, and drop all NIHR pairs (NIHR is a test set). Save the counts removed at each stage to results/cells_dedup.json.
- Train bart_cells with the exact same settings as bart_base (only the data changes). Write val outputs.
- If there isn't time, skip this and add a row to the "Proposal to final" table explaining why (it's a secondary question, and the leakage handling is already documented).
```

Earns: data handling and leakage (2), project evolution (1).

---

## Step 7: Faithfulness checker

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 7: plainmed/checker.py, the part the project's contribution rests on.
- For each output sentence: run an NLI model (premise = each source sentence, hypothesis = the output sentence) and take the max entailment probability over source sentences, SummaC-ZS style. Keep which source sentence gave the max (the demo uses it for "show me the source").
- Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (name in config). Log why: trained on MNLI, FEVER and ANLI, and small enough to run on a laptop CPU in the live demo. Source sentences longer than the model's limit get truncated; log how often that happens.
- Number check: extract numbers from both sides (digits, decimals, percentages, "x in y", number words like "twelve"), normalize them, and flag an output sentence that has a number the source doesn't. Unit tests with 15 tricky cases ("25 trials" vs "twenty-five trials", "4,788" vs "4788", "50%" vs "half").
- flag = nli_score < threshold OR number_mismatch. Threshold is a placeholder (0.5) until Step 8 tunes it.
- scripts/07_check.py --system --dataset --split writes the .checked.jsonl from CONTRACT Section 3 and adds flag_rate to the summary.
- Test with a stub NLI model here; I run it in Colab on val for every system.
```

Earns: implementation of the core contribution (1).

---

## Step 8: Hand labels and checker validation

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 8: validate the checker against my own labels.
- Sampling: 120 output sentences, half from bart_base and half from the chosen LLM, stratified across checker scores so both flagged and unflagged sentences show up. 60 come from Cochrane val (for tuning the threshold) and 60 from Cochrane test (for the final precision/recall). This is a change from the proposal's 100, and it's exactly what the reviewer asked for (keep the tuning labels separate from the reporting labels). It's already in the DECISIONS.md table; cite the feedback in the report.
- labels/sheet.csv: id, sentence, full source, then empty columns: supported (yes/no), error_type (number / overstated certainty / omission-not-applicable / unexplained jargon / other), note. Hide the checker scores and the system name so they can't bias me.
- After I fill it in: scripts/08_validate.py picks the threshold on the 60 val sentences (max F1 for "unsupported"), then reports precision, recall and F1 on the 60 test sentences, plus the NLI-only and number-rule-only versions (a small ablation). Write results/checker_validation.csv and a precision-recall plot.
- Self-agreement: re-sample 30 of the labeled sentences into labels/relabel.csv in a shuffled order without my first labels. After I relabel them (at least a few days later), report Cohen's kappa.
- Freeze the threshold in config.
```

Earns: human evaluation (2), ablation (2), valid tuning without leakage (2).

---

## Step 9: Full evaluation (the only time test is used)

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 9: run every frozen system on Cochrane test and NIHR test.
- First check that config.yaml has no placeholders left (prompt variant, decoding, threshold). Stop and tell me if it does.
- Colab notebook 09_eval.ipynb: generate outputs for every system on both test sets, run the checker, run all metrics with BERTScore on.
- results/main_table.csv: one row per system x dataset with FKGL, words, digits/100, SARI, ROUGE-L, BERTScore and flag rate, each with a 95% CI. Also paired bootstraps for the claims we make (LLM vs BART, each vs extractive).
- Diagnostics (pick the ones that answer real questions, then plot them):
  - performance and flag rate by input length (tertiles)
  - by how number-heavy the source is
  - Cochrane test vs NIHR (does it generalize to a new source?)
  - how often each system drops the certainty language ("may", "uncertain", "low-certainty evidence") that the source had
- Save every figure to results/ with a numbered caption in results/captions.md.
```

Earns: evaluation and results (2), most of the evidence for interpretation (3).

---

## Step 10: Error analysis and interpretation notes

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 10: turn the results into claims.
- Count the error types from my labels by system (results/error_types.csv), and pull 2 clear examples of each type with their source sentence.
- Pick the demo cases: 3 representative successes, 1 ambiguous case and 2 failure cases (one the checker catches, one it misses). Save them to demo/cases.json with why each was chosen.
- Write results/interpretation.md: for each research question, what the results support, what they don't, and alternative explanations (for example, "lower SARI may reflect weak references rather than worse outputs"); limitations; and 4-5 specific next steps tied to specific failures.
- Every number in it must come from a results/ file. Name the file next to each claim.
```

Earns: interpretation and future work (3).

---

## Step 11: Demo

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 11: demo/app.py in Gradio.
- Input: paste an abstract, or pick from demo/cases.json and a few test examples.
- Choose a system (bart_base, the LLM, extractive), or compare two side by side.
- Output: the simplified paragraph with flagged sentences highlighted; clicking or expanding a flagged sentence shows its best-matching source sentence and the reason (low NLI score or a number mismatch); FKGL before and after; and a "reading aid, not medical advice" notice that is always visible.
- Two modes so the demo can't die during the presentation:
  cached mode serves the precomputed test outputs and checker results (no GPU or internet needed); live mode runs BART-base and the checker on CPU for pasted text. The LLM appears only in cached mode unless it's fast enough on my laptop.
- Don't log what users paste.
- demo/RUNBOOK.md: launch commands, the order of cases to show in the 3 minutes, and a checklist for recording the backup video.
Test that the app launches and every case in cases.json renders in cached mode.
```

Earns: interactive demo (4), including the failure-case requirement.

---

## Step 12: Report (ACL format)

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 12: the final report and README.
- Use the ACL LaTeX template. Sections: Introduction (problem + motivation), Related work, Data (construction, stats, licensing and ethics), Methods (preprocessing, each system, prompts in an appendix, model settings), Evaluation setup (splits, metrics, human labels, bootstrap), Results, Analysis (diagnostics + errors), Changes from the proposal (from the DECISIONS.md table), Discussion and future work, Limitations and ethics, References.
- Tables are generated from results/*.csv by a script (scripts/12_tables.py writes LaTeX), so the report can't drift from the code.
- Reuse proposal text only where it's still true, and update it everywhere else.
- Citations: Devaraj 2021, Guo 2024 (CELLS), Lewis 2020 (BART), the LLM's model card, the NLI model's card, Laban 2022 (SummaC), Xu 2016 (SARI), Zhang 2020 (BERTScore), Lin 2004 (ROUGE), Kincaid 1975, plus any AI tools used, per course policy.
- README.md: setup, data download, the exact commands for each step in order, which steps need Colab, and how to launch the demo.
```

Earns: report and reproducibility (6), plus the written half of categories 1–3.

---

## Step 13: Slides and oral defense prep

```
Follow CONTRACT.md in /home/claude/plainmed exactly. Read DECISIONS.md first.

Step 13: at most 5 slides for a 3-minute talk: (1) problem + what PlainMed does, (2) methods tried, (3) main results with CIs, (4) what the checker catches and misses, (5) takeaways + next steps. Numbers come only from results/. Add a 3-minute speaker script.
Then drill me for the oral defense: ask me one question at a time across the whole pipeline (data, leakage, each metric, why BART-base, prompt choice, checker threshold, kappa, a result I can't fully explain, and a new hypothetical like "what if the references were written by doctors?"). Grade each answer and tell me what I missed.
```

Earns: oral defense (5), slides for categories 1–3.

---

## Step 14: Consistency audit (do it last)

```
Follow CONTRACT.md in /home/claude/plainmed exactly.

Step 14: check that the report, slides, demo and code all agree.
- Every number in the report and slides matches a results/ file. List any mismatch.
- A fresh clone + README commands reproduces the baseline and metric numbers in my workspace.
- Demo cached outputs are the same files the tables were computed from.
- The "Proposal to final" table covers every difference between the proposal and the final system.
- Rubric checklist: go through all 6 categories and point to where each bullet is satisfied.
```

Earns: "consistency among report, code, demonstration and results" (6).

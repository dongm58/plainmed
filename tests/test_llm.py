"""
tests/test_llm.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks the prompted-LLM code without a GPU.
- Prompts, few-shot selection (train only, seeded, matches config), chat
  messages, output cleaning and the one-paragraph check.
- run_system with the stub model: writes records with the right fields and
  resumes without redoing finished ids.
- The prompt selection rule on made-up tables (normal, tie-break, fallback).
- HFGenerator on a tiny randomly initialized Qwen2 model with a hand-built
  tokenizer: batched generation with left padding gives exactly the same
  tokens as one-at-a-time generation, and the token-limit flag works. This
  runs the real transformers generate() path that Colab will use.
- The whole script (smoke, sensitivity, choose, run) end to end with --stub.

Input: data/processed/cochrane/ (skips the data-dependent tests if Step 1 has not been run).
Output: pass/fail results from pytest.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest

from plainmed.io import load_config, read_jsonl, repo_path
from plainmed.llm import (PROMPTS, StubGenerator, build_messages, clean_output, is_one_paragraph, render_prompt,
                          run_system, select_fewshot)

sys.path.insert(0, str(repo_path("scripts")))
TRAIN = repo_path("data/processed/cochrane/train.jsonl")
needs_data = pytest.mark.skipif(not TRAIN.exists(), reason="run scripts/01_data.py first")


def test_prompts_share_the_same_requirements():
    assert set(PROMPTS) == {"A", "B", "C"}
    for text in PROMPTS.values():
        assert "{source}" in text and "Results:\n{source}" in text
        lowered = text.lower()
        assert "paragraph" in lowered
        assert ("8th-grade" in lowered) or ("13-year-old" in lowered)
        assert ("certain" in lowered) or ("sure" in lowered)
        assert ("do not add" in lowered) or ("only use information" in lowered)


@needs_data
def test_fewshot_selection_matches_config_and_uses_train_only():
    config = load_config()
    shots = select_fewshot(read_jsonl(TRAIN), 3, config["seed"])
    assert [s["id"] for s in shots] == config["llm"]["fewshot_ids"]
    assert all(s["split"] == "train" for s in shots)
    assert all(len(s["source"].split()) <= 250 and len(s["target"].split()) <= 180 for s in shots)


def test_build_messages_structure():
    shots = [{"id": "s1", "source": "Shot source.", "target": "Shot summary."}]
    messages = build_messages("Real source.", "A", shots)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "Shot summary."
    assert messages[-1]["content"].endswith("Results:\nReal source.")
    assert [m["role"] for m in build_messages("x", "B")] == ["system", "user"]


def test_clean_output():
    assert clean_output("  Plain-language summary: Fewer people got sick.  ") == "Fewer people got sick."
    assert clean_output("**Summary:** Fewer people got sick.") == "Fewer people got sick."
    assert clean_output("Here is a plain-language version: Fewer people got sick.") == "Fewer people got sick."
    assert clean_output("Summary of the evidence is unclear.") == "Summary of the evidence is unclear."
    assert clean_output("Plain-language summary\nFewer people got sick.") == "Fewer people got sick."
    assert clean_output("Here's what the review found: fewer people got sick.") == "fewer people got sick."


def test_is_one_paragraph():
    assert is_one_paragraph("One paragraph. Two sentences.")
    assert not is_one_paragraph("First paragraph.\n\nSecond paragraph.")
    assert not is_one_paragraph("Findings:\n- less pain\n- more side effects")
    assert not is_one_paragraph("1. Less pain\n2. More side effects")


def test_run_system_with_stub_writes_and_resumes(tmp_path):
    examples = [{"id": f"cochrane-val-{i:04d}", "source": f"Sentence one of {i}. Sentence two. Sentence three."} for i in range(5)]
    path = tmp_path / "out.jsonl"
    run_system(examples[:3], "llm_zeroshot", "A", [], StubGenerator(), path, batch_size=2, log_every=0)
    run_system(examples, "llm_zeroshot", "A", [], StubGenerator(), path, batch_size=2, log_every=0)
    records = read_jsonl(path)
    assert sorted(r["id"] for r in records) == [e["id"] for e in examples]
    first = records[0]
    assert first["output"].startswith("Sentence one of") and first["output"].endswith("Sentence two.")
    assert set(first["meta"]) >= {"model", "variant", "fewshot_ids", "seconds", "input_tokens", "output_tokens",
                                  "hit_max_new_tokens", "one_paragraph", "raw_output"}


def _table(rows):
    df = pd.DataFrame(rows, columns=["variant", "shots", "sari", "fkgl", "one_paragraph_rate", "truncated_rate"])
    df["reference_fkgl_same_sample"] = 12.0
    return df


def test_choose_rule():
    from importlib import import_module
    choose = import_module("04_llm").choose_variant
    normal = _table([["A", "zeroshot", 38, 11, 1, 0], ["A", "fewshot", 40, 11, 1, 0],
                     ["B", "zeroshot", 41, 10, 1, 0], ["B", "fewshot", 43, 10, 1, 0],
                     ["C", "zeroshot", 45, 13, 1, 0], ["C", "fewshot", 46, 13, 1, 0]])
    result = choose(normal)
    assert result["chosen_variant"] == "B" and not result["used_tie_break"]
    tie = _table([["A", "zeroshot", 40, 11, 1, 0], ["A", "fewshot", 40, 11, 1, 0],
                  ["B", "zeroshot", 40.3, 9, 1, 0], ["B", "fewshot", 40.3, 9, 1, 0],
                  ["C", "zeroshot", 30, 9, 0.5, 0], ["C", "fewshot", 30, 9, 1, 0]])
    result = choose(tie)
    assert result["chosen_variant"] == "B" and result["used_tie_break"]
    none = _table([["A", "zeroshot", 40, 13, 1, 0], ["A", "fewshot", 40, 13, 1, 0],
                   ["B", "zeroshot", 42, 14, 1, 0], ["B", "fewshot", 42, 14, 1, 0],
                   ["C", "zeroshot", 41, 15, 1, 0.2], ["C", "fewshot", 41, 15, 1, 0]])
    result = choose(none)
    assert result["chosen_variant"] == "B" and result["all_variants_dropped_so_rule_relaxed"]


def _tiny_model_and_tokenizer():
    """A 2-layer random Qwen2 model and a word-level tokenizer with a chat template, built offline."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = ("<pad> <eos> <unk> <|system|> <|user|> <|assistant|> results : the a of in and to trials were people "
             "fewer got sick evidence low certainty pain less more we included rct risk ratio ci 0.82 . , "
             "rewrite following review").split()
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words))}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         pad_token_id=vocab["<pad>"], eos_token_id=vocab["<eos>"])
    model = Qwen2ForCausalLM(config)
    model.generation_config.eos_token_id = vocab["<eos>"]
    model.generation_config.pad_token_id = vocab["<pad>"]
    return model, tokenizer


def test_hf_generator_batched_equals_unbatched():
    pytest.importorskip("torch")
    from plainmed.llm import HFGenerator

    model, tokenizer = _tiny_model_and_tokenizer()
    gen = HFGenerator("tiny-random-qwen2", max_new_tokens=6, dtype="float32", device="cpu", tokenizer=tokenizer, model=model)
    sources = ["we included trials .", "fewer people got sick and the evidence of low certainty was in the review .", "less pain ."]
    prompts = [render_prompt([{"role": "user", "content": s}], tokenizer) for s in sources]
    batched = gen.generate(prompts)
    single = [gen.generate([p])[0] for p in prompts]
    assert [b["text"] for b in batched] == [s["text"] for s in single]
    assert [b["input_tokens"] for b in batched] == [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    assert all(b["output_tokens"] <= 6 for b in batched)
    assert all(b["hit_max_new_tokens"] == (b["output_tokens"] == 6) for b in batched)


@needs_data
def test_script_end_to_end_with_stub(tmp_path):
    script = str(repo_path("scripts/04_llm.py"))
    run = lambda *a: subprocess.run([sys.executable, script, *a, "--stub", "--out-root", str(tmp_path)],
                                    check=True, capture_output=True, text=True)
    run("smoke")
    run("sensitivity", "--batch-size", "16")
    table = pd.read_csv(tmp_path / "results" / "prompt_sensitivity.csv")
    assert len(table) == 6 and set(table["variant"]) == {"A", "B", "C"} and (table["n"] == 100).all()
    run("choose")
    choice = json.loads((tmp_path / "results" / "prompt_choice.json").read_text())
    assert choice["chosen_variant"] in {"A", "B", "C"}
    run("run", "--shots", "few", "--variant", choice["chosen_variant"], "--batch-size", "32")
    records = read_jsonl(tmp_path / "outputs" / "llm_fewshot" / "cochrane_val.jsonl")
    assert len(records) == 411 and records[0]["meta"]["fewshot_ids"] == load_config()["llm"]["fewshot_ids"]

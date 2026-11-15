"""
tests/test_checker.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks the faithfulness checker.
- 15 number cases: digits vs words, thousands separators (commas and spaces),
  rates out of 100, fractions, decimals, the 0/1 exception, and real misses.
- Source pieces include neighboring-sentence pairs.
- check_output with the stub NLI model: supported sentences pass, invented
  numbers and unsupported sentences are flagged, the best source piece is kept.
- HFNLI on a tiny randomly initialized DeBERTa-v2 classifier built offline
  (the real transformers path Colab uses): returns one probability per pair.
- The script end to end with --stub.

Input: data/processed/ for the script test (skipped if Step 1 has not been run).
Output: pass/fail results from pytest.
"""

import subprocess
import sys

import pandas as pd
import pytest

from plainmed.checker import StubNLI, check_output, extract_numbers, source_pieces, unsupported_numbers
from plainmed.io import read_jsonl, repo_path, write_jsonl

SOURCE = ("We included 25 RCTs involving 4,788 participants. The rate of response was 17% (95% CI 10 to 28%) "
          "with interferon 3 MU thrice weekly versus 3% in controls. In 13 451 participants, deaths did not differ. "
          "Twelve trials reported adverse events.")
SOURCE_NUMBERS = extract_numbers(SOURCE, generous=True)


@pytest.mark.parametrize("sentence,missing", [
    ("Twenty-five trials with 4788 people were included.", []),
    ("There were 25 trials and 4,788 people.", []),
    ("About 17 out of every 100 people responded.", []),
    ("About 17 in 100 people responded, compared with 3 in 100.", []),
    ("Interferon was given three times a week.", []),
    ("It was given thrice weekly.", []),
    ("The study had 13,451 participants.", []),
    ("Twelve trials looked at side effects.", []),
    ("One trial found a benefit.", []),
    ("Responses ranged from 10% to 28%.", []),
    ("Fifty-four trials were included.", [54.0]),
    ("About 30% fewer people died.", [30.0]),
    ("The study ran in 2019.", [2019.0]),
    ("About 20 out of 100 people responded.", [20.0]),
    ("Response was 1.7 times higher.", [1.7]),
])
def test_number_cases(sentence, missing):
    assert unsupported_numbers(sentence, SOURCE_NUMBERS) == missing


def test_source_pieces_include_pairs():
    pieces = source_pieces("First sentence here. Second sentence here. Third one.")
    assert len(pieces) == 3 + 2
    assert pieces[3] == (0, "First sentence here. Second sentence here.")


def test_check_output_flags_and_keeps_best_source():
    output = "We included 25 RCTs. Fifty-four trials were included. Interferon cures cancer forever."
    result = check_output(output, SOURCE, StubNLI(), threshold=0.5)
    first, second, third = result["sentences"]
    assert not first["flag"] and first["best_source_idx"] == 0
    assert second["number_mismatch"] and second["flag"]
    assert third["low_nli"] and third["flag"]
    assert result["n_sentences"] == 3 and result["n_flagged"] == 2
    assert result["flag_rate"] == pytest.approx(2 / 3)


def test_check_output_empty():
    result = check_output("", SOURCE, StubNLI())
    assert result["n_sentences"] == 0 and result["flag_rate"] == 0.0


def _tiny_nli():
    """A 2-layer random DeBERTa-v2 sequence classifier with a word-level tokenizer, built offline."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, PreTrainedTokenizerFast

    words = "[PAD] [UNK] [CLS] [SEP] we included trials people the of 25 17 response rate interferon cures cancer .".split()
    vocab = {w: i for i, w in enumerate(words)}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    raw.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                                       special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    torch.manual_seed(0)
    config = DebertaV2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=64, max_position_embeddings=128, num_labels=3,
                             id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                             label2id={"entailment": 0, "neutral": 1, "contradiction": 2}, pad_token_id=0)
    return DebertaV2ForSequenceClassification(config).eval(), tok


def test_hfnli_tiny_model_gives_probabilities():
    pytest.importorskip("torch")
    from plainmed.checker import HFNLI

    model, tok = _tiny_nli()
    nli = HFNLI.__new__(HFNLI)
    nli.name, nli.device, nli.tokenizer, nli.model = "tiny", "cpu", tok, model
    nli.entail_index, nli.batch_size, nli.max_length, nli.truncated = 0, 2, 16, 0
    pairs = [("we included 25 trials .", "we included trials ."), ("the response rate .", "interferon cures cancer ."),
             ("people .", "the rate .")]
    scores = nli.entailment(pairs)
    assert len(scores) == 3 and all(0.0 <= s <= 1.0 for s in scores)
    result = check_output("We included trials. Interferon cures cancer.", "We included 25 trials. The response rate.", nli)
    assert result["n_sentences"] == 2


needs_data = pytest.mark.skipif(not repo_path("data/processed/cochrane/val.jsonl").exists(), reason="run scripts/01_data.py first")


@needs_data
def test_script_end_to_end_with_stub(tmp_path):
    val = read_jsonl(repo_path("data/processed/cochrane/val.jsonl"))[:5]
    path = tmp_path / "outputs" / "llm_zeroshot" / "cochrane_val.jsonl"
    write_jsonl(path, [{"id": e["id"], "system": "llm_zeroshot", "output": e["target"], "meta": {}} for e in val])
    subprocess.run([sys.executable, str(repo_path("scripts/07_check.py")), "--inputs", str(path), "--stub",
                    "--out-root", str(tmp_path)], check=True, capture_output=True, text=True)
    checked = read_jsonl(tmp_path / "checked" / "llm_zeroshot" / "cochrane_val.checked.jsonl")
    assert len(checked) == 5 and all("sentences" in c and "flag_rate" in c for c in checked)
    summary = pd.read_csv(tmp_path / "results" / "checker_summary.csv")
    assert summary.loc[0, "n_outputs"] == 5 and summary.loc[0, "threshold"] == 0.5

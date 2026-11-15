"""
plainmed/checker.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: The faithfulness checker. For every sentence of a system's
output it asks two questions and flags the sentence ("check this") if either
answer is bad.

1. Is it supported by the source? An NLI (natural language inference) model
   scores how strongly each piece of the source entails the output sentence.
   The pieces are every source sentence and every pair of neighboring source
   sentences, because one plain-language sentence often merges two source
   sentences. The sentence's score is the highest entailment probability over
   all pieces (the zero-shot SummaC idea, Laban et al. 2022), and the piece
   that gave it is kept so the demo can show "here is where the source says
   this". Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.

2. Does it contain a number the source does not? A rule extracts numbers from
   both texts (digits, decimals, percentages, thousands separators and number
   words such as "twenty-five") and flags output numbers missing from the
   source. Two deliberate exceptions, to avoid false alarms: 0 and 1 are
   ignored ("one trial" is often a rewording of "a single study"), and 100 is
   ignored when it is the base of a rate ("17 out of every 100" for "17%").

flag = nli_score < threshold OR the sentence has an unsupported number.
The threshold starts at 0.5 and is tuned on hand-labeled validation sentences in Step 8.

Input: a system output and its source abstract.
Output: per-sentence scores, the best supporting source text, number problems,
flags, and the share of flagged sentences.
"""

import re

from plainmed.text import split_sentences

UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
MULTIPLES = {"once": 1, "twice": 2, "thrice": 3, "half": None, "dozen": 12}

NUMBER_WORD = re.compile(
    r"\b(" + "|".join(sorted(TENS, key=len, reverse=True)) + r")(?:[-\s](" + "|".join(sorted(UNITS, key=len, reverse=True)) + r"))?\b"
    r"|\b(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r"|once|twice|thrice|dozen)\b",
    re.IGNORECASE,
)
DIGITS = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\w.])-?\d+(?:\.\d+)?")
SPACE_THOUSANDS = re.compile(r"(?<![\w.])\d{1,3}(?: \d{3})+(?![\d.,])")
RATE_BASE = re.compile(r"\b(?:out of (?:every )?|in (?:every )?|per |of every )100\b", re.IGNORECASE)
IGNORED = {0.0, 1.0}


def _to_number(token: str) -> float | None:
    """Parse a digit token like '4,788', '0.82' or '-1.09' into a float."""
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def extract_numbers(text: str, generous: bool = False) -> set[float]:
    """All numbers mentioned in a text, as floats, from digits and from number words.

    With generous=True (used for the source) numbers written with spaces as
    thousands separators ("13 451") are also added joined (13451), so an output
    that writes "13,451" is not flagged.
    """
    numbers = set()
    if generous:
        for match in SPACE_THOUSANDS.finditer(text):
            numbers.add(float(match.group(0).replace(" ", "")))
    for match in DIGITS.finditer(text):
        value = _to_number(match.group(0))
        if value is not None:
            numbers.add(abs(value))
    for match in NUMBER_WORD.finditer(text):
        tens, unit, single = match.group(1), match.group(2), match.group(3)
        if tens:
            numbers.add(float(TENS[tens.lower()] + (UNITS[unit.lower()] if unit else 0)))
        elif single:
            word = single.lower()
            value = UNITS.get(word, MULTIPLES.get(word))
            if value is not None:
                numbers.add(float(value))
    return numbers


def unsupported_numbers(sentence: str, source_numbers: set[float]) -> list[float]:
    """Numbers in an output sentence that never appear in the source (after the 0, 1 and rate-base exceptions)."""
    found = extract_numbers(sentence)
    if RATE_BASE.search(sentence):
        found.discard(100.0)
    return sorted(n for n in found if n not in source_numbers and n not in IGNORED)


def source_pieces(source: str) -> list[tuple[int, str]]:
    """Every source sentence and every pair of neighboring sentences, with the index of the first sentence."""
    sentences = split_sentences(source)
    pieces = [(i, s) for i, s in enumerate(sentences)]
    pieces += [(i, sentences[i] + " " + sentences[i + 1]) for i in range(len(sentences) - 1)]
    return pieces


class StubNLI:
    """Fake NLI model for tests: entailment = share of the hypothesis's words found in the premise."""

    name = "stub"

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for premise, hypothesis in pairs:
            p = set(re.findall(r"[a-z0-9]+", premise.lower()))
            h = re.findall(r"[a-z0-9]+", hypothesis.lower())
            scores.append(sum(w in p for w in h) / len(h) if h else 0.0)
        return scores


class HFNLI:
    """Entailment probabilities from a Hugging Face NLI model, in batches."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "entailment" not in labels:
            raise ValueError(f"model labels {self.model.config.id2label} have no 'entailment'")
        self.entail_index = labels["entailment"]
        self.batch_size = batch_size
        self.max_length = max_length
        self.truncated = 0

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        scores = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            enc = self.tokenizer([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                                 max_length=self.max_length, padding=True, return_tensors="pt",
                                 return_overflowing_tokens=False)
            self.truncated += int((enc["attention_mask"].sum(dim=1) >= self.max_length).sum())
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            scores += torch.softmax(logits.float(), dim=-1)[:, self.entail_index].tolist()
        return scores


def check_output(output: str, source: str, nli, threshold: float = 0.5) -> dict:
    """Score every output sentence against the source and flag the doubtful ones."""
    sentences = split_sentences(output)
    pieces = source_pieces(source)
    source_numbers = extract_numbers(source, generous=True)
    pairs = [(piece, sentence) for sentence in sentences for _, piece in pieces]
    scores = nli.entailment(pairs) if pairs else []
    results = []
    for j, sentence in enumerate(sentences):
        row = scores[j * len(pieces):(j + 1) * len(pieces)]
        best = max(range(len(row)), key=row.__getitem__) if row else None
        missing = unsupported_numbers(sentence, source_numbers)
        nli_score = float(row[best]) if row else 0.0
        results.append({
            "text": sentence,
            "nli_score": round(nli_score, 4),
            "best_source_idx": pieces[best][0] if row else None,
            "best_source_text": pieces[best][1] if row else "",
            "unsupported_numbers": missing,
            "number_mismatch": bool(missing),
            "low_nli": nli_score < threshold,
            "flag": nli_score < threshold or bool(missing),
        })
    flagged = sum(r["flag"] for r in results)
    return {"sentences": results, "n_sentences": len(results), "n_flagged": flagged,
            "flag_rate": flagged / len(results) if results else 0.0}

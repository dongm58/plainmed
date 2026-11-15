"""
plainmed/text.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Small text utilities shared by every step.
- normalize(): Unicode NFKC plus whitespace cleanup, applied to every source and
  target once in Step 1 so all systems see identical input.
- split_sentences(): a rule-based sentence splitter built for medical abstracts.
  It never splits inside decimals ("0.82", "95% CI 0.70 to 0.96") or after
  common abbreviations ("e.g.", "vs.", "et al."). A rule-based splitter was
  chosen over spaCy so the same deterministic splitter runs in my workspace,
  in Colab and on the laptop without downloading a language model.
- mask_numbers(): replaces every number with NUM before readability scoring,
  because textstat counts decimal points as sentence ends (see DECISIONS.md).
- word_count(), digit_count(), digits_per_100(): simple length and number-density counts.

Input: raw strings.
Output: cleaned strings, lists of sentences, and counts.
"""

import re
import unicodedata

ABBREVIATIONS = {
    "e.g", "i.e", "vs", "etc", "al", "approx", "ca", "cf", "fig", "figs", "no", "nos",
    "dr", "mr", "mrs", "ms", "prof", "st", "vol", "min", "max", "resp", "incl", "est",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
WORD = re.compile(r"\S+")


def normalize(text: str) -> str:
    """Apply Unicode NFKC normalization and collapse all whitespace runs to single spaces."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _ends_with_abbreviation(chunk: str) -> bool:
    """True if the chunk ends in a period that belongs to an abbreviation or a single initial."""
    match = re.search(r"([A-Za-z][A-Za-z.]*)\.[\"'”’)\]]*$", chunk)
    if not match:
        return False
    token = match.group(1).lower()
    return token in ABBREVIATIONS or len(token) == 1


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking on decimals or abbreviations."""
    text = normalize(text)
    if not text:
        return []
    pieces = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        chunk = text[start:match.start()]
        if _ends_with_abbreviation(chunk):
            continue
        pieces.append(text[start:match.end()].strip())
        start = match.end()
    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def mask_numbers(text: str) -> str:
    """Replace every number (including decimals and thousands separators) with the token NUM."""
    return NUMBER.sub("NUM", text)


def word_count(text: str) -> int:
    """Number of whitespace-separated tokens."""
    return len(WORD.findall(text))


def digit_count(text: str) -> int:
    """Number of digit characters."""
    return sum(ch.isdigit() for ch in text)


def digits_per_100(text: str) -> float:
    """Digit characters per 100 words, a simple measure of how statistic-heavy a text is."""
    words = word_count(text)
    return 100.0 * digit_count(text) / words if words else 0.0

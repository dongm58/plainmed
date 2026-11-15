"""
plainmed/baselines.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: The two baselines every real system has to beat.
- copy_source(): returns the source unchanged. Any system should beat it on
  readability; it also shows what SARI and ROUGE give for doing nothing.
- extractive(): keeps the k source sentences with the fewest digits, breaking
  ties by fewest words, and returns them in their original order. It tests the
  idea from our data exploration that simplification here is mostly dropping
  statistics: if this baseline does about as well as a neural model, the model
  is not adding much beyond deleting numbers.
Sentences come from plainmed.text.split_sentences, the same splitter the
checker and the demo use.

Input: a source abstract (string).
Output: the baseline's text, and for extractive() the indices of the kept sentences.
"""

from plainmed.text import digit_count, split_sentences, word_count


def copy_source(source: str) -> str:
    """Baseline 1: the source, unchanged."""
    return source


def extractive(source: str, k: int = 3) -> tuple[str, list[int]]:
    """Baseline 2: the k sentences with the fewest digits (ties: fewest words), in original order."""
    sentences = split_sentences(source)
    ranked = sorted(range(len(sentences)), key=lambda i: (digit_count(sentences[i]), word_count(sentences[i]), i))
    kept = sorted(ranked[:k])
    return " ".join(sentences[i] for i in kept), kept

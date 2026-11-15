"""
plainmed/sari.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: SARI (Xu et al., 2016), the standard text-simplification metric.
It compares the system output with both the source and the reference(s) and
averages three scores over 1- to 4-grams:
  keep    F1 of n-grams the system kept from the source that the reference also kept
  delete  precision of n-grams the system deleted that the reference also deleted
  add     F1 of n-grams the system added that the reference also added

This is a copy of the SARI code in Hugging Face's `evaluate` library
(metrics/sari/sari.py, Apache License 2.0, which itself follows Xu's original
code and the EASSE normalization), rewritten in this project's style with the
same logic. It is copied instead of calling evaluate.load("sari") because
that call downloads the metric script from the Hugging Face Hub at run time,
which fails offline and in my workspace. tests/test_metrics.py checks that it
reproduces the value documented in the evaluate library.

Input: a source text, a system output and a list of reference texts.
Output: a SARI score from 0 to 100.
"""

from collections import Counter

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

_TOKENIZER = Tokenizer13a()


def sari_normalize(text: str) -> str:
    """Lowercase and tokenize with sacrebleu's 13a tokenizer, as evaluate's SARI does."""
    return _TOKENIZER(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[str]:
    """All n-grams of a token list, as space-joined strings."""
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(sgrams: list[str], cgrams: list[str], rgramslist: list[list[str]], numref: int) -> tuple[float, float, float]:
    """Keep F1, delete precision and add F1 for one n-gram order (same formulas as evaluate's SARIngram)."""
    rgramcounter = Counter(g for rgrams in rgramslist for g in rgrams)
    sgramcounter_rep = Counter({g: c * numref for g, c in Counter(sgrams).items()})
    cgramcounter_rep = Counter({g: c * numref for g, c in Counter(cgrams).items()})

    keep_rep = sgramcounter_rep & cgramcounter_rep
    keep_good = keep_rep & rgramcounter
    keep_all = sgramcounter_rep & rgramcounter
    keep_tmp1 = sum(keep_good[g] / keep_rep[g] for g in keep_good)
    keep_tmp2 = sum(keep_good[g] for g in keep_good)
    keep_precision = keep_tmp1 / len(keep_rep) if len(keep_rep) > 0 else 1
    keep_recall = keep_tmp2 / sum(keep_all.values()) if len(keep_all) > 0 else 1
    keep = 0
    if keep_precision > 0 or keep_recall > 0:
        keep = 2 * keep_precision * keep_recall / (keep_precision + keep_recall)

    del_rep = sgramcounter_rep - cgramcounter_rep
    del_good = del_rep - rgramcounter
    del_tmp1 = sum(del_good[g] / del_rep[g] for g in del_good)
    delete = del_tmp1 / len(del_rep) if len(del_rep) > 0 else 1

    add = set(cgramcounter_rep) - set(sgramcounter_rep)
    add_good = add & set(rgramcounter)
    add_all = set(rgramcounter) - set(sgramcounter_rep)
    add_precision = len(add_good) / len(add) if len(add) > 0 else 1
    add_recall = len(add_good) / len(add_all) if len(add_all) > 0 else 1
    add_f1 = 0
    if add_precision > 0 or add_recall > 0:
        add_f1 = 2 * add_precision * add_recall / (add_precision + add_recall)

    return keep, delete, add_f1


def sari_sentence(source: str, output: str, references: list[str]) -> float:
    """SARI for one example, on a 0-1 scale, from already-normalized strings."""
    s1, c1 = source.split(" "), output.split(" ")
    r1s = [r.split(" ") for r in references]
    keeps, deletes, adds = [], [], []
    for n in range(1, 5):
        keep, delete, add = _sari_ngram(_ngrams(s1, n), _ngrams(c1, n), [_ngrams(r, n) for r in r1s], len(references))
        keeps.append(keep)
        deletes.append(delete)
        adds.append(add)
    return (sum(keeps) / 4 + sum(deletes) / 4 + sum(adds) / 4) / 3


def sari(source: str, output: str, references: list[str]) -> float:
    """SARI for one example on a 0-100 scale. The corpus score is the mean over examples."""
    return 100 * sari_sentence(sari_normalize(source), sari_normalize(output), [sari_normalize(r) for r in references])

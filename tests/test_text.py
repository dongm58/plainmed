"""
tests/test_text.py

Author: Michael Dong
Date: November 14, 2026
Project: PlainMed (COSC 72/272 final project)

What it does: Checks normalization, number masking and the sentence splitter.
The splitter test uses 10 real sentences copied from Cochrane abstracts
(test split examples 4 and 0, train example 0), chosen because they contain
decimals, confidence intervals, fractions, semicolons and quotes.

Input: none (test strings are written below).
Output: pass/fail results from pytest.
"""

from plainmed.text import digits_per_100, mask_numbers, normalize, split_sentences, word_count

REAL_SENTENCES = [
    "A total of 1831 participants were randomised to drain (915 participants) versus 'no drain' (916 participants) in 12 trials included in this review.",
    "Only two trials including 199 participants were of low risk of bias.",
    "The proportion of females ranged between 55.0% and 79.0% in the 11 trials that provided this information.",
    "There was no significant difference between the drain group (1/840) (adjusted proportion: 0.1%) and the 'no drain' group (2/841) (0.2%) (RR 0.41; 95% CI 0.04 to 4.37) in short-term mortality in the ten trials with 1681 participants reporting on this outcome.",
    "There was no significant difference in the quality of life between the two groups (one trial; 93 participants; SMD 0.22; 95% CI -0.19 to 0.63).",
    "Two trials met the inclusion criteria.",
    "One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner.",
    "Both trials were small, reported results inadequately, and were of low methodological quality.",
    "In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing.",
    "One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group.",
]


def test_splitter_on_real_abstract_sentences():
    assert split_sentences(" ".join(REAL_SENTENCES)) == REAL_SENTENCES


def test_splitter_keeps_abbreviations_together():
    text = "Smith et al. found no effect, e.g. on pain. The drug vs. placebo comparison was unclear. P = 0.03. Overall, evidence was low."
    assert split_sentences(text) == [
        "Smith et al. found no effect, e.g. on pain.",
        "The drug vs. placebo comparison was unclear.",
        "P = 0.03.",
        "Overall, evidence was low.",
    ]


def test_splitter_splits_before_digits_and_quotes():
    assert split_sentences('Results were mixed. 25 trials reported pain. "Very low" certainty applied.') == [
        "Results were mixed.", "25 trials reported pain.", '"Very low" certainty applied.'
    ]


def test_splitter_keeps_numbered_list_markers():
    text = "Three comparisons were made. 1. Chlorpromazine versus olanzapine: no clear difference. 2. Chlorpromazine versus risperidone: no difference. The mean age was 63. Most were women."
    assert split_sentences(text) == [
        "Three comparisons were made.",
        "1. Chlorpromazine versus olanzapine: no clear difference.",
        "2. Chlorpromazine versus risperidone: no difference.",
        "The mean age was 63.",
        "Most were women.",
    ]


def test_splitter_unit_ending_in_single_letter():
    text = "The mean dose of omega-3 PUFA used in the trials was 3.5 g/d. No trials with vascular events were identified."
    assert split_sentences(text) == [
        "The mean dose of omega-3 PUFA used in the trials was 3.5 g/d.",
        "No trials with vascular events were identified.",
    ]


def test_splitter_empty():
    assert split_sentences("   ") == []


def test_normalize():
    assert normalize("  risk ratio\n\n0.82 \t (ﬁnal)  ") == "risk ratio 0.82 (final)"


def test_mask_numbers():
    assert mask_numbers("RR 0.41; 95% CI 0.04 to 4.37 in 1,681 people") == "RR NUM; NUM% CI NUM to NUM in NUM people"


def test_counts():
    assert word_count("Fewer people got sick.") == 4
    assert digits_per_100("In 12 trials with 95 people") == 100.0 * 4 / 6
    assert digits_per_100("") == 0.0

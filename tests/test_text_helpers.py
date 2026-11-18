"""Tests for the text clean-up helpers used on Whisper and translator output.

@pytest.mark.parametrize runs one test function several times with different
inputs, so each case appears as its own line in the results.
"""
import pytest

from interpreter import (clean_translation, collapse_repetitions,
                         continuation_join, trailing_marks_length)


# ── continuation_join() ───────────────────────────────────────────────────

@pytest.mark.parametrize("before, new, expected", [
    # Both sides marked with "..." -> join, and strip the leading marks
    ("и тогда они...", "...включаться", (True, "включаться")),
    # The single-character ellipsis "…" counts too
    ("и тогда они…", "…включаться", (True, "включаться")),
    # Only one side marked -> no join, new text returned unchanged
    ("и тогда они...", "включаться", (False, "включаться")),
    ("Привет.", "...как дела", (False, "...как дела")),
    # Neither side marked -> no join
    ("Привет.", "Как дела?", (False, "Как дела?")),
    # Spaces around the marks don't stop a join
    ("и тогда они...  ", "  ...включаться", (True, "включаться")),
    # A space after the leading marks is stripped too
    ("и тогда они...", "... включаться", (True, "включаться")),
    # Mixed marks: "..." on one side, "…" on the other
    ("и тогда они...", "…включаться", (True, "включаться")),
    # Two dots is not an ellipsis
    ("и тогда они..", "...включаться", (False, "...включаться")),
    # Empty strings never join
    ("", "...включаться", (False, "...включаться")),
    ("и тогда они...", "", (False, "")),
])
def test_continuation_join(before, new, expected):
    assert continuation_join(before, new) == expected


# ── trailing_marks_length() ───────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("и тогда они...", 3),
    ("и тогда они… ", 2),       # the "…" and the trailing space
    ("и тогда они", 0),         # nothing to delete
    ("и тогда они ...  ", 6),   # spaces before and after the marks
    ("", 0),
    ("...", 3),                 # text that is nothing but marks
])
def test_trailing_marks_length(text, expected):
    assert trailing_marks_length(text) == expected


def test_trailing_marks_length_also_counts_a_single_full_stop():
    # Pins current behaviour: rstrip(".…") removes ANY trailing dots, so a
    # normal full stop counts as well. That's harmless only because the GUI
    # calls this after continuation_join() has said True. If that ever
    # changes, this test reminds you the full stop would be deleted.
    assert trailing_marks_length("Привет.") == 1


def test_join_and_trim_together_give_one_clean_line():
    # How the GUI uses the pair: trim the end of the old line, add the new.
    before, new = "и тогда они...", "...включаться"
    joined, text = continuation_join(before, new)
    assert joined
    line = before[:len(before) - trailing_marks_length(before)] + " " + text
    assert line == "и тогда они включаться"


# ── clean_translation() ───────────────────────────────────────────────────

def test_clean_translation_removes_invented_repeat():
    # A real case from the logs: one short word, a repeated invented sentence.
    assert clean_translation("алым.", "I'm so sorry. I'm so sorry.") == "I'm so sorry."


def test_clean_translation_keeps_a_normal_translation():
    assert clean_translation("Как дела?", "How are you?") == "How are you?"


def test_clean_translation_trusts_a_multi_sentence_source():
    # If the speaker said several sentences, a repeat may be genuine.
    source = "Да. Да."
    assert clean_translation(source, "Yes. Yes.") == "Yes. Yes."


# ── collapse_repetitions() ────────────────────────────────────────────────

def test_collapse_repetitions_leaves_normal_text_alone():
    text = "Я хочу пойти в магазин."
    assert collapse_repetitions(text) == text


def test_collapse_repetitions_shortens_a_word_loop():
    result = collapse_repetitions("А-а-а-а-а-а-а-а")
    assert len(result) < len("А-а-а-а-а-а-а-а")

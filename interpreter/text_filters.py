"""Checks and clean-ups for the text Whisper and the translators produce.

Pure functions: text in, text (or a verdict) out. They need no audio,
GPU or window, which makes them quick to test (tests/test_text_helpers.py).

- is_hallucination() / hallucination_reason(): text Whisper invents on
  silence or music ("Thank you for watching").
- suspicious_reasons(): a result that looks wrong for the audio it came from.
- continuation_join() / trailing_marks_length(): join a sentence that
  Whisper split across two phrases with "...".
- clean_translation() / collapse_repetitions(): remove repeats the
  translator or Whisper made up.
"""

from .config import (
    CONTINUATION_MARKS, HALLUCINATION_FRAGMENTS, HALLUCINATIONS, LOGPROB_FLOOR,
    MAX_COMPRESSION, MIN_CHARS_PER_SEC, SUSPECT_PHRASES)


def _words(text: str) -> list[str]:
    """The words of a text, lower-case, without punctuation (to compare two
    transcriptions by content)."""
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()


def suspicious_reasons(text: str, speech_sec: float, avg_logprob: float) -> list:
    """Why a final transcription looks wrong (empty list = looks fine)."""
    import zlib
    reasons = []
    normalized = " ".join("".join(c for c in text.lower() if c.isalnum() or c == " ").split())
    if speech_sec >= 1.5 and len(text) / speech_sec < MIN_CHARS_PER_SEC:
        reasons.append("too_short")
    if normalized in SUSPECT_PHRASES and speech_sec >= 1.5:
        reasons.append("stock_phrase")
    if avg_logprob < LOGPROB_FLOOR:
        reasons.append("low_confidence")
    if text and len(text) > 20:
        ratio = len(text.encode()) / len(zlib.compress(text.encode()))
        if ratio > MAX_COMPRESSION:
            reasons.append("repetitive")
    return reasons


def continuation_join(text_before: str, new_text: str):
    """Whisper marks a sentence cut between two phrases with "..." at the end
    of one and the start of the next ("и тогда они..." / "...включаться").
    Joining them on one line reads far better. Returns (joined, text):
    joined=True means remove the trailing "..." from text_before and append
    text (already stripped of its leading "...") after a space."""
    before = text_before.rstrip()
    new = new_text.lstrip()
    if before.endswith(CONTINUATION_MARKS) and new.startswith(CONTINUATION_MARKS):
        return True, new.lstrip(".… ")
    return False, new_text


def trailing_marks_length(text_before: str) -> int:
    """How many characters to delete from the end of text_before to remove
    its trailing "..." (and spaces) before joining a continuation."""
    return len(text_before) - len(text_before.rstrip().rstrip(".…").rstrip())


def clean_translation(source: str, translated: str) -> str:
    """Guard against the translator inventing text. Seen in the logs: the
    one-word fragment "алым." became "I'm so sorry. I'm so sorry." So:
    a sentence repeated back-to-back is kept once, and a translation far
    longer than a very short source is cut to its first sentence."""
    import re
    split = lambda t: re.split(r"(?<=[.!?…])\s+", t.strip())
    sentences = split(translated)
    if len(split(source)) > 1:
        return translated      # the speaker may really have repeated themselves
    kept = [x for i, x in enumerate(sentences)
            if i == 0 or x.strip().lower() != sentences[i - 1].strip().lower()]
    out = " ".join(kept)
    src_words, out_words = len(source.split()), len(out.split())
    if src_words <= 2 and out_words > 3 * src_words + 3 and len(kept) > 1:
        out = kept[0]
    return out


def collapse_repetitions(text: str) -> str:
    """Shorten runaway repetition: one word ("А-а-а-а-а-а…" ×50) to three
    repeats, and a repeated phrase ("я хочу быть человеком, я хочу быть
    человеком, …" ×20 in the logs, a Whisper loop on a song chorus) to two."""
    import re
    # Phrase loops first: 2-12 words repeated 3+ times -> keep two
    text = re.sub(r"(\b(?:\w+[\s,]+){1,11}?\w+)(?:[\s,.!?…-]+\1\b){2,}",
                  lambda m: f"{m.group(1)}, {m.group(1)}…", text, flags=re.IGNORECASE)
    def shorten(m):
        word, sep = m.group(1), m.group(2)
        return sep.join([word] + [word.lower()] * 2)
    return re.sub(r"\b(\w+)([-\s,.!?]+)(?:\1\2?){3,}\b", shorten, text, flags=re.IGNORECASE)


def is_music_loop(text: str) -> bool:
    """True for the loops Whisper produces on music or humming: one letter
    held ("Аааааааааа"), or one word repeated ("Спустя. Спустя. Спустя.
    Спустя." in the logs). Real speech almost never looks like this."""
    import re
    letters = [c for c in text.lower() if c.isalpha()]
    if len(letters) >= 8 and max(letters.count(c) for c in set(letters)) / len(letters) > 0.8:
        return True
    words = re.findall(r"\w+", text.lower())
    return len(words) >= 4 and len(set(words)) == 1


def hallucination_reason(text: str):
    """Why the text is treated as invented by Whisper, or None if it isn't:
    "empty", "subtitle credit" (a YouTube caption fragment such as
    "Субтитры сделал…"), "stock phrase" ("thank you", "you"… invented on
    silence) or "music loop" (one letter or word repeated)."""
    text_lower = text.lower()
    if any(f in text_lower for f in HALLUCINATION_FRAGMENTS):
        return "subtitle credit"
    normalized = "".join(ch for ch in text_lower if ch.isalnum() or ch == " ").strip()
    if not normalized:
        return "empty"
    if " ".join(normalized.split()) in HALLUCINATIONS:
        return "stock phrase"
    return "music loop" if is_music_loop(text) else None


def is_hallucination(text: str) -> bool:
    """True if the text is empty, a known stock phrase Whisper invents on silence
    ("thank you", "you"…), a subtitle-credit fragment or a music loop."""
    return hallucination_reason(text) is not None

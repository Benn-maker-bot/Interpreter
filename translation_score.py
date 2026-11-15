"""
translation_score.py - score the app's TRANSLATIONS, not just its recognition
============================================================================

WHAT THIS FILE IS FOR
---------------------
The simulator's WER (word error rate) says whether the app HEARD the right
words. It says nothing about whether the TRANSLATION, the thing the user
actually reads, was any good. In one test run, a Ukrainian clip came out as
"the kisses follow and sew the odor of the beasts" and WER couldn't show it.

This file fills that gap. The test sentences come from Google FLEURS, and
FLEURS has the SAME sentences in every language: each row of its tables
carries a sentence number ("id"), and the same id means the same sentence
in Russian, English, German … So for a Russian clip, FLEURS already holds a
human English translation of it: the "reference translation".

The app's translation is compared with that reference using chrF.

chrF ("character F-score")
--------------------------
Compares the two texts in short runs of characters (1 to 6 letters long),
so "organise" and "organize" still mostly match, and a word with the right
stem but the wrong ending earns partial credit. Score 0-100: 100 = the
same text. A rough guide for one sentence:
     60+   good: the meaning comes through, wording differs
    40-60  understandable, with mistakes
    <40    the meaning is largely lost
Two correct translations can word things differently, so even a perfect
translation rarely scores 100. Use the scores to COMPARE runs or settings,
not as absolute marks. chrF is calculated by the standard "sacrebleu"
package, so the numbers are comparable with published results.

HOW TO RUN IT (from the project folder)
---------------------------------------
    .venv\\Scripts\\python translation_score.py               the newest result file
    .venv\\Scripts\\python translation_score.py A.json B.json  these result files

The result files are in testdata\\sim_results\\. New simulator runs already
include the chrF scores (simulate.py calls score_rows() below); this script
also scores older files, which don't.

The FLEURS sentence tables download once into testdata\\fleurs_text\\.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import requests

from fetch_testdata import FLEURS, HF   # FLEURS: folder name -> language code

APP_DIR = Path(__file__).resolve().parent
CACHE = APP_DIR / "testdata" / "fleurs_text"     # Path: downloaded sentence tables
RESULTS = APP_DIR / "testdata" / "sim_results"   # Path: the simulator's result files
SPLITS = ("dev", "test", "train")                # FLEURS's three tables per language
# FOLDER: language code -> FLEURS folder, e.g. "ru" -> "ru_ru" (FLEURS reversed).
FOLDER = {lang: folder for folder, lang in FLEURS.items()}
# The simulator stores the target language by its menu name ("English").
# NAME_TO_CODE: menu name -> language code, for the languages FLEURS has here.
NAME_TO_CODE = {"English": "en", "Russian": "ru", "German": "de", "French": "fr",
                "Spanish": "es", "Italian": "it", "Ukrainian": "uk", "Polish": "pl",
                "Bulgarian": "bg", "Turkish": "tr"}


@lru_cache(maxsize=None)
def _table(lang):
    """One language's FLEURS sentences, both ways round.

    Returns:
        (by_text, by_id): two dicts.
          by_text  sentence -> (split, id), e.g. "Привет." -> ("dev", "1234")
          by_id    (split, id) -> sentence
    PYTHON: @lru_cache remembers the result per language, so each table is
    read only once per run.
    """
    by_text, by_id = {}, {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = CACHE / f"{FOLDER[lang]}_{split}.tsv"
        if not path.exists():
            r = requests.get(f"{HF}/google/fleurs/resolve/main/data/{FOLDER[lang]}/{split}.tsv",
                             timeout=120)
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
        for row in path.read_text(encoding="utf-8").splitlines():
            cols = row.split("\t")       # column 0 = sentence id, column 2 = the sentence
            if len(cols) > 2:
                key = (split, cols[0])
                by_text.setdefault(cols[2].strip(), key)
                by_id.setdefault(key, cols[2].strip())
    return by_text, by_id


def reference(text, src, tgt):
    """The human translation of a clip's text, or None if there isn't one.

    Inputs:
        text  str: the clip's correct text (the manifest's "text"). Songs
              hold two sentences joined by a space; each is looked up.
        src   str: its language code, e.g. "ru".
        tgt   str: the target language code, e.g. "en".

    Example:
        reference("У него также не было полномочий …", "ru", "en")
        -> "He also had no authority to …"
    """
    if src not in FOLDER or tgt not in FOLDER or src == tgt:
        return None
    src_text, _ = _table(src)
    _, tgt_by_id = _table(tgt)

    def one(sentence):
        key = src_text.get(sentence)
        return tgt_by_id.get(key) if key else None

    if (whole := one(text)) is not None:
        return whole
    # A song: try every space as the join between two sentences.
    # PYTHON: enumerate(text) gives (position, character) pairs.
    for i, ch in enumerate(text):
        if ch == " ":
            a, b = one(text[:i]), one(text[i + 1:])
            if a and b:
                return f"{a} {b}"
    return None


def chrf(hyp, ref):
    """chrF score of one translation, 0-100 (see the top of the file)."""
    from sacrebleu.metrics import CHRF
    return round(CHRF().sentence_score(hyp or "", [ref]).score, 1)


def score_rows(rows, tgt_name):
    """Add "ref_translation" and "chrf" to each result row (in place).

    Inputs:
        rows      list of dicts: the simulator's per-clip results.
        tgt_name  str: the target language's menu name, e.g. "English".
    Rows without a reference (unknown sentence, same language, live audio)
    get chrf None. Network or file problems are reported once, not raised,
    so scoring can never break a simulation run.
    """
    tgt = NAME_TO_CODE.get(tgt_name)
    for r in rows:
        r["ref_translation"], r["chrf"] = None, None
        if not tgt or not r.get("lang") or not r.get("ref"):
            continue
        try:
            ref = reference(r["ref"], r["lang"], tgt)
        except Exception as e:                      # no internet, FLEURS moved, …
            print(f"  translation scoring skipped: {e}")
            return rows
        if ref:
            r["ref_translation"], r["chrf"] = ref, chrf(r.get("translation"), ref)
    return rows


def summary(rows):
    """Average chrF of rows that have one, as text, e.g. "54.2 (20)", or "-"."""
    s = [r["chrf"] for r in rows if r.get("chrf") is not None]
    return f"{sum(s) / len(s):.1f} ({len(s)})" if s else "-"


def report(path):
    """Score one result file and print chrF per kind and language, and the
    worst translations."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = score_rows(d["clips"], d["args"].get("tgt", "English"))
    print(f"\n{Path(path).name}  data={d['args'].get('data')}  "
          f"{d['args'].get('src')} -> {d['args'].get('tgt')}")
    print("  chrF, average (clips scored), higher is better:")
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        langs = sorted({r["lang"] for r in rs})
        per = "  ".join(f"{l} {summary([r for r in rs if r['lang'] == l]).split()[0]}" for l in langs)
        print(f"    {kind:<7} all {summary(rs):<12} {per}")
    worst = sorted((r for r in rows if r["chrf"] is not None), key=lambda r: r["chrf"])[:5]
    if worst:
        print("  Worst translations:")
        for r in worst:
            print(f"    {r['chrf']:5.1f}  {r['file']}\n"
                  f"           app: {r['translation'][:90]!r}\n"
                  f"           ref: {r['ref_translation'][:90]!r}")


if __name__ == "__main__":
    # Default: the newest result file (live runs have no correct text, so skip them).
    files = sys.argv[1:] or sorted(p for p in RESULTS.glob("*.json") if "-live" not in p.name)[-1:]
    for f in files:
        report(RESULTS / f if not Path(f).exists() else f)

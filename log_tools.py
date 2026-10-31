"""
log_tools.py — analyse the Voice Interpreter's log files
=========================================================

The interpreter writes a detailed log (interpreter.log.jsonl: one JSON object
per line) and saves the audio of recent phrases in recordings/. This script
reads them back to answer two questions: "how fast is it?" and "how accurate
is it?". It does not change anything in the app.

Run it from the project folder with the project's Python, for example:

    .venv\\Scripts\\python log_tools.py                 write interpreter.log.summary.json
    .venv\\Scripts\\python log_tools.py report          speed breakdown of every stage
    .venv\\Scripts\\python log_tools.py report 2026-10-31T14:00
                                                    …only entries from that time on
    .venv\\Scripts\\python log_tools.py accuracy        check the recognised words
    .venv\\Scripts\\python log_tools.py accuracy 2026-10-31T14:00 large-v3

report
    Reads the "phrase_trace" lines (one per finished phrase), "phrase_displayed"
    lines and the 10-second stats lines, and prints the median (p50), 95th
    percentile (p95) and maximum time of each stage: waiting in the queue,
    speech detection, the GPU, getting the text on screen, and so on, plus
    resource use (CPU, GPU, memory) and a list of findings.

accuracy
    There is no human transcript of real calls, so a larger, slower model
    (large-v3 by default) re-transcribes each saved phrase recording with
    careful settings. Its text is the "reference". The app's text is compared
    word by word: the word error rate (WER) is the share of reference words
    that the app got wrong, missed or added. The reference is cached next to
    each recording (.ref.json) so the big model only runs once per phrase.
    Results are broken down by model and by error type, and every phrase
    with differences is listed so you can see exactly what went wrong.

Timestamps in the log look like 2026-10-31T14:00:00.000+00:00; any prefix of
one (such as 2026-10-31T14) works as a start time.
"""

import json
import gzip
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG_PATH = APP_DIR / "interpreter.log.jsonl"
ERROR_LOG_PATH = APP_DIR / "interpreter.error.log.jsonl"
SUMMARY_PATH = APP_DIR / "interpreter.log.summary.json"


def iter_log_files(base_path: Path):
    """List the current log file plus its rotated, gzipped older copies."""
    files = []
    if base_path.exists():
        files.append(base_path)
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*.gz")))
    return files


def read_jsonl(path: Path):
    """Yield one dictionary per line of a .jsonl or .jsonl.gz file, skipping bad lines."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_summary():
    """Count entries by level and event, list recent problems, and write the
    summary JSON file (same as the app's Export Log Summary button).
    """
    # A Counter is a dict that counts: levels['ERROR'] += 1 starts at 0.
    levels = Counter()
    events = Counter()
    issues = []
    total = 0
    # Go through every log entry (newest file and the older gzipped ones),
    # counting levels and event names, and keeping warnings/errors aside.
    for path in iter_log_files(LOG_PATH):
        for entry in read_jsonl(path):
            total += 1
            levels[entry.get("level", "UNKNOWN")] += 1
            events[entry.get("event", "unknown")] += 1
            if entry.get("level") in {"WARNING", "ERROR", "CRITICAL"}:
                issues.append({
                    "ts": entry.get("ts"),
                    "event": entry.get("event"),
                    "message": entry.get("message"),
                })
    # Put the counts together and save them as a nicely indented JSON file.
    summary = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "log_path": str(LOG_PATH),
        "error_log_path": str(ERROR_LOG_PATH),
        "total_entries": total,
        "levels": dict(levels),
        "top_events": dict(events.most_common(25)),
        "recent_issues": issues[-25:],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _pct(values, fraction):
    """The value below which a fraction `q` of the values fall (q=0.5 is the
    median, q=0.95 the 95th percentile). None values are ignored.
    """
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(len(v) * fraction))]


def _row(label, values, unit="ms"):
    """Format one line of the report: count, median, 95th percentile and maximum."""
    vals = [x for x in values if x is not None]
    if not vals:
        return f"  {label:<34} (no data)"
    return (f"  {label:<34} n={len(vals):<5} p50={_pct(vals, .5):>8.1f}  "
            f"p95={_pct(vals, .95):>8.1f}  max={max(vals):>8.1f} {unit}")


def build_report(since=None):
    """Latency/efficiency breakdown of the speech -> text -> translation
    pipeline from the phrase_trace, phrase_displayed and *_stats log lines.
    `since` is an ISO prefix such as '2026-10-31T14:00' (UTC offset as logged)."""
    # ── Step 1: read every log entry, optionally only those from `since` on.
    #    (Timestamps are text like '2026-10-31T14:05:…', so comparing them
    #    as text also compares them in time order.)
    entries = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)]
    if since:
        entries = [e for e in entries if e.get("ts", "") >= since]
    # ── Step 2: sort the entries by event name, e.g. by['phrase_trace'] is
    #    the list of all phrase_trace entries.
    #    tr   = one entry per finished phrase (timings, text, scores …)
    #    displayed = one entry per phrase previews_processed on screen (screen delays)
    by_event = {}
    for e in entries:
        by_event.setdefault(e.get("event"), []).append(e)
    traces, displayed = by_event.get("phrase_trace", []), by_event.get("phrase_displayed", [])
    # col(rows, 'x') = the list of each entry's 'x' value (a 'column').
    # gpu = encode + decode time: the two GPU stages added together, because
    # GPU work is asynchronous and can be counted in either one.
    column = lambda rows, k: [r.get(k) for r in rows]
    gpu_ms = [(r.get("encode_ms") or 0) + (r.get("decode_ms") or 0) for r in traces if "decode_ms" in r]
    out = [f"Pipeline report  ({entries[0]['ts'][:19]} .. {entries[-1]['ts'][:19]}, {len(entries)} log lines)"
           if entries else "No log entries."]

    # ── Step 3: the finished-phrase section. `out` collects the report's
    #    lines; they are joined into one text at the end. _row() prints
    #    count, median (p50), 95th percentile (p95) and maximum.
    out.append("\nFinished phrases (speech -> final text)")
    out.append(f"  phrases: {len(traces)}   cut reasons: {dict(Counter(column(traces, 'reason')))}   "
               f"models: {dict(Counter(column(traces, 'model')))}   "
               f"translation: {dict(Counter(column(traces, 'translation_engine')))}")
    out.append(_row("local translation (Opus-MT)", [r.get("mt_ms") for r in traces if r.get("mt_ms")]))
    out.append(f"  suspicious results: {sum(bool(r.get('suspicious')) for r in traces)}   "
               f"retried: {sum(bool(r.get('retry')) for r in traces)}   "
               f"retry used: {sum(bool((r.get('retry') or {}).get('used')) for r in traces)}   "
               f"still suspicious: {sum(bool(r.get('still_suspicious')) for r in traces)}   "
               f"context resets: {sum(bool(r.get('context_reset')) for r in traces)}   "
               f"no-speech drops: {sum(bool(r.get('dropped_text')) for r in traces)}")
    out.append(_row("retry cost", [(r.get("retry") or {}).get("ms") for r in traces if r.get("retry")]))
    # One report row per stage of the pipeline, in the order a phrase goes
    # through them: capture, queue, VAD, features, GPU, screen …
    for label, vals, unit in [
            ("audio length", [x * 1000 if x else None for x in column(traces, "audio_sec")], "ms"),
            ("speech start -> phrase end", column(traces, "capture_ms"), "ms"),
            ("phrase end -> worker picks up", column(traces, "queue_wait_ms"), "ms"),
            ("  VAD trim (CPU)", column(traces, "vad_ms"), "ms"),
            ("  mel features (CPU)", column(traces, "mel_ms"), "ms"),
            ("  GPU lock wait", column(traces, "lock_wait_ms"), "ms"),
            ("  GPU encode+decode", gpu_ms, "ms"),
            ("worker total", column(traces, "process_ms"), "ms"),
            ("result -> on screen (UI)", column(displayed, "ui_ms"), "ms"),
            ("PAUSE -> FINAL TEXT ON SCREEN", column(displayed, "speech_end_to_text_ms"), "ms"),
            ("SPEECH START -> FIRST PREVIEW", column(displayed, "first_preview_ms"), "ms"),
            ("pause -> online translation shown",
             column(by_event.get("translation_displayed", []), "speech_end_to_translation_ms"), "ms"),
            ("tokens generated", column(traces, "tokens"), "tok"),
            ("speech level", column(traces, "rms_dbfs"), "dBFS"),
            ("peak level", column(traces, "peak_dbfs"), "dBFS"),
            ("AGC gain applied", column(traces, "agc_gain_db"), "dB"),
            ("mean VAD speech probability", column(traces, "vad_mean"), ""),
            ("previews per phrase", column(traces, "previews"), "")]:
        out.append(_row(label, vals, unit))

    # ── Step 4: live previews. preview_stats lines are written every 10 s;
    #    each holds its own p50/p95/max, so we summarise those values.
    preview_stats = by_event.get("preview_stats", [])
    out.append("\nLive previews (10 s windows; p50 of each window's p50)")
    for k in ("age_ms", "process_ms", "encode_ms", "decode_ms", "vad_ms"):
        out.append(_row(k, [w[k]["p50"] for w in preview_stats if isinstance(w.get(k), dict)]))
    previews_processed = sum(w["process_ms"]["n"] for w in preview_stats if isinstance(w.get("process_ms"), dict))
    out.append(f"  previews processed: {previews_processed}   skipped as stale: "
               f"{sum(w.get('skipped_stale', 0) for w in preview_stats)}   not shown (empty/unchanged): "
               f"{sum(w['not_shown']['n'] for w in preview_stats if isinstance(w.get('not_shown'), dict))}   "
               f"preview model: {dict(Counter(w.get('preview_model') for w in preview_stats))}")

    # ── Step 5: audio capture health. read_gap_ms is the time between two
    #    audio reads; it should be about one block (~32 ms). Much more means
    #    the capture loop fell behind and audio may have been lost.
    capture_stats = by_event.get("capture_stats", [])
    out.append("\nAudio capture (10 s windows; worst value per window)")
    for k in ("read_gap_ms", "loop_ms", "vad_ms", "backlog_frames"):
        out.append(_row(k + " (max)", [w[k]["max"] for w in capture_stats if isinstance(w.get(k), dict)],
                        "frames" if k == "backlog_frames" else "ms"))
    out.append(_row("level p50", [w["level_dbfs"]["p50"] for w in capture_stats
                                  if isinstance(w.get("level_dbfs"), dict)], "dBFS"))
    out.append(_row("speech frames", column(capture_stats, "speech_pct"), "%"))

    # ── Step 6: CPU, memory, GPU and how promptly the window updates.
    resource_stats = by_event.get("resource_stats", [])
    out.append("\nResources and UI (10 s windows)")
    for k, unit in (("process_cpu_pct", "%"), ("system_cpu_pct", "%"), ("rss_mb", "MB"),
                    ("gpu_util_pct", "%"), ("gpu_mem_used_mb", "MB"), ("gpu_sm_clock_mhz", "MHz"),
                    ("gpu_temp_c", "C")):
        out.append(_row(k, column(resource_stats, k), unit))
    out.append(_row("UI poll lag (p95 per window)",
                    [w["poll_lag_ms"]["p95"] for w in resource_stats if isinstance(w.get("poll_lag_ms"), dict)]))

    # ── Step 7: for each sensitivity change, compare the measured input level
    #    just before and just after it (it should not change: sensitivity
    #    only decides what counts as speech, not how loud the audio is).
    sensitivity_changes = by_event.get("sensitivity_changed", [])
    if sensitivity_changes:
        out.append("\nSensitivity changes vs measured input level (capture level p50, dBFS)")
        for e in sensitivity_changes:
            before = [w for w in capture_stats if w["ts"] < e["ts"] and isinstance(w.get("level_dbfs"), dict)]
            after = [w for w in capture_stats if w["ts"] > e["ts"] and isinstance(w.get("level_dbfs"), dict)]
            level_before = before[-1]["level_dbfs"]["p50"] if before else None
            level_after = after[0]["level_dbfs"]["p50"] if after else None
            out.append(f"  {e['ts'][11:19]}  sensitivity={e.get('value')}  "
                       f"threshold={e.get('vad_threshold')}  level before={level_before}  after={level_after}")

    # ── Step 8: automatic findings. Each rule checks for a known problem
    #    and adds a plain-language explanation if it is found.
    flags = []
    if (_pct(column(traces, "queue_wait_ms"), .95) or 0) > 100:
        flags.append("Phrases wait >100 ms for the worker: GPU busy with previews or backlog.")
    if any((w.get("read_gap_ms") or {}).get("max", 0) > 150 for w in capture_stats):
        flags.append("Audio reads stalled >150 ms: capture loop fell behind the device.")
    if (_pct([w["poll_lag_ms"]["p95"] for w in resource_stats if isinstance(w.get("poll_lag_ms"), dict)],
             .5) or 0) > 50:
        flags.append("UI thread is busy: results wait >50 ms before being drawn.")
    if preview_stats and sum(w.get("skipped_stale", 0) for w in preview_stats) > previews_processed:
        flags.append("More previews skipped than processed: preview model can't keep up.")
    if (_pct(column(traces, "agc_gain_db"), .5) or 0) > 20:
        flags.append("AGC is boosting >20 dB: input is very quiet, raise the source volume.")
    if sum(bool(r.get("still_suspicious")) for r in traces) > 0.1 * max(1, len(traces)):
        flags.append("Over 10% of phrases still look wrong after a retry: check the audio "
                     "(music, overlapping speech) or try a larger model.")
    if (_pct(column(traces, "peak_dbfs"), .95) or -100) > -0.5:
        flags.append("Peaks reach 0 dBFS: input is clipping, lower the source volume.")
    out.append("\nFindings")
    out.extend(f"  - {f}" for f in flags) if flags else out.append("  - nothing abnormal")
    return "\n".join(out)


def _words(text):
    """Split text into lower-case words for comparison (ё is treated as е, and
    punctuation is ignored).
    """
    import re
    text = (text or "").lower().replace("ё", "е")
    return re.findall(r"\w+", text)


def _align(ref_words, app_words):
    """Word edit distance with backtrace: (errors, missing words, substitutions)."""
    # This is the standard 'edit distance' method (Levenshtein), on words.
    # d is a table: d[i][j] = the fewest edits (insert, delete or replace a
    # word) needed to turn the first i reference words into the first j app
    # words. The answer for the whole texts ends up in the bottom-right cell.
    # Example: ref 'мы поедем домой', app 'мы едем домой' -> 1 edit.
    edits = [[0] * (len(app_words) + 1) for _ in range(len(ref_words) + 1)]
    # Edges of the table: turning i words into nothing takes i deletions,
    # and turning nothing into j words takes j insertions.
    for i in range(len(ref_words) + 1):
        edits[i][0] = i
    for j in range(len(app_words) + 1):
        edits[0][j] = j
    # Fill the table. Each cell takes the cheapest of three moves:
    #   from above (+1): the reference word is missing in the app's text
    #   from the left (+1): the app added an extra word
    #   diagonally (+0 if the words match, +1 if one replaced the other)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(app_words) + 1):
            edits[i][j] = min(edits[i - 1][j] + 1, edits[i][j - 1] + 1,
                          edits[i - 1][j - 1] + (ref_words[i - 1] != app_words[j - 1]))
    # Walk back from the bottom-right corner along the cheapest path to see
    # WHICH words were missing or replaced, not just how many errors.
    # The lists come out backwards, so they are reversed ([::-1]) at the end.
    i, j, missing, wrong = len(ref_words), len(app_words), [], []
    while i > 0 or j > 0:
        if i and j and edits[i][j] == edits[i - 1][j - 1] + (ref_words[i - 1] != app_words[j - 1]):
            if ref_words[i - 1] != app_words[j - 1]:
                wrong.append(f"{ref_words[i - 1]}→{app_words[j - 1]}")
            i, j = i - 1, j - 1
        elif i and edits[i][j] == edits[i - 1][j] + 1:
            missing.append(ref_words[i - 1])
            i -= 1
        else:
            j -= 1
    return edits[len(ref_words)][len(app_words)], missing[::-1], wrong[::-1]


# Text the reference model itself invents on music/silence (YouTube subtitle
# credits etc.); phrases whose reference contains these are skipped.
REF_HALLUCINATIONS = ("субтитр", "dimatorzok", "продолжение следует", "редактор",
                      "спасибо за просмотр", "подписывайтесь", "thanks for watching")


def _reference_text(model_holder, ref_model, wav_path, lang):
    """Reference transcript for one recording, cached next to it as .ref.json
    so the large model only ever runs once per recording."""
    cache = wav_path.with_suffix(".ref.json")
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("model") == ref_model:
            return cached["text"]
    if model_holder.get("model") is None:
        import ctranslate2
        from faster_whisper import WhisperModel
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        model_holder["model"] = WhisperModel(
            ref_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    # Read the WAV ourselves: faster-whisper's file decoder is incompatible
    # with the installed PyAV version.
    import wave
    import numpy as np
    with wave.open(str(wav_path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    segs, _ = model_holder["model"].transcribe(audio, language=lang, beam_size=5,
                                               vad_filter=False,
                                               condition_on_previous_text=False)
    text = " ".join(s.text for s in segs).strip()
    cache.write_text(json.dumps({"model": ref_model, "text": text}, ensure_ascii=False),
                     encoding="utf-8")
    return text


FILLER_WORDS = {"ну", "вот", "и", "а", "то", "так", "это", "же", "уж", "да", "um", "uh",
                "well", "so", "and"}


def _classify(ref_words, app_words, errors, missing, wrong):
    """Rough error category for one phrase, to show where accuracy is lost."""
    if not errors:
        return None
    if not app_words or errors >= 0.7 * len(ref_words):
        return "whole phrase lost/wrong"
    words = missing + [w.split("→")[0] for w in wrong]
    if words and all(w in FILLER_WORDS for w in words):
        return "filler words only"
    if words and all(w in (ref_words[0], ref_words[-1]) for w in words):
        return "first/last word (phrase cut)"
    return "word errors"


def build_accuracy(since=None, ref_model="large-v3"):
    """Re-transcribe saved phrase recordings with a larger reference model
    (beam search, no filtering) and compare with what the app logged, to find
    words and whole phrases the app failed to recognise. Results are broken
    down by recognition model and translation engine."""
    # Borrow two things from the app: the function that makes the NVIDIA
    # libraries loadable (so the reference model can use the GPU) and the
    # folder where phrase recordings are saved.
    import sys
    sys.path.insert(0, str(APP_DIR))
    from interpreter import _add_cuda_dll_dirs, RECORD_DIR
    _add_cuda_dll_dirs()

    # ── Step 1: find every finished phrase that has a saved recording.
    traces = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)
              if e.get("event") == "phrase_trace" and e.get("audio_file")
              and (not since or e.get("ts", "") >= since)]
    traces = [trace for trace in traces if (RECORD_DIR / trace["audio_file"]).exists()]
    if not traces:
        return "No phrase recordings found (they are saved from this version on)."

    # ── Step 2: set up the tallies.
    #    model_holder      keeps the reference model once loaded (loaded only if
    #                some recording has no cached reference yet)
    #    per_model   errors and words per (model / translation engine)
    #    categories  errors per kind of mistake (see _classify)
    model_holder = {}
    details, per_model, all_missing, categories = [], {}, Counter(), Counter()
    retries = Counter()
    total_errors = total_ref_words = missed_phrases = skipped = 0
    for trace in traces:
        # ── Step 3, for each phrase: get the reference text (what the bigger
        #    model heard) and split both texts into words.
        ref_text = _reference_text(model_holder, ref_model, RECORD_DIR / trace["audio_file"],
                                   trace.get("src_lang"))
        ref_words, app_words = _words(ref_text), _words(trace.get("text"))
        # Skip phrases where the reference itself is empty or invented (the big
        # model also hallucinates on music, e.g. subtitle credits).
        if not ref_words or any(h in ref_text.lower() for h in REF_HALLUCINATIONS):
            skipped += 1
            continue
        # Line the two word lists up (like a spell-checker's 'diff') to count
        # the errors and list which words were missing or wrong.
        errors, missing, wrong = _align(ref_words, app_words)
        category = _classify(ref_words, app_words, errors, missing, wrong)
        if category:
            categories[category] += errors
        if trace.get("retry"):
            retries["used" if trace["retry"].get("used") else "kept original"] += 1
        total_errors += errors
        total_ref_words += len(ref_words)
        missed_phrases += not app_words
        all_missing.update(missing)
        key = f"{trace.get('model')} / {trace.get('translation_engine') or '-'}"
        e, n, c = per_model.get(key, (0, 0, 0))
        per_model[key] = (e + errors, n + len(ref_words), c + 1)
        # Keep a detailed description of every phrase that had an error, so you
        # can read exactly what the app wrote and what was really said.
        if errors:
            details.append(f"  {trace['ts'][11:19]} seq={trace['seq']} model={trace.get('model')} "
                           f"reason={trace.get('reason')} errors={errors}/{len(ref_words)} "
                           f"no_speech={trace.get('no_speech_prob')} logprob={trace.get('avg_logprob')}")
            details.append(f"     category: {category}"
                           + (f"   retry: {trace['retry'].get('original_text')!r} -> "
                              f"{trace['retry'].get('text')!r} (used={trace['retry'].get('used')})"
                              if trace.get("retry") else ""))
            details.append(f"     app: {trace.get('text') or '(nothing)'}")
            details.append(f"     ref: {ref_text}")
            if trace.get("translation"):
                details.append(f"     en:  {trace['translation']}")
            if missing:
                details.append(f"     missing: {' '.join(missing)}")
            if wrong:
                details.append(f"     wrong:   {', '.join(wrong)}")

    # ── Step 4: the summary at the top, then the per-phrase details.
    #    WER = errors / reference words. :.1% formats 0.123 as 12.3%.
    out = [f"Accuracy check: {len(traces)} phrases, app vs reference model '{ref_model}'",
           f"  word error rate vs reference: {total_errors / max(1, total_ref_words):.1%} "
           f"({total_errors} errors / {total_ref_words} words)   phrases with no text at all: "
           f"{missed_phrases}   skipped (reference hallucinated/empty): {skipped}",
           "  by model / translation engine:"]
    out += [f"    {k:<26} WER {e / max(1, n):6.1%}  ({e} errors / {n} words, {c} phrases)"
            for k, (e, n, c) in sorted(per_model.items())]
    out.append("  errors by category: " + (", ".join(
        f"{c} {n}" for c, n in categories.most_common()) or "-"))
    out.append(f"  suspicious-result retries: {dict(retries) or 'none'}")
    out.append("  most often missed: " + (", ".join(
        f"{w}×{n}" for w, n in all_missing.most_common(10)) or "-"))
    out.append("  (the reference is itself a model, not a human transcript: small "
               "differences such as '10' vs 'десять' count as errors)\n")
    return "\n".join(out + details)



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        print(build_report(sys.argv[2] if len(sys.argv) > 2 else None))
    elif len(sys.argv) > 1 and sys.argv[1] == "accuracy":
        print(build_accuracy(sys.argv[2] if len(sys.argv) > 2 else None,
                             sys.argv[3] if len(sys.argv) > 3 else "large-v3"))
    else:
        summary = build_summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2))

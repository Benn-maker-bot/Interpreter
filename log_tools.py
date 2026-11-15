"""
log_tools.py - reports from the interpreter's log files
=======================================================

WHAT THIS FILE IS FOR
---------------------
While it runs, the interpreter writes down everything it does in a log
file, interpreter.log.jsonl. It also saves the audio of recent phrases in
the folder recordings/. This script READS those files back and turns them
into reports that answer questions like:
  * How fast is it? (Which step takes the time?)
  * How accurate is it? (Which words does it get wrong?)
  * Which settings work best?
  * Is it getting better from day to day?
It never changes anything in the app; it only reads.

WHAT A LOG LINE LOOKS LIKE
--------------------------
The log is "JSON Lines": every line is one JSON object (a set of
"name": value pairs, like a Python dict) describing one event. Example of
the most important kind, written once per finished phrase:

  {"ts": "2026-11-01T01:45:49.123+00:00", "level": "INFO",
   "event": "phrase_trace", "session_id": "3927d4c7-…", "seq": 12,
   "text": "Привет, как дела?", "translation": "Hi, how are you?",
   "audio_sec": 2.4, "queue_wait_ms": 73, "encode_ms": 142, "decode_ms": 39,
   "mt_ms": 31, "lang_used": "ru", "end_rule": "speech", …}

  ts          the time it happened ("timestamp"), in ISO format:
              year-month-day T hour:minute:second.milliseconds +timezone
  level       how important: INFO (normal), WARNING, ERROR, CRITICAL
  event       what happened, e.g. phrase_trace, phrase_displayed,
              capture_stats, resource_stats, music_mode_changed
  the rest    details that belong to that event

A timestamp prefix works as a start time for the reports: comparing the
text "2026-11-01T01:45:49" >= "2026-11-01T01" is the same as comparing the
times, because the most significant parts come first.

HOW TO RUN IT (from the project folder)
---------------------------------------
    .venv\\Scripts\\python log_tools.py                 write interpreter.log.summary.json
    .venv\\Scripts\\python log_tools.py report          speed breakdown of every stage
    .venv\\Scripts\\python log_tools.py report 2026-10-31T14:00
                                                    …only entries from that time on
    .venv\\Scripts\\python log_tools.py accuracy        check the recognised words
    .venv\\Scripts\\python log_tools.py accuracy 2026-10-31T14:00 large-v3
    .venv\\Scripts\\python log_tools.py compare         speed and accuracy per setting
    .venv\\Scripts\\python log_tools.py compare 2026-10-31T16:00 --ref
                                                    …also create missing references
    .venv\\Scripts\\python log_tools.py sensitivity     how sensitivity and input level
                                                    affect detection and accuracy
    .venv\\Scripts\\python log_tools.py trend           headline figures per day, real use
                                                    and simulations side by side

THE REPORTS
-----------
  (none)       a summary file: how many log lines of each level and event,
               and the 25 most recent warnings/errors.
  report       for every step a phrase goes through (detecting speech,
               waiting, the GPU, putting text on screen …) the typical time
               (median), a slow-case time (95th percentile) and the worst
               time; plus CPU/GPU use and a list of findings in plain words.
  accuracy     real calls have no correct transcript, so a bigger, slower
               Whisper model ("large-v3") listens to each saved phrase
               recording carefully. Its text is the "reference". The app's
               text is compared word by word with it: the word error rate
               (WER) is the share of reference words the app got wrong,
               missed or added. The reference is saved next to each
               recording (.ref.json), so the big model only runs once per
               phrase.
  compare      one row per combination of settings you have used (model,
               preview model, sensitivity, compute type): speed and
               accuracy side by side, to choose the best settings.
  sensitivity  how the Speech sensitivity slider and the input loudness
               affect detection and accuracy, including a "what if" table.
  trend        one row per day, real use and simulations side by side.

WORDS USED IN THE REPORTS
-------------------------
  n       how many values were measured.
  p50     the MEDIAN: half the values are lower, half higher. A "typical"
          value that one freak result can't distort.
  p95     the 95th PERCENTILE: 95 % of values are lower. Shows the slow cases.
  max     the single worst value.
  Example: times 100, 110, 120, 130, 900 ms -> p50 = 120, max = 900.

WHAT IS IN THIS FILE
--------------------
  iter_log_files, read_jsonl  find and read the log files.
  build_summary                the summary file.
  _pct, _row                   percentiles, and formatting one report row.
  build_report                 the "report" command.
  _words, _align               splitting text into words; comparing two
                               word lists (counting errors).
  _reference_text, _classify   the reference transcript; error categories.
  build_accuracy               the "accuracy" command.
  build_compare                the "compare" command.
  build_sensitivity            the "sensitivity" command.
  build_trend                  the "trend" command.
  (bottom)                     which command to run, from what was typed.

Python ideas new to this file are explained where they appear ("PYTHON:").
The basics are explained at the top of music.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import json                                    # read/write JSON text
import gzip                                    # read compressed (.gz) files
from collections import Counter                # a dict that counts things (see build_summary)
from datetime import datetime, timezone        # dates and times
from pathlib import Path                       # file paths as objects

# ─────────────────────────────────────────────────────────────────────────────
# Settings: where the files are
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent                  # Path: the project folder
LOG_PATH = APP_DIR / "interpreter.log.jsonl"               # Path: the main log
ERROR_LOG_PATH = APP_DIR / "interpreter.error.log.jsonl"   # Path: warnings and errors only
SUMMARY_PATH = APP_DIR / "interpreter.log.summary.json"    # Path: written by build_summary


# ─────────────────────────────────────────────────────────────────────────────
# Reading the log files
# ─────────────────────────────────────────────────────────────────────────────

def iter_log_files(base_path: Path):
    """List every file that belongs to a log: the current one and the older copies.

    Why there are several files:
        A log can't grow forever. When interpreter.log.jsonl reaches 5 MB,
        the app renames it to interpreter.log.jsonl.1 and starts a new one
        ("log rotation"); older copies become .2, .3 … Normally the old
        copies are compressed (.1.gz); if another program had the file
        open at that moment, compressing fails and it stays plain (.1).

    Inputs:
        base_path  Path of the current log, e.g. .../interpreter.log.jsonl

    Returns:
        list of Paths, current file first, e.g.
          [.../interpreter.log.jsonl, .../interpreter.log.jsonl.1,
           .../interpreter.log.jsonl.2.gz]
    """
    files = []
    if base_path.exists():
        files.append(base_path)
    # Older copies: gzipped (.1.gz …) or, if compressing failed because
    # another process held the file, plain (.1 …).
    # PYTHON: folder.glob("pattern") finds files matching a pattern; "*" stands
    # for any characters. sorted(...) puts them in name order.
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*")))
    return files


def read_jsonl(path: Path):
    """Read a log file line by line, giving back one dict per log entry.

    Inputs:
        path  Path of a .jsonl file, or a compressed .jsonl.gz file.

    Gives back ("yields"):
        one dict per line, e.g. {"ts": "…", "event": "phrase_trace", …}.
        Empty or damaged lines (e.g. half-written when the app crashed)
        are skipped.

    PYTHON: "yield" makes this a GENERATOR. Instead of reading the whole
    file into a list first (log files can be large), it hands over one
    entry at a time as the caller asks for them, in a for loop:
        for entry in read_jsonl(path):
            print(entry["event"])
    """
    # opener: the right function to open the file - gzip.open for compressed
    # files, the normal open otherwise. PYTHON: functions can be stored in
    # variables like any other value.
    opener = gzip.open if path.suffix == ".gz" else open
    # "rt" = read as text; errors="replace" puts a ? in place of any damaged
    # characters instead of stopping with an error.
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:                       # a file gives its lines one by one
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)        # text -> dict
            except Exception:
                continue                      # a damaged line: skip it


def build_summary():
    """Count the log's entries and save a short summary file.

    What it does:
        Goes through every entry of every log file, counts how many there
        are of each level (INFO, WARNING, ERROR…) and of each event name,
        and keeps the warnings and errors aside. Then writes the result to
        interpreter.log.summary.json (the app's "Export Log Summary"
        button does the same).

    Returns:
        the summary as a dict, e.g.
          {"generated_at": "2026-11-01T02:22:30+00:00",
           "total_entries": 12784,
           "levels": {"INFO": 12544, "WARNING": 31, "ERROR": 205},
           "top_events": {"phrase_flushed": 2107, "status_updated": 1690, …},
           "recent_issues": [{"ts": "…", "event": "source_silent", "message": "…"}, …]}
    """
    # PYTHON: a Counter is a dict for counting. A name that hasn't been seen
    # yet counts as 0, so levels["ERROR"] += 1 works straight away.
    #   c = Counter(); c["a"] += 1; c["a"] += 1; c["b"] += 1
    #   c -> Counter({"a": 2, "b": 1});  c.most_common(1) -> [("a", 2)]
    levels = Counter()                  # level name -> count
    events = Counter()                  # event name -> count
    issues = []                         # list of dicts: warnings and errors
    total = 0                           # int: all entries
    # Go through every log entry (current file and the older ones),
    # counting levels and event names, and keeping warnings/errors aside.
    for path in iter_log_files(LOG_PATH):
        for entry in read_jsonl(path):
            total += 1
            # entry.get("level", "UNKNOWN"): the value, or "UNKNOWN" if missing.
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
        # The time now, in this computer's time zone, as ISO text.
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "log_path": str(LOG_PATH),
        "error_log_path": str(ERROR_LOG_PATH),
        "total_entries": total,
        "levels": dict(levels),
        "top_events": dict(events.most_common(25)),    # the 25 most frequent events
        "recent_issues": issues[-25:],                  # [-25:] = the last 25
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers for the reports
# ─────────────────────────────────────────────────────────────────────────────

def _pct(values, fraction):
    """A percentile: the value below which `fraction` of the values fall.

    Inputs:
        values    list of numbers; None entries (not measured) are ignored.
        fraction  float: 0.5 for the median, 0.95 for the 95th percentile.

    Returns:
        the value, or None if there were no numbers.

    How: sort the values, then pick the one at position fraction x count.
    Example:
        _pct([300, 100, None, 200, 900], 0.5)
        sorted: [100, 200, 300, 900] -> position int(4 x 0.5) = 2 -> 300
    """
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    # min(len(v) - 1, …) makes sure the position never runs past the end.
    return v[min(len(v) - 1, int(len(v) * fraction))]


def _row(label, values, unit="ms"):
    """Format one line of a report: count, median, 95th percentile and maximum.

    Inputs:
        label   str: what was measured, e.g. "GPU encode+decode".
        values  list of numbers (None entries are ignored).
        unit    str (default "ms"): shown after the numbers.

    Returns:
        str, e.g.
        "  GPU encode+decode                  n=135   p50=   183.0  p95=   311.0  max=   402.0 ms"
        or "  GPU encode+decode                  (no data)"

    PYTHON: in f-strings, {x:<34} pads x to 34 characters (left-aligned);
    {x:>8.1f} shows a number with 1 decimal, right-aligned in 8 characters.
    That's what makes the columns line up.
    """
    vals = [x for x in values if x is not None]
    if not vals:
        return f"  {label:<34} (no data)"
    return (f"  {label:<34} n={len(vals):<5} p50={_pct(vals, .5):>8.1f}  "
            f"p95={_pct(vals, .95):>8.1f}  max={max(vals):>8.1f} {unit}")


# ─────────────────────────────────────────────────────────────────────────────
# "report": where does the time go?
# ─────────────────────────────────────────────────────────────────────────────

def build_report(since: str | None = None) -> str:
    """The speed report: how long each step takes, and what looks wrong.

    Inputs:
        since  str or None: only use log lines from this time on, e.g.
               "2026-10-31T14:00". None (default) = everything.
               (PYTHON: "str | None" in the type hint means "a str or None".)

    Returns:
        the whole report as one text (str) with line breaks, ready to print.

    Sections, in order:
      1. Finished phrases: how many, why they ended, which models and
         translators, retries, and one row per step a phrase goes through:
           audio length                    how long the phrase was
           speech start -> phrase end      recording it (incl. the pause)
           phrase end -> worker picks up   waiting in the queue
             VAD trim, mel features        preparing the audio (CPU)
             GPU lock wait, encode+decode  Whisper on the graphics card
           worker total                    all the worker's processing
           result -> on screen             the window drawing it
           PAUSE -> FINAL TEXT ON SCREEN   what you experience
      2. Live previews (the grey text).
      3. Audio capture health.
      4. Computer resources (CPU, memory, GPU) and window responsiveness.
      5. Sensitivity changes vs the measured input level.
      6. Languages detected.
      7. Music and singing.
      8. Times the audio source went silent.
      9. Findings: plain-language warnings from rules (e.g. "Peaks reach
         0 dBFS: input is clipping, lower the source volume.").
    """
    # ── Step 1: read every log entry, optionally only those from `since` on.
    #    (Timestamps are text like '2026-10-31T14:05:…', so comparing them
    #    as text also compares them in time order.)
    # PYTHON: a list comprehension with TWO "for"s goes through every entry of
    # every file: for p in files: for e in read_jsonl(p): keep e.
    entries = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)]
    if since:
        entries = [e for e in entries if e.get("ts", "") >= since]
    # ── Step 2: group the entries by event name, e.g. by_event["phrase_trace"]
    #    is the list of all phrase_trace entries.
    #    traces    = one entry per finished phrase (timings, text, scores …)
    #    displayed = one entry per phrase shown on screen (screen delays)
    by_event = {}
    for e in entries:
        by_event.setdefault(e.get("event"), []).append(e)
    traces, displayed = by_event.get("phrase_trace", []), by_event.get("phrase_displayed", [])
    # column(rows, "x") = the list of each entry's "x" value (a "column" of the
    # table), e.g. column(traces, "mt_ms") -> [31.2, 28.9, None, 44.0, …]
    column = lambda rows, k: [r.get(k) for r in rows]
    # gpu_ms = encode + decode time: the two GPU stages added together, because
    # GPU work is asynchronous and its time can be counted in either one.
    # "(x or 0)" turns a missing value (None) into 0.
    gpu_ms = [(r.get("encode_ms") or 0) + (r.get("decode_ms") or 0) for r in traces if "decode_ms" in r]
    # out: list of the report's lines; joined into one text at the end.
    # The first line names the time range: entries[0] is the first entry,
    # entries[-1] the last; [:19] keeps "2026-10-31T14:05:12" (no milliseconds).
    out = [f"Pipeline report  ({entries[0]['ts'][:19]} .. {entries[-1]['ts'][:19]}, {len(entries)} log lines)"
           if entries else "No log entries."]

    # ── Step 3: the finished-phrase section.
    out.append("\nFinished phrases (speech -> final text)")
    # Counter(list) counts each distinct value, e.g. reasons {"end": 81, "max": 28}.
    out.append(f"  phrases: {len(traces)}   cut reasons: {dict(Counter(column(traces, 'reason')))}   "
               f"models: {dict(Counter(column(traces, 'model')))}   "
               f"translation: {dict(Counter(column(traces, 'translation_engine')))}")
    out.append(_row("local translation (Opus-MT)", [r.get("mt_ms") for r in traces if r.get("mt_ms")]))
    # PYTHON: sum(bool(x) for …) counts how many are "truthy" (True counts as 1).
    out.append(f"  suspicious results: {sum(bool(r.get('suspicious')) for r in traces)}   "
               f"retried: {sum(bool(r.get('retry')) for r in traces)}   "
               f"retry used: {sum(bool((r.get('retry') or {}).get('used')) for r in traces)}   "
               f"still suspicious: {sum(bool(r.get('still_suspicious')) for r in traces)}   "
               f"context resets: {sum(bool(r.get('context_reset')) for r in traces)}   "
               f"no-speech drops: {sum(bool(r.get('dropped_text')) for r in traces)}")
    out.append(_row("retry cost", [(r.get("retry") or {}).get("ms") for r in traces if r.get("retry")]))
    # One report row per stage of the pipeline, in the order a phrase goes
    # through them: capture, queue, VAD, features, GPU, screen …
    # PYTHON: "for a, b, c in [(…), (…)]" unpacks each 3-item tuple of the list
    # into three variables.
    for label, vals, unit in [
            # audio_sec is in seconds; x 1000 for ms (None stays None).
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
    #    each holds its own small summary per measurement, like
    #    "process_ms": {"n": 28, "p50": 149.4, "p95": 191.0, "max": 219.8},
    #    so the report summarises those per-window medians.
    preview_stats = by_event.get("preview_stats", [])
    out.append("\nLive previews (10 s windows; p50 of each window's p50)")
    for k in ("age_ms", "process_ms", "encode_ms", "decode_ms", "vad_ms"):
        # isinstance(x, dict): is x a dict? (Skips windows without that measurement.)
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
        # "A if condition else B" picks the unit.
        out.append(_row(k + " (max)", [w[k]["max"] for w in capture_stats if isinstance(w.get(k), dict)],
                        "frames" if k == "backlog_frames" else "ms"))
    out.append(_row("level p50", [w["level_dbfs"]["p50"] for w in capture_stats
                                  if isinstance(w.get("level_dbfs"), dict)], "dBFS"))
    out.append(_row("speech frames", column(capture_stats, "speech_pct"), "%"))

    # ── Step 6: CPU, memory, GPU and how promptly the window updates.
    resource_stats = by_event.get("resource_stats", [])
    out.append("\nResources and UI (10 s windows)")
    for k, unit in (("process_cpu_pct", "%"), ("process_cpu_machine_pct", "%"),
                    ("system_cpu_pct", "%"), ("cpu_freq_mhz", "MHz"), ("rss_mb", "MB"),
                    ("mem_available_mb", "MB"), ("gpu_util_pct", "%"), ("gpu_mem_used_mb", "MB"),
                    ("gpu_mem_free_mb", "MB"), ("gpu_sm_clock_mhz", "MHz"), ("gpu_temp_c", "C"),
                    ("threads", ""), ("probe_ms", "ms"),
                    ("process_inventory_ms", "ms"), ("gpu_snapshot_ms", "ms")):
        out.append(_row(k, column(resource_stats, k), unit))
    # Average CPU per thread of ours (% of one core) and the busiest other apps.
    # Each window holds a dict like {"audio-capture": 12.0, "transcription-worker": 31.5}.
    for field, title in (("thread_cpu_pct", "CPU by app thread (average % of one core)"),
                         ("top_other_apps", "Busiest other apps (average % of one core)")):
        totals = Counter()
        windows = [w[field] for w in resource_stats if isinstance(w.get(field), dict)]
        for w in windows:
            # Counter.update(dict) ADDS the dict's numbers to the totals.
            totals.update(w)
        if windows:
            # Divide each total by the number of windows for the average;
            # {v / n:.0f} shows it with no decimals.
            out.append(f"  {title}: " + ", ".join(
                f"{k} {v / len(windows):.0f}" for k, v in totals.most_common(6)))
    # The last 5 resource warnings. e['ts'][11:19] = just "HH:MM:SS" from the timestamp.
    for e in by_event.get("resource_warning", [])[-5:]:
        out.append(f"  WARNING {e['ts'][11:19]}: {'; '.join(e.get('problems', []))}")
    out.append(_row("UI poll lag (p95 per window)",
                    [w["poll_lag_ms"]["p95"] for w in resource_stats if isinstance(w.get("poll_lag_ms"), dict)]))

    # ── Step 7: for each sensitivity change, compare the measured input level
    #    just before and just after it. It should NOT change: sensitivity only
    #    decides what counts as speech, not how loud the audio is.
    sensitivity_changes = by_event.get("sensitivity_changed", [])
    if sensitivity_changes:
        out.append("\nSensitivity changes vs measured input level (capture level p50, dBFS)")
        for e in sensitivity_changes:
            before = [w for w in capture_stats if w["ts"] < e["ts"] and isinstance(w.get("level_dbfs"), dict)]
            after = [w for w in capture_stats if w["ts"] > e["ts"] and isinstance(w.get("level_dbfs"), dict)]
            # The last window before, and the first window after the change.
            level_before = before[-1]["level_dbfs"]["p50"] if before else None
            level_after = after[0]["level_dbfs"]["p50"] if after else None
            out.append(f"  {e['ts'][11:19]}  sensitivity={e.get('value')}  "
                       f"threshold={e.get('vad_threshold')}  level before={level_before}  after={level_after}")

    # ── Step 8: automatic findings. Each rule checks for a known problem
    #    and adds a plain-language explanation if it is found.
    flags = []                               # list of str: the findings
    # "(x or 0) > 100": a missing value (None) counts as 0, so the rule doesn't fire.
    if (_pct(column(traces, "queue_wait_ms"), .95) or 0) > 100:
        flags.append("Phrases wait >100 ms for the worker: GPU busy with previews or backlog.")
    # any(...): True if at least one window matches.
    if any((w.get("read_gap_ms") or {}).get("max", 0) > 150 for w in capture_stats):
        flags.append("Audio reads stalled >150 ms: capture loop fell behind the device.")
    if (_pct([w["poll_lag_ms"]["p95"] for w in resource_stats if isinstance(w.get("poll_lag_ms"), dict)],
             .5) or 0) > 50:
        flags.append("UI thread is busy: results wait >50 ms before being drawn.")
    if preview_stats and sum(w.get("skipped_stale", 0) for w in preview_stats) > previews_processed:
        flags.append("More previews skipped than processed: preview model can't keep up.")
    if (_pct(column(traces, "agc_gain_db"), .5) or 0) > 20:
        flags.append("AGC is boosting >20 dB: input is very quiet, raise the source volume.")
    # max(1, n) avoids dividing by zero when there are no phrases.
    if sum(bool(r.get("still_suspicious")) for r in traces) > 0.1 * max(1, len(traces)):
        flags.append("Over 10% of phrases still look wrong after a retry: check the audio "
                     "(music, overlapping speech) or try a larger model.")
    if (_pct(column(traces, "peak_dbfs"), .95) or -100) > -0.5:
        flags.append("Peaks reach 0 dBFS: input is clipping, lower the source volume.")
    # Languages Whisper detected (automatic language detection)
    with_lang = [t for t in traces if t.get("lang")]
    if with_lang:
        from collections import Counter as _Counter
        detected = _Counter(t["lang"] for t in with_lang)
        switched = _Counter(t["lang_switched"] for t in traces if t.get("lang_switched"))
        out.append(chr(10) + "Languages (automatic detection)")      # chr(10) is a line break
        out.append("  detected (top guess per phrase): " +
                   ", ".join(f"{k} {v}" for k, v in detected.most_common(8)))
        out.append("  phrases switched to another language: " +
                   (", ".join(f"{k} {v}" for k, v in switched.most_common()) or "none"))
        out.append(_row("expected-language probability", column(traces, "src_lang_prob")))
        # Phrases that looked like another language (>= 50 % sure) but stayed.
        unsure = [t for t in with_lang if not t.get("lang_switched")
                  and t["lang"] != t.get("src_lang") and (t.get("lang_prob") or 0) >= 0.5]
        if unsure:
            out.append(f"  {len(unsure)} phrase(s) looked like another language but were not "
                       "switched (too short or not clear enough), e.g. "
                       + "; ".join(f"{t['lang']} {t.get('lang_prob')}: {(t.get('text') or '')[:30]!r}"
                                   for t in unsure[:3]))
    # Music mode (music.py): how often music was detected and what it cost.
    music_traces = [t for t in traces if t.get("music_mode")]
    if any("sound_music" in t for t in traces):
        out.append("\nMusic and singing (sound classifier, vocal separation)")
        out.append(f"  phrases in music mode: {len(music_traces)} of {len(traces)}   "
                   f"with singing: {sum(1 for t in music_traces if t.get('singing'))}   "
                   f"instrumental skipped: {len(by_event.get('transcription_skipped_instrumental', []))}")
        out.append(_row("music score (all phrases)", column(traces, "sound_music")))
        out.append(_row("speech score (all phrases)", column(traces, "sound_speech")))
        out.append(_row("classifier time", column(traces, "sound_classify_ms"), "ms"))
        out.append(_row("vocal separation time", column(music_traces, "separation_ms"), "ms"))
        out.append(_row("music removed (level)", column(music_traces, "music_removed_db"), "dBFS"))
        out.append(_row("audio length in music mode",
                        [x * 1000 if x else None for x in column(music_traces, "audio_sec")], "ms"))
        # The last 6 music-mode switches, e.g. "01:45:12  music mode ON (music; last seconds [0.1, 0.6, 0.7])"
        for e in by_event.get("music_mode_changed", [])[-6:]:
            out.append(f"  {e['ts'][11:19]}  music mode {'ON' if e.get('active') else 'off'} "
                       f"({e.get('reason') or 'phrase median ' + str(e.get('music_median'))}; "
                       f"last seconds {e.get('last_seconds', '-')})")
        scene = by_event.get("music_scene_stats", [])
        if scene:
            out.append(_row("music score per second (10 s mean)", column(scene, "music_mean")))
            out.append(f"  seconds judged: {sum(w.get('windows', 0) for w in scene)}   silent: "
                       f"{sum(w.get('silent_windows', 0) for w in scene)}   mode switches: "
                       f"{sum(w.get('switches', 0) for w in scene)}")
        # PYTHON: "+" on two lists joins them into one list.
        failures = by_event.get("music_processing_failed", []) + by_event.get("music_scene_failed", [])
        if failures:
            flags.append(f"Music processing failed {len(failures)} time(s): see the error log.")
    # Times the captured device received no sound at all (source paused,
    # or playing to another device), with the apps that were playing.
    silences = by_event.get("source_silent", [])
    resumed = by_event.get("source_resumed", [])
    if silences:
        out.append("\nSource silent (no sound reached the device for 5 s or more)")
        for e in silences[-8:]:
            # PYTHON: next((x for x in list if condition), None) finds the FIRST
            # matching item, or None if there is none. Here: when sound came back.
            back = next((r for r in resumed if r["ts"] > e["ts"]), None)
            until = f"{back['ts'][11:19]} ({back.get('silent_sec')} s)" if back else "end of log"
            apps = ", ".join(a["app"] for a in e.get("playing_apps", []) if "app" in a)
            out.append(f"  {e['ts'][11:19]}  silent until {until}; "
                       f"apps with sound sessions: {apps or 'none'}")
        flags.append(f"The audio source went silent {len(silences)} time(s): see 'Source silent' above.")
    out.append("\nFindings")
    # One line per finding, or "nothing abnormal" if there are none.
    # PYTHON: "A if condition else B" used as a whole statement: runs A or B.
    out.extend(f"  - {f}" for f in flags) if flags else out.append("  - nothing abnormal")
    # "\n".join(list) glues the lines together with line breaks between them.
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Comparing texts word by word
# ─────────────────────────────────────────────────────────────────────────────

def _words(text):
    """Split text into simple lower-case words, ready to compare.

    Punctuation is ignored, capital letters don't matter, and the Russian
    letter "ё" counts as "е" (people and models use them interchangeably).

    Inputs:
        text  str or None.

    Returns:
        list of str.

    Examples:
        _words("Hello, World!")      ->  ["hello", "world"]
        _words("Всё хорошо.")        ->  ["все", "хорошо"]
        _words(None)                 ->  []
    """
    import re
    text = (text or "").lower().replace("ё", "е")
    # re.findall(r"\w+", text): every run of "word characters" (letters,
    # digits, underscore) - i.e. the words, without punctuation and spaces.
    return re.findall(r"\w+", text)


def _align(ref_words, app_words):
    """Compare the correct words with the app's words and count the errors.

    Inputs:
        ref_words  list of str: the correct ("reference") words.
        app_words  list of str: what the app wrote.

    Returns (three values):
        errors   int: the fewest word changes (a word missing, an extra
                 word, or a word replaced by another) that turn the
                 correct text into the app's text.
        missing  list of str: correct words the app left out.
        wrong    list of str: replaced words, as "correct→app".

    Examples:
        _align(["мы", "поедем", "домой"], ["мы", "едем", "домой"])
          -> (1, [], ["поедем→едем"])
        _align(["guten", "morgen", "alle"], ["guten", "morgen"])
          -> (1, ["alle"], [])
        word error rate = errors / number of correct words: 1 / 3 = 33 %

    How it works ("edit distance", also called Levenshtein distance):
        Build a table d where d[i][j] = the fewest changes needed to turn
        the first i correct words into the first j app words. Each cell
        is worked out from three neighbours that are already known:
          from above      d[i-1][j]   + 1   a correct word is missing
          from the left   d[i][j-1]   + 1   the app added a word
          diagonally      d[i-1][j-1] + 0   the words match
                                      + 1   one word replaced another
        and takes the cheapest. The answer for the whole texts ends up in
        the bottom-right cell. Then the path back from that cell shows
        WHICH words were missing or replaced.

        Table for "мы поедем домой" (rows) vs "мы едем домой" (columns):
                      ""   мы   едем  домой
              ""       0    1    2     3
              мы       1    0    1     2
              поедем   2    1    1     2
              домой    3    2    2     1      <- 1 error
    """
    # edits: the table, (correct words + 1) rows x (app words + 1) columns,
    # all 0 to begin with. PYTHON: [0] * n makes a list of n zeros; the outer
    # comprehension makes one such row per correct word (+1).
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
    # PYTHON: (a != b) is True or False, which count as 1 or 0 in arithmetic.
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(app_words) + 1):
            edits[i][j] = min(edits[i - 1][j] + 1, edits[i][j - 1] + 1,
                          edits[i - 1][j - 1] + (ref_words[i - 1] != app_words[j - 1]))
    # Walk back from the bottom-right corner along the cheapest path to see
    # WHICH words were missing or replaced, not just how many errors.
    # The lists come out backwards, so they are reversed ([::-1]) at the end.
    i, j, missing, wrong = len(ref_words), len(app_words), [], []
    while i > 0 or j > 0:
        # Came diagonally? (a match, or a replaced word)
        if i and j and edits[i][j] == edits[i - 1][j - 1] + (ref_words[i - 1] != app_words[j - 1]):
            if ref_words[i - 1] != app_words[j - 1]:
                wrong.append(f"{ref_words[i - 1]}→{app_words[j - 1]}")
            i, j = i - 1, j - 1
        # Came from above? (a missing word)
        elif i and edits[i][j] == edits[i - 1][j] + 1:
            missing.append(ref_words[i - 1])
            i -= 1
        # Otherwise from the left (an extra word in the app's text).
        else:
            j -= 1
    return edits[len(ref_words)][len(app_words)], missing[::-1], wrong[::-1]


# Text the reference model itself invents on music or silence (YouTube
# subtitle credits, "to be continued", "thanks for watching" …). Phrases whose
# reference contains any of these are skipped: the reference is wrong there.
REF_HALLUCINATIONS = ("субтитр", "dimatorzok", "продолжение следует", "редактор",
                      "спасибо за просмотр", "подписывайтесь", "thanks for watching")


def _reference_text(model_holder, ref_model, wav_path, lang):
    """The reference transcript of one phrase recording ("what was really said").

    Inputs:
        model_holder  dict: holds the big model once it is loaded, as
                      model_holder["model"], so it is loaded only once for
                      all phrases. (PYTHON: a dict passed in can be changed
                      here and the caller sees the change.)
        ref_model     str: which Whisper model to use, e.g. "large-v3".
        wav_path      Path of the phrase's recording.
        lang          str or None: the language to transcribe, e.g. "ru".

    Returns:
        str: the reference text, e.g. "Привет, как дела?"

    Caching: the result is saved next to the recording as <name>.ref.json,
    e.g. {"model": "large-v3", "text": "Привет, как дела?"}. Next time it
    is read from there - the big model is slow (about 1-2 s per phrase).

    The big model runs with careful settings: beam search (tries 5
    candidate sentences and keeps the best), no filtering, and no context
    from previous phrases, so each reference stands on its own.
    """
    # .with_suffix(".ref.json"): the same path with a different ending.
    cache = wav_path.with_suffix(".ref.json")
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("model") == ref_model:               # made by the same model?
            return cached["text"]
    if model_holder.get("model") is None:                  # load the big model the first time
        import ctranslate2
        from faster_whisper import WhisperModel
        # The graphics card if there is one, otherwise the processor.
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        # float16 / int8: number formats; smaller formats are faster.
        model_holder["model"] = WhisperModel(
            ref_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    # Read the WAV ourselves: faster-whisper's file decoder is incompatible
    # with the installed PyAV version.
    import wave
    import numpy as np
    with wave.open(str(wav_path)) as w:
        # whole numbers -> decimals from -1.0 to 1.0
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    # transcribe returns the text in "segments" (pieces) and some info (unused: "_").
    segs, _ = model_holder["model"].transcribe(audio, language=lang, beam_size=5,
                                               vad_filter=False,
                                               condition_on_previous_text=False)
    text = " ".join(s.text for s in segs).strip()
    cache.write_text(json.dumps({"model": ref_model, "text": text}, ensure_ascii=False),
                     encoding="utf-8")
    return text


# Small words that carry little meaning ("well", "so", "and"; Russian "ну",
# "вот", "и" …). Errors that only involve these matter less (see _classify).
FILLER_WORDS = {"ну", "вот", "и", "а", "то", "так", "это", "же", "уж", "да", "um", "uh",
                "well", "so", "and"}


def _classify(ref_words, app_words, errors, missing, wrong):
    """Put a phrase's errors into a rough category, to show WHERE accuracy is lost.

    Inputs:
        the correct words, the app's words, and _align's three results.

    Returns (str, or None if there were no errors):
        "whole phrase lost/wrong"      the app wrote nothing, or 70 %+ wrong
        "filler words only"            only small words like "ну", "and"
        "first/last word (phrase cut)" only the first or last word: usually
                                       the phrase was cut at a bad moment
        "word errors"                  ordinary misheard words

    Example:
        correct "ну вот мы дома", app "мы дома": missing ["ну", "вот"]
        -> "filler words only"
    """
    if not errors:
        return None
    if not app_words or errors >= 0.7 * len(ref_words):
        return "whole phrase lost/wrong"
    # The correct words involved: the missing ones, plus the correct half of
    # each "correct→app" replacement (.split("→")[0]).
    words = missing + [w.split("→")[0] for w in wrong]
    if words and all(w in FILLER_WORDS for w in words):
        return "filler words only"
    if words and all(w in (ref_words[0], ref_words[-1]) for w in words):
        return "first/last word (phrase cut)"
    return "word errors"


# ─────────────────────────────────────────────────────────────────────────────
# "accuracy": which words does the app get wrong?
# ─────────────────────────────────────────────────────────────────────────────

def build_accuracy(since=None, ref_model="large-v3"):
    """The accuracy report: the app's text vs a careful reference transcript.

    Inputs:
        since      str or None: only phrases from this time on.
        ref_model  str (default "large-v3"): the reference Whisper model.

    Returns:
        the report as one text. At the top, the summary:
          word error rate vs reference: 7.5% (112 errors / 1490 words) …
          by model / translation engine:
            large-v3-turbo / opus-mt   WER   7.5%  (112 errors / 1490 words, 109 phrases)
          errors by category: word errors 60, first/last word (phrase cut) 30, …
          most often missed: и×5, не×3, …
        then, for every phrase with errors:
          01:45:49 seq=12 model=large-v3-turbo reason=end errors=1/4 …
             category: word errors
             app: Привет как дила
             ref: Привет, как дела?
             wrong:   дела→дила

    Steps:
      1. Find every finished phrase (phrase_trace) whose recording exists.
      2. Set up the counters.
      3. For each phrase: get the reference, compare the words, count the
         errors by model and category, and write the details.
      4. Put the summary on top of the details.

    Note: the reference is itself an AI model, not a person, so small
    differences like "10" vs "десять" also count as errors.
    """
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
    #    model_holder  keeps the reference model once loaded (loaded only if
    #                  some recording has no cached reference yet)
    #    details       list of str: the per-phrase lines
    #    per_model     (model / translation engine) -> (errors, words, phrases)
    #    all_missing   Counter: how often each word was missed
    #    categories    Counter: errors per kind of mistake (see _classify)
    #    retries       Counter: how retries of suspicious results ended
    model_holder = {}
    details, per_model, all_missing, categories = [], {}, Counter(), Counter()
    retries = Counter()
    # PYTHON: "a = b = c = 0" sets all of them to 0.
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
        # Line the two word lists up (like a spell-checker's "diff") to count
        # the errors and list which words were missing or wrong.
        errors, missing, wrong = _align(ref_words, app_words)
        category = _classify(ref_words, app_words, errors, missing, wrong)
        if category:
            categories[category] += errors
        if trace.get("retry"):
            retries["used" if trace["retry"].get("used") else "kept original"] += 1
        total_errors += errors
        total_ref_words += len(ref_words)
        missed_phrases += not app_words          # True (1) if the app wrote nothing
        all_missing.update(missing)              # count each missing word
        key = f"{trace.get('model')} / {trace.get('translation_engine') or '-'}"
        # Add to this model's running totals: (errors, words, phrases).
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
    #    WER = errors / reference words. {x:.1%} formats 0.123 as "12.3%".
    out = [f"Accuracy check: {len(traces)} phrases, app vs reference model '{ref_model}'",
           f"  word error rate vs reference: {total_errors / max(1, total_ref_words):.1%} "
           f"({total_errors} errors / {total_ref_words} words)   phrases with no text at all: "
           f"{missed_phrases}   skipped (reference hallucinated/empty): {skipped}",
           "  by model / translation engine:"]
    # PYTHON: "out += [...]" adds all items of the list to out.
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


# ─────────────────────────────────────────────────────────────────────────────
# "compare": which settings work best?
# ─────────────────────────────────────────────────────────────────────────────

def build_compare(since=None, fill_refs=False, ref_model="large-v3"):
    """Compare speed and accuracy between the settings you have tried.

    Every finished phrase is logged with the settings in force (fields
    starting "cfg_", e.g. cfg_model, cfg_preview_model, cfg_sensitivity).
    Phrases are grouped by (model, preview model, sensitivity, compute type)
    and each group gets one row:
      phrases       how many phrases used these settings
      pause->text   end of speech -> final text on screen (median, ms)
      1st preview   speech start -> first grey preview (median, ms)
      1st text      speech start -> first text of any kind (so "Previews:
                    Off" can be compared with the preview settings)
      process       the worker's processing time (median, ms)
      early         % of phrases whose early transcription was used
      doubtful      % of phrases whose result looked suspicious
      preview ok    how much of the final text the last preview already
                    had right (%)
      WER           word error rate against the reference transcripts

    Inputs:
        since      str or None: only phrases from this time on.
        fill_refs  bool (default False): also CREATE missing reference
                   transcripts (slow: runs the big model). Otherwise only
                   phrases whose reference already exists get a WER.
        ref_model  str: the reference model's name.

    Returns:
        the table as one text, e.g.
          model          preview          sens compute     phrases pause->text …
          large-v3-turbo small             0.5 int8_float16    109      366 ms …
    """
    import sys
    sys.path.insert(0, str(APP_DIR))
    from interpreter import RECORD_DIR, _add_cuda_dll_dirs

    entries = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)
               if not since or e.get("ts", "") >= since]
    traces = [e for e in entries if e.get("event") == "phrase_trace"]
    # phrase_displayed lines hold the screen timings; match them to their
    # phrase by (app session, listening run, phrase number).
    # PYTHON: a DICT COMPREHENSION {key: value for …} builds a dict in one line.
    displayed = {(e.get("session_id"), e.get("run_id"), e.get("seq")): e
                 for e in entries if e.get("event") == "phrase_displayed"}
    if fill_refs:
        _add_cuda_dll_dirs()
    model_holder = {}

    # groups: settings tuple -> dict of lists and counters for that group.
    groups = {}
    for t in traces:
        sens = t.get("cfg_sensitivity")
        # The group's key. "?" = not recorded (phrases logged before settings
        # were written into each phrase). {sens:.1f} = 1 decimal: 0.5 -> "0.5".
        key = (t.get("cfg_model") or t.get("model") or "?",
               t.get("cfg_preview_model") or "?",
               f"{sens:.1f}" if isinstance(sens, (int, float)) else "?",
               t.get("cfg_compute_type") or "?")
        g = groups.setdefault(key, {"pause_to_text": [], "first_preview": [], "process": [],
                                    "early": [], "suspicious": [], "preview_agree": [], "first_text": [],
                                    "errors": 0, "ref_words": 0, "n": 0})
        g["n"] += 1
        d = displayed.get((t.get("session_id"), t.get("run_id"), t.get("seq")))
        if d:
            g["pause_to_text"].append(d.get("speech_end_to_text_ms"))
            g["first_preview"].append(d.get("first_preview_ms"))
            # First text of any kind: the first preview, or with previews Off
            # the finished phrase (speech start -> pause -> text on screen).
            # This makes "Off" comparable with the preview settings.
            first = d.get("first_preview_ms")
            if first is None and t.get("capture_ms") is not None and d.get("speech_end_to_text_ms") is not None:
                first = t["capture_ms"] + d["speech_end_to_text_ms"]
            g["first_text"].append(first)
        g["process"].append(t.get("process_ms"))
        g["early"].append(1 if t.get("early_used") else 0)
        g["suspicious"].append(1 if t.get("suspicious") else 0)
        # How much of the final text the last preview already had right:
        # 1 - (errors / final words), as a percentage. Example: 10 final words,
        # 2 different in the preview -> 80 %.
        final_w, prev_w = _words(t.get("text")), _words(t.get("last_preview"))
        if final_w and prev_w:
            errs, _, _ = _align(final_w, prev_w)
            g["preview_agree"].append(max(0.0, 1 - errs / len(final_w)) * 100)
        # Word error rate against the reference, where one is available.
        wav = RECORD_DIR / t["audio_file"] if t.get("audio_file") else None
        if wav and wav.exists() and (fill_refs or wav.with_suffix(".ref.json").exists()):
            ref_text = _reference_text(model_holder, ref_model, wav, t.get("src_lang"))
            ref_words = _words(ref_text)
            if ref_words and not any(h in ref_text.lower() for h in REF_HALLUCINATIONS):
                errs, _, _ = _align(ref_words, _words(t.get("text")))
                g["errors"] += errs
                g["ref_words"] += len(ref_words)

    # Two tiny helpers: med = the median; fmt = a number as text with a unit,
    # or "-" if it is missing. PYTHON: unit="" gives the lambda a default value.
    med = lambda v: _pct(v, 0.5)
    fmt = lambda v, unit="": "-" if v is None else f"{v:.0f}{unit}"
    out = [f"Settings comparison ({len(traces)} phrases)",
           "  pause->text = end of speech to final text on screen (median); "
           "1st preview = speech start to first live text;",
           "  1st text = speech start to the first text of any kind (with previews off, the final text);",
           "  early = phrases whose early transcription was used; preview ok = share of "
           "the final words the last preview already had right.\n",
           f"  {'model':<15}{'preview':<15}{'sens':>5} {'compute':<13}{'phrases':>7}"
           f"{'pause->text':>12}{'1st preview':>12}{'1st text':>10}{'process':>9}{'early':>7}"
           f"{'doubtful':>9}{'preview ok':>11}{'WER':>8}"]
    # Most-used settings first. PYTHON: sorted(..., key=f) sorts by f(item);
    # the minus sign sorts from largest to smallest.
    for key in sorted(groups, key=lambda k: -groups[k]["n"]):
        g = groups[key]
        n = g["n"]
        wer = f"{g['errors'] / g['ref_words']:.1%}" if g["ref_words"] else "-"
        out.append(f"  {key[0]:<15}{key[1]:<15}{key[2]:>5} {key[3]:<13}{n:>7}"
                   f"{fmt(med(g['pause_to_text']), ' ms'):>12}"
                   f"{fmt(med(g['first_preview']), ' ms'):>12}"
                   f"{fmt(med(g['first_text']), ' ms'):>10}"
                   f"{fmt(med(g['process']), ' ms'):>9}"
                   f"{100 * sum(g['early']) / n:>6.0f}%{100 * sum(g['suspicious']) / n:>8.0f}%"
                   f"{fmt(med(g['preview_agree']), '%'):>11}{wer:>8}")
    out.append("\n  '?' = not recorded (phrases logged before settings were added to each "
               "phrase). WER '-' = no reference yet: run `compare --ref` to create them.")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# "sensitivity": how do the sensitivity slider and input loudness matter?
# ─────────────────────────────────────────────────────────────────────────────

def build_sensitivity(since=None):
    """How the Speech sensitivity setting and the input level affect results.

    What sensitivity does:
        It does NOT change the volume. It sets how sure the speech detector
        must be before audio counts as speech:
            threshold = 0.85 - 0.7 x sensitivity
            sensitivity 0 %   -> threshold 0.85 (only very clear speech)
            sensitivity 50 %  -> threshold 0.50
            sensitivity 100 % -> threshold 0.15 (almost anything)
        Too low: quiet or unclear speech is missed, and phrase starts are
        detected late (first words clipped).
        Too high: noise or music starts phrases that turn out empty.

    The report has three parts:
      1. results per sensitivity setting you have used;
      2. results per input level (how loud the speech was);
      3. a "what if" table: how much of the audio WOULD count as speech at
         each sensitivity, worked out from the logged speech-score
         histograms - so a good setting can be chosen without trying them
         all.

    Inputs:
        since  str or None: only log lines from this time on.

    Returns:
        the report as one text.
    """
    import sys
    sys.path.insert(0, str(APP_DIR))
    from interpreter import RECORD_DIR

    entries = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)
               if not since or e.get("ts", "") >= since]
    traces = [e for e in entries if e.get("event") == "phrase_trace"]

    def ref_words_for(t):
        """The reference words of a phrase, if its reference transcript has
        already been made (by `accuracy` or `compare --ref`); otherwise None.
        This report never runs the big model itself."""
        if not t.get("audio_file"):
            return None
        r = (RECORD_DIR / t["audio_file"]).with_suffix(".ref.json")
        if not r.exists():
            return None
        text = json.loads(r.read_text(encoding="utf-8"))["text"]
        if any(h in text.lower() for h in REF_HALLUCINATIONS):
            return None
        return _words(text) or None

    def summarise(rows):
        """The figures for one table row (a group of phrases).

        Returns a dict:
            n         number of phrases
            empty     % of phrases that had no words (started by noise)
            onset     median onset delay: how late the start of speech
                      was detected, in ms
            wer       word error rate as text, e.g. "7.5%", or "-"
            first     % of phrases whose first word was missed, or "-"
            level     median speech level (dBFS)
            nospeech  median "no speech" probability from Whisper
            doubtful  % of phrases with a suspicious result
            agc       median automatic gain added (dB)
        """
        errs = words = first_missed = with_ref = 0
        for t in rows:
            ref = ref_words_for(t)
            if ref:
                app = _words(t.get("text"))
                e, _, _ = _align(ref, app)
                errs += e
                words += len(ref)
                with_ref += 1
                # The first correct word isn't among the app's first 3 words.
                first_missed += ref[0] not in app[:3]
        n = len(rows)
        return {
            "n": n,
            "empty": 100 * sum(not t.get("text") for t in rows) / n,
            "onset": _pct([t.get("onset_delay_ms") for t in rows], 0.5),
            "wer": f"{errs / words:.1%}" if words else "-",
            "first": f"{100 * first_missed / with_ref:.0f}%" if with_ref else "-",
            "level": _pct([t.get("rms_dbfs") for t in rows], 0.5),
            # no_speech_prob is a list (one per task); take its largest value.
            "nospeech": _pct([max(t["no_speech_prob"]) if isinstance(t.get("no_speech_prob"), list)
                              and t["no_speech_prob"] else None for t in rows], 0.5),
            "doubtful": 100 * sum(bool(t.get("suspicious")) for t in rows) / n,
            "agc": _pct([t.get("agc_gain_db") for t in rows], 0.5),
        }

    # fmt: a number as text using a format pattern (default: no decimals), or "-".
    # PYTHON: "{:.0f}".format(3.7) -> "4" (the same as f"{3.7:.0f}").
    fmt = lambda v, f="{:.0f}": "-" if v is None else f.format(v)
    out = [f"Sensitivity and input level ({len(traces)} phrases)\n",
           "1) By Speech sensitivity  (threshold = how sure the detector must be)",
           "   empty = phrases started by noise that contained no words; onset = how late",
           "   the start of speech was detected (ms); first word = phrases whose first",
           "   word was missed (needs references: run `compare --ref` first).",
           f"   {'sensitivity':>11} {'threshold':>9} {'phrases':>8} {'empty':>6} {'onset':>7}"
           f" {'first word':>10} {'WER':>7}"]
    # Part 1: group phrases by the sensitivity in force. The log stores the
    # threshold, so turn it back into the slider value (the formula above,
    # solved for sensitivity), rounded to the nearest 10 %.
    by_sens = {}
    for t in traces:
        thr = t.get("vad_threshold")
        if isinstance(thr, (int, float)):
            sens = round((0.85 - thr) / 0.7 * 10) / 10   # back to the 0..1 slider value
            # max(0.0, min(1.0, x)) keeps the value between 0 and 1.
            by_sens.setdefault(max(0.0, min(1.0, sens)), []).append(t)
    for sens in sorted(by_sens):
        r = summarise(by_sens[sens])
        out.append(f"   {sens * 100:>10.0f}% {0.85 - 0.7 * sens:>9.2f} {r['n']:>8} {r['empty']:>5.0f}%"
                   f" {fmt(r['onset']):>7} {r['first']:>10} {r['wer']:>7}")

    # Part 2: group phrases by how loud the speech was, in bands of a few dB.
    out += ["", "2) By input level of the speech  (AGC = gain added before Whisper)",
            f"   {'level (dBFS)':>13} {'phrases':>8} {'WER':>7} {'no-speech':>10}"
            f" {'doubtful':>9} {'AGC dB':>7}"]
    bands = [(-99, -45), (-45, -38), (-38, -32), (-32, -26), (-26, -20), (-20, 0)]
    for lo, hi in bands:
        rows = [t for t in traces if isinstance(t.get("rms_dbfs"), (int, float))
                and lo <= t["rms_dbfs"] < hi]
        if rows:
            r = summarise(rows)
            out.append(f"   {f'{lo} to {hi}':>13} {r['n']:>8} {r['wer']:>7} "
                       f"{fmt(r['nospeech'], '{:.2f}'):>10} {r['doubtful']:>8.0f}% "
                       f"{fmt(r['agc'], '{:.1f}'):>7}")

    # Part 3: the "what if" table. Every 10 s the capture thread logs a
    # HISTOGRAM of the speech scores it saw: 10 counters, one per score band
    # (0.0-0.1, 0.1-0.2, … 0.9-1.0). Example: [5000, 800, 300, …, 2500].
    # Adding them all up shows how the scores are spread; for any threshold,
    # the share of scores above it is the share of audio that would count as
    # speech at that setting.
    hist = [0] * 10
    for e in entries:
        if e.get("event") == "capture_stats" and isinstance(e.get("speech_score_histogram"), list):
            # zip pairs up the two lists' counters; add them band by band.
            hist = [a + b for a, b in zip(hist, e["speech_score_histogram"])]
    total = sum(hist)
    out += ["", "3) What-if: share of audible audio that would count as speech at each setting"]
    if total:
        out.append("   (from the speech-score histograms; scores fall into 0.1-wide bins, so"
                   " each value is approximate)")
        for sens in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
            thr = 0.85 - 0.7 * sens
            # A band counts if its middle ((i + 0.5) / 10, e.g. 0.55 for band 5)
            # is at or above the threshold.
            above = sum(hist[i] for i in range(10) if (i + 0.5) / 10 >= thr)
            out.append(f"   sensitivity {sens * 100:>3.0f}%  threshold {thr:.2f}  ->"
                       f" {100 * above / total:5.1f}% of audible audio counted as speech")
    else:
        out.append("   (no histograms yet: they are logged from this version on)")

    out += ["", "How to read this: pick the lowest sensitivity whose 'first word' and WER",
            "are still good (fewer noise-triggered 'empty' phrases), or raise it if",
            "onset delays are long and first words go missing."]
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# "trend": is it getting better, day by day?
# ─────────────────────────────────────────────────────────────────────────────

def build_trend(since=None):
    """Headline figures per day, real use and simulations side by side.

    Why: a change can look good in the simulator but behave differently on
    real calls. This table shows both, day by day, so you can see whether
    a change helped where it matters.

    Columns (one row per day and source):
      sess        sessions (listening runs) that day
      phrases     finished phrases
      empty       % thrown away as non-speech or invented text
      retry       % transcribed a second time because they looked wrong
      cut         % forced to end at the time limit
      music       % in music mode
      lang chg    times the language changed
      end->text   median end of speech -> text on screen (real use only, ms)
      recog       median recognition time (ms)
      errors      ERROR/CRITICAL log lines
      reported    "Report a mistake" uses
      GPU shared  log lines where another program was using the graphics card
      version     the app's code version(s) that day (git commit)

    Example:
      day        source       sess phrases  empty  retry   cut …
      2026-10-31 real use       17    1437    19%     5%   23% …
      2026-11-01 simulation     11     747    23%    11%    6% …

    Inputs:
        since  str or None: only log lines from this time on.

    Returns:
        the table as one text.
    """
    import statistics                 # Python's built-in statistics (median etc.)
    # days: (date, source) -> dict of counters, e.g. ("2026-11-01", "real use") -> {...}
    days = {}

    def day(entry, sim):
        """The counters for this log entry's day and source, created empty
        the first time that day/source is seen.

        Inputs:
            entry  a log entry (dict); its "ts" gives the date (first 10
                   characters, "2026-11-01").
            sim    bool: True for a simulation, False for real use.
        """
        # PYTHON: a SET (set()) is used for sessions and versions: adding the
        # same value twice keeps it once, so len() counts distinct values.
        d = days.setdefault((entry["ts"][:10], "simulation" if sim else "real use"),
                            {"sessions": set(), "phrases": 0, "empty": 0, "retried": 0,
                             "forced_cut": 0, "lang_changes": 0, "music": 0, "pause_to_text": [],
                             "recognise": [], "errors": 0, "reported": 0, "gpu_shared": 0,
                             "versions": set()})
        return d

    for path in iter_log_files(LOG_PATH):
        for e in read_jsonl(path):
            if since and e.get("ts", "") < since:
                continue
            # ev: the event name; sid: the session id as text ("" if missing).
            ev, sid = e.get("event"), str(e.get("session_id") or "")
            sim = sid.startswith("sim")      # simulations use ids starting "sim-"
            if ev == "phrase_trace":
                d = day(e, sim)
                d["sessions"].add(sid)
                d["phrases"] += 1
                d["empty"] += not e.get("text")
                d["retried"] += "retry" in e          # True/False count as 1/0
                d["forced_cut"] += e.get("reason") == "max"
                d["lang_changes"] += bool(e.get("lang_changed_from"))
                d["music"] += bool(e.get("music_mode"))
                # Recognition time: if the early transcription was used, that
                # work happened during the pause (early_ms); otherwise process_ms.
                rec = e.get("early_ms") if e.get("early_used") else e.get("process_ms")
                if rec is not None:
                    d["recognise"].append(rec)
            elif ev == "phrase_displayed" and e.get("speech_end_to_text_ms") is not None:
                day(e, sim)["pause_to_text"].append(e["speech_end_to_text_ms"])
            elif ev == "user_reported_mistake":
                day(e, False)["reported"] += 1
            # PYTHON: a "\" at the end of a line continues the condition on the next.
            elif ev in ("session_context", "resource_stats") and \
                    (e.get("gpu_other_programs_pct") or {}):
                day(e, sim)["gpu_shared"] += 1
            if ev == "session_context" and e.get("app_version"):
                day(e, sim)["versions"].add(e["app_version"])
            if e.get("level") in ("ERROR", "CRITICAL"):
                day(e, sim)["errors"] += 1

    # med: the median as whole-number text, or "-"; pct: n out of d as a percentage.
    med = lambda v: f"{statistics.median(v):.0f}" if v else "-"
    pct = lambda n, d: f"{100 * n / d:.0f}%" if d else "-"
    lines = ["Daily trend (median times in ms; % of phrases)", "",
             f"{'day':<11}{'source':<12}{'sess':>5}{'phrases':>8}{'empty':>7}{'retry':>7}"
             f"{'cut':>6}{'music':>7}{'lang chg':>9}{'end->text':>10}{'recog':>7}{'errors':>7}"
             f"{'reported':>9}{'GPU shared':>11}  version"]
    # PYTHON: sorted(dict.items()) sorts by key, here (date, source): oldest day first.
    for (date, source), d in sorted(days.items()):
        if not d["phrases"] and not d["errors"]:
            continue                                  # nothing to show for this row
        n = d["phrases"]
        lines.append(f"{date:<11}{source:<12}{len(d['sessions']):>5}{n:>8}{pct(d['empty'], n):>7}"
                     f"{pct(d['retried'], n):>7}{pct(d['forced_cut'], n):>6}{pct(d['music'], n):>7}"
                     f"{d['lang_changes']:>9}{med(d['pause_to_text']):>10}{med(d['recognise']):>7}"
                     f"{d['errors']:>7}{d['reported']:>9}{d['gpu_shared']:>11}  "
                     f"{', '.join(sorted(d['versions'])) or '-'}")
    lines += ["", "empty: transcriptions discarded as non-speech or invented; retry: decoded a "
              "second time because the text looked wrong; cut: phrase forced to end at the time "
              "limit; end->text: end of speech to text on screen (real use only); recog: "
              "recognition time; reported: 'Report a mistake' uses; GPU shared: log lines where "
              "another program was computing on the graphics card."]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Which report to make (when run from the command line)
# ─────────────────────────────────────────────────────────────────────────────
# PYTHON: sys.argv is the list of words typed on the command line, e.g.
#   "python log_tools.py report 2026-10-31T14:00"
#   -> ["log_tools.py", "report", "2026-10-31T14:00"]
# so sys.argv[1] is the command and sys.argv[2] (if present) the start time.
# "if len(sys.argv) > 2 else None" uses None when no start time was typed.
if __name__ == "__main__":
    import sys
    # Reports include original transcripts, which may contain Cyrillic or
    # other characters that the Windows console's legacy code page cannot
    # encode. Keep the CLI useful from the default PowerShell/Command Prompt.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass  # stdout may be redirected or wrapped by an embedding application
    if len(sys.argv) > 1 and sys.argv[1] == "trend":
        print(build_trend(sys.argv[2] if len(sys.argv) > 2 else None))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        print(build_report(sys.argv[2] if len(sys.argv) > 2 else None))
    elif len(sys.argv) > 1 and sys.argv[1] == "sensitivity":
        print(build_sensitivity(sys.argv[2] if len(sys.argv) > 2 else None))
    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Everything after "compare" except the "--ref" switch; the first of
        # those (if any) is the start time.
        args = [a for a in sys.argv[2:] if a != "--ref"]
        print(build_compare(args[0] if args else None, fill_refs="--ref" in sys.argv))
    elif len(sys.argv) > 1 and sys.argv[1] == "accuracy":
        print(build_accuracy(sys.argv[2] if len(sys.argv) > 2 else None,
                             sys.argv[3] if len(sys.argv) > 3 else "large-v3"))
    else:
        # No command: write the summary file and print it.
        summary = build_summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2))

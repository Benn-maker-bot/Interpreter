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
    files = []
    if base_path.exists():
        files.append(base_path)
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*.gz")))
    return files


def read_jsonl(path: Path):
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
    levels = Counter()
    events = Counter()
    issues = []
    total = 0
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


def _pct(values, q):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(len(v) * q))]


def _row(label, values, unit="ms"):
    vals = [x for x in values if x is not None]
    if not vals:
        return f"  {label:<34} (no data)"
    return (f"  {label:<34} n={len(vals):<5} p50={_pct(vals, .5):>8.1f}  "
            f"p95={_pct(vals, .95):>8.1f}  max={max(vals):>8.1f} {unit}")


def build_report(since=None):
    """Latency/efficiency breakdown of the speech -> text -> translation
    pipeline from the phrase_trace, phrase_displayed and *_stats log lines.
    `since` is an ISO prefix such as '2026-10-31T14:00' (UTC offset as logged)."""
    ev = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)]
    if since:
        ev = [e for e in ev if e.get("ts", "") >= since]
    by = {}
    for e in ev:
        by.setdefault(e.get("event"), []).append(e)
    tr, disp = by.get("phrase_trace", []), by.get("phrase_displayed", [])
    col = lambda rows, k: [r.get(k) for r in rows]
    gpu = [(r.get("encode_ms") or 0) + (r.get("decode_ms") or 0) for r in tr if "decode_ms" in r]
    out = [f"Pipeline report  ({ev[0]['ts'][:19]} .. {ev[-1]['ts'][:19]}, {len(ev)} log lines)"
           if ev else "No log entries."]

    out.append("\nFinished phrases (speech -> final text)")
    out.append(f"  phrases: {len(tr)}   cut reasons: {dict(Counter(col(tr, 'reason')))}   "
               f"models: {dict(Counter(col(tr, 'model')))}")
    for label, vals, unit in [
            ("audio length", [x * 1000 if x else None for x in col(tr, "audio_sec")], "ms"),
            ("speech start -> phrase end", col(tr, "capture_ms"), "ms"),
            ("phrase end -> worker picks up", col(tr, "queue_wait_ms"), "ms"),
            ("  VAD trim (CPU)", col(tr, "vad_ms"), "ms"),
            ("  mel features (CPU)", col(tr, "mel_ms"), "ms"),
            ("  GPU lock wait", col(tr, "lock_wait_ms"), "ms"),
            ("  GPU encode+decode", gpu, "ms"),
            ("worker total", col(tr, "process_ms"), "ms"),
            ("result -> on screen (UI)", col(disp, "ui_ms"), "ms"),
            ("PAUSE -> FINAL TEXT ON SCREEN", col(disp, "speech_end_to_text_ms"), "ms"),
            ("SPEECH START -> FIRST PREVIEW", col(disp, "first_preview_ms"), "ms"),
            ("pause -> online translation shown",
             col(by.get("translation_displayed", []), "speech_end_to_translation_ms"), "ms"),
            ("tokens generated", col(tr, "tokens"), "tok"),
            ("speech level", col(tr, "rms_dbfs"), "dBFS"),
            ("peak level", col(tr, "peak_dbfs"), "dBFS"),
            ("AGC gain applied", col(tr, "agc_gain_db"), "dB"),
            ("mean VAD speech probability", col(tr, "vad_mean"), ""),
            ("previews per phrase", col(tr, "previews"), "")]:
        out.append(_row(label, vals, unit))

    ps = by.get("preview_stats", [])
    out.append("\nLive previews (10 s windows; p50 of each window's p50)")
    for k in ("age_ms", "process_ms", "encode_ms", "decode_ms", "vad_ms"):
        out.append(_row(k, [w[k]["p50"] for w in ps if isinstance(w.get(k), dict)]))
    shown = sum(w["process_ms"]["n"] for w in ps if isinstance(w.get("process_ms"), dict))
    out.append(f"  previews processed: {shown}   skipped as stale: "
               f"{sum(w.get('skipped_stale', 0) for w in ps)}   not shown (empty/unchanged): "
               f"{sum(w['not_shown']['n'] for w in ps if isinstance(w.get('not_shown'), dict))}   "
               f"preview model: {dict(Counter(w.get('preview_model') for w in ps))}")

    cs = by.get("capture_stats", [])
    out.append("\nAudio capture (10 s windows; worst value per window)")
    for k in ("read_gap_ms", "loop_ms", "vad_ms", "backlog_frames"):
        out.append(_row(k + " (max)", [w[k]["max"] for w in cs if isinstance(w.get(k), dict)],
                        "frames" if k == "backlog_frames" else "ms"))
    out.append(_row("level p50", [w["level_dbfs"]["p50"] for w in cs
                                  if isinstance(w.get("level_dbfs"), dict)], "dBFS"))
    out.append(_row("speech frames", col(cs, "speech_pct"), "%"))

    rs = by.get("resource_stats", [])
    out.append("\nResources and UI (10 s windows)")
    for k, unit in (("process_cpu_pct", "%"), ("system_cpu_pct", "%"), ("rss_mb", "MB"),
                    ("gpu_util_pct", "%"), ("gpu_mem_used_mb", "MB"), ("gpu_sm_clock_mhz", "MHz"),
                    ("gpu_temp_c", "C")):
        out.append(_row(k, col(rs, k), unit))
    out.append(_row("UI poll lag (p95 per window)",
                    [w["poll_lag_ms"]["p95"] for w in rs if isinstance(w.get("poll_lag_ms"), dict)]))

    sc = by.get("sensitivity_changed", [])
    if sc:
        out.append("\nSensitivity changes vs measured input level (capture level p50, dBFS)")
        for e in sc:
            before = [w for w in cs if w["ts"] < e["ts"] and isinstance(w.get("level_dbfs"), dict)]
            after = [w for w in cs if w["ts"] > e["ts"] and isinstance(w.get("level_dbfs"), dict)]
            b = before[-1]["level_dbfs"]["p50"] if before else None
            a = after[0]["level_dbfs"]["p50"] if after else None
            out.append(f"  {e['ts'][11:19]}  sensitivity={e.get('value')}  "
                       f"threshold={e.get('vad_threshold')}  level before={b}  after={a}")

    flags = []
    if (_pct(col(tr, "queue_wait_ms"), .95) or 0) > 100:
        flags.append("Phrases wait >100 ms for the worker: GPU busy with previews or backlog.")
    if any((w.get("read_gap_ms") or {}).get("max", 0) > 150 for w in cs):
        flags.append("Audio reads stalled >150 ms: capture loop fell behind the device.")
    if (_pct([w["poll_lag_ms"]["p95"] for w in rs if isinstance(w.get("poll_lag_ms"), dict)],
             .5) or 0) > 50:
        flags.append("UI thread is busy: results wait >50 ms before being drawn.")
    if ps and sum(w.get("skipped_stale", 0) for w in ps) > shown:
        flags.append("More previews skipped than processed: preview model can't keep up.")
    if (_pct(col(tr, "agc_gain_db"), .5) or 0) > 20:
        flags.append("AGC is boosting >20 dB: input is very quiet, raise the source volume.")
    if (_pct(col(tr, "peak_dbfs"), .95) or -100) > -0.5:
        flags.append("Peaks reach 0 dBFS: input is clipping, lower the source volume.")
    out.append("\nFindings")
    out.extend(f"  - {f}" for f in flags) if flags else out.append("  - nothing abnormal")
    return "\n".join(out)


def _words(text):
    import re
    text = (text or "").lower().replace("ё", "е")
    return re.findall(r"\w+", text)


def _align(ref, hyp):
    """Word edit distance with backtrace: (errors, missing words, substitutions)."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, missing, wrong = len(ref), len(hyp), [], []
    while i > 0 or j > 0:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                wrong.append(f"{ref[i - 1]}→{hyp[j - 1]}")
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            missing.append(ref[i - 1])
            i -= 1
        else:
            j -= 1
    return d[len(ref)][len(hyp)], missing[::-1], wrong[::-1]


def build_accuracy(since=None, ref_model="large-v3"):
    """Re-transcribe saved phrase recordings with a larger reference model
    (beam search, no filtering) and compare with what the app logged, to find
    words and whole phrases the app failed to recognise."""
    import sys
    sys.path.insert(0, str(APP_DIR))
    from interpreter import _add_cuda_dll_dirs, RECORD_DIR
    _add_cuda_dll_dirs()
    from faster_whisper import WhisperModel
    import ctranslate2

    traces = [e for p in iter_log_files(LOG_PATH) for e in read_jsonl(p)
              if e.get("event") == "phrase_trace" and e.get("audio_file")
              and (not since or e.get("ts", "") >= since)]
    traces = [t for t in traces if (RECORD_DIR / t["audio_file"]).exists()]
    if not traces:
        return "No phrase recordings found (they are saved from this version on)."
    device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    model = WhisperModel(ref_model, device=device,
                         compute_type="float16" if device == "cuda" else "int8")
    out = [f"Accuracy check: {len(traces)} phrases, app vs reference model '{ref_model}'\n"]
    tot_err = tot_ref = 0
    all_missing = Counter()
    missed_phrases = 0
    for t in traces:
        # Read the WAV ourselves: faster-whisper's file decoder is
        # incompatible with the installed PyAV version.
        import wave
        import numpy as np
        with wave.open(str(RECORD_DIR / t["audio_file"])) as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        segs, _ = model.transcribe(audio,
                                   language=t.get("src_lang"), beam_size=5,
                                   vad_filter=False, condition_on_previous_text=False)
        ref_text = " ".join(s.text for s in segs).strip()
        ref, hyp = _words(ref_text), _words(t.get("text"))
        if not ref:
            continue
        err, missing, wrong = _align(ref, hyp)
        tot_err += err
        tot_ref += len(ref)
        all_missing.update(missing)
        missed_phrases += not hyp
        if err:
            out.append(f"  {t['ts'][11:19]} seq={t['seq']} reason={t.get('reason')} "
                       f"errors={err}/{len(ref)} no_speech={t.get('no_speech_prob')} "
                       f"logprob={t.get('avg_logprob')}")
            out.append(f"     app: {t.get('text') or '(nothing)'}")
            out.append(f"     ref: {ref_text}")
            if missing:
                out.append(f"     missing: {' '.join(missing)}")
            if wrong:
                out.append(f"     wrong:   {', '.join(wrong)}")
    out.insert(1, f"  word error rate vs reference: {tot_err / max(1, tot_ref):.1%} "
                  f"({tot_err} errors / {tot_ref} words)   phrases with no text at all: "
                  f"{missed_phrases}\n  most often missed: "
                  f"{', '.join(f'{w}×{n}' for w, n in all_missing.most_common(10)) or '-'}\n")
    return "\n".join(out)


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

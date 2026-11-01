"""
simulate.py — test the interpreter with recorded audio instead of a device
========================================================================

Plays a playlist of test clips (from make_testdata.py, or fetch_testdata.py
with --data real) through the REAL
pipeline: the same audio capture thread (speech detection, phrase cutting,
music detection), transcription worker (Whisper, language detection, vocal
separation, translation) and settings as the app. Only the sound card is
replaced: a fake PyAudio "device" delivers the clips at real-time speed,
with a pause between them, switching language from clip to clip.

Afterwards it compares what came out with the correct text of each clip:

  language   did Whisper detect the right language?
  WER        word error rate of the transcription (0 % = perfect)
  translated did a translation appear, and by which engine?
  latency    end of the clip -> final text; start -> first preview

and prints a table per language plus the overall figures. The details are
saved to testdata/sim_results/<time>.json, and every phrase is also in the
app's normal log (session_id "sim-…"), so log_tools.py works on it too.

Usage (from this folder):
    .venv\\Scripts\\python simulate.py                       all clips, Auto-detect -> English
    .venv\\Scripts\\python simulate.py --src Russian         expect Russian (detection switches)
    .venv\\Scripts\\python simulate.py --tgt Russian --kinds song --limit 3
    .venv\\Scripts\\python simulate.py --langs de,fr,ru --preview tiny --gap 2
Options:
    --src NAME      source language as in the window ("Auto-detect", "Russian" …)
    --tgt NAME      target language (default English)
    --kinds K       speech, song or speech,song (default both)
    --langs L       comma-separated language codes to include (default all)
    --limit N       clips per language and kind (default all)
    --gap SEC       silence between clips (default 1.5)
    --preview P     off, tiny, small or main (default small)
    --model M       Whisper model size (default large-v3-turbo)
    --no-music      turn music mode off
    --offline       don't use the online translator
    --data D        synth (default: single-language TTS clips) or real
"""

import argparse
import json
import queue
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import interpreter as I                         # noqa: E402
import log_tools as LT                          # noqa: E402

REAL_PYAUDIO = I.pyaudio                      # build() swaps in a fake; live mode needs this
TESTDATA = APP_DIR / "testdata"
DATA_SETS = {"synth": TESTDATA / "synth",     # make_testdata.py: TTS voices, made-up music
             "real": TESTDATA,                # fetch_testdata.py: FLEURS + Jamendo
             "feedback": TESTDATA / "feedback"}   # phrases reported with "Report a mistake"
DATA_DIR = DATA_SETS["synth"]                 # chosen with --data
RATE = 16000


# ─────────────────────────────────────────────────────────────────────────────
# A fake sound card: PyAudio's interface, fed from the playlist
# ─────────────────────────────────────────────────────────────────────────────

def speech_end(audio, block=320):
    """Sample index just after the last 20 ms block louder than -45 dBFS."""
    x = audio.astype(np.float32) / 32768
    n = len(x) // block
    if n == 0:
        return len(x)
    rms = np.sqrt((x[:n * block].reshape(n, block) ** 2).mean(axis=1))
    loud = np.nonzero(rms > 10 ** (-45 / 20))[0]
    return int((loud[-1] + 1) * block) if len(loud) else len(x)


def reference_words(clip):
    """Words of the correct text. For songs, lines in brackets are backing
    vocals ("(Rufen gemeinsam)"), which nobody expects to be transcribed."""
    import re
    text = clip["text"]
    if clip["kind"] == "song":
        text = re.sub(r"\([^)]*\)", " ", text)
    return LT._words(text)


LEVEL_TARGET_DBFS = -20     # each clip's voice is brought to this loudness (as the app's AGC)
LEVEL_PEAK_DBFS   = -1      # without peaks going above this
LEVEL_RANGE_DB    = (-20, 30)   # at most 20 dB quieter / 30 dB louder


def level_clip(audio):
    """Automatic gain for one clip: measure the loudness of the part that is
    actually sounding (20 ms blocks within 25 dB of the loudest, so pauses
    don't count) and scale the clip so that part sits at LEVEL_TARGET_DBFS,
    limited so peaks stay below LEVEL_PEAK_DBFS. Real recordings differ by
    20 dB or more; a quiet one could otherwise sit under the speech detector's
    threshold. Returns (int16 audio, gain in dB)."""
    x = audio.astype(np.float32) / 32768
    n = len(x) // 320
    if n == 0:
        return audio, 0.0
    power = (x[:n * 320].reshape(n, 320) ** 2).mean(axis=1)
    db = 10 * np.log10(power + 1e-12)
    active = power[(db > db.max() - 25) & (db > -70)]
    if not len(active):
        return audio, 0.0
    gain_db = LEVEL_TARGET_DBFS - 10 * np.log10(active.mean())
    peak_db = 20 * np.log10(np.abs(x).max() + 1e-9)
    gain_db = float(np.clip(min(gain_db, LEVEL_PEAK_DBFS - peak_db), *LEVEL_RANGE_DB))
    out = np.clip(x * 10 ** (gain_db / 20), -1, 1)
    return (out * 32767).astype(np.int16), round(gain_db, 1)


class Playlist:
    """All clips joined into one stream with silence between them. Remembers
    when (monotonic clock) each clip starts and ends playing. level=True
    applies level_clip() to each clip (its gain is stored as clip["gain_db"])."""

    def __init__(self, clips, gap_sec, level=True):
        """Load every clip's WAV file and join them, with gap_sec of silence
        between clips and 4 s at the end. self.spans holds, per clip, the
        sample positions where its voice starts and ends (used for scoring)."""
        parts, self.spans, pos = [], [], 0
        silence = np.zeros(int(gap_sec * RATE), np.int16)
        parts.append(silence); pos += len(silence)
        for clip in clips:
            with wave.open(str(DATA_DIR / clip["file"])) as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if level:
                audio, clip["gain_db"] = level_clip(audio)
            parts.append(audio)
            # The span ends where the speech ends: FLEURS clips have ~1 s of
            # silence at the end, which made "pause -> text" negative.
            # Synthetic clips know exactly where the vocals end (songs carry
            # on with music after them).
            end = (int(clip["vocal_end"] * RATE) if "vocal_end" in clip else speech_end(audio))
            start = int(clip.get("vocal_start", 0) * RATE)
            self.spans.append([pos + start, pos + end])
            pos += len(audio)
            parts.append(silence); pos += len(silence)
        parts.append(np.zeros(4 * RATE, np.int16))        # tail: let the last phrase end
        self.samples = np.concatenate(parts)
        self.t0 = None                                    # set when playback starts

    def time_of(self, sample):
        return self.t0 + sample / RATE

    @property
    def duration(self):
        return len(self.samples) / RATE


class FakeStream:
    """Delivers the playlist in real time, like a microphone would."""

    def __init__(self, playlist):
        self.p, self.pos = playlist, 0
        playlist.t0 = time.monotonic()

    def _due(self):
        return int((time.monotonic() - self.p.t0) * RATE)

    def get_read_available(self):
        return max(0, self._due() - self.pos)

    def read(self, frames, exception_on_overflow=False):
        """Like a real microphone: wait until `frames` samples' worth of
        time has passed, then return them (silence after the playlist)."""
        while self._due() < self.pos + frames:          # wait like a real device
            time.sleep(0.002)
        chunk = self.p.samples[self.pos:self.pos + frames]
        if len(chunk) < frames:                          # after the end: silence
            chunk = np.concatenate([chunk, np.zeros(frames - len(chunk), np.int16)])
        self.pos += frames
        return chunk.tobytes()

    def close(self):
        pass


class FakePyAudio:
    """Just enough of pyaudio.PyAudio for the capture thread: one 16 kHz
    mono input device that plays the playlist."""
    playlist = None

    def get_device_info_by_index(self, index):
        return {"name": "simulator", "maxInputChannels": 1, "defaultSampleRate": RATE,
                "isLoopbackDevice": False, "index": 0}

    get_default_input_device_info = lambda self: self.get_device_info_by_index(0)

    def is_format_supported(self, rate, **kw):
        if rate != RATE:
            raise ValueError("simulator plays 16 kHz only")
        return True

    def open(self, **kw):
        return FakeStream(FakePyAudio.playlist)

    def terminate(self):
        pass


class FakePyAudioModule:
    paInt16 = 8
    PyAudio = FakePyAudio


# ─────────────────────────────────────────────────────────────────────────────
# Building the pipeline (what InterpreterApp._start_session does, headless)
# ─────────────────────────────────────────────────────────────────────────────

class Loaders:
    """Local translators, created on demand (stands in for the window)."""
    def __init__(self):
        self._mt_loaders = {}
    _translator_for = I.InterpreterApp._translator_for


def build(args, playlist, say=print, level_cb=None):
    """level_cb(level_db, peak_db, speaking, speech_prob, threshold) is called
    for every 20 ms block, from the capture thread.
    playlist None = live: record the real device args.device (a loopback
    device captures whatever Windows plays on that output)."""
    if playlist is None:
        I.pyaudio = REAL_PYAUDIO                        # a real device
    else:
        I.pyaudio = FakePyAudioModule                   # the fake sound card
        FakePyAudio.playlist = playlist
    src_key, tgt_key = args.src, args.tgt
    src_whisper = I.LANGUAGES[src_key][0]
    tgt_code = I.LANGUAGES[tgt_key][0]

    say(f"Loading Whisper {args.model} …")
    main = I.WhisperModelLoader(model_size=args.model, status_cb=lambda m: None)
    main.start()
    preview = None
    if args.preview in ("tiny", "small"):
        preview = I.WhisperModelLoader(model_size=args.preview, status_cb=lambda m: None)
        preview.start()
    main.ready.wait()
    if main.model is None:
        raise RuntimeError(f"Whisper '{args.model}' failed to load: {main.error}")
    if getattr(main, "used_fallback", False):
        # The app falls back to Tiny on the CPU; a simulation with a different
        # model than asked for would give misleading figures, so stop.
        raise RuntimeError(f"Whisper '{args.model}' failed to load ({main.error or 'see log'}); "
                           "not simulating with the CPU fallback model")
    if preview:
        preview.ready.wait()
    scene = None
    if not args.no_music:
        from music import MusicScene
        scene = MusicScene(log=lambda e, **k: I.log_event(20, e, **k))
        t = time.monotonic()
        while scene.classifier is None and time.monotonic() - t < 60:
            time.sleep(0.2)
    loaders = Loaders()
    # Pre-load the local translators the clips will need, so the first
    # phrases aren't counted as "online" while a model downloads.
    langs = ({c["lang"] for c in playlist.clips} if playlist is not None
             else {src_whisper} - {"auto"})              # live: languages not known in advance
    for code in langs:
        for pair in I.local_route(code, tgt_code):
            loaders._translator_for(*pair)
    for loader in loaders._mt_loaders.values():
        loader.ready.wait()

    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    results = queue.Queue()
    phrases = queue.Queue(maxsize=32)
    partial_slot = None if args.preview == "off" else I.LatestSlot()
    early_slot = I.LatestSlot()
    capture = I.AudioCaptureThread(device_index=0 if playlist is not None else args.device,
                                   phrase_queue=phrases,
                                   status_cb=lambda m: results.put(("status", m)),
                                   level_cb=level_cb or (lambda *a: None), partial_slot=partial_slot,
                                   early_slot=early_slot)
    capture.set_sensitivity(args.sensitivity)
    if playlist is not None:
        # The gaps between clips are digital silence on purpose: don't warn that
        # "no sound reaches the device" (it would list the real apps playing)
        capture._report_source = lambda **kw: None
    capture.music_scene = scene
    worker = I.TranscriptionWorker(
        wid=0, model=main.model, src_whisper=src_whisper, src_lang_key=src_key,
        tgt_lang_key=tgt_key, phrase_queue=phrases, result_queue=results,
        status_cb=lambda m: results.put(("status", m)), partial_slot=partial_slot,
        preview_model=(preview.model if preview else None), run_id=1, record_audio=False,
        translator=loaders._translator_for(src_whisper, tgt_code), early_slot=early_slot,
        session_id=session_id,
        config={"model": args.model, "preview_choice": args.preview, "simulation": True,
                "sensitivity": args.sensitivity, "src_lang": src_key, "tgt_lang": tgt_key})
    worker.music_scene = scene
    worker.auto_language = True
    worker.translator_for = loaders._translator_for
    return capture, worker, results, session_id


# ─────────────────────────────────────────────────────────────────────────────
# Running and scoring
# ─────────────────────────────────────────────────────────────────────────────

def choose_clips(args):
    """The clips to play, from the chosen data set's manifest.jsonl, filtered
    by kind (speech/song) and language, at most args.limit per language and
    kind. They are interleaved (one of each group in turn) so the language
    changes from clip to clip, as it would in real use."""
    rows =[json.loads(l) for l in open(DATA_DIR / "manifest.jsonl", encoding="utf-8")]
    kinds = set(args.kinds.split(","))
    langs = set(args.langs.split(",")) if args.langs else None
    groups = {}
    for r in rows:
        if not r.get("file"):                 # a report saved without audio
            continue
        if r["kind"] in kinds and (not langs or r["lang"] in langs):
            groups.setdefault((r["kind"], r["lang"]), []).append(r)
    if args.limit:
        groups = {k: v[:args.limit] for k, v in groups.items()}
    # Interleave: one clip from each group in turn, so the language keeps changing
    order, lists = [], [list(v) for v in groups.values()]
    while any(lists):
        for lst in lists:
            if lst:
                order.append(lst.pop(0))
    return order


def strip_label(text):
    """Remove the "[German] " label the app adds before a language change."""
    import re
    return re.sub(r"^\[[^\]]+\]\s*", "", text or "")


def score(playlist, finals, previews):
    """Match each final phrase to the clip it overlaps most, then score clips."""
    per_clip = [{"finals": []} for _ in playlist.clips]
    for f in finals:
        tr = f["trace"] or {}
        end = tr.get("t_end")
        start = end - (tr.get("capture_ms") or 0) / 1000 if end else None
        if end is None:
            continue
        best, overlap = None, 0.0
        for i, (a, b) in enumerate(playlist.spans):
            o = min(end, playlist.time_of(b)) - max(start, playlist.time_of(a))
            if o > overlap:
                best, overlap = i, o
        if best is not None:
            per_clip[best]["finals"].append(f)
    out = []
    for clip, span, got in zip(playlist.clips, playlist.spans, per_clip):
        fs = got["finals"]
        hyp = " ".join(strip_label(f["text"]) for f in fs if f["text"]).strip()
        ref_words, hyp_words = reference_words(clip), LT._words(hyp)
        errors = LT._align(ref_words, hyp_words)[0] if ref_words else 0
        detected = [l for l in ((f["trace"] or {}).get("lang_used") or (f["trace"] or {}).get("lang")
                                for f in fs if f["text"]) if l]
        clip_end, clip_start = playlist.time_of(span[1]), playlist.time_of(span[0])
        # pause -> text: from the end of the clip to its last result
        last_final = max((f["t"] for f in fs if f["text"]), default=None)
        first_prev = min((t for t, s in previews if clip_start - 0.2 <= t), default=None)
        out.append({
            "file": clip["file"], "kind": clip["kind"], "lang": clip["lang"],
            "ref": clip["text"], "hyp": hyp,
            "translation": " ".join(f["translated"] or "" for f in fs).strip(),
            "engines": sorted({(f["trace"] or {}).get("translation_engine") or "?" for f in fs if f["text"]}),
            "detected": max(set(detected), key=detected.count) if detected else None,
            "ref_words": len(ref_words), "errors": errors,
            "latency_ms": round((last_final - clip_end) * 1000) if last_final else None,
            "first_preview_ms": round((first_prev - clip_start) * 1000)
                                if first_prev and first_prev < clip_end + 2 else None,
            "music_mode": any((f["trace"] or {}).get("music_mode") for f in fs),
            "phrases": len(fs),
        })
    return out


def report(rows, args, wall_sec):
    """Print one line per (kind, language) and the overall figures."""
    def summary(rs):
        """The figures for one table row (a group of clips): language right,
        word error rate, translated, median times, music mode."""
        words =sum(r["ref_words"] for r in rs)
        errs = sum(min(r["errors"], r["ref_words"]) for r in rs)
        lid = [r for r in rs if r["detected"]]
        lat = [r["latency_ms"] for r in rs if r["latency_ms"] is not None]
        prev = [r["first_preview_ms"] for r in rs if r["first_preview_ms"] is not None]
        return {"clips": len(rs),
                "lang_ok": f"{sum(r['detected'] == r['lang'] for r in lid)}/{len(rs)}",
                "wer": f"{errs / words:.0%}" if words else "-",
                "translated": f"{sum(bool(r['translation']) for r in rs)}/{len(rs)}",
                "latency": f"{int(np.median(lat))} ms" if lat else "-",
                "preview": f"{int(np.median(prev))} ms" if prev else "-",
                "music": f"{sum(r['music_mode'] for r in rs)}/{len(rs)}"}
    print(f"\nSimulation: {args.src} -> {args.tgt}, model {args.model}, previews {args.preview}, "
          f"music mode {'off' if args.no_music else 'on'}, {wall_sec / 60:.1f} min")
    head = f"  {'kind':<7}{'lang':<6}{'clips':>6}{'lang ok':>9}{'WER':>6}{'transl.':>9}" \
           f"{'pause->text':>12}{'1st preview':>12}{'music':>7}"
    print(head)
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["lang"]), []).append(r)
    for (kind, lang), rs in sorted(groups.items()):
        s = summary(rs)
        print(f"  {kind:<7}{lang:<6}{s['clips']:>6}{s['lang_ok']:>9}{s['wer']:>6}{s['translated']:>9}"
              f"{s['latency']:>12}{s['preview']:>12}{s['music']:>7}")
    for kind in sorted({r["kind"] for r in rows}):
        s = summary([r for r in rows if r["kind"] == kind])
        print(f"  {kind + ' all':<13}{s['clips']:>6}{s['lang_ok']:>9}{s['wer']:>6}{s['translated']:>9}"
              f"{s['latency']:>12}{s['preview']:>12}{s['music']:>7}")
    wrong = [r for r in rows if r["detected"] and r["detected"] != r["lang"]]
    if wrong:
        print("\n  Wrong language detected:")
        for r in wrong[:10]:
            print(f"    {r['file']}: {r['lang']} heard as {r['detected']}: {r['hyp'][:60]!r}")
    print("\n  Examples (reference | recognised | translation):")
    for r in rows[:: max(1, len(rows) // 8)]:
        print(f"    [{r['kind']} {r['lang']}] {r['ref'][:45]!r}\n"
              f"        -> {r['hyp'][:60]!r}\n        => {r['translation'][:60]!r}")


def main():
    """Command-line use: read the options (see the top of this file), run
    the simulation printing each phrase, then print the results table."""
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="Auto-detect")
    ap.add_argument("--tgt", default="English")
    ap.add_argument("--kinds", default="speech,song")
    ap.add_argument("--langs", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--preview", default="small", choices=("off", "tiny", "small", "main"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--data", default="synth", choices=tuple(DATA_SETS))
    args = ap.parse_args()
    global DATA_DIR
    DATA_DIR = DATA_SETS[args.data]
    if not (DATA_DIR / "manifest.jsonl").exists():
        maker = "make_testdata.py" if args.data == "synth" else "fetch_testdata.py"
        sys.exit(f"No test data: run  .venv\\Scripts\\python {maker}  first.")

    def show(kind, data):
        if kind == "status":
            print(data, flush=True)
        elif kind == "final" and data["text"]:
            print(f"  {strip_label(data['text'])[:70]!r} -> {str(data['translated'])[:50]!r}", flush=True)

    rows, path, wall_sec = run(args, show)
    report(rows, args, wall_sec)
    print(f"\nDetails: {path}")


def run(args, on_event=lambda kind, data: None, stop=None, level_cb=None):
    """Play the clips through the pipeline, score them and save the results.
    Returns (rows, results path, wall-clock seconds).

    on_event(kind, data) is called from this thread with
      ("status", text)          loading progress
      ("started", playlist)     playback begins (playlist.t0 is set)
      ("partial", (seq, heard, translated, t_start, t))  a live preview
      ("final", final)          a finished phrase: dict seq/text/translated/trace/t
      ("translation", final)    an online translation arrived later for that final
    stop: a threading.Event that ends the run early (clips not yet played
    are left out of the results). level_cb: see build().

    args.data == "live": no clips; records the real device args.device until
    stop is set. Each phrase is a result row (no correct text to compare)."""
    from concurrent.futures import ThreadPoolExecutor
    live = args.data == "live"
    if live:
        playlist, clips = None, []
        on_event("status", f"Recording live from {getattr(args, 'device_name', args.device)}")
    else:
        clips = choose_clips(args)
        playlist = Playlist(clips, args.gap, level=getattr(args, "level_clips", True))
        playlist.clips = clips
        on_event("status", f"{len(clips)} clips, {playlist.duration / 60:.1f} min of audio")
    capture, worker, results, session_id = build(args, playlist, lambda m: on_event("status", m),
                                                 level_cb)
    I.log_event(20, "simulation_started", session_id=session_id, clips=len(clips), **vars(args))
    I.log_session_context(session_id, simulation=True)

    worker.start(); capture.start()
    if not live:
        while playlist.t0 is None:                        # the capture thread opens the stream
            time.sleep(0.01)
    on_event("started", playlist)
    finals, previews, wall0 = [], [], time.monotonic()
    # Online translation in the background: waiting for it here held up every
    # later result by the round trip (~0.2-0.5 s per phrase)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sim-translate")
    done_translations = queue.Queue()

    def translate_online(final, lang):
        """Runs on a pool thread: translate one phrase via the internet,
        store it in the phrase (final), then hand it back to the loop."""
        try:
            final["translated"] = I._translate_cached(strip_label(final["text"]),
                                                      "auto" if lang == "auto" else lang,
                                                      I.LANGUAGES[args.tgt][1])
            if final["trace"] is not None:
                final["trace"]["translation_engine"] = "online"
        except Exception as e:
            I.log_exception("sim_online_translation_failed", e, seq=final["seq"], lang=lang)
            final["translated"] = f"[online translation failed: {e}]"
        done_translations.put(final)

    while True:
        try:
            kind, data = results.get(timeout=0.05)
        except queue.Empty:
            kind = None
        now = time.monotonic()
        if kind == "final":
            seq, text, translated, trace = data
            final = {"seq": seq, "text": text, "translated": translated, "trace": trace, "t": now}
            finals.append(final)
            on_event("final", final)
            if translated is None and text and not args.offline:
                pool.submit(translate_online, final, (trace or {}).get("lang_used") or worker.src_whisper)
        elif kind == "partial":
            previews.append((now, data[1]))
            on_event("partial", data)
        elif kind == "status":
            on_event("status", data)
        while not done_translations.empty():
            on_event("translation", done_translations.get())
        if not live and now - playlist.t0 > playlist.duration + 3 and results.empty():
            break
        if stop is not None and stop.is_set():
            break
    capture.stop(); worker.stop()
    pool.shutdown(wait=True)
    while not done_translations.empty():
        on_event("translation", done_translations.get())
    if live:
        rows = live_rows(finals)
    else:
        rows = score(playlist, finals, previews)
        played = sum(playlist.time_of(b) < time.monotonic() - 3 for _, b in playlist.spans)
        rows = rows[:played] if stop is not None and stop.is_set() else rows
    out_dir = TESTDATA / "sim_results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}{'-live' if live else ''}.json"
    path.write_text(json.dumps({"args": vars(args), "session_id": session_id, "clips": rows},
                               ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return rows, path, time.monotonic() - wall0


def live_row(final):
    """One result row for a phrase heard live (nothing to compare it with)."""
    tr = final["trace"] or {}
    return {"seq": final["seq"], "kind": "live", "lang": None,
            "detected": tr.get("lang_used") or tr.get("lang"),
            "hyp": strip_label(final["text"]), "translation": final["translated"] or "",
            "latency_ms": round((tr["t_result"] - tr["t_end"]) * 1000) if tr.get("t_end") and tr.get("t_result") else None,
            "audio_sec": tr.get("audio_sec"), "music_mode": bool(tr.get("music_mode")),
            "engine": tr.get("translation_engine")}


def live_rows(finals):
    return [live_row(f) for f in finals if f["text"]]


if __name__ == "__main__":
    main()

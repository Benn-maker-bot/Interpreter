"""
simulate.py — test the interpreter with recorded audio instead of a device
========================================================================

Plays a playlist of test clips (from fetch_testdata.py) through the REAL
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

TESTDATA = APP_DIR / "testdata"
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


class Playlist:
    """All clips joined into one stream with silence between them. Remembers
    when (monotonic clock) each clip starts and ends playing."""

    def __init__(self, clips, gap_sec):
        parts, self.spans, pos = [], [], 0
        silence = np.zeros(int(gap_sec * RATE), np.int16)
        parts.append(silence); pos += len(silence)
        for clip in clips:
            with wave.open(str(TESTDATA / clip["file"])) as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            parts.append(audio)
            # The span ends where the speech ends: FLEURS clips have ~1 s of
            # silence at the end, which made "pause -> text" negative.
            self.spans.append([pos, pos + speech_end(audio)])
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


def build(args, playlist):
    I.pyaudio = FakePyAudioModule                       # the fake sound card
    FakePyAudio.playlist = playlist
    src_key, tgt_key = args.src, args.tgt
    src_whisper = I.LANGUAGES[src_key][0]
    tgt_code = I.LANGUAGES[tgt_key][0]

    print(f"Loading Whisper {args.model} …", flush=True)
    main = I.WhisperModelLoader(model_size=args.model, status_cb=lambda m: None)
    main.start()
    preview = None
    if args.preview in ("tiny", "small"):
        preview = I.WhisperModelLoader(model_size=args.preview, status_cb=lambda m: None)
        preview.start()
    main.ready.wait()
    if main.model is None:
        sys.exit(f"Whisper failed to load: {main.error}")
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
    for code in {c["lang"] for c in playlist.clips}:
        for pair in I.local_route(code, tgt_code):
            loaders._translator_for(*pair)
    for loader in loaders._mt_loaders.values():
        loader.ready.wait()

    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    results = queue.Queue()
    phrases = queue.Queue(maxsize=32)
    partial_slot = None if args.preview == "off" else I.LatestSlot()
    early_slot = I.LatestSlot()
    capture = I.AudioCaptureThread(device_index=0, phrase_queue=phrases,
                                   status_cb=lambda m: results.put(("status", m)),
                                   level_cb=lambda *a: None, partial_slot=partial_slot,
                                   early_slot=early_slot)
    capture.set_sensitivity(args.sensitivity)
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
    rows = [json.loads(l) for l in open(TESTDATA / "manifest.jsonl", encoding="utf-8")]
    kinds = set(args.kinds.split(","))
    langs = set(args.langs.split(",")) if args.langs else None
    groups = {}
    for r in rows:
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
        words = sum(r["ref_words"] for r in rs)
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = ap.parse_args()
    if not (TESTDATA / "manifest.jsonl").exists():
        sys.exit("No test data: run  .venv\\Scripts\\python fetch_testdata.py  first.")

    clips = choose_clips(args)
    playlist = Playlist(clips, args.gap)
    playlist.clips = clips
    print(f"{len(clips)} clips, {playlist.duration / 60:.1f} min of audio", flush=True)
    capture, worker, results, session_id = build(args, playlist)
    I.log_event(20, "simulation_started", session_id=session_id, clips=len(clips), **vars(args))

    worker.start(); capture.start()
    finals, previews, wall0 = [], [], time.monotonic()
    while True:
        try:
            kind, data = results.get(timeout=0.2)
        except queue.Empty:
            kind = None
        now = time.monotonic()
        if kind == "final":
            seq, text, translated, trace = data
            if translated is None and text and not args.offline:
                lang = (trace or {}).get("lang_used") or worker.src_whisper
                try:
                    translated = I._translate_cached(strip_label(text),
                                                     "auto" if lang == "auto" else lang,
                                                     I.LANGUAGES[args.tgt][1])
                    if trace is not None:
                        trace["translation_engine"] = "online"
                except Exception as e:
                    translated = f"[online translation failed: {e}]"
            finals.append({"seq": seq, "text": text, "translated": translated,
                           "trace": trace, "t": now})
            if text:
                print(f"  {strip_label(text)[:70]!r} -> {str(translated)[:50]!r}", flush=True)
        elif kind == "partial":
            previews.append((now, data[1]))
        if playlist.t0 and now - playlist.t0 > playlist.duration + 3 and results.empty():
            break
    capture.stop(); worker.stop()
    rows = score(playlist, finals, previews)
    report(rows, args, time.monotonic() - wall0)
    out_dir = TESTDATA / "sim_results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"args": vars(args), "session_id": session_id, "clips": rows},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDetails: {path}")


if __name__ == "__main__":
    main()

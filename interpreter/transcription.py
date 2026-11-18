"""Speech to text with Whisper (faster-whisper on CTranslate2).

- WhisperModelLoader loads a Whisper model on a background thread, on the
  GPU if there is one, falling back to the CPU.
- TranscriptionWorker takes finished phrases from the capture thread,
  transcribes them, checks the result (text_filters.py), translates it
  (translation.py) and puts it on the window's result queue. It also makes
  the quick live previews while a phrase is still being spoken.
- save_feedback() stores a phrase the user reported with "Report a mistake",
  so it can be replayed in the simulator.
"""

import queue
import threading
import time
import logging
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable

from .config import (
    AUTO_LANG_PROB, CONTEXT_RESET_SEC, CONTEXT_TOKENS, DEFAULT_MODEL_CPU,
    FEEDBACK_DIR, GPU_CPU_THREADS, LANG_MIN_SPEECH_SEC, LANG_SOURCE_MAX,
    LANG_SWITCH_PROB, LANGUAGE_NAMES, LANGUAGES, LOGPROB_FLOOR, MODELS,
    NEW_LANG_MUSIC_PROB, NEW_LANG_PROB, NO_SPEECH_PROB, PREVIEW_OFF,
    RECENT_AUDIO_PHRASES, RECORD_DIR, RECORD_KEEP, RETRY_USELESS, SAMPLE_RATE,
    SUSPECT_PHRASES, VOICE_GAP_DB)
from .logging_setup import log_event, log_exception
from .system import (
    COMPUTE_TYPE, _DEFAULT_MODEL, INFERENCE_LOCK, NUM_WORKERS, WHISPER_DEVICE)
from .audio import apply_agc, LatestSlot, PerfWindow
from .text_filters import (
    clean_translation, collapse_repetitions, hallucination_reason,
    is_hallucination, suspicious_reasons, _words)
from .translation import local_route, _translate_cached


# ─────────────────────────────────────────────────────────────────────────────
# Whisper model loader - GPU with automatic CPU fallback
#
# Loading a model takes seconds, so it happens on a background thread.
# ─────────────────────────────────────────────────────────────────────────────

class WhisperModelLoader(threading.Thread):
    """
    Loads a Whisper model on a background thread so the window stays usable.
    
    This class handles the complex process of loading AI models in the
    background without freezing the user interface. Loading can take seconds
    (or minutes on the first download). When done, `ready` is set and
    `model` (or `error`) is filled in. If loading on the GPU fails it falls
    back to the Tiny model on the CPU.
    
    The Whisper models are large AI models that perform speech recognition.
    Loading these models can take considerable time, especially the first time
    when they need to be downloaded from the internet and converted for use.
    
    This threading approach allows the application to remain responsive while
    the model is loading in the background.
    """
    def __init__(self, model_size: str, status_cb: Callable[[str], None]) -> None:
        """
        Prepare to load `model_size` ("tiny", "small", "medium", "large-v3-turbo").
        
        This method initializes the model loader with the specified model size
        and a callback function to report progress.
        
        Args:
            model_size (str): The size of the Whisper model to load
            status_cb (Callable[[str], None]): Function to show progress messages
        """
        super().__init__(daemon=True, name="whisper-loader")
        # model_size: which model to load; status_cb: shows progress messages.
        # model / error: the loaded model, or why loading failed; ready: set when
        # finished either way. active_device / active_compute_type: what was
        # actually used (they change if it fell back to the CPU); worker_count,
        # default_model_key, used_fallback: details of that fallback.
        self.model_size          = model_size
        self.status_cb           = status_cb
        self.model               = None
        self.error               = None
        self.ready               = threading.Event()
        self.active_device       = WHISPER_DEVICE
        self.active_compute_type = COMPUTE_TYPE
        self.worker_count        = NUM_WORKERS
        self.default_model_key   = _DEFAULT_MODEL
        self.used_fallback       = False

    def _try_load(self, model_size, device, compute_type):
        """Load the model on one device, then "warm it up" with one second of silence
        so the first real phrase is not slowed down by start-up work.

        Returns (model, milliseconds taken).
        """
        from faster_whisper import WhisperModel
        label = "GPU" if device == "cuda" else "CPU"
        self.status_cb(f"Loading Whisper '{model_size}' on {label}…")
        t0    = time.perf_counter()
        
        # INFERENCE_LOCK serialises inference, so a second worker only adds
        # contention. One worker gets every physical core instead (logical
        # count / 2 on SMT CPUs; hyperthreads slow CTranslate2 down). On GPU the
        # CPU threads still run feature extraction and VAD.
        kwargs = dict(device=device, compute_type=compute_type)
        try:
            import psutil
            cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        except ImportError:
            cores = max(1, (os.cpu_count() or 4) // 2)
        # On the GPU the CPU threads only do small jobs; 24 of them per model
        # (one model each for final text, previews, ...) just compete with
        # Teams/Telemost for the CPU, so the GPU gets a few.
        kwargs["cpu_threads"] = min(GPU_CPU_THREADS, cores) if device == "cuda" else cores
            
        try:
            from faster_whisper.utils import download_model
            download_model(model_size, local_files_only=True)
        except Exception:
            self.status_cb(f"Downloading Whisper '{model_size}' (first time only, "
                           f"can take a few minutes)…")
            log_event(logging.INFO, "whisper_model_download_started", model_size=model_size)
        if device == "cuda" and "large" in model_size:
            # Turbo: 10% faster (189 vs 209 ms/phrase), half the GPU memory and
            # no accuracy loss (7.6% vs 8.0% WER on the recordings). Smaller
            # models showed no speed gain, so they stay float16.
            kwargs["compute_type"] = "int8_float16"
        model = WhisperModel(model_size, **kwargs)
        self.active_compute_type = kwargs["compute_type"]
        model._size_name = model_size
        self.status_cb("Pre-warming model…")
        import numpy as np
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        # One token is enough to warm the encoder and decoder; letting it
        # decode freely on silence wastes seconds generating junk.
        list(model.transcribe(dummy, language="en", beam_size=1, max_new_tokens=1,
                               vad_filter=False, without_timestamps=True)[0])
        return model, round((time.perf_counter() - t0) * 1000, 2)

    def run(self):
        """Thread entry point: load on the chosen device, fall back to CPU if needed."""
        try:
            # ── Primary attempt ───────────────────────────────────────────
            try:
                # First attempt: load the requested model on the chosen device (normally
                # the GPU). If that works we are done.
                self.model, ms = self._try_load(self.model_size, WHISPER_DEVICE, COMPUTE_TYPE)
                label = "GPU" if WHISPER_DEVICE == "cuda" else "CPU"
                self.status_cb(f"✔ Whisper '{self.model_size}' ready on {label}.")
                log_event(logging.INFO, "whisper_model_ready",
                          model_size=self.model_size, device=label,
                          compute_type=self.active_compute_type, elapsed_ms=ms, fallback_used=False)
            # Loading failed. On the CPU there is nothing else to try, so the error
            # is passed on. On the GPU, fall back to the small Tiny model on the CPU
            # so the app still works (slowly) instead of not at all.
            except Exception as first_err:
                log_exception("whisper_model_primary_load_failed", first_err,
                              model_size=self.model_size, device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE)
                if WHISPER_DEVICE != "cuda":
                    raise first_err
                # ── CPU fallback ──────────────────────────────────────────
                fallback_size = MODELS.get(DEFAULT_MODEL_CPU, "tiny")
                self.status_cb("GPU load failed - falling back to CPU (Tiny model)…")
                self.model, ms     = self._try_load(fallback_size, "cpu", "int8")
                self.model_size    = fallback_size
                self.active_device = "cpu"
                self.active_compute_type = "int8"
                self.worker_count  = 1
                self.default_model_key = DEFAULT_MODEL_CPU
                self.used_fallback = True
                self.status_cb(f"✔ Whisper '{fallback_size}' ready on CPU (fallback).")
                log_event(logging.WARNING, "whisper_model_cpu_fallback_ready",
                          original_device=WHISPER_DEVICE, fallback_device="cpu",
                          active_model_size=fallback_size, elapsed_ms=ms)
        # Nothing worked: remember the error so the window can show it.
        except Exception as e:
            self.error = str(e)
            self.status_cb(f"ERROR loading model: {e}")
            log_exception("whisper_model_load_failed", e,
                          model_size=self.model_size, device=self.active_device,
                          compute_type=self.active_compute_type)
        # Whatever happened, signal 'ready' so the window stops waiting.
        finally:
            self.ready.set()


# ─────────────────────────────────────────────────────────────────────────────
# Transcription worker
#
# This section turns audio into text: save_feedback() for "Report a
# mistake", then TranscriptionWorker itself, the heart of the program. The
# checks it runs on Whisper's text (hallucinations, results that look
# wrong) are in text_filters.py.
# ─────────────────────────────────────────────────────────────────────────────


def save_feedback(
    seq: int,
    audio: bytes | None,
    trace: dict[str, Any],
    recognised: str,
    translation: str,
    problems: list[str],
    correct_text: str = "",
    actual_lang: str | None = None,
    note: str = "",
) -> Path | None:
    """Save a phrase the user reported as wrong: its audio (16 kHz WAV), what
    was shown, what was wrong and the full phrase_trace. It is added to
    testdata/feedback/manifest.jsonl in the simulator's format, so the
    simulator can replay every reported phrase ("Your reported phrases") and
    a fix can be checked against the user's real mistakes.
    Returns the WAV path (None if the audio is no longer kept)."""
    import json, wave
    trace = dict(trace or {})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = None
    clips = FEEDBACK_DIR / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    if audio:
        path = clips / f"{stamp}_{seq}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(audio)
    row = {
        "file": str(path.relative_to(FEEDBACK_DIR)) if path else None,
        "kind": "song" if trace.get("music_mode") else "speech",
        "lang": actual_lang or trace.get("lang_used") or trace.get("lang") or trace.get("src_lang"),
        # The correct text, if the user typed it; otherwise nothing to score
        # word errors against (the simulator then checks language and speed)
        "text": correct_text.strip(), "verified": bool(correct_text.strip()),
        "recognised": recognised, "translation": translation, "problems": problems,
        "note": note, "reported_at": stamp, "session_id": trace.get("session_id"), "seq": seq,
        "trace": {k: v for k, v in trace.items() if not k.startswith("t_")},
    }
    with open(FEEDBACK_DIR / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    log_event(logging.INFO, "user_reported_mistake", seq=seq, problems=problems,
              recognised=recognised[:200], correct_text=correct_text[:200],
              actual_lang=actual_lang, detected_lang=trace.get("lang_used"),
              audio_saved=bool(path), note=note[:200], music_mode=trace.get("music_mode"),
              end_rule=trace.get("end_rule"), lang_reason=trace.get("lang_reason"))
    return path


class TranscriptionWorker(threading.Thread):
    """
    This is the core processing thread that performs speech recognition and
    translation on captured audio phrases.
    
    Finished phrases (phrase_queue) always take priority; live previews
    (partial_slot) only run when the model would otherwise sit idle.
    
    When the target is English, Whisper translates directly (task="translate")
    on the GPU: no network round trip, no rate limits, and it makes live
    previews of the *translation* cheap enough to show while the friend talks.
    
    This worker thread processes audio data from the capture thread, runs
    the Whisper speech recognition model, translates text when needed, and
    sends results back to the main UI thread for display.
    """
    def __init__(self, wid, model, src_whisper, src_lang_key, tgt_lang_key,
                 phrase_queue, result_queue, status_cb,
                 partial_slot: "LatestSlot | None" = None,
                 preview_model=None, run_id=0, record_audio=False, translator=None,
                 early_slot: "LatestSlot | None" = None, session_id: str = "",
                 config: "dict | None" = None, channel: str = ""):
        """
        Store everything the worker needs for one listening session.
        
        This method initializes a transcription worker with all necessary
        parameters for processing audio and generating text.
        
        Args:
            wid: Worker number
            model: The main Whisper model (used for finished phrases)
            src_whisper: Language code of the speech, e.g. "ru"
            src_lang_key: Source language menu name
            tgt_lang_key: Target language menu name
            phrase_queue: Finished phrases from AudioCaptureThread
            result_queue: Where results go back to the window
            status_cb: Function to show status messages
            partial_slot: Preview audio (LatestSlot)
            preview_model: Smaller model for previews, or a function returning one
            run_id: Session identifier
            record_audio: Save each phrase as a WAV for later accuracy checks
            translator: Opus-MT translator, or a function returning one (or None)
            early_slot: Slot for early finals from capture thread
            session_id: Unique identifier for the session
            config: Settings in force
            channel: Name of the channel ("Me", "Friend") when listening to two devices
        """
        super().__init__(daemon=True, name=f"transcription-worker-{channel or wid}")
        self.channel      = channel
        # wid: worker number; model: main Whisper model; model_lock: the shared
        # INFERENCE_LOCK; src_whisper: speech language code ("ru"); src/tgt_lang_key:
        # language menu names; phrase_queue / result_queue: input / output;
        # status_cb: shows status messages.
        self.wid          = wid
        self.model        = model
        self.model_lock   = INFERENCE_LOCK
        self.src_whisper  = src_whisper
        # "Auto-detect": the language is decided per phrase (lang_now is the
        # latest one, used for previews and the next prompt)
        self.auto_source  = src_whisper == "auto"
        self.lang_now     = "en" if self.auto_source else src_whisper
        self._last_shown_lang = None
        # Languages heard recently (newest first), plus the target language:
        # the candidates for a short, unclear phrase
        self._recent_langs = [LANGUAGES.get(tgt_lang_key, ("en", "en"))[0]]
        self.translator_for = None    # window's _translator_for: any local pair
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key
        self.phrase_queue = phrase_queue
        self.result_queue = result_queue
        self.status_cb    = status_cb
        self.partial_slot = partial_slot
        # Optional smaller model just for live previews (e.g. Tiny: ~30 ms vs
        # ~110 ms for Small); finished phrases still use the main model. May be
        # a callable, re-checked per preview, so a model that finishes loading
        # after the session started is picked up straight away.
        self._preview_model_src = preview_model
        # run_id: which listening session this is (also in the logs);
        # record_audio: save phrase WAVs; _stats: preview timings for the logs;
        # _stop_evt: set by stop().
        self.run_id       = run_id
        self.session_id   = session_id      # unique per app launch (run_id restarts at 1)
        self.config       = config or {}    # settings in force, copied into every phrase_trace
        self.early_slot   = early_slot      # early finals from the capture thread
        self.music_scene  = None            # music.MusicScene (set by the window)
        self._skip_context = False          # True while decoding a phrase in music mode
        self._last_vocals  = None           # separated voice of the last phrase (saved by _record)
        self._sep_cache    = {}             # vocal separation done ahead for the phrase in progress
        # Audio of the last RECENT_AUDIO_PHRASES phrases (seq -> 16 kHz PCM), so
        # the window's "Report a mistake" can save the phrase the user flags
        self.recent_audio: OrderedDict[int, bytes] = OrderedDict()
        # Automatic language detection (set by the window) and the target
        # language code (a phrase already in it needs no translation)
        self.auto_language = True
        self.tgt_whisper = LANGUAGES.get(tgt_lang_key, ("en", "en"))[0]
        self._early       = None            # (seq, phrase blocks, recognise() result, ms) kept
                                            # until the matching final arrives
        self.record_audio = record_audio
        self._stats       = PerfWindow()
        self._stop_evt    = threading.Event()
        tgt_code          = LANGUAGES.get(tgt_lang_key, ("en", "en"))[0]
        # Whisper can translate to English itself (not the turbo model);
        # a local Opus-MT translator, when loaded, is preferred for any pair.
        self._whisper_can_translate = (tgt_code == "en" and src_whisper not in ("en", "auto"))
        self._translator_src = translator
        # _last_final: number of the last finished phrase (older previews are
        # skipped); _last_preview: the last preview sent (repeats are skipped);
        # _prev_phrase_end: when the previous phrase ended (for the context reset).
        self._last_final  = -1
        self._last_preview = None
        self._prev_phrase_end = 0.0
        self._previews_by_seq = {}   # seq -> [shown count, last preview text]
        self._session = Counter()    # totals for the session_summary log line

    def stop(self):
        """Ask the worker to finish (None wakes it if it is waiting for a phrase)."""
        self._stop_evt.set()
        self.phrase_queue.put(None)

    def _record(self, seq, audio_bytes):
        """Save the phrase audio (16 kHz mono WAV) for accuracy analysis;
        keeps the newest RECORD_KEEP files. Returns the path or None."""
        if not self.record_audio:
            return None
        try:
            import wave
            RECORD_DIR.mkdir(exist_ok=True)
            path = RECORD_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_r{self.run_id}_s{seq}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                w.writeframes(audio_bytes)
            # Music mode: also keep the separated voice ("…_s12.vocals.wav"),
            # so log_tools.py can compare recognition on the mix and the voice.
            vocals, self._last_vocals = self._last_vocals, None
            if vocals is not None:
                import numpy as np
                with wave.open(str(path.with_suffix(".vocals.wav")), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                    w.writeframes((np.clip(vocals, -1, 1) * 32767).astype(np.int16).tobytes())
            old = sorted(f for f in RECORD_DIR.glob("*.wav")
                         if not f.name.endswith(".vocals.wav"))[:-RECORD_KEEP]
            for f in old:
                # A file can be locked (open in another program): skip it and
                # try again next time, instead of failing this recording.
                try:
                    f.unlink(missing_ok=True)
                    f.with_suffix(".ref.json").unlink(missing_ok=True)
                    f.with_suffix(".vocals.wav").unlink(missing_ok=True)
                except OSError:
                    pass
            return path.name
        except Exception as e:
            log_exception("phrase_record_failed", e)
            return None

    @property
    def translator(self):
        """The local translator if it has finished loading, else None."""
        src = self._translator_src
        return src() if callable(src) else src

    @property
    def local_translate(self):
        """True when Whisper's batched translate task is the engine."""
        model = self.model
        return (self.translator is None and self._whisper_can_translate
                and "turbo" not in (getattr(model, "_size_name", "") or ""))

    def _mt(self, text, src=None):
        """Translate locally with Opus-MT from `src` (default: the expected
        language) to the target, directly or via English. Returns (text, ms),
        or (None, 0) if no local model is available or ready (the window
        then uses the online translator)."""
        src = src or self.src_whisper
        if not text:
            return None, 0.0
        if self.translator_for is not None:
            hops = local_route(src, self.tgt_whisper)
            translators = [self.translator_for(*pair) for pair in hops]
            translators = [t() if callable(t) else t for t in translators]
        else:                                # tests: only the one given translator
            translators = [self.translator] if src == self.src_whisper else []
        if not translators or any(t is None for t in translators):
            return None, 0.0
        t0 = time.perf_counter()
        try:
            out = text
            for translator in translators:
                out = clean_translation(out, translator.translate(out))
        except Exception as e:
            log_exception("local_translation_failed", e, chars=len(text), src=src)
            return None, 0.0
        return out, round((time.perf_counter() - t0) * 1000, 2)

    @property
    def preview_model(self):
        """The model to use for previews (None means: use the main model)."""
        src = self._preview_model_src
        return src() if callable(src) else src

    def _prepare(self):
        """Prompt tokens and VAD options, built once per worker."""
        from faster_whisper.vad import VadOptions
        self._special = {}   # id(model) -> its special tokens (see _tokens_for)
        # Previous finished text per task, fed back as context so each chunk
        # is decoded knowing what came before (fixes words lost at cuts).
        self._history   = {"transcribe": [], "translate": []}
        self._history_t = 0.0
        self._vad_opts  = VadOptions(min_silence_duration_ms=300, speech_pad_ms=200)
        self._n_frames  = self.model.feature_extractor.nb_max_frames

    def _decode(self, audio_np, tasks, model=None, beam_size=1, detect=False):
        """Return one text per task. VAD runs once (CPU) and trims silence,
        AGC levels the clip, the GPU encodes it once, then all tasks decode as
        one batch from copies of that encoding. Whisper's encoder always
        processes a padded 30 s window, so encoding once instead of per task
        saves ~20-30% of the GPU time."""
        import numpy as np
        import ctranslate2
        from faster_whisper.vad import get_speech_timestamps
        model = model or self.model
        info = {"audio_sec": round(len(audio_np) / SAMPLE_RATE, 3)}
        t = time.perf_counter()
        stamps = get_speech_timestamps(audio_np, self._vad_opts)
        info["vad_ms"] = round((time.perf_counter() - t) * 1000, 2)
        if not stamps:
            info["no_speech"] = True
            return ["" for _ in tasks], info
        audio_np, info["agc_gain_db"] = apply_agc(audio_np[stamps[0]["start"]:stamps[-1]["end"]])
        info["agc_gain_db"] = round(info["agc_gain_db"], 1)
        info["speech_sec"] = round(len(audio_np) / SAMPLE_RATE, 3)
        t = time.perf_counter()
        feats = model.feature_extractor(audio_np)[:, :self._n_frames]
        # Always the full 30 s window: a shorter one encodes 6x faster but
        # raised Turbo's word errors from 8% to 64% on the recordings.
        if feats.shape[1] < self._n_frames:
            feats = np.pad(feats, ((0, 0), (0, self._n_frames - feats.shape[1])))
        single = ctranslate2.StorageView.from_array(
            np.ascontiguousarray(feats[None], dtype=np.float32))
        info["mel_ms"] = round((time.perf_counter() - t) * 1000, 2)
        lock_t0 = time.perf_counter()
        with self.model_lock:
            t = time.perf_counter()
            info["lock_wait_ms"] = round((t - lock_t0) * 1000, 2)
            if len(tasks) == 1:
                enc = model.model.encode(single, to_cpu=False)
            else:
                enc = model.model.encode(single, to_cpu=True)
                enc = ctranslate2.StorageView.from_array(
                    np.repeat(np.asarray(enc), len(tasks), axis=0))
            # Note: CUDA calls are asynchronous, so encode_ms can be partly
            # absorbed into decode_ms; their sum is the reliable GPU figure.
            info["encode_ms"] = round((time.perf_counter() - t) * 1000, 2)
            if detect and self.auto_source:
                self._detect_auto(model, enc, info)
            t = time.perf_counter()
            results = model.model.generate(
                enc, self._prompts_for(tasks, model),
                beam_size=beam_size, max_length=self._max_length(info, tasks, model),
                return_no_speech_prob=True,
                return_scores=True, suppress_blank=True)
            info["decode_ms"] = round((time.perf_counter() - t) * 1000, 2)
            # Language check (finished phrases only). If the phrase is
            # clearly in another language, decode it again in that
            # language, plus Whisper's own translation into English.
            other = (self._detect_language(model, enc, info)
                     if detect and not self.auto_source else None)
            if other:
                t = time.perf_counter()
                single_enc = enc
                if len(tasks) > 1:   # enc holds one copy per task: keep one
                    single_enc = ctranslate2.StorageView.from_array(np.asarray(enc)[:1])
                # Transcribe only: Turbo was trained without the translate
                # task (tests: its "English" just repeated the German).
                alt_tasks = ["transcribe"]
                results = [model.model.generate(
                    single_enc, [self._lang_prompt(model, other, task)], beam_size=beam_size,
                    max_length=self._max_length(info, ["transcribe"], model),
                    return_no_speech_prob=True, return_scores=True, suppress_blank=True)[0]
                    for task in alt_tasks]
                info["lang_switched"] = other
                info["lang_decode_ms"] = round((time.perf_counter() - t) * 1000, 2)
        texts, dropped = [], []
        info["no_speech_prob"], info["avg_logprob"] = [], []
        for r in results:
            text = self._tokens_for(model)["tokenizer"].decode(r.sequences_ids[0]).strip()
            # CTranslate2 scores are length-normalised log-probs (~ avg_logprob)
            avg_lp = float(r.scores[0]) if r.scores else 0.0
            info["no_speech_prob"].append(round(r.no_speech_prob, 3))
            info["avg_logprob"].append(round(avg_lp, 3))
            # Whisper discards only when BOTH say "no speech"; using the
            # no-speech probability alone threw away half the phrases of real
            # call audio (music/noise raise it even when words are clear).
            if r.no_speech_prob > NO_SPEECH_PROB and avg_lp < LOGPROB_FLOOR:
                dropped.append(text)
                text = ""
            texts.append(text)
        if any(dropped):
            info["dropped_text"] = dropped
        info["tokens"] = sum(len(r.sequences_ids[0]) for r in results)
        info["_token_ids"] = [list(r.sequences_ids[0]) for r in results]
        if info.get("lang_switched"):
            # One transcription in the other language; same shape as tasks
            texts = [texts[0]] * len(tasks)
            info["_token_ids"] = info["_token_ids"][:1] * len(tasks)
        return texts, info

    def _retry_if_suspicious(self, audio_np, tasks, texts, token_ids, info):
        """If the result looks wrong (far too short for the audio, a stock
        phrase, low confidence or a repetition loop), decode again without the
        context prompt (a common trigger) using beam search, and keep the
        better result. Costs extra GPU time only on suspect phrases."""
        speech_sec = info.get("speech_sec") or len(audio_np) / SAMPLE_RATE
        logprob = (info.get("avg_logprob") or [0.0])[0]
        reasons = suspicious_reasons(texts[0], speech_sec, logprob)
        info["chars_per_sec"] = round(len(texts[0]) / max(0.1, speech_sec), 1)
        if not reasons or info.get("no_speech"):
            return texts, token_ids, info
        info["suspicious"] = reasons
        # Log analysis (real calls + simulator): retrying a "stock_phrase"
        # result never gave a better text. A "repetitive" loop can be fixed
        # (simulator, tr_04: 32 -> 4 word errors) once the result is checked
        # with the rule below; earlier retries of loops only looked useless
        # because that rule rejected every result shorter than the loop.
        if not set(reasons) - RETRY_USELESS:
            info["retry_skipped"] = "reasons that never improved on retry"
            return texts, token_ids, info
        # A 1-2 word fragment of doubtful speech (speech detector < 0.5) that
        # is only "low confidence": the audio is unclear, and a second decode
        # just guesses differently ("Oh" -> "Huh?"). Simulator replay with the
        # correct text: 16 such retries, net 2 word errors fewer, ~160 ms each.
        if (reasons == ["low_confidence"] and (info.get("vad_mean") or 0) < 0.5
                and len(texts[0].split()) <= 2):
            info["retry_skipped"] = "short unclear fragment"
            return texts, token_ids, info
        saved = self._history
        self._history = {k: [] for k in self._history}
        t0 = time.perf_counter()
        try:
            retry_texts, retry_info = self._decode(audio_np, tasks, beam_size=5)
        finally:
            self._history = saved
        retry_token_ids = retry_info.pop("_token_ids", [])
        retry_logprob = (retry_info.get("avg_logprob") or [-9.0])[0]
        retry_reasons = suspicious_reasons(retry_texts[0], speech_sec, retry_logprob)
        # Accept the retry only if it is real text (not a caption hallucination
        # such as "Субтитры делал DimaTorzok", not a stock phrase such as
        # "Продолжение следует...") and either
        #   - not suspect at all and more confident: its length doesn't matter
        #     (simulator: clean retries rejected only for being shorter would
        #     have cut word errors 11->1, 9->5, 13->11 and 32->4), or
        #   - no shorter than the original and less suspect (or equally
        #     suspect but more confident).
        retry_text = retry_texts[0]
        real = (bool(retry_text) and not is_hallucination(retry_text)
                and " ".join(_words(retry_text)) not in SUSPECT_PHRASES)
        better = real and ((not retry_reasons and retry_logprob > logprob)
                           or (len(retry_text) >= len(texts[0])
                               and (len(retry_reasons) < len(reasons)
                                    or (len(retry_reasons) == len(reasons)
                                        and retry_logprob > logprob))))
        info["retry"] = {"text": retry_texts[0], "avg_logprob": round(retry_logprob, 3),
                         "reasons": retry_reasons, "used": better,
                         "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "original_text": texts[0],
                         # False when only punctuation/capitals differ ("today" -> "Today.")
                         "words_changed": _words(retry_texts[0]) != _words(texts[0])}
        if better:
            texts, token_ids, reasons = retry_texts, retry_token_ids, retry_reasons
            info["avg_logprob"] = retry_info.get("avg_logprob")
        info["still_suspicious"] = reasons
        return texts, token_ids, info

    def _tokens_for(self, model):
        """Special-token ids for this model. They differ between model
        families: large-v3/turbo inserted a language token, shifting
        <|transcribe|> etc. by one, so Turbo's ids sent to a Tiny/Small
        preview model asked it for the wrong task (garbled, 4x slower
        previews). Ordinary text tokens are shared, so context carries over."""
        model_key = (id(model), self.lang_now)     # Auto-detect: per language
        if model_key not in self._special:
            tokenizer = model.hf_tokenizer
            token_id_of = lambda t: tokenizer.token_to_id(t)
            head = [token_id_of("<|startoftranscript|>"), token_id_of(f"<|{self.lang_now}|>")]
            tail = [token_id_of("<|notimestamps|>")]
            self._special[model_key] = {
                "prompts": {"transcribe": head + [token_id_of("<|transcribe|>")] + tail,
                            "translate": head + [token_id_of("<|translate|>")] + tail},
                "sot_prev": token_id_of("<|startofprev|>"),
                "tokenizer": tokenizer,
                "text_vocab": token_id_of("<|endoftext|>"),   # ids below this are plain text
            }
        return self._special[model_key]

    def _lang_prompt(self, model, lang, task):
        """Start-of-decoding tokens for another language (no context):
        <|startoftranscript|> <|de|> <|transcribe|> <|notimestamps|>."""
        tok = self._tokens_for(model)["tokenizer"]
        return [tok.token_to_id("<|startoftranscript|>"), tok.token_to_id(f"<|{lang}|>"),
                tok.token_to_id(f"<|{task}|>"), tok.token_to_id("<|notimestamps|>")]

    def _detect_auto(self, model, enc, info):
        """Auto-detect mode: decide this phrase's language before decoding
        it. A clear verdict (or enough speech) changes lang_now; an unsure
        one on a short clip keeps the previous language (a second of singing
        can look like Spanish). Earlier text in another language is dropped
        from the context."""
        try:
            guesses = model.model.detect_language(enc)[0]
        except Exception as e:
            info["lang_error"] = str(e)
            return
        probs = {code.strip("<|>"): float(p) for code, p in guesses}
        top, top_p = max(probs.items(), key=lambda kv: kv[1])
        info["lang"], info["lang_prob"] = top, round(top_p, 3)
        info["lang_top3"] = {k: round(v, 3) for k, v in
                             sorted(probs.items(), key=lambda kv: -kv[1])[:3]}
        sure = top_p >= AUTO_LANG_PROB or (info.get("speech_sec") or 0) >= LANG_MIN_SPEECH_SEC * 2
        why = (f"confident ({top_p:.2f})" if top_p >= AUTO_LANG_PROB
               else "long enough speech" if sure else None)
        # Unsure (a short or unclear clip): prefer a language already heard in
        # this session, or the target language, if it is at all plausible.
        # The simulator showed both bad choices: keeping a language that
        # scores ~0 ("Annyeong!" for Russian) and trusting a weak top guess
        # (Japanese 0.4 for the first second of Russian speech).
        if not sure:
            familiar = [l for l in self._recent_langs if probs.get(l, 0.0) >= 0.05]
            if familiar:
                top = max(familiar, key=lambda l: probs.get(l, 0.0))
                sure = True
                why = f"unsure ({top_p:.2f}): chose recently heard {top}"
            elif probs.get(self.lang_now, 0.0) < 0.10:
                sure = True              # nothing familiar fits: the best guess
                why = f"unsure ({top_p:.2f}): best guess, current language unlikely"
            else:
                why = f"unsure ({top_p:.2f}): kept {self.lang_now}"
            info["lang_unsure"] = True
        # A language new to this session must be clearly identified
        in_music = bool(self.music_scene is not None and getattr(self.music_scene, "active", False))
        need = NEW_LANG_MUSIC_PROB if in_music else NEW_LANG_PROB
        if sure and top != self.lang_now and top not in self._recent_langs and top_p < need:
            sure = False
            info["lang_new_rejected"] = f"{top} {top_p:.2f} < {need}"
            why = f"kept {self.lang_now}: new language {top} not certain enough ({top_p:.2f} < {need})"
        info["lang_reason"] = why
        if sure and top != self.lang_now:
            info["lang_changed_from"] = self.lang_now
            self.lang_now = top
            self._history = {k: [] for k in self._history}
        info["lang_used"] = self.lang_now
        if self.lang_now in self._recent_langs:
            self._recent_langs.remove(self.lang_now)
        self._recent_langs.insert(0, self.lang_now)
        del self._recent_langs[6:]

    def _detect_language(self, model, enc, info):
        """Whisper's language guess for this phrase (from the encoder output,
        ~4 ms). Records the top languages in info; returns the language to
        switch to, or None to keep the expected one."""
        try:
            guesses = model.model.detect_language(enc)[0]
        except Exception as e:
            info["lang_error"] = str(e)
            return None
        probs = {code.strip("<|>"): float(p) for code, p in guesses}
        top, top_p = max(probs.items(), key=lambda kv: kv[1])
        src_p = probs.get(self.src_whisper, 0.0)
        info["lang"], info["lang_prob"] = top, round(top_p, 3)
        info["src_lang_prob"] = round(src_p, 3)
        info["lang_top3"] = {k: round(v, 3) for k, v in
                             sorted(probs.items(), key=lambda kv: -kv[1])[:3]}
        if (top != self.src_whisper and top_p >= LANG_SWITCH_PROB and src_p <= LANG_SOURCE_MAX
                and (info.get("speech_sec") or 0) >= LANG_MIN_SPEECH_SEC):
            return top
        return None

    def _max_length(self, info, tasks, model):
        """Longest output allowed for this clip: about 15 tokens per second of
        speech plus a margin (Russian speech is ~8-10 tokens/s), instead of a
        flat 224. A stuck model ("а-а-а-а…" in the logs) then stops quickly,
        which also saves GPU time. The prompt's own tokens are added on top."""
        speech_sec = info.get("speech_sec") or 1.0
        cap = min(224, int(speech_sec * 15) + 20)
        return min(448, len(self._prompts_for(tasks, model)[0]) + cap)

    def _prompts_for(self, tasks, model):
        """Build the start-of-decoding instructions ("prompt") for each task.

        A Whisper prompt looks like:
            <|startofprev|> …previous sentence tokens… <|startoftranscript|> <|ru|>
            <|transcribe|> <|notimestamps|>
        i.e. "here is what was said before; now transcribe this Russian audio,
        without timestamps". The special-token numbers depend on the model
        (see _tokens_for).
        """
        special_tokens = self._tokens_for(model)
        histories = [[i for i in (self._history.get(t) or []) if i < special_tokens["text_vocab"]]
                 for t in tasks]
        if self._skip_context:   # lyrics: earlier phrases (talk) would mislead
            histories = [[] for _ in tasks]
        # A batch needs <|startoftranscript|> at the same position in every
        # prompt, so give each task the same number of context tokens.
        context_len = min(len(h) for h in histories)
        return [([special_tokens["sot_prev"]] + h[len(h) - context_len:] if context_len else []) + special_tokens["prompts"][t]
                for t, h in zip(tasks, histories)]

    def _remember(self, tasks, token_ids, texts):
        """Keep accepted text as context for the next chunk."""
        for task, ids, text in zip(tasks, token_ids, texts):
            if text:
                self._history[task] = (self._history[task] + ids)[-CONTEXT_TOKENS:]
        self._history_t = time.monotonic()

    def run(self):
        """Thread entry point. Runs the main loop and restarts it after an unexpected
        error (so one bad phrase cannot stop transcription for the whole session),
        then logs a session_summary when the session ends.
        """
        # An unexpected error used to kill this thread for good: phrases then
        # piled up and were dropped for the rest of the session. Log it and
        # carry on instead.
        while not self._stop_evt.is_set():
            try:
                self._run_loop()
                break
            except Exception as e:
                self._session["worker_errors"] += 1
                log_exception("transcription_worker_recovered", e, worker_id=self.wid)
                time.sleep(0.05)
        totals = dict(self._session)
        if totals.get("phrases"):
            totals["process_ms_mean"] = round(totals.pop("process_ms_total") / totals["phrases"], 1)
        log_event(logging.INFO, "session_summary", session_id=self.session_id,
                  channel=self.channel, run_id=self.run_id, config=self.config,
                  model=getattr(self.model, "_size_name", None), src_lang=self.src_whisper,
                  **totals)

    def _run_loop(self):
        """The worker's main loop.

        Finished phrases always come first. Only when there is no finished phrase
        waiting does it process the newest preview. For each finished phrase it:
        resets the context after a long silence, transcribes (with a retry if
        the result looks wrong), translates, saves the audio, logs a detailed
        "phrase_trace", and sends ("final", …) to the window.
        """
        import numpy as np
        # One-time set-up for this session (VAD settings, empty context).
        self._prepare()
        while not self._stop_evt.is_set():
            try:
                # ── Step 1: is there a finished phrase waiting?
                #    Wait at most 10 ms. If nothing arrives, item = False, which means
                #    'no finished phrase: use the spare time for a preview instead'.
                #    item = None is the stop signal sent by stop().
                item = self.phrase_queue.get(timeout=0.01)
            except queue.Empty:
                item = False
            if item is None:
                break
            # Every 10 s, log a summary of the previews made (preview_stats).
            if self._stats.due():
                preview_slot = self.partial_slot
                log_event(logging.INFO, "preview_stats", session_id=self.session_id,
                          channel=self.channel, run_id=self.run_id,
                          preview_model=getattr(self.preview_model or self.model,
                                                "_size_name", None),
                          skipped_stale=preview_slot.overwritten if preview_slot else 0,
                          **self._stats.summary())
                if preview_slot:
                    preview_slot.overwritten = 0
            # ── Step 2 (no finished phrase): make a preview of the newest audio.
            #    take() empties the preview_slot, so each preview is only made once. A
            #    preview for a phrase that has already been finished is skipped.
            if item is False and self.early_slot is not None:
                early = self.early_slot.take()
                if early and early[0] > self._last_final:
                    # Pause just started: transcribe the phrase now, keep the
                    # result for when its final arrives (EARLY_FINAL_SEC).
                    self._stats.add("early_started", 1)
                    early_seq, early_bytes, early_len = early
                    t_early = time.perf_counter()
                    try:
                        early_audio = (np.frombuffer(early_bytes, dtype=np.int16)
                                       .astype(np.float32) / 32768.0)
                        self._early = (early_seq, early_len, self._recognise(early_audio),
                                       round((time.perf_counter() - t_early) * 1000, 1))
                    except Exception as e:
                        self._early = None
                        log_exception("early_final_failed", e, seq=early_seq)
                    continue
            if item is False:
                partial = self.partial_slot.take() if self.partial_slot else None
                if partial and partial[0] > self._last_final:
                    self._process_partial(*partial)
                elif partial:
                    self._stats.add("dropped_after_final", 1)
                continue
            # ── Step 3: a finished phrase. Unpack it:
            #    seq        the phrase's number in this session (0, 1, 2 …)
            #    audio_bytes 16 kHz mono PCM audio
            #    meta       facts from the capture thread: when the speech started
            #               and ended (t_start/t_end), its level, VAD score …
            seq, audio_bytes, meta = item
            self._last_final = seq
            if audio_bytes:
                self.recent_audio[seq] = audio_bytes
                while len(self.recent_audio) > RECENT_AUDIO_PHRASES:
                    self.recent_audio.popitem(last=False)
            # Reset the context only after a real pause in the speech
            # gap = seconds of silence between the previous phrase and this one.
            # After a long silence the conversation has probably moved on, so the
            # previous text is no longer useful context and is forgotten.
            gap = (meta.get("t_start") or 0) - self._prev_phrase_end
            context_reset = self._prev_phrase_end > 0 and gap > CONTEXT_RESET_SEC
            if context_reset:
                self._history = {k: [] for k in self._history}
            self._prev_phrase_end = meta.get("t_end") or time.monotonic()
            t_dequeue = time.monotonic()
            t0 = time.perf_counter()
            try:
                # An empty phrase (too short to be speech): tell the window to clear
                # its preview, and move on.
                if not audio_bytes:
                    self.result_queue.put(("final", (seq, "", "", None)))
                    continue
                # Was this phrase already transcribed as an early final? It
                # matches if the capture thread offered exactly this many
                # blocks and only silence followed (and the context is the
                # same, i.e. it wasn't just reset).
                early = self._early
                early_used = bool(early and early[0] == seq and meta.get("early_blocks")
                                  and early[1] == meta["early_blocks"] and not context_reset)
                self._early = None
                if early_used:
                    rec = early[2]
                    self._session["early_used"] += 1
                else:
                    audio_np = (np.frombuffer(audio_bytes, dtype=np.int16)
                                .astype(np.float32) / 32768.0)
                    rec = self._recognise(audio_np)
                    self._session["early_missed"] += bool(meta.get("early_blocks"))
                tasks, token_ids, info = rec["tasks"], rec["token_ids"], rec["info"]
                if rec.get("switch_log"):
                    log_event(logging.INFO, "language_switched", seq=seq, **rec["switch_log"])
                whisper_translates = rec["whisper_translates"]
                text, translated = rec["text"], rec["translated"]
                engine, mt_ms = rec["engine"], rec["mt_ms"]
                lock_wait_ms = info.get("lock_wait_ms", 0.0)
                # Text still suspect after a retry is not fed back as context:
                # that is how one "Спасибо." turned the next phrase into another.
                if text and not info.get("still_suspicious") and not info.get("lang_switched"):
                    self._remember(tasks, token_ids,
                                   [text, translated] if whisper_translates else [text])
                # ── Step 6: record everything about this phrase in one log line
                #    (phrase_trace): timings of each stage, levels, confidence, the text
                #    itself and the saved audio file. log_tools.py analyses these lines.
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                t_result = time.monotonic()
                trace = {
                    "session_id": self.session_id, "channel": self.channel,
                    "run_id": self.run_id, "seq": seq, "reason": meta.get("reason"),
                    # early transcription on silence: was the speculative result
                    # used, and how long that transcription took (time saved)
                    "early_used": early_used,
                    "early_ms": early[3] if early_used else None,
                    "tasks": len(tasks), "translation_engine": engine if text else None,
                    "mt_ms": mt_ms,
                    "model": getattr(self.model, "_size_name", None),
                    "audio_sec": meta.get("audio_sec"), "rms_dbfs": meta.get("rms_dbfs"),
                    "peak_dbfs": meta.get("peak_dbfs"), "vad_mean": meta.get("vad_mean"),
                    "vad_threshold": meta.get("vad_threshold"), "previews": meta.get("previews"),
                    "onset_delay_ms": meta.get("onset_delay_ms"),
                    # why the phrase ended: pause rule (+ pause length and the
                    # speech score falling), or a forced cut (+ the cut point)
                    **{k: meta[k] for k in ("end_rule", "pause_ms", "end_probs", "cut_prob",
                                            "cut_level_db") if k in meta},
                    # speech start -> phrase end detected (the audio itself + pause wait)
                    "capture_ms": round((meta["t_end"] - meta["t_start"]) * 1000, 1)
                                  if meta.get("t_start") else None,
                    "queue_wait_ms": round((t_dequeue - meta["t_end"]) * 1000, 2),
                    "process_ms": elapsed_ms,
                    "text_chars": len(text),
                    "translated_chars": len(translated) if translated else 0,
                    # The words themselves, so recognition can be checked
                    # against the saved audio (log_tools.py accuracy)
                    "text": text, "translation": translated, "src_lang": self.src_whisper,
                    "context_tokens": len(self._history.get("transcribe", [])),
                    "context_reset": context_reset,
                    # What the live preview showed last, to measure how far
                    # previews are from the final text
                    "previews_shown": self._previews_by_seq.get(seq, [0, ""])[0],
                    "last_preview": self._previews_by_seq.pop(seq, [0, ""])[1],
                    "gap_before_sec": round(gap, 2) if self._prev_phrase_end else None,
                    "audio_file": self._record(seq, audio_bytes),
                    "queue_backlog": self.phrase_queue.qsize(),
                }
                trace.update({k: v for k, v in info.items() if k != "audio_sec"})
                # The settings in force (model, preview model, sensitivity …),
                # so log_tools.py compare can group results by setting.
                trace.update({f"cfg_{k}": v for k, v in self.config.items()})
                trace["cfg_preview_model"] = (PREVIEW_OFF if self.partial_slot is None else
                                              getattr(self.preview_model or self.model,
                                                      "_size_name", None))
                log_event(logging.INFO, "phrase_trace", **trace)
                totals = self._session
                totals["phrases"] += 1
                totals["words"] += len(text.split())
                totals["empty"] += not text
                totals["suspicious"] += bool(info.get("suspicious"))
                totals["retried"] += "retry" in info
                totals["retry_used"] += bool(info.get("retry", {}).get("used"))
                totals["still_suspicious"] += bool(info.get("still_suspicious"))
                totals["dropped_no_speech"] += bool(info.get("dropped_text"))
                totals["context_resets"] += context_reset
                totals["process_ms_total"] += elapsed_ms
                trace["t_end"], trace["t_result"] = meta["t_end"], t_result
                self.result_queue.put(("metric", ("transcription_ms", elapsed_ms)))
                if mt_ms:   # local (Opus-MT) translation time, for the Diagnostics tab
                    self.result_queue.put(("metric", ("translation_ms", mt_ms)))
                self.result_queue.put(("metric", ("lock_wait_ms",     lock_wait_ms)))
                # ── Step 7: send the result to the window, unless the session was
                #    stopped meanwhile (the window has already moved on).
                if self._stop_evt.is_set():
                    break   # session restarted: don't leak into the new one
                # translated None => the UI sends it for online translation
                self.result_queue.put(("final", (seq, text, translated, trace)))
            except Exception as e:
                self.result_queue.put(("final", (seq, "", "", None)))
                self.result_queue.put(("status", f"Worker {self.wid}: {e}"))
                log_exception("transcription_failed", e,
                              worker_id=self.wid,
                              input_bytes=len(audio_bytes) if audio_bytes else 0)

    def _recognise(self, audio_np):
        """Full recognition of one phrase: Whisper (with the retry for doubtful
        results), the hallucination check, and translation. Used for finished
        phrases and for early (speculative) finals. Changes nothing permanent:
        the caller decides whether to keep the result and update the context.
        Returns a dict with the texts, translation and timing details."""
        whisper_translates = self.local_translate
        tasks = ["transcribe", "translate"] if whisper_translates else ["transcribe"]
        music = self._music_prepare(audio_np)
        audio_np = music.pop("_audio", audio_np)
        if music.pop("_no_voice", False):     # instrumental music: nothing to say
            log_event(logging.INFO, "transcription_skipped_instrumental", worker_id=self.wid,
                      **{k: v for k, v in music.items() if not k.startswith("_")})
            return {"tasks": tasks, "token_ids": [[] for _ in tasks], "info": dict(music),
                    "whisper_translates": whisper_translates, "text": "",
                    "translated": "" if whisper_translates else None,
                    "engine": "none", "mt_ms": 0.0}
        self._skip_context = bool(music.get("music_mode"))
        try:
            texts, info = self._decode(audio_np, tasks, detect=True)
            token_ids = info.pop("_token_ids", [])
            if not info.get("lang_switched"):    # retries are tuned for the expected language
                texts, token_ids, info = self._retry_if_suspicious(
                    audio_np, tasks, texts, token_ids, info)
        finally:
            self._skip_context = False
        info.update(music)        # scores and mode go into the phrase's log line
        lang = info.get("lang_switched")
        if lang:
            return self._other_language_result(lang, texts, token_ids, info, tasks,
                                               whisper_translates)
        text       = collapse_repetitions(texts[0])
        translated = collapse_repetitions(texts[1]) if whisper_translates else None
        engine     = "whisper" if whisper_translates else "online"
        mt_ms      = 0.0
        if is_hallucination(text):
            info["discarded"] = hallucination_reason(text)
            log_event(logging.INFO, "transcription_discarded", worker_id=self.wid, text=text,
                      why=info["discarded"], speech_sec=info.get("speech_sec"),
                      vad_mean=info.get("vad_mean"), music_mode=bool(music.get("music_mode")),
                      no_speech_prob=info.get("no_speech_prob"), avg_logprob=info.get("avg_logprob"))
            text = ""
            translated = "" if whisper_translates else None
        elif whisper_translates and is_hallucination(translated):
            translated = ""
        elif self.auto_source:
            lang = info.get("lang_used") or self.lang_now
            if lang == self.tgt_whisper:
                translated, engine = text, "none (already target language)"
            else:
                out, mt_ms = self._mt(text, src=lang)
                if out is not None:
                    translated, engine = out, f"opus-mt-{lang}"
            # Label the text when the language changes ("[German] …")
            if lang != self._last_shown_lang:
                text = f"[{LANGUAGE_NAMES.get(lang, lang.upper())}] {text}"
                self._last_shown_lang = lang
        elif not whisper_translates:
            # None from _mt means "no local translator": the window will
            # then send the text to the online translator.
            out, mt_ms = self._mt(text)
            if out is not None:
                translated, engine = out, "opus-mt"
        return {"tasks": tasks, "token_ids": token_ids, "info": info,
                "whisper_translates": whisper_translates, "text": text,
                "translated": translated, "engine": engine, "mt_ms": mt_ms}

    def _other_language_result(self, lang, texts, token_ids, info, tasks, whisper_translates):
        """A phrase in another language than expected (German song, Bulgarian
        advert …). Show it labelled ("[German] …") and translate it:
          - already in the target language: shown as it is
          - otherwise: the online translator, told the detected language
            (no local model exists for German/Bulgarian/…; Turbo cannot
            translate). Offline, it says so instead of guessing.
        With "Detect other languages" off, it is labelled but not translated
        (the Russian model only produced nonsense from German)."""
        text = collapse_repetitions(texts[0])
        name = LANGUAGE_NAMES.get(lang, lang.upper())
        engine, mt_ms, translated = f"whisper-{lang}", 0.0, ""
        if is_hallucination(text):
            info["discarded"] = hallucination_reason(text)
            log_event(logging.INFO, "transcription_discarded", worker_id=self.wid, text=text,
                      lang=lang, why=info["discarded"], speech_sec=info.get("speech_sec"))
            text = ""
        tgt_google = LANGUAGES.get(self.tgt_lang_key, ("en", "en"))[1]
        if text and not self.auto_language:
            info["foreign_not_translated"] = True
            translated = f"({name} - not translated: 'Detect other languages' is off)"
        elif text and lang == self.tgt_whisper:
            translated, engine = text, "none (already target language)"
        elif text and (local := self._mt(text, src=lang))[0] is not None:
            translated, mt_ms = local      # translated once, used as it is
            engine = f"opus-mt-{lang}"
        elif text:
            t0 = time.perf_counter()
            try:
                translated = _translate_cached(text, lang, tgt_google) or ""
                engine = f"online-{lang}"
            except Exception as e:
                info["translation_error"] = f"{type(e).__name__}: {e}"
                translated = f"(no offline translation for {name})"
            mt_ms = round((time.perf_counter() - t0) * 1000, 1)
        # Logged by the caller once the result is kept: early finals that are
        # thrown away would otherwise log the same phrase several times.
        switch_log = dict(worker_id=self.wid, lang=lang, lang_prob=info.get("lang_prob"),
                          expected=self.src_whisper, src_lang_prob=info.get("src_lang_prob"),
                          text=text[:120], auto=self.auto_language, engine=engine)
        return {"tasks": tasks, "token_ids": token_ids, "info": info,
                "switch_log": switch_log,
                "whisper_translates": whisper_translates,
                "text": f"[{name}] {text}" if text else "",
                "translated": translated if text else ("" if whisper_translates else None),
                "engine": engine, "mt_ms": mt_ms}

    def _music_prepare(self, audio_np):
        """Music mode: classify the phrase (speech / singing / music) and, if
        music is playing and the vocal separator is ready, remove the
        instruments so Whisper hears the voice. Returns the fields to log,
        plus "_audio" (the cleaned audio) when separation ran."""
        import numpy as np
        scene = self.music_scene
        if scene is None:
            return {}
        try:
            out = scene.judge(audio_np)
            if out.get("music_mode") and scene.separator is not None:
                t0 = time.perf_counter()
                voice = scene.separator.separate(audio_np, cache=self._sep_cache)
                peak = float(np.abs(voice).max()) if len(voice) else 0.0
                out["separation_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                # chunks separated ahead of time while the phrase was still going
                out["separation_reused"] = f"{self._sep_cache.get('reused', 0)}/" \
                                           f"{self._sep_cache.get('chunks', 0)}"
                # Level kept (voice) vs removed (music), in dBFS
                rms = lambda x: 10 * np.log10(float(np.mean(x ** 2)) + 1e-12)
                out["voice_db"], out["music_removed_db"] = (round(rms(voice), 1),
                                                            round(rms(audio_np - voice), 1))
                mix_db = rms(audio_np)
                # A voice in the music (singing) if what is kept is within
                # VOICE_GAP_DB of the whole mix; otherwise instrumental only:
                # skip Whisper, which would only invent text.
                out["singing"] = scene.singing = out["voice_db"] > mix_db - VOICE_GAP_DB
                if not out["singing"]:
                    out["_no_voice"] = True
                elif peak > 1e-3:
                    out["_audio"] = (voice / peak * min(0.9, max(peak, 0.3))).astype(np.float32)
                self._last_vocals = out.get("_audio", voice)
            return out
        except Exception as e:
            self._music_failures = getattr(self, "_music_failures", 0) + 1
            log_exception("music_processing_failed", e, worker_id=self.wid,
                          failures=self._music_failures)
            if self._music_failures >= 3:
                # Same error on every phrase: switch music mode off for this
                # session (speech recognition carries on) instead of flooding
                # the window and log with the same error.
                self.music_scene = None
                log_event(logging.WARNING, "music_processing_disabled",
                          reason=f"{type(e).__name__}: {e}", failures=self._music_failures)
                self.status_cb("⚠ Music mode switched off after repeated errors "
                               "(see Diagnostics)")
            return {}

    def _process_partial(self, seq, audio_bytes, t_capture, t_start):
        """Make a quick preview of an unfinished phrase and send ("partial", …).

        Uses the preview model and does not update the context: previews are
        temporary and will be replaced by the finished text.
        """
        import numpy as np
        try:
            # How old the preview audio already is when we pick it up
            # age_ms: how old this preview's audio already is when we start on it.
            self._stats.add("age_ms", (time.monotonic() - t_capture) * 1000)
            audio_np = (np.frombuffer(audio_bytes, dtype=np.int16)
                        .astype(np.float32) / 32768.0)
            # Use the (usually smaller, faster) preview model, and decide whether
            # Whisper should also translate (only when no Opus-MT translator exists
            # and the model is able to translate; Turbo cannot).
            preview_model = self.preview_model
            whisper_translates = self._whisper_can_translate and self.translator is None and \
                "turbo" not in (getattr(preview_model or self.model, "_size_name", "") or "")
            tasks = ["transcribe", "translate"] if whisper_translates else ["transcribe"]
            t0 = time.perf_counter()
            texts, info = self._decode(audio_np, tasks, model=preview_model)
            self._stats.add("preview_is_main_model", int(self.preview_model is None))
            info.pop("_token_ids", None)
            if "dropped_text" in info:
                self._stats.add("dropped_no_speech", 1)
            self._stats.add("process_ms", (time.perf_counter() - t0) * 1000)
            self._stats.add("audio_sec", info.get("audio_sec", 0))
            for k in ("vad_ms", "mel_ms", "encode_ms", "decode_ms", "lock_wait_ms"):
                if k in info:
                    self._stats.add(k, info[k])
            # Throw away invented text, then translate the preview with Opus-MT so
            # the English preview appears together with the Russian one.
            heard = texts[0] if not is_hallucination(texts[0]) else ""
            translated = (texts[1] if whisper_translates and heard
                          and not is_hallucination(texts[1]) else None)
            if heard and not whisper_translates:
                translated, mt_ms = self._mt(heard)
                if mt_ms:
                    self._stats.add("mt_ms", mt_ms)
            # Only send the preview if it differs from the last one sent; repeating
            # an identical preview would just redraw the window for nothing.
            preview = (seq, heard, translated)
            if heard and preview != self._last_preview and not self._stop_evt.is_set():
                self._last_preview = preview
                preview_record = self._previews_by_seq.setdefault(seq, [0, ""])
                preview_record[0] += 1
                preview_record[1] = heard
                self.result_queue.put(("partial", preview + (t_start, time.monotonic())))
            else:
                self._stats.add("not_shown", 1)   # empty, hallucination or unchanged
            self.result_queue.put(("metric", ("partial_ms",
                                              round((time.perf_counter() - t0) * 1000, 2))))
        except Exception as e:
            log_exception("partial_transcription_failed", e, worker_id=self.wid, seq=seq)
        self._prefetch_separation(audio_np if "audio_np" in locals() else None)

    def _prefetch_separation(self, audio_np):
        """Music mode, while a phrase is still going: separate the voice of
        one more finished 6 s chunk now (~0.3 s), so the phrase's final
        transcription only has its last chunk left to do."""
        scene = self.music_scene
        if audio_np is None or scene is None or not scene.active or scene.separator is None \
                or self.phrase_queue.qsize():
            return
        try:
            t0 = time.perf_counter()
            if scene.separator.prefetch(audio_np, self._sep_cache):
                self._stats.add("separation_prefetch_ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._sep_cache = {}
            log_exception("separation_prefetch_failed", e, worker_id=self.wid)

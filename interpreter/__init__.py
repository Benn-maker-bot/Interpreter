"""
Voice Interpreter
=================

A desktop app that listens to speech, writes down what was said, and
translates it, live. Typical use: you are on a Teams/Telemost call with a
Russian-speaking friend whose voice plays through your FiiO headphones; the
app "listens" to the FiiO, shows the Russian words as they are spoken, and
shows the English translation next to them. It can also do the reverse
("My Voice": your microphone, English -> Russian).

This file contains the shared application logic and the classic window. It is
long, so this guide explains the big picture first. Section comments and
docstrings explain the main components; use the names below to jump to the
implementation and follow the data flow.

NEW TO THE PROJECT? Start with docs/START_HERE.md: short guides with
diagrams (docs/MAP.md), a file guide, how-to recipes, the tuning numbers
and a glossary. In VS Code press Ctrl+Shift+O to jump to any class or
function in this file by name.


HOW A PIECE OF SPEECH TRAVELS THROUGH THE PROGRAM
-------------------------------------------------

    sound card / FiiO  ──►  AudioCaptureThread  ──►  TranscriptionWorker  ──►  InterpreterApp (window)
                            (records audio,           (Whisper AI turns            (shows text,
                             finds speech,             audio into text;             speaks it,
                             cuts phrases)             Opus-MT translates it)       writes logs)

  1. CAPTURE  AudioCaptureThread reads the sound in small blocks (~32 ms).
     For "Friend" mode it records what an *output* device is playing (WASAPI
     "loopback"), so it hears the call audio you hear in your headphones.

  2. DETECT SPEECH  Each block is scored by Silero VAD (a tiny AI model that
     answers "is this speech?"). Speech blocks are collected into a phrase.
     A phrase ends at a short pause (0.3 s; 0.2 s once it is over 7 s long)
     or, at most 10 s long, is cut at the quietest moment. During music
     the pauses are longer (see MUSIC AND LANGUAGES below).

  3. PREVIEW  While someone is still talking, the unfinished phrase is sent
     every 0.2 s for a quick "preview" transcription, so text appears almost
     as the words are spoken. Words that two previews in a row agree on are
     shown normally; the still-changing end is grey.

  4. TRANSCRIBE  As soon as a pause starts (0.1 s), the phrase is already
     transcribed "early"; if the pause lasts, that result is used at once.
     Otherwise, when the phrase ends, TranscriptionWorker runs Whisper
     (OpenAI's speech-recognition model, via the fast "faster-whisper" /
     CTranslate2 engine) on the GPU. The previous sentences are given to
     Whisper as context, which improves accuracy. Results that look wrong
     are retried.

  5. TRANSLATE  The text is translated by Opus-MT (a small translation model,
     also on the GPU). If no local model exists for the language pair, the
     app falls back to Whisper's own translation, then to online services.

  6. DISPLAY  The window replaces the grey preview with the final text and
     translation, can read the translation aloud (Speaker), and logs timing
     and accuracy data for every phrase.


MUSIC AND LANGUAGES
-------------------
  MUSIC MODE  Once a second a sound classifier (YAMNet, in music.py)
     decides whether music is playing. If so: phrases may be longer, the
     speech detector is more sensitive, and a second AI model (UVR) removes
     the instruments so Whisper hears only the voice.

  AUTO-DETECT  With "Auto-detect" as the source language, Whisper guesses
     the language of every phrase. A short or unsure guess keeps the
     current language; a language not heard recently must be very certain
     (TranscriptionWorker._detect_auto explains the rules).

  WHY  Every phrase's log line ("phrase_trace") records why things
     happened: which rule ended it (end_rule), why the language was chosen
     (lang_reason), why text was thrown away (discarded) or retried.


TESTING AND IMPROVING
---------------------
  simulate.py / simulator_qt.py  play test recordings through this exact
                    code instead of a sound card and score the results.
  "Report a mistake" (Fluent window)  saves a phrase the app got wrong,
                    with its audio, to testdata/feedback; the simulator can
                    replay these reports to check a fix.
  log_tools.py trend   headline figures per day, to see if a change helped.


THREADS (things that run at the same time)
------------------------------------------
  A GUI must never freeze, so slow work runs on background threads that talk
  to the window through queues (thread-safe mailboxes):

    MainThread            the window (Tk/customtkinter). Only this thread may
                          touch widgets. It checks result_queue every 15 ms.
    audio-capture         AudioCaptureThread: records, detects speech.
    transcription-worker  TranscriptionWorker: Whisper + translation.
    translation-worker    TranslationWorker: online translation fallback.
    whisper-loader        WhisperModelLoader: loads AI models without freezing
                          the window. mt-loader-* does the same for Opus-MT.
    tts                   Speaker: text-to-speech.


FILES THE PROGRAM READS AND WRITES (all next to this file)
-----------------------------------------------------------
  interpreter.settings.json    your choices (model, devices, sensitivity…)
  interpreter.log.jsonl        detailed log, one JSON object per line
  interpreter.error.log.jsonl  warnings and errors only
  recordings/                  audio of recent phrases, for accuracy checks
  log_tools.py                 separate script: `python log_tools.py report`
                               (speed breakdown) and `... accuracy` (checks
                               the recognised words against a bigger model)

  AI models are downloaded once into the Hugging Face cache
  (%USERPROFILE%\\.cache\\huggingface) and reused.


GLOSSARY
--------
  Whisper      speech-to-text AI model. Sizes: tiny, small, medium, and
               large-v3-turbo ("Turbo"), the most accurate one used here.
  VAD          Voice Activity Detection: deciding which audio is speech.
  dBFS         decibels relative to full scale; 0 dBFS is the loudest a
               digital signal can be, -20 dBFS is typical speech.
  AGC          Automatic Gain Control: making quiet audio louder.
  Loopback     recording what an output device (speakers) is playing.
  Encoder/     the two halves of Whisper: the encoder "listens" to the
  decoder      audio, the decoder writes the words one token at a time.
  Token        a piece of a word that the AI models work with.
  Context      the previous sentences, fed back so Whisper knows what came
               before.
  Hallucination  text a model invents that was never said (Whisper
               sometimes writes "Спасибо." or YouTube subtitle credits).
  WER          word error rate: % of words wrong, missing or extra.
  CUDA         NVIDIA's system for running code on the graphics card (GPU).

  Sound and audio devices
  Sample       one number describing the sound wave at one instant. Here
               each sample is a 16-bit whole number ("int16", -32768..32767).
  Sample rate  how many samples per second: 48000 Hz (48 kHz) for most
               sound cards, 16000 Hz for Whisper. Converting between them
               is called resampling.
  PCM          "raw" audio: just the list of samples, no compression. The
               app passes audio around as PCM bytes.
  Mono/stereo  one channel / two channels (left and right). Downmixing
               averages the channels into one mono signal.
  Block/chunk  the app reads sound in small pieces of ~32 ms at a time.
  PortAudio    the cross-platform audio library that does the actual
               recording and playback. PyAudio (here PyAudioWPatch) is the
               Python wrapper around it; `pa = pyaudio.PyAudio()` starts it.
  Host API     Windows offers the same device through several audio systems
               (WASAPI, DirectSound, MME). WASAPI is the modern one and the
               only one that can do loopback.
  SAPI         Windows' built-in Speech API, used for reading text aloud.
  COM          the Windows system SAPI is built on; `comtypes` lets Python
               talk to it.

  AI models
  CTranslate2  a fast engine that runs the Whisper and Opus-MT models (on
               the GPU here). faster-whisper is built on it.
  Opus-MT      small neural translation models, one per language pair.
  Silero VAD   a tiny neural network that scores each 32 ms of audio for
               "is this speech?" (0 = no, 1 = yes).
  Batch        several inputs processed by the GPU in one go, which is
               almost as fast as processing one.
  Greedy /     two ways the decoder picks words: greedy takes the single
  beam search  most likely next token; beam search keeps the 5 best
               candidate sentences and picks the best whole sentence
               (slower, sometimes better).
  Log-prob     the model's confidence: the average logarithm of the
               probability of each token it wrote. 0 means certain;
               -0.2 is confident; below -1.0 means very unsure.
  No-speech    Whisper's own estimate that the audio contains no speech.
  probability
  Compression  how much a text shrinks when zipped. Text that repeats
  ratio        itself shrinks a lot (ratio above 2.4), a sign that Whisper
               got stuck in a loop.
  float16 /    number formats for the model's weights. Smaller formats are
  int8         faster and use less GPU memory.
  Hugging Face the website the AI models are downloaded from.
  Latency      delay: the time from something happening (speech) to its
               result (text on screen).

  Programming and files
  Thread       a part of the program that runs at the same time as the
               rest (see THREADS above).
  Queue        a thread-safe mailbox: one thread put()s items in, another
               get()s them out in the same order. Threads must communicate
               through queues instead of sharing variables carelessly.
  Lock         makes threads take turns at something that only one may do
               at a time (here: using the GPU). `with LOCK:` waits for the
               turn, runs the block, then lets the next thread in.
  Event        a flag one thread sets and others check (threading.Event),
               e.g. "please stop" or "model is ready".
  Callback     a function you hand to someone else to call later, e.g.
               status_cb: the capture thread calls it to show a message.
  JSON         a simple text format for data ({"name": value, …}). The log
               writes one JSON object per line ("JSON Lines", .jsonl).
  gzip         a compression format (like .zip) used for old log files.
  DLL          a Windows library file (.dll) a program loads at run time;
               the NVIDIA CUDA libraries are DLLs.
  HTTP 429     the web's "too many requests" error: a service asks you to
               slow down.
  Asynchronous an operation that starts now and finishes later while the
               program carries on (e.g. SAPI speaking, or GPU work).
  Watchdog     code that regularly checks that something is still working
               and restarts it if not.
  Monotonic    time.monotonic(): a clock that only ever moves forward,
  clock        used for measuring durations (the wall clock can jump).
  p50 / p95    the median (half the values are lower) and the value that
               95 % of measurements stay under (shows the slow cases).


PYTHON FEATURES USED IN THIS FILE
---------------------------------
  If you know basic Python (variables, if/for, functions, lists, dicts),
  these are the less common features you will meet:

  class Name(threading.Thread):  a class that runs as a thread; its run()
                       method is the code that runs in the background after
                       .start() is called.
  self                 inside a class, the object itself (self.model is
                       "this object's model").
  __init__             the method that sets up a new object.
  @property            makes a method look like a plain attribute:
                       `worker.translator` actually calls a function.
  @staticmethod        a function stored in a class that doesn't need self.
  @lru_cache           remembers a function's results so calling it again
                       with the same arguments returns instantly.
  with x:              runs a block with something set up and always cleaned
                       up afterwards (a lock released, a file closed).
  try / except /       run code, catch errors in "except" instead of
  finally              crashing, and always run "finally" at the end.
  lambda a: a + 1      a tiny unnamed function written in one line.
  f"{x:.1f} dB"        an f-string: puts values into text; ":.1f" means one
                       decimal place.
  yield                makes a function produce values one at a time (a
                       generator) instead of building a whole list.
  x if cond else y     choose between two values on one line.
  [f(x) for x in xs]   a list comprehension: build a list in one line.
  *args / **kwargs     pass a variable number of arguments / named arguments;
                       log_event(..., **fields) stores any named values.
  _name                a leading underscore means "internal, used only
                       inside this file/class".
  queue.get(timeout=…) wait for an item, but give up (queue.Empty) after the
                       timeout so the loop can check whether to stop.
  self.after(ms, f)    Tk's timer: call function f after ms milliseconds on
                       the main thread (used for every repeating job).


READING THE CODE
----------------
  Sections are marked with banner comments (# ─── ...). In order:
  settings/constants -> logging -> GPU detection -> online translation ->
  audio device helpers -> audio capture -> settings file -> model loading ->
  transcription worker (with the level meter, local translator and speaker
  helpers just before it) -> the main window (InterpreterApp).
  Values in CAPITALS near the top are tuning knobs, each with a comment
  saying what it does and, where it was measured, why it has that value.
"""

import queue
import threading
import time
import struct
import math
import json
import logging
import os
import sys
import traceback
import uuid
import gzip
import shutil
from collections import Counter, deque, OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
import subprocess
import customtkinter as ctk
from loguru import logger   # the logging library behind log_event()

# The package's modules. "import *" brings in every public name, so
# `interpreter.SAMPLE_RATE` keeps working for the other scripts and tests.
from .config import *  # noqa: F401,F403
from .logging_setup import *  # noqa: F401,F403
from .system import *  # noqa: F401,F403
from .system import _add_cuda_dll_dirs, _DEFAULT_MODEL  # noqa: F401  (log_tools.py uses the first)
from .audio import *  # noqa: F401,F403
from .audio import _PYAUDIO_AVAILABLE  # noqa: F401  (session_controller.py uses it)
from .text_filters import *  # noqa: F401,F403
from .text_filters import _words  # noqa: F401
from .translation import *  # noqa: F401,F403
from .translation import _MT_LOADERS_LOCK, _translate_cached  # noqa: F401  (simulate.py uses the second)
from .speech import *  # noqa: F401,F403
from .settings import *  # noqa: F401,F403

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


log_event(logging.INFO, "app_module_loaded",
          app_dir=str(APP_DIR), log_path=str(LOG_PATH), error_log_path=str(ERROR_LOG_PATH))


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
# This section turns audio into text. First come the checks that spot
# text Whisper may have invented (hallucinations) or results that look
# wrong, then helper classes (level meter, local translator, speaker),
# then TranscriptionWorker itself, the heart of the program.
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


def ui_scale(widget) -> float:
    """The Windows display scaling customtkinter applies (e.g. 2.5 at 250 %).
    Plain canvases are NOT scaled automatically, so the meters multiply their
    sizes by this; without it they were thin with unreadable labels on a 4K
    screen."""
    try:
        from customtkinter import ScalingTracker
        return ScalingTracker.get_widget_scaling(widget)
    except Exception:
        return 1.0


class DbMeter(ctk.CTkCanvas):
    """
    Segmented LED-style level meter in dBFS, after broadcast PPMs:
    -60..0 dBFS, green below -18, amber to -6, red above; instant attack,
    20 dB/s fall-back, and a peak-hold marker that holds 1.5 s. Segments are
    drawn once and only recoloured, so updating costs almost nothing.
    
    This is the audio level meter displayed in the main window that shows
    how loud the input audio currently is. It provides visual feedback to
    help users adjust their microphone levels for optimal performance.
    """
    FLOOR_DB, SEG_DB = -60.0, 1.5
    FALL_DB_PER_SEC, HOLD_SEC = 20.0, 1.5
    TICKS = (-60, -48, -36, -24, -18, -12, -6, -3, 0)
    ZONES = ((-18, "#2ECC71", "#123d24"), (-6, "#F1C40F", "#4a3f0c"), (0.1, "#E74C3C", "#4a1712"))

    def __init__(self, master, **kw):
        """Create the canvas; the LED segments are drawn when it gets its size."""
        self._s = ui_scale(master)          # display scaling factor
        super().__init__(master, height=int(34 * self._s), highlightthickness=0,
                         bg="#1d1e1e", **kw)
        # _segs: one (dB value, rectangle id, lit colour, unlit colour) per LED;
        # _lit: each LED's current colour (so only changes are redrawn).
        # _rms / _peak: the level and peak currently shown, in dB; _peak_t: when
        # the peak was reached (for peak hold); _last_t: time of the last update.
        self._segs, self._lit = [], []
        self._rms = self._peak = self.FLOOR_DB
        self._peak_t = 0.0
        self._last_t = time.monotonic()
        self.bind("<Configure>", lambda e: self._layout())

    def _colours(self, db):
        """Return the (lit, unlit) colours for a level in dB (green/amber/red zone)."""
        for top, lit_colour, unlit_colour in self.ZONES:
            if db < top:
                return lit_colour, unlit_colour
        return self.ZONES[-1][1:]

    def _layout(self):
        """Draw all the LED segments and the scale numbers to fit the current width."""
        self.delete("all")
        width = max(100, self.winfo_width())
        sc = self._s
        segment_count = int(-self.FLOOR_DB / self.SEG_DB)
        seg_w = (width - 8) / segment_count
        self._segs = []
        for i in range(segment_count):
            db = self.FLOOR_DB + (i + 1) * self.SEG_DB
            lit_colour, unlit_colour = self._colours(db - 0.01)
            seg_left = 4 + i * seg_w
            rect_id = self.create_rectangle(seg_left, 3 * sc, seg_left + seg_w - 2 * sc, 19 * sc,
                                            fill=unlit_colour, width=0)
            self._segs.append((db, rect_id, lit_colour, unlit_colour))
        self._lit = [None] * segment_count
        for t in self.TICKS:
            x = 4 + (t - self.FLOOR_DB) / -self.FLOOR_DB * (width - 8)
            # negative font size = pixels, so it scales with the display
            self.create_text(min(max(x, 8 * sc), width - 8 * sc), 27 * sc, text=str(t),
                             fill="gray60", font=("Segoe UI", -int(10 * sc)))

    def update_level(self, rms_db, peak_db):
        """Show a new level. Rises instantly, falls slowly (20 dB/s) like a real
        meter, and keeps a peak marker for HOLD_SEC seconds.
        """
        now = time.monotonic()
        elapsed_sec, self._last_t = now - self._last_t, now
        fall_db = self.FALL_DB_PER_SEC * elapsed_sec
        self._rms = max(rms_db, self._rms - fall_db)
        if peak_db >= self._peak:
            self._peak, self._peak_t = peak_db, now
        elif now - self._peak_t > self.HOLD_SEC:
            self._peak = max(peak_db, self._peak - fall_db)
        self._paint()

    def reset(self):
        """Clear the meter (when listening stops)."""
        self._rms = self._peak = self.FLOOR_DB
        self._paint()

    def _paint(self):
        """Recolour only the segments whose state changed (cheap to call often)."""
        hold_idx = int((self._peak - self.FLOOR_DB) / self.SEG_DB) - 1
        for i, (db, rect_id, lit_colour, unlit_colour) in enumerate(self._segs):
            state = lit_colour if (db <= self._rms or i == hold_idx) else unlit_colour
            if self._lit[i] != state:
                self.itemconfigure(rect_id, fill=state)
                self._lit[i] = state


class SpeechMeter(ctk.CTkCanvas):
    """
    The speech detector's score (0-100 %) as a bar, with a white line at
    the threshold set by the Speech sensitivity slider. While the bar passes
    the line, the audio counts as speech (bar turns green).
    
    This is a different thing from the level meter above it: the level meter
    shows how LOUD the sound is; this shows how much it SOUNDS LIKE SPEECH.
    Sensitivity moves the threshold line, never the volume.
    
    This meter shows how confident the speech detection algorithm (Silero VAD)
    is that the current audio contains speech. It helps users understand
    whether their microphone is correctly detecting speech or if there's
    interference from background noise.
    """
    TICKS = (0, 25, 50, 75, 100)

    def __init__(self, master, **kw):
        self._s = ui_scale(master)
        super().__init__(master, height=int(34 * self._s), highlightthickness=0,
                         bg="#1d1e1e", **kw)
        self._shown, self._threshold, self._speaking = 0.0, 0.5, False
        self.bind("<Configure>", lambda e: self._layout())

    def _layout(self):
        """Draw the meter's fixed parts (background, threshold line, labels)
        again for the current width; runs when the window is resized."""
        sc, w = self._s, max(100, self.winfo_width())
        self.delete("all")
        self._x0, self._x1 = 4 * sc, w - 4 * sc
        self.create_rectangle(self._x0, 3 * sc, self._x1, 19 * sc, fill="#2a2d2e", width=0)
        self._bar = self.create_rectangle(self._x0, 3 * sc, self._x0, 19 * sc,
                                          fill="#2980B9", width=0)
        self._mark = self.create_line(0, 0, 0, 22 * sc, fill="white", width=max(2, int(2 * sc)))
        for t in self.TICKS:
            x = self._x0 + (self._x1 - self._x0) * t / 100
            self.create_text(min(max(x, 10 * sc), w - 12 * sc), 27 * sc, text=f"{t}%",
                             fill="gray60", font=("Segoe UI", -int(10 * sc)))
        self._paint()

    def update_score(self, prob, threshold, speaking):
        """Show a new score; it rises at once and falls back gently."""
        self._shown = max(prob, self._shown * 0.8)
        self._threshold, self._speaking = threshold, speaking
        self._paint()

    def _paint(self):
        """Move the bar and the threshold line to the newest values (cheap:
        only changes coordinates, nothing is redrawn from scratch)."""
        if not hasattr(self, "_bar"):
            return
        sc, span = self._s, self._x1 - self._x0
        self.coords(self._bar, self._x0, 3 * sc, self._x0 + span * self._shown, 19 * sc)
        self.itemconfigure(self._bar, fill="#2ECC71" if self._speaking else "#2980B9")
        x = self._x0 + span * self._threshold
        self.coords(self._mark, x, 0, x, 22 * sc)


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


# ─────────────────────────────────────────────────────────────────────────────
# Main application
#
# InterpreterApp is the window and the coordinator of everything above.
# Methods are grouped by purpose: window placement, models, building the
# controls, the level meter, mode switching, starting/stopping listening,
# text-to-speech, settings changes, handling results (_poll_queue),
# diagnostics, the watchdog and logging.
# ─────────────────────────────────────────────────────────────────────────────


class InterpreterApp(ctk.CTk):

    """
    The classic desktop view and UI adapter for the shared session controller.
    
    This class implements the main user interface for the Voice Interpreter.
    It builds all the controls, validates model state, turns current widget values
    into a SessionConfig, and hands worker lifecycle to SessionController.
    
    Background threads never touch the window directly: they put messages on
    `result_queue`, and _poll_queue() reads them on the UI thread.
    
    This is the traditional interface that provides all the functionality of
    the application with a simple, easy-to-use layout.
    """
    def __init__(self):
        """
        Create the window: load settings, find audio devices, build all the
        controls, place the window on screen, start loading the AI models and
        start listening.
        
        This method initializes the main application window and sets up all
        components including audio device detection, UI controls, and background
        model loading.
        """
        super().__init__()
        self.title("Voice Interpreter")
        self.geometry("980x720")
        self.minsize(760, 480)

        # session_id: random id for this run of the app (appears in the logs).
        # result_queue: messages from background threads to the window.
        # phrase_queue: finished phrases from capture to the worker (replaced
        # for every listening session). _perf_stats: figures for Diagnostics.
        self.session_id = str(uuid.uuid4())
        self.result_queue = queue.Queue()
        from session_controller import SessionController
        self.session_controller = SessionController(self.result_queue, sys.modules[__name__])
        self.phrase_queue = self.session_controller.phrase_queue
        self._perf_stats  = {
            "transcriptions":  0,
            "translations":    0,
            "transcription_ms": deque(maxlen=200),
            "translation_ms":   deque(maxlen=200),
            "lock_wait_ms":     deque(maxlen=200),
        }
        # The current listening session's threads (None / empty when stopped).
        self._capture_thread         = None
        self._loopback_out        = None
        self._workers             = []
        self._translation_workers = []
        self._translation_queue   = self.session_controller.translation_queue
        # _model_loader: thread loading the main model; _whisper_model: the model
        # once loaded; is_listening: a session is running; _muted: user pressed
        # Mute; _run_id: session counter; _last_poll_t / _psutil_process: for the
        # diagnostics; _preview_loader / _preview_enabled: the preview model.
        self._model_loader        = None
        self._whisper_model       = None
        self.is_listening         = False
        self._muted               = False
        self._run_id              = 0
        self._traces              = {}   # seq -> trace awaiting online translation
        self._first_preview       = {}   # seq -> ms from speech start to first preview
        self._ui_stats            = PerfWindow()
        self._last_poll_t         = None
        self._psutil_process                = None
        self._preview_loader      = None
        self._preview_enabled     = False
        self._previews_off        = False   # Previews set to "Off"
        self._mt_loaders          = {}   # (src, tgt) -> TranslatorLoader
        self._initialised         = False
        self._active_mode         = None
        self._start_retry_after_id = None
        # ── TTS ───────────────────────────────────────────────────────────
        self._tts_on   = False
        self.speaker   = Speaker()
        self.speaker.start()
        # Live preview state: per text box, where the grey preview line starts
        # and which phrase it belongs to.
        self._live       = {}
        self._last_preview_words = {}   # box -> (phrase seq, words of its last preview),
                                        # for showing the stable part of previews
        self._mute_evt   = None
        self._partial_slot = None

        # ── Audio devices ─────────────────────────────────────────────────
        self._input_devices, self._output_devices = get_audio_devices()
        # _input_devices / _output_devices: lists of (device number, label);
        # _input_map / _output_map: label -> device number, to look labels up.
        self._input_map  = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        input_labels     = [lbl for _, lbl in self._input_devices]
        output_labels    = [lbl for _, lbl in self._output_devices]

        nvidia_lbl  = find_device_by_keywords(self._input_devices,  NVIDIA_KEYWORDS)
        vb_lbl      = find_best_vb_input(self._input_devices)       # prefers CABLE-A
        friend_lbl  = find_best_friend_input(self._input_devices)   # FiiO loopback > VB cable
        fiio_lbl    = find_best_output_device(self._output_devices)  # K7 > Q series etc.
        default_out = output_labels[0] if output_labels else ""

        # _mode_state: for each mode (0 = My Voice, 1 = Friend):
        # [device label, source language, target language].
        self._mode_state = [
            [nvidia_lbl or (input_labels[0] if input_labels else ""), "English", "Russian"],
            [friend_lbl or (input_labels[0] if input_labels else ""), "Russian", "English"],
        ]
        saved_devices = load_settings().get("mode_devices", [])
        for i, lbl in enumerate(saved_devices[:2]):
            if lbl in self._input_map:
                self._mode_state[i][0] = lbl

        # *_var: Tk variables linked to the controls; reading .get() gives what
        # the control currently shows, .set() changes it. device_var: input
        # device; out_device_var: output device; loopback_var: replay checkbox;
        # model_var, sensitivity_var, src_var, tgt_var: the other menus/slider.
        self.device_var     = ctk.StringVar(value=self._mode_state[0][0])
        self.out_device_var = ctk.StringVar(value=fiio_lbl or default_out)
        # Replaying audio is only for VB-Audio cables; with a [Loopback]
        # input the audio is already playing on the device.
        self.loopback_var   = ctk.BooleanVar(
            value=bool(vb_lbl) and "[loopback]" not in (friend_lbl or "").lower())
        saved_model = load_settings().get("model")
        if isinstance(saved_model, str):
            saved_model = saved_model.replace(chr(0x2014), "-")
        self.model_var      = ctk.StringVar(
            value=saved_model if saved_model in MODELS else _DEFAULT_MODEL)
        self.sensitivity_var = ctk.DoubleVar(
            value=float(load_settings().get("sensitivity", DEFAULT_SENS)))
        self._sens_log_after = None
        self.src_var        = ctk.StringVar(value="English")
        self.tgt_var        = ctk.StringVar(value="Russian")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self._build_header()
        self._build_quick_switch()
        self._build_controls(input_labels, output_labels)
        self._build_meter()
        self._build_text_area()
        self._build_status_bar()

        # Start in the mode used last time (Friend, for most calls)
        self._apply_mode(int(load_settings().get("mode", 0)) % 2, switch_source=True)
        self._fit_to_screen()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        log_event(logging.INFO, "app_initialized",
                  session_id=self.session_id,
                  whisper_device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE,
                  num_workers=NUM_WORKERS,
                  input_device_count=len(input_labels),
                  output_device_count=len(output_labels))
        self._preload_model()
        self._ensure_preview_model()
        self._initialised = True
        self._start_session()
        self.after(15,   self._poll_queue)
        self.after(WATCHDOG_SEC * 1000, self._watchdog)
        self.after(STATS_INTERVAL_SEC * 1000, self._log_resources)
        self.after(1000, self._refresh_diagnostics)

    # ── Window placement ──────────────────────────────────────────────────

    WANT_W, WANT_H = 1000, 760   # logical px; shrunk to fit smaller screens

    @staticmethod
    def _work_area():
        """Usable screen area in physical pixels (excludes the taskbar,
        wherever it is docked): (left, top, right, bottom)."""
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            r = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(r), 0):  # SPI_GETWORKAREA
                return r.left, r.top, r.right, r.bottom
        return None

    def _fit_to_screen(self):
        """Size the window to fit the usable screen area and centre it there.
        Windows' default placement put the 800 px-tall window (2000 physical
        px at 250 % scaling) partly below the screen on a 4K monitor."""
        self.update_idletasks()
        from customtkinter import ScalingTracker
        scale = ScalingTracker.get_window_scaling(self)
        area = self._work_area()
        if area is None:
            screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
            area = (0, 0, int(screen_w * scale), int(screen_h * scale))
        left, top, right, bottom = area
        # Title bar + borders, which geometry() sizes exclude (~40 px at 100 %)
        chrome = int(40 * scale)
        avail_w = (right - left) / scale
        avail_h = (bottom - top - chrome) / scale
        need_h = self.winfo_reqheight() / scale    # what the layout asks for
        win_w = int(min(self.WANT_W, avail_w * 0.96))
        win_h = int(min(max(self.WANT_H, need_h), avail_h * 0.96))
        win_x = left + int(((right - left) - win_w * scale) / 2)
        win_y = top + int(((bottom - top) - (win_h * scale + chrome)) / 2)
        self.minsize(min(760, win_w), min(480, win_h))
        self.geometry(f"{win_w}x{win_h}+{max(left, win_x)}+{max(top, win_y)}")
        log_event(logging.INFO, "window_placed", scaling=scale, work_area=list(area),
                  size_logical=[win_w, win_h], layout_needs_h=round(need_h), position=[win_x, win_y])

    # ── Model ─────────────────────────────────────────────────────────────

    def _preload_model(self):
        """Start loading the selected Whisper model in the background."""
        self._whisper_model = None
        model_size = MODELS[self.model_var.get()]
        log_event(logging.INFO, "whisper_model_preload_started", model_size=model_size)
        self._model_loader = WhisperModelLoader(
            model_size=model_size,
            status_cb=lambda m: self.result_queue.put(("status", m)),
        )
        self._model_loader.start()

    def _reload_model_if_changed(self):
        """Called when the Model menu changes: save the choice, load the new model
        and restart listening.
        """
        log_event(logging.INFO, "whisper_model_reload_requested", selected_model=self.model_var.get())
        save_settings(model=self.model_var.get())
        if MODELS[self.model_var.get()] == "tiny" and self.src_var.get() != "English":
            self.after(1500, lambda: self.status_var.set(
                "Note: Tiny makes many mistakes in non-English speech; use Turbo or Medium."))
        if self.is_listening:
            self._stop_session(keep_mode=True)
        self._whisper_model = None
        self._preload_model()
        self._ensure_preview_model()
        self._restart_session()

    # ── Header ────────────────────────────────────────────────────────────

    def _build_header(self):
        """Build the title bar: app name, GPU/CPU badge and the LISTENING/MUTED badge."""
        header = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray85", "gray17"))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.grid(row=0, column=0, pady=(8, 8))

        ctk.CTkLabel(title_row, text="🎙  Voice Interpreter",
                     font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0)

        hw_text  = " GPU " if WHISPER_DEVICE == "cuda" else " CPU "
        hw_color = COL_OK  if WHISPER_DEVICE == "cuda" else COL_INFO
        ctk.CTkLabel(title_row, text=hw_text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     fg_color=hw_color, corner_radius=4,
                     text_color="white").grid(row=0, column=1, padx=(10, 0))

        self.live_status_badge = ctk.CTkLabel(
            title_row, text=" IDLE ",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=COL_WARN, corner_radius=4, text_color="white")
        self.live_status_badge.grid(row=0, column=2, padx=(10, 0))

        ctk.CTkLabel(title_row,
                     text="   Whisper AI  ·  real-time translation",
                     font=ctk.CTkFont(size=12),
                     text_color=("gray50", "gray60")).grid(row=0, column=3, padx=(6, 0))

    # ── Quick switch ──────────────────────────────────────────────────────

    def _build_quick_switch(self):
        """Build the mode buttons (My Voice / Friend) and Mute / Speak / Clear."""
        # A frame is a container for other widgets. 'grid' places widgets in
        # rows and columns; weight=1 lets a column stretch when the window widens.
        quick_switch_frame = ctk.CTkFrame(self, fg_color=("gray88", "gray20"), corner_radius=10)
        quick_switch_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 0))
        quick_switch_frame.grid_columnconfigure(0, weight=1)
        quick_switch_frame.grid_columnconfigure(1, weight=1)
        self._mode_btns = []
        # The two mode buttons, side by side. enumerate() numbers them 0 and 1.
        # `lambda i=col: …` makes each button call _on_mode_btn with its own
        # number (i=col captures the value now; without it both would get 1).
        for col, (label, emoji, src, tgt) in enumerate([
            ("My Voice", "🎤", "English", "Russian"),
            ("Friend",   "🔊", "Russian", "English"),
        ]):
            mode_button = ctk.CTkButton(
                quick_switch_frame, text=f"{emoji}  {label}   {src} → {tgt}",
                font=ctk.CTkFont(size=14, weight="bold"),
                height=44, corner_radius=8,
                fg_color=COL_INACTIVE, hover_color=COL_IN_HOVER,
                command=lambda i=col: self._on_mode_btn(i))
            mode_button.grid(row=0, column=col, sticky="ew",
                     padx=(10, 5) if col == 0 else (5, 5), pady=10)
            self._mode_btns.append(mode_button)

        # Mute, Speak and Clear: `command=` is the method called when clicked.
        # Action buttons share the row (they used to take a row of their own)
        action_buttons = ctk.CTkFrame(quick_switch_frame, fg_color="transparent")
        action_buttons.grid(row=0, column=2, padx=(5, 10), pady=10)
        self.mute_btn = ctk.CTkButton(
            action_buttons, text="🔇  Mute", width=110, height=44,
            command=self._toggle_mute)
        self.mute_btn.grid(row=0, column=0, padx=(0, 6))
        self.tts_btn = ctk.CTkButton(
            action_buttons, text="🔇  Speak: Off", width=125, height=44,
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
            text_color=("gray10", "gray90"),
            command=self._toggle_tts)
        self.tts_btn.grid(row=0, column=1, padx=(0, 6))
        ctk.CTkButton(action_buttons, text="Clear", width=64, height=44,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._clear).grid(row=0, column=2)

    # ── Controls ──────────────────────────────────────────────────────────

    def _build_controls(self, input_labels, output_labels):
        """Build the settings rows: audio input, audio output, languages, model,
        speech sensitivity and preview model.
        """
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 0))
        outer.grid_columnconfigure(1, weight=1)

        # Audio input
        ctk.CTkLabel(outer, text="Audio input:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=0, column=0, sticky="w", pady=2)
        input_row = ctk.CTkFrame(outer, fg_color="transparent")
        input_row.grid(row=0, column=1, sticky="w")
        self.device_menu = ctk.CTkOptionMenu(
            input_row, variable=self.device_var, values=input_labels, width=320,
            command=self._on_manual_device_change)
        self.device_menu.grid(row=0, column=0)
        ctk.CTkButton(input_row, text="⟳", width=34,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._refresh_devices).grid(row=0, column=1, padx=(8, 0))
        ctk.CTkLabel(input_row, text="🔊 = sound playing on an output device (e.g. FiiO in a call)",
                     font=ctk.CTkFont(size=11), text_color=("gray45", "gray60")
                     ).grid(row=0, column=2, padx=(10, 0))

        # Audio output + loopback toggle
        ctk.CTkLabel(outer, text="Audio output:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=1, column=0, sticky="w", pady=2)
        output_row = ctk.CTkFrame(outer, fg_color="transparent")
        output_row.grid(row=1, column=1, sticky="w")
        self.out_device_menu = ctk.CTkOptionMenu(
            output_row, variable=self.out_device_var, values=output_labels, width=260,
            command=self._on_manual_device_change)
        self.out_device_menu.grid(row=0, column=0)
        self.loopback_chk = ctk.CTkCheckBox(
            output_row, text="Hear input through output  (VB-Audio cables only)",
            variable=self.loopback_var, font=ctk.CTkFont(size=12),
            command=self._on_manual_device_change)
        self.loopback_chk.grid(row=0, column=1, padx=(16, 0))

        # Translate
        ctk.CTkLabel(outer, text="Translate:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=2, column=0, sticky="w", pady=2)
        language_row = ctk.CTkFrame(outer, fg_color="transparent")
        language_row.grid(row=2, column=1, sticky="w")
        lang_names = sorted(TARGET_LANGUAGES)
        source_names = ["Auto-detect"] + lang_names   # the language of each phrase is detected
        self.src_menu = ctk.CTkOptionMenu(
            language_row, variable=self.src_var, values=source_names, width=155,
            command=self._on_manual_lang_change)
        self.src_menu.grid(row=0, column=0)
        ctk.CTkLabel(language_row, text="  →  ",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=1)
        self.tgt_menu = ctk.CTkOptionMenu(
            language_row, variable=self.tgt_var, values=lang_names, width=155,
            command=self._on_manual_lang_change)
        self.tgt_menu.grid(row=0, column=2)

        # Model + sensitivity
        ctk.CTkLabel(outer, text="Model:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=3, column=0, sticky="w", pady=(6, 2))
        model_row = ctk.CTkFrame(outer, fg_color="transparent")
        model_row.grid(row=3, column=1, sticky="ew", pady=(6, 2))
        self.model_menu = ctk.CTkOptionMenu(
            model_row, variable=self.model_var, values=list(MODELS.keys()), width=230,
            command=lambda _: self._reload_model_if_changed())
        self.model_menu.grid(row=0, column=0, padx=(0, 20))
        ctk.CTkLabel(model_row, text="Speech sensitivity:",
                     font=ctk.CTkFont(size=13)).grid(row=0, column=1)
        self.sens_slider = ctk.CTkSlider(
            model_row, from_=0.0, to=1.0, variable=self.sensitivity_var,
            width=160, command=self._on_sensitivity)
        self.sens_slider.grid(row=0, column=2, padx=(8, 4))
        self.sens_label = ctk.CTkLabel(
            model_row, text=f"{round(self.sensitivity_var.get() * 100)}%",
            font=ctk.CTkFont(size=12), width=40)
        self.sens_label.grid(row=0, column=3)
        saved = load_settings()
        choice = saved.get("preview_model")
        if choice not in PREVIEW_CHOICES:
            # Older settings had an on/off Tiny checkbox. "Off" meant previews
            # from the main model, which kept the GPU 88-100 % busy with Turbo
            # and delayed finished phrases, so it now maps to Small.
            choice = "Tiny (instant)" if saved.get("fast_preview", False) else "Small (balanced)"
        self.preview_choice_var = ctk.StringVar(value=choice)
        ctk.CTkLabel(model_row, text="Previews:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=4, padx=(16, 4))
        ctk.CTkOptionMenu(model_row, variable=self.preview_choice_var, values=list(PREVIEW_CHOICES),
                          width=150, command=lambda _: self._on_preview_choice()).grid(
            row=0, column=5)

    # ── Input-level meter ─────────────────────────────────────────────────

    def _build_meter(self):
        """Build the input-level meter row (DbMeter plus the dBFS readout)."""
        meter_frame = ctk.CTkFrame(self, fg_color="transparent")
        meter_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(6, 0))
        meter_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(meter_frame, text="Input level:",
                     font=ctk.CTkFont(size=12),
                     text_color=("gray50", "gray55"),
                     width=95, anchor="w").grid(row=0, column=0, sticky="w")
        self.meter = DbMeter(meter_frame)
        self.meter.grid(row=0, column=1, sticky="ew", padx=(4, 8))
        self.level_text = ctk.CTkLabel(meter_frame, text="-",
                                       font=ctk.CTkFont(family="Consolas", size=11),
                                       text_color=("gray50", "gray55"),
                                       width=190, anchor="w")
        self.level_text.grid(row=0, column=2)
        # Second row: how much the sound resembles speech, against the
        # threshold set by Speech sensitivity (see SpeechMeter).
        ctk.CTkLabel(meter_frame, text="Speech:", font=ctk.CTkFont(size=12),
                     text_color=("gray50", "gray55"), width=95, anchor="w").grid(
            row=1, column=0, sticky="w", pady=(2, 0))
        self.speech_meter = SpeechMeter(meter_frame)
        self.speech_meter.grid(row=1, column=1, sticky="ew", padx=(4, 8), pady=(2, 0))
        self.speech_text = ctk.CTkLabel(meter_frame, text="-",
                                        font=ctk.CTkFont(family="Consolas", size=11),
                                        text_color=("gray50", "gray55"), width=190, anchor="w")
        self.speech_text.grid(row=1, column=2, pady=(2, 0))

    # ── Main tabbed area ──────────────────────────────────────────────────

    def _build_text_area(self):
        """Build the tabs: "Interpreter" (heard text + translation side by side)
        and "Diagnostics" (live health and performance figures).
        """
        tabs = ctk.CTkTabview(self, fg_color="transparent")
        tabs.grid(row=4, column=0, sticky="nsew", padx=20, pady=(2, 0))
        tabs.add("Interpreter")
        tabs.add("Diagnostics")
        self.main_tabs = tabs

        # ─── Interpreter tab ──────────────────────────────────────────────
        interp_tab = tabs.tab("Interpreter")
        interp_tab.grid_columnconfigure(0, weight=1)
        interp_tab.grid_columnconfigure(1, weight=1)
        interp_tab.grid_rowconfigure(1, weight=1)

        self.heard_label = ctk.CTkLabel(interp_tab, text="Heard in English",
                                        font=ctk.CTkFont(size=12, weight="bold"))
        self.heard_label.grid(row=0, column=0, sticky="w", padx=4, pady=(4, 4))
        self.translated_label = ctk.CTkLabel(interp_tab, text="Translation → Russian",
                                             font=ctk.CTkFont(size=12, weight="bold"))
        self.translated_label.grid(row=0, column=1, sticky="w", padx=(12, 4), pady=(4, 4))
        self._text_size = int(load_settings().get("text_size", 14))
        self.heard_box = ctk.CTkTextbox(interp_tab, wrap="word", height=120,
                                        font=ctk.CTkFont(size=self._text_size))
        self.heard_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.translated_box = ctk.CTkTextbox(interp_tab, wrap="word", height=120,
                                             font=ctk.CTkFont(size=self._text_size))
        self.translated_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        # ─── Diagnostics tab ──────────────────────────────────────────────
        diag_tab = tabs.tab("Diagnostics")
        diag_tab.grid_columnconfigure(0, weight=1)
        diag_tab.grid_rowconfigure(2, weight=1)   # textbox row gets all spare space

        # Health summary (fixed-height, multi-line label replacement)
        self.diag_summary_box = ctk.CTkTextbox(
            diag_tab, height=90, wrap="word",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("gray92", "gray18"), activate_scrollbars=False)
        self.diag_summary_box.grid(row=0, column=0, sticky="ew", padx=2, pady=(4, 6))
        self.diag_summary_box.configure(state="disabled")

        # Shortcut buttons
        btn_frame = ctk.CTkFrame(diag_tab, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for col, (lbl, cmd) in enumerate([
            ("Open App Folder", self._open_app_folder),
            ("Open Main Log",   lambda: self._open_path(LOG_PATH)),
            ("Open Error Log",  lambda: self._open_path(ERROR_LOG_PATH)),
            ("Open Summary",    lambda: self._open_path(LOG_SUMMARY_PATH)),
        ]):
            ctk.CTkButton(btn_frame, text=lbl, width=130,
                          fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                          text_color=("gray10", "gray90"),
                          command=cmd).grid(row=0, column=col, padx=(0, 8))

        # Scrollable detail box
        self.diag_box = ctk.CTkTextbox(diag_tab, wrap="word", height=80,
                                       font=ctk.CTkFont(size=12))
        self.diag_box.grid(row=2, column=0, sticky="nsew")

    # ── Status bar ────────────────────────────────────────────────────────

    def _build_status_bar(self):
        """Build the bottom bar: status message and the Export Log Summary button."""
        status_bar = ctk.CTkFrame(self, fg_color="transparent")
        status_bar.grid(row=5, column=0, sticky="ew", padx=22, pady=(2, 8))
        status_bar.grid_columnconfigure(0, weight=1)
        self.status_var = ctk.StringVar(value="Loading Whisper model…")
        ctk.CTkLabel(status_bar, textvariable=self.status_var,
                     font=ctk.CTkFont(size=11),
                     text_color=("gray45", "gray55"), anchor="w").grid(
            row=0, column=0, sticky="ew")
        # Text size (caption guidance: readers must control the size)
        for col, (label, step) in enumerate((("A−", -1), ("A+", 1)), start=1):
            ctk.CTkButton(status_bar, text=label, width=40,
                          fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                          text_color=("gray10", "gray90"),
                          command=lambda d=step: self._change_text_size(d)).grid(
                row=0, column=col, padx=(6, 0))
        ctk.CTkButton(status_bar, text="Export Log Summary", width=150,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._export_log_summary).grid(row=0, column=3, padx=(10, 0))

    # ── Level meter ───────────────────────────────────────────────────────

    def _on_level(self, rms_db: float, peak_db: float, speaking: bool,
                  prob: float = 0.0, threshold: float = 0.5) -> None:
        """Called by the capture thread with each block's level; forwards it to the
        main thread through result_queue (threads may not touch widgets).
        """
        self.result_queue.put(("level", (rms_db, peak_db, speaking, prob, threshold)))

    def _update_meter(self, rms_db, peak_db, speaking, prob=0.0, threshold=0.5):
        """Show a level on the meter and the dBFS readout, and the speech score
        against the threshold on the speech meter (main thread)."""
        self.meter.update_level(rms_db, peak_db)
        self.speech_meter.update_score(prob, threshold, speaking)
        self.level_text.configure(text="  -∞ dBFS  (silent)" if rms_db <= -99
                                  else f"{rms_db:5.1f} dBFS")
        verdict = "SPEECH" if speaking else "no speech"
        self.speech_text.configure(text=f"{prob * 100:3.0f}% / needs {threshold * 100:.0f}%  {verdict}",
                                   text_color=COL_OK if speaking else ("gray50", "gray55"))

    # ── Mode switching ────────────────────────────────────────────────────

    def _on_mode_btn(self, mode_idx: int) -> None:
        """A mode button was clicked: remember the current mode's settings, switch."""
        if self._active_mode is not None:
            self._mode_state[self._active_mode] = [
                self.device_var.get(), self.src_var.get(), self.tgt_var.get()]
        self._apply_mode(mode_idx, switch_source=True)

    def _apply_mode(self, mode_idx, switch_source=False):
        """Switch to mode 0 (My Voice) or 1 (Friend): set its device and languages,
        highlight its button and restart listening.
        """
        if self.is_listening:
            self._stop_session(keep_mode=True)
        self._active_mode = mode_idx
        save_settings(mode=mode_idx)   # remembered for the next start
        device_lbl, src_lang, tgt_lang = self._mode_state[mode_idx]
        if switch_source and device_lbl:
            self.device_var.set(device_lbl)
        self.src_var.set(src_lang)
        self.tgt_var.set(tgt_lang)
        self.heard_label.configure(text=f"Heard in {src_lang}")
        self.translated_label.configure(text=f"Translation → {tgt_lang}")
        colours = [(COL_MY_VOICE, COL_MY_HOVER), (COL_FRIEND, COL_FR_HOVER)]
        for i, btn in enumerate(self._mode_btns):
            if i == mode_idx:
                btn.configure(fg_color=colours[i][0], hover_color=colours[i][1])
            else:
                btn.configure(fg_color=COL_INACTIVE, hover_color=COL_IN_HOVER)
        if self._initialised:
            self._restart_session()

    def _on_manual_device_change(self, _: object | None = None) -> None:
        """An audio device menu changed: restart listening with the new device."""
        self._restart_session()

    def _on_manual_lang_change(self, _=None):
        """A language menu changed: update the box titles and restart listening."""
        self.heard_label.configure(text=f"Heard in {self.src_var.get()}")
        self.translated_label.configure(text=f"Translation → {self.tgt_var.get()}")
        self._restart_session()

    def _refresh_devices(self):
        """The ⟳ button: re-scan audio devices (e.g. after plugging one in)."""
        if self.is_listening:
            self._stop_session(keep_mode=True)
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map  = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        in_labels  = [lbl for _, lbl in self._input_devices]
        out_labels = [lbl for _, lbl in self._output_devices]
        self.device_menu.configure(values=in_labels)
        self.out_device_menu.configure(values=out_labels)
        if self.device_var.get() not in in_labels:
            self.device_var.set(in_labels[0] if in_labels else "")
        if self.out_device_var.get() not in out_labels:
            self.out_device_var.set(out_labels[0] if out_labels else "")
        nvidia_lbl = find_device_by_keywords(self._input_devices, NVIDIA_KEYWORDS)
        friend_lbl = find_best_friend_input(self._input_devices)
        if self._mode_state[0][0] not in self._input_map and nvidia_lbl:
            self._mode_state[0][0] = nvidia_lbl
        if self._mode_state[1][0] not in self._input_map and friend_lbl:
            self._mode_state[1][0] = friend_lbl
        self.status_var.set(f"Refreshed - {len(in_labels)} in / {len(out_labels)} out device(s).")
        log_event(logging.INFO, "audio_devices_refreshed",
                  input_count=len(in_labels), output_count=len(out_labels))
        self._refresh_diagnostics_once()
        self._restart_session()

    # ── Session start / stop ──────────────────────────────────────────────
    # The app listens continuously; settings changes restart the session and
    # Mute is the only way to stop it.

    def _restart_session(self):
        """Stop listening and start again with the current settings (unless muted)."""
        if self.is_listening:
            self._stop_session(keep_mode=True)
        if not self._muted:
            self._start_session()

    def _toggle_mute(self):
        """The Mute button: stop listening (and release the device), or resume."""
        self._muted = not self._muted
        if self._muted:
            self._stop_session()
            self.mute_btn.configure(text="🎙  Unmute", fg_color=COL_STOP,
                                    hover_color=COL_STOP_HOV)
            self._set_live_status("MUTED", COL_ERR)
            self.status_var.set("Muted - not listening.")
        else:
            self.mute_btn.configure(text="🔇  Mute", fg_color=["#3B8ED0", "#1F6AA5"],
                                    hover_color=["#3281BC", "#1A5C8F"])
            self._start_session()
        log_event(logging.INFO, "mute_toggled", muted=self._muted)

    def _sync_session_controller(self):
        """Expose controller-owned runtime references to existing UI diagnostics."""
        controller = self.session_controller
        self.phrase_queue = controller.phrase_queue
        self._translation_queue = controller.translation_queue
        self._capture_thread = controller.capture_thread
        self._loopback_out = controller.loopback_out
        self._workers = controller.workers
        self._translation_workers = controller.translation_workers
        self._partial_slot = controller.partial_slot
        self._early_slot = controller.early_slot
        self._stopping_capture = controller.stopping_capture

    def _start_session(self):
        """Validate the selected model and ask the shared controller to start it."""
        if self.is_listening or self._muted:
            return
        if self._model_loader and not self._model_loader.ready.is_set():
            self.status_var.set("⏳ Loading Whisper model - please wait…")
            if self._start_retry_after_id is None:
                self._start_retry_after_id = self.after(600, self._retry_start_session)
            return
        self._start_retry_after_id = None
        self._run_id += 1
        self._traces, self._first_preview = {}, {}
        if self._whisper_model is None and self._model_loader:
            if self._model_loader.error:
                self.status_var.set(f"Cannot start: {self._model_loader.error}")
                self._set_live_status("MODEL ERROR", COL_ERR)
                log_event(logging.ERROR, "session_start_blocked_model_error",
                          error=self._model_loader.error)
                return
            self._whisper_model = self._model_loader.model
            if getattr(self._model_loader, "used_fallback", False):
                self._set_live_status("CPU FALLBACK", COL_WARN)
                self.model_var.set(self._model_loader.default_model_key)
        if self._whisper_model is None:
            self.status_var.set("Whisper model unavailable.")
            log_event(logging.ERROR, "session_start_blocked_model_unavailable")
            return

        device_label = self.device_var.get()
        input_index = self._input_map.get(device_label, -1)
        input_index = None if input_index == -1 else input_index
        log_event(logging.INFO, "audio_device_selected",
                  device_label=device_label, device_index=input_index)
        if self._active_mode is not None:
            self._mode_state[self._active_mode][0] = device_label
        save_settings(mode_devices=[mode[0] for mode in self._mode_state])

        output_label = self.out_device_var.get()
        output_index = self._output_map.get(output_label, -1)
        output_index = None if output_index == -1 else output_index
        if self._tts_on:
            self._configure_speaker()
        self._live = {}
        self.is_listening = True
        self._set_live_status("LISTENING", COL_OK)
        self.status_var.set("Listening…")

        from session_controller import SessionConfig
        settings = load_settings()
        config = SessionConfig(
            session_id=self.session_id,
            run_id=self._run_id,
            device_label=device_label,
            device_index=input_index,
            output_label=output_label,
            output_index=output_index,
            loopback=bool(self.loopback_var.get()),
            src_key=self.src_var.get(),
            tgt_key=self.tgt_var.get(),
            model_label=self.model_var.get(),
            sensitivity=round(self.sensitivity_var.get(), 2),
            preview_choice=self.preview_choice_var.get(),
            preview_enabled=self._preview_enabled,
            previews_off=self._previews_off,
            tts_on=self._tts_on,
            record_audio=bool(settings.get("record_audio", True)),
            auto_language=bool(settings.get("auto_language", True)),
            mode_state=tuple(tuple(mode) for mode in self._mode_state),
        )
        try:
            self.session_controller.start(
                config,
                model=self._whisper_model,
                model_loader=self._model_loader,
                speaker=self.speaker,
                on_level=self._on_level,
                preview_model=self._preview_model,
                translator_for=self._translator_for,
                music_scene=self._ensure_music_scene(),
            )
            self._sync_session_controller()
        except Exception as exc:
            self.is_listening = False
            self.session_controller.stop_workers()
            self._sync_session_controller()
            self._set_live_status("START FAILED", COL_ERR)
            self.status_var.set(f"Could not start listening: {exc}")
            log_exception("session_start_failed", exc, device_label=device_label,
                          model=config.model_label, run_id=config.run_id)

    def _retry_start_session(self):
        """Timer callback: try _start_session again (model was still loading)."""
        self._start_retry_after_id = None
        self._start_session()

    def _stop_session(self, keep_mode=False):
        """Stop the controller's workers, then reset UI state and indicators."""
        self.is_listening = False
        if self._start_retry_after_id is not None:
            self.after_cancel(self._start_retry_after_id)
            self._start_retry_after_id = None
        self.session_controller.stop_workers()
        self._sync_session_controller()
        self.meter.reset()
        self.level_text.configure(text="-")
        self._set_live_status("IDLE", COL_WARN)
        log_event(logging.INFO, "session_stopped",
                  session_id=self.session_id, keep_mode=keep_mode)
        if not keep_mode:
            self.status_var.set("Stopped.")
    # ── TTS - fixed: lock prevents "run loop already started" ────────────

    def _toggle_tts(self):
        """On: each new translation is spoken as it arrives, through the
        selected Audio output device. Off: speech stops immediately."""
        if self.speaker.error and not self.speaker.is_alive():
            self.status_var.set(f"Text-to-speech unavailable: {self.speaker.error}")
            return
        self._tts_on = not self._tts_on
        if self._tts_on:
            self._configure_speaker()
            self.tts_btn.configure(text="🔊  Speak: On", fg_color=COL_OK,
                                   hover_color=COL_MY_HOVER, text_color="white")
        else:
            self.speaker.stop_speech()
            self.tts_btn.configure(text="🔇  Speak: Off", fg_color=("gray75", "gray30"),
                                   hover_color=("gray65", "gray40"),
                                   text_color=("gray10", "gray90"))
        log_event(logging.INFO, "tts_toggled", enabled=self._tts_on)

    def _configure_speaker(self):
        """Point the text-to-speech voice at the chosen output device and language."""
        self.speaker.configure(self.out_device_var.get(), self.tgt_var.get())
        self.after(300, lambda: self.speaker.error and self.status_var.set(self.speaker.error))

    # ── Clear ─────────────────────────────────────────────────────────────

    def _clear(self):
        """The Clear button: empty both text boxes (listening carries on)."""
        self.heard_box.delete("1.0", "end")
        self.translated_box.delete("1.0", "end")
        self._live = {}
        self.status_var.set("Cleared.")
        log_event(logging.INFO, "ui_cleared")

    # ── Sensitivity ───────────────────────────────────────────────────────

    def _on_sensitivity(self, val):
        """The sensitivity slider moved: apply it at once, save and log it once the
        slider has stopped moving for 0.5 s (dragging fires many events).
        """
        v = round(float(val), 2)
        self.sens_label.configure(text=f"{round(v * 100)}%")
        if self._capture_thread:
            self._capture_thread.set_sensitivity(v)
        # Dragging fires dozens of events; log and save once it settles.
        if self._sens_log_after is not None:
            self.after_cancel(self._sens_log_after)
        def _settled():
            self._sens_log_after = None
            save_settings(sensitivity=v)
            log_event(logging.INFO, "sensitivity_changed", value=v, run_id=self._run_id,
                      session_id=self.session_id,
                      vad_threshold=round(vad_threshold_for(v), 2),
                      # what the detector was seeing when the change was made
                      speech_score_now=round(self.speech_meter._shown, 2))
        self._sens_log_after = self.after(500, _settled)

    def _on_preview_choice(self):
        """The Previews menu changed: save, load that model, restart listening."""
        save_settings(preview_model=self.preview_choice_var.get())
        self._ensure_preview_model()
        self._restart_session()

    def _ensure_music_scene(self):
        """The music detector (music.py), created once and kept across
        sessions so its models load only once. "music_mode": false in
        interpreter.settings.json turns it off; "vocal_separation": false
        keeps the detection but skips removing the instruments."""
        scene = getattr(self, "_music_scene", None)
        if scene is None:
            settings = load_settings()
            try:
                from music import MusicScene
                scene = MusicScene(enabled=bool(settings.get("music_mode", True)),
                                   separate=bool(settings.get("vocal_separation", True)),
                                   log=lambda event, **k: log_event(logging.INFO, event, **k))
            except Exception as e:
                log_exception("music_scene_unavailable", e)
                scene = None
            self._music_scene = scene
        return scene if scene is not None and scene.enabled else None

    def _ensure_preview_model(self):
        """Load the chosen preview model alongside the main model. Not needed
        when it is the same as the main model."""
        size = PREVIEW_CHOICES[self.preview_choice_var.get()]
        # Plain mirror of the Tk settings: _preview_model() runs on the worker
        # thread, and Tk variables may only be touched from the UI thread.
        self._previews_off = size == PREVIEW_OFF
        self._preview_enabled = (size not in (None, PREVIEW_OFF)
                                 and size != MODELS[self.model_var.get()])
        if not self._preview_enabled:
            return
        if self._preview_loader is None or self._preview_loader.model_size != size:
            self._preview_loader = WhisperModelLoader(model_size=size, status_cb=lambda m: None)
            self._preview_loader.start()
            log_event(logging.INFO, "preview_model_load_started", model_size=size)

    def _translator_for(
        self, src: str, tgt: str
    ) -> Callable[[], LocalTranslator | None] | None:
        """Start loading the Opus-MT model for this pair (first use downloads
        ~300 MB) and return a callable the worker polls; until it is ready,
        translation falls back to Whisper/online."""
        if opus_repo(src, tgt) is None:
            return None
        # Workers call this too (a phrase in a new language): one loader per pair
        with _MT_LOADERS_LOCK:
            loader = self._mt_loaders.get((src, tgt))
            if loader is None:
                loader = self._mt_loaders[(src, tgt)] = TranslatorLoader(src, tgt)
                loader.start()
                log_event(logging.INFO, "local_translator_load_started", pair=f"{src}-{tgt}",
                          repo=opus_repo(src, tgt))
        return lambda l=loader: l.translator if l.ready.is_set() else None

    def _preview_model(self):
        """Called from the transcription worker thread: no Tk access here."""
        if not self._preview_enabled:
            return None
        loader = self._preview_loader
        if loader is not None and loader.ready.is_set() and loader.model is not None:
            return loader.model
        return None   # still loading: previews use the main model meanwhile

    # ── Queue poll ────────────────────────────────────────────────────────

    def _log_displayed(self, seq: int, trace: dict[str, Any], online: bool) -> None:
        """Log how long a finished phrase took to reach the screen ("phrase_displayed")."""
        now = time.monotonic()
        log_event(logging.INFO, "phrase_displayed", session_id=self.session_id,
                  run_id=trace["run_id"], seq=seq,
                  ui_ms=round((now - trace["t_result"]) * 1000, 1),
                  # pause detected -> final text on screen
                  speech_end_to_text_ms=round((now - trace["t_end"]) * 1000, 1),
                  first_preview_ms=self._first_preview.pop(seq, None),
                  online_translation_pending=online)
        if online:
            self._traces[seq] = trace

    def _watchdog(self):
        """Restart listening if the capture thread died or gave up (device
        unplugged, driver reset) or a transcription worker exited, instead of
        showing LISTENING while nothing is heard. Backs off up to 30 s."""
        try:
            if self.is_listening and not self._muted:
                capture = self._capture_thread
                problem = None
                if capture is not None and not capture.is_alive():
                    problem = capture.failed or "capture thread stopped"
                elif any(not w.is_alive() for w in self._workers):
                    problem = "transcription worker stopped"
                if problem:
                    self._watchdog_failures = getattr(self, "_watchdog_failures", 0) + 1
                    wait = min(30, 2 ** self._watchdog_failures)
                    log_event(logging.WARNING, "watchdog_restart", problem=problem,
                              attempt=self._watchdog_failures, next_check_sec=wait)
                    self.status_var.set(f"⚠ {problem} - restarting listening…")
                    self._stop_session(keep_mode=True)
                    self.after(500, self._reconnect)
                    self.after(wait * 1000, self._watchdog)
                    return
                self._watchdog_failures = 0
        except Exception as e:
            log_exception("watchdog_failed", e)
        self.after(WATCHDOG_SEC * 1000, self._watchdog)

    def _reconnect(self):
        """Re-scan devices and resume on the *same* device once it is back
        (device indexes change when a USB DAC is re-plugged). Never silently
        switches to another device."""
        if self._muted or self.is_listening:
            return
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map  = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        self.device_menu.configure(values=[lbl for _, lbl in self._input_devices])
        if self.device_var.get() in self._input_map:
            log_event(logging.INFO, "watchdog_reconnected", device=self.device_var.get())
            self._start_session()
        else:
            self.status_var.set(f"⚠ Waiting for {plain_device_name(self.device_var.get())} "
                                f"to reconnect…")
            log_event(logging.WARNING, "watchdog_device_missing", device=self.device_var.get())
            # stay "listening" so the watchdog keeps retrying
            self.is_listening = True
            self._capture_thread = None
            self.after(3000, lambda: (setattr(self, "is_listening", False), self._reconnect()))

    def _log_resources(self):
        """Every STATS_INTERVAL_SEC: CPU, memory, GPU load and the UI thread's
        responsiveness, so slowdowns can be tied to resource limits. Also
        which of our threads use the CPU, how much is left for other apps
        (Teams, Telemost) and which apps use the most, and a resource_warning
        when the computer is running short."""
        # Measuring takes ~100 ms (it lists every process), so it runs on a
        # short-lived background thread instead of freezing the window.
        base = dict(listening=self.is_listening, run_id=self._run_id,
                    result_queue=self.result_queue.qsize(),
                    phrase_queue=self.phrase_queue.qsize(),
                    translation_queue=self._translation_queue.qsize(),
                    **self._ui_stats.summary())

        def measure():
            """Background thread: read CPU, memory and GPU figures (slow
            Windows calls), log them, and warn if anything is running low."""
            t0 = time.perf_counter()
            fields = collect_resource_stats(self)
            fields["probe_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._last_resources = fields     # for the Diagnostics page
            log_event(logging.INFO, "resource_stats", **base, **fields)
            warnings = resource_warnings(fields)
            if warnings:
                log_event(logging.WARNING, "resource_warning", problems=warnings,
                          **{k: fields.get(k) for k in ("system_cpu_pct", "mem_available_mb",
                                                         "gpu_mem_free_mb", "gpu_util_pct",
                                                         "top_other_apps")})
        threading.Thread(target=measure, name="resource-probe", daemon=True).start()
        # Run again in STATS_INTERVAL_SEC seconds.
        self.after(STATS_INTERVAL_SEC * 1000, self._log_resources)

    def _poll_queue(self):
        """Runs every 15 ms on the main thread: handle every message the background
        threads have sent (previews, finished phrases, translations, status,
        levels, metrics) and update the window. This is the only place results
        reach the screen.
        """
        # Measure how late this call is. It was scheduled 15 ms after the last
        # one; if the window was busy (e.g. redrawing), it runs later, and that
        # extra delay is recorded as poll_lag_ms for the diagnostics.
        now = time.monotonic()
        if self._last_poll_t is not None:
            # >0 means the Tk thread was busy and results waited on screen
            self._ui_stats.add("poll_lag_ms", (now - self._last_poll_t) * 1000 - 15)
        self._last_poll_t = now
        # Take every message waiting in result_queue. get_nowait() returns
        # immediately; when the queue is empty it raises queue.Empty, which ends
        # the loop (see the 'except queue.Empty' at the bottom).
        # Each message is a pair: (kind, data). 'kind' says what it is.
        handled = 0
        try:
            while True:
                kind, data = self.result_queue.get_nowait()
                handled += 1
                # ── 'partial': a live preview of a phrase that is still being spoken.
                #    Shown in grey; replaced each time a newer preview arrives.
                #    seq = the phrase's number, t_start = when its speech began.
                if kind == "partial":
                    seq, heard, translated, t_start, t_ready = data
                    self._set_live(self.heard_box, seq, heard)
                    if translated:
                        self._set_live(self.translated_box, seq, translated)
                    now = time.monotonic()
                    # Record how long the preview waited to be drawn, and (once per phrase)
                    # how long after the speech started the first preview appeared.
                    self._ui_stats.add("preview_ui_ms", (now - t_ready) * 1000)
                    if seq not in self._first_preview and t_start:
                        self._first_preview[seq] = round((now - t_start) * 1000, 1)
                # ── 'final': the finished, accurate text of a phrase.
                #    It replaces the grey preview. `translated` is the translation if the
                #    worker already made one (Opus-MT or Whisper); None means it still
                #    has to be translated online, so a 'translate' job is queued.
                #    An empty text means the phrase turned out to be silence/noise: the
                #    preview is simply removed.
                elif kind == "final":
                    seq, text, translated, trace = data
                    self._commit(self.heard_box, seq, text)
                    if trace and text:
                        self._log_displayed(seq, trace, online=translated is None)
                    if translated is not None:
                        self._commit_translation(seq, translated)
                    elif text:
                        self.result_queue.put(("translate", (seq, text, self.src_var.get(),
                                                             self.tgt_var.get())))
                    else:
                        self._commit(self.translated_box, seq, "")
                # ── 'translate': hand a phrase to the online TranslationWorker thread.
                #    If its queue is full (online service far too slow), the phrase is
                #    dropped and a warning logged rather than freezing the window.
                elif kind == "translate":
                    seq, text, src_lang_key, tgt_lang_key = data
                    try:
                        self._translation_queue.put((seq, text, src_lang_key, tgt_lang_key), timeout=0.1)
                        backlog = self._translation_queue.qsize()
                        log_event(logging.INFO, "translation_queued",
                                  input_chars=len(text), src_lang=src_lang_key, tgt_lang=tgt_lang_key,
                                  queue_backlog=backlog)
                        if backlog >= TRANSLATION_QUEUE_BACKLOG_WARN:
                            log_event(logging.WARNING, "translation_queue_backlog_high",
                                      backlog=backlog)
                    except queue.Full:
                        log_event(logging.WARNING, "translation_dropped_queue_full",
                                  input_chars=len(text), src_lang=src_lang_key, tgt_lang=tgt_lang_key)
                # ── 'translated': an online translation came back; show it and log how
                #    long it took from the end of the speech.
                elif kind == "translated":
                    seq, text = data
                    self._commit_translation(seq, text)
                    trace = self._traces.pop(seq, None)
                    if trace:
                        log_event(logging.INFO, "translation_displayed",
                                  run_id=trace["run_id"], seq=seq,
                                  speech_end_to_translation_ms=round(
                                      (time.monotonic() - trace["t_end"]) * 1000, 1))
                # ── 'status': a message for the status bar at the bottom of the window.
                elif kind == "status":
                    self.status_var.set(data)
                    log_event(logging.INFO, "status_updated", status=data)
                # ── 'level': the current input level, for the meter (~30 per second).
                elif kind == "level":
                    self._update_meter(*data)
                # ── 'metric': a timing figure for the Diagnostics tab.
                elif kind == "metric":
                    self._update_metric(*data)
        except queue.Empty:
            pass
        if handled:
            self._ui_stats.add("msgs_per_poll", handled)
        # Schedule the next check in 15 ms. This is how the function repeats
        # forever without blocking the window (see self.after in the guide).
        self.after(15, self._poll_queue)

    def _update_metric(self, name: str, value: float) -> None:
        """Keep running statistics for the Diagnostics tab."""
        if name == "transcription_ms":
            self._perf_stats["transcriptions"] += 1
            self._perf_stats["transcription_ms"].append(value)
        elif name == "translation_ms":
            self._perf_stats["translations"] += 1
            self._perf_stats["translation_ms"].append(value)
        elif name == "lock_wait_ms":
            self._perf_stats["lock_wait_ms"].append(value)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def _refresh_diagnostics(self):
        """Refresh the Diagnostics tab once a second."""
        self._refresh_diagnostics_once()
        self.after(1000, self._refresh_diagnostics)

    def _refresh_diagnostics_once(self):
        """Rewrite the Diagnostics tab with current health and performance figures."""
        # Averages of recent timings, and how many items wait in each queue.
        avg_transcribe_ms  = self._avg(self._perf_stats["transcription_ms"])
        avg_translate_ms  = self._avg(self._perf_stats["translation_ms"])
        avg_lock_wait_ms  = self._avg(self._perf_stats["lock_wait_ms"])
        phrase_backlog     = self.phrase_queue.qsize()
        translation_backlog     = self._translation_queue.qsize()

        # Choose a health message and colour: red if work is piling up in a
        # queue (the app can't keep up), green if listening, amber if idle.
        if phrase_backlog >= PHRASE_QUEUE_BACKLOG_WARN or translation_backlog >= TRANSLATION_QUEUE_BACKLOG_WARN:
            color, health = COL_ERR,  "⚠ Backlog high - translation lagging"
        elif self.is_listening:
            color, health = COL_OK,   "✔ Healthy - listening"
        else:
            color, health = COL_WARN, "● Idle"

        # The summary box at the top of the Diagnostics tab.
        summary = (
            f"{health}\n"
            f"Transcriptions: {self._perf_stats['transcriptions']}   "
            f"Avg transcribe: {avg_transcribe_ms:.0f} ms   "
            f"Avg translate: {avg_translate_ms:.0f} ms   "
            f"Avg lock wait: {avg_lock_wait_ms:.0f} ms\n"
            f"Phrase Q: {phrase_backlog}   Translation Q: {translation_backlog}   "
            f"Online translators: {len(self._translation_workers)}"
        )
        # A text box must be 'normal' to change its text; it is set back to
        # 'disabled' afterwards so the user can't type into it.
        self.diag_summary_box.configure(state="normal", text_color=color)
        self.diag_summary_box.delete("1.0", "end")
        self.diag_summary_box.insert("1.0", summary)
        self.diag_summary_box.configure(state="disabled")

        # The detail box below: current settings and where the log files are.
        detail_lines = [
            f"session_id            {self.session_id}",
            f"input device          {self.device_var.get()}",
            f"output device         {self.out_device_var.get()}",
            f"replay to output      {self.loopback_var.get()}  (VB-Audio cables only)",
            f"model                 {self.model_var.get()}",
            f"source language       {self.src_var.get()}",
            f"target language       {self.tgt_var.get()}",
            f"listening             {self.is_listening}",
            f"transcription workers {len(self._workers)}",
            f"translation workers   {len(self._translation_workers)}",
            f"whisper device        {WHISPER_DEVICE}",
            f"compute type          {getattr(self._model_loader, 'active_compute_type', COMPUTE_TYPE)}",
            f"phrase queue backlog  {phrase_backlog}",
            f"translation Q backlog {translation_backlog}",
            f"",
            f"log          {LOG_PATH}",
            f"error log    {ERROR_LOG_PATH}",
            f"summary      {LOG_SUMMARY_PATH}",
        ]
        self.diag_box.configure(state="normal")
        self.diag_box.delete("1.0", "end")
        self.diag_box.insert("1.0", "\n".join(detail_lines))
        self.diag_box.configure(state="disabled")

    def _set_live_status(self, text, color):
        """Set the coloured status badge in the header (LISTENING, MUTED, …)."""
        self.live_status_badge.configure(text=f" {text} ", fg_color=color)

    def _open_path(self, path: Path):
        """Open a file or folder in Windows Explorer / its default program."""
        try:
            target = path if path.exists() else APP_DIR
            os.startfile(str(target))
            log_event(logging.INFO, "open_path_requested", path=str(target))
        except Exception as e:
            self.status_var.set(f"Open failed: {e}")
            log_exception("open_path_failed", e, path=str(path))

    def _open_app_folder(self):
        """Open the folder that contains the app and its log files."""
        self._open_path(APP_DIR)

    def _export_log_summary(self):
        """The Export Log Summary button: write the summary file and open it."""
        try:
            summary = build_log_summary()
            self.status_var.set(f"Log summary written → {LOG_SUMMARY_PATH.name}")
            log_event(logging.INFO, "log_summary_exported",
                      path=str(LOG_SUMMARY_PATH),
                      total_entries=summary.get("total_entries", 0))
            self._refresh_diagnostics_once()
        except Exception as e:
            self.status_var.set(f"Export failed: {e}")
            log_exception("log_summary_export_failed", e)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _avg(values: Iterable[float]) -> float:
        """Average of a list of numbers (0 for an empty list)."""
        return (sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _trim_textbox(box):
        """Delete the oldest text so a box never holds more than
        MAX_TEXT_HISTORY_CHARS characters (keeps the window fast in long calls).
        """
        text = box.get("1.0", "end-1c")
        if len(text) > MAX_TEXT_HISTORY_CHARS:
            box.delete("1.0", f"1.0 + {len(text) - MAX_TEXT_HISTORY_CHARS} chars")

    @staticmethod
    def _at_bottom(box):
        """True if the text box is scrolled to the end. If the user has
        scrolled up to re-read something, new text must not yank the view
        back down (real-time caption design guidance)."""
        try:
            return box.yview()[1] >= 0.98
        except Exception:
            return True

    def _set_live(self, box, seq, text):
        """Show (or replace) the grey in-progress line for phrase `seq`."""
        follow = self._at_bottom(box)
        start, live_seq = self._live.get(box, (None, None))
        if live_seq is not None and seq < live_seq:
            return
        if start is not None:
            box.delete(start, "end")
        else:
            start = box.index("end-1c")
        prefix = "\n" if box.get("1.0", start).strip() else ""
        # Stable words (idea from whisper_streaming's "LocalAgreement"): the
        # words at the start of this preview that are the same as in the
        # previous preview of the same phrase have stopped changing, so they
        # are shown in normal text; only the still-changing rest is grey.
        prev_seq, prev_words = self._last_preview_words.get(box, (None, []))
        words = text.split()
        stable = 0
        if prev_seq == seq:
            while (stable < min(len(words), len(prev_words))
                   and words[stable] == prev_words[stable]):
                stable += 1
        self._last_preview_words[box] = (seq, words)
        stable_text = " ".join(words[:stable])
        changing_text = " ".join(words[stable:])
        box.insert(start, prefix + "… " + stable_text + (" " if stable_text else ""), "stable")
        box.insert("end-1c", changing_text, "live")
        box.tag_config("live", foreground="gray55")
        if follow:
            box.see("end")
        self._live[box] = (start, seq)

    def _commit(self, box, seq, text):
        """Replace the live line (if it belongs to `seq` or earlier) with the
        final text."""
        follow = self._at_bottom(box)
        start, live_seq = self._live.get(box, (None, None))
        if start is not None and live_seq <= seq:
            box.delete(start, "end")
            self._live.pop(box, None)
        if not text:
            return
        # Insert at the end of the finished text: before a newer phrase's
        # preview if one is showing, otherwise at the very end.
        live = self._live.get(box)
        anchor = live[0] if live else box.index("end-1c")
        before = box.get("1.0", anchor)
        joined, piece = continuation_join(before, text)
        if joined:      # "...они..." + "...включаться" -> one line
            cut, insert = trailing_marks_length(before), " " + piece
        else:
            cut, insert = 0, ("\n" if before.strip() else "") + text
        pos = box.index(f"{anchor} - {cut} chars")
        if cut:
            box.delete(pos, anchor)
        box.insert(pos, insert)
        if live:
            self._live[box] = (box.index(f"{pos} + {len(insert)} chars"), live[1])
        else:
            self._trim_textbox(box)
        if follow:
            box.see("end")

    def _change_text_size(self, step):
        """A−/A+ buttons: change the transcript font size (10-32), saved."""
        self._text_size = max(10, min(32, self._text_size + step))
        for box in (self.heard_box, self.translated_box):
            box.configure(font=ctk.CTkFont(size=self._text_size))
        save_settings(text_size=self._text_size)

    def _commit_translation(self, seq: int, text: str) -> None:
        """Show a finished translation, and speak it if Speak is on."""
        self._commit(self.translated_box, seq, text)
        if text and self._tts_on and not text.startswith("[translation failed"):
            self.speaker.say(text)
        self.status_var.set("Listening…" if self.is_listening else "Done.")

    @staticmethod
    def _append(box, text):
        """Add a line of text to the end of a text box and scroll to it."""
        existing_text = box.get("1.0", "end-1c")
        box.insert("end", ("\n" if existing_text else "") + text)
        box.see("end")

    def _on_close(self):
        """The window is closing: stop listening and the speaker, log, and exit."""
        try:
            if self.is_listening:
                self._stop_session()
            log_event(logging.INFO, "app_closing", session_id=self.session_id)
            self.speaker.shutdown()
        finally:
            self.destroy()


def run_classic():
    """Start the original customtkinter window."""
    log_event(logging.INFO, "app_main_start", gui="classic")
    app = InterpreterApp()
    app.mainloop()
    log_event(logging.INFO, "app_main_exit", gui="classic")

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
from .transcription import *  # noqa: F401,F403

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


log_event(logging.INFO, "app_module_loaded",
          app_dir=str(APP_DIR), log_path=str(LOG_PATH), error_log_path=str(ERROR_LOG_PATH))


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

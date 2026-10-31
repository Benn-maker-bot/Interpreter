"""
Voice Interpreter
=================

A desktop app that listens to speech, writes down what was said, and
translates it, live. Typical use: you are on a Teams/Telemost call with a
Russian-speaking friend whose voice plays through your FiiO headphones; the
app "listens" to the FiiO, shows the Russian words as they are spoken, and
shows the English translation next to them. It can also do the reverse
("My Voice": your microphone, English -> Russian).

This file is the whole application. It is long, so this guide explains the
big picture first. Every class and function below also has its own comment.


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
     A phrase ends at a short pause (0.3 s) or after 7 s at the best gap.

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
import logging.handlers
import os
import sys
import traceback
import uuid
import gzip
import shutil
from collections import Counter, deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests
import subprocess
import customtkinter as ctk

# PyAudioWPatch is a drop-in PyAudio fork that adds WASAPI loopback devices,
# letting us capture what an output device (e.g. FiiO K7) is playing.
try:
    import pyaudiowpatch as pyaudio
    _PYAUDIO_AVAILABLE = True
    _LOOPBACK_AVAILABLE = True
except ImportError:
    _LOOPBACK_AVAILABLE = False
    try:
        import pyaudio
        _PYAUDIO_AVAILABLE = True
    except ImportError:
        _PYAUDIO_AVAILABLE = False


def _add_cuda_dll_dirs():
    """Make the pip-installed NVIDIA CUDA DLLs (nvidia-cublas-cu12,
    nvidia-cudnn-cu12) loadable by ctranslate2 on Windows."""
    if os.name != "nt":
        return []
    roots = [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(getattr(sys, "_MEIPASS", APP_DIR)) / "nvidia")
    added = []
    for root in roots:
        for bin_dir in root.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(bin_dir))
    return added


# Device labels contain emoji; a cp1252 console would raise on every log line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paths / log config ────────────────────────────────────────────────────────
# Where the app keeps its files. APP_DIR is the folder containing this
# script, so logs and settings always sit next to the program.
APP_DIR               = Path(__file__).resolve().parent
# LOG_PATH: the main log (every event); ERROR_LOG_PATH: warnings/errors only;
# LOG_SUMMARY_PATH: written by Export Log Summary; SETTINGS_PATH: saved choices.
LOG_PATH              = APP_DIR / "interpreter.log.jsonl"
ERROR_LOG_PATH        = APP_DIR / "interpreter.error.log.jsonl"
LOG_SUMMARY_PATH      = APP_DIR / "interpreter.log.summary.json"
SETTINGS_PATH         = APP_DIR / "interpreter.settings.json"


def _app_version() -> str:
    """The git commit this copy of the program comes from (e.g. "366520e"),
    logged with each session so results can be tied to a code version."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=APP_DIR,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


APP_VERSION = _app_version()
# Start a new log file after 5 MB, keeping LOG_BACKUP_COUNT older ones.
MAX_LOG_BYTES         = 5 * 1024 * 1024
LOG_BACKUP_COUNT      = 5

# ── Tuning ────────────────────────────────────────────────────────────────────
# General limits. A *backlog* is work waiting in a queue; if it grows, the
# app is falling behind, so a warning is logged.
MAX_TEXT_HISTORY_CHARS         = 12_000
# Warn when this many phrases / translations are waiting to be processed.
PHRASE_QUEUE_BACKLOG_WARN      = 6
TRANSLATION_QUEUE_BACKLOG_WARN = 16
TRANSLATION_WORKERS            = 1          # >1 lets results arrive out of order
TRANSLATION_QUEUE_MAX          = 64


# ─────────────────────────────────────────────────────────────────────────────
# Structured JSONL logging
#
# Everything the app does is recorded with log_event("event_name", key=value).
# Each entry becomes one line of JSON in interpreter.log.jsonl (warnings and
# errors also go to interpreter.error.log.jsonl) and a readable line in the
# console. log_tools.py reads these files to measure speed and accuracy.
# ─────────────────────────────────────────────────────────────────────────────

class JsonLineFormatter(logging.Formatter):
    """Turns each log message into one line of JSON.

    JSON lines are easy for programs (like log_tools.py) to read back: every
    line is a dictionary with the time, level, event name and any extra fields
    passed to log_event().
    """
    def format(self, record):
        """Build the JSON text for one log record (called by Python's logging)."""
        payload = {
            "ts":      datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
            "level":   record.levelname,
            "logger":  record.name,
            "event":   getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "thread":  record.threadName,
            "process": record.process,
        }
        extra_fields = getattr(record, "fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class GZipRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A log file handler that starts a new file when the log gets big.

    When interpreter.log.jsonl passes MAX_LOG_BYTES it is renamed to .1, .2 …
    and the old copies are compressed with gzip so they take little space.
    """
    def doRollover(self):
        """Rotate the log files, then gzip the rotated copies."""
        super().doRollover()
        for i in range(self.backupCount, 0, -1):
            rotated = Path(f"{self.baseFilename}.{i}")
            gz_path = rotated.with_name(rotated.name + ".gz")
            if rotated.exists() and not gz_path.exists():
                try:
                    with open(rotated, "rb") as src, gzip.open(gz_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    rotated.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Log compression failed: {e}", file=sys.stderr)


class PlaintextFormatter(logging.Formatter):
    """Formats log messages as short readable lines for the console window."""
    def format(self, record):
        """Build one readable console line: time, level, thread, event, fields."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        event = getattr(record, "event", record.getMessage())
        msg = f"[{ts}] {record.levelname:^7} | {record.threadName[:15]:<15} | {event}"
        extra_fields = getattr(record, "fields", None)
        if isinstance(extra_fields, dict):
            # Format kwargs nicely
            args = ", ".join(f"{k}={v}" for k, v in extra_fields.items() if k not in ["exception", "event"])
            if args:
                msg += f" | {args}"
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


def _build_logger(name: str, path: Path, level=logging.INFO) -> logging.Logger:
    """Create a logger that writes JSON to `path` and readable text to the console.

    `level` is the minimum importance that gets written (INFO, WARNING …).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    
    # JSONL file handler
    f_handler = GZipRotatingFileHandler(
        path, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    f_handler.setFormatter(JsonLineFormatter())
    logger.addHandler(f_handler)
    
    # Plaintext console handler
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(PlaintextFormatter())
    logger.addHandler(s_handler)
    
    return logger


# The two loggers used by log_event(): APP_LOG gets everything (INFO and
# up), ERROR_LOG only WARNING and up.
APP_LOG   = _build_logger("interpreter.app",   LOG_PATH,       logging.INFO)
ERROR_LOG = _build_logger("interpreter.error", ERROR_LOG_PATH, logging.WARNING)


def log_event(level: int, event: str, **fields):
    """Write one event to the log. This is the function the whole app uses.

    `event` is a short name such as "phrase_trace"; any keyword arguments are
    stored as extra JSON fields. Warnings and errors also go to the error log.
    """
    APP_LOG.log(level, event, extra={"event": event, "fields": fields})
    if level >= logging.WARNING:
        ERROR_LOG.log(level, event, extra={"event": event, "fields": fields})


def log_exception(event: str, exc: Exception, **fields):
    """Log an error together with its full traceback (the list of calls that
    led to it), so problems can be diagnosed later from the log files.
    """
    merged = dict(fields)
    merged.setdefault("exception_type",    type(exc).__name__)
    merged.setdefault("exception_message", str(exc))
    APP_LOG.error(   event, exc_info=(type(exc), exc, exc.__traceback__), extra={"event": event, "fields": merged})
    ERROR_LOG.error( event, exc_info=(type(exc), exc, exc.__traceback__), extra={"event": event, "fields": merged})


def install_global_exception_hooks():
    """Make sure crashes that nobody caught are still written to the log.

    Python normally just prints such errors to the console (which may not be
    visible). These hooks log them as CRITICAL instead. Only the app itself
    installs them (see the bottom of the file), not scripts that import it.
    """
    def _excepthook(exc_type, exc_value, exc_tb):
        for log in (APP_LOG, ERROR_LOG):
            log.critical("unhandled_exception", exc_info=(exc_type, exc_value, exc_tb),
                         extra={"event": "unhandled_exception", "fields": {}})

    def _threading_excepthook(args):
        for log in (APP_LOG, ERROR_LOG):
            log.critical("unhandled_thread_exception",
                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                         extra={"event": "unhandled_thread_exception",
                                "fields": {"thread": args.thread.name if args.thread else None}})

    sys.excepthook      = _excepthook
    threading.excepthook = _threading_excepthook


log_event(logging.INFO, "app_module_loaded",
          app_dir=str(APP_DIR), log_path=str(LOG_PATH), error_log_path=str(ERROR_LOG_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Log summary helpers
#
# Small helpers behind the "Export Log Summary" button: read the log files
# back and count what happened.
# ─────────────────────────────────────────────────────────────────────────────

def iter_log_files(base_path: Path):
    """List the current log file plus its rotated, gzipped older copies."""
    files = [base_path] if base_path.exists() else []
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*.gz")))
    return files


def read_jsonl_entries(path: Path):
    """Read a .jsonl (or .jsonl.gz) log file and yield one dictionary per line,
    skipping any line that is not valid JSON.
    """
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


def build_log_summary():
    """Count log entries by level and event and write interpreter.log.summary.json.

    Used by the "Export Log Summary" button.
    """
    # Counters for log levels and event names, plus a list of problems.
    # (A Counter is a dict that counts: levels['ERROR'] += 1 starts at 0.)
    levels, events, issues, total = Counter(), Counter(), [], 0
    # Read every entry of every log file, counting as we go and keeping
    # warnings and errors aside.
    for path in iter_log_files(LOG_PATH):
        for entry in read_jsonl_entries(path):
            total += 1
            levels[entry.get("level", "UNKNOWN")] += 1
            events[entry.get("event",  "unknown")] += 1
            if entry.get("level") in {"ERROR", "CRITICAL", "WARNING"}:
                issues.append({
                    "ts":        entry.get("ts"),
                    "event":     entry.get("event"),
                    "message":   entry.get("message"),
                    "exception": entry.get("exception"),
                })
    # Gather the results and save them as an indented, readable JSON file.
    summary = {
        "generated_at":   datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "log_path":        str(LOG_PATH),
        "error_log_path":  str(ERROR_LOG_PATH),
        "total_entries":   total,
        "levels":          dict(levels),
        "top_events":      dict(events.most_common(20)),
        "recent_issues":   issues[-20:],
    }
    with open(LOG_SUMMARY_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


# ── Audio / VAD ────────────────────────────────────────────────────────────────
# Settings for recording and for deciding where phrases start and end.
# Whisper expects 16 kHz mono 16-bit audio (SAMPLE_RATE / CHANNELS /
# SAMPLE_WIDTH); devices usually record at 44.1 or 48 kHz, so audio is
# resampled. Times ending in _SEC are in seconds.
SAMPLE_RATE      = 16_000
# Whisper's audio format: 16 kHz (SAMPLE_RATE), mono (CHANNELS = 1),
# 16-bit samples (SAMPLE_WIDTH = 2 bytes). CHUNK_SAMPLES = 512 samples is
# one 32 ms block at 16 kHz; blocks at other rates last the same 32 ms.
CHUNK_SAMPLES    = 512
CHANNELS         = 1
SAMPLE_WIDTH     = 2
SILENCE_SEC      = 0.3     # pause that ends a phrase (sentence gaps are often ~0.25-0.4 s)
MAX_PHRASE_SEC   = 7       # longest phrase before a forced cut. Previews show text
                           # live, so a longer phrase costs no visible delay but
                           # means fewer cuts through words (the main error source)
CUT_SEARCH_SEC   = 1.5     # ...made where speech is least likely in this final window
PARTIAL_INTERVAL_SEC = 0.2 # live preview cadence; the worker skips stale ones,
                           # so this self-limits to what the GPU can keep up with
PARTIAL_MIN_SEC  = 0.5     # less audio than this makes Whisper guess wildly
MIN_PHRASE_SEC   = 0.5     # shorter clips are mostly silence and make Whisper hallucinate
MAX_READ_FAILURES = 20     # consecutive audio read errors (~1 s) before capture gives up
WATCHDOG_SEC      = 2      # how often the UI checks that capture/workers are alive
VAD_WINDOW_SEC   = 0.25    # audio context given to Silero VAD per decision
VAD_HYSTERESIS   = 0.15    # once in speech, stay until probability drops this far below
PREROLL_SEC      = 0.6     # audio kept from before VAD fired, so first words aren't clipped
                           # when the detector reacts late (more likely at low sensitivity).
                           # RealtimeSTT keeps 1 s. Leading silence is trimmed off again
                           # before Whisper, so a longer pre-roll costs almost nothing.
EARLY_FINAL_SEC  = 0.1     # "early transcription on silence" (idea from RealtimeSTT): this far
                           # into a pause, start the final transcription speculatively. If the
                           # pause reaches SILENCE_SEC the result is used at once (saving the
                           # ~0.2 s transcription time); if speech resumes it is thrown away.
SILENCE_FLOOR_DBFS = -65   # below this it's digital silence: skip VAD entirely
ONSET_PROB       = 0.2     # a speech score this high counts as "possibly speech": the time
                           # from then until the detector actually starts a phrase is logged
                           # as onset_delay_ms (how late, at this sensitivity, speech is noticed)
DEFAULT_SENS     = 0.5     # speech sensitivity 0..1; higher catches quieter/less clear speech
AGC_TARGET_DBFS  = -20     # clips are normalised to this RMS before Whisper...
AGC_MAX_GAIN_DB  = 30      # ...boosting quiet audio by at most this much
AGC_PEAK_DBFS    = -1      # and never letting peaks exceed this
NO_SPEECH_PROB   = 0.6     # Whisper's rule: discard only if no-speech prob is above
LOGPROB_FLOOR    = -1.0    # this AND the average token log-prob is below this
CONTEXT_TOKENS   = 96      # previous text fed back as context (Whisper's default)
CONTEXT_RESET_SEC = 6      # forget the context after this long a *silence* between
                           # phrases (timing it from the last finished phrase reset it
                           # mid-speech, since phrases last up to 7 s: that caused the
                           # "Спасибо." failures seen in the logs)
RECORD_DIR       = Path(__file__).resolve().parent / "recordings"
RECORD_KEEP      = 400     # newest phrase recordings kept for accuracy checks


def vad_threshold_for(sensitivity: float) -> float:
    """Map the 0..1 sensitivity slider to a Silero speech probability
    threshold: 0 -> 0.85 (only clear speech), 1 -> 0.15 (almost anything)."""
    return 0.85 - 0.7 * max(0.0, min(1.0, sensitivity))


def levels_dbfs(data: bytes):
    """RMS and peak of an int16 block in dBFS (0 dBFS = full scale)."""
    import numpy as np
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return -100.0, -100.0
    rms  = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    to_db = lambda v: 20 * math.log10(v / 32768.0) if v > 0 else -100.0
    return max(-100.0, to_db(rms)), max(-100.0, to_db(peak))


def apply_agc(audio_np):
    """Automatic gain control for one clip (float32, -1..1): bring speech to
    a consistent level for Whisper and VAD, with a gain cap so silence isn't
    amplified into noise and a peak limit so nothing clips."""
    import numpy as np
    rms = float(np.sqrt(np.mean(audio_np * audio_np))) if audio_np.size else 0.0
    if rms < 1e-6:
        return audio_np, 0.0
    gain = 10 ** (AGC_TARGET_DBFS / 20) / rms
    gain = min(gain, 10 ** (AGC_MAX_GAIN_DB / 20))
    peak = float(np.max(np.abs(audio_np))) * gain
    limit = 10 ** (AGC_PEAK_DBFS / 20)
    if peak > limit:
        gain *= limit / peak
    return (audio_np * gain).astype(np.float32), 20 * math.log10(gain)

# VB-Audio virtual cables appear as "CABLE-A Output", "CABLE Output", etc.
# Voicemeeter mix buses are deliberately excluded so we capture the raw cable.
VB_KEYWORDS     = ["cable output", "cable-a output", "cable-b output",
                   "vb-audio virtual", "virtual cable"]
# Words that identify an NVIDIA Broadcast (noise-removing) microphone,
# preferred as the default for My Voice mode.
NVIDIA_KEYWORDS = ["nvidia", "broadcast", "rtx voice"]
# Fiio output devices: FiiO K7, K3, K5, K9, Q3, Q5, Q7, Q series, KA1, KA3, KA5
FIIO_KEYWORDS   = ["fiio", "fio k", "fio q", "fio dac",
                   "k3)", "k5)", "k7)", "k9)",
                   "q3)", "q5)", "q7)",
                   "ka1", "ka3", "ka5",
                   "btr", "realtek usb audio"]

# ── Whisper models ────────────────────────────────────────────────────────────
# MODELS maps the names shown in the Model menu to Whisper model sizes.
# Bigger models are more accurate but slower. LANGUAGES maps menu names to
# (Whisper language code, translation-service language code).
MODELS = {
    "Tiny  — fastest":        "tiny",
    "Small — recommended ✓":  "small",
    "Medium — most accurate": "medium",
    # Measured on real Russian call audio vs large-v3: 7.5% word errors at
    # ~225 ms, against Medium 9.9% at ~276 ms. It cannot translate (trained
    # for transcription only), so translation comes from Opus-MT/online.
    "Turbo — best accuracy ★": "large-v3-turbo",
}
# Model chosen on first start: Tiny without a GPU (anything bigger is too
# slow on a CPU), Turbo with one.
DEFAULT_MODEL_CPU = "Tiny  — fastest"
DEFAULT_MODEL_GPU = "Turbo — best accuracy ★"

# Model for live previews (finished phrases always use the main model).
# Measured preview cost on the RTX 4060 Ti: Tiny ~40 ms, Small ~90 ms,
# Turbo ~190 ms (which saturated the GPU and delayed finished phrases).
PREVIEW_CHOICES = {"Tiny (instant)": "tiny", "Small (balanced)": "small", "Main model": None}

# Local neural translation (Opus-MT via CTranslate2, ~30-55 ms on GPU).
# Replaces the online services for these pairs: no network, no rate limits.
OPUS_MT_REPOS = {
    ("ru", "en"): "ooeoeo/opus-mt-ru-en-ct2-float16",
    ("en", "ru"): "ooeoeo/opus-mt-en-ru-ct2-float16",
}

# Menu name -> (Whisper language code, online-translation language code).
# They differ only for Chinese (Google needs "zh-CN").
LANGUAGES = {
    "English":            ("en", "en"),
    "Russian":            ("ru", "ru"),
    "Arabic":             ("ar", "ar"),
    "Chinese (Mandarin)": ("zh", "zh-CN"),
    "Dutch":              ("nl", "nl"),
    "French":             ("fr", "fr"),
    "German":             ("de", "de"),
    "Greek":              ("el", "el"),
    "Hindi":              ("hi", "hi"),
    "Italian":            ("it", "it"),
    "Japanese":           ("ja", "ja"),
    "Korean":             ("ko", "ko"),
    "Polish":             ("pl", "pl"),
    "Portuguese":         ("pt", "pt"),
    "Spanish":            ("es", "es"),
    "Turkish":            ("tr", "tr"),
    "Ukrainian":          ("uk", "uk"),
}

# ── UI colours ────────────────────────────────────────────────────────────────
# Colours used by the window. A pair means (light theme, dark theme).
COL_MY_VOICE = ("#1A6B3A", "#1A6B3A")
# COL_MY_* / COL_FR_*: My Voice (green) / Friend (blue) button colours,
# normal and when the mouse is over them; COL_INACTIVE: unselected mode;
# COL_STOP: the red Unmute button; COL_OK/WARN/ERR/INFO: status colours.
COL_MY_HOVER = ("#155730", "#155730")
COL_FRIEND   = ("#1A4A7A", "#1A4A7A")
COL_FR_HOVER = ("#153D66", "#153D66")
COL_INACTIVE = ("gray30",  "gray25")
COL_IN_HOVER = ("gray40",  "gray35")
COL_STOP     = "#C0392B"
COL_STOP_HOV = "#A93226"
COL_OK       = "#27AE60"
COL_WARN     = "#F39C12"
COL_ERR      = "#C0392B"
COL_INFO     = "#2980B9"


# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
#
# Whisper is much faster on an NVIDIA graphics card (GPU). This checks, once
# at start-up, that the GPU and its CUDA libraries are usable, and picks
# sensible defaults for GPU or CPU.
# ─────────────────────────────────────────────────────────────────────────────

def detect_device():
    """Decide whether Whisper runs on the NVIDIA GPU ("cuda") or the CPU.

    Returns (device, compute_type, worker_count, default_model_label). The GPU
    is only chosen if the CUDA libraries really load; otherwise Whisper would
    crash later in the middle of transcribing.
    """
    try:
        dll_dirs = _add_cuda_dll_dirs()
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        # On Windows, just because get_cuda_device_count() > 0 doesn't mean
        # the cuBLAS/cuDNN libraries are loadable. If they are missing,
        # CTranslate2 throws a RuntimeError at transcribe() time, which can deadlock.
        # We explicitly verify we can load the DLLs before trusting 'cuda'.
        if n > 0 and os.name == "nt":
            import ctypes
            for dll in ("cublas64_12.dll", "cudnn64_9.dll"):
                try:
                    ctypes.cdll.LoadLibrary(dll)
                except Exception:
                    log_event(logging.WARNING, "cuda_detection_dll_missing",
                              dll=dll, searched=dll_dirs)
                    n = 0
                    break

        log_event(logging.INFO, "cuda_detection_result", cuda_device_count=n,
                  ctranslate2_version=ctranslate2.__version__)
        if n > 0:
            return "cuda", "float16", 1, DEFAULT_MODEL_GPU
    except Exception as e:
        log_exception("cuda_detection_failed", e)
    return "cpu", "int8", 1, DEFAULT_MODEL_CPU


# Decided once at start-up:
#   WHISPER_DEVICE  "cuda" (GPU) or "cpu"
#   COMPUTE_TYPE    number format for the models (float16 on GPU, int8 on CPU)
#   NUM_WORKERS     transcription threads (always 1, see WhisperModelLoader)
#   _DEFAULT_MODEL  the Model-menu entry to use if none was saved
WHISPER_DEVICE, COMPUTE_TYPE, NUM_WORKERS, _DEFAULT_MODEL = detect_device()
# Only one AI model may use the GPU at a time; everything that runs a
# model does it inside `with INFERENCE_LOCK:` (see Lock in the glossary).
INFERENCE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Online translation (fallback)
#
# Normally translation is done locally by Opus-MT (see LocalTranslator) or by
# Whisper itself. These functions are the last resort, for language pairs
# with no local model: they call free web services (Google, then MyMemory)
# over one shared, kept-open HTTPS connection (_http).
# ─────────────────────────────────────────────────────────────────────────────

# One shared web connection for the online translators. Re-using it saves
# ~200 ms per request (no new secure-connection handshake each time).
# _adapter sets how many connections may be open and turns off automatic
# retries (a refused request should go to the next provider instead).
_http = requests.Session()
_http.headers.update({"User-Agent": "Mozilla/5.0"})
_http.adapters.clear()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
_http.mount("http://",  _adapter)
_http.mount("https://", _adapter)


class RateLimitedError(Exception):
    """Raised when an online translation service says "too many requests" (HTTP 429)."""
    pass


def _google_translate(text: str, src: str, tgt: str) -> str:
    """Translate text with Google's free web endpoint (online fallback only).

    Google often blocks this endpoint (HTTP 429); the caller then pauses it.
    """
    url    = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
    response   = _http.get(url, params=params, timeout=6)
    if response.status_code == 429:
        raise RateLimitedError("Google returned 429 Too Many Requests")
    response.raise_for_status()
    data = response.json()
    return "".join(p[0] for p in data[0] if p[0]).strip()


# MyMemory's list of supported languages, fetched the first time it's needed.
_MYMEMORY_CODES: dict | None = None

def _mymemory_code(code: str) -> str:
    """Convert a short language code ("ru") into the regional code MyMemory
    expects ("ru-RU"), using MyMemory's own list of supported languages.
    """
    # MyMemory wants region codes ("ru-RU"), not bare ISO codes ("ru").
    global _MYMEMORY_CODES
    if _MYMEMORY_CODES is None:
        from deep_translator import MyMemoryTranslator
        _MYMEMORY_CODES = MyMemoryTranslator(source="en-GB", target="ru-RU") \
            .get_supported_languages(as_dict=True)
    codes = list(_MYMEMORY_CODES.values())
    if code in codes:
        return code
    for c in codes:
        if c.split("-")[0].lower() == code.split("-")[0].lower():
            return c
    raise ValueError(f"MyMemory does not support language '{code}'")


def _mymemory_translate(text: str, src: str, tgt: str) -> str:
    """Translate text with the MyMemory online service (second online fallback)."""
    # Called directly over the shared keep-alive session: deep-translator
    # opens a fresh TLS connection per call, adding ~200 ms each time.
    response = _http.get("https://api.mymemory.translated.net/get",
                     params={"q": text,
                             "langpair": f"{_mymemory_code(src)}|{_mymemory_code(tgt)}"},
                     timeout=6)
    if response.status_code == 429:
        raise RateLimitedError("MyMemory returned 429 Too Many Requests")
    response.raise_for_status()
    data = response.json()
    if data.get("responseStatus") == 429 or "QUOTA" in str(data.get("responseDetails", "")).upper():
        raise RateLimitedError(str(data.get("responseDetails")))
    if data.get("responseStatus") not in (200, "200"):
        raise RuntimeError(f"MyMemory error: {data.get('responseDetails')}")
    return (data["responseData"]["translatedText"] or "").strip()


# Providers are tried in order. A provider that rate-limits us is skipped for
# an increasing cooldown so we stop hammering it (which prolongs the block).
_PROVIDERS = [("google", _google_translate), ("mymemory", _mymemory_translate)]
# How long to stop using a provider that said 'too many requests': 30 s
# the first time, doubling each time up to 600 s (10 min).
# _provider_state: per provider, 'until' = when it may be used again and
# 'cooldown' = the current pause length. _provider_lock protects it
# because more than one thread may translate at once.
_PROVIDER_COOLDOWN_BASE = 30.0
_PROVIDER_COOLDOWN_MAX  = 600.0
_provider_state = {name: {"until": 0.0, "cooldown": 0.0} for name, _ in _PROVIDERS}
_provider_lock  = threading.Lock()


@lru_cache(maxsize=512)
def _translate_cached(text: str, src: str, tgt: str) -> str:
    """Translate text online, trying each provider in turn (Google, MyMemory).

    Results are cached (lru_cache) so repeated phrases are instant. A provider
    that rate-limits us is skipped for a cooldown that doubles each time.
    Only used when no local Opus-MT model exists for the language pair.
    """
    if src == tgt:
        return text
    # Try each online provider in order (_PROVIDERS: Google, then MyMemory)
    # until one succeeds. `errors` collects why each one failed, for the log.
    errors = []
    for name, translate_fn in _PROVIDERS:
        # Is this provider 'cooling down' after refusing us recently? If so,
        # skip it. (_provider_lock: several threads may read/change the state.)
        with _provider_lock:
            state = _provider_state[name]
            wait  = state["until"] - time.monotonic()
        if wait > 0:
            errors.append(f"{name}: cooling down ({wait:.0f}s)")
            continue
        try:
            # Call the provider. On success, reset its cooldown and return the text.
            t0         = time.perf_counter()
            translated = translate_fn(text, src, tgt)
            with _provider_lock:
                state["cooldown"] = 0.0
            log_event(logging.INFO, "translation_completed",
                      provider=name, src=src, tgt=tgt,
                      input_chars=len(text), output_chars=len(translated),
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
            return translated
        # The provider said 'too many requests' (HTTP 429): don't ask it again
        # for a while. The pause doubles each time (30 s, 60 s, 120 s …, up to
        # 10 min) - called 'exponential backoff' - so we stop pestering it.
        except RateLimitedError as e:
            with _provider_lock:
                state["cooldown"] = min(_PROVIDER_COOLDOWN_MAX,
                                        max(_PROVIDER_COOLDOWN_BASE, state["cooldown"] * 2))
                state["until"]    = time.monotonic() + state["cooldown"]
            # INFO: a blocked service is retried after each cooldown by design
            log_event(logging.INFO, "translation_rate_limited",
                      provider=name, cooldown_s=state["cooldown"], error=str(e))
            errors.append(f"{name}: rate limited")
        except Exception as e:
            log_exception("translation_provider_failed", e,
                          provider=name, src=src, tgt=tgt, input_chars=len(text))
            errors.append(f"{name}: {type(e).__name__}")
    # Every provider failed: log it and raise an error; the caller then shows
    # '[translation failed: …]' in the translation box.
    log_event(logging.ERROR, "translation_failed",
              src=src, tgt=tgt, input_chars=len(text), errors=errors)
    raise RuntimeError("; ".join(errors))


class TranslationWorker(threading.Thread):
    """Background thread that does *online* translation, one phrase at a time.

    The window puts (seq, text, source, target) items on a queue; this thread
    translates them and puts the result on result_queue. Kept to one thread so
    results come back in the same order the phrases were spoken.
    """
    def __init__(self, wid: int, request_queue: queue.Queue, result_queue: queue.Queue):
        """Set up the worker. `wid` is just a number used in the thread's name."""
        super().__init__(daemon=True, name=f"translation-worker-{wid}")
        # wid: this worker's number (for its thread name); request_queue: phrases
        # to translate; result_queue: where results go; _stop_evt: set to stop.
        self.wid           = wid
        self.request_queue = request_queue
        self.result_queue  = result_queue
        self._stop_evt     = threading.Event()

    def stop(self):
        """Ask the thread to finish (the None item wakes it up if it is waiting)."""
        self._stop_evt.set()
        self.request_queue.put(None)

    def run(self):
        """Main loop: take a phrase from the queue, translate it, send the result."""
        while not self._stop_evt.is_set():
            item = self.request_queue.get()
            if item is None:
                break
            seq, text, src_lang_key, tgt_lang_key = item
            _, src = LANGUAGES.get(src_lang_key, ("en", "en"))
            _, tgt = LANGUAGES.get(tgt_lang_key, ("ru", "ru"))
            try:
                t0  = time.perf_counter()
                out = _translate_cached(text, src, tgt)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                self.result_queue.put(("translated", (seq, out or text)))
                self.result_queue.put(("metric", ("translation_ms", elapsed_ms)))
            except Exception as e:
                # Show the failure in the translation box, not just the status
                # bar, which "Listening…" overwrites almost immediately.
                self.result_queue.put(("translated", (seq, f"[translation failed: {text}]")))
                self.result_queue.put(("status", f"Translation error: {e}"))


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
#
# Functions for finding audio devices and their capabilities. Windows
# exposes each device several times through different "host APIs"
# (WASAPI, DirectSound, MME…); these helpers keep the best one and pick
# sensible defaults (FiiO for output, the FiiO's loopback for Friend mode).
# ─────────────────────────────────────────────────────────────────────────────

def get_device_native_rate(portaudio, device_index) -> int:
    """Return the native sample rate reported by PortAudio for a device.
    Falls back to 44100 if the index is invalid or the query fails."""
    if device_index is None or device_index < 0:
        return 44100
    try:
        info = portaudio.get_device_info_by_index(device_index)
        return int(info.get("defaultSampleRate", 44100))
    except Exception:
        return 44100


def rms(data: bytes) -> float:
    """Root-mean-square loudness of a block of 16-bit audio (a raw number,
    not decibels). Uses NumPy because a plain Python loop is far slower.
    """
    import numpy as np
    n = len(data) // SAMPLE_WIDTH
    if n == 0:
        return 0.0
    samples = np.frombuffer(data, dtype=np.int16, count=n).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


# Host API preference order for PortAudio on Windows.
# WDM-KS (usually index 3) requires exclusive device access and will fail with
# -9999 "Unanticipated host error" when the Windows audio engine already has
# the device open.  We skip it entirely.
_PREFERRED_HOST_API_ORDER = ["Windows WASAPI", "Windows DirectSound", "MME"]
# Audio systems never to use (WDM-KS needs exclusive access and fails
# while Windows is using the device).
_SKIP_HOST_API_NAMES      = {"Windows WDM-KS"}


def _get_host_api_map(portaudio):
    """Return {api_index: api_name} and the preferred-order api_index list."""
    api_map  = {}
    pref_idx = []
    other    = []
    skip_set = set()
    for i in range(portaudio.get_host_api_count()):
        info = portaudio.get_host_api_info_by_index(i)
        api_map[i] = info["name"]
        if info["name"] in _SKIP_HOST_API_NAMES:
            skip_set.add(i)
    for name in _PREFERRED_HOST_API_ORDER:
        for idx, n in api_map.items():
            if n == name:
                pref_idx.append(idx)
    # Any remaining (not preferred and not skipped)
    for idx in api_map:
        if idx not in pref_idx and idx not in skip_set:
            other.append(idx)
    return api_map, pref_idx + other, skip_set


def probe_openable_rate(portaudio, device_index, is_input):
    """Try common sample rates and return the first one PortAudio accepts."""
    candidates = [48000, 44100, 16000, 22050, 32000, 8000]
    for rate in candidates:
        try:
            if is_input:
                portaudio.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=1,
                    input_format=pyaudio.paInt16,
                )
            else:
                portaudio.is_format_supported(
                    rate,
                    output_device=device_index,
                    output_channels=2,
                    output_format=pyaudio.paInt16,
                )
            return rate
        except ValueError:
            continue
    return 44100  # last-resort guess


# Ordered priority list for VB-Audio input auto-selection.
# CABLE-A is the conventional "loopback capture" cable in most setups;
# prefer it over CABLE-B or the un-lettered CABLE.
VB_INPUT_PRIORITY_KEYWORDS = [
    "cable-a output",          # VB-Audio Virtual Cable A
    "cable output (vb-audio",  # CABLE Output (VB-Audio Virtual Cable)
    "cable-b output",          # last resort: Cable B
    "voicemeeter out",         # Voicemeeter mix buses
]


def find_best_vb_input(input_devices):
    """Return the label of the best VB-Audio input, preferring Cable A."""
    for priority_kw in VB_INPUT_PRIORITY_KEYWORDS:
        for _idx, label in input_devices:
            if priority_kw in label.lower():
                return label
    return None


def get_audio_devices():
    """Return (input_devices, output_devices) as lists of (index, label).

    - Skips WDM-KS devices (need exclusive access, fail under Windows audio engine).
    - For each physical device name, keeps only the best host-API variant
      (WASAPI > DirectSound > MME).
    """
    if not _PYAUDIO_AVAILABLE:
        log_event(logging.WARNING, "pyaudio_unavailable")
        return [(-1, "Default Microphone")], [(-1, "Default Output")]

    # inputs/outputs: the lists we build, as (device number, label).
    # seen_in/seen_out: names already added. Windows lists each device once
    # per host API (WASAPI, DirectSound, MME); we keep only the first, best one.
    inputs, outputs, seen_in, seen_out = [], [], {}, {}
    portaudio = None
    try:
        portaudio = pyaudio.PyAudio()
        # Visit the host APIs best-first (WASAPI first), so the first time we
        # meet a device name it is through the best audio system.
        api_map, pref_order, skip_apis = _get_host_api_map(portaudio)

        # Collect all devices grouped by name, in preferred-API order
        for api_idx in pref_order:
            for i in range(portaudio.get_device_count()):
                info = portaudio.get_device_info_by_index(i)
                if info.get("hostApi") != api_idx:
                    continue
                name = info["name"]
                name_lower   = name.lower()
                # A device that can record (has input channels) goes in the input list.
                # A symbol is put in front of the name to help you spot it:
                #   speaker = loopback (what an output device is playing),
                #   microphone = NVIDIA Broadcast, star = VB-Audio virtual cable.
                if info.get("maxInputChannels", 0) >= 1 and name not in seen_in:
                    seen_in[name] = True
                    prefix = "\U0001f50a " if info.get("isLoopbackDevice")           else \
                             "\U0001f3a4 " if any(k in name_lower for k in NVIDIA_KEYWORDS) else \
                             "\u2605 "      if any(k in name_lower for k in VB_KEYWORDS)     else ""
                    inputs.append((i, prefix + name))
                # A device that can play (has output channels) goes in the output list;
                # headphone symbol = FiiO, star = VB-Audio cable.
                if info.get("maxOutputChannels", 0) >= 1 and name not in seen_out:
                    seen_out[name] = True
                    is_fiio = any(k in name_lower for k in FIIO_KEYWORDS)
                    is_vb   = any(k in name_lower for k in VB_KEYWORDS)
                    prefix  = "\U0001f3a7 " if is_fiio else \
                              "\u2605 "      if is_vb   else ""
                    outputs.append((i, prefix + name))

        try:
            default_in  = portaudio.get_default_input_device_info().get("index")
            default_out = portaudio.get_default_output_device_info().get("index")
        except Exception:
            default_in = default_out = None
        log_event(logging.INFO, "audio_devices_enumerated",
                  input_count=len(inputs), output_count=len(outputs),
                  default_in=default_in, default_out=default_out,
                  preferred_api_order=[api_map.get(i) for i in pref_order[:4]])
    except Exception as e:
        log_exception("audio_device_enumeration_failed", e)
    finally:
        # A leaked PortAudio instance breaks WASAPI streams opened later
        # from other threads ("Unanticipated host error").
        if portaudio is not None:
            # Always shut PortAudio down again: a leftover instance makes streams
            # opened later by other threads fail ("Unanticipated host error").
            portaudio.terminate()
    # If nothing was found, offer the system defaults (-1 = default device).
    return (inputs or [(-1, "Default Microphone")],
            outputs or [(-1, "Default Output")])


# Ordered priority list for output device auto-selection: the first keyword
# that matches any device name wins.  This lets K7 beat Q series.
FIIO_PRIORITY_KEYWORDS = [
    "fiio k7", "fiio k9", "fiio k5", "fiio k3",  # desktop DAC/amps first
    "fiio q", "fiio ka",                           # portable
    "fiio",                                        # any other Fiio
    "realtek usb audio",                           # fallback (Fiio uses Realtek USB chip)
]


def find_best_friend_input(input_devices):
    """Default input for Friend mode: capture what the FiiO is playing (the
    call audio from Teams/Telemost) via WASAPI loopback, else a VB-Audio cable."""
    loopbacks = [(i, l) for i, l in input_devices if "[loopback]" in l.lower()]
    for priority_kw in FIIO_PRIORITY_KEYWORDS:
        for _idx, label in loopbacks:
            if priority_kw in label.lower():
                return label
    return find_best_vb_input(input_devices)


def find_device_by_keywords(devices, keywords):
    """Return the first device label whose name (lowercased) contains any keyword.
    Loopback (output-capture) devices are skipped: "nvidia" would otherwise
    match "NVIDIA High Definition Audio [Loopback]" (a monitor's speakers)."""
    for _idx, label in devices:
        if "[loopback]" in label.lower():
            continue
        if any(kw in label.lower() for kw in keywords):
            return label
    return None


def find_best_output_device(output_devices):
    """Return the label for the best Fiio output, using FIIO_PRIORITY_KEYWORDS
    so that K7 > Q series etc., rather than whichever appears first in the
    PortAudio device list."""
    for priority_kw in FIIO_PRIORITY_KEYWORDS:
        for _idx, label in output_devices:
            if priority_kw in label.lower():
                return label
    return None


def open_loopback_keepalive(portaudio, loopback_info, attempts=6):
    """Retry with growing waits: PortAudio can stay busy for a second or more
    while a previous session's stream shuts down (seen at 14:51:16)."""
    for attempt in range(attempts):
        stream = _open_loopback_keepalive(portaudio, loopback_info, log_failure=attempt == attempts - 1)
        if stream is not None:
            if attempt:
                log_event(logging.INFO, "loopback_keepalive_retry_succeeded", attempt=attempt + 1)
            return stream
        time.sleep(0.15 * (attempt + 1))
    return None


def _open_loopback_keepalive(portaudio, loopback_info, log_failure=True):
    """WASAPI loopback delivers no data while nothing is playing, so a blocking
    read() would hang. Playing silence to the same render device keeps the
    loopback stream ticking. Returns the stream (caller closes it) or None."""
    try:
        wasapi = portaudio.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        name   = loopback_info["name"].replace(" [Loopback]", "")
        for i in range(portaudio.get_device_count()):
            render_device = portaudio.get_device_info_by_index(i)
            if (render_device["hostApi"] == wasapi and not render_device.get("isLoopbackDevice")
                    and render_device["maxOutputChannels"] > 0 and render_device["name"] == name):
                ch = min(2, int(render_device["maxOutputChannels"]))
                silence_cb = lambda _in, frames, _t, _s: (b"\x00" * frames * 2 * ch,
                                                          pyaudio.paContinue)
                stream = portaudio.open(format=pyaudio.paInt16, channels=ch,
                                 rate=int(render_device["defaultSampleRate"]), output=True,
                                 output_device_index=i, stream_callback=silence_cb)
                log_event(logging.INFO, "loopback_keepalive_opened",
                          render_device=name, render_index=i)
                return stream
        log_event(logging.WARNING, "loopback_keepalive_device_not_found", name=name)
    except Exception as e:
        if log_failure:
            log_exception("loopback_keepalive_failed", e)
    return None


def downmix(data: bytes, channels: int) -> bytes:
    """Average all channels to mono (taking only the left channel loses audio
    that is panned right or sent on the right channel only)."""
    if channels <= 1:
        return data
    import numpy as np
    arr = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
    return arr.mean(axis=1).astype(np.int16).tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# VB-Audio loopback output thread
# Reads raw PCM from a ring-buffer and writes it to the chosen output device
# so the user hears the captured audio through their headphones (Fiio etc.).
# ─────────────────────────────────────────────────────────────────────────────

class LoopbackOutputThread(threading.Thread):
    """Plays captured audio back out of another device.

    Only used with VB-Audio virtual cables ("Hear input through output"): the
    call audio goes into the cable, so without this you would not hear it.
    Not used when capturing an output device directly (it is already playing).
    """
    def __init__(self, output_device_index, input_sample_rate, status_cb):
        """Remember which device to play to; audio arrives later through push()."""
        super().__init__(daemon=True, name="loopback-output")
        # output_device_index: device to play to; input_sample_rate: rate of the
        # audio we receive; _buf: queue of audio blocks waiting to be played.
        self.output_device_index = output_device_index
        self.input_sample_rate   = input_sample_rate   # rate the capture thread uses
        self.status_cb           = status_cb
        self._stop_evt           = threading.Event()
        self._buf: queue.Queue   = queue.Queue(maxsize=200)

    def push(self, chunk: bytes):
        """Upmix mono chunk to stereo if the output device is stereo."""
        ch = getattr(self, "_out_channels", 1)
        if ch == 2:
            import struct as _s
            n     = len(chunk) // 2
            mono  = _s.unpack_from("<" + str(n) + "h", chunk)
            chunk = _s.pack("<" + str(n * 2) + "h",
                            *[v for sample in mono for v in (sample, sample)])
        try:
            self._buf.put_nowait(chunk)
        except queue.Full:
            pass  # drop under pressure

    def stop(self):
        """Ask the thread to finish."""
        self._stop_evt.set()
        self._buf.put(b"")

    def run(self):
        """Open the output device and keep writing queued audio blocks to it."""
        if not _PYAUDIO_AVAILABLE:
            return
        pa     = pyaudio.PyAudio()
        stream = None
        try:
            # Probe the actual rate and channel count the device accepts.
            # WDM-KS devices are excluded from enumeration, but if we still
            # land on a device that refuses to open, we catch below.
            # Find a format the output device accepts: stereo or mono, and a sample
            # rate it supports. is_format_supported() raises ValueError if not, so
            # we try the options in order and stop at the first that works.
            out_channels = 1
            out_rate     = probe_openable_rate(pa, self.output_device_index, is_input=False)
            # Try stereo (most DACs), then mono, at multiple rates
            found = False
            for ch in [2, 1]:
                for rate in [out_rate, 48000, 44100, 16000]:
                    try:
                        pa.is_format_supported(
                            rate,
                            output_device=self.output_device_index,
                            output_channels=ch,
                            output_format=pyaudio.paInt16,
                        )
                        out_rate     = rate
                        out_channels = ch
                        found = True
                        break
                    except ValueError:
                        continue
                if found:
                    break
            # Open the output stream with the format found. frames_per_buffer is how
            # many samples are written at a time (about one 32 ms block).
            self._out_channels = out_channels
            chunk_sz = int(out_rate * CHUNK_SAMPLES / SAMPLE_RATE) or CHUNK_SAMPLES
            kwargs = dict(format=pyaudio.paInt16, channels=out_channels,
                          rate=out_rate, output=True,
                          frames_per_buffer=chunk_sz)
            if self.output_device_index is not None and self.output_device_index >= 0:
                kwargs["output_device_index"] = self.output_device_index
            stream = pa.open(**kwargs)
            log_event(logging.INFO, "loopback_output_opened",
                      device_index=self.output_device_index,
                      rate=out_rate, channels=out_channels, chunk_sz=chunk_sz)
        except Exception as e:
            if "Invalid output device" in str(e) or "Invalid device" in str(e):
                self.status_cb("ERROR: Output device disconnected or changed. Please restart app.")
            log_exception("loopback_output_open_failed", e,
                          device_index=self.output_device_index)
            return

        # Main loop: take audio blocks from our queue and play them. An empty
        # block (b"") is the stop signal sent by stop().
        while not self._stop_evt.is_set():
            try:
                chunk = self._buf.get(timeout=0.05)
            except queue.Empty:
                continue
            if not chunk:
                break
            try:
                stream.write(chunk)
            except Exception as e:
                log_exception("loopback_output_write_failed", e)
                break

        try:
            if stream:
                stream.close()
        finally:
            pa.terminate()
        log_event(logging.INFO, "loopback_output_stopped")

# ─────────────────────────────────────────────────────────────────────────────
# Audio capture + VAD
#
# AudioCaptureThread is the start of the pipeline: it records audio and
# decides what is speech. PerfWindow and LatestSlot are small helpers it
# uses. VAD = Voice Activity Detection (Silero VAD, a tiny neural network).
# ─────────────────────────────────────────────────────────────────────────────

STATS_INTERVAL_SEC = 10   # how often each component logs its perf summary


class PerfWindow:
    """Collects samples per metric and summarises them as n/p50/p95/max for a
    periodic stats log line, then resets. Cheap enough for the audio loop."""
    def __init__(self):
        """Start an empty collection of samples."""
        # _data: metric name -> list of samples; started: when this window began.
        self._data = {}
        self.started = time.monotonic()

    def add(self, name, value):
        """Record one measurement (e.g. add("vad_ms", 2.4))."""
        self._data.setdefault(name, []).append(value)

    def due(self):
        """True when STATS_INTERVAL_SEC has passed and a summary should be logged."""
        return time.monotonic() - self.started >= STATS_INTERVAL_SEC

    def summary(self):
        """Return {metric: {n, p50, p95, max}} for the collected samples, then reset.

        p50 is the median (half the values are lower); p95 means 95 % of values
        were at or below it, which shows the slow cases without the very worst.
        """
        out = {}
        for name, vals in self._data.items():
            v = sorted(vals)
            out[name] = {"n": len(v), "p50": round(v[len(v) // 2], 2),
                         "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))], 2),
                         "max": round(v[-1], 2)}
        self._data = {}
        self.started = time.monotonic()
        return out


class LatestSlot:
    """Single-item mailbox: put() overwrites, take() empties. Live previews
    only matter if current, so a stale one is replaced rather than queued."""
    def __init__(self):
        """Start empty."""
        # _lock: lets two threads use the slot safely; _item: the stored item.
        self._lock = threading.Lock()
        self._item = None
        self.overwritten = 0   # previews dropped because a newer one arrived

    def put(self, item):
        """Store an item, replacing (and counting) any item not yet taken."""
        with self._lock:
            if self._item is not None:
                self.overwritten += 1
            self._item = item

    def take(self):
        """Return the stored item (or None) and empty the slot."""
        with self._lock:
            item, self._item = self._item, None
            return item


class AudioCaptureThread(threading.Thread):
    """Records audio from one device and turns it into phrases of speech.

    For every ~32 ms block of sound it: measures the level (for the meter),
    asks Silero VAD whether it is speech, and adds speech to the current
    phrase. Finished phrases go to `phrase_queue` (for the final, accurate
    transcription); while a phrase is still growing, a copy is put in
    `partial_slot` every PARTIAL_INTERVAL_SEC for a quick preview.

    For "loopback" devices (what an output device is playing) it also plays
    silence to that device, because Windows delivers no loopback audio at all
    while the device is silent, which would make reading hang.
    """
    def __init__(self, device_index, phrase_queue, status_cb, level_cb,
                 loopback_out: "LoopbackOutputThread | None" = None,
                 partial_slot: "LatestSlot | None" = None,
                 mute_evt: "threading.Event | None" = None,
                 early_slot: "LatestSlot | None" = None):
        """Store settings; the device is opened later, in the thread itself (run).

        device_index  which PortAudio device to record from.
        phrase_queue  where finished phrases are sent as (seq, audio, info).
        status_cb     function to show a status message in the window.
        level_cb      function that receives (rms_dbfs, peak_dbfs, is_speech).
        loopback_out  optional LoopbackOutputThread to copy audio to.
        partial_slot  where preview audio is placed (LatestSlot).
        mute_evt      while set, audio is replaced by silence (our own speech).
        """
        super().__init__(daemon=True, name="audio-capture")
        # device_index: which device to record (PortAudio's number for it).
        # _keepalive: the silence stream for loopback devices (see the class doc).
        # is_loopback: True when recording what an OUTPUT device plays.
        # failed: set to a reason if capture gives up, for the watchdog.
        # partial_slot / phrase_queue: where previews / finished phrases go.
        # status_cb / level_cb: callbacks to show status and the input level.
        # loopback_out: optional thread that plays the audio back (VB cables).
        # _stop_evt: set by stop(); vad_threshold: speech-probability threshold.
        self.device_index      = device_index
        self._keepalive        = None
        self.is_loopback       = False
        self.failed            = None   # set to a reason if capture gave up
        self.partial_slot      = partial_slot
        self.early_slot        = early_slot  # early (speculative) finals, see EARLY_FINAL_SEC
        self.mute_evt          = mute_evt   # set while our own TTS is audible on this device
        self.phrase_queue      = phrase_queue
        self.status_cb         = status_cb
        self.level_cb          = level_cb
        self.loopback_out      = loopback_out
        self._stop_evt         = threading.Event()
        self.vad_threshold     = vad_threshold_for(DEFAULT_SENS)

    def set_sensitivity(self, sensitivity: float):
        """Change the speech sensitivity (0..1) while running; takes effect at once."""
        self.vad_threshold = vad_threshold_for(sensitivity)

    def stop(self):
        """Ask the thread to stop; it finishes after its current audio block."""
        self._stop_evt.set()

    def _flush(self, seq, blocks, min_blocks, label="Phrase", resample=None, meta=None):
        """Send one finished phrase to the transcription worker.

        The audio is converted to 16 kHz (what Whisper needs). A phrase that is
        too short is still sent, but empty, so the window can clear its preview.
        """
        # Always emit, even an empty payload, so the UI can clear the live
        # preview line for this phrase.
        payload = b""
        if len(blocks) >= min_blocks:
            raw_audio     = b"".join(blocks)
            payload = resample(raw_audio) if resample else raw_audio
        try:
            meta = dict(meta or {}, reason=label, t_end=time.monotonic())
            self.phrase_queue.put((seq, payload, meta), timeout=0.25)
            backlog = self.phrase_queue.qsize()
            log_event(logging.INFO, "phrase_flushed",
                      reason=label, seq=seq, bytes=len(payload),
                      chunks=len(blocks), queue_backlog=backlog)
            if backlog >= PHRASE_QUEUE_BACKLOG_WARN:
                log_event(logging.WARNING, "phrase_queue_backlog_high", backlog=backlog)
        except queue.Full:
            log_event(logging.WARNING, "phrase_dropped_queue_full",
                      reason=label, bytes=len(payload), chunks=len(blocks))

    def run(self):
        """Thread entry point: create PortAudio, run the capture, always clean up."""
        pa     = pyaudio.PyAudio()
        try:
            self._run(pa)
        finally:
            if self._keepalive is not None:
                try:
                    self._keepalive.close()
                except Exception:
                    pass
            pa.terminate()

    def _run(self, pa):
        """The capture itself: open the device, then loop reading audio blocks.

        Steps: find the device's sample rate and channels, open the stream,
        then for each block: downmix to mono, measure level, run VAD, build
        phrases, send previews and finished phrases, and log stats every 10 s.
        """
        stream = None
        self._keepalive = None
        try:
            dev_info = (pa.get_device_info_by_index(self.device_index)
                        if self.device_index is not None and self.device_index >= 0 else {})
        except Exception:
            dev_info = {}
        self.is_loopback = bool(dev_info.get("isLoopbackDevice"))
        if self.is_loopback:
            self._keepalive = open_loopback_keepalive(pa, dev_info)
        # ── Open stream at the device's native sample rate ────────────────────
        # PortAudio on Windows will refuse to open a VB-Audio virtual cable
        # at 16000 Hz if the device is configured for 44100/48000 Hz in the
        # Windows audio properties.  We query the native rate first.
        # Probe actual openable sample rate for this device.
        # get_device_native_rate() gives the advertised rate, but PortAudio
        # may refuse it (e.g. DirectSound CABLE-A reports 44100 but only
        # accepts 44100; WASAPI CABLE-A reports 48000 and only accepts 48000).
        # Probe actual rate and channels the input device accepts.
        in_channels = 1
        open_rate   = 44100
        found = False
        # Get advertised max channels as a starting point
        try:
            info = pa.get_device_info_by_index(self.device_index) if self.device_index >= 0 else pa.get_default_input_device_info()
            max_ch = int(info.get("maxInputChannels", 1))
        except Exception:
            max_ch = 1
            
        for ch in ([max_ch, 2, 1] if max_ch > 0 else [2, 1]):
            for rate in [48000, 44100, 16000, 22050, 32000, 8000]:
                try:
                    pa.is_format_supported(
                        rate,
                        input_device=self.device_index if self.device_index >= 0 else None,
                        input_channels=ch,
                        input_format=pyaudio.paInt16,
                    )
                    open_rate   = rate
                    in_channels = ch
                    found = True
                    log_event(logging.INFO, "audio_probe_success", ch=ch, rate=rate)
                    break
                except ValueError as e:
                    # log_event(logging.DEBUG, "audio_probe_value_error", ch=ch, rate=rate, error=str(e))
                    continue
                except Exception as e:
                    log_event(logging.WARNING, "audio_probe_exception", ch=ch, rate=rate, error=str(e))
                    continue
            if found:
                break
        
        if not found:
            log_event(logging.WARNING, "audio_probe_failed_all", device_index=self.device_index)
            # Last-resort fallback to the device's advertised native format
            try:
                info = pa.get_device_info_by_index(self.device_index) if self.device_index >= 0 else pa.get_default_input_device_info()
                open_rate   = int(info.get("defaultSampleRate", 44100))
                in_channels = int(info.get("maxInputChannels", 1))
                if in_channels < 1:
                    in_channels = 1
            except Exception:
                open_rate   = 44100
                in_channels = 1
        
        chunk_sz  = int(open_rate * CHUNK_SAMPLES / SAMPLE_RATE) or CHUNK_SAMPLES
        self._open_rate  = open_rate
        self._in_channels = in_channels
        self._chunk_size = chunk_sz
        try:
            kwargs = dict(format=pyaudio.paInt16, channels=in_channels,
                          rate=open_rate, input=True,
                          frames_per_buffer=chunk_sz, start=True)
            if self.device_index is not None and self.device_index >= 0:
                kwargs["input_device_index"] = self.device_index
            stream = pa.open(**kwargs)
            log_event(logging.INFO, "audio_stream_opened",
                      device_index=self.device_index,
                      open_rate=open_rate, chunk_sz=chunk_sz)
        except Exception as e:
            if "Invalid audio channels" in str(e) or "Invalid device" in str(e):
                self.status_cb("ERROR: Audio device disconnected or changed. Please re-select it.")
            else:
                self.status_cb("ERROR opening audio device: " + str(e))
            log_exception("audio_stream_open_failed", e,
                          device_index=self.device_index, open_rate=open_rate)
            self.failed = f"could not open audio device: {e}"
            return

        # No calibration: speech is detected by Silero VAD, which judges
        # whether audio *sounds like speech* rather than how loud it is, so it
        # copes with any input level and with background noise or music.
        from faster_whisper.vad import get_vad_model
        vad_model = get_vad_model()

        # The settings near the top of the file are in seconds; the loop
        # counts audio blocks (~32 ms each), so convert seconds -> blocks.
        blocks_per_sec          = open_rate / chunk_sz                 # ~31 blocks per second
        silence_blocks_needed   = max(1, int(SILENCE_SEC * blocks_per_sec))       # pause that ends a phrase
        max_phrase_blocks       = int(MAX_PHRASE_SEC * blocks_per_sec)            # longest phrase before a cut
        cut_search_blocks       = int(CUT_SEARCH_SEC * blocks_per_sec)            # where to look for the cut
        preview_interval_blocks = max(1, int(PARTIAL_INTERVAL_SEC * blocks_per_sec))  # time between previews
        min_phrase_blocks       = max(1, int(MIN_PHRASE_SEC * blocks_per_sec))    # shorter = treat as noise
        min_preview_blocks      = max(1, int(PARTIAL_MIN_SEC * blocks_per_sec))   # audio needed for a preview
        vad_window_blocks       = max(1, int(VAD_WINDOW_SEC * blocks_per_sec))    # context given to the VAD
        preroll_blocks          = max(1, int(PREROLL_SEC * blocks_per_sec))       # kept from before speech
        early_blocks_needed     = max(1, int(EARLY_FINAL_SEC * blocks_per_sec))   # pause before an early final
        early_offer_len         = None  # phrase length (blocks) when an early final was offered

        recent_blocks = deque(maxlen=max(vad_window_blocks, preroll_blocks))  # last few blocks (a
                                            # deque with maxlen drops the oldest automatically)
        phrase_blocks, phrase_speech_probs = [], []  # the current phrase's audio blocks and the
                                                     # VAD speech score of each block
        chunk_sec     = chunk_sz / open_rate  # duration of one block, in seconds
        stats         = PerfWindow()          # timings for the capture_stats log line
        last_read_t   = None                  # when the previous block was read
        speech_frames = frames = 0            # blocks judged speech / all blocks (for stats)
        # Per-phrase measurements, logged with the phrase's trace:
        # when its speech started, the level of each block, its loudest peak,
        # and how many previews were made of it.
        phrase_start_time, phrase_levels_db, phrase_peak_db, phrase_preview_count = 0.0, [], -100.0, 0

        def phrase_meta(n_chunks):
            return {"t_start": phrase_start_time, "audio_sec": round(n_chunks * chunk_sec, 3),
                    "rms_dbfs": round(sum(phrase_levels_db) / len(phrase_levels_db), 1) if phrase_levels_db else None,
                    "peak_dbfs": round(phrase_peak_db, 1),
                    "vad_mean": round(sum(phrase_speech_probs) / len(phrase_speech_probs), 3) if phrase_speech_probs else None,
                    "vad_threshold": round(self.vad_threshold, 3), "previews": phrase_preview_count,
                    "early_blocks": early_offer_len, "onset_delay_ms": onset_delay_ms}
        silent_blocks        = 0      # quiet blocks in a row since the last speech
        in_speech            = False  # True while a phrase is being collected
        blocks_since_preview = 0      # blocks added since the last preview was offered
        seq                  = 0      # number of the current phrase (0, 1, 2 …)
        silence              = bytes(chunk_sz * SAMPLE_WIDTH)  # one block of digital silence
        possible_onset_t     = None   # when the speech score first rose above ONSET_PROB
        onset_delay_ms       = None   # how late the current phrase's start was detected
        # How often each speech score occurred (10 bins: 0.0-0.1 … 0.9-1.0), for
        # audible blocks only. Logged every 10 s; from it you can work out how
        # much audio would count as speech at ANY sensitivity, not just the
        # current one (log_tools.py sensitivity).
        prob_histogram       = [0] * 10
        self.status_cb("Listening…")

        # ── Resampler: captured at open_rate; Whisper needs SAMPLE_RATE (16 kHz)
        resampler = "none"
        if open_rate != SAMPLE_RATE:
            try:
                # soxr (python-soxr) is a C resampling library: ~3x faster
                # than scipy here (1.2 vs 3.6 ms for 7 s of audio), same quality.
                import soxr
                import numpy as _np_rs
                def _resample(raw: bytes) -> bytes:
                    arr = _np_rs.frombuffer(raw, dtype=_np_rs.int16)
                    return soxr.resample(arr, open_rate, SAMPLE_RATE, quality="HQ").tobytes()
                resampler = "soxr"
            except ImportError:
                resampler = None
        if open_rate != SAMPLE_RATE and resampler is None:
            try:
                from scipy.signal import resample_poly
                from fractions import Fraction
                import numpy as _np_rs
                rate_ratio = Fraction(SAMPLE_RATE, open_rate).limit_denominator(100)
                upsample_factor, downsample_factor = rate_ratio.numerator, rate_ratio.denominator
                def _resample(raw: bytes) -> bytes:
                    arr = _np_rs.frombuffer(raw, dtype=_np_rs.int16).astype(_np_rs.float32)
                    out = resample_poly(arr, upsample_factor, downsample_factor)
                    return out.clip(-32768, 32767).astype(_np_rs.int16).tobytes()
                log_event(logging.INFO, "resampler_scipy",
                          open_rate=open_rate, whisper_rate=SAMPLE_RATE, up=upsample_factor, down=downsample_factor)
            except ImportError:
                import numpy as _np_rs
                _ratio = max(1, open_rate // SAMPLE_RATE)
                def _resample(raw: bytes) -> bytes:
                    arr = _np_rs.frombuffer(raw, dtype=_np_rs.int16)
                    return arr[::_ratio].tobytes()
                log_event(logging.WARNING, "resampler_decimation",
                          open_rate=open_rate, whisper_rate=SAMPLE_RATE, ratio=_ratio)
        elif open_rate == SAMPLE_RATE:
            def _resample(raw: bytes) -> bytes:
                return raw
        if resampler == "soxr":
            log_event(logging.INFO, "resampler_soxr", open_rate=open_rate, whisper_rate=SAMPLE_RATE)

        import numpy as np

        def speech_prob() -> float:
            # Silero scores 512-sample (32 ms) frames at 16 kHz; a ~250 ms
            # window gives it context. ~0.4 ms per call on one CPU core.
            pcm_samples = np.frombuffer(_resample(b"".join(list(recent_blocks)[-vad_window_blocks:])), dtype=np.int16)
            n = (len(pcm_samples) // 512) * 512
            if n == 0:
                return 0.0
            t0 = time.perf_counter()
            out = vad_model(pcm_samples[-n:].astype(np.float32) / 32768.0)
            stats.add("vad_ms", (time.perf_counter() - t0) * 1000)
            return float(out[-2:].mean())

        next_keepalive_try = 0.0
        keepalive_tries = 0
        read_failures = 0
        while not self._stop_evt.is_set():
            # Without the keep-alive, a silent loopback delivers nothing and a
            # blocking read() would hang (and Stop would wait for audio). Poll
            # instead, and keep trying to restore the keep-alive.
            if self.is_loopback and self._keepalive is None:
                if time.monotonic() >= next_keepalive_try:
                    keepalive_tries += 1
                    # Log the failure once, not on every 1 s retry (that flooded
                    # the error log), then one line when it comes back.
                    self._keepalive = _open_loopback_keepalive(pa, dev_info, log_failure=False)
                    next_keepalive_try = time.monotonic() + 1.0
                    if self._keepalive is not None:
                        log_event(logging.INFO, "loopback_keepalive_restored",
                                  retries=keepalive_tries)
                try:
                    if stream.get_read_available() < chunk_sz:
                        time.sleep(0.005)
                        continue
                except Exception:
                    pass
            try:
                data = stream.read(chunk_sz, exception_on_overflow=False)
                now = time.monotonic()
                if last_read_t is not None:
                    # >> chunk duration means the loop fell behind the device
                    stats.add("read_gap_ms", (now - last_read_t) * 1000)
                last_read_t = now
                try:
                    stats.add("backlog_frames", stream.get_read_available())
                except Exception:
                    pass
                if in_channels > 1:
                    data = downmix(data, in_channels)
            except Exception as e:
                # A vanished device fails every read; logging each one flooded
                # the log ~100x a second. Give up after a run of failures and
                # let the UI watchdog reopen the device.
                read_failures += 1
                if read_failures == 1:
                    log_exception("audio_stream_read_failed", e)
                if read_failures >= MAX_READ_FAILURES:
                    self.failed = f"audio device stopped responding: {e}"
                    log_event(logging.ERROR, "audio_capture_failed",
                              device_index=self.device_index, failures=read_failures,
                              error=str(e))
                    self.status_cb("⚠ Audio device stopped responding — reconnecting…")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0
            loop_t0 = time.perf_counter()

            # Our own text-to-speech playing on the captured device must not
            # be transcribed as the friend speaking.
            if self.mute_evt is not None and self.mute_evt.is_set():
                data = silence[:len(data)]

            # ── Feed raw audio to loopback so headphones receive it ──────
            if self.loopback_out is not None:
                self.loopback_out.push(data)

            # ── Step 1: measure this block ───────────────────────────────
            # `recent_blocks` keeps the last few blocks (for VAD context and pre-roll).
            # speech_probability is Silero's estimate (0..1) that this block is speech;
            # digital silence is skipped to save work.
            recent_blocks.append(data)
            level_db, peak_db = levels_dbfs(data)
            speech_probability = speech_prob() if level_db > SILENCE_FLOOR_DBFS else 0.0
            speech_threshold  = self.vad_threshold
            # Hysteresis: easier to stay in speech than to enter it, so brief
            # dips inside words don't end the phrase.
            speaking = speech_probability >= (speech_threshold - VAD_HYSTERESIS if in_speech else speech_threshold)
            self.level_cb(level_db, peak_db, speaking, speech_probability, speech_threshold)
            if level_db > SILENCE_FLOOR_DBFS:
                prob_histogram[min(9, int(speech_probability * 10))] += 1
            # Onset tracking: note when speech *might* have begun, so the delay
            # until the detector is sure (it depends on the sensitivity) can be
            # measured. A late onset is what clips the first word.
            if not in_speech:
                if speech_probability >= ONSET_PROB:
                    possible_onset_t = possible_onset_t or time.monotonic()
                else:
                    possible_onset_t = None
            frames += 1
            speech_frames += speaking
            if level_db > -99:
                stats.add("level_dbfs", level_db)
            if stats.due():
                log_event(logging.INFO, "capture_stats", device_index=self.device_index,
                          chunk_ms=round(chunk_sec * 1000, 1),
                          speech_pct=round(100 * speech_frames / max(1, frames), 1),
                          vad_threshold=round(speech_threshold, 3),
                          speech_score_histogram=prob_histogram, **stats.summary())
                frames = speech_frames = 0
                prob_histogram = [0] * 10

            # ── Step 2: track whether we are inside a phrase ─────────────
            # in_speech = "a phrase has started and not ended yet".
            # silent_blocks = how many quiet blocks in a row since the last speech.
            if speaking:
                early_offer_len = None  # speech resumed: any early final is now out of date
                if not in_speech:
                    in_speech = True
                    # Pre-roll: VAD needs a frame or two to fire, so include
                    # the audio just before it did or the first word is clipped.
                    preroll = list(recent_blocks)[-preroll_blocks:-1]
                    phrase_blocks, phrase_speech_probs = preroll, [speech_threshold] * len(preroll)
                    blocks_since_preview = len(phrase_blocks)
                    phrase_start_time = time.monotonic() - len(preroll) * chunk_sec
                    onset_delay_ms = (round((time.monotonic() - possible_onset_t) * 1000)
                                      if possible_onset_t else 0)
                    phrase_levels_db, phrase_peak_db, phrase_preview_count = [], -100.0, 0
                    self.status_cb("🎙 Speech…")
                silent_blocks = 0
            elif in_speech:
                silent_blocks += 1
            if not in_speech:
                continue

            # ── Step 3: add the block to the phrase, then decide what to do ─
            # phrase_blocks holds the phrase's audio blocks; phrase_speech_probs
            # their speech scores.
            phrase_blocks.append(data)
            phrase_speech_probs.append(speech_probability)
            phrase_levels_db.append(level_db)
            phrase_peak_db = max(phrase_peak_db, peak_db)
            blocks_since_preview += 1
            # (a) a long enough pause -> the phrase is finished
            if silent_blocks >= silence_blocks_needed:
                self._flush(seq, phrase_blocks, min_phrase_blocks, "end", _resample, phrase_meta(len(phrase_blocks)))
                seq += 1
                phrase_blocks, phrase_speech_probs, silent_blocks, blocks_since_preview, in_speech = [], [], 0, 0, False
                early_offer_len = None
            # (a2) the pause has just begun -> offer an early final (the worker
            #      transcribes it now; used only if the pause continues)
            elif (silent_blocks == early_blocks_needed and self.early_slot is not None
                    and len(phrase_blocks) >= min_phrase_blocks):
                early_offer_len = len(phrase_blocks)
                self.early_slot.put((seq, _resample(b"".join(phrase_blocks)), early_offer_len))
            # (b) the phrase is too long (people who never pause) -> cut it
            elif len(phrase_blocks) >= max_phrase_blocks:
                # Cut where speech is least likely (a word gap) near the end,
                # carrying the remainder into the next phrase.
                start = len(phrase_speech_probs) - cut_search_blocks
                cut   = start + min(range(cut_search_blocks), key=lambda i: phrase_speech_probs[start + i])
                early_offer_len = None  # an early final never applies to a forced cut
                self._flush(seq, phrase_blocks[:cut + 1], min_phrase_blocks, "max", _resample,
                            phrase_meta(cut + 1))
                seq += 1
                phrase_blocks, phrase_speech_probs = phrase_blocks[cut + 1:], phrase_speech_probs[cut + 1:]
                blocks_since_preview = len(phrase_blocks)
                phrase_start_time = time.monotonic() - len(phrase_blocks) * chunk_sec
                phrase_levels_db, phrase_peak_db, phrase_preview_count = phrase_levels_db[cut + 1:], -100.0, 0
            # (c) still talking -> every PARTIAL_INTERVAL_SEC, offer a preview
            elif (blocks_since_preview >= preview_interval_blocks and len(phrase_blocks) >= min_preview_blocks and silent_blocks == 0
                    and self.partial_slot is not None):
                # silent_blocks == 0: once a pause starts the final is ~0.3 s away;
                # a preview started now would hold the GPU and delay it.
                self.partial_slot.put((seq, _resample(b"".join(phrase_blocks)),
                                       time.monotonic(), phrase_start_time))
                phrase_preview_count += 1
                blocks_since_preview = 0
            stats.add("loop_ms", (time.perf_counter() - loop_t0) * 1000)

        if phrase_blocks:
            self._flush(seq, phrase_blocks, min_phrase_blocks, "final", _resample, phrase_meta(len(phrase_blocks)))
        if stream is not None:
            stream.close()
        log_event(logging.INFO, "audio_capture_stopped", open_rate=open_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Persisted user settings
#
# The user's choices are saved in interpreter.settings.json so they are
# remembered the next time the app starts.
# ─────────────────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    """Read interpreter.settings.json and return it as a dict ({} if missing or broken)."""
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        log_exception("settings_load_failed", e, path=str(SETTINGS_PATH))
        return {}


def save_settings(**changes):
    """Update some settings (e.g. save_settings(model="…")) and write the file."""
    settings = load_settings()
    settings.update(changes)
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception as e:
        log_exception("settings_save_failed", e, path=str(SETTINGS_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Whisper model loader — GPU with automatic CPU fallback
#
# Loading a model takes seconds, so it happens on a background thread.
# ─────────────────────────────────────────────────────────────────────────────

class WhisperModelLoader(threading.Thread):
    """Loads a Whisper model on a background thread so the window stays usable.

    Loading can take seconds (or minutes on the first download). When done,
    `ready` is set and `model` (or `error`) is filled in. If loading on the
    GPU fails it falls back to the Tiny model on the CPU.
    """
    def __init__(self, model_size, status_cb):
        """Prepare to load `model_size` ("tiny", "small", "medium", "large-v3-turbo")."""
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
        kwargs["cpu_threads"] = cores
            
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
                self.status_cb("GPU load failed — falling back to CPU (Tiny model)…")
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

# Whisper's stock outputs on silence/noise (learned from YouTube captions).
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "please subscribe", "subscribe", "you", "bye",
    "hmm", "um", "uh", "oh", "so",
}


# Short stock phrases Whisper emits in place of real speech it failed to
# decode (seen in the logs: "Спасибо." for 11-12 spoken words). They are real
# words, so they are only treated as suspect when far too short for the audio.
SUSPECT_PHRASES = {"спасибо", "спасибо за внимание", "продолжение следует",
                   "субтитры сделал dimatorzok", "редактор субтитров а синецкая",
                   "thank you", "thanks for watching"}
MIN_CHARS_PER_SEC = 5.0   # good phrases ran 9-17 chars/s in the logs; at 5 the check
                          # caught both real losses in 52 phrases (2 false alarms)
MAX_COMPRESSION   = 2.4   # Whisper's own repetition-loop check


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


# Fragments that only ever come from Whisper's training captions (YouTube
# subtitle credits), never from real speech; any text containing one is dropped.
HALLUCINATION_FRAGMENTS = ("субтитр", "dimatorzok", "редактор субтитров", "корректор",
                           "amara.org", "subtitles by")


def is_hallucination(text: str) -> bool:
    """True if the text is empty, a known stock phrase Whisper invents on silence
    ("thank you", "you"…), or contains a YouTube subtitle-credit fragment.
    """
    text_lower = text.lower()
    if any(f in text_lower for f in HALLUCINATION_FRAGMENTS):
        return True
    normalized = "".join(ch for ch in text_lower if ch.isalnum() or ch == " ").strip()
    return not normalized or " ".join(normalized.split()) in HALLUCINATIONS


def plain_device_name(label: str) -> str:
    """'🔊 Speakers (FiiO Q series) [Loopback]' -> 'Speakers (FiiO Q series)'."""
    import re
    return re.sub(r"^[^\w(]+", "", label).replace(" [Loopback]", "").strip()


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
    """Segmented LED-style level meter in dBFS, after broadcast PPMs:
    -60..0 dBFS, green below -18, amber to -6, red above; instant attack,
    20 dB/s fall-back, and a peak-hold marker that holds 1.5 s. Segments are
    drawn once and only recoloured, so updating costs almost nothing."""
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


class LocalTranslator:
    """Opus-MT translation model for one language pair. Text is split into
    sentences (Opus-MT is trained on sentences; fragments translate badly)
    and translated as one batch."""
    def __init__(self, src, tgt):
        """Download (first time) and load the Opus-MT model for (src, tgt).

        `sentencepiece` splits text into the sub-word pieces the model was trained
        on; there is one tokenizer for the source and one for the target language.
        """
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download
        # pair: (source, target) language codes, e.g. ("ru", "en").
        # _translator: the Opus-MT model running on CTranslate2.
        # _src_tokenizer / _tgt_tokenizer: split text into, and join it back from,
        # the sub-word pieces the model uses (sentencepiece).
        self.pair = (src, tgt)
        path = snapshot_download(OPUS_MT_REPOS[(src, tgt)])
        device = "cuda" if WHISPER_DEVICE == "cuda" else "cpu"
        self._translator = ctranslate2.Translator(
            path, device=device, compute_type="float16" if device == "cuda" else "int8")
        self._src_tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(path, "source.spm"))
        self._tgt_tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(path, "target.spm"))

    def translate(self, text: str) -> str:
        """Translate `text`: split into sentences, translate them as one batch on the
        GPU, and join the results.
        """
        import re
        sentences = [x for x in re.split(r"(?<=[.!?…])\s+", text.strip()) if x]
        if not sentences:
            return ""
        batch = [self._src_tokenizer.encode(x, out_type=str) + ["</s>"] for x in sentences]
        with INFERENCE_LOCK:
            results = self._translator.translate_batch(batch, beam_size=2, max_decoding_length=256)
        import html
        # Opus-MT models were trained on text with HTML-escaped punctuation, so
        # they sometimes output "&apos;" for ' or "&quot;" for ": unescape it.
        return " ".join(html.unescape(self._tgt_tokenizer.decode(
            [t for t in r.hypotheses[0] if t != "</s>"]))
                        for r in results).strip()


class TranslatorLoader(threading.Thread):
    """Loads a LocalTranslator on a background thread (first use downloads ~300 MB).

    Until it is ready the app uses Whisper or online translation instead.
    """
    def __init__(self, src, tgt):
        """Prepare to load the translator for the language pair (src, tgt)."""
        super().__init__(daemon=True, name=f"mt-loader-{src}-{tgt}")
        # src/tgt: language pair; translator: the loaded LocalTranslator (or None
        # if loading failed); ready: set when loading finished either way.
        self.src, self.tgt = src, tgt
        self.translator = None
        self.ready = threading.Event()

    def run(self):
        """Load the model, translate one word to warm it up, then signal `ready`."""
        t0 = time.perf_counter()
        try:
            self.translator = LocalTranslator(self.src, self.tgt)
            self.translator.translate("Hello." if self.src == "en" else "Привет.")  # warm-up
            log_event(logging.INFO, "local_translator_ready", pair=f"{self.src}-{self.tgt}",
                      elapsed_ms=round((time.perf_counter() - t0) * 1000))
        except Exception as e:
            log_exception("local_translator_failed", e, pair=f"{self.src}-{self.tgt}")
        finally:
            self.ready.set()


class SpeechMeter(ctk.CTkCanvas):
    """The speech detector's score (0-100 %) as a bar, with a white line at
    the threshold set by the Speech sensitivity slider. While the bar passes
    the line, the audio counts as speech (bar turns green).

    This is a different thing from the level meter above it: the level meter
    shows how LOUD the sound is; this shows how much it SOUNDS LIKE SPEECH.
    Sensitivity moves the threshold line, never the volume."""
    TICKS = (0, 25, 50, 75, 100)

    def __init__(self, master, **kw):
        self._s = ui_scale(master)
        super().__init__(master, height=int(34 * self._s), highlightthickness=0,
                         bg="#1d1e1e", **kw)
        self._shown, self._threshold, self._speaking = 0.0, 0.5, False
        self.bind("<Configure>", lambda e: self._layout())

    def _layout(self):
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
        if not hasattr(self, "_bar"):
            return
        sc, span = self._s, self._x1 - self._x0
        self.coords(self._bar, self._x0, 3 * sc, self._x0 + span * self._shown, 19 * sc)
        self.itemconfigure(self._bar, fill="#2ECC71" if self._speaking else "#2980B9")
        x = self._x0 + span * self._threshold
        self.coords(self._mark, x, 0, x, 22 * sc)


class Speaker(threading.Thread):
    """Text-to-speech via Windows SAPI, owned by one thread (COM objects are
    apartment-bound). Speech is asynchronous, so stop() purges it instantly,
    unlike pyttsx3's blocking runAndWait(). `speaking` is set while audio is
    playing so the capture thread can ignore our own voice."""
    SVSF_ASYNC, SVSF_PURGE = 1, 2
    HANGOVER_SEC = 0.4   # loopback capture lags playback slightly

    def __init__(self):
        """Set up the command queue; the SAPI voice is created inside the thread."""
        super().__init__(daemon=True, name="tts")
        # _commands: queue of commands for the speech thread; speaking: set while
        # audio is playing; ready: set once the voice is created; error: last
        # problem (e.g. no voice for the language), shown in the status bar.
        self._commands    = queue.Queue()
        self.speaking = threading.Event()
        self.ready    = threading.Event()
        self.error    = None

    def configure(self, output_label: str, language: str):
        """Choose the output device and a voice for `language` (e.g. "English")."""
        self._commands.put(("config", output_label, language))

    def say(self, text: str):
        """Queue text to be spoken (does not wait for it to finish)."""
        self._commands.put(("say", text))

    def stop_speech(self):
        """Stop speaking immediately and drop anything queued."""
        self._commands.put(("stop",))

    def shutdown(self):
        """Stop speaking and end the thread (when the app closes)."""
        self._commands.put(("quit",))

    def _configure(self, voice, output_label, language):
        """Apply an output device and language to the SAPI voice.

        Returns an error message if no voice for the language is installed.
        """
        want_out = plain_device_name(output_label).lower()
        outputs  = voice.GetAudioOutputs()
        for i in range(outputs.Count):
            tok = outputs.Item(i)
            if tok.GetDescription().lower() == want_out:
                voice.AudioOutput = tok
                break
        voices = voice.GetVoices()
        chosen = None
        for i in range(voices.Count):
            tok = voices.Item(i)
            if language.lower() in tok.GetDescription().lower():
                chosen = tok
                break
        if chosen is not None:
            voice.Voice = chosen
        log_event(logging.INFO, "tts_configured", output=want_out, language=language,
                  voice=chosen.GetDescription() if chosen else None)
        if chosen is None:
            return f"No {language} text-to-speech voice installed (Windows Settings → Speech)."
        return None

    def run(self):
        """Thread loop: carry out queued commands and keep `speaking` up to date."""
        try:
            # Create the Windows speech engine (a SAPI 'SpVoice' COM object).
            # CoInitialize prepares this thread to use COM; COM objects must be
            # used from the thread that created them, which is why all speaking
            # happens on this one thread, driven by commands from a queue.
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except Exception as e:
            self.error = str(e)
            log_exception("tts_init_failed", e)
            self.ready.set()
            return
        self.ready.set()
        # Main loop: wait up to 50 ms for a command, then carry it out.
        # Commands are tuples: ('say', text), ('stop',), ('config', device,
        # language) or ('quit',).
        quiet_since = None
        while True:
            try:
                cmd = self._commands.get(timeout=0.05)
            except queue.Empty:
                cmd = None
            try:
                if cmd is None:
                    pass
                # SVSF_ASYNC = start speaking and return at once (don't wait);
                # SVSF_PURGE = throw away anything still being said or queued.
                # Speaking an empty string with PURGE is SAPI's way to stop instantly.
                elif cmd[0] == "quit":
                    voice.Speak("", self.SVSF_ASYNC | self.SVSF_PURGE)
                    break
                elif cmd[0] == "stop":
                    voice.Speak("", self.SVSF_ASYNC | self.SVSF_PURGE)
                elif cmd[0] == "say":
                    voice.Speak(cmd[1], self.SVSF_ASYNC)
                    self.speaking.set()
                elif cmd[0] == "config":
                    self.error = self._configure(voice, cmd[1], cmd[2])
                # Keep the `speaking` flag up to date. The capture thread reads it to
                # ignore our own voice when it plays on the device being captured. It
                # stays set for HANGOVER_SEC after speech ends, because the recorded
                # audio lags slightly behind what is played.
                # RunningState 2 = SRSEIsSpeaking
                if voice.Status.RunningState == 2:
                    self.speaking.set()
                    quiet_since = None
                elif self.speaking.is_set():
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > self.HANGOVER_SEC:
                        self.speaking.clear()
            except Exception as e:
                log_exception("tts_command_failed", e, command=cmd[0] if cmd else None)
        self.speaking.clear()


class TranscriptionWorker(threading.Thread):
    """Finished phrases (phrase_queue) always take priority; live previews
    (partial_slot) only run when the model would otherwise sit idle.

    When the target is English, Whisper translates directly (task="translate")
    on the GPU: no network round trip, no rate limits, and it makes live
    previews of the *translation* cheap enough to show while the friend talks.
    """
    def __init__(self, wid, model, src_whisper, src_lang_key, tgt_lang_key,
                 phrase_queue, result_queue, status_cb,
                 partial_slot: "LatestSlot | None" = None,
                 preview_model=None, run_id=0, record_audio=False, translator=None,
                 early_slot: "LatestSlot | None" = None, session_id: str = "",
                 config: "dict | None" = None):
        """Store everything the worker needs for one listening session.

        model          the main Whisper model (used for finished phrases).
        src_whisper    language code of the speech, e.g. "ru".
        phrase_queue   finished phrases from AudioCaptureThread.
        result_queue   where results go back to the window.
        partial_slot   preview audio (LatestSlot).
        preview_model  smaller model for previews, or a function returning one.
        translator     Opus-MT translator, or a function returning one (or None).
        record_audio   save each phrase as a WAV for later accuracy checks.
        """
        super().__init__(daemon=True, name=f"transcription-worker-{wid}")
        # wid: worker number; model: main Whisper model; model_lock: the shared
        # INFERENCE_LOCK; src_whisper: speech language code ("ru"); src/tgt_lang_key:
        # language menu names; phrase_queue / result_queue: input / output;
        # status_cb: shows status messages.
        self.wid          = wid
        self.model        = model
        self.model_lock   = INFERENCE_LOCK
        self.src_whisper  = src_whisper
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
        self._early       = None            # (seq, phrase blocks, recognise() result, ms) kept
                                            # until the matching final arrives
        self.record_audio = record_audio
        self._stats       = PerfWindow()
        self._stop_evt    = threading.Event()
        tgt_code          = LANGUAGES.get(tgt_lang_key, ("en", "en"))[0]
        # Whisper can translate to English itself (not the turbo model);
        # a local Opus-MT translator, when loaded, is preferred for any pair.
        self._whisper_can_translate = (tgt_code == "en" and src_whisper != "en")
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
            old = sorted(RECORD_DIR.glob("*.wav"))[:-RECORD_KEEP]
            for f in old:
                f.unlink(missing_ok=True)
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

    def _mt(self, text):
        """Translate with Opus-MT; returns (text, ms) or (None, 0) if unavailable."""
        translator = self.translator
        if translator is None or not text:
            return None, 0.0
        t0 = time.perf_counter()
        try:
            out = translator.translate(text)
        except Exception as e:
            log_exception("local_translation_failed", e, chars=len(text))
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

    def _decode(self, audio_np, tasks, model=None, beam_size=1):
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
            t = time.perf_counter()
            results = model.model.generate(
                enc, self._prompts_for(tasks, model),
                beam_size=beam_size, max_length=224, return_no_speech_prob=True,
                return_scores=True, suppress_blank=True)
            info["decode_ms"] = round((time.perf_counter() - t) * 1000, 2)
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
        # Accept the retry only if it is real text, no shorter than the
        # original, and less suspect (or equally suspect but more confident).
        # A looser rule let a retry swap the correct "с друзьями" for
        # "Субтитры делал DimaTorzok" because that scored as more confident.
        better = (bool(retry_texts[0]) and not is_hallucination(retry_texts[0])
                  and len(retry_texts[0]) >= len(texts[0])
                  and (len(retry_reasons) < len(reasons)
                       or (len(retry_reasons) == len(reasons) and retry_logprob > logprob)))
        info["retry"] = {"text": retry_texts[0], "avg_logprob": round(retry_logprob, 3),
                         "reasons": retry_reasons, "used": better,
                         "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "original_text": texts[0]}
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
        model_key = id(model)
        if model_key not in self._special:
            tokenizer = model.hf_tokenizer
            token_id_of = lambda t: tokenizer.token_to_id(t)
            head = [token_id_of("<|startoftranscript|>"), token_id_of(f"<|{self.src_whisper}|>")]
            tail = [token_id_of("<|notimestamps|>")]
            self._special[model_key] = {
                "prompts": {"transcribe": head + [token_id_of("<|transcribe|>")] + tail,
                            "translate": head + [token_id_of("<|translate|>")] + tail},
                "sot_prev": token_id_of("<|startofprev|>"),
                "tokenizer": tokenizer,
                "text_vocab": token_id_of("<|endoftext|>"),   # ids below this are plain text
            }
        return self._special[model_key]

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
                  run_id=self.run_id, config=self.config,
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
                          run_id=self.run_id,
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
                whisper_translates = rec["whisper_translates"]
                text, translated = rec["text"], rec["translated"]
                engine, mt_ms = rec["engine"], rec["mt_ms"]
                lock_wait_ms = info.get("lock_wait_ms", 0.0)
                # Text still suspect after a retry is not fed back as context:
                # that is how one "Спасибо." turned the next phrase into another.
                if text and not info.get("still_suspicious"):
                    self._remember(tasks, token_ids,
                                   [text, translated] if whisper_translates else [text])
                # ── Step 6: record everything about this phrase in one log line
                #    (phrase_trace): timings of each stage, levels, confidence, the text
                #    itself and the saved audio file. log_tools.py analyses these lines.
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                t_result = time.monotonic()
                trace = {
                    "session_id": self.session_id,
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
                trace["cfg_preview_model"] = getattr(self.preview_model or self.model,
                                                     "_size_name", None)
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
        texts, info = self._decode(audio_np, tasks)
        token_ids = info.pop("_token_ids", [])
        texts, token_ids, info = self._retry_if_suspicious(
            audio_np, tasks, texts, token_ids, info)
        text       = texts[0]
        translated = texts[1] if whisper_translates else None
        engine     = "whisper" if whisper_translates else "online"
        mt_ms      = 0.0
        if is_hallucination(text):
            log_event(logging.INFO, "transcription_discarded", worker_id=self.wid, text=text)
            text = ""
            translated = "" if whisper_translates else None
        elif whisper_translates and is_hallucination(translated):
            translated = ""
        elif not whisper_translates:
            # None from _mt means "no local translator": the window will
            # then send the text to the online translator.
            out, mt_ms = self._mt(text)
            if out is not None:
                translated, engine = out, "opus-mt"
        return {"tasks": tasks, "token_ids": token_ids, "info": info,
                "whisper_translates": whisper_translates, "text": text,
                "translated": translated, "engine": engine, "mt_ms": mt_ms}

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

    """The main window, and the "conductor" that starts and stops everything.

    It builds the controls, starts a listening "session" (one capture thread,
    one transcription worker, one translation worker) and restarts it whenever
    a setting changes. Background threads never touch the window directly:
    they put messages on `result_queue`, and _poll_queue() (on the main
    thread) reads them every 15 ms and updates the screen.
    """
    def __init__(self):
        """Create the window: load settings, find audio devices, build all the
        controls, place the window on screen, start loading the AI models and
        start listening.
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
        self.phrase_queue = queue.Queue()
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
        self._translation_queue   = queue.Queue(maxsize=TRANSLATION_QUEUE_MAX)
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

        self._apply_mode(0, switch_source=True)
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
        lang_names = sorted(LANGUAGES.keys())
        self.src_menu = ctk.CTkOptionMenu(
            language_row, variable=self.src_var, values=lang_names, width=155,
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
        self.level_text = ctk.CTkLabel(meter_frame, text="—",
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
        self.speech_text = ctk.CTkLabel(meter_frame, text="—",
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
        self.heard_box = ctk.CTkTextbox(interp_tab, wrap="word", height=120,
                                        font=ctk.CTkFont(size=14))
        self.heard_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.translated_box = ctk.CTkTextbox(interp_tab, wrap="word", height=120,
                                             font=ctk.CTkFont(size=14))
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
        ctk.CTkButton(status_bar, text="Export Log Summary", width=150,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._export_log_summary).grid(row=0, column=1, padx=(10, 0))

    # ── Level meter ───────────────────────────────────────────────────────

    def _on_level(self, rms_db, peak_db, speaking, prob=0.0, threshold=0.5):
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

    def _on_mode_btn(self, mode_idx):
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

    def _on_manual_device_change(self, _=None):
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
        self.status_var.set(f"Refreshed — {len(in_labels)} in / {len(out_labels)} out device(s).")
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
            self.status_var.set("Muted — not listening.")
        else:
            self.mute_btn.configure(text="🔇  Mute", fg_color=["#3B8ED0", "#1F6AA5"],
                                    hover_color=["#3281BC", "#1A5C8F"])
            self._start_session()
        log_event(logging.INFO, "mute_toggled", muted=self._muted)

    def _start_session(self):
        """Start listening: create the capture thread, transcription worker and
        translation worker for the current device, languages and models.

        If the Whisper model is still loading, it tries again every 0.6 s.
        """
        if self.is_listening or self._muted:
            return
        # ── Step 1: make sure the Whisper model is loaded ────────────────
        # If it is still loading, show a message and try again in 0.6 s.
        if self._model_loader and not self._model_loader.ready.is_set():
            self.status_var.set("⏳ Loading Whisper model — please wait…")
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

        self.is_listening = True
        self._set_live_status("LISTENING", COL_OK)
        self.status_var.set("Listening…")

        # ── Step 2: work out the device, languages and queues ────────────
        # in_idx is PortAudio's number for the selected input device.
        in_idx = self._input_map.get(self.device_var.get(), -1)
        if in_idx == -1:
            in_idx = None
        log_event(logging.INFO, "audio_device_selected",
                  device_label=self.device_var.get(), device_index=in_idx)
        if self._active_mode is not None:
            self._mode_state[self._active_mode][0] = self.device_var.get()
        save_settings(mode_devices=[m[0] for m in self._mode_state])
        capturing_output = "[loopback]" in self.device_var.get().lower()

        src_key     = self.src_var.get()
        tgt_key     = self.tgt_var.get()
        src_whisper, _ = LANGUAGES.get(src_key, ("en", "en"))

        # phrase_queue carries finished phrases (capture -> worker);
        # partial_slot carries only the newest preview audio.
        worker_count = getattr(self._model_loader, "worker_count", NUM_WORKERS)
        self.phrase_queue  = queue.Queue(maxsize=max(worker_count * 4, 8))
        self._partial_slot = LatestSlot()
        self._early_slot   = LatestSlot()   # early finals (EARLY_FINAL_SEC)
        # Everything that can change speed or accuracy, recorded once per
        # session (session_config) and copied into every phrase_trace, so
        # results can be compared by setting (log_tools.py compare).
        loader = self._model_loader
        session_config = {
            "model": MODELS.get(self.model_var.get()),
            "compute_type": getattr(loader, "active_compute_type", COMPUTE_TYPE),
            "device": getattr(loader, "active_device", WHISPER_DEVICE),
            "preview_choice": self.preview_choice_var.get(),
            "sensitivity": round(self.sensitivity_var.get(), 2),
            "src_lang": self.src_var.get(), "tgt_lang": self.tgt_var.get(),
            "speak": self._tts_on,
        }
        self._live         = {}
        # Spoken translations played on the device we're capturing would be
        # heard as the friend; mute capture while the speaker is talking.
        same_device = (capturing_output and plain_device_name(self.device_var.get())
                       == plain_device_name(self.out_device_var.get()))
        self._mute_evt = self.speaker.speaking if same_device else None
        if self._tts_on:
            self._configure_speaker()

        # ── Optional loopback output (VB-Audio → Fiio headphones) ────────
        # Look up the native sample rate of the input device so the loopback
        # output thread opens at the same rate and PortAudio accepts it.
        _pa_tmp = pyaudio.PyAudio() if _PYAUDIO_AVAILABLE else None
        in_native_rate = get_device_native_rate(_pa_tmp, in_idx) if _pa_tmp else SAMPLE_RATE
        if _pa_tmp:
            _pa_tmp.terminate()
        self._loopback_out = None
        if self.loopback_var.get() and capturing_output:
            # The audio is already playing on that device; replaying it would
            # double it up (or feed back if it's the same device).
            log_event(logging.INFO, "loopback_skipped_output_capture",
                      device_label=self.device_var.get())
        elif self.loopback_var.get():
            out_lbl = self.out_device_var.get()
            out_idx = self._output_map.get(out_lbl, -1)
            if out_idx == -1:
                out_idx = None
            self._loopback_out = LoopbackOutputThread(
                output_device_index=out_idx,
                input_sample_rate=in_native_rate,
                status_cb=lambda m: self.result_queue.put(("status", m)))
            self._loopback_out.start()
            log_event(logging.INFO, "loopback_started",
                      output_label=out_lbl, output_index=out_idx,
                      input_native_rate=in_native_rate)

        # Opening a new PortAudio instance while the previous capture thread is
        # still terminating its own fails with "Unanticipated host error".
        prev_capture = getattr(self, "_stopping_capture", None)
        if prev_capture is not None and prev_capture.is_alive():
            prev_capture.join(timeout=1.0)
        self._stopping_capture = None
        self._capture_thread = AudioCaptureThread(
            device_index=in_idx,
            phrase_queue=self.phrase_queue,
            status_cb=lambda m: self.result_queue.put(("status", m)),
            level_cb=self._on_level,
            loopback_out=self._loopback_out,
            partial_slot=self._partial_slot,
            mute_evt=self._mute_evt,
            early_slot=self._early_slot,
        )
        self._capture_thread.set_sensitivity(round(self.sensitivity_var.get(), 2))
        self._capture_thread.start()

        # ── Translation workers (TRANSLATION_WORKERS parallel) ────────────
        self._translation_workers = []
        for wid in range(TRANSLATION_WORKERS):
            translation_worker = TranslationWorker(wid, self._translation_queue, self.result_queue)
            translation_worker.start()
            self._translation_workers.append(translation_worker)

        # ── Transcription workers ─────────────────────────────────────────
        self._workers = []
        for wid in range(worker_count):
            worker = TranscriptionWorker(
                wid=wid, model=self._whisper_model,
                src_whisper=src_whisper,
                src_lang_key=src_key, tgt_lang_key=tgt_key,
                phrase_queue=self.phrase_queue,
                result_queue=self.result_queue,
                status_cb=lambda m: self.result_queue.put(("status", m)),
                partial_slot=self._partial_slot,
                preview_model=self._preview_model,
                run_id=self._run_id,
                record_audio=bool(load_settings().get("record_audio", True)),
                translator=self._translator_for(src_whisper,
                                                LANGUAGES.get(tgt_key, ("en", "en"))[0]),
                early_slot=self._early_slot,
                session_id=self.session_id,
                config=session_config)
            worker.start()
            self._workers.append(worker)

        log_event(logging.INFO, "session_config", run_id=self._run_id,
                  session_id=self.session_id, device_label=self.device_var.get(),
                  output_label=self.out_device_var.get(), **session_config,
                  # the tuning values from the top of the file, in case they change
                  tuning={"silence_sec": SILENCE_SEC, "max_phrase_sec": MAX_PHRASE_SEC,
                          "preview_interval_sec": PARTIAL_INTERVAL_SEC,
                          "preroll_sec": PREROLL_SEC, "early_final_sec": EARLY_FINAL_SEC,
                          "context_tokens": CONTEXT_TOKENS},
                  app_version=APP_VERSION)
        log_event(logging.INFO, "session_started",
                  session_id=self.session_id,
                  device_label=self.device_var.get(), device_index=in_idx,
                  src_lang=src_key, tgt_lang=tgt_key,
                  model=self.model_var.get(),
                  sensitivity=round(self.sensitivity_var.get(), 2),
                  worker_count=len(self._workers),
                  translation_workers=len(self._translation_workers),
                  loopback=self.loopback_var.get(),
                  model_device=getattr(self._model_loader, "active_device", WHISPER_DEVICE),
                  model_compute_type=getattr(self._model_loader, "active_compute_type", COMPUTE_TYPE))

    def _retry_start_session(self):
        """Timer callback: try _start_session again (model was still loading)."""
        self._start_retry_after_id = None
        self._start_session()

    def _stop_session(self, keep_mode=False):
        """Stop listening: tell all the session's threads to finish and reset the
        meter. `keep_mode` is True when the stop is part of a restart.
        """
        self.is_listening = False
        if self._start_retry_after_id is not None:
            self.after_cancel(self._start_retry_after_id)
            self._start_retry_after_id = None
        if self._capture_thread:
            self._capture_thread.stop()
            self._stopping_capture = self._capture_thread
            self._capture_thread = None
        if self._loopback_out:
            self._loopback_out.stop()
            self._loopback_out = None
        for worker in self._workers:
            worker.stop()
        self._workers = []
        for translation_worker in self._translation_workers:
            translation_worker.stop()
        self._translation_workers = []
        self.meter.reset()
        self.level_text.configure(text="—")
        self._set_live_status("IDLE", COL_WARN)
        log_event(logging.INFO, "session_stopped",
                  session_id=self.session_id, keep_mode=keep_mode)
        if not keep_mode:
            self.status_var.set("Stopped.")

    # ── TTS — fixed: lock prevents "run loop already started" ────────────

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

    def _ensure_preview_model(self):
        """Load the chosen preview model alongside the main model. Not needed
        when it is the same as the main model."""
        size = PREVIEW_CHOICES[self.preview_choice_var.get()]
        # Plain mirror of the Tk settings: _preview_model() runs on the worker
        # thread, and Tk variables may only be touched from the UI thread.
        self._preview_enabled = size is not None and size != MODELS[self.model_var.get()]
        if not self._preview_enabled:
            return
        if self._preview_loader is None or self._preview_loader.model_size != size:
            self._preview_loader = WhisperModelLoader(model_size=size, status_cb=lambda m: None)
            self._preview_loader.start()
            log_event(logging.INFO, "preview_model_load_started", model_size=size)

    def _translator_for(self, src, tgt):
        """Start loading the Opus-MT model for this pair (first use downloads
        ~300 MB) and return a callable the worker polls; until it is ready,
        translation falls back to Whisper/online."""
        if (src, tgt) not in OPUS_MT_REPOS:
            return None
        loader = self._mt_loaders.get((src, tgt))
        if loader is None:
            loader = self._mt_loaders[(src, tgt)] = TranslatorLoader(src, tgt)
            loader.start()
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

    def _log_displayed(self, seq, trace, online):
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
                    self.status_var.set(f"⚠ {problem} — restarting listening…")
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
        responsiveness, so slowdowns can be tied to resource limits."""
        # Collect figures into `fields`, then log them all as one line.
        # Each measurement is wrapped in try/except: if a library is missing or
        # fails, the other figures are still logged.
        fields = {}
        try:
            # CPU and memory, from psutil. cpu_percent() measures since its previous
            # call, so the very first call only starts the measurement. 100 % =
            # one CPU core fully busy. rss = memory the app is using.
            import psutil
            if self._psutil_process is None:
                self._psutil_process = psutil.Process()
                self._psutil_process.cpu_percent(None)
            fields.update(
                process_cpu_pct=round(self._psutil_process.cpu_percent(None), 1),
                system_cpu_pct=psutil.cpu_percent(None),
                rss_mb=round(self._psutil_process.memory_info().rss / 2**20),
                threads=self._psutil_process.num_threads())
        except Exception:
            pass
        # GPU figures from NVIDIA's management library (NVML, via pynvml):
        # how busy the GPU is, memory used, temperature and clock speed.
        if WHISPER_DEVICE == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                fields.update(gpu_util_pct=gpu_util.gpu, gpu_mem_used_mb=gpu_mem.used // 2**20,
                              gpu_temp_c=pynvml.nvmlDeviceGetTemperature(gpu_handle, 0),
                              gpu_sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(gpu_handle, 1))
            except Exception:
                pass
        # Also log how many messages are waiting in each queue (a growing queue
        # means something is falling behind) and the window's responsiveness.
        log_event(logging.INFO, "resource_stats", listening=self.is_listening,
                  run_id=self._run_id, result_queue=self.result_queue.qsize(),
                  phrase_queue=self.phrase_queue.qsize(),
                  translation_queue=self._translation_queue.qsize(),
                  **fields, **self._ui_stats.summary())
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

    def _update_metric(self, name, value):
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
            color, health = COL_ERR,  "⚠ Backlog high — translation lagging"
        elif self.is_listening:
            color, health = COL_OK,   "✔ Healthy — listening"
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
    def _avg(values):
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

    def _set_live(self, box, seq, text):
        """Show (or replace) the grey in-progress line for phrase `seq`."""
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
        box.see("end")
        self._live[box] = (start, seq)

    def _commit(self, box, seq, text):
        """Replace the live line (if it belongs to `seq` or earlier) with the
        final text."""
        start, live_seq = self._live.get(box, (None, None))
        if start is not None and live_seq <= seq:
            box.delete(start, "end")
            self._live.pop(box, None)
        if text:
            if box in self._live:      # a newer phrase is previewing: insert before it
                start, live_seq = self._live[box]
                prefix = "\n" if box.get("1.0", start).strip() else ""
                box.insert(start, prefix + text)
                self._live[box] = (box.index(f"{start} + {len(prefix + text)} chars"), live_seq)
            else:
                self._append(box, text)
                self._trim_textbox(box)

    def _commit_translation(self, seq, text):
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


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    install_global_exception_hooks()
    log_event(logging.INFO, "app_main_start")
    app = InterpreterApp()
    app.mainloop()
    log_event(logging.INFO, "app_main_exit")

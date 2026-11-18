"""
Voice Interpreter
=================

A desktop app that listens to speech, writes down what was said, and
translates it, live. Typical use: you are on a Teams/Telemost call with a
Russian-speaking friend whose voice plays through your FiiO headphones; the
app "listens" to the FiiO, shows the Russian words as they are spoken, and
shows the English translation next to them. It can also do the reverse
("My Voice": your microphone, English -> Russian).

The app's code is the interpreter package (this folder). This file, the
package's __init__.py, explains the big picture; each module's docstring
explains its own part. Use the names below to jump to the implementation
and follow the data flow.

NEW TO THE PROJECT? Start with docs/START_HERE.md: short guides with
diagrams (docs/MAP.md), a file guide, how-to recipes, the tuning numbers
and a glossary. In VS Code press Ctrl+T to jump to any class or function
in the project by name (Ctrl+Shift+O searches the open file only).


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


FILES THE PROGRAM READS AND WRITES (all in the project folder)
---------------------------------------------------------------
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


PYTHON FEATURES USED IN THIS PACKAGE
------------------------------------
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
                       inside this module/class".
  from .audio import x a relative import: the dot means "from this package".
  queue.get(timeout=…) wait for an item, but give up (queue.Empty) after the
                       timeout so the loop can check whether to stop.
  self.after(ms, f)    Tk's timer: call function f after ms milliseconds on
                       the main thread (used for every repeating job).


THE MODULES (in import order: each one only imports from those above it)
------------------------------------------------------------------------
  config.py         fixed settings: paths, tuning values, menus, colours
  logging_setup.py  log_event(), the log files, error explanations
  system.py         GPU detection, Windows scheduling, resource use
  audio.py          devices, AudioCaptureThread, levels, LatestSlot
  text_filters.py   checks and clean-ups for Whisper's and translators' text
  translation.py    LocalTranslator (Opus-MT), online services, TranslationWorker
  speech.py         Speaker (text-to-speech)
  settings.py       the user's saved choices (interpreter.settings.json)
  transcription.py  WhisperModelLoader, TranscriptionWorker
  gui_tk.py         InterpreterApp, the classic window
  __main__.py       start-up: `python -m interpreter` or `python run.py`

  Importing the package (`import interpreter`) loads them all, and this
  file re-exports their names, so interpreter.SAMPLE_RATE or
  interpreter.AudioCaptureThread work without knowing which module they
  are in. Inside the package, modules import from each other directly
  (`from .config import SAMPLE_RATE`), which shows what each one needs.
  Values in CAPITALS (config.py) are tuning knobs, each with a comment
  saying what it does and, where it was measured, why it has that value.
"""

import logging

# "import *" brings in every public name of each module. Names starting with
# "_" are skipped, so the few that other scripts use are imported by name.
from .config import *  # noqa: F401,F403
from .logging_setup import *  # noqa: F401,F403
from .system import *  # noqa: F401,F403
from .system import _add_cuda_dll_dirs  # noqa: F401  (log_tools.py)
from .audio import *  # noqa: F401,F403
from .audio import _PYAUDIO_AVAILABLE  # noqa: F401  (session_controller.py)
from .text_filters import *  # noqa: F401,F403
from .translation import *  # noqa: F401,F403
from .translation import _translate_cached  # noqa: F401  (simulate.py)
from .speech import *  # noqa: F401,F403
from .settings import *  # noqa: F401,F403
from .transcription import *  # noqa: F401,F403
from .gui_tk import *  # noqa: F401,F403

from . import config, logging_setup, system, audio, text_filters  # noqa: F401
from . import translation, speech, settings, transcription, gui_tk  # noqa: F401


log_event(logging.INFO, "app_module_loaded",
          app_dir=str(APP_DIR), log_path=str(LOG_PATH), error_log_path=str(ERROR_LOG_PATH))  # noqa: F405

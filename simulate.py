"""
simulate.py - test the interpreter with recordings instead of a microphone
==========================================================================

WHAT THIS FILE IS FOR
---------------------
To know whether a change to the app makes it better or worse, we need to
test it the same way every time, with audio whose correct text we know.
Talking into a microphone can't do that: you never say things exactly the
same way twice, and nobody writes down exactly what you said.

So this file plays a list of recorded test clips ("the playlist") through
the REAL interpreter - the same speech detection, phrase cutting, music
detection, Whisper, language detection and translation code the app uses.
Only one thing is swapped: instead of a sound card, a FAKE sound card
delivers the clips, at normal speed, as if someone were speaking them.

Afterwards it compares what the app wrote with the correct text of each
clip and measures:

  language     did Whisper recognise the right language?
  WER          "word error rate": the share of words that were wrong,
               missing or extra. 0 % is perfect; 10 % means 1 word in 10.
  translated   did a translation appear, and which translator made it?
  pause->text  how long after the clip ended its final text appeared.
  1st preview  how long after the clip started the first grey preview
               appeared.

It prints a table per language, and saves all details to
testdata/sim_results/<date-time>.json. Every phrase also goes into the app's
normal log (with a session id starting "sim-"), so log_tools.py can analyse
simulations exactly like real use.

AN ANALOGY
----------
Like a driving examiner with a fixed test route: the same roads (clips)
every time, a known correct answer for each one, and a score sheet. If the
score improves after a change, the change helped.

THE TEST CLIPS ("data sets")
----------------------------
  synth     made by make_testdata.py: computer voices reading known
            sentences, one language per clip, some over generated music.
            The most reliable: the text is known exactly.
  real      downloaded by fetch_testdata.py: real people reading (FLEURS)
            and real songs (JamendoLyrics).
  feedback  phrases you reported with "Report a mistake" in the app.
  live      (only from the simulator window) whatever Windows is playing,
            e.g. a radio stream; no correct text, so only speed and language.

Each data set has a manifest.jsonl: a list of its clips, one per line:
  {"file": "speech\\de_00.wav", "kind": "speech", "lang": "de",
   "text": "Die Polizei sagte, …", "vocal_start": 0.3, "vocal_end": 4.1}

HOW TO RUN IT (in the VS Code terminal, from the project folder)
----------------------------------------------------------------
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
    --data D        synth (default), real or feedback
The simulator window (simulator_qt.py) uses the same code with buttons.

WHAT IS IN THIS FILE (top to bottom)
------------------------------------
  Settings            paths and numbers used below.
  speech_end()        finds where the speech in a clip really ends.
  reference_words()   the correct words of a clip, ready for comparing.
  level_clip()        makes each clip equally loud ("automatic gain").
  Playlist            all clips joined into one long recording.
  FakeStream          plays the playlist like a live microphone.
  FakePyAudio,        a pretend sound-card library that the app's capture
  FakePyAudioModule   code talks to without knowing it isn't real.
  build()             sets up the app's pipeline without its window.
  choose_clips()      picks which clips to play.
  strip_label()       removes "[German] " labels from text.
  score()             matches results to clips and scores them.
  report()            prints the results table.
  main()              command-line start.
  run()               the simulation itself: play, collect, score, save.
  live_row(s)()       result rows for live mode.

PYTHON NOTES
------------
Python ideas new to this file are explained where they first appear, marked
"PYTHON:". The basics (variables, functions, classes, self, lists, dicts,
NumPy arrays) are explained at the top of interpreter/music.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse                 # reads command-line options like "--limit 3"
import json                     # reads/writes JSON text: {"name": value, …}
import queue                    # thread-safe "mailboxes" between threads
import sys                      # system things: sys.path, sys.exit, sys.version
import threading                # running work at the same time (not used directly now)
import time                     # clocks and sleeping
import uuid                     # makes random unique ids, for the session id
import wave                     # reads .wav audio files
from pathlib import Path        # file paths as objects: folder / "file.txt"

import numpy as np              # fast maths on arrays of numbers (see interpreter/music.py)

# PYTHON: __file__ is the path of this file. Path(__file__).resolve() makes it
# a full path, and .parent is its folder, e.g.
#   C:\Users\you\Code\python_projects\Interpreter
APP_DIR = Path(__file__).resolve().parent
# PYTHON: sys.path is the list of folders Python searches when you "import"
# something. Putting the app folder first makes sure "import interpreter"
# finds the interpreter\ package next to this file.
sys.path.insert(0, str(APP_DIR))
# "as I" / "as LT" give the modules short nicknames: I.log_event(...) etc.
# "# noqa: E402" tells code checkers "yes, this import is deliberately not at
# the very top" (it has to come after the sys.path line).
import interpreter as I                         # noqa: E402
import log_tools as LT                          # noqa: E402
import translation_score as TS                  # noqa: E402  chrF translation scores

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

# The real sound-card library, kept safe here. build() replaces interpreter's
# I.pyaudio with the fake one for clips; live mode needs the real one back.
REAL_PYAUDIO = I.pyaudio
# PYTHON: with Path objects, "/" joins folder names: APP_DIR / "testdata"
# -> C:\...\Interpreter\testdata
TESTDATA = APP_DIR / "testdata"                  # Path: where all test data lives
# DATA_SETS: dict, data-set name -> folder holding its manifest.jsonl and clips.
DATA_SETS = {"synth": TESTDATA / "synth",        # make_testdata.py: computer voices, made-up music
             "synth_piper": TESTDATA / "synth_piper",  # make_testdata.py --piper: Piper voices
             "real": TESTDATA,                  # fetch_testdata.py: FLEURS + Jamendo recordings
             "feedback": TESTDATA / "feedback"}  # phrases reported with "Report a mistake"
DATA_DIR = DATA_SETS["synth"]                    # Path: the data set in use (changed by --data)
RATE = 16000                                     # int: samples per second of all clips (16 kHz)


# ─────────────────────────────────────────────────────────────────────────────
# Preparing the clips
# ─────────────────────────────────────────────────────────────────────────────

def speech_end(audio, block=320):
    """Find where the speech in a clip really ends.

    Why:
        Recorded clips (FLEURS) often have about 1 second of silence after
        the last word. The "pause -> text" time must be measured from the
        last WORD, not from the end of the file - otherwise the text seemed
        to appear before the speech ended (negative times!).

    Inputs:
        audio  1-D NumPy array of int16 samples (whole numbers, 16 kHz).
        block  int (default 320): size of the pieces to check, in samples.
               320 samples at 16 kHz = 0.02 s = 20 ms.

    Returns:
        int: the sample position just after the last 20 ms piece louder than
        -45 dBFS (quieter than that counts as silence). If nothing is loud,
        the clip's full length.

    Example:
        a 5-second clip (80,000 samples) whose last word ends at 4.0 s
        -> 64,000  (4.0 s x 16,000)
    """
    # x: the same audio as decimals from -1.0 to 1.0.
    x = audio.astype(np.float32) / 32768
    n = len(x) // block                    # int: how many whole 20 ms pieces fit
    if n == 0:
        return len(x)                      # shorter than 20 ms: nothing to check
    # Loudness of each piece:
    #   x[:n * block]           drop the leftover samples that don't fill a piece
    #   .reshape(n, block)      fold the list into a table: one row per piece
    #   ** 2  ...  .mean(axis=1)  average of the squared samples in each row
    #   np.sqrt(...)            square root -> "RMS", the usual measure of loudness
    # rms: 1-D array, one loudness value (0.0 - 1.0) per piece.
    rms = np.sqrt((x[:n * block].reshape(n, block) ** 2).mean(axis=1))
    # 10 ** (-45 / 20) converts -45 dBFS into the same scale: 0.0056.
    # np.nonzero(condition)[0] gives the positions where the condition is True.
    # loud: array of the numbers of the pieces louder than -45 dBFS, e.g. [3, 4, …, 199].
    loud = np.nonzero(rms > 10 ** (-45 / 20))[0]
    # loud[-1] is the last loud piece; +1 for "just after it"; x block for samples.
    return int((loud[-1] + 1) * block) if len(loud) else len(x)


def reference_words(clip):
    """The correct words of a clip, as a list, ready to be compared.

    Inputs:
        clip  dict: one row of the manifest, e.g.
              {"file": "songs\\de_02.wav", "kind": "song", "lang": "de",
               "text": "Komm mit mir (Rufen gemeinsam) in die Nacht"}

    Returns:
        list of str: the words, in the simple form log_tools uses for
        comparing (lower case, no punctuation), e.g.
        ["komm", "mit", "mir", "in", "die", "nacht"]

    For songs, text in round brackets is removed first: those are backing
    vocals or stage directions ("(Rufen gemeinsam)" = "shouting together")
    that nobody expects the app to write down.
    """
    import re                    # "regular expressions": patterns for finding text
    text = clip["text"]
    if clip["kind"] == "song":
        # PYTHON: re.sub(pattern, replacement, text) replaces every match.
        # The pattern \([^)]*\) means: "(" then any characters except ")" then ")".
        # The r"" before the string ("raw string") keeps the backslashes as typed.
        text = re.sub(r"\([^)]*\)", " ", text)
    return LT._words(text)


# How loud each clip is made (see level_clip below).
LEVEL_TARGET_DBFS = -20         # int: target loudness of the voice; the app's own AGC uses the same
LEVEL_PEAK_DBFS   = -1          # int: the loudest single sample may reach this, never above
LEVEL_RANGE_DB    = (-20, 30)   # tuple: change the volume by at most -20 dB (quieter) / +30 dB (louder)


def level_clip(audio):
    """Automatic gain: make a clip as loud as the others.

    Why:
        Real recordings differ in loudness by 20 dB or more (one speaker is
        close to the microphone, another far away). A very quiet clip could
        be too quiet for the speech detector to notice at all - then the
        test would measure the recording, not the app.

    How:
        1. Cut the clip into 20 ms pieces and measure each piece's loudness.
        2. Keep only the pieces that are actually sounding: within 25 dB of
           the loudest piece and above -70 dBFS. Pauses between words don't
           count, otherwise a clip with many pauses would seem quieter than
           it is.
        3. The gain (volume change, in dB) is the difference between the
           target (-20 dBFS) and the loudness of those pieces.
        4. Limit the gain so the loudest single sample stays at or below
           -1 dBFS (louder would distort, "clipping"), and to the range
           -20 .. +30 dB.
        5. Apply the gain.

    Inputs:
        audio  1-D NumPy array of int16 samples.

    Returns (two values):
        the levelled audio (int16 array), and the gain used in dB (float,
        rounded to 0.1), e.g. +11.0 means "made 11 dB louder".

    Example:
        a quiet clip whose speech is at -31 dBFS  -> gain +11.0 dB -> now -20 dBFS
        a loud clip at -17 dBFS                   -> gain  -3.0 dB -> now -20 dBFS

    About decibels: +6 dB doubles the sample values, +20 dB multiplies them
    by 10. The formula 10 ** (dB / 20) turns dB into that multiplier:
        10 ** (11 / 20) = 3.55   (+11 dB = 3.55 times larger samples)
    """
    x = audio.astype(np.float32) / 32768         # decimals, -1.0 .. 1.0
    n = len(x) // 320                            # number of whole 20 ms pieces
    if n == 0:
        return audio, 0.0                        # too short: leave it as it is
    # Step 1. power: average squared sample value of each piece (see speech_end).
    power = (x[:n * 320].reshape(n, 320) ** 2).mean(axis=1)
    # db: each piece's loudness in dBFS, e.g. [-62.1, -24.3, -21.8, …]
    db = 10 * np.log10(power + 1e-12)
    # Step 2. PYTHON (NumPy): array[condition] keeps only the items where the
    # condition is True. "&" means "and" for arrays (Python's "and" doesn't
    # work on arrays). db.max() is the loudest piece.
    active = power[(db > db.max() - 25) & (db > -70)]
    if not len(active):
        return audio, 0.0                        # nothing audible: leave it
    # Step 3. 10 * log10(mean power) = the loudness of the sounding pieces in dBFS.
    gain_db = LEVEL_TARGET_DBFS - 10 * np.log10(active.mean())
    # Step 4. peak_db: the single loudest sample in dBFS (20 * log10 for sample values).
    peak_db = 20 * np.log10(np.abs(x).max() + 1e-9)
    # min(...) keeps the gain small enough that the peak lands at -1 dBFS at most.
    # PYTHON: np.clip(value, low, high) limits a value to the range low..high.
    # *LEVEL_RANGE_DB "unpacks" the tuple (-20, 30) into two separate arguments.
    gain_db = float(np.clip(min(gain_db, LEVEL_PEAK_DBFS - peak_db), *LEVEL_RANGE_DB))
    # Step 5. Multiply every sample by the gain; np.clip keeps it within -1..1.
    out = np.clip(x * 10 ** (gain_db / 20), -1, 1)
    # Back to whole numbers (int16) like the original.
    return (out * 32767).astype(np.int16), round(gain_db, 1)


class Playlist:
    """All test clips joined into one long recording, with silence between them.

    Why one long recording:
        The fake microphone plays it from start to end without stopping,
        just like a real conversation, so the app's phrase cutting, music
        detection and language switching all work as in real use.

    What it looks like (gap_sec = 1.5):
        [1.5 s silence][clip 1][1.5 s silence][clip 2][1.5 s silence] … [4 s silence]
    The 4 seconds at the very end give the last phrase time to finish.

    Attributes:
        samples  1-D int16 array: the whole recording, 16,000 per second.
        spans    list of [start, end] pairs, one per clip: the sample
                 positions where its voice starts and ends inside `samples`.
                 Used by score() to know which result belongs to which clip.
                 Example: [[24000, 89600], [113600, 190400], …]
        t0       float or None: the clock time (time.monotonic()) when
                 playback started; set by FakeStream.
        clips    list of dicts: the manifest rows (set by run() after
                 creating the playlist).

    Example:
        playlist = Playlist(clips, gap_sec=1.5)
        playlist.duration      ->  1043.2  (seconds, ~17 minutes)
    """

    def __init__(self, clips, gap_sec, level=True):
        """Load every clip's audio file and join them into one recording.

        Inputs:
            clips    list of dicts: the clips to play (manifest rows).
            gap_sec  float: seconds of silence between clips, e.g. 1.5.
            level    bool (default True): make every clip equally loud first
                     (level_clip). The gain used is stored in the clip's
                     dict as clip["gain_db"], so the window can show it.

        Where each clip's voice starts and ends:
            Synthetic clips know it exactly ("vocal_start"/"vocal_end" in
            the manifest; songs carry on with music after the voice).
            Recorded clips don't, so speech_end() measures the end.
        """
        # parts: list of arrays to join at the end; spans: see above;
        # pos: int, how many samples have been added so far (where the next part starts).
        parts, self.spans, pos = [], [], 0
        # silence: gap_sec seconds of zeros, e.g. 1.5 x 16000 = 24,000 zeros.
        silence = np.zeros(int(gap_sec * RATE), np.int16)
        # PYTHON: ";" lets two short statements share one line.
        parts.append(silence); pos += len(silence)
        for clip in clips:
            # Read the WAV file. w.getnframes() = how many samples it has;
            # w.readframes(n) returns them as raw bytes; np.frombuffer turns
            # those bytes into an int16 array.
            with wave.open(str(DATA_DIR / clip["file"])) as w:
                audio = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if level:
                # Two results at once: the levelled audio, and the gain stored in the clip.
                audio, clip["gain_db"] = level_clip(audio)
            parts.append(audio)
            # The span ends where the speech ends: FLEURS clips have ~1 s of
            # silence at the end, which made "pause -> text" negative.
            # Synthetic clips know exactly where the vocals end (songs carry
            # on with music after them).
            end = (int(clip["vocal_end"] * RATE) if "vocal_end" in clip else speech_end(audio))
            # clip.get("vocal_start", 0): the value, or 0 if the clip has none.
            start = int(clip.get("vocal_start", 0) * RATE)
            # Positions inside the WHOLE recording = where the clip starts + inside the clip.
            self.spans.append([pos + start, pos + end])
            pos += len(audio)
            parts.append(silence); pos += len(silence)
        parts.append(np.zeros(4 * RATE, np.int16))        # 4 s at the end: let the last phrase end
        self.samples = np.concatenate(parts)               # join everything into one array
        self.t0 = None                                     # set when playback starts

    def time_of(self, sample):
        """The clock time at which a sample position is played.

        Example: playback started at t0 = 1000.0; sample 48,000 is 3 s in
        -> 1003.0. Used to compare clip positions with result times.
        """
        return self.t0 + sample / RATE

    # PYTHON: @property lets a method be read like a plain attribute, without
    # brackets: playlist.duration instead of playlist.duration(). The value is
    # calculated each time it is read.
    @property
    def duration(self):
        """The whole recording's length in seconds (float), e.g. 1043.2."""
        return len(self.samples) / RATE


# ─────────────────────────────────────────────────────────────────────────────
# The fake sound card
# ─────────────────────────────────────────────────────────────────────────────
# The app's capture thread (AudioCaptureThread in interpreter/audio.py) records
# through a library called PyAudio, like this:
#     pa = pyaudio.PyAudio()                  # start the library
#     stream = pa.open(...)                   # open a microphone
#     data = stream.read(640)                 # wait for and read 640 samples
# The three classes below have methods with exactly those names, so the
# capture thread can use them without knowing they are fake. This trick is
# called a "fake" or "mock" object: same interface, different insides.
# PYTHON: Python doesn't check an object's class, only whether it has the
# methods being called ("duck typing": if it quacks like a duck…).

class FakeStream:
    """Plays the playlist in real time, like a live microphone.

    The key idea: a real microphone can't deliver sound from the future.
    read() waits until enough time has passed for the requested samples
    to have been "spoken", then returns them.

    Attributes:
        p    the Playlist being played.
        pos  int: how many samples have been delivered so far.
    """

    def __init__(self, playlist):
        """Start "playing" now: remember the start time in the playlist."""
        self.p, self.pos = playlist, 0
        # time.monotonic(): a clock that only moves forward (good for timing).
        playlist.t0 = time.monotonic()

    def _due(self):
        """How many samples should have been played by now (int).

        Example: 2.5 s after the start -> 2.5 x 16000 = 40,000 samples.
        """
        return int((time.monotonic() - self.p.t0) * RATE)

    def get_read_available(self):
        """How many samples are ready to read without waiting (PyAudio
        has this method; the capture thread uses it to catch up).
        max(0, ...) never returns a negative number."""
        return max(0, self._due() - self.pos)

    def read(self, frames, exception_on_overflow=False):
        """Return the next `frames` samples, waiting until their time has come.

        Inputs:
            frames                 int: how many samples to read, e.g. 640 (40 ms).
            exception_on_overflow  ignored; PyAudio's real read() has it.

        Returns:
            bytes: the samples as raw data (2 bytes per int16 sample), exactly
            what a real microphone returns. After the playlist ends: silence.

        Example: at 2.00 s, the capture thread asks for 640 samples starting
        at 32,000. They are due at 32,640 / 16,000 = 2.04 s, so read() waits
        0.04 s, then returns them.
        """
        # PYTHON: "while condition:" repeats the indented block as long as the
        # condition is True. time.sleep(0.002) pauses this thread for 2 ms.
        while self._due() < self.pos + frames:          # not time yet: wait like a real device
            time.sleep(0.002)
        chunk = self.p.samples[self.pos:self.pos + frames]
        if len(chunk) < frames:                          # past the end: top up with silence
            chunk = np.concatenate([chunk, np.zeros(frames - len(chunk), np.int16)])
        self.pos += frames
        return chunk.tobytes()                           # array -> raw bytes

    def close(self):
        """Nothing to close (a real stream releases the microphone here)."""
        pass


class FakePyAudio:
    """A pretend PyAudio: just enough for the capture thread to "open" one
    16 kHz mono microphone that plays the playlist.

    Class attribute:
        playlist  the Playlist to play. Set by build() before the capture
                  thread starts; shared by all FakePyAudio objects.
    """
    playlist = None

    def get_device_info_by_index(self, index):
        """Describe the one fake device, in the same dict format PyAudio uses."""
        return {"name": "simulator", "maxInputChannels": 1, "defaultSampleRate": RATE,
                "isLoopbackDevice": False, "index": 0}

    # PYTHON: a lambda can also be a method. This one returns the same
    # description as above, for "the default microphone".
    get_default_input_device_info = lambda self: self.get_device_info_by_index(0)

    def is_format_supported(self, rate, **kw):
        """Can the device record at this sample rate? Only 16 kHz.

        PYTHON: **kw collects any other named arguments (channels=..., etc.)
        into a dict, so the method accepts whatever the capture thread passes.
        PYTHON: "raise" creates an error on purpose; the capture thread
        catches it and tries the next rate, as with a real device.
        """
        if rate != RATE:
            raise ValueError("simulator plays 16 kHz only")
        return True

    def open(self, **kw):
        """"Open the microphone": return a FakeStream playing the playlist."""
        return FakeStream(FakePyAudio.playlist)

    def terminate(self):
        """Nothing to shut down (a real PyAudio releases the audio system)."""
        pass


class FakePyAudioModule:
    """Stands in for the whole pyaudio MODULE (interpreter/audio.py uses
    pyaudio.paInt16 and pyaudio.PyAudio).

    paInt16  int 8: PyAudio's code for "16-bit whole-number samples".
    PyAudio  the fake class above.
    """
    paInt16 = 8
    PyAudio = FakePyAudio


# ─────────────────────────────────────────────────────────────────────────────
# Building the pipeline (what the window's _start_session does, without a window)
# ─────────────────────────────────────────────────────────────────────────────

def build(args, playlist, say=print, level_cb=None):
    """Set up the app's pipeline (capture thread + transcription worker),
    exactly as the window does when you start listening, but without a window.

    Inputs:
        args      the settings (see main()): src, tgt, model, preview,
                  sensitivity, no_music, … (an argparse "Namespace": read
                  them as args.model etc.)
        playlist  the Playlist to play through the fake sound card. None =
                  live mode: record the real device args.device instead (a
                  "loopback" device records what Windows plays on an output).
        say       function to show progress messages (default: print).
        level_cb  optional function called for every audio block (20-30 ms)
                  from the capture thread, as
                  level_cb(level_db, peak_db, speaking, speech_prob, threshold).
                  The simulator window uses it to move its meters.

    Returns (four values):
        capture     the AudioCaptureThread (not started yet).
        worker      the TranscriptionWorker (not started yet).
        results     queue.Queue: where the worker puts its results.
        session_id  str: this run's id in the log, e.g. "sim-3b917432".

    Steps:
      1. Swap in the fake sound card (or the real one for live mode).
      2. Load the Whisper model(s) and wait until they are ready.
      3. Load music detection (unless --no-music), and wait for it.
      4. Pre-load the translators the clips will need.
      5. Create the queues, the capture thread and the worker, and connect them.
    """
    # Step 1. Replace the "pyaudio" name inside interpreter/audio.py, the
    # module whose functions use it. (Setting I.pyaudio would only change
    # the package's copy of the name; audio.py would never see it.)
    if playlist is None:
        I.audio.pyaudio = REAL_PYAUDIO                  # a real device
    else:
        I.audio.pyaudio = FakePyAudioModule             # the fake sound card
        FakePyAudio.playlist = playlist
    # src_key/tgt_key: the language names, e.g. "Auto-detect", "English".
    src_key, tgt_key = args.src, args.tgt
    # I.LANGUAGES maps names to codes, e.g. "English" -> ("en", "en");
    # [0] takes the first: Whisper's code ("auto" for Auto-detect).
    src_whisper = I.LANGUAGES[src_key][0]
    tgt_code = I.LANGUAGES[tgt_key][0]

    # Step 2. Load Whisper in a background thread (WhisperModelLoader is a
    # thread; .start() starts it). status_cb=lambda m: None ignores its messages.
    say(f"Loading Whisper {args.model} …")
    main = I.WhisperModelLoader(model_size=args.model, status_cb=lambda m: None)
    main.start()
    preview = None                                      # a second, smaller model for previews
    if args.preview in ("tiny", "small"):
        preview = I.WhisperModelLoader(model_size=args.preview, status_cb=lambda m: None)
        preview.start()
    # PYTHON: .ready is a threading.Event, a flag one thread sets and others
    # wait for. .wait() pauses here until the loader sets it.
    main.ready.wait()
    if main.model is None:
        # PYTHON: "raise RuntimeError(...)" stops this function with an error;
        # the caller (the simulator window) catches it and explains it.
        raise RuntimeError(f"Whisper '{args.model}' failed to load: {main.error}")
    # getattr(obj, "name", default): read an attribute, or the default if it doesn't exist.
    if getattr(main, "used_fallback", False):
        # The app falls back to Tiny on the CPU; a simulation with a different
        # model than asked for would give misleading figures, so stop.
        raise RuntimeError(f"Whisper '{args.model}' failed to load ({main.error or 'see log'}); "
                           "not simulating with the CPU fallback model")
    if preview:
        preview.ready.wait()
    # Step 3. Music detection (see interpreter/music.py).
    scene = None
    if not args.no_music:
        from interpreter.music import MusicScene
        # The scene logs through the app's log: 20 is the number for "INFO" level.
        scene = MusicScene(log=lambda e, **k: I.log_event(20, e, **k))
        # Wait (up to 60 s) for its sound classifier to load, checking every 0.2 s.
        t = time.monotonic()
        while scene.classifier is None and time.monotonic() - t < 60:
            time.sleep(0.2)
    # Step 4. Pre-load the local translators the clips will need, so the first
    # phrases aren't counted as "online" while a model downloads.
    # ModelManager (interpreter/model_manager.py) is what the windows use to
    # load models. The simulator uses one too, for the translators only.
    # loaders.mt_loaders: (source, target) -> its TranslatorLoader,
    # e.g. {("de", "en"): <TranslatorLoader>}
    loaders = I.ModelManager()
    # PYTHON: {expression for …} with curly brackets builds a SET: like a list,
    # but each value only once. {c["lang"] for c in clips} -> {"de", "fr", "ru", …}
    # In live mode the languages aren't known in advance; only the chosen
    # source language (if it isn't Auto-detect): {"ru"} - {"auto"} removes "auto".
    langs = ({c["lang"] for c in playlist.clips} if playlist is not None
             else {src_whisper} - {"auto"})
    for code in langs:
        # local_route gives the translation steps, e.g. bg -> en is [("bg", "en")],
        # but bg -> ru may go via English: [("bg", "en"), ("en", "ru")].
        for pair in I.local_route(code, tgt_code):
            # PYTHON: *pair unpacks ("bg", "en") into two arguments: f("bg", "en").
            loaders.translator_for(*pair)
    for loader in loaders.mt_loaders.values():          # .values(): the dict's values
        loader.ready.wait()

    # Step 5. Connect everything, as in the app.
    # session_id: "sim-" + 8 random hexadecimal characters, e.g. "sim-3b917432".
    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    # results: worker -> this file (finals, previews, status messages).
    results = queue.Queue()
    # phrases: capture thread -> worker (finished phrases). maxsize=32: at most
    # 32 waiting phrases (the app uses the same size).
    phrases = queue.Queue(maxsize=32)
    # LatestSlot: a one-item "mailbox" that only keeps the newest item, used for
    # previews (an old preview is useless once a newer one exists).
    partial_slot = None if args.preview == "off" else I.LatestSlot()
    early_slot = I.LatestSlot()                          # for early transcription at a pause
    capture = I.AudioCaptureThread(device_index=0 if playlist is not None else args.device,
                                   phrase_queue=phrases,
                                   status_cb=lambda m: results.put(("status", m)),
                                   level_cb=level_cb or (lambda *a: None), partial_slot=partial_slot,
                                   early_slot=early_slot)
    capture.set_sensitivity(args.sensitivity)
    if playlist is not None:
        # The gaps between clips are digital silence on purpose: don't warn that
        # "no sound reaches the device" (it would list the real apps playing).
        # This replaces the capture thread's warning function with one that does nothing.
        capture._report_source = lambda **kw: None
    capture.music_scene = scene
    worker = I.TranscriptionWorker(
        wid=0, model=main.model, src_whisper=src_whisper, src_lang_key=src_key,
        tgt_lang_key=tgt_key, phrase_queue=phrases, result_queue=results,
        status_cb=lambda m: results.put(("status", m)), partial_slot=partial_slot,
        preview_model=(preview.model if preview else None), run_id=1, record_audio=False,
        translator=loaders.translator_for(src_whisper, tgt_code), early_slot=early_slot,
        session_id=session_id,
        # config: settings written into every phrase's log line, so log_tools
        # can compare runs with different settings.
        config={"model": args.model, "preview_choice": args.preview, "simulation": True,
                "sensitivity": args.sensitivity, "src_lang": src_key, "tgt_lang": tgt_key})
    worker.music_scene = scene
    worker.auto_language = True                          # translate phrases in other languages too
    worker.translator_for = loaders.translator_for       # how the worker gets more translators
    return capture, worker, results, session_id


# ─────────────────────────────────────────────────────────────────────────────
# Choosing, running and scoring
# ─────────────────────────────────────────────────────────────────────────────

def choose_clips(args):
    """Pick the clips to play and put them in a sensible order.

    Inputs:
        args  the settings: kinds ("speech,song"), langs ("de,fr" or "" for
              all), limit (clips per language and kind; 0 = all).

    Returns:
        list of dicts: the manifest rows to play, in playing order.

    Steps:
      1. Read every line of the data set's manifest.jsonl.
      2. Keep the clips of the wanted kinds and languages, grouped by
         (kind, language).
      3. Keep at most `limit` clips per group.
      4. Interleave the groups (one clip from each group in turn), so the
         language changes from clip to clip, as it might in real use, and
         language detection is really tested.

    Example (2 languages, 2 speech clips each):
        groups:  ("speech","de"): [de_00, de_01]   ("speech","fr"): [fr_00, fr_01]
        order:   de_00, fr_00, de_01, fr_01
    """
    # Step 1. json.loads(line) turns one line of JSON text into a dict.
    rows =[json.loads(l) for l in open(DATA_DIR / "manifest.jsonl", encoding="utf-8")]
    # .split(",") cuts text at commas: "speech,song" -> ["speech", "song"].
    # set(...) makes it a set, which is fast for "is x in it?" checks.
    kinds = set(args.kinds.split(","))
    langs = set(args.langs.split(",")) if args.langs else None    # None = all languages
    # groups: dict, (kind, lang) -> list of clips.
    groups = {}
    for r in rows:                                  # step 2
        if not r.get("file"):                       # a report saved without audio: skip
            continue
        if r["kind"] in kinds and (not langs or r["lang"] in langs):
            # PYTHON: d.setdefault(key, []) returns d[key], first creating it as
            # an empty list if it doesn't exist yet. Then .append adds the clip.
            groups.setdefault((r["kind"], r["lang"]), []).append(r)
    if args.limit:                                  # step 3 (0 counts as False = no limit)
        # A dict comprehension: same keys, lists cut to the first `limit` items.
        groups = {k: v[:args.limit] for k, v in groups.items()}
    # Step 4. Interleave: one clip from each group in turn.
    # order: the result; lists: a copy of each group's list (they get emptied).
    order, lists = [], [list(v) for v in groups.values()]
    # PYTHON: any(lists) is True while at least one list still has items
    # (an empty list counts as False).
    while any(lists):
        for lst in lists:
            if lst:
                order.append(lst.pop(0))            # .pop(0) removes and returns the first item
    return order


def strip_label(text):
    """Remove the "[German] " label the app adds when the language changes.

    Example:
        "[German] Guten Morgen"  ->  "Guten Morgen"
        "Hello"                  ->  "Hello"   (no label: unchanged)
        None                     ->  ""        ("text or ''" turns None into "")

    The pattern ^\\[[^\\]]+\\]\\s* means: at the start (^), a "[", one or more
    characters that aren't "]", a "]", then any spaces.
    """
    import re
    return re.sub(r"^\[[^\]]+\]\s*", "", text or "")


def score(playlist, finals, previews):
    """Work out which results belong to which clip, and score every clip.

    Inputs:
        playlist  the Playlist that was played (its spans say where each
                  clip's voice was).
        finals    list of dicts, one per finished phrase the app produced:
                  {"seq": 5, "text": "Guten Morgen", "translated": "Good morning",
                   "trace": {...the phrase's log details...}, "t": 1003.4}
        previews  list of (time, preview text) tuples.

    Returns:
        list of dicts, one per clip, e.g.
          {"file": "speech\\de_00.wav", "kind": "speech", "lang": "de",
           "ref": the correct text, "hyp": what the app wrote,
           "translation": "...", "engines": ["opus-mt-de"],
           "detected": "de", "ref_words": 12, "errors": 1,
           "latency_ms": 372, "first_preview_ms": 410,
           "music_mode": False, "phrases": 1}
        ("hyp", short for "hypothesis", is the usual name for the
        recogniser's output; "ref", "reference", is the correct text.)

    Step 1, matching:
        The app doesn't know which clip a phrase came from - it just hears
        sound. So each phrase is given to the clip it overlaps most in time.
        A phrase's time span is known from its log details: it ended at
        t_end and was capture_ms long.
          Example: a phrase from 1003.2 to 1007.9 s overlaps clip 4
          (1002.5 - 1008.1 s) by 4.7 s and clip 5 by 0 s -> clip 4.
        A long clip can have several phrases; a missed clip has none.

    Step 2, scoring each clip:
        * its phrases' texts are joined, and compared word by word with the
          correct text -> number of word errors (log_tools._align counts
          the fewest wrong/missing/extra words that explain the difference);
        * the language used most often among its phrases is its
          "detected" language;
        * latency = the time of its last phrase's text minus the end of the
          clip's voice;
        * first preview = the first preview's time minus the clip's start.
    """
    # Step 1. per_clip: one dict per clip, collecting its phrases.
    # PYTHON: "_" as a loop variable means "I don't need its value".
    per_clip = [{"finals": []} for _ in playlist.clips]
    for f in finals:
        tr = f["trace"] or {}                       # the phrase's log details (or empty)
        end = tr.get("t_end")                       # when the phrase ended (clock time)
        # When it started = end minus its length (capture_ms is in milliseconds).
        start = end - (tr.get("capture_ms") or 0) / 1000 if end else None
        if end is None:
            continue                                # no timing: can't be matched
        best, overlap = None, 0.0                   # the best clip so far, and its overlap
        for i, (a, b) in enumerate(playlist.spans):
            # Overlap of two time ranges = earliest end - latest start
            # (negative if they don't overlap at all).
            o = min(end, playlist.time_of(b)) - max(start, playlist.time_of(a))
            if o > overlap:
                best, overlap = i, o
        if best is not None:
            per_clip[best]["finals"].append(f)
    # Step 2. out: the result rows.
    out = []
    # PYTHON: zip(a, b, c) walks through several lists side by side:
    # (clip 0, span 0, results 0), (clip 1, span 1, results 1), …
    for clip, span, got in zip(playlist.clips, playlist.spans, per_clip):
        fs = got["finals"]                          # this clip's phrases
        # " ".join(list) glues texts together with spaces between them.
        hyp = " ".join(strip_label(f["text"]) for f in fs if f["text"]).strip()
        ref_words, hyp_words = reference_words(clip), LT._words(hyp)
        # _align returns several values; [0] is the number of word errors.
        errors = LT._align(ref_words, hyp_words)[0] if ref_words else 0
        # detected: the language of each phrase that has text, e.g. ["de", "de", "en"].
        # (lang_used = what the app finally used; lang = Whisper's first guess.)
        detected = [l for l in ((f["trace"] or {}).get("lang_used") or (f["trace"] or {}).get("lang")
                                for f in fs if f["text"]) if l]
        clip_end, clip_start = playlist.time_of(span[1]), playlist.time_of(span[0])
        # pause -> text: from the end of the clip to its last result.
        # PYTHON: max(..., default=None) gives None instead of an error if there are none.
        last_final = max((f["t"] for f in fs if f["text"]), default=None)
        # The first preview shown from (just before) this clip's start onwards.
        first_prev = min((t for t, s in previews if clip_start - 0.2 <= t), default=None)
        out.append({
            "file": clip["file"], "kind": clip["kind"], "lang": clip["lang"],
            "ref": clip["text"], "hyp": hyp,
            "translation": " ".join(f["translated"] or "" for f in fs).strip(),
            # Which translators were used, without repeats ({...} is a set), sorted.
            "engines": sorted({(f["trace"] or {}).get("translation_engine") or "?" for f in fs if f["text"]}),
            # The most common language: max over the distinct values, compared by
            # how often each occurs (key=detected.count).
            "detected": max(set(detected), key=detected.count) if detected else None,
            "ref_words": len(ref_words), "errors": errors,
            "latency_ms": round((last_final - clip_end) * 1000) if last_final else None,
            # Only counted if it came within 2 s after the clip ended (otherwise
            # it belongs to the next clip).
            "first_preview_ms": round((first_prev - clip_start) * 1000)
                                if first_prev and first_prev < clip_end + 2 else None,
            "music_mode": any((f["trace"] or {}).get("music_mode") for f in fs),
            "phrases": len(fs),
        })
    return out


def report(rows, args, wall_sec):
    """Print the results table in the terminal.

    Inputs:
        rows      list of dicts from score().
        args      the settings (shown in the heading).
        wall_sec  float: how long the whole simulation took, in seconds.

    What it prints:
        Simulation: Auto-detect -> English, model large-v3-turbo, …, 17.4 min
          kind   lang   clips  lang ok   WER  transl. pause->text 1st preview  music
          speech de         4      4/4    7%      4/4      372 ms      947 ms    0/4
          …
          speech all       40    39/40    6%    40/40      366 ms      959 ms   0/40
        then the clips heard in the wrong language, and a few examples.
    """
    def summary(rs):
        """The figures for one table row (a group of clips).

        PYTHON: a function defined inside another function is only usable
        inside it - it's a private helper of report().

        Returns a dict of ready-to-print text values, e.g.
            {"clips": 4, "lang_ok": "4/4", "wer": "7%", "translated": "4/4",
             "latency": "372 ms", "preview": "947 ms", "music": "0/4"}
        """
        words =sum(r["ref_words"] for r in rs)                     # all correct words
        # min(errors, words): a clip can't count as more than 100 % wrong.
        errs = sum(min(r["errors"], r["ref_words"]) for r in rs)
        lid = [r for r in rs if r["detected"]]                      # clips with a detected language
        lat = [r["latency_ms"] for r in rs if r["latency_ms"] is not None]
        prev = [r["first_preview_ms"] for r in rs if r["first_preview_ms"] is not None]
        # PYTHON: in f-strings, {x:.0%} shows a fraction as a percentage with no
        # decimals: 0.071 -> "7%".
        # np.median: the middle value (half are lower, half higher).
        return {"clips": len(rs),
                "lang_ok": f"{sum(r['detected'] == r['lang'] for r in lid)}/{len(rs)}",
                "wer": f"{errs / words:.0%}" if words else "-",
                "translated": f"{sum(bool(r['translation']) for r in rs)}/{len(rs)}",
                "latency": f"{int(np.median(lat))} ms" if lat else "-",
                "preview": f"{int(np.median(prev))} ms" if prev else "-",
                "music": f"{sum(r['music_mode'] for r in rs)}/{len(rs)}",
                # Average chrF of the translations (translation_score.py), e.g. "58".
                "chrf": TS.summary(rs).split()[0]}
    print(f"\nSimulation: {args.src} -> {args.tgt}, model {args.model}, previews {args.preview}, "
          f"music mode {'off' if args.no_music else 'on'}, {wall_sec / 60:.1f} min")
    # PYTHON: in f-strings, {x:<7} pads x to 7 characters, left-aligned;
    # {x:>6} pads to 6, right-aligned. That's how the columns line up.
    head = f"  {'kind':<7}{'lang':<6}{'clips':>6}{'lang ok':>9}{'WER':>6}{'transl.':>9}" \
           f"{'pause->text':>12}{'1st preview':>12}{'music':>7}{'chrF':>7}"
    print(head)
    # One row per (kind, language), in alphabetical order.
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["lang"]), []).append(r)
    for (kind, lang), rs in sorted(groups.items()):
        s = summary(rs)
        print(f"  {kind:<7}{lang:<6}{s['clips']:>6}{s['lang_ok']:>9}{s['wer']:>6}{s['translated']:>9}"
              f"{s['latency']:>12}{s['preview']:>12}{s['music']:>7}{s['chrf']:>7}")
    # Then one total row per kind ("speech all", "song all").
    for kind in sorted({r["kind"] for r in rows}):
        s = summary([r for r in rows if r["kind"] == kind])
        print(f"  {kind + ' all':<13}{s['clips']:>6}{s['lang_ok']:>9}{s['wer']:>6}{s['translated']:>9}"
              f"{s['latency']:>12}{s['preview']:>12}{s['music']:>7}{s['chrf']:>7}")
    # The clips heard in the wrong language (at most 10).
    wrong = [r for r in rows if r["detected"] and r["detected"] != r["lang"]]
    if wrong:
        print("\n  Wrong language detected:")
        for r in wrong[:10]:
            # {x!r} shows x with quotes, so empty text or spaces are visible.
            print(f"    {r['file']}: {r['lang']} heard as {r['detected']}: {r['hyp'][:60]!r}")
    print("\n  Examples (reference | recognised | translation):")
    # rows[::k] takes every k-th row (about 8 examples spread over the run).
    for r in rows[:: max(1, len(rows) // 8)]:
        print(f"    [{r['kind']} {r['lang']}] {r['ref'][:45]!r}\n"
              f"        -> {r['hyp'][:60]!r}\n        => {r['translation'][:60]!r}")


def main():
    """Command-line start: read the options, run, print the table.

    Steps:
      1. Read the options typed after "simulate.py" (argparse).
      2. Choose the data set's folder; stop with a message if it has no clips.
      3. Run the simulation, printing each phrase as it is recognised.
      4. Print the results table and where the details were saved.
    """
    # Step 1. argparse describes the options; parse_args() reads what was typed.
    # Example: "simulate.py --limit 1 --no-music" -> args.limit == 1, args.no_music == True.
    ap =argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="Auto-detect")
    ap.add_argument("--tgt", default="English")
    ap.add_argument("--kinds", default="speech,song")
    ap.add_argument("--langs", default="")
    ap.add_argument("--limit", type=int, default=0)            # type=int: turn the text "3" into 3
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--preview", default="small", choices=("off", "tiny", "small", "main"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--sensitivity", type=float, default=0.5)
    ap.add_argument("--no-music", action="store_true")         # present = True, absent = False
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--data", default="synth", choices=tuple(DATA_SETS))
    args = ap.parse_args()
    # Step 2. PYTHON: "global DATA_DIR" lets this function change the module-level
    # variable DATA_DIR (without it, Python would create a new local variable).
    global DATA_DIR
    DATA_DIR = DATA_SETS[args.data]
    if not (DATA_DIR / "manifest.jsonl").exists():
        maker = "make_testdata.py" if args.data == "synth" else "fetch_testdata.py"
        # sys.exit("message") prints the message and ends the program.
        sys.exit(f"No test data: run  .venv\\Scripts\\python {maker}  first.")

    def show(kind, data):
        """Print what the simulation reports while it runs: status messages,
        and each finished phrase as  'recognised text' -> 'translation'."""
        if kind == "status":
            # flush=True: print immediately (otherwise output may be held back).
            print(data, flush=True)
        elif kind == "final" and data["text"]:
            print(f"  {strip_label(data['text'])[:70]!r} -> {str(data['translated'])[:50]!r}", flush=True)

    rows, path, wall_sec = run(args, show)                       # step 3
    report(rows, args, wall_sec)                                 # step 4
    print(f"\nDetails: {path}")


def run(args, on_event=lambda kind, data: None, stop=None, level_cb=None):
    """Run a whole simulation: play the clips, collect the results, score
    them and save them. Used by main() and by the simulator window.

    Inputs:
        args      the settings (see main()).
        on_event  a function called with progress as on_event(kind, data);
                  the default does nothing. Kinds and their data:
                    "status"       text, e.g. "Loading Whisper large-v3-turbo …"
                    "started"      the Playlist (its t0 is now set), or None live
                    "partial"      (seq, heard, translated, t_start, t): a preview
                    "final"        a finished phrase: dict seq/text/translated/trace/t
                    "translation"  the same phrase dict, after an online
                                   translation arrived for it
        stop      a threading.Event (or None). When it is set (the window's
                  Stop button), the run ends early; clips not yet played are
                  left out of the results.
        level_cb  passed on to build() (for the window's meters).

    Returns (three values):
        rows      list of result dicts (see score(), or live_row() in live mode)
        path      Path of the saved results file
        wall_sec  float: how long it took, in seconds

    Steps:
      1. Choose the clips and build the playlist (not in live mode).
      2. Build the pipeline and start its two threads.
      3. Loop: collect results as they arrive, until the playlist has been
         played (+3 s) or stop is set.
      4. Stop the threads; wait for outstanding online translations.
      5. Score, and save everything to testdata/sim_results/.

    Live mode (args.data == "live"): no clips; it records the real device
    args.device until stop is set. Each phrase becomes a result row (there
    is no correct text to compare with).
    """
    # ThreadPoolExecutor: a small team of threads that run submitted jobs.
    from concurrent.futures import ThreadPoolExecutor
    live = args.data == "live"                           # bool
    # Step 1
    if live:
        playlist, clips = None, []
        on_event("status", f"Recording live from {getattr(args, 'device_name', args.device)}")
    else:
        clips = choose_clips(args)
        playlist = Playlist(clips, args.gap, level=getattr(args, "level_clips", True))
        playlist.clips = clips
        on_event("status", f"{len(clips)} clips, {playlist.duration / 60:.1f} min of audio")
    # Step 2
    capture, worker, results, session_id = build(args, playlist, lambda m: on_event("status", m),
                                                 level_cb)
    # PYTHON: vars(args) turns the settings into a dict; **dict passes each entry
    # as a named argument, so every setting goes into the log line.
    I.log_event(20, "simulation_started", session_id=session_id, clips=len(clips), **vars(args))
    I.log_session_context(session_id, simulation=True)

    worker.start(); capture.start()
    if not live:
        # Wait until the capture thread has "opened the microphone" (FakeStream
        # sets playlist.t0), so the clock times below make sense.
        while playlist.t0 is None:
            time.sleep(0.01)
    on_event("started", playlist)
    # finals: list of finished phrases; previews: list of (time, text);
    # wall0: start time, for the total duration.
    finals, previews, wall0 = [], [], time.monotonic()
    # Online translation in the background: waiting for it here held up every
    # later result by the round trip to the internet (~0.2-0.5 s per phrase).
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sim-translate")
    done_translations = queue.Queue()                   # finished online translations come back here

    def translate_online(final, lang):
        """Translate one phrase via the internet (runs on a pool thread).

        Stores the translation in the phrase's dict (final["translated"])
        and puts the dict into done_translations, where the main loop picks
        it up and reports it. If the internet fails, the error is logged and
        the translation says so, e.g. "[online translation failed: timeout]".

        PYTHON: this inner function can use `args` and `done_translations`
        from run(), because inner functions see their outer function's
        variables (a "closure").
        """
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

    # Step 3. PYTHON: "while True:" loops forever, until a "break" inside it.
    while True:
        try:
            # Wait up to 0.05 s for the next result from the worker.
            kind, data = results.get(timeout=0.05)
        except queue.Empty:
            kind = None                                 # nothing arrived; check the end conditions
        now = time.monotonic()
        if kind == "final":
            # A finished phrase: (number, text, translation or None, log details).
            seq, text, translated, trace = data
            final = {"seq": seq, "text": text, "translated": translated, "trace": trace, "t": now}
            finals.append(final)
            on_event("final", final)
            # None = no local translator for it: ask the online one, in the background.
            if translated is None and text and not args.offline:
                pool.submit(translate_online, final, (trace or {}).get("lang_used") or worker.src_whisper)
        elif kind == "partial":
            previews.append((now, data[1]))             # data[1] is the preview text
            on_event("partial", data)
        elif kind == "status":
            on_event("status", data)
        # Report any online translations that have come back.
        while not done_translations.empty():
            on_event("translation", done_translations.get())
        # Finished: the whole playlist has played (+3 s) and no results are waiting.
        if not live and now - playlist.t0 > playlist.duration + 3 and results.empty():
            break
        if stop is not None and stop.is_set():          # the Stop button was pressed
            break
    # Step 4
    capture.stop(); worker.stop()
    pool.shutdown(wait=True)                            # wait for translations still running
    while not done_translations.empty():
        on_event("translation", done_translations.get())
    # Step 5
    if live:
        rows = live_rows(finals)
    else:
        rows = score(playlist, finals, previews)
        # When stopped early, keep only clips that had finished at least 3 s ago.
        # (True counts as 1 in sum(), so this counts those clips.)
        played = sum(playlist.time_of(b) < time.monotonic() - 3 for _, b in playlist.spans)
        rows = rows[:played] if stop is not None and stop.is_set() else rows
        # Score the translations against FLEURS's human translations (adds
        # "ref_translation" and "chrf" to each row; see translation_score.py).
        on_event("status", "Scoring translations…")
        TS.score_rows(rows, args.tgt)
    out_dir = TESTDATA / "sim_results"
    out_dir.mkdir(exist_ok=True)                        # create the folder if needed
    # File name from the date and time, e.g. 20261101-014549.json (or …-live.json).
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}{'-live' if live else ''}.json"
    # json.dumps turns the dict into JSON text: ensure_ascii=False keeps
    # letters like "ü" and "я" readable; indent=1 adds line breaks;
    # default=str writes anything JSON can't store (e.g. a Path) as text.
    path.write_text(json.dumps({"args": vars(args), "session_id": session_id, "clips": rows},
                               ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return rows, path, time.monotonic() - wall0


def live_row(final):
    """One result row for a phrase heard in live mode.

    There is no correct text to compare with, so the row records what can
    be measured: the language, the text, how fast it came, and how.

    Inputs:
        final  dict: one finished phrase (see run()).

    Returns:
        dict, e.g. {"seq": 12, "kind": "live", "lang": None, "detected": "ru",
                    "hyp": "Привет", "translation": "Hi", "latency_ms": 310,
                    "audio_sec": 1.8, "music_mode": False, "engine": "opus-mt-ru"}
        latency_ms here is from the end of the phrase (when its pause was
        detected) to its result.
    """
    tr = final["trace"] or {}
    return {"seq": final["seq"], "kind": "live", "lang": None,
            "detected": tr.get("lang_used") or tr.get("lang"),
            "hyp": strip_label(final["text"]), "translation": final["translated"] or "",
            "latency_ms": round((tr["t_result"] - tr["t_end"]) * 1000) if tr.get("t_end") and tr.get("t_result") else None,
            "audio_sec": tr.get("audio_sec"), "music_mode": bool(tr.get("music_mode")),
            "engine": tr.get("translation_engine")}


def live_rows(finals):
    """Result rows for every live phrase that has text (empty ones are skipped)."""
    return [live_row(f) for f in finals if f["text"]]


# PYTHON: this block only runs when the file is started directly
# ("python simulate.py"), not when another file imports it (the simulator
# window imports it to use run(), and must not start main()).
if __name__ == "__main__":
    main()

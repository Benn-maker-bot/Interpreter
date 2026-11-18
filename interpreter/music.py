"""
music.py - noticing music and singing, and helping Whisper hear the words
=========================================================================

WHAT THIS FILE IS FOR
---------------------
Whisper (the speech-recognition AI used by this app) learned mostly from
people *talking*. When there is music, two problems appear:

  1. Phrases get chopped up too small.
     The app normally ends a phrase at a short pause. Singing has almost no
     clear pauses between notes, so a song was being cut into 1-second
     scraps. Whisper cannot understand one sung syllable on its own and
     wrote nonsense.
       Example from the logs, the same song:
         cut into 1 s pieces  ->  "Спустя."  "Готень за мной."  "Аааааа…"
         joined into ~9 s     ->  "Крыла спустил… по тропинке шел со мной"
     So during music, phrases must be allowed to be longer.

  2. The instruments drown out the voice.
     Drums and guitars are loud. Whisper hears them as noise on top of the
     words and makes more mistakes.

This file solves both, using two small AI models.

AN ANALOGY
----------
Imagine a friend listening to the radio for you and writing down what is
said:
  * First they notice "this is a song, not a talk show"   -> SoundClassifier
  * Then they wait for whole sung lines instead of single
    words before writing anything down                   -> the capture
                                                            thread does this
                                                            when music mode
                                                            is on
  * And they mentally "turn down" the band to focus on
    the singer                                            -> VocalSeparator
  * "Is a song playing right now?" is what they keep in
    mind the whole time                                   -> MusicScene

WHAT IS IN THIS FILE (top to bottom)
------------------------------------
  Settings (CAPITAL names)  numbers and names the code below uses.
  _onnx_session()  helper function: loads an AI model file so it can be run.
  SoundClassifier  class: "Does this sound like speech, singing or music?"
                   Uses YAMNet, a free Google model that recognises 521
                   kinds of sound. Small (~15 MB) and fast (~5 ms per
                   second of audio on the processor).
  VocalSeparator   class: "Remove the instruments, keep the voice."
                   Uses UVR MDX-Net "Kim_Vocal_2" (63 MB), from the free
                   Ultimate Vocal Remover project. Needs the graphics card
                   (GPU): ~250 ms per 6 seconds of audio there.
  MusicScene       class: decides once per second whether music mode is on,
                   and holds both models. The audio capture thread and the
                   transcription worker (in interpreter/audio.py and transcription.py) use it.

WORDS YOU WILL MEET
-------------------
  Sample       one number describing the sound wave at one instant.
               Sound is stored as a long list of samples.
  16 kHz       16,000 samples per second: the rate Whisper and YAMNet use.
               1 second of audio = 16,000 numbers.
  44.1 kHz     44,100 samples per second ("CD quality"). The vocal model was
               trained on this rate, so audio is converted to it first.
  float32      a number with a decimal point, e.g. -0.25. Audio as float32
               goes from -1.0 (lowest) to +1.0 (highest).
  int16        a whole number from -32768 to 32767. Sound cards deliver audio
               like this. Dividing by 32768 turns it into float32:
                 16384 / 32768 = 0.5
  dBFS         loudness on a logarithmic scale. 0 dBFS is the loudest
               possible; normal speech is about -20; -55 is very quiet;
               every -20 dB means 10 times quieter.
  ONNX         a standard file format for AI models. "onnxruntime" is the
               library that runs such a file.
  GPU / CPU    the graphics card / the main processor. AI models run much
               faster on a GPU.
  Thread       a part of the program that runs at the same time as the rest.
  Log          the file interpreter.log.jsonl where the app writes down what
               happened, e.g. {"event": "music_mode_changed", "active": true}.

PYTHON BASICS USED IN THIS FILE (explained again where they first appear)
-------------------------------------------------------------------------
  Variable     a name for a value:  level = -23.5
  Function     a named, reusable piece of code: def classify(audio): …
               Calling it runs it:  scores = classify(audio)
  Class        a blueprint for objects that bundle data and functions
               together. class SoundClassifier: … describes it;
               SoundClassifier() builds one object ("instance") from it.
  Method       a function that belongs to a class, e.g. classify().
  self         inside a method, "this object". self.session is "this
               object's session".
  __init__     the method Python runs automatically when an object is built,
               to set up its starting values.
  Attribute    a variable stored on an object: self.active = False
  List         an ordered, changeable collection:  [1, 2, 3]
  Tuple        an ordered collection that can't be changed:  ("a", "b")
  Dict         name -> value pairs:  {"speech": 0.1, "music": 0.9}
               scores["music"] reads one value (0.9 here).
  Array        NumPy's fast list of numbers (see "import numpy" below).
  None         Python's "nothing / not set yet" value.
  True/False   the two yes/no values ("booleans").

Both models are downloaded the first time from Hugging Face (a website
that hosts AI models) and are reused from the disk afterwards.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports: code from other modules that this file uses
# ─────────────────────────────────────────────────────────────────────────────
# PYTHON: "import x" loads another module (a file of ready-made code) so this
# file can use its functions as x.something. "from x import y" brings in just
# y, so it can be used without the "x." in front. "import numpy as np" gives
# numpy a shorter nickname.
import csv                          # reads .csv tables (the list of YAMNet's sound names)
import os                           # operating-system helpers (not used at the moment)
import threading                    # runs work at the same time ("threads"), and locks
import time                         # clocks: time.perf_counter() and time.monotonic()
from collections import deque       # a list with a maximum length; old items fall off

import numpy as np                  # fast maths on long lists of numbers ("arrays")

# PYTHON: a NumPy ARRAY is like a list of numbers, but all the same type and
# much faster for maths. Maths on an array works on every number at once:
#     a = np.array([1, 2, 3])
#     a * 2          ->  array([2, 4, 6])
#     a.max()        ->  3
#     a[0]           ->  1          (the first item; counting starts at 0)
#     a[1:]          ->  array([2, 3])   ("slice": from position 1 to the end)
#     a[-1]          ->  3          (negative = counted from the end)
# Arrays can have several dimensions, like a table. Its SHAPE says how big
# each dimension is: a table with 2 rows and 3 columns has shape (2, 3).
#     t[0, 2]        ->  row 0, column 2
#     t[:, 2]        ->  column 2 of every row (":" means "all")


# ─────────────────────────────────────────────────────────────────────────────
# Settings: fixed values used below
# ─────────────────────────────────────────────────────────────────────────────
# PYTHON: names in CAPITALS are "constants" by convention: values that are set
# once here and never changed while the program runs. Python doesn't enforce
# it; it's a signal to the reader.

# Where the AI models are downloaded from. Hugging Face organises files in
# "repositories" named "owner/name", like folders.
YAMNET_REPO = "audiomagic/yamnet-onnx"     # str (text): the sound classifier's repository
# PYTHON: "a, b = 1, 2" sets two variables in one line (a = 1 and b = 2).
VOCAL_REPO, VOCAL_FILE = "seanghay/uvr_models", "Kim_Vocal_2.onnx"   # the vocal separator

# Which of YAMNet's 521 sound names count as speech, singing and music.
# YAMNet gives a score for each of its sounds ("Dog", "Car horn", "Speech",
# "Rock music" …); we only care about these three groups.
# PYTHON: these are TUPLES (round brackets): fixed lists that can't be changed.
SPEECH_CLASSES  = ("Speech", "Narration, monologue", "Conversation")
SINGING_CLASSES = ("Singing", "Choir", "Chant", "Rapping", "Humming", "Vocal music",
                   "Song", "Male singing", "Female singing", "Child singing")
MUSIC_CLASSES   = ("Music", "Musical instrument", "Pop music", "Rock music",
                   "Background music", "Theme music", "Soundtrack music")

# When music mode switches on and off. A music score goes from 0.0
# (definitely not music) to 1.0 (definitely music).
# Two DIFFERENT thresholds are used on purpose ("hysteresis"):
#   * music mode turns ON  when the score is 0.45 or more,
#   * and only turns OFF   when it drops below 0.25.
# Example, scores second by second:  0.50  0.38  0.42  0.30  0.20  0.15
#   one threshold of 0.40:   ON   off   ON   off   off   off   (flickers!)
#   0.45 on / 0.25 off:      ON   ON    ON   ON    (off after 3 s below 0.25)
# Like a thermostat that heats at 19 °C but only stops at 21 °C.
MUSIC_ON, MUSIC_OFF = 0.45, 0.25     # float: score needed to turn on / to stay on
SILENT_DBFS = -55     # float: a second of audio quieter than this counts as silence
STATS_SEC = 10        # int: write a summary line to the log every 10 seconds


# ─────────────────────────────────────────────────────────────────────────────
# Helper: loading an AI model
# ─────────────────────────────────────────────────────────────────────────────

def _onnx_session(path, gpu):
    """Load an AI model file so it can be run, on the graphics card if possible.

    What it does:
        Opens the ONNX model file at `path` with onnxruntime and returns a
        "session": an object whose .run() method feeds data through the model
        and gives back its answer.

    Why it exists:
        Both models in this file need the same set-up, so it is written once
        and used twice (by SoundClassifier and VocalSeparator).

    Inputs (PYTHON: "parameters" - the values a function is given):
        path  str: where the model file is on disk,
              e.g. "C:\\Users\\you\\.cache\\huggingface\\…\\yamnet.onnx"
        gpu   bool: True = use the NVIDIA graphics card if one is available,
              False = always use the processor.

    Returns:
        an onnxruntime InferenceSession, ready to use.

    Example:
        session = _onnx_session("yamnet.onnx", gpu=False)
        result = session.run(None, {"waveform": audio})

    PYTHON: the leading underscore in the name (_onnx_session) is a habit
    meaning "only meant to be used inside this file".
    """
    # PYTHON: an import inside a function only happens when the function is
    # first called. onnxruntime takes a moment to load, so the app starts faster.
    import onnxruntime as ort
    # providers: list of str. Where the model may run, in order of preference.
    # The processor ("CPUExecutionProvider") always works, so it is the fallback.
    providers = ["CPUExecutionProvider"]
    # PYTHON: "and" - both sides must be True. "x in list" - is x one of the items?
    if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        # PYTHON: try/except - try the code; if it raises an error ("exception"),
        # run the except part instead of crashing.
        try:
            # Load NVIDIA's graphics-card libraries (installed with pip) first,
            # otherwise onnxruntime may not find them.
            ort.preload_dlls()
        except Exception:
            pass        # PYTHON: "pass" means "do nothing". Old versions lack this; carry on.
        # insert(0, x) puts x at the FRONT of the list = "try the GPU first".
        # providers is now ["CUDAExecutionProvider", "CPUExecutionProvider"].
        providers.insert(0, "CUDAExecutionProvider")
    # options: settings for how the model runs.
    options = ort.SessionOptions()
    # Use only a few processor threads: Teams/Telemost (the call itself) need
    # the processor more than these models do.
    options.intra_op_num_threads = 2    # int: threads used inside one calculation
    options.inter_op_num_threads = 1    # int: separate calculations run side by side
    return ort.InferenceSession(path, options, providers=providers)


# ─────────────────────────────────────────────────────────────────────────────
# SoundClassifier: speech, singing or music?
# ─────────────────────────────────────────────────────────────────────────────

class SoundClassifier:
    """Tells how much a piece of audio sounds like speech, singing and music.

    What it is:
        A wrapper around YAMNet, Google's sound-recognition model. Give it
        audio, and it scores 521 kinds of sound. This class boils those down
        to the three numbers we care about: speech, singing and music, each
        from 0.0 (no) to 1.0 (certainly).

    PYTHON: a CLASS is a blueprint. "class SoundClassifier:" describes what a
    sound classifier has (its data, called ATTRIBUTES) and can do (its
    functions, called METHODS). Writing SoundClassifier() builds an actual
    object from the blueprint. You can build several objects from one class.

    Attributes (the data each object keeps):
        session    the loaded YAMNet model (an onnxruntime session).
        _speech    list of int: the positions of the speech sounds among
                   YAMNet's 521 scores, e.g. [0, 1, 2].
        _singing   list of int: the positions of the singing sounds.
        _music     list of int: the positions of the music sounds.

    Example:
        classifier = SoundClassifier()      # load once (about 1 second)
        scores = classifier.classify(audio) # then call as often as needed (~5 ms)
        scores  ->  {"speech": 0.02, "singing": 0.31, "music": 0.88, "classify_ms": 4.7}
        scores["music"]  ->  0.88           # "very likely music"
    """

    def __init__(self):
        """Set up a new classifier: load YAMNet and find the sounds we need.

        PYTHON: __init__ (two underscores on each side) is a special method.
        Python runs it automatically when you write SoundClassifier(). Its
        job is to give the new object its starting attributes. "self" is the
        new object itself, so "self.session = …" stores the session ON the
        object, where the other methods can find it later.

        Steps:
          1. Download the model file (only the first time; afterwards it is
             found on disk) and load it on the processor. YAMNet is so small
             that the processor is fast enough, which keeps the graphics card
             free for Whisper.
          2. Read the list of YAMNet's sound names from a .csv table.
             Each row looks like:  0,/m/09x0r,Speech
             (number, code, name).
          3. Remember the POSITIONS of the names we want. If "Music" is the
             133rd name, its score will be at position 132 (counting from 0)
             in every answer YAMNet gives. classify() uses these positions.
        """
        # hf_hub_download(repo, file) returns the file's path on disk,
        # downloading it first if it isn't there yet.
        from huggingface_hub import hf_hub_download
        # Step 1
        self.session = _onnx_session(hf_hub_download(YAMNET_REPO, "yamnet.onnx"), gpu=False)
        # Step 2
        # PYTHON: "with open(path) as f:" opens the file, gives it the name f
        # for the indented block, and closes it again automatically afterwards
        # (even if something goes wrong).
        with open(hf_hub_download(YAMNET_REPO, "yamnet_class_map.csv"), encoding="utf-8") as f:
            # PYTHON: this is a LIST COMPREHENSION - a one-line way to build a list:
            #     [what_to_keep for item in collection]
            # csv.reader(f) gives each row as a list, e.g. ["0", "/m/09x0r", "Speech"].
            # [1:] skips the first row (the column headings).
            # row[2] is the third column (counting from 0): the sound's name.
            # names: list of str, 521 items: ["Speech", "Child speech", …, "Field recording"]
            names = [row[2] for row in list(csv.reader(f))[1:]]
        # Step 3
        # PYTHON: a LAMBDA is a tiny one-line function without a name:
        #     lambda wanted: <result>    is like    def pick(wanted): return <result>
        # PYTHON: enumerate(names) gives pairs (position, item):
        #     (0, "Speech"), (1, "Child speech"), (2, "Conversation"), …
        # So pick(("Speech", "Conversation")) returns the positions of those names,
        # e.g. [0, 2].
        pick = lambda wanted: [i for i, n in enumerate(names) if n in wanted]
        # Three attributes set in one line (see "a, b = 1, 2" above).
        self._speech, self._singing, self._music = (pick(SPEECH_CLASSES), pick(SINGING_CLASSES),
                                                    pick(MUSIC_CLASSES))

    # PYTHON: "audio_16k: np.ndarray" and "-> dict" are TYPE HINTS: notes saying
    # what kind of value goes in (a NumPy array) and comes out (a dict). Python
    # doesn't check them, but VS Code uses them to help you.
    def classify(self, audio_16k: np.ndarray) -> dict:
        """Score a piece of audio for speech, singing and music.

        Inputs:
            audio_16k  NumPy array of float numbers: the audio at 16 kHz,
                       e.g. 1 second = 16,000 numbers between -1.0 and 1.0.

        Returns:
            a dict with four entries:
              "speech", "singing", "music"   float, 0.0 (no) to 1.0 (certainly)
              "classify_ms"                  float, how long it took in ms

        How the scores are made:
            YAMNet looks at the audio in short overlapping slices ("frames",
            each ~0.96 s long, starting every ~0.48 s) and scores all 521
            sounds for every frame. Its answer is a table:
                           Speech  …  Music  …  Singing  …  (521 columns)
                frame 0     0.02      0.91       0.40
                frame 1     0.01      0.88       0.05
            * speech and music: average each column over all frames, then
              take the highest average in the group.
                music = average of (0.91, 0.88) = 0.895 -> 0.895
            * singing: take the highest single frame (0.40 here) instead of
              the average (0.225), because YAMNet only notices singing in
              places and an average would hide it.

        Example:
            classify(one_second_of_pop_song)
            -> {"speech": 0.03, "singing": 0.40, "music": 0.895, "classify_ms": 4.9}
        """
        # audio: the same sound as 32-bit decimals (what the model expects).
        audio = audio_16k.astype(np.float32)
        # YAMNet needs at least 1 second (16,000 samples).
        if len(audio) < 16000:
            # np.pad(audio, (0, extra)) adds `extra` zeros (= silence) at the end.
            # Example: 10,000 samples -> 6,000 zeros added -> 16,000 samples.
            audio = np.pad(audio, (0, 16000 - len(audio)))
        # t0: float, the time now in seconds. perf_counter() is a precise stopwatch;
        # only the DIFFERENCE between two readings is meaningful.
        t0 = time.perf_counter()
        # Run the model. self.session.run(...) returns a list of outputs; [0] is
        # the first one: the score table, shape (frames, 521).
        # frames: 2-D array, e.g. shape (2, 521) for 1 second of audio.
        frames = self.session.run(None, {"waveform": audio})[0]
        # mean: 1-D array of 521 scores, the average of each column.
        # .mean(0) averages along dimension 0 (down the rows).
        mean = frames.mean(0)
        # mean[self._speech] picks only the speech positions, e.g. mean[[0, 1, 2]];
        # .max() keeps the highest. round(x, 3) keeps 3 decimals (0.89512 -> 0.895).
        # float(...) turns NumPy's number into a plain Python float, because the
        # JSON log can't store NumPy numbers.
        return {"speech":  round(float(mean[self._speech].max()), 3),
                "singing": round(float(frames[:, self._singing].max()), 3),  # all rows, singing columns
                "music":   round(float(mean[self._music].max()), 3),
                # (now - start) is in seconds; x 1000 gives milliseconds, e.g. 0.0049 -> 4.9
                "classify_ms": round((time.perf_counter() - t0) * 1000, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# VocalSeparator: remove the instruments, keep the voice
# ─────────────────────────────────────────────────────────────────────────────

class VocalSeparator:
    """Removes the instruments from audio and keeps the voice.

    What it is:
        A wrapper around UVR MDX-Net "Kim_Vocal_2", an AI model trained to
        separate vocals from music. In goes a song, out comes just the singer.

    How it works, in plain words:
        1. The audio is turned into a SPECTROGRAM: a picture of the sound,
           with time going left to right and pitch (frequency) bottom to top.
           A deep voice is a bright line low down; a cymbal is a smear high
           up. Voices and instruments occupy different parts of the picture.
        2. The AI looks at the picture and keeps only the voice parts.
        3. The picture is turned back into sound.
        The maths for steps 1 and 3 is the Fourier transform ("FFT"): it
        splits sound into the frequencies it is made of, and back again.
        Like separating white light into a rainbow with a prism.

    Why the audio is cut into chunks:
        The model always looks at exactly ~5.9 seconds at a time (a
        "chunk"). Longer audio is processed chunk after chunk. The edges of
        each chunk's result are less accurate (the model can't see what
        comes before or after), so chunks overlap and their edges are thrown
        away ("trim"). Example for a 12-second phrase:
            chunk 1: 0.0 - 5.9 s   keep  0.0 - 5.75 s
            chunk 2: 5.66 - 11.6 s keep 5.75 - 11.5 s
            chunk 3: 11.3 - 17.2 s keep 11.5 - 12.0 s  (the rest is padding)

    Class attributes (fixed numbers shared by every VocalSeparator; they come
    from how the model was trained and must not be changed):
        N_FFT       int 7680: samples in each FFT slice (~0.17 s)
        HOP         int 1024: how far each slice starts after the previous
                    one (~0.023 s), so slices overlap a lot
        DIM_F       int 3072: frequency rows the model uses (the highest
                    frequencies are left out; voices don't reach them)
        DIM_T       int 256: time columns (slices) per chunk
        COMPENSATE  float 1.009: a small volume correction the model needs
                    (+0.9 %)
        RATE        int 44100: the model works at CD quality

    Attributes set in __init__ (each object's own data):
        session      the loaded model.
        on_gpu       bool: True if it really runs on the graphics card.
        chunk        int 261,120: samples per chunk (~5.9 s at 44.1 kHz).
        trim         int 3,840: samples thrown away at each edge (~0.09 s).
        step         int 253,440: distance between chunk starts (~5.75 s).
        window       array of 7680 floats: the "fade in / fade out" curve.
        _norm        array: correction for overlapping slices.
        _frame_index 2-D array (256 x 7680): where each slice starts and ends.

    The model also expects stereo (left and right), so mono audio is simply
    copied into both channels.

    Example:
        separator = VocalSeparator()            # load once (a few seconds)
        voice = separator.separate(song_audio)  # ~250 ms per 6 s on the GPU
        len(voice) == len(song_audio)           # same length, instruments gone
    """

    # PYTHON: variables written directly inside the class (not inside a method)
    # are CLASS ATTRIBUTES: shared by every object. Read them as self.N_FFT.
    N_FFT, HOP, DIM_F, DIM_T, COMPENSATE, RATE = 7680, 1024, 3072, 256, 1.009, 44100

    def __init__(self):
        """Set up a new separator: load the model and work out the chunk sizes.

        The three sizes (all counted in samples at 44.1 kHz):
          chunk = HOP x (DIM_T - 1) = 1024 x 255 = 261,120 samples = ~5.92 s
                  how much audio one run of the model looks at
          trim  = N_FFT // 2 = 7680 // 2 = 3,840 samples = ~0.09 s
                  how much of each edge of a chunk's result is thrown away
          step  = chunk - 2 x trim = 261,120 - 7,680 = 253,440 = ~5.75 s
                  how far apart chunks start, so the kept middles join up
                  exactly, with no gap and no overlap

        Three things used on every chunk are also prepared now, so they
        don't have to be recalculated each time:
          window        the smooth "fade in, fade out" curve (a Hann window)
          _norm         the overlap correction
          _frame_index  the positions of all 256 slices
        """
        from huggingface_hub import hf_hub_download
        self.session = _onnx_session(hf_hub_download(VOCAL_REPO, VOCAL_FILE), gpu=True)
        # on_gpu: bool. get_providers() lists where the model actually runs.
        self.on_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        self.chunk = self.HOP * (self.DIM_T - 1)     # 261,120
        # PYTHON: "//" divides and drops the decimals: 7 // 2 = 3 (but 7 / 2 = 3.5).
        self.trim = self.N_FFT // 2                   # 3,840
        self.step = self.chunk - 2 * self.trim        # 253,440
        # n: array 0, 1, 2, … 7679 (np.arange(k) makes the numbers 0 to k-1).
        n = np.arange(self.N_FFT)
        # window: the Hann window, a smooth hump: 0 at both ends, 1 in the middle.
        #   position:  0     1920    3840    5760    7679
        #   value:     0.0   0.5     1.0     0.5     0.0
        # Each slice of audio is multiplied by it so its edges fade in and out
        # instead of starting abruptly (which would add a "click" to the maths).
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * n / self.N_FFT)).astype(np.float32)
        # When the overlapping slices are added back together (in _istft), the
        # middle of the chunk is covered by more slices than its edges, so it
        # would come out too loud. norm adds up how much each position is covered…
        # np.zeros(k) makes an array of k zeros.
        norm = np.zeros(self.N_FFT + self.HOP * (self.DIM_T - 1), np.float32)
        # PYTHON: "for t in range(256):" repeats the indented line with t = 0, 1, 2, … 255.
        for t in range(self.DIM_T):
            # a[start:end] += x adds x to positions start .. end-1.
            # Slice t covers samples t*1024 to t*1024+7679.
            norm[t * self.HOP:t * self.HOP + self.N_FFT] += self.window ** 2   # ** = "to the power of"
        # …so _istft can divide by it to even things out.
        # np.maximum(norm, 1e-8) replaces anything smaller than 0.00000001 by
        # 0.00000001, so we never divide by zero at the very edges.
        self._norm = np.maximum(norm, 1e-8)
        # _frame_index: a table with one row per slice, listing the sample
        # positions in that slice:
        #   row 0:  0,    1,    2,    …  7679
        #   row 1:  1024, 1025, 1026, …  8703
        #   row 2:  2048, …
        # [:, None] turns a list into a column, [None, :] into a row; adding a
        # column and a row makes NumPy build the whole table ("broadcasting").
        self._frame_index = (np.arange(self.DIM_T)[:, None] * self.HOP
                             + np.arange(self.N_FFT)[None, :])

    def _stft(self, x):
        """Turn sound into a spectrogram ("Short-Time Fourier Transform").

        Inputs:
            x  2-D array, shape (2, 261120): stereo audio, one row per channel
               (left, right), one chunk long.

        Returns:
            3-D array of complex numbers, shape (2, 3841, 256):
              2      channels (left, right)
              3841   frequencies, from 0 Hz (bottom) to 22,050 Hz (top)
              256    time slices, left to right
            A complex number stores two things: how strong that frequency is
            ("magnitude") and its exact timing ("phase"). Both are needed to
            turn the picture back into sound later.

        Steps:
          1. Pad half a slice of mirrored audio at both ends, so the first
             and last slices are centred on real audio, not on the edge.
          2. Cut the audio into 256 overlapping slices in one go (using
             _frame_index) and fade each slice's edges with the window.
          3. Run the FFT on every slice at once.

        Speed note: scipy's FFT with 32-bit numbers and 2 threads is used;
        NumPy's 64-bit FFT took ~150 ms of each ~300 ms chunk.
        """
        from scipy import fft
        # Step 1. The pad sizes are given per dimension: ((0, 0), (3840, 3840)) means
        # "channels: add nothing; samples: add 3840 before and 3840 after".
        # mode="reflect" mirrors the audio: [1, 2, 3] padded by 2 -> [3, 2, 1, 2, 3, 2, 1].
        x = np.pad(x, ((0, 0), (self.N_FFT // 2, self.N_FFT // 2)), mode="reflect")
        # Step 2. Indexing with a table of positions picks ALL slices at once.
        # frames: shape (2 channels, 256 slices, 7680 samples).
        frames = (x[:, self._frame_index] * self.window).astype(np.float32)
        # Step 3. rfft = FFT for real (not complex) input; axis=-1 = along the
        # last dimension (the samples of each slice); workers=2 = two threads.
        # .transpose(0, 2, 1) swaps the last two dimensions so frequencies come
        # before time slices: the order the model expects.
        return fft.rfft(frames, axis=-1, workers=2).transpose(0, 2, 1)

    def _istft(self, spec):
        """Turn a spectrogram back into sound (the reverse of _stft).

        Inputs:
            spec  3-D complex array, shape (2, 3841, 256): the voice-only
                  spectrogram from the model.

        Returns:
            2-D array, shape (2, 261120): one chunk of audio per channel.

        Steps:
          1. Inverse FFT: turn each time slice back into 7680 samples.
          2. Fade each slice's edges with the window again.
          3. Add the overlapping slices together at their positions
             ("overlap-add"): slice 0 at sample 0, slice 1 at 1024, …
          4. Divide by _norm to undo the extra loudness where slices overlap.
          5. Remove the half-slice padding _stft added at both ends.
        """
        from scipy import fft
        # Steps 1-2. frames: shape (2, 256, 7680).
        frames = fft.irfft(spec.transpose(0, 2, 1), n=self.N_FFT, axis=-1, workers=2) * self.window
        # out: the result, silent to begin with. spec.shape[0] = 2 channels.
        out = np.zeros((spec.shape[0], len(self._norm)), np.float32)
        for t in range(self.DIM_T):                                       # step 3
            out[:, t * self.HOP:t * self.HOP + self.N_FFT] += frames[:, t]
        out /= self._norm                                                 # step 4
        # Step 5: keep samples 3840 .. 3840+261120 (the real audio).
        return out[:, self.N_FFT // 2:self.N_FFT // 2 + self.chunk]

    # ── Re-using work on a phrase that is still growing ─────────────────────
    # While someone sings, the phrase gets longer every moment. Chunks at the
    # start of it won't change any more, so they can be separated early and
    # kept ("cached") until the phrase ends (see prefetch()).
    # But the audio is converted from 16 kHz to 44.1 kHz first, and the
    # conversion of the last few thousand samples can still change slightly
    # when more audio arrives. So a chunk is only kept if it ends at least
    # this many samples (at 44.1 kHz: 4096 / 44100 = ~0.09 s) before the end
    # of the audio that has arrived.
    STABLE_MARGIN = 4096      # int, samples

    def _chunk_voice(self, piece):
        """Separate the voice in ONE chunk.

        Inputs:
            piece  2-D array, shape (2, 261120): stereo audio at 44.1 kHz.

        Returns:
            1-D array of 253,440 floats: the voice only, mono, just the
            trimmed middle of the chunk (~5.75 s).

        Steps:
          1. Make the spectrogram; keep only the 3072 lowest frequency rows
             (up to ~17.6 kHz), which is what the model was trained on.
          2. Split each complex number into its two parts ("real" and
             "imaginary"): the model wants 4 plain tables ("planes"):
             left-real, left-imaginary, right-real, right-imaginary.
          3. Run the AI model.
          4. Put the model's 4 planes back together into complex numbers,
             adding empty rows for the frequencies left out in step 1.
          5. Back to sound, cut off the inaccurate edges, and average left
             and right into one channel (mono).

        Speed note: chunks go through the model one at a time, never as a
        group ("batch"). The graphics card's maths library (cuDNN) picks its
        fastest method once for a given input size. A batch had a different
        size for every phrase, so it searched again each time: 2-4x slower.
        """
        # Step 1. [:, :3072] = all channels, frequency rows 0 to 3071.
        spec = self._stft(piece)[:, :self.DIM_F]
        # Step 2. .real / .imag take the two parts of complex numbers.
        # np.stack([...]) piles the 4 tables into one array: shape (4, 3072, 256).
        planes = np.stack([spec[0].real, spec[0].imag, spec[1].real, spec[1].imag])
        # Step 3. planes[None] adds a dimension of size 1 at the front: shape
        # (1, 4, 3072, 256). Models expect a "batch" (a group), even of one.
        # The model's answer: [0] = its first output, [0] = the first (only)
        # item of the batch -> out: shape (4, 3072, 256).
        out = self.session.run(None, {"input": planes[None].astype(np.float32)})[0][0]
        # Step 4. full: empty complex spectrogram of the full size (2, 3841, 256).
        full = np.zeros((2, self.N_FFT // 2 + 1, self.DIM_T), np.complex64)
        # PYTHON: "1j" is Python's imaginary unit; a + 1j * b builds the complex number a+bi.
        full[0, :self.DIM_F] = out[0] + 1j * out[1]        # left channel
        full[1, :self.DIM_F] = out[2] + 1j * out[3]        # right channel
        # Step 5. [:, 3840:-3840] = drop 3840 samples at both ends.
        # .mean(0) = average of the two channels -> mono.
        return self._istft(full)[:, self.trim:-self.trim].mean(0)

    def _prepare(self, audio_16k, cache):
        """Get audio ready for chunking: convert it to 44.1 kHz stereo and pad it.

        Inputs:
            audio_16k  1-D float array: the phrase's audio at 16 kHz (mono).
            cache      dict or None: the chunks already separated for this
                       phrase. If it belongs to a DIFFERENT phrase, it is
                       emptied here.

        Returns (three values):
            mix   2-D array (2, length): padded stereo audio at 44.1 kHz.
            n     int: how many samples of real audio it contains.
            pad   int: how much silence was added at the end, so the length
                  is a whole number of steps.

        Example (a 12-second phrase):
            16 kHz: 192,000 samples  ->  44.1 kHz: n = 529,200 samples
            529,200 / 253,440 = 2.09 steps -> round up to 3 steps = 760,320
            pad = 760,320 - 529,200 = 231,120 samples of silence
            mix = 3,840 silence + 529,200 audio + 231,120 + 3,840 silence

        How "a different phrase" is spotted:
            the cache remembers the first 0.1 s of its phrase ("head"). A
            phrase that is still growing always starts the same way; a new
            phrase starts differently.

        The cache dict looks like this:
            {"head": array of 1600 samples,
             "pieces": {0: voice of chunk 1, 253440: voice of chunk 2},
             "reused": 2, "chunks": 3}
        """
        import soxr                        # a fast, high-quality sample-rate converter
        # PYTHON: "is not None" checks that a value was given.
        if cache is not None:
            # head: the first 1600 samples = 0.1 s at 16 kHz.
            head = audio_16k[:1600]
            # PYTHON: cache.get("head") reads a dict value, or gives None if it
            # isn't there (cache["head"] would raise an error instead).
            # A "\" at the end of a line continues the statement on the next line.
            if cache.get("head") is None or len(cache["head"]) != len(head) \
                    or not np.array_equal(cache["head"], head):
                cache.clear()                                    # forget the old phrase
                # .copy(): store an independent copy, not a view into audio_16k.
                cache.update(head=head.copy(), pieces={})        # pieces: chunk start -> voice
        # x: the audio at 44.1 kHz (2.76 times as many samples).
        x = soxr.resample(audio_16k.astype(np.float32), 16000, self.RATE).astype(np.float32)
        n = len(x)                         # len(...) = how many items
        # PYTHON: "%" is the remainder after division: 10 % 4 = 2.
        pad = self.step - n % self.step
        # np.concatenate([a, b, c]) joins arrays end to end.
        mix = np.concatenate([np.zeros(self.trim, np.float32), x,
                              np.zeros(pad + self.trim, np.float32)])
        # PYTHON: "return a, b, c" returns three values at once (as a tuple); the
        # caller unpacks them: mix, n, pad = self._prepare(...)
        # np.stack([mix, mix]): the same audio as left AND right channel = stereo.
        return np.stack([mix, mix]), n, pad

    def _stable(self, i, n):
        """Is the chunk starting at sample i safe to keep for later?

        Returns True if the chunk only covers audio that won't change any
        more: its last kept sample is at least STABLE_MARGIN before the end
        of the audio that has arrived (n).

        Example: n = 529,200 (12 s so far)
            chunk at i = 0:        ends at 257,280 -> True  (safe)
            chunk at i = 253,440:  ends at 510,720 -> True  (safe)
            chunk at i = 506,880:  ends at 764,160 -> False (not finished yet)
        """
        return i + self.chunk - self.trim <= n - self.STABLE_MARGIN

    def prefetch(self, audio_16k, cache, max_chunks=1):
        """Do some of the separation work early, while a phrase is still going.

        Why:
            Separating a 12-second sung phrase takes ~0.75 s if it is all done
            after the singer stops, so the text appears that much later. But
            the first chunks of the phrase stopped changing long ago. Doing
            them in advance leaves only the last chunk (~0.25 s) for the end.
            Measured: 662 ms -> 227 ms for a 16-second phrase.

        When it is called:
            by the transcription worker after each live preview while music
            is playing (interpreter/transcription.py: _prefetch_separation).

        Inputs:
            audio_16k   1-D array: the phrase so far, at 16 kHz.
            cache       dict: remembers finished chunks between calls.
            max_chunks  int (default 1): at most this many chunks per call
                        (~0.25 s each), so the worker is never busy for long.

        PYTHON: "max_chunks=1" gives the parameter a DEFAULT value: callers
        may leave it out, and it is then 1.

        Returns:
            int: how many chunks were separated in this call (0 if none was
            ready yet).

        Example, while someone sings a 12-second line:
            after 4 s:  prefetch(...) -> 0   (no chunk complete yet)
            after 7 s:  prefetch(...) -> 1   (chunk 1 done and kept)
            after 12 s: prefetch(...) -> 1   (chunk 2 done and kept)
            then separate(...) only has chunk 3 left to do.
        """
        mix, n, pad = self._prepare(audio_16k, cache)
        done = 0                                         # int: chunks separated so far
        # PYTHON: range(start, stop, step) counts start, start+step, … below stop.
        # range(0, 760320, 253440) -> 0, 253440, 506880: the start of every chunk.
        for i in range(0, n + pad, self.step):
            if done >= max_chunks or not self._stable(i, n):
                break                        # PYTHON: "break" leaves the loop at once
            if i not in cache["pieces"]:     # not separated before
                # mix[:, i:i + chunk] = both channels, samples i to i+chunk-1
                cache["pieces"][i] = self._chunk_voice(mix[:, i:i + self.chunk])
                done += 1                    # PYTHON: "+= 1" adds one (done = done + 1)
        return done

    def separate(self, audio_16k: np.ndarray, cache=None) -> np.ndarray:
        """Remove the instruments from a phrase and return just the voice.

        Inputs:
            audio_16k  1-D float array: the finished phrase, 16 kHz mono.
            cache      dict or None (default): what prefetch() collected while
                       the phrase was still going. Chunks found there are
                       re-used instead of being separated again.

        Returns:
            1-D float array: the voice only, 16 kHz mono, exactly as long as
            the input.

        Steps:
          1. Prepare the audio (44.1 kHz stereo, padded).
          2. For every chunk: take it from the cache if it is there,
             otherwise separate it now (and keep it if it is stable).
          3. Join the chunks' voices, remove the padding, apply the small
             volume correction, and convert back to 16 kHz.
        It also writes into the cache how many chunks were re-used, so the
        phrase's log line can show it ("separation_reused": "2/3").

        Example:
            voice = separator.separate(phrase, cache)
            cache["reused"], cache["chunks"]  ->  (2, 3)
            # 2 of the 3 chunks came from prefetch(); only 1 was done now.
        """
        import soxr
        mix, n, pad = self._prepare(audio_16k, cache)                    # step 1
        # pieces: list of 1-D arrays, the voice of each chunk in order.
        # reused: int, how many came from the cache.
        pieces, reused = [], 0
        for i in range(0, n + pad, self.step):                           # step 2
            if cache is not None and i in cache["pieces"]:
                pieces.append(cache["pieces"][i])     # .append(x) adds x to the end of a list
                reused += 1
                continue                              # PYTHON: "continue" jumps to the next round
            piece = self._chunk_voice(mix[:, i:i + self.chunk])
            if cache is not None and self._stable(i, n):
                cache["pieces"][i] = piece
            pieces.append(piece)
        if cache is not None:
            cache["reused"], cache["chunks"] = reused, len(pieces)
        # Step 3. [:n] drops the padding at the end; x COMPENSATE = +0.9 % volume.
        voice = np.concatenate(pieces)[:n] * self.COMPENSATE
        # Back to 16 kHz, cut to exactly the input's length (rounding can add a sample).
        return soxr.resample(voice, self.RATE, 16000).astype(np.float32)[:len(audio_16k)]


# ─────────────────────────────────────────────────────────────────────────────
# MusicScene: is music playing right now?
# ─────────────────────────────────────────────────────────────────────────────

class MusicScene:
    """Keeps track of whether music is playing right now ("music mode").

    Who uses it:
        * The audio capture thread hands it the last second of audio every
          second (feed()), and reads `active` to know how long phrases may be.
        * The transcription worker reads `active`, uses `separator` to remove
          instruments, and calls judge() to score each finished phrase.

    Attributes:
        enabled          bool: music detection switched on in the settings?
        want_separation  bool: also remove instruments (if a GPU allows)?
        active           bool: True while music mode is on.
        singing          bool: True if the last music phrase had a voice in it
                         (set by the transcription worker).
        classifier       SoundClassifier, or None until it has loaded.
        separator        VocalSeparator, or None until loaded (and stays None
                         without a usable graphics card).
        error            str or None: why loading failed, if it did.
        _windows         deque of the last 8 seconds' results, each a tuple
                         (music score, was it silent?), e.g.
                         deque([(0.12, False), (0.0, True), (0.71, False)])
        _busy            bool: True while one second is being checked.
        _stats           dict: counters for the next summary log line.
        _log             function used to write log lines.
        _lock            threading.Lock: lets only one thread update the
                         decision at a time.

    Loading the models takes a few seconds, so it happens in the background.
    Until they are ready, `active` simply stays False: the app works
    normally, just without music mode.

    Example:
        scene = MusicScene(log=my_log_function)  # returns at once; loads in the background
        ...                                      # a few seconds later:
        scene.feed(last_second_of_audio)         # once per second, from the capture thread
        if scene.active:                         # music playing?
            voice = scene.separator.separate(phrase)
    """

    def __init__(self, enabled=True, separate=True, log=None, models_from=None):
        """Create the music scene and start loading the models in the background.

        Inputs:
            enabled      bool (default True). False = never detect music (music
                         mode switched off in the settings).
            separate     bool (default True). False = detect music but don't
                         remove the instruments.
            log          a function to write log lines with, called as
                         log("event_name", field=value, …). Default None = a
                         function that does nothing.
            models_from  another MusicScene whose models to share (default
                         None = load our own). The interpreter listens to two
                         audio sources at once; each needs its own scene (its
                         own "is music playing?" decision), but both can use
                         one copy of the models, which saves memory and time.

        This returns immediately; the models load in a separate thread (_load).
        `loaded` is set once loading has finished (or failed).
        """
        # Two attributes set in one line from the two parameters.
        self.enabled, self.want_separation = enabled, separate
        self.active = False                 # bool: music mode on?
        self.singing = False                # bool: ...and a voice in the music?
        # PYTHON: "a = b = None" sets both to None.
        self.classifier = self.separator = None
        self.error = None                   # str or None
        # PYTHON: a DEQUE ("deck", from "double-ended queue") with maxlen=8 keeps
        # at most 8 items: adding a 9th automatically drops the oldest. Perfect
        # for "the last 8 seconds".
        self._windows = deque(maxlen=8)
        self._busy = False
        self._stats = self._new_stats()
        # PYTHON: "x or y" gives x if x is something, otherwise y. So: use the log
        # function we were given, or else a lambda that accepts anything and does
        # nothing. (*a collects any positional arguments, **k any named ones.)
        self._log = log or (lambda *a, **k: None)
        # PYTHON: a LOCK makes threads take turns. "with self._lock:" waits until
        # no other thread is inside a block guarded by the same lock.
        self._lock = threading.Lock()
        # PYTHON: an EVENT is a flag that threads can wait for: loaded.wait()
        # blocks until another thread calls loaded.set().
        self.loaded = threading.Event()
        if enabled and models_from is not None:
            # Share the other scene's models once it has finished loading them.
            threading.Thread(target=self._borrow_models, args=(models_from,),
                             name="music-models-share", daemon=True).start()
        elif enabled:
            # PYTHON: threading.Thread(target=f) prepares a thread that will run f;
            # .start() starts it. The rest of the program carries on immediately.
            # daemon=True: this thread won't keep the program open when it closes.
            threading.Thread(target=self._load, name="music-models", daemon=True).start()
        else:
            self.loaded.set()               # nothing to load

    def _borrow_models(self, source):
        """Wait until `source` has loaded its models, then use the same ones
        (runs in a background thread). The models only read their settings
        while working, and onnxruntime lets several threads run one model,
        so two scenes can safely share them."""
        source.loaded.wait()
        self.classifier, self.separator, self.error = (source.classifier, source.separator,
                                                       source.error)
        self.loaded.set()
        self._log("music_models_shared", classifier=self.classifier is not None,
                  separator=self.separator is not None)

    def _load(self):
        """Load the models, then set `loaded` (also if loading failed)."""
        try:
            self._load_models()
        finally:
            self.loaded.set()

    def _load_models(self):
        """Load the models (runs in a background thread).

        Steps:
          1. Load YAMNet. If that fails, stop here: no music detection.
          2. If instrument removal is wanted, load the vocal separator. Only
             keep it if it runs on the graphics card: on the processor it
             takes ~1.5 s per phrase and would make the call stutter.
          3. "Warm up" the separator once on 1 second of silence, because
             the very first run on a graphics card is always slow (it sets
             itself up).

        Errors are written to the log, never raised: the worst case is that
        music mode stays off and everything else keeps working.

        Log lines it writes, for example:
            {"event": "sound_classifier_ready", "elapsed_ms": 812}
            {"event": "vocal_separator_ready", "elapsed_ms": 3410}
            {"event": "vocal_separator_skipped", "reason": "GPU not available to onnxruntime"}
        """
        t0 = time.perf_counter()
        try:                                                              # step 1
            self.classifier = SoundClassifier()
            self._log("sound_classifier_ready", elapsed_ms=round((time.perf_counter() - t0) * 1000))
        # PYTHON: "except Exception as e" catches the error and names it e.
        # str(e) is its message, e.g. "No such file or directory".
        except Exception as e:
            # PYTHON: an f-STRING (f"...") puts values inside text: {e} is replaced
            # by the error message -> "classifier: No such file or directory".
            self.error = f"classifier: {e}"
            self._log("sound_classifier_failed", error=str(e))
            return                          # PYTHON: "return" leaves the function now
        if self.want_separation:                                          # step 2
            t0 = time.perf_counter()
            try:
                separator = VocalSeparator()
                if not separator.on_gpu:
                    self._log("vocal_separator_skipped", reason="GPU not available to onnxruntime")
                    return
                # Step 3: np.zeros(16000, np.float32) = 1 second of silence.
                separator.separate(np.zeros(16000, np.float32))
                self.separator = separator  # only now is it made available to others
                self._log("vocal_separator_ready", elapsed_ms=round((time.perf_counter() - t0) * 1000))
            except Exception as e:
                self._log("vocal_separator_failed", error=str(e))

    # ── Deciding about music mode, once per second ──────────────────────────
    # The first version judged whole phrases, which was too slow: it needed
    # three phrases (up to 15 s each during music) to switch, and during
    # silence no phrases arrive at all, so it got stuck in music mode. The
    # logs showed ~14 s to notice music. Now the capture thread hands over
    # the last second of audio every second; YAMNet needs only ~5 ms for it,
    # so music mode switches within 2-3 seconds.

    def feed(self, audio_16k):
        """Hand over the latest second of audio to be checked for music.

        Called by: the audio capture thread, once per second.

        Inputs:
            audio_16k  1-D array: one second of 16 kHz audio (16,000 numbers,
                       whole numbers or decimals).

        It never makes the caller wait: the checking runs in its own short
        background thread (_classify_window). If the previous second is
        still being checked, this second is simply skipped - the audio
        capture must never be held up, or sound would be lost.
        """
        if self.classifier is None or self._busy:
            return                          # not loaded yet, or still busy: skip this second
        self._busy = True
        # args=(audio_16k,) passes the audio to _classify_window.
        # PYTHON: (x,) with a comma is a tuple of ONE item; (x) without it is just x.
        threading.Thread(target=self._classify_window, args=(audio_16k,),
                         name="music-scene", daemon=True).start()

    def _classify_window(self, audio):
        """Check one second of audio (runs in a short background thread).

        Steps:
          1. Make sure the audio is in decimal form (-1.0 .. 1.0).
          2. Measure its loudness in dBFS. Below SILENT_DBFS (-55) it counts
             as silence, and YAMNet isn't run at all (saves time).
          3. Otherwise score it with YAMNet.
          4. Update the music-mode decision (_update).

        Loudness examples:
            normal speech         about -20 dBFS
            quiet background      about -45 dBFS
            silence (nothing on)  below -55 dBFS  -> not classified

        PYTHON: "finally:" runs at the end whether or not there was an error,
        so _busy is always set back to False and the next second can be
        checked.
        """
        try:
            audio = np.asarray(audio)                         # step 1: make sure it's an array
            # .dtype is the type of the numbers in an array.
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0    # e.g. 16384 -> 0.5
            # Step 2: loudness = 10 x log10(average of the squared samples).
            # Example: samples averaging 0.1 in size -> squares average 0.01 ->
            #          10 x log10(0.01) = 10 x -2 = -20 dBFS.
            # The tiny 1e-12 avoids log10(0) (impossible) for perfect silence.
            # float(...) / bool(...) turn NumPy values into plain Python ones,
            # which the JSON log can store.
            level = float(10 * np.log10(float(np.mean(audio ** 2)) + 1e-12))
            silent = bool(level < SILENT_DBFS)                # e.g. -62.0 < -55 -> True
            # Step 3. PYTHON: "A if condition else B" chooses between two values.
            scores = ({"speech": 0.0, "singing": 0.0, "music": 0.0, "classify_ms": 0.0}
                      if silent else self.classifier.classify(audio))
            self._update(scores, silent, level)               # step 4
        except Exception as e:
            self._log("music_scene_failed", error=str(e))
        finally:
            self._busy = False

    def _update(self, scores, silent, level):
        """Add one second's result and decide whether music mode is on.

        Inputs:
            scores  dict from classify(), e.g. {"music": 0.62, "speech": 0.05, …}
            silent  bool: was this second silent?
            level   float: its loudness in dBFS, e.g. -21.3

        The rules:
          Turn ON   when at least 2 of the last 3 seconds scored as music
                    (MUSIC_ON, 0.45, or more).
          Turn OFF  after 2 silent seconds in a row, or 3 seconds in a row
                    with no music (below MUSIC_OFF, 0.25) and not silent.
        Needing 2 or 3 seconds (not just 1) stops a single drum hit or a
        cough from switching the mode.

        Examples (newest second last):
            OFF, last 3 scores 0.10, 0.62, 0.71     -> 2 of 3 >= 0.45 -> ON ("music")
            ON,  last 2 seconds silent, silent      -> OFF ("silence")
            ON,  last 3 scores 0.12, 0.08, 0.20     -> all < 0.25     -> OFF ("speech")
            ON,  last 3 scores 0.30, 0.12, 0.08     -> 0.30 isn't < 0.25 -> stays ON

        It also writes to the log:
          * every switch, with the reason and the last 3 seconds:
              {"event": "music_mode_changed", "active": true, "reason": "music",
               "last_seconds": [0.1, 0.62, 0.71], "level_dbfs": -18.4}
          * a summary every STATS_SEC (10) seconds ("music_scene_stats").
        """
        # Only one thread at a time may run this block (see _lock in __init__).
        with self._lock:
            self._windows.append((float(scores["music"]), silent))
            # Add this second to the summary counters. st is a short name for the
            # same dict (not a copy): changing st changes self._stats.
            st = self._stats
            st["windows"] += 1
            st["silent"] += silent           # PYTHON: True counts as 1, False as 0
            st["music_sum"] += scores["music"]
            st["speech_sum"] += scores["speech"]
            # .get("classify_ms", 0): the value, or 0 if it's missing.
            st["classify_ms_max"] = max(st["classify_ms_max"], scores.get("classify_ms", 0))
            # recent: plain list of the stored seconds, oldest first.
            recent = list(self._windows)
            # was: the mode before this second; reason: why it changed (None = it didn't).
            was, reason = self.active, None
            if not self.active:
                # Count the seconds among the last 3 that were clearly music.
                # In "for m, s in recent[-3:]", each item is a tuple (score, silent),
                # unpacked into m (the music score) and s (silent?).
                # sum(1 for … if …) counts how many match.
                if sum(1 for m, s in recent[-3:] if m >= MUSIC_ON) >= 2:
                    self.active, reason = True, "music"
            else:
                # PYTHON: all(...) is True only if EVERY item matches.
                # "_" is the usual name for a value we don't need.
                if len(recent) >= 2 and all(s for _, s in recent[-2:]):
                    self.active, reason = False, "silence"
                elif len(recent) >= 3 and all(m < MUSIC_OFF and not s for m, s in recent[-3:]):
                    self.active, reason = False, "speech"
            # PYTHON: "!=" means "is not equal to".
            if self.active != was:
                st["switches"] += 1
                self._log("music_mode_changed", active=self.active, reason=reason,
                          # a list comprehension with "A if … else B" inside:
                          # each second's score (2 decimals), or "silent".
                          last_seconds=[round(m, 2) if not s else "silent" for m, s in recent[-3:]],
                          level_dbfs=round(level, 1))
            # Time for the periodic summary? time.monotonic() is a clock that only
            # moves forward (unlike the wall clock, it can't jump), good for timing.
            if time.monotonic() - st["t0"] >= STATS_SEC and st["windows"]:
                n = st["windows"]
                self._log("music_scene_stats", windows=n, silent_windows=st["silent"],
                          music_mean=round(st["music_sum"] / n, 3),    # average music score
                          speech_mean=round(st["speech_sum"] / n, 3),
                          classify_ms_max=st["classify_ms_max"], switches=st["switches"],
                          music_mode=self.active, singing=self.singing)
                self._stats = self._new_stats()              # start counting afresh

    # PYTHON: @staticmethod (a "decorator", written above a function) marks a
    # method that doesn't need `self`: it doesn't use the object's data.
    @staticmethod
    def _new_stats():
        """Fresh, empty counters for the next summary log line.

        Returns a dict:
            t0               float: when counting started (time.monotonic())
            windows          int: seconds checked
            silent           int: of which silent
            music_sum        float: music scores added up (for the average)
            speech_sum       float: speech scores added up
            classify_ms_max  float: slowest check, in ms
            switches         int: how often music mode switched
        """
        return {"t0": time.monotonic(), "windows": 0, "silent": 0, "music_sum": 0.0,
                "speech_sum": 0.0, "classify_ms_max": 0.0, "switches": 0}

    def judge(self, audio_16k):
        """Score a finished phrase, for its line in the log.

        Called by: the transcription worker, once per finished phrase.

        Inputs:
            audio_16k  1-D array: the whole phrase at 16 kHz.

        Returns:
            a dict, or {} (empty) if the classifier isn't loaded yet:
              {"sound_speech": 0.1, "sound_singing": 0.4, "sound_music": 0.8,
               "sound_classify_ms": 5.1, "music_mode": True, "singing": True}

        Note: this does NOT change music mode. Only feed() (once per second)
        decides that; judge() only records what this phrase sounded like.
        """
        if self.classifier is None:
            return {}
        scores = self.classifier.classify(audio_16k)
        # PYTHON: {f"sound_{k}": v for k, v in scores.items()} is a DICT
        # COMPREHENSION: it builds a new dict with "sound_" put before every name.
        # .items() gives each (name, value) pair of a dict.
        # {**a, **b} unpacks two dicts into one new dict.
        return {**{f"sound_{k}": v for k, v in scores.items()},
                "music_mode": self.active, "singing": self.singing}

"""
make_testdata.py - make test clips whose language and text are known exactly
============================================================================

WHAT THIS FILE IS FOR
---------------------
The simulator (simulate.py) checks the app against clips whose correct text
is known. Real recordings turned out to be unreliable for this: many songs
mix languages (an English chorus in a German song), so the app was marked
"wrong language" when it was actually right.

This script MAKES its own clips instead, so every clip is guaranteed to be
in exactly one language, with exactly known words:

  speech   one sentence read aloud by a computer voice ("text-to-speech").
           The voices are Microsoft's free neural voices (the "edge-tts"
           package), two per language - usually one female, one male - and
           they read at three speeds.
  songs    two sentences read by the same voices, mixed OVER MUSIC that this
           script composes itself in Python: chords, bass, a melody and
           drums, in a random key and tempo, with 2 s of music before the
           voice and 1.5 s after it.

The sentences come from Google FLEURS (licence CC BY 4.0): sentences written
by people, one language each. Only the small text files are downloaded, not
FLEURS's audio. Two kinds of sentence are skipped:
  * sentences with digits ("18 medals"): a voice says "eighteen", which
    can't be compared word-for-word with "18";
  * sentences with words in another alphabet (an English name in a Russian
    sentence), which would make the language less clear-cut.

What it writes (in testdata/synth/):
  speech/de_00.wav …   songs/de_00.wav …     (16 kHz, mono, 16-bit WAV)
  manifest.jsonl, one line per clip, e.g.
    {"file": "songs\\de_00.wav", "kind": "song", "lang": "de",
     "text": "Die Polizei sagte, … Der Satellit …",
     "voice": "de-DE-KatjaNeural", "snr_db": 6.2,
     "vocal_start": 2.0, "vocal_end": 11.84}
  vocal_start / vocal_end: where the voice starts and ends, in seconds,
  so the simulator can time things exactly. snr_db: how much louder the
  voice is than the music (see build()).

The same clips come out every time: the random choices (sentence order,
tempo, key, melody) all use a fixed starting number, SEED.

HOW TO RUN IT (from the project folder, needs internet)
-------------------------------------------------------
    .venv\\Scripts\\python make_testdata.py         4 clips per language and kind
    .venv\\Scripts\\python make_testdata.py 8       8 per language and kind
    .venv\\Scripts\\python make_testdata.py --piper  the same, read by Piper voices

SECOND VOICE ENGINE: PIPER (--piper)
------------------------------------
With --piper the voices come from Piper (the "piper-tts" package), a
neural text-to-speech engine that runs on this PC. Its clips go to a
SEPARATE data set, testdata/synth_piper/, so the edge-tts set and its
baseline results never change. Run the simulator on it with:
    .venv\\Scripts\\python simulate.py --data synth_piper
(The sentences only match the edge-tts set's when both were made with the
same clip count and neither engine skipped a sentence.)
Why a second engine: a test made with one engine's voices can flatter or
punish the app in ways real speech wouldn't; two engines show that.
Piper's output is also reproducible (edge-tts voices can change on
Microsoft's servers). The voice files (~20-110 MB each) download once
from Hugging Face into testdata/piper_voices/; after that, only the
FLEURS sentences need the internet.
Install it WITHOUT its dependencies, or pip replaces onnxruntime-gpu with
the CPU-only onnxruntime (see requirements.txt):
    .venv\\Scripts\\python -m pip install --no-deps piper-tts pathvalidate

WHAT IS IN THIS FILE
--------------------
  Settings          where to save; which languages and voices.
  sentences()       downloads and filters the FLEURS sentences.
  tts()             turns one sentence into audio with a computer voice.
  voices_by_lang()  picks two voices per language.
  _tone(), _env()   building blocks for music: a note, and its volume shape.
  music()           composes a backing track.
  rms()             measures loudness.
  build()           makes all the clips and the manifest.

A FEW MUSIC FACTS USED BELOW
----------------------------
  Frequency   how high a note is, in Hz (vibrations per second).
              The note A above middle C is 440 Hz.
  Octave      a note 12 semitones higher has twice the frequency
              (A = 440 Hz, the next A up = 880 Hz).
  MIDI number a number per note: middle C = 60, A440 = 69, each +1 is one
              semitone higher. Frequency = 440 x 2^((number - 69) / 12).
  Chord       several notes played together; "major" chords sound happy.
  BPM         beats per minute: the tempo. 120 BPM = 2 beats per second.
  Bar         a group of 4 beats here.
  Harmonics   real instruments don't play a pure tone: they add quieter
              copies at 2x, 3x, 4x … the frequency, which give the sound its
              character. A flute has few, a trumpet many.

PYTHON NOTES
------------
This file uses "async" code (explained at tts()). Other new ideas are marked
"PYTHON:" where they appear; the basics are explained at the top of interpreter/music.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import asyncio                  # runs "async" code (see tts() below)
import io                       # treat bytes in memory as a file
import json                     # write the manifest as JSON lines
import re                       # "regular expressions": text patterns
import sys                      # sys.argv: the number typed after the script name
from pathlib import Path        # file paths as objects

import numpy as np              # arrays of numbers: all the audio maths
import requests                 # downloading from the internet

# Re-use from fetch_testdata.py: the FLEURS language list, the download
# address, and the function that writes WAV files.
from fetch_testdata import FLEURS, HF, save_wav

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent      # Path: the project folder
# OUT_DIRS: dict, voice engine -> the data set's folder (see PIPER above).
OUT_DIRS = {"edge": APP_DIR / "testdata" / "synth",
            "piper": APP_DIR / "testdata" / "synth_piper"}
OUT = OUT_DIRS["edge"]                         # Path: where the clips are saved (set by build())
RATE = 16000                                   # int: samples per second
SEED = 7                                       # int: starting number for all random choices

# The voice service names languages with a country, e.g. "de-DE" = German
# as spoken in Germany. LOCALES: dict, Whisper code -> voice locale.
LOCALES = {"ru": "ru-RU", "en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES",
           "it": "it-IT", "uk": "uk-UA", "pl": "pl-PL", "bg": "bg-BG", "tr": "tr-TR"}
# Languages written in the Cyrillic alphabet (Russian, Ukrainian, Bulgarian).
# PYTHON: {…} with single values is a SET (like a list, but no repeats, fast "in").
CYRILLIC = {"ru", "uk", "bg"}


# ─────────────────────────────────────────────────────────────────────────────
# Text
# ─────────────────────────────────────────────────────────────────────────────

def sentences(folder, lang):
    """Download FLEURS's sentences for one language and keep the usable ones.

    Inputs:
        folder  str: FLEURS's folder name, e.g. "de_de".
        lang    str: the language code, e.g. "de" (decides which alphabet
                counts as "other").

    Returns:
        list of str: usable sentences, each 6 to 22 words, no digits, no
        words in another alphabet, no repeats. Usually 100-200 of them.

    Steps:
      1. Download the sentence tables ("dev", then "test", then "train"),
         stopping once more than 200 sentences have been collected.
         Each table has one row per recording; column 3 is the sentence.
      2. Filter: remove repeats, keep 6-22 words, drop digits and other
         alphabets.

    Example:
        sentences("de_de", "de")
        -> ["Die Polizei sagte, der Körper schien seit etwa einem Tag dort gelegen zu haben.", …]
    """
    out = []                                      # list of all sentences found
    for split in ("dev", "test", "train"):        # step 1
        r = requests.get(f"{HF}/google/fleurs/resolve/main/data/{folder}/{split}.tsv", timeout=60)
        r.raise_for_status()                      # stop with an error if the download failed
        for row in r.text.splitlines():
            cols = row.split("\t")                # columns are separated by tabs
            if len(cols) > 2:
                out.append(cols[2].strip())       # .strip() removes spaces at both ends
        if len(out) > 200:
            break
    # Step 2. The "other" alphabet to reject:
    #   for Cyrillic languages: Latin letters, pattern [A-Za-z] (A to Z, a to z);
    #   for the others: Cyrillic letters, the range [Ѐ-ӿ] (Unicode 0400-04FF,
    #   which contains all Cyrillic letters).
    other = r"[A-Za-z]" if lang in CYRILLIC else r"[Ѐ-ӿ]"
    # PYTHON: dict.fromkeys(list) makes a dict with the list's items as keys:
    # repeats disappear and the order is kept - a neat way to remove duplicates.
    # re.search(pattern, s) finds the pattern anywhere in s ("\d" = any digit).
    # len(s.split()) = the number of words.
    keep = [s for s in dict.fromkeys(out)
            if 6 <= len(s.split()) <= 22 and not re.search(r"\d", s) and not re.search(other, s)]
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Voices
# ─────────────────────────────────────────────────────────────────────────────
# PYTHON: ASYNC CODE.
# Talking to an internet service means a lot of waiting. "async def" marks a
# function that can pause while it waits ("await …") without freezing the
# whole program. You call it with "await function(...)" from another async
# function, and the very first one is started with asyncio.run(...) (see the
# bottom of this file). edge-tts only offers async functions, which is why
# this file uses them.

async def tts(text, voice, rate):
    """Read a sentence aloud with a computer voice ("text-to-speech").

    Inputs:
        text   str: the sentence, e.g. "Guten Morgen, wie geht es?"
        voice  str: the voice's name, e.g. "de-DE-KatjaNeural".
        rate   str: speaking speed change, e.g. "-10%" (slower), "+0%",
               "+8%" (faster).

    Returns:
        1-D float32 array: the spoken sentence at 16 kHz, with the silence
        before and after the words trimmed off.

    Steps:
      1. Ask the voice service to speak; it sends the audio back in pieces,
         as MP3 (a compressed audio format). Collect all the pieces.
      2. Decode the MP3 into plain samples, converted to 16 kHz mono
         decimals, using the "av" library (a Python wrapper for FFmpeg).
      3. Trim the silence: keep from just before the first sample louder
         than 2 % of the loudest, to just after the last one (+-10 ms).
         Then vocal_start/vocal_end mark where the words really are.

    Example:
        audio = await tts("Hallo Welt", "de-DE-KatjaNeural", "+0%")
        len(audio) / 16000  ->  0.9   (seconds)
    """
    import av
    import edge_tts
    # Step 1. PYTHON: bytearray() is a growable list of bytes; += adds more.
    mp3 = bytearray()
    # PYTHON: "async for" loops over items that arrive over time (here: pieces
    # of audio coming from the internet), pausing while waiting for each one.
    # Each chunk is a dict, e.g. {"type": "audio", "data": b"..."}.
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    # Step 2. parts: list of arrays, the decoded pieces of audio.
    parts = []
    # av.open needs a file; io.BytesIO turns the bytes into an in-memory file.
    with av.open(io.BytesIO(bytes(mp3))) as f:
        # The resampler converts to: "flt" (32-bit decimals), "mono", 16 kHz.
        resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
        # f.decode(audio=0) gives the first audio track's frames (short pieces).
        for frame in f.decode(audio=0):
            # .to_ndarray() turns a frame into a NumPy array; .reshape(-1) makes
            # it a flat 1-D list (-1 = "however long it needs to be").
            parts += [o.to_ndarray().reshape(-1) for o in resampler.resample(frame)]
        # resample(None) = "no more input": gives back what it still held.
        parts += [o.to_ndarray().reshape(-1) for o in resampler.resample(None)]
    audio = np.concatenate(parts).astype(np.float32)
    # Step 3. Trim the voice's own leading/trailing silence, so vocal_start/vocal_end
    # are where the words really are.
    # loud: positions of all samples louder than 2 % of the loudest one.
    loud = np.nonzero(np.abs(audio) > 0.02 * np.abs(audio).max())[0]
    # Keep from 160 samples (10 ms) before the first to 160 after the last.
    # max(0, …) stops it going below position 0.
    return audio[max(0, loud[0] - 160):loud[-1] + 160] if len(loud) else audio


async def voices_by_lang():
    """Choose two voices for each language: a female and a male one where
    possible, so the test isn't tuned to a single voice.

    Returns:
        dict: language code -> list of voice names, e.g.
          {"de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
           "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"], …}

    Each voice in the service's list is a dict like
        {"ShortName": "de-DE-KatjaNeural", "Locale": "de-DE", "Gender": "Female", …}
    """
    import edge_tts
    all_voices = await edge_tts.list_voices()          # list of dicts, ~400 voices
    out = {}
    for lang, locale in LOCALES.items():
        vs = [v for v in all_voices if v["Locale"] == locale]          # this language's voices
        fem = [v["ShortName"] for v in vs if v["Gender"] == "Female"]
        male = [v["ShortName"] for v in vs if v["Gender"] == "Male"]
        # fem[:1] = the first female voice as a list (or [] if there is none).
        # Adding two lists joins them: ["Katja"] + ["Conrad"] = ["Katja", "Conrad"].
        # "or" falls back to the first two voices of any kind if both were empty.
        out[lang] = (fem[:1] + male[:1]) or [v["ShortName"] for v in vs[:2]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Piper: the second voice engine (used with --piper)
# ─────────────────────────────────────────────────────────────────────────────
# PIPER_VOICES: dict, language -> Piper voice names, female first, then male
# where both exist. A name reads "<locale>-<speaker>-<quality>"; quality is
# x_low / low / medium / high (bigger file, more natural). Spanish uses Latin
# American voices to match FLEURS's es_419 sentences. Bulgarian and Turkish
# have only one voice each.
PIPER_VOICES = {
    "ru": ["ru_RU-irina-medium", "ru_RU-denis-medium"],
    "en": ["en_US-hfc_female-medium", "en_US-hfc_male-medium"],
    "de": ["de_DE-kerstin-low", "de_DE-thorsten-medium"],
    "fr": ["fr_FR-siwis-medium", "fr_FR-tom-medium"],
    "es": ["es_AR-daniela-high", "es_MX-ald-medium"],
    "it": ["it_IT-paola-medium", "it_IT-riccardo-x_low"],
    "uk": ["uk_UA-tetiana-high", "uk_UA-mykyta-high"],
    "pl": ["pl_PL-gosia-medium", "pl_PL-darkman-medium"],
    "bg": ["bg_BG-dimitar-medium"],
    "tr": ["tr_TR-dfki-medium"],
}
PIPER_DIR = APP_DIR / "testdata" / "piper_voices"   # Path: downloaded voice files
# edge-tts speeds -> Piper's length_scale (how long each sound lasts:
# 1.1 = 10 % slower, 0.92 = about 8 % faster).
PIPER_SPEED = {"-10%": 1.1, "+0%": 1.0, "+8%": 0.92}
_piper_loaded = {}                                  # dict: voice name -> loaded PiperVoice


def _piper_voice(name):
    """Load a Piper voice, downloading its two files the first time.

    The voices live on Hugging Face in rhasspy/piper-voices, in folders made
    from the name: "ru_RU-irina-medium" ->
        ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx       (the model)
        ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json  (its settings)
    """
    if name not in _piper_loaded:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
        locale, speaker, quality = name.split("-")
        folder = f"{locale.split('_')[0]}/{locale}/{speaker}/{quality}"
        for ext in (".onnx.json", ".onnx"):
            hf_hub_download("rhasspy/piper-voices", f"{folder}/{name}{ext}", local_dir=PIPER_DIR)
        _piper_loaded[name] = PiperVoice.load(PIPER_DIR / folder / f"{name}.onnx")
    return _piper_loaded[name]


def piper_tts(text, voice, rate):
    """Piper's version of tts(): same inputs, same kind of result.

    Piper returns the speech in chunks of float samples at the voice's own
    rate (usually 22050 Hz); soxr converts them to 16 kHz. The silence is
    trimmed exactly as in tts().
    """
    import soxr
    from piper import SynthesisConfig
    v = _piper_voice(voice)
    chunks = list(v.synthesize(text, SynthesisConfig(length_scale=PIPER_SPEED[rate])))
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    audio = soxr.resample(audio, chunks[0].sample_rate, RATE).astype(np.float32)
    loud = np.nonzero(np.abs(audio) > 0.02 * np.abs(audio).max())[0]
    return audio[max(0, loud[0] - 160):loud[-1] + 160] if len(loud) else audio


# ─────────────────────────────────────────────────────────────────────────────
# Music
# ─────────────────────────────────────────────────────────────────────────────
# Sound is a wave. A pure note is a SINE wave: sin(2 x pi x frequency x time).
# Adding sine waves together mixes the sounds. That is all a synthesizer does,
# and all this section does: it adds up notes, bass, drums and a melody into
# one array of samples.

def _tone(freq, n, harmonics=(1.0,)):
    """Make a note: a sine wave plus optional harmonics.

    Inputs:
        freq       float: the note's frequency in Hz, e.g. 440.0.
        n          int: how many samples long, e.g. 16000 = 1 second.
        harmonics  tuple of floats: the volume of the note itself and of
                   each harmonic. (1.0,) = a pure tone; (1, .5, .25) = the
                   note at full volume, plus 2x its frequency at half
                   volume, plus 3x at a quarter: a richer, organ-like sound.

    Returns:
        1-D array of n samples (values roughly -1.x .. 1.x; they are
        scaled down later).

    Example:
        _tone(440.0, 16000)            1 second of a pure A
        _tone(110.0, 8000, (1, .4))    half a second of a bass A with one harmonic
    """
    # t: the time of each sample in seconds: 0, 1/16000, 2/16000, …
    t = np.arange(n) / RATE
    # For each harmonic k (0, 1, 2 …) with volume a: a x sin(2 pi x freq x (k+1) x t).
    # sum(...) adds all those waves together.
    return sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))


def _env(n, attack=0.01, decay=None):
    """The volume shape ("envelope") of a note: how it starts and fades.

    Real notes don't switch on and off instantly: a piano note starts fast
    and fades away; an organ note holds. Multiplying a tone by its envelope
    shapes it like that.

    Inputs:
        n       int: length in samples.
        attack  float seconds (default 0.01 = 10 ms): fade-in time.
        decay   float seconds or None: if given, the note dies away with
                this "time constant" (after `decay` seconds it is at about
                37 % volume, after twice that about 14 %). None = it holds.

    Returns:
        1-D array of n volume factors from 0.0 to 1.0.

    It always fades out over the last 20 ms, so notes never end with a click.

    Example (n = 16000, attack 0.01, decay 0.25):
        volume  0.0 -> 1.0 within 10 ms, then 0.37 at 0.25 s, 0.14 at 0.5 s,
        and 0 at the very end.
    """
    e = np.ones(n)                                    # start: full volume everywhere
    a = min(n, int(attack * RATE))                    # attack length in samples
    # np.linspace(0, 1, a): a evenly spaced numbers from 0 to 1 (a straight ramp up).
    e[:a] = np.linspace(0, 1, a)
    if decay:
        # np.exp(-x) falls from 1 towards 0: the classic "dying away" curve.
        e *= np.exp(-np.arange(n) / (decay * RATE))
    r = min(n, int(0.02 * RATE))                      # 20 ms in samples
    e[n - r:] *= np.linspace(1, 0, r)                 # ramp down at the end: no clicks
    return e


def music(seconds, rng):
    """Compose a simple pop-style backing track.

    Inputs:
        seconds  float: how long the track must be, e.g. 12.8.
        rng      a NumPy random-number generator (made from SEED in build()),
                 so the "random" choices are the same every time.

    Returns:
        1-D float array, seconds x 16000 samples, scaled so its average
        loudness (RMS) is exactly 1.0 (build() then sets the real volume).

    What it plays, per bar of 4 beats:
        pad     the bar's chord, held softly for the whole bar
        bass    the chord's root note, low, on every beat
        kick    a drum "boom" on beats 1 and 3 (a sine wave falling from
                120 Hz to 45 Hz)
        snare   a short noise burst on beats 2 and 4
        hi-hat  a very short, bright tick on every half beat
        melody  on most half beats, a random note from the chord or the scale
    The 4 chords repeat: I - vi - IV - V (for example in C major:
    C - A minor - F - G), the most common chord pattern in pop music,
    starting at a random one of the four.

    Example:
        music(12.8, rng)  -> 204,800 samples: 12.8 s at a random tempo
                             between 80 and 130 BPM, in a random key.
    """
    n_total = int(seconds * RATE)                     # int: total samples needed
    bpm = rng.uniform(80, 130)                        # float: a random tempo, e.g. 104.3
    beat = int(60 / bpm * RATE)                       # int: samples per beat, e.g. 9204
    # root: the key's MIDI note number, randomly C3 (48) to B3 (59).
    # rng.integers(0, 12) gives a random whole number 0..11.
    root = 48 + int(rng.integers(0, 12))
    # The major scale, in semitones above the root (C D E F G A B for C).
    major = [0, 2, 4, 5, 7, 9, 11]
    # The 4 chords, each as 3 notes in semitones above the root:
    #   I = [0, 4, 7] (C E G);  vi = [9, 12, 16] (A C E);
    #   IV = [5, 9, 12] (F A C);  V = [7, 11, 14] (G B D).
    progression = [[0, 4, 7], [9, 12, 16], [5, 9, 12], [7, 11, 14]]   # I vi IV V
    # Start at a random chord: rotate the list. With shift 2:
    # [IV, V] + [I, vi] = [IV, V, I, vi].
    shift = int(rng.integers(0, 4))
    progression = progression[shift:] + progression[:shift]
    # hz: turns a MIDI note number into Hz. hz(69) = 440.0; hz(57) = 220.0.
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    # out: the track, silent to begin with. 8 extra beats of room, so notes
    # near the end don't run past the end of the array; cut off later.
    out = np.zeros(n_total + 8 * beat)
    # noise: one beat of random values (sounds like "shhh"), re-used for the drums.
    noise = rng.standard_normal(beat)
    bar = 0                                           # int: which bar we are in
    # One round of this loop = one bar (4 beats), starting at sample `start`.
    for start in range(0, n_total, 4 * beat):
        # PYTHON: "%" is the remainder: bar % 4 cycles 0, 1, 2, 3, 0, 1, …
        chord = progression[bar % 4]
        for note in chord:                            # pad: whole bar, soft
            # One octave above the root (+12), 4 harmonics, a slow 0.15 s fade-in.
            out[start:start + 4 * beat] += 0.08 * _tone(hz(root + 12 + note), 4 * beat,
                                                       (1, .5, .25, .12)) * _env(4 * beat, 0.15)
        for b in range(4):                            # b: the beat in the bar, 0..3
            s = start + b * beat                      # where this beat starts
            # Bass: the chord's lowest note, one octave below the root (-12),
            # plucked (fast attack, 0.25 s decay).
            out[s:s + beat] += 0.25 * _tone(hz(root - 12 + chord[0]), beat, (1, .4)) * _env(beat, 0.005, 0.25)
            if b % 2 == 0:                            # beats 1 and 3: kick drum
                k = int(0.15 * RATE)                  # 0.15 s long
                f = np.linspace(120, 45, k)           # the pitch falls from 120 to 45 Hz
                # np.cumsum adds up the frequencies sample by sample: that is the
                # way to make a sine wave whose pitch changes as it plays.
                out[s:s + k] += 0.5 * np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(k, 0.001, 0.05)
            else:                                     # beats 2 and 4: snare (noise burst)
                k = int(0.12 * RATE)
                out[s:s + k] += 0.2 * noise[:k] * _env(k, 0.001, 0.04)
            for half in (0, beat // 2):               # hi-hat on every half beat
                k = int(0.03 * RATE)                  # 30 ms
                # np.diff (each value minus the one before it) removes the low
                # sounds from the noise, leaving a bright "tss".
                out[s + half:s + half + k] += 0.06 * np.diff(noise[:k + 1]) * _env(k, 0.001, 0.01)
            for half in (0, beat // 2):               # melody: chord and scale notes
                if rng.random() < 0.7:                # a note on 70 % of half beats
                    # Two octaves above the root (+24): 60 % of the time a chord
                    # note, otherwise any note of the scale.
                    m = root + 24 + (chord[int(rng.integers(0, 3))] if rng.random() < .6
                                     else major[int(rng.integers(0, 7))])
                    k = beat // 2
                    out[s + half:s + half + k] += 0.1 * _tone(hz(m), k, (1, .3, .1)) * _env(k, 0.01, 0.15)
        bar += 1
    out = out[:n_total]                               # cut off the extra room
    # Divide by the RMS (average loudness) so the result's RMS is exactly 1.0.
    return out / (np.sqrt((out ** 2).mean()) + 1e-9)    # unit RMS


def rms(x):
    """Average loudness ("root mean square") of an array of samples.

    The square root of the average of the squared samples. Squaring makes
    negative and positive values count the same.

    Example:
        rms(np.array([0.5, -0.5, 0.5, -0.5]))  ->  0.5
    The tiny 1e-9 avoids dividing by zero later for complete silence.
    """
    return float(np.sqrt((x ** 2).mean()) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Building the clips
# ─────────────────────────────────────────────────────────────────────────────

async def build(count, engine="edge"):
    """Make all the clips and the manifest.

    Inputs:
        count   int: how many speech clips AND how many songs to make per
                language (default 4, so 10 languages x 8 = 80 clips).
        engine  str: "edge" (edge-tts, into testdata/synth) or "piper"
                (Piper, into testdata/synth_piper).

    For each language:
      1. Get its sentences and shuffle them (in the same "random" order
         every time, thanks to SEED).
      2. Speech clips: one sentence each, read by alternating voices and
         speeds, with 0.3 s of silence before and after, at a fixed
         loudness (RMS 0.08 = about -22 dBFS).
      3. Song clips: two sentences read by one voice with 0.4 s between
         them; a backing track as long as that plus 2 s intro and 1.5 s
         outro; the voice mixed in 3 to 9 dB louder than the music
         ("snr_db", chosen at random per song); the whole clip turned down
         if its loudest sample would go above 0.95.
      4. Save each clip and add it to the manifest.
    Finally write manifest.jsonl.

    About snr_db ("signal-to-noise ratio" in decibels): how much louder the
    voice is than the music. +6 dB = the voice's samples are about twice as
    large as the music's. Lower values make the song harder to understand.

    Example output:
        ru: 4 speech, 4 songs
        en: 4 speech, 4 songs
        …
        80 clips -> C:\\...\\testdata\\synth

    PYTHON: this is an async function (see tts()), because it waits for the
    voice service; it is started at the bottom of the file with asyncio.run.
    """
    # rng: the random-number generator. Starting it from SEED makes every run
    # produce the same sequence of "random" numbers, so the same clips.
    rng = np.random.default_rng(SEED)
    global OUT                                        # PYTHON: lets this function change OUT
    OUT = OUT_DIRS[engine]
    if engine == "piper":
        voices = PIPER_VOICES                         # dict: language -> 1-2 voice names

        async def say(text, voice, rate):             # async wrapper, same shape as tts()
            return piper_tts(text, voice, rate)
    else:
        voices = await voices_by_lang()               # dict: language -> 2 voice names
        say = tts
    rates = ["-10%", "+0%", "+8%"]                    # the three speaking speeds, used in turn
    for kind in ("speech", "songs"):
        (OUT / kind).mkdir(parents=True, exist_ok=True)
    manifest = []                                     # list of dicts: every clip made
    for folder, lang in FLEURS.items():
        texts = sentences(folder, lang)               # step 1
        rng.shuffle(texts)                            # mixes the list's order in place
        # PYTHON: iter(list) makes an ITERATOR: something you take items from one
        # at a time, and which remembers where it got to. Both loops below take
        # sentences from the same `pool`, so no sentence is used twice.
        pool = iter(texts)

        async def next_vocal(voice, rate):
            """The next sentence the voice manages to speak, as (text, audio).

            The voice service refuses a few sentences (for no clear reason,
            usually with "No audio was received"). Each sentence gets 2 tries;
            if both fail, it is skipped with a message and the next one is
            tried. Returns (None, None) when the sentences run out.

            PYTHON: this function is defined INSIDE the loop, so it can use the
            loop's current `pool` and `lang`.
            """
            for text in pool:                         # takes the next sentence from pool
                for _ in range(2):                    # two attempts
                    try:
                        return text, await say(text, voice, rate)
                    except Exception as e:
                        err = e                       # remember the error for the message
                # {text[:40]!r}: the first 40 characters, in quotes.
                print(f"  {lang}: skipped {text[:40]!r} ({err})", flush=True)
            return None, None

        made = {"speech": 0, "song": 0}               # dict: clips made so far, by kind
        # Step 2: speech clips.
        for i in range(count):
            # Alternate voices and speeds: i % 2 -> voice 0, 1, 0, 1 …; i % 3 -> speed.
            voice = voices[lang][i % len(voices[lang])]
            text, vocal = await next_vocal(voice, rates[i % 3])
            if text is None:
                break                                 # no sentences left
            made["speech"] += 1
            pad = np.zeros(int(0.3 * RATE), np.float32)          # 0.3 s of silence
            # vocal / rms(vocal) * 0.08: scale the voice to an RMS of exactly 0.08.
            audio = np.concatenate([pad, vocal / rms(vocal) * 0.08, pad])
            path = OUT / "speech" / f"{lang}_{i:02d}.wav"         # e.g. speech\de_03.wav
            save_wav(path, audio, RATE)                           # step 4
            manifest.append({"file": str(path.relative_to(OUT)), "kind": "speech", "lang": lang,
                             "text": text, "voice": voice, "engine": engine,
                             "vocal_start": 0.3, "vocal_end": round(0.3 + len(vocal) / RATE, 2)})
        # Step 3: songs.
        for i in range(count):
            # (i + 1): start with the OTHER voice than the speech clips did.
            voice = voices[lang][(i + 1) % len(voices[lang])]
            gap = np.zeros(int(0.4 * RATE), np.float32)          # 0.4 s between the sentences
            # pair: the two sentences' texts; parts: their audio, each followed by a gap.
            pair, parts = [], []
            for _ in range(2):
                text, v = await next_vocal(voice, rates[i % 3])
                if text is not None:
                    pair.append(text); parts += [v, gap]
            if len(pair) < 2:
                break                                 # not enough sentences left
            made["song"] += 1
            # parts[:-1]: everything except the last gap.
            vocal = np.concatenate(parts[:-1])
            intro, outro = 2.0, 1.5                   # seconds of music before / after the voice
            backing = music(intro + len(vocal) / RATE + outro, rng)
            # A random voice-to-music ratio, 3 to 9 dB.
            snr_db = rng.uniform(3, 9)                       # vocals 3..9 dB above the music
            # The music's RMS: 0.08 divided by the ratio. 10 ** (dB / 20) turns dB
            # into a factor: 6 dB -> 2.0, so the music's RMS would be 0.04.
            mix = backing * (0.08 / 10 ** (snr_db / 20))
            s = int(intro * RATE)                     # where the voice starts (sample)
            # Add the voice (RMS 0.08) on top of the music, starting after the intro.
            mix[s:s + len(vocal)] += vocal / rms(vocal) * 0.08
            # If the loudest sample is above 0.95, turn the whole clip down so it
            # isn't; otherwise max(1.0, …) is 1.0 and nothing changes.
            mix /= max(1.0, np.abs(mix).max() / 0.95)
            path = OUT / "songs" / f"{lang}_{i:02d}.wav"
            save_wav(path, mix, RATE)                 # step 4
            manifest.append({"file": str(path.relative_to(OUT)), "kind": "song", "lang": lang,
                             "text": " ".join(pair), "voice": voice, "engine": engine, "snr_db": round(snr_db, 1),
                             "vocal_start": intro, "vocal_end": round(intro + len(vocal) / RATE, 2)})
        print(f"  {lang}: {made['speech']} speech, {made['song']} songs", flush=True)
    # Write the clip list, one JSON object per line.
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(manifest)} clips -> {OUT}")


# Start here when run directly. asyncio.run(...) starts the async build()
# and waits until it has finished. The number of clips per language is the
# number typed after the script's name, or 4; "--piper" picks the Piper voices.
if __name__ == "__main__":
    args = sys.argv[1:]
    numbers = [a for a in args if a.isdigit()]
    asyncio.run(build(int(numbers[0]) if numbers else 4,
                      "piper" if "--piper" in args else "edge"))

"""
fetch_testdata.py - download real recordings for the simulator to test with
===========================================================================

WHAT THIS FILE IS FOR
---------------------
The simulator (simulate.py) needs test clips: recordings whose correct text
is known. This script downloads a small set of REAL recordings - real
people's voices and real songs - with their correct text, into the folder
testdata/ (about 30 MB). The folder isn't part of the project in git
(.gitignore excludes it): anyone can download it again with this script.

Two collections are used, both free to use for research:

  speech   Google FLEURS (licence CC BY 4.0): people reading sentences
           aloud, in about 100 languages. We use 10 of them.
           The full collection is hundreds of megabytes per language, so
           only the first few clips of each language are taken: the file
           is read as it downloads ("streamed"), and the download stops as
           soon as enough clips have been found.
  songs    JamendoLyrics "jam-alt-lines" (Creative Commons songs from the
           music site Jamendo): single sung lines with the music, in
           German, English, Spanish and French.

What it writes:
  testdata/speech/de_00.wav, de_01.wav, …    one file per clip
  testdata/songs/fr_00.wav, …                (16 kHz, mono, 16-bit WAV)
  testdata/manifest.jsonl                    the list of clips, one line each:
      {"file": "speech\\de_00.wav", "kind": "speech", "lang": "de",
       "text": "Die Polizei sagte, …"}

A NOTE ON THESE RECORDINGS
--------------------------
Real recordings are realistic, but less reliable as a test: many songs mix
languages (an English chorus in a German song), so "language right" can be
marked wrong when the app was actually right. That is why make_testdata.py
was added: it makes synthetic clips that are guaranteed to be in one
language. The simulator calls this data set "real" / "Recorded clips".

HOW TO RUN IT (from the project folder, needs internet)
-------------------------------------------------------
    .venv\\Scripts\\python fetch_testdata.py            6 clips per language
    .venv\\Scripts\\python fetch_testdata.py 10         10 clips per language

WHAT IS IN THIS FILE
--------------------
  Settings        where to download from and save to; which languages.
  save_wav()      writes audio to a .wav file in the format the simulator uses.
                  (make_testdata.py uses it too.)
  fetch_fleurs()  downloads read speech for one language.
  fetch_songs()   downloads sung lines for one language.
  main()          downloads everything and writes the manifest.

Python ideas new to this file are explained where they appear ("PYTHON:").
The basics are explained at the top of interpreter/music.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import io                       # treat bytes in memory as if they were a file
import json                     # read/write JSON text
import sys                      # sys.argv: what was typed after the script's name
import tarfile                  # read .tar.gz archives (bundles of many files)
from pathlib import Path        # file paths as objects (folder / "name")

import numpy as np              # fast maths on arrays of numbers
import requests                 # downloading from the internet ("HTTP requests")

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent      # Path: the project folder
OUT = APP_DIR / "testdata"                     # Path: where the clips are saved
HF = "https://huggingface.co/datasets"         # str: start of every download address

# FLEURS names its folders "<language>_<country>"; Whisper uses 2-letter codes.
# FLEURS: dict, FLEURS folder name -> Whisper language code.
# (es_419 is "Spanish as spoken in Latin America"; 419 is the region's number.)
FLEURS = {"ru_ru": "ru", "en_us": "en", "de_de": "de", "fr_fr": "fr", "es_419": "es",
          "it_it": "it", "uk_ua": "uk", "pl_pl": "pl", "bg_bg": "bg", "tr_tr": "tr"}
# SONG_LANGS: tuple of the languages JamendoLyrics has.
SONG_LANGS = ("de", "en", "es", "fr")


def save_wav(path, audio, rate):
    """Save audio as a WAV file in the format all test clips use:
    16,000 samples per second, mono, 16-bit whole numbers.

    Inputs:
        path   where to save, e.g. Path(".../testdata/speech/de_00.wav")
        audio  the sound: a NumPy array (or list) of decimal numbers from
               -1.0 to 1.0. It may be stereo: a table with one column per
               channel, shape (samples, 2).
        rate   int: its current sample rate, e.g. 48000 or 16000.

    Steps:
      1. Turn it into a float32 NumPy array.
      2. Stereo -> mono: average the left and right channels.
      3. Convert the sample rate to 16 kHz if needed.
      4. Write the WAV file: limit the values to -1..1, scale to whole
         numbers (-32767..32767) and store them as 16-bit.

    Example:
        save_wav(Path("x.wav"), np.zeros(48000), 48000)
        -> x.wav: 1 second of silence, 16,000 samples
    """
    import wave                  # Python's built-in WAV reader/writer
    import soxr                  # high-quality sample-rate conversion
    audio = np.asarray(audio, dtype=np.float32)          # step 1
    # PYTHON (NumPy): .ndim is the number of dimensions: 1 for a plain list of
    # samples (mono), 2 for a table (several channels).
    if audio.ndim > 1:
        audio = audio.mean(axis=1)                       # step 2: average across each row
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000)        # step 3
    # Step 4. "wb" = open for writing, in binary (not text) mode.
    with wave.open(str(path), "wb") as w:
        # 1 channel, 2 bytes (16 bits) per sample, 16,000 samples per second.
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        # np.clip keeps every value within -1..1 (louder would wrap around and crackle);
        # x 32767 turns 0.5 into 16383; .astype(np.int16) makes whole numbers;
        # .tobytes() gives the raw bytes that go into the file.
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def fetch_fleurs(folder, lang, count, manifest):
    """Download `count` read-speech clips of one language from FLEURS.

    Inputs:
        folder    str: FLEURS's folder for the language, e.g. "de_de".
        lang      str: the Whisper code to file it under, e.g. "de".
        count     int: how many clips to keep, e.g. 6.
        manifest  list: the clip list being built. Each clip saved here is
                  added to it as a dict. (PYTHON: a list passed to a function
                  is the SAME list, not a copy, so adding to it here changes
                  the caller's list too.)

    How FLEURS is organised (for each language):
        dev.tsv           a table with one row per recording; column 2 is the
                          file name, column 3 the sentence that was read.
                          (.tsv = "tab-separated values": columns divided by
                          tab characters.)
        audio/dev.tar.gz  an archive holding all the recordings.

    Steps:
      1. Download dev.tsv and build a dict: file name -> correct sentence.
      2. Open the archive AS IT DOWNLOADS, and go through the files in it
         one by one.
      3. For each recording that has a sentence: read the audio, save it
         as a WAV, and add it to the manifest.
      4. Stop after `count` clips: the rest of the archive is never
         downloaded.
    """
    # Step 1. requests.get(address) downloads; timeout=60: give up after 60 s.
    tsv = requests.get(f"{HF}/google/fleurs/resolve/main/data/{folder}/dev.tsv", timeout=60)
    # raise_for_status() raises an error if the download failed (e.g. "404 Not Found").
    tsv.raise_for_status()
    texts = {}                                    # dict: file name -> sentence
    # tsv.text is the downloaded text; .splitlines() cuts it into lines.
    for row in tsv.text.splitlines():
        cols = row.split("\t")                    # "\t" is the tab character
        if len(cols) > 2:
            texts[cols[1]] = cols[2]              # file name -> raw transcription
    import soundfile as sf                        # reads audio files (FLEURS uses .wav)
    # Step 2. stream=True: don't download everything first; read it bit by bit.
    with requests.get(f"{HF}/google/fleurs/resolve/main/data/{folder}/audio/dev.tar.gz",
                      stream=True, timeout=60) as r:
        r.raise_for_status()
        kept = 0                                  # int: clips saved so far
        # mode="r|gz": read a gzip-compressed tar archive as a stream (the "|"),
        # straight from the download (r.raw).
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:                    # each file inside the archive
                # Path(...).name: just the file name, without folders.
                name = Path(member.name).name
                if not member.isfile() or name not in texts:
                    continue                      # a folder, or no sentence for it: skip
                # Step 3. Read the file's bytes, wrap them as an in-memory file
                # (io.BytesIO) and let soundfile decode the audio.
                # audio: array of samples; rate: int, e.g. 16000.
                audio, rate = sf.read(io.BytesIO(tar.extractfile(member).read()))
                # PYTHON: {kept:02d} formats a number with at least 2 digits: 3 -> "03".
                out = OUT / "speech" / f"{lang}_{kept:02d}.wav"
                save_wav(out, audio, rate)
                # relative_to(OUT): the path inside testdata, e.g. "speech\de_03.wav".
                manifest.append({"file": str(out.relative_to(OUT)), "kind": "speech",
                                 "lang": lang, "text": texts[name]})
                kept += 1
                if kept >= count:                 # step 4
                    break
    print(f"  speech {lang}: {kept} clips")


def fetch_songs(lang, count, manifest):
    """Download `count` sung lines (with music) of one language from
    JamendoLyrics.

    Inputs:
        lang      str: language code, e.g. "fr".
        count     int: how many clips to keep.
        manifest  list: the clip list being built (clips are added to it).

    How the collection is organised (for each language):
        groups/<lang>/metadata.jsonl   one line per sung line, e.g.
            {"file_name": "123_4.mp3", "text": "Je chante pour toi"}
        groups/<lang>/<file name>      the audio of that line.

    Steps:
      1. Download the metadata list.
      2. For each entry: download its audio, skip it if it's shorter than
         2 seconds (a word or two says little about recognition), save it
         and add it to the manifest - until `count` clips are kept.
    """
    import soundfile as sf
    base = f"{HF}/jamendolyrics/jam-alt-lines/resolve/main/groups/{lang}"   # str: folder address
    meta = requests.get(f"{base}/metadata.jsonl", timeout=60)              # step 1
    meta.raise_for_status()
    kept = 0
    for line in meta.text.splitlines():                                    # step 2
        row = json.loads(line)                    # dict for one sung line
        # The collection's field names vary; "a or b or ''" takes the first that exists.
        audio_name = row.get("file_name") or row.get("audio") or ""
        text = row.get("text") or row.get("lyrics") or ""
        if not audio_name or not text:
            continue
        # quote() makes the file name safe for a web address (spaces -> %20, etc.).
        r = requests.get(f"{base}/{requests.utils.quote(audio_name)}", timeout=60)
        if r.status_code != 200:                  # 200 = "OK"; anything else = failed
            continue
        # r.content: the downloaded bytes. Decode them as audio.
        audio, rate = sf.read(io.BytesIO(r.content))
        # len(audio) / rate = length in seconds, e.g. 44100 samples / 44100 = 1.0 s
        if len(audio) / rate < 2.0:               # very short lines say little
            continue
        out = OUT / "songs" / f"{lang}_{kept:02d}.wav"
        save_wav(out, audio, rate)
        manifest.append({"file": str(out.relative_to(OUT)), "kind": "song", "lang": lang,
                         "text": text})
        kept += 1
        if kept >= count:
            break
    print(f"  songs  {lang}: {kept} clips")


def main():
    """Download everything and write the clip list (testdata/manifest.jsonl).

    How many clips per language: the number typed after the script's name,
    or 6 if none was given.
      PYTHON: sys.argv is the list of words typed on the command line:
        "python fetch_testdata.py 10"  ->  sys.argv == ["fetch_testdata.py", "10"]
      so sys.argv[1] is "10" (text), and int("10") makes it the number 10.

    A language that fails to download (no internet, site down…) is reported
    and skipped; the others still work.

    Example output:
        speech ru: 6 clips
        speech en: 6 clips
        …
        songs  fr: 6 clips
        84 clips -> C:\\...\\testdata\\manifest.jsonl
    """
    count =int(sys.argv[1]) if len(sys.argv) > 1 else 6
    # mkdir(parents=True, exist_ok=True): create the folder (and any missing
    # parent folders); no error if it already exists.
    (OUT / "speech").mkdir(parents=True, exist_ok=True)
    (OUT / "songs").mkdir(parents=True, exist_ok=True)
    manifest = []                                 # list of dicts: every clip saved
    # PYTHON: .items() gives the (key, value) pairs of a dict: ("ru_ru", "ru"), …
    for folder, lang in FLEURS.items():
        try:
            fetch_fleurs(folder, lang, count, manifest)
        except Exception as e:
            print(f"  speech {lang}: failed ({e})")
    for lang in SONG_LANGS:
        try:
            fetch_songs(lang, count, manifest)
        except Exception as e:
            print(f"  songs  {lang}: failed ({e})")
    # Write the clip list: one JSON object per line ("JSON Lines").
    # "w" = open for writing (replacing the old file).
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            # ensure_ascii=False keeps letters like "ü" and "я" readable in the file.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")     # "\n" = new line
    print(f"{len(manifest)} clips -> {OUT / 'manifest.jsonl'}")


# Only run main() when this file is started directly, not when imported
# (make_testdata.py imports save_wav, FLEURS and HF from here).
if __name__ == "__main__":
    main()

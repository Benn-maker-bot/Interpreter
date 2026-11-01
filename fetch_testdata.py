"""
fetch_testdata.py — download test audio for the simulator (simulate.py)
======================================================================

Downloads a small set of real recordings with their correct text, into
testdata/ (about 30 MB; not part of the project in git):

  speech   Google FLEURS (CC BY 4.0): people reading sentences, ~100
           languages. A few clips per language are streamed from the
           start of each archive, so the full archives (100s of MB) are
           never downloaded.
  songs    JamendoLyrics "jam-alt-lines" (Creative Commons songs from
           Jamendo): single sung lines with music, in de/en/es/fr.

Each clip is saved as a 16 kHz mono WAV, and testdata/manifest.jsonl gets
one line per clip: {"file", "kind", "lang", "text"}.

Usage:
    .venv\\Scripts\\python fetch_testdata.py            (default languages)
    .venv\\Scripts\\python fetch_testdata.py 10         (10 clips per language)
"""

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import requests

APP_DIR = Path(__file__).resolve().parent
OUT = APP_DIR / "testdata"
HF = "https://huggingface.co/datasets"

# FLEURS language folders -> Whisper language codes
FLEURS = {"ru_ru": "ru", "en_us": "en", "de_de": "de", "fr_fr": "fr", "es_419": "es",
          "it_it": "it", "uk_ua": "uk", "pl_pl": "pl", "bg_bg": "bg", "tr_tr": "tr"}
SONG_LANGS = ("de", "en", "es", "fr")


def save_wav(path, audio, rate):
    """Write float/int audio as 16 kHz mono 16-bit WAV."""
    import wave
    import soxr
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def fetch_fleurs(folder, lang, count, manifest):
    """Stream the dev archive and keep the first `count` clips."""
    tsv = requests.get(f"{HF}/google/fleurs/resolve/main/data/{folder}/dev.tsv", timeout=60)
    tsv.raise_for_status()
    texts = {}
    for row in tsv.text.splitlines():
        cols = row.split("\t")
        if len(cols) > 2:
            texts[cols[1]] = cols[2]          # file name -> raw transcription
    import soundfile as sf
    with requests.get(f"{HF}/google/fleurs/resolve/main/data/{folder}/audio/dev.tar.gz",
                      stream=True, timeout=60) as r:
        r.raise_for_status()
        kept = 0
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in texts:
                    continue
                audio, rate = sf.read(io.BytesIO(tar.extractfile(member).read()))
                out = OUT / "speech" / f"{lang}_{kept:02d}.wav"
                save_wav(out, audio, rate)
                manifest.append({"file": str(out.relative_to(OUT)), "kind": "speech",
                                 "lang": lang, "text": texts[name]})
                kept += 1
                if kept >= count:
                    break
    print(f"  speech {lang}: {kept} clips")


def fetch_songs(lang, count, manifest):
    """Sung lines (with music) from JamendoLyrics jam-alt-lines."""
    import soundfile as sf
    base = f"{HF}/jamendolyrics/jam-alt-lines/resolve/main/groups/{lang}"
    meta = requests.get(f"{base}/metadata.jsonl", timeout=60)
    meta.raise_for_status()
    kept = 0
    for line in meta.text.splitlines():
        row = json.loads(line)
        audio_name = row.get("file_name") or row.get("audio") or ""
        text = row.get("text") or row.get("lyrics") or ""
        if not audio_name or not text:
            continue
        r = requests.get(f"{base}/{requests.utils.quote(audio_name)}", timeout=60)
        if r.status_code != 200:
            continue
        audio, rate = sf.read(io.BytesIO(r.content))
        if len(audio) / rate < 2.0:          # very short lines say little
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
    """Download `count` clips per language (6 unless given on the command
    line) and write the list of clips, testdata/manifest.jsonl. A language
    that fails to download is reported and skipped; the others still work."""
    count =int(sys.argv[1]) if len(sys.argv) > 1 else 6
    (OUT / "speech").mkdir(parents=True, exist_ok=True)
    (OUT / "songs").mkdir(parents=True, exist_ok=True)
    manifest = []
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
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(manifest)} clips -> {OUT / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()

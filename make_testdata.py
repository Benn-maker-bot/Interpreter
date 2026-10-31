"""
make_testdata.py — build synthetic, single-language test audio for simulate.py
==============================================================================

Real recordings are unreliable test data when a clip mixes languages (many
songs have an English chorus). This script makes clips whose language and
text are known exactly:

  speech   one sentence read by a neural text-to-speech voice (edge-tts,
           Microsoft's free online voices; two voices per language).
  songs    two sentences read by the same voices, mixed over music made
           here in Python: chords, bass, a melody and drums in a random key
           and tempo, with an intro and outro of music only.

The sentences come from Google FLEURS (CC BY 4.0): text written by people,
one language per sentence. Only the small text files are downloaded.
Sentences with digits (spoken differently from how they are written) or
words in another alphabet are skipped.

Output: testdata/synth/{speech,songs}/<lang>_<nn>.wav (16 kHz mono) and
testdata/synth/manifest.jsonl, one line per clip:
{"file", "kind", "lang", "text", "voice", "vocal_start", "vocal_end"}.

Usage:
    .venv\\Scripts\\python make_testdata.py              (4 clips per language and kind)
    .venv\\Scripts\\python make_testdata.py 8            (8 per language and kind)
Needs an internet connection (FLEURS text and edge-tts voices).
"""

import asyncio
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

from fetch_testdata import FLEURS, HF, save_wav

APP_DIR = Path(__file__).resolve().parent
OUT = APP_DIR / "testdata" / "synth"
RATE = 16000
SEED = 7

# Whisper code -> edge-tts locale
LOCALES = {"ru": "ru-RU", "en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES",
           "it": "it-IT", "uk": "uk-UA", "pl": "pl-PL", "bg": "bg-BG", "tr": "tr-TR"}
CYRILLIC = {"ru", "uk", "bg"}


# ─────────────────────────────────────────────────────────────────────────────
# Text
# ─────────────────────────────────────────────────────────────────────────────

def sentences(folder, lang):
    """Usable FLEURS sentences for one language (all splits' text, no audio)."""
    out = []
    for split in ("dev", "test", "train"):
        r = requests.get(f"{HF}/google/fleurs/resolve/main/data/{folder}/{split}.tsv", timeout=60)
        r.raise_for_status()
        for row in r.text.splitlines():
            cols = row.split("\t")
            if len(cols) > 2:
                out.append(cols[2].strip())
        if len(out) > 200:
            break
    other = r"[A-Za-z]" if lang in CYRILLIC else r"[Ѐ-ӿ]"
    keep = [s for s in dict.fromkeys(out)
            if 6 <= len(s.split()) <= 22 and not re.search(r"\d", s) and not re.search(other, s)]
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Voices
# ─────────────────────────────────────────────────────────────────────────────

async def tts(text, voice, rate):
    """Speak `text` with an edge-tts voice; returns float32 audio at 16 kHz."""
    import av
    import edge_tts
    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
    parts = []
    with av.open(io.BytesIO(bytes(mp3))) as f:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
        for frame in f.decode(audio=0):
            parts += [o.to_ndarray().reshape(-1) for o in resampler.resample(frame)]
        parts += [o.to_ndarray().reshape(-1) for o in resampler.resample(None)]
    audio = np.concatenate(parts).astype(np.float32)
    # Trim the voice's own leading/trailing silence, so vocal_start/vocal_end
    # are where the words really are
    loud = np.nonzero(np.abs(audio) > 0.02 * np.abs(audio).max())[0]
    return audio[max(0, loud[0] - 160):loud[-1] + 160] if len(loud) else audio


async def voices_by_lang():
    """Two voices (a female and a male one where possible) per language."""
    import edge_tts
    all_voices = await edge_tts.list_voices()
    out = {}
    for lang, locale in LOCALES.items():
        vs = [v for v in all_voices if v["Locale"] == locale]
        fem = [v["ShortName"] for v in vs if v["Gender"] == "Female"]
        male = [v["ShortName"] for v in vs if v["Gender"] == "Male"]
        out[lang] = (fem[:1] + male[:1]) or [v["ShortName"] for v in vs[:2]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Music
# ─────────────────────────────────────────────────────────────────────────────

def _tone(freq, n, harmonics=(1.0,)):
    t = np.arange(n) / RATE
    return sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))


def _env(n, attack=0.01, decay=None):
    """Attack ramp, then exponential decay (time constant `decay` s) or hold."""
    e = np.ones(n)
    a = min(n, int(attack * RATE))
    e[:a] = np.linspace(0, 1, a)
    if decay:
        e *= np.exp(-np.arange(n) / (decay * RATE))
    r = min(n, int(0.02 * RATE))
    e[n - r:] *= np.linspace(1, 0, r)               # no clicks at the end
    return e


def music(seconds, rng):
    """A simple pop-style backing track: chords, bass, melody, drums."""
    n_total = int(seconds * RATE)
    bpm = rng.uniform(80, 130)
    beat = int(60 / bpm * RATE)
    root = 48 + int(rng.integers(0, 12))             # MIDI note of the key (C3..B3)
    major = [0, 2, 4, 5, 7, 9, 11]
    progression = [[0, 4, 7], [9, 12, 16], [5, 9, 12], [7, 11, 14]]   # I vi IV V
    shift = int(rng.integers(0, 4))
    progression = progression[shift:] + progression[:shift]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    out = np.zeros(n_total + 8 * beat)
    noise = rng.standard_normal(beat)
    bar = 0
    for start in range(0, n_total, 4 * beat):
        chord = progression[bar % 4]
        for note in chord:                           # pad: whole bar, soft
            out[start:start + 4 * beat] += 0.08 * _tone(hz(root + 12 + note), 4 * beat,
                                                       (1, .5, .25, .12)) * _env(4 * beat, 0.15)
        for b in range(4):
            s = start + b * beat
            out[s:s + beat] += 0.25 * _tone(hz(root - 12 + chord[0]), beat, (1, .4)) * _env(beat, 0.005, 0.25)
            if b % 2 == 0:                           # kick: falling sine
                k = int(0.15 * RATE)
                f = np.linspace(120, 45, k)
                out[s:s + k] += 0.5 * np.sin(2 * np.pi * np.cumsum(f) / RATE) * _env(k, 0.001, 0.05)
            else:                                    # snare: noise burst
                k = int(0.12 * RATE)
                out[s:s + k] += 0.2 * noise[:k] * _env(k, 0.001, 0.04)
            for half in (0, beat // 2):              # hi-hat: high-passed noise
                k = int(0.03 * RATE)
                out[s + half:s + half + k] += 0.06 * np.diff(noise[:k + 1]) * _env(k, 0.001, 0.01)
            for half in (0, beat // 2):              # melody: chord and scale notes
                if rng.random() < 0.7:
                    m = root + 24 + (chord[int(rng.integers(0, 3))] if rng.random() < .6
                                     else major[int(rng.integers(0, 7))])
                    k = beat // 2
                    out[s + half:s + half + k] += 0.1 * _tone(hz(m), k, (1, .3, .1)) * _env(k, 0.01, 0.15)
        bar += 1
    out = out[:n_total]
    return out / (np.sqrt((out ** 2).mean()) + 1e-9)    # unit RMS


def rms(x):
    return float(np.sqrt((x ** 2).mean()) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Building the clips
# ─────────────────────────────────────────────────────────────────────────────

async def build(count):
    rng = np.random.default_rng(SEED)
    voices = await voices_by_lang()
    rates = ["-10%", "+0%", "+8%"]
    for kind in ("speech", "songs"):
        (OUT / kind).mkdir(parents=True, exist_ok=True)
    manifest = []
    for folder, lang in FLEURS.items():
        texts = sentences(folder, lang)
        rng.shuffle(texts)
        pool = iter(texts)

        async def next_vocal(voice, rate):
            """The next sentence the voice can speak (edge-tts refuses a few)."""
            for text in pool:
                for _ in range(2):
                    try:
                        return text, await tts(text, voice, rate)
                    except Exception as e:
                        err = e
                print(f"  {lang}: skipped {text[:40]!r} ({err})", flush=True)
            return None, None

        made = {"speech": 0, "song": 0}
        for i in range(count):
            voice = voices[lang][i % len(voices[lang])]
            text, vocal = await next_vocal(voice, rates[i % 3])
            if text is None:
                break
            made["speech"] += 1
            pad = np.zeros(int(0.3 * RATE), np.float32)
            audio = np.concatenate([pad, vocal / rms(vocal) * 0.08, pad])
            path = OUT / "speech" / f"{lang}_{i:02d}.wav"
            save_wav(path, audio, RATE)
            manifest.append({"file": str(path.relative_to(OUT)), "kind": "speech", "lang": lang,
                             "text": text, "voice": voice,
                             "vocal_start": 0.3, "vocal_end": round(0.3 + len(vocal) / RATE, 2)})
        for i in range(count):
            voice = voices[lang][(i + 1) % len(voices[lang])]
            gap = np.zeros(int(0.4 * RATE), np.float32)
            pair, parts = [], []
            for _ in range(2):
                text, v = await next_vocal(voice, rates[i % 3])
                if text is not None:
                    pair.append(text); parts += [v, gap]
            if len(pair) < 2:
                break
            made["song"] += 1
            vocal = np.concatenate(parts[:-1])
            intro, outro = 2.0, 1.5
            backing = music(intro + len(vocal) / RATE + outro, rng)
            snr_db = rng.uniform(3, 9)                       # vocals 3..9 dB above the music
            mix = backing * (0.08 / 10 ** (snr_db / 20))
            s = int(intro * RATE)
            mix[s:s + len(vocal)] += vocal / rms(vocal) * 0.08
            mix /= max(1.0, np.abs(mix).max() / 0.95)
            path = OUT / "songs" / f"{lang}_{i:02d}.wav"
            save_wav(path, mix, RATE)
            manifest.append({"file": str(path.relative_to(OUT)), "kind": "song", "lang": lang,
                             "text": " ".join(pair), "voice": voice, "snr_db": round(snr_db, 1),
                             "vocal_start": intro, "vocal_end": round(intro + len(vocal) / RATE, 2)})
        print(f"  {lang}: {made['speech']} speech, {made['song']} songs", flush=True)
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(manifest)} clips -> {OUT}")


if __name__ == "__main__":
    asyncio.run(build(int(sys.argv[1]) if len(sys.argv) > 1 else 4))

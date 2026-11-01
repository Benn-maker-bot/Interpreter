"""
music.py — knowing when music or singing is playing, and hearing the words
=========================================================================

Whisper is trained mostly on speech. With music under the voice, two things
went wrong in the logs:

  1. The speech detector chopped singing into 1-second pieces (sung notes
     have no clear pauses), and Whisper cannot make sense of a single sung
     syllable: "Спустя.", "Готень за мной.", "Аааааа…". Joined into ~9 s,
     the same audio gave the real lyrics ("Крыла спустил… по тропинке шел
     со мной"). So in music the phrases must be longer.
  2. Loud instruments mask the voice.

This module provides:

  SoundClassifier  YAMNet (Google's AudioSet model, 521 sound classes, ONNX,
                   ~15 MB, ~5 ms per phrase on the CPU): how much a phrase
                   sounds like speech, music and singing.
  VocalSeparator   UVR MDX-Net "Kim_Vocal_2" (ONNX, 63 MB): removes the
                   instruments and keeps the voice. ~250 ms per 6 s chunk on
                   the GPU. It only runs while music plays, only if
                   onnxruntime can use the GPU, and does the chunks of a
                   phrase that is still going in advance (prefetch), so
                   little is left to do when the phrase ends.
  MusicScene       remembers the last few phrases and decides whether the
                   program is in "music" mode; the audio capture and the
                   transcription worker read it to adjust themselves.

Both models are downloaded once from Hugging Face (Apache-2.0 YAMNet export
by audiomagic; the UVR models are from the Ultimate Vocal Remover project).
"""

import csv
import os
import threading
import time
from collections import deque

import numpy as np

YAMNET_REPO = "audiomagic/yamnet-onnx"
VOCAL_REPO, VOCAL_FILE = "seanghay/uvr_models", "Kim_Vocal_2.onnx"

# AudioSet class names grouped into what matters here
SPEECH_CLASSES  = ("Speech", "Narration, monologue", "Conversation")
SINGING_CLASSES = ("Singing", "Choir", "Chant", "Rapping", "Humming", "Vocal music",
                   "Song", "Male singing", "Female singing", "Child singing")
MUSIC_CLASSES   = ("Music", "Musical instrument", "Pop music", "Rock music",
                   "Background music", "Theme music", "Soundtrack music")

# Music mode: entered when most of the last few phrases had music in them,
# left when they clearly did not. Two thresholds (hysteresis) so it doesn't
# flip on every phrase.
MUSIC_ON, MUSIC_OFF = 0.45, 0.25
SILENT_DBFS = -55        # a second quieter than this counts as silence
STATS_SEC = 10           # how often music_scene_stats is logged


def _onnx_session(path, gpu):
    """An onnxruntime session, on the GPU if asked and possible."""
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    if gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()          # the pip NVIDIA libraries (CUDA 12)
        except Exception:
            pass
        providers.insert(0, "CUDAExecutionProvider")
    options = ort.SessionOptions()
    # Few CPU threads: Teams/Telemost need the CPU more than this does
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, options, providers=providers)


class SoundClassifier:
    """YAMNet: how much a 16 kHz clip sounds like speech, singing and music."""

    def __init__(self):
        """Download YAMNet once (cached), and find which of its 521 sound
        classes count as speech, singing and music."""
        from huggingface_hub import hf_hub_download
        self.session = _onnx_session(hf_hub_download(YAMNET_REPO, "yamnet.onnx"), gpu=False)
        with open(hf_hub_download(YAMNET_REPO, "yamnet_class_map.csv"), encoding="utf-8") as f:
            names = [row[2] for row in list(csv.reader(f))[1:]]
        pick = lambda wanted: [i for i, n in enumerate(names) if n in wanted]
        self._speech, self._singing, self._music = (pick(SPEECH_CLASSES), pick(SINGING_CLASSES),
                                                    pick(MUSIC_CLASSES))

    def classify(self, audio_16k: np.ndarray) -> dict:
        """Scores 0-1 for speech, singing and music (each the strongest of its
        classes, averaged over the clip's ~0.5 s frames; singing uses the
        loudest frame, because YAMNet hears it only in places)."""
        audio = audio_16k.astype(np.float32)
        if len(audio) < 16000:                       # YAMNet needs ~1 s
            audio = np.pad(audio, (0, 16000 - len(audio)))
        t0 = time.perf_counter()
        frames = self.session.run(None, {"waveform": audio})[0]    # [frames, 521]
        mean = frames.mean(0)
        return {"speech":  round(float(mean[self._speech].max()), 3),
                "singing": round(float(frames[:, self._singing].max()), 3),
                "music":   round(float(mean[self._music].max()), 3),
                "classify_ms": round((time.perf_counter() - t0) * 1000, 1)}


class VocalSeparator:
    """UVR MDX-Net: keep the voice, remove the instruments.

    The model works on a spectrogram (the sound split into frequencies over
    time) of 44.1 kHz stereo audio: 7680-point FFT, 1024-sample hop, 3072
    frequency bins x 256 frames per chunk, real and imaginary parts of both
    channels as 4 input planes. These numbers are fixed by how the model was
    trained (the standard UVR settings for Kim_Vocal_2)."""

    N_FFT, HOP, DIM_F, DIM_T, COMPENSATE, RATE = 7680, 1024, 3072, 256, 1.009, 44100

    def __init__(self):
        """Download the UVR model once (cached), load it on the GPU, and
        work out the chunk sizes and the window shapes used for every chunk.

        chunk  audio one model call looks at (~5.9 s at 44.1 kHz)
        trim   the edges of each chunk's result are thrown away (they are
               less accurate), so chunks overlap by this much
        step   how far apart chunks start (chunk minus both trims)"""
        from huggingface_hub import hf_hub_download
        self.session = _onnx_session(hf_hub_download(VOCAL_REPO, VOCAL_FILE), gpu=True)
        self.on_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        self.chunk = self.HOP * (self.DIM_T - 1)
        self.trim = self.N_FFT // 2
        self.step = self.chunk - 2 * self.trim
        n = np.arange(self.N_FFT)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * n / self.N_FFT)).astype(np.float32)
        # Overlap-add normalisation for the inverse transform (same every chunk)
        norm = np.zeros(self.N_FFT + self.HOP * (self.DIM_T - 1), np.float32)
        for t in range(self.DIM_T):
            norm[t * self.HOP:t * self.HOP + self.N_FFT] += self.window ** 2
        self._norm = np.maximum(norm, 1e-8)
        self._frame_index = (np.arange(self.DIM_T)[:, None] * self.HOP
                             + np.arange(self.N_FFT)[None, :])

    # scipy.fft in single precision with 2 threads: numpy's double-precision
    # FFT took ~150 ms of each ~300 ms chunk (the GPU part is ~145 ms).
    def _stft(self, x):
        from scipy import fft
        x = np.pad(x, ((0, 0), (self.N_FFT // 2, self.N_FFT // 2)), mode="reflect")
        frames = (x[:, self._frame_index] * self.window).astype(np.float32)
        return fft.rfft(frames, axis=-1, workers=2).transpose(0, 2, 1)

    def _istft(self, spec):
        """The reverse of _stft: turn a spectrogram (frequencies over time)
        back into sound, adding the overlapping frames together."""
        from scipy import fft
        frames = fft.irfft(spec.transpose(0, 2, 1), n=self.N_FFT, axis=-1, workers=2) * self.window
        out = np.zeros((spec.shape[0], len(self._norm)), np.float32)
        for t in range(self.DIM_T):
            out[:, t * self.HOP:t * self.HOP + self.N_FFT] += frames[:, t]
        out /= self._norm
        return out[:, self.N_FFT // 2:self.N_FFT // 2 + self.chunk]

    # A chunk is reused from the cache only if its audio ends this far (44.1 kHz
    # samples) before the end of what has arrived: near the end, resampling
    # the longer phrase later can still change a few samples.
    STABLE_MARGIN = 4096

    def _chunk_voice(self, piece):
        """Separate one chunk (2 x chunk samples, 44.1 kHz) -> mono voice of
        `step` samples. One chunk per ONNX call: the input shape never
        changes, so cuDNN picks its convolution method once (a batch of all
        chunks changed shape with every phrase and made it search again: 2-4x
        slower)."""
        spec = self._stft(piece)[:, :self.DIM_F]
        planes = np.stack([spec[0].real, spec[0].imag, spec[1].real, spec[1].imag])
        out = self.session.run(None, {"input": planes[None].astype(np.float32)})[0][0]
        full = np.zeros((2, self.N_FFT // 2 + 1, self.DIM_T), np.complex64)
        full[0, :self.DIM_F] = out[0] + 1j * out[1]
        full[1, :self.DIM_F] = out[2] + 1j * out[3]
        return self._istft(full)[:, self.trim:-self.trim].mean(0)

    def _prepare(self, audio_16k, cache):
        """Resample to 44.1 kHz and pad. Resets `cache` if it belongs to a
        different phrase (its first 0.1 s differ)."""
        import soxr
        if cache is not None:
            head = audio_16k[:1600]
            if cache.get("head") is None or len(cache["head"]) != len(head) \
                    or not np.array_equal(cache["head"], head):
                cache.clear()
                cache.update(head=head.copy(), pieces={})
        x = soxr.resample(audio_16k.astype(np.float32), 16000, self.RATE).astype(np.float32)
        n = len(x)
        pad = self.step - n % self.step
        mix = np.concatenate([np.zeros(self.trim, np.float32), x,
                              np.zeros(pad + self.trim, np.float32)])
        return np.stack([mix, mix]), n, pad

    def _stable(self, i, n):
        """Chunk at offset i only uses audio that won't change any more."""
        return i + self.chunk - self.trim <= n - self.STABLE_MARGIN

    def prefetch(self, audio_16k, cache, max_chunks=1):
        """Separate up to max_chunks chunks of a phrase that is still growing,
        into `cache`, so separate() at the end of the phrase only has the last
        chunk left (a 12 s sung phrase: ~1 s -> ~0.3 s). Returns how many
        chunks it did."""
        mix, n, pad = self._prepare(audio_16k, cache)
        done = 0
        for i in range(0, n + pad, self.step):
            if done >= max_chunks or not self._stable(i, n):
                break
            if i not in cache["pieces"]:
                cache["pieces"][i] = self._chunk_voice(mix[:, i:i + self.chunk])
                done += 1
        return done

    def separate(self, audio_16k: np.ndarray, cache=None) -> np.ndarray:
        """Return the voice only, as 16 kHz mono float32 (same length).
        cache: a dict kept for one growing phrase; chunks prefetch() already
        separated are reused."""
        import soxr
        mix, n, pad = self._prepare(audio_16k, cache)
        pieces, reused = [], 0
        for i in range(0, n + pad, self.step):
            if cache is not None and i in cache["pieces"]:
                pieces.append(cache["pieces"][i])
                reused += 1
                continue
            piece = self._chunk_voice(mix[:, i:i + self.chunk])
            if cache is not None and self._stable(i, n):
                cache["pieces"][i] = piece
            pieces.append(piece)
        if cache is not None:
            cache["reused"], cache["chunks"] = reused, len(pieces)
        voice = np.concatenate(pieces)[:n] * self.COMPENSATE
        return soxr.resample(voice, self.RATE, 16000).astype(np.float32)[:len(audio_16k)]


class MusicScene:
    """Shared between the capture thread and the transcription worker: is
    music playing (judged from the last SCENE_WINDOW phrases)? Loads the
    models in the background; until they are ready nothing changes."""

    def __init__(self, enabled=True, separate=True, log=None):
        """enabled: music detection on at all. separate: also remove the
        instruments (needs the GPU). log(event, **fields): where to log.
        The models load in a background thread (_load), so creating this is
        instant; `active` stays False until the classifier is ready."""
        self.enabled, self.want_separation = enabled, separate
        self.active = False            # music mode on?
        self.singing = False           # ...and a voice in it
        self.classifier = self.separator = None
        self.error = None
        self._windows = deque(maxlen=8)    # (music score, silent) per second
        self._busy = False
        self._stats = self._new_stats()
        self._log = log or (lambda *a, **k: None)
        self._lock = threading.Lock()
        if enabled:
            threading.Thread(target=self._load, name="music-models", daemon=True).start()

    def _load(self):
        """Background thread: load YAMNet, then (if wanted and a GPU is
        available) the vocal separator, and run it once on silence so its
        first real use isn't slow. Problems are logged, never raised: music
        mode just stays off."""
        t0 = time.perf_counter()
        try:
            self.classifier = SoundClassifier()
            self._log("sound_classifier_ready", elapsed_ms=round((time.perf_counter() - t0) * 1000))
        except Exception as e:
            self.error = f"classifier: {e}"
            self._log("sound_classifier_failed", error=str(e))
            return
        if self.want_separation:
            t0 = time.perf_counter()
            try:
                separator = VocalSeparator()
                if not separator.on_gpu:
                    # ~1.5 s per phrase on many CPU cores: too heavy next to Teams
                    self._log("vocal_separator_skipped", reason="GPU not available to onnxruntime")
                    return
                separator.separate(np.zeros(16000, np.float32))      # warm-up
                self.separator = separator
                self._log("vocal_separator_ready", elapsed_ms=round((time.perf_counter() - t0) * 1000))
            except Exception as e:
                self._log("vocal_separator_failed", error=str(e))

    # ── Live decision, once per second (fast switching) ──────────────────
    # Judging whole phrases was too slow: three phrases (up to 15 s each in
    # music mode) before switching, and during silence no phrases arrive,
    # so the mode never changed (logs: ~14 s to notice music, stuck in music
    # mode through silence). Now the capture thread hands over the last
    # second of audio every second; YAMNet takes ~5 ms for it.

    def feed(self, audio_16k):
        """Called by the capture thread with the last second of 16 kHz audio
        (int16 or float). Never blocks: if the previous second is still being
        classified, this one is skipped."""
        if self.classifier is None or self._busy:
            return
        self._busy = True
        threading.Thread(target=self._classify_window, args=(audio_16k,),
                         name="music-scene", daemon=True).start()

    def _classify_window(self, audio):
        """Short-lived thread: measure the second's loudness, score it with
        YAMNet (skipped if it is silent) and update the decision."""
        try:
            audio = np.asarray(audio)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            # plain Python numbers: numpy types cannot be written to the JSON log
            level = float(10 * np.log10(float(np.mean(audio ** 2)) + 1e-12))
            silent = bool(level < SILENT_DBFS)
            scores = ({"speech": 0.0, "singing": 0.0, "music": 0.0, "classify_ms": 0.0}
                      if silent else self.classifier.classify(audio))
            self._update(scores, silent, level)
        except Exception as e:
            self._log("music_scene_failed", error=str(e))
        finally:
            self._busy = False

    def _update(self, scores, silent, level):
        """Add one second's result and decide whether music mode is on:
          ON   when 2 of the last 3 seconds score as music (>= MUSIC_ON)
          OFF  after 2 silent seconds, or 3 seconds without music (< MUSIC_OFF)
        Two different thresholds stop the mode flipping back and forth. Every
        change, and a summary every STATS_SEC, is logged."""
        with self._lock:
            self._windows.append((float(scores["music"]), silent))
            st = self._stats
            st["windows"] += 1
            st["silent"] += silent
            st["music_sum"] += scores["music"]
            st["speech_sum"] += scores["speech"]
            st["classify_ms_max"] = max(st["classify_ms_max"], scores.get("classify_ms", 0))
            recent = list(self._windows)
            was, reason = self.active, None
            if not self.active:
                # ON: 2 of the last 3 seconds clearly music
                if sum(1 for m, s in recent[-3:] if m >= MUSIC_ON) >= 2:
                    self.active, reason = True, "music"
            else:
                # OFF: 2 seconds of silence, or 3 seconds without music
                if len(recent) >= 2 and all(s for _, s in recent[-2:]):
                    self.active, reason = False, "silence"
                elif len(recent) >= 3 and all(m < MUSIC_OFF and not s for m, s in recent[-3:]):
                    self.active, reason = False, "speech"
            if self.active != was:
                st["switches"] += 1
                self._log("music_mode_changed", active=self.active, reason=reason,
                          last_seconds=[round(m, 2) if not s else "silent" for m, s in recent[-3:]],
                          level_dbfs=round(level, 1))
            if time.monotonic() - st["t0"] >= STATS_SEC and st["windows"]:
                n = st["windows"]
                self._log("music_scene_stats", windows=n, silent_windows=st["silent"],
                          music_mean=round(st["music_sum"] / n, 3),
                          speech_mean=round(st["speech_sum"] / n, 3),
                          classify_ms_max=st["classify_ms_max"], switches=st["switches"],
                          music_mode=self.active, singing=self.singing)
                self._stats = self._new_stats()

    @staticmethod
    def _new_stats():
        """Empty counters for the next music_scene_stats log line."""
        return {"t0": time.monotonic(), "windows": 0, "silent": 0, "music_sum": 0.0,
                "speech_sum": 0.0, "classify_ms_max": 0.0, "switches": 0}

    def judge(self, audio_16k):
        """Scores for one finished phrase, for its log line ({} if the
        classifier is not ready). The mode itself is decided by feed()."""
        if self.classifier is None:
            return {}
        scores = self.classifier.classify(audio_16k)
        return {**{f"sound_{k}": v for k, v in scores.items()},
                "music_mode": self.active, "singing": self.singing}

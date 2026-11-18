"""Sound in and out: finding devices, recording, and deciding what is speech.

- get_audio_devices() and the find_best_* helpers list the devices and
  pick sensible defaults from their names.
- AudioCaptureThread records one device in ~32 ms blocks, measures the
  level, asks Silero VAD whether it is speech and cuts the speech into
  phrases. It is the start of the pipeline.
- LoopbackOutputThread plays to an output device, and the keep-alive
  helpers play silence so Windows keeps delivering loopback audio.
- rms(), levels_dbfs() and apply_agc() measure and adjust audio levels.
- PerfWindow (timing statistics) and LatestSlot (a one-item mailbox for
  live previews) are small helpers used here and by the other threads.

PortAudio (through PyAudio) is the library that talks to the sound card.
Everything here reads the module-level name `pyaudio`, so simulate.py can
swap in a fake sound card by setting interpreter.audio.pyaudio.
"""

import queue
import threading
import time
import math
import logging
from collections import deque
from typing import Iterable

from .config import (
    AGC_MAX_GAIN_DB, AGC_PEAK_DBFS, AGC_TARGET_DBFS, CHUNK_SAMPLES, CUT_SEARCH_SEC,
    DEFAULT_SENS, EARLY_FINAL_SEC, FIIO_KEYWORDS, FIIO_PRIORITY_KEYWORDS,
    KEEPALIVE_ERROR_AFTER_TRIES, MAX_PHRASE_SEC, MAX_READ_FAILURES, MIC_SILENT_SEC,
    MIN_PHRASE_SEC, MME_NAME_MAX, MUSIC_MAX_PHRASE_SEC, MUSIC_SILENCE_SEC,
    MUSIC_SOFT_PHRASE_SEC, MUSIC_SOFT_SILENCE_SEC, MUSIC_VAD_DROP, NVIDIA_KEYWORDS,
    ONSET_PROB, PARTIAL_INTERVAL_SEC, PARTIAL_MIN_SEC, PHRASE_QUEUE_BACKLOG_WARN,
    PREROLL_SEC, SAMPLE_RATE, SAMPLE_WIDTH, SILENCE_FLOOR_DBFS, SILENCE_SEC,
    SOURCE_SILENT_SEC, SPEECH_SOFT_PHRASE_SEC, SPEECH_SOFT_SILENCE_SEC,
    STATS_INTERVAL_SEC, VAD_HYSTERESIS, VAD_WINDOW_SEC, VB_INPUT_PRIORITY_KEYWORDS,
    VB_KEYWORDS)
from .logging_setup import log_event, log_exception
from .system import audio_sessions_snapshot, boost_audio_thread


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


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
#
# Functions for finding audio devices and their capabilities. Windows
# exposes each device several times through different "host APIs"
# (WASAPI, DirectSound, MME…); these helpers keep the best one and pick
# sensible defaults (FiiO for output, the FiiO's loopback for Friend mode).
# ─────────────────────────────────────────────────────────────────────────────

# PortAudio counts how many PyAudio objects are open: it really starts on
# the first and really shuts down (closing every stream) when the last one
# is terminated. Updating that count is NOT thread-safe. With two channels,
# one thread may create a PyAudio object (starting a channel, re-scanning
# devices) while another thread creates or terminates one; the count can go
# wrong and PortAudio shuts down under a running stream: a crash. So every
# PyAudio() and terminate() goes through these two helpers, which take turns
# using PORTAUDIO_LOCK.
PORTAUDIO_LOCK = threading.Lock()


def portaudio_start():
    """Create a PyAudio object (starts PortAudio), safely across threads."""
    with PORTAUDIO_LOCK:
        return pyaudio.PyAudio()


def portaudio_stop(portaudio) -> None:
    """Terminate a PyAudio object created by portaudio_start()."""
    with PORTAUDIO_LOCK:
        portaudio.terminate()


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


def find_best_vb_input(input_devices: Iterable[tuple[int, str]]) -> str | None:
    """Return the label of the best VB-Audio input, preferring Cable A."""
    for priority_kw in VB_INPUT_PRIORITY_KEYWORDS:
        for _idx, label in input_devices:
            if priority_kw in label.lower():
                return label
    return None


def get_audio_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
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
    inputs: list[tuple[int, str]] = []
    outputs: list[tuple[int, str]] = []
    seen_in: dict[str, bool] = {}
    seen_out: dict[str, bool] = {}

    def known(name: str, seen: dict[str, bool]) -> bool:
        """True if this device is already listed. MME cuts names to 31
        characters ("SPDIF Interface (Realtek USB Au"), so a cut name that
        starts a name already listed (from WASAPI) is the same device; it
        used to appear twice in the menus, once with a truncated name."""
        return name in seen or (len(name) == MME_NAME_MAX
                                and any(s.startswith(name) for s in seen))

    portaudio = None
    try:
        portaudio = portaudio_start()
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
                if info.get("maxInputChannels", 0) >= 1 and not known(name, seen_in):
                    seen_in[name] = True
                    prefix = "\U0001f50a " if info.get("isLoopbackDevice")           else \
                             "\U0001f3a4 " if any(k in name_lower for k in NVIDIA_KEYWORDS) else \
                             "\u2605 "      if any(k in name_lower for k in VB_KEYWORDS)     else ""
                    inputs.append((i, prefix + name))
                # A device that can play (has output channels) goes in the output list;
                # headphone symbol = FiiO, star = VB-Audio cable.
                if info.get("maxOutputChannels", 0) >= 1 and not known(name, seen_out):
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
            portaudio_stop(portaudio)
    # If nothing was found, offer the system defaults (-1 = default device).
    return (inputs or [(-1, "Default Microphone")],
            outputs or [(-1, "Default Output")])


def find_best_friend_input(input_devices: Iterable[tuple[int, str]]) -> str | None:
    """Default input for Friend mode: capture what the FiiO is playing (the
    call audio from Teams/Telemost) via WASAPI loopback, else a VB-Audio cable."""
    loopbacks = [(i, l) for i, l in input_devices if "[loopback]" in l.lower()]
    for priority_kw in FIIO_PRIORITY_KEYWORDS:
        for _idx, label in loopbacks:
            if priority_kw in label.lower():
                return label
    return find_best_vb_input(input_devices)


def find_device_by_keywords(devices: Iterable[tuple[int, str]],
                            keywords: Iterable[str]) -> str | None:
    """Return the first device label whose name (lowercased) contains any keyword.
    Loopback (output-capture) devices are skipped: "nvidia" would otherwise
    match "NVIDIA High Definition Audio [Loopback]" (a monitor's speakers)."""
    for _idx, label in devices:
        if "[loopback]" in label.lower():
            continue
        if any(kw in label.lower() for kw in keywords):
            return label
    return None


def find_best_output_device(output_devices: Iterable[tuple[int, str]]) -> str | None:
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
            # A WARNING, not an error: the capture loop keeps retrying, and it
            # usually works within seconds. (The RODECaster refuses this
            # silent output stream for ~5 s while its microphone is already
            # recording.) It becomes an ERROR after KEEPALIVE_ERROR_AFTER_TRIES.
            log_event(logging.WARNING, "loopback_keepalive_failed",
                      exception_type=type(e).__name__, exception_message=str(e),
                      device=loopback_info.get("name"))
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
        pa     = portaudio_start()
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
            portaudio_stop(pa)
        log_event(logging.INFO, "loopback_output_stopped")

# ─────────────────────────────────────────────────────────────────────────────
# Audio capture + VAD
#
# AudioCaptureThread is the start of the pipeline: it records audio and
# decides what is speech. PerfWindow and LatestSlot are small helpers it
# uses. VAD = Voice Activity Detection (Silero VAD, a tiny neural network).
# ─────────────────────────────────────────────────────────────────────────────


class PerfWindow:
    """
    Collects samples per metric and summarises them as n/p50/p95/max for a
    periodic stats log line, then resets. Cheap enough for the audio loop.
    
    This class is used to collect performance metrics over time periods and
    summarize them with statistical information (count, median, 95th percentile,
    maximum). It's used in performance monitoring and diagnostics.
    """
    def __init__(self):
        """
        Start an empty collection of samples.
        
        This method initializes the performance window to collect metrics
        for performance analysis.
        """
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
    """
    Single-item mailbox: put() overwrites, take() empties. Live previews
    only matter if current, so a stale one is replaced rather than queued.
    
    This class is used to manage live preview text updates. It's designed to
    ensure that only the most recent preview is displayed, since older
    previews would be irrelevant once new ones arrive.
    """
    def __init__(self):
        """
        Start empty.
        
        This method initializes an empty slot for storing preview data.
        """
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
    """
    Records audio from one device and turns it into phrases of speech.
    
    This thread is responsible for capturing raw audio data from the selected
    input device (microphone or loopback). It processes this audio in small
    chunks (~32 ms each) to determine when speech begins and ends, then
    groups these chunks into meaningful phrases.
    
    For every ~32 ms block of sound it: measures the level (for the meter),
    asks Silero VAD whether it is speech, and adds speech to the current
    phrase. Finished phrases go to `phrase_queue` (for the final, accurate
    transcription); while a phrase is still growing, a copy is put in
    `partial_slot` every PARTIAL_INTERVAL_SEC for a quick preview.
    
    For "loopback" devices (what an output device is playing) it also plays
    silence to that device, because Windows delivers no loopback audio at all
    while the device is silent, which would make reading hang.
    
    This thread is crucial for the real-time processing of audio. It handles
    the timing and detection of speech boundaries to ensure proper phrase
    segmentation.
    """
    def __init__(self, device_index, phrase_queue, status_cb, level_cb,
                 loopback_out: "LoopbackOutputThread | None" = None,
                 partial_slot: "LatestSlot | None" = None,
                 mute_evt: "threading.Event | None" = None,
                 early_slot: "LatestSlot | None" = None,
                 channel: str = ""):
        """
        Store settings; the device is opened later, in the thread itself (run).
        
        This method initializes the audio capture thread with all necessary
        parameters for capturing and processing audio.
        
        Args:
            device_index: Which PortAudio device to record from
            phrase_queue: Where finished phrases are sent as (seq, audio, info)
            status_cb: Function to show a status message in the window
            level_cb: Function that receives (rms_dbfs, peak_dbfs, is_speech)
            loopback_out: Optional LoopbackOutputThread to copy audio to
            partial_slot: Where preview audio is placed (LatestSlot)
            mute_evt: While set, audio is replaced by silence (our own speech)
            channel: Name of the channel this device belongs to ("Me", "Friend")
        """
        super().__init__(daemon=True, name=f"audio-capture-{channel}" if channel else "audio-capture")
        self.channel           = channel
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
        self.preview_interval_sec = PARTIAL_INTERVAL_SEC   # may be raised by the window
        self.music_scene = None      # music.MusicScene, set by the window (music mode)
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

    def _report_source(self, silent, silent_sec):
        """The captured device went silent (no app playing) or came back.
        Logs it with the apps that have sound sessions, and tells the user.
        The app list is read on a short-lived thread (Windows audio API
        calls can take tens of ms; the capture loop must not wait).

        A microphone is different: its noise gate (or mute) sends exact
        silence whenever nobody speaks, which is normal. It is reported only
        after MIC_SILENT_SEC, as information. (On 2026-11-15, 24 of 28
        warnings in 8 minutes were a RODECaster microphone's pauses.)"""
        if not self.is_loopback:
            log_event(logging.INFO, "source_silent" if silent else "source_resumed",
                      channel=self.channel, device_index=self.device_index,
                      silent_sec=silent_sec, kind="microphone")
            self.status_cb(f"🔇 Microphone silent for {silent_sec:.0f} s (muted?)" if silent
                           else "Listening…")
            return

        def report():
            sessions = audio_sessions_snapshot()
            if silent:
                log_event(logging.WARNING, "source_silent", channel=self.channel,
                          device_index=self.device_index,
                          silent_sec=silent_sec, playing_apps=sessions,
                          hint="no sound reaches the device: the playing app stopped or "
                               "plays to a different device")
            else:
                log_event(logging.INFO, "source_resumed", channel=self.channel,
                          device_index=self.device_index,
                          silent_sec=silent_sec, playing_apps=sessions)
        threading.Thread(target=report, name="source-check", daemon=True).start()
        self.status_cb("🔇 No sound is playing on this device" if silent else "Listening…")

    @staticmethod
    def _best_cut(levels_db, speech_probs, search_blocks):
        """Index of the block to end a forced cut on: the quietest moment in
        the last `search_blocks` blocks. In continuous speech the speech score
        stays ~0.9 even inside words, so using it alone cut through words
        ("Вродали" for "продали" in the logs). Loudness dips mark the real
        gaps between words, so each block is scored by its loudness averaged
        with its neighbours (one loud click doesn't matter), plus the speech
        score as a tie-breaker (20 dB per 100 %)."""
        n = len(levels_db)
        start = max(1, n - search_blocks)
        def score(i):
            window = range(i - 1, min(n, i + 2))
            return (sum(levels_db[j] for j in window) / len(window)
                    + 20 * sum(speech_probs[j] for j in window) / len(window))
        return min(range(start, n - 1), key=score)

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
            log_event(logging.INFO, "phrase_flushed", channel=self.channel,
                      reason=label, seq=seq, bytes=len(payload),
                      chunks=len(blocks), queue_backlog=backlog)
            if backlog >= PHRASE_QUEUE_BACKLOG_WARN:
                log_event(logging.WARNING, "phrase_queue_backlog_high", backlog=backlog)
        except queue.Full:
            log_event(logging.ERROR, "phrase_dropped_queue_full", seq=seq,
                      reason=label, audio_sec=round(len(payload) / 2 / SAMPLE_RATE, 1),
                      queue_size=self.phrase_queue.maxsize)

    def run(self):
        """Thread entry point: create PortAudio, run the capture, always clean up."""
        log_event(logging.INFO, "audio_thread_priority", applied=boost_audio_thread())
        pa     = portaudio_start()
        try:
            self._run(pa)
        finally:
            if self._keepalive is not None:
                try:
                    self._keepalive.close()
                except Exception:
                    pass
            portaudio_stop(pa)

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
            log_event(logging.INFO, "audio_stream_opened", channel=self.channel,
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
        # The same two limits while music plays (music mode)
        speech_limits = (silence_blocks_needed, max_phrase_blocks)
        music_limits  = (max(1, int(MUSIC_SILENCE_SEC * blocks_per_sec)),
                         int(MUSIC_MAX_PHRASE_SEC * blocks_per_sec))
        music_soft_blocks = int(MUSIC_SOFT_PHRASE_SEC * blocks_per_sec)
        music_soft_silence_blocks = max(1, int(MUSIC_SOFT_SILENCE_SEC * blocks_per_sec))
        speech_soft_blocks = int(SPEECH_SOFT_PHRASE_SEC * blocks_per_sec)
        speech_soft_silence_blocks = max(1, int(SPEECH_SOFT_SILENCE_SEC * blocks_per_sec))
        # Every second, the last second of audio goes to the music detector
        # (music.MusicScene.feed), so it can switch within ~2 s, also when
        # nothing is said and no phrase arrives.
        scene_blocks = max(1, int(round(blocks_per_sec)))
        scene_buffer = deque(maxlen=scene_blocks)
        was_music = False
        cut_search_blocks       = int(CUT_SEARCH_SEC * blocks_per_sec)            # where to look for the cut
        preview_interval_blocks = max(1, int(self.preview_interval_sec * blocks_per_sec))  # time between previews
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
        silent_report_sec = SOURCE_SILENT_SEC if self.is_loopback else MIC_SILENT_SEC
        source_silent_blocks = int(silent_report_sec / chunk_sec)
        stats         = PerfWindow()          # timings for the capture_stats log line
        last_read_t   = None                  # when the previous block was read
        speech_frames = frames = 0            # blocks judged speech / all blocks (for stats)
        silent_frames = silent_run = 0        # blocks of no sound at all (total / current run)
        # Per-phrase measurements, logged with the phrase's trace:
        # when its speech started, the level of each block, its loudest peak,
        # and how many previews were made of it.
        phrase_start_time, phrase_levels_db, phrase_peak_db, phrase_preview_count = 0.0, [], -100.0, 0

        def phrase_meta(n_chunks):
            """Facts about the phrase being sent (its first n_chunks blocks):
            when speech started, length, loudness, speech score, previews.
            They travel with the audio and end up in its phrase_trace."""
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
                                  channel=self.channel, retries=keepalive_tries)
                    elif keepalive_tries == KEEPALIVE_ERROR_AFTER_TRIES:
                        # Not a passing hiccup any more: now it is an error.
                        log_event(logging.ERROR, "loopback_keepalive_still_failing",
                                  channel=self.channel, device_index=self.device_index,
                                  retries=keepalive_tries,
                                  hint="no sound arrives from this output device while "
                                       "nothing plays on it; still retrying every second")
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
                    log_event(logging.ERROR, "audio_capture_failed", channel=self.channel,
                              device_index=self.device_index, failures=read_failures,
                              error=str(e))
                    self.status_cb("⚠ Audio device stopped responding - reconnecting…")
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
            # Music mode: longer phrases and a lower speech threshold
            if self.music_scene is not None:
                scene_buffer.append(data)
                if len(scene_buffer) == scene_blocks:
                    self.music_scene.feed(np.frombuffer(_resample(b"".join(scene_buffer)),
                                                        dtype=np.int16))
                    scene_buffer.clear()
            in_music = bool(self.music_scene is not None and self.music_scene.active)
            if in_music != was_music:
                was_music = in_music
                self.status_cb("♪ Music detected: tuned for lyrics" if in_music
                               else "Listening…")
            silence_blocks_needed, max_phrase_blocks = music_limits if in_music else speech_limits
            if in_music:
                speech_threshold = max(0.15, speech_threshold - MUSIC_VAD_DROP)
                if len(phrase_blocks) >= music_soft_blocks:
                    silence_blocks_needed = music_soft_silence_blocks
            elif len(phrase_blocks) >= speech_soft_blocks:
                silence_blocks_needed = speech_soft_silence_blocks
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
            # ── Is anything playing? A loopback device delivers exact zeros
            #    when no app plays sound; the stream itself keeps running.
            #    Log when the source goes quiet / comes back, with which apps
            #    were playing, so "nothing was heard" can be traced to the
            #    source (a paused browser tab) or to this program.
            if level_db <= SILENCE_FLOOR_DBFS:
                silent_frames += 1
                silent_run += 1
                if silent_run == source_silent_blocks:
                    self._report_source(silent=True, silent_sec=silent_report_sec)
            else:
                if silent_run >= source_silent_blocks:
                    self._report_source(silent=False, silent_sec=round(silent_run * chunk_sec, 1))
                silent_run = 0
            if stats.due():
                log_event(logging.INFO, "capture_stats", channel=self.channel,
                          device_index=self.device_index,
                          chunk_ms=round(chunk_sec * 1000, 1),
                          speech_pct=round(100 * speech_frames / max(1, frames), 1),
                          silent_pct=round(100 * silent_frames / max(1, frames), 1),
                          vad_threshold=round(speech_threshold, 3),
                          speech_score_histogram=prob_histogram, **stats.summary())
                frames = speech_frames = silent_frames = 0
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
                    # When speech most likely began: the moment the speech score
                    # first rose (possible_onset_t), not the start of the 0.6 s
                    # pre-roll, which made "speech -> first preview" look ~0.5 s
                    # slower than it was.
                    phrase_start_time = possible_onset_t or time.monotonic()
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
                # Why it ended (which pause rule) and how the speech score fell
                # at the end: shows whether phrases end too early or too late.
                rule = ("music, long phrase" if in_music and silence_blocks_needed == music_soft_silence_blocks
                        else "music" if in_music
                        else "speech, long phrase" if silence_blocks_needed == speech_soft_silence_blocks
                        else "speech")
                self._flush(seq, phrase_blocks, min_phrase_blocks, "end", _resample,
                            dict(phrase_meta(len(phrase_blocks)), end_rule=rule,
                                 pause_ms=round(silent_blocks * chunk_sec * 1000),
                                 end_probs=[round(p, 2) for p in phrase_speech_probs[-silent_blocks - 3:]][:8]))
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
                cut = self._best_cut(phrase_levels_db, phrase_speech_probs, cut_search_blocks)
                early_offer_len = None  # an early final never applies to a forced cut
                self._flush(seq, phrase_blocks[:cut + 1], min_phrase_blocks, "max", _resample,
                            dict(phrase_meta(cut + 1), end_rule=f"max length ({'music' if in_music else 'speech'})",
                                 cut_prob=round(phrase_speech_probs[cut], 2),
                                 cut_level_db=round(phrase_levels_db[cut], 1)))
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


def plain_device_name(label: str) -> str:
    """'🔊 Speakers (FiiO Q series) [Loopback]' -> 'Speakers (FiiO Q series)'."""
    import re
    return re.sub(r"^[^\w(]+", "", label).replace(" [Loopback]", "").strip()

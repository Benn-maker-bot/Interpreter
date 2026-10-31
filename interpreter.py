"""
Voice Interpreter — optimised for minimum latency.

Speed stack:
  • Auto-detects NVIDIA GPU; falls back to CPU int8 automatically if CUDA
    libraries are missing (e.g. cublas64_12.dll not installed).
  • Model pre-warmed with a dummy buffer after load.
  • STREAM_CHUNK_SEC = 1.2 s mid-speech flush; SILENCE_SEC = 0.25 s end-phrase.
  • 3 parallel workers on CPU (1 on GPU).
  • Persistent requests.Session + LRU cache for translation.
  • VB-Audio loopback: captures system audio for translation; a separate
    loopback output thread simultaneously plays back audio to the Fiio output.

Bug fixes in this revision:
  • VB-Audio loopback output: audio from the virtual cable is now also written
    to the selected output device (Fiio) so you can hear it through headphones.
  • TTS "run loop already started": pyttsx3 is now guarded with a lock and the
    engine is re-initialised on failure instead of crashing silently.
  • Translation queue unbounded backlog: the queue is now bounded (64 items)
    and excess phrases are dropped with a warning; a second parallel translation
    worker is spawned on CPU to drain the queue faster.
  • Diagnostics tab text invisible: rowconfigure weight fixed so the textbox
    fills the available space and is scrollable.
  • Diagnostics summary label wraps correctly with a larger fixed height frame.
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

try:
    import pyaudio
    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paths / log config ────────────────────────────────────────────────────────
APP_DIR               = Path(__file__).resolve().parent
LOG_PATH              = APP_DIR / "interpreter.log.jsonl"
ERROR_LOG_PATH        = APP_DIR / "interpreter.error.log.jsonl"
LOG_SUMMARY_PATH      = APP_DIR / "interpreter.log.summary.json"
MAX_LOG_BYTES         = 5 * 1024 * 1024
LOG_BACKUP_COUNT      = 5

# ── Tuning ────────────────────────────────────────────────────────────────────
MAX_TEXT_HISTORY_CHARS         = 12_000
PHRASE_QUEUE_BACKLOG_WARN      = 6
TRANSLATION_QUEUE_BACKLOG_WARN = 16
TRANSLATION_WORKERS            = 2          # parallel translation threads
TRANSLATION_QUEUE_MAX          = 64


# ─────────────────────────────────────────────────────────────────────────────
# Structured JSONL logging
# ─────────────────────────────────────────────────────────────────────────────

class JsonLineFormatter(logging.Formatter):
    def format(self, record):
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
    def doRollover(self):
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
    def format(self, record):
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


APP_LOG   = _build_logger("interpreter.app",   LOG_PATH,       logging.INFO)
ERROR_LOG = _build_logger("interpreter.error", ERROR_LOG_PATH, logging.WARNING)


def log_event(level: int, event: str, **fields):
    APP_LOG.log(level, event, extra={"event": event, "fields": fields})
    if level >= logging.WARNING:
        ERROR_LOG.log(level, event, extra={"event": event, "fields": fields})


def log_exception(event: str, exc: Exception, **fields):
    merged = dict(fields)
    merged.setdefault("exception_type",    type(exc).__name__)
    merged.setdefault("exception_message", str(exc))
    APP_LOG.error(   event, exc_info=(type(exc), exc, exc.__traceback__), extra={"event": event, "fields": merged})
    ERROR_LOG.error( event, exc_info=(type(exc), exc, exc.__traceback__), extra={"event": event, "fields": merged})


def install_global_exception_hooks():
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


install_global_exception_hooks()
log_event(logging.INFO, "app_module_loaded",
          app_dir=str(APP_DIR), log_path=str(LOG_PATH), error_log_path=str(ERROR_LOG_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Log summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def iter_log_files(base_path: Path):
    files = [base_path] if base_path.exists() else []
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*.gz")))
    return files


def read_jsonl_entries(path: Path):
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
    levels, events, issues, total = Counter(), Counter(), [], 0
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
SAMPLE_RATE      = 16_000
CHUNK_SAMPLES    = 512
CHANNELS         = 1
SAMPLE_WIDTH     = 2
SILENCE_SEC      = 0.25
STREAM_CHUNK_SEC = 1.2
MAX_PHRASE_SEC   = 6
MIN_PHRASE_SEC   = 0.10
CAL_MULTIPLIER   = 1.3
DEFAULT_SENS     = 0.6

# VB-Audio virtual cables appear as "CABLE-A Output", "CABLE Output", etc.
# Voicemeeter mix buses are deliberately excluded so we capture the raw cable.
VB_KEYWORDS     = ["cable output", "cable-a output", "cable-b output",
                   "vb-audio virtual", "virtual cable"]
NVIDIA_KEYWORDS = ["nvidia", "broadcast", "rtx voice"]
# Fiio output devices: FiiO K7, K3, K5, K9, Q3, Q5, Q7, Q series, KA1, KA3, KA5
FIIO_KEYWORDS   = ["fiio", "fio k", "fio q", "fio dac",
                   "k3)", "k5)", "k7)", "k9)",
                   "q3)", "q5)", "q7)",
                   "ka1", "ka3", "ka5",
                   "btr", "realtek usb audio"]

# ── Whisper models ────────────────────────────────────────────────────────────
MODELS = {
    "Tiny  — fastest":        "tiny",
    "Small — recommended ✓":  "small",
    "Medium — most accurate": "medium",
}
DEFAULT_MODEL_CPU = "Tiny  — fastest"
DEFAULT_MODEL_GPU = "Small — recommended ✓"

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
COL_MY_VOICE = ("#1A6B3A", "#1A6B3A")
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
# ─────────────────────────────────────────────────────────────────────────────

def detect_device():
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        # On Windows, just because get_cuda_device_count() > 0 doesn't mean
        # the cuBLAS libraries are actually in the PATH. If they are missing,
        # CTranslate2 throws a RuntimeError at transcribe() time, which can deadlock.
        # We explicitly verify we can load the DLL before trusting 'cuda'.
        import os
        if n > 0 and os.name == "nt":
            import ctypes
            try:
                ctypes.cdll.LoadLibrary("cublas64_12.dll")
            except Exception:
                try:
                    ctypes.cdll.LoadLibrary("cublas64_11.dll")
                except Exception:
                    log_event(logging.WARNING, "cuda_detection_dll_missing")
                    n = 0
                    
        log_event(logging.INFO, "cuda_detection_result", cuda_device_count=n)
        if n > 0:
            return "cuda", "float16", 1, DEFAULT_MODEL_GPU
    except Exception as e:
        log_exception("cuda_detection_failed", e)
    return "cpu", "int8", 2, DEFAULT_MODEL_CPU


WHISPER_DEVICE, COMPUTE_TYPE, NUM_WORKERS, _DEFAULT_MODEL = detect_device()
INFERENCE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Translation — persistent session + LRU cache + multiple workers
# ─────────────────────────────────────────────────────────────────────────────

_http = requests.Session()
_http.headers.update({"User-Agent": "Mozilla/5.0"})
_http.adapters.clear()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
_http.mount("http://",  _adapter)
_http.mount("https://", _adapter)


@lru_cache(maxsize=512)
def _translate_cached(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    try:
        url    = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
        t0     = time.perf_counter()
        resp   = _http.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data       = resp.json()
        translated = "".join(p[0] for p in data[0] if p[0]).strip()
        log_event(logging.INFO, "translation_completed",
                  src=src, tgt=tgt,
                  input_chars=len(text), output_chars=len(translated),
                  elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                  cache_info=_translate_cached.cache_info()._asdict())
        return translated
    except Exception as e:
        log_exception("translation_failed", e, src=src, tgt=tgt, input_chars=len(text))
        raise RuntimeError(f"Translation failed: {e}") from e


class TranslationWorker(threading.Thread):
    def __init__(self, wid: int, request_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__(daemon=True, name=f"translation-worker-{wid}")
        self.wid           = wid
        self.request_queue = request_queue
        self.result_queue  = result_queue
        self._stop_evt     = threading.Event()

    def stop(self):
        self._stop_evt.set()
        self.request_queue.put(None)

    def run(self):
        while not self._stop_evt.is_set():
            item = self.request_queue.get()
            if item is None:
                break
            text, src_lang_key, tgt_lang_key = item
            _, src = LANGUAGES.get(src_lang_key, ("en", "en"))
            _, tgt = LANGUAGES.get(tgt_lang_key, ("ru", "ru"))
            try:
                t0  = time.perf_counter()
                out = _translate_cached(text, src, tgt)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                self.result_queue.put(("translated",   out or text))
                self.result_queue.put(("metric", ("translation_ms", elapsed_ms)))
            except Exception as e:
                self.result_queue.put(("status", f"Translation error: {e}"))


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device_native_rate(pa_instance, device_index) -> int:
    """Return the native sample rate reported by PortAudio for a device.
    Falls back to 44100 if the index is invalid or the query fails."""
    if device_index is None or device_index < 0:
        return 44100
    try:
        info = pa_instance.get_device_info_by_index(device_index)
        return int(info.get("defaultSampleRate", 44100))
    except Exception:
        return 44100


def rms(data: bytes) -> float:
    n = len(data) // SAMPLE_WIDTH
    if n == 0:
        return 0.0
    samples = struct.unpack(f"{n}h", data[:n * SAMPLE_WIDTH])
    return math.sqrt(sum(s * s for s in samples) / n)


# Host API preference order for PortAudio on Windows.
# WDM-KS (usually index 3) requires exclusive device access and will fail with
# -9999 "Unanticipated host error" when the Windows audio engine already has
# the device open.  We skip it entirely.
_PREFERRED_HOST_API_ORDER = ["Windows WASAPI", "Windows DirectSound", "MME"]
_SKIP_HOST_API_NAMES      = {"Windows WDM-KS"}


def _get_host_api_map(pa):
    """Return {api_index: api_name} and the preferred-order api_index list."""
    api_map  = {}
    pref_idx = []
    other    = []
    skip_set = set()
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
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


def probe_openable_rate(pa, device_index, is_input):
    """Try common sample rates and return the first one PortAudio accepts."""
    candidates = [48000, 44100, 16000, 22050, 32000, 8000]
    for rate in candidates:
        try:
            if is_input:
                pa.is_format_supported(
                    rate,
                    input_device=device_index,
                    input_channels=1,
                    input_format=pyaudio.paInt16,
                )
            else:
                pa.is_format_supported(
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

    inputs, outputs, seen_in, seen_out = [], [], {}, {}
    try:
        pa = pyaudio.PyAudio()
        api_map, pref_order, skip_apis = _get_host_api_map(pa)

        # Collect all devices grouped by name, in preferred-API order
        for api_idx in pref_order:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("hostApi") != api_idx:
                    continue
                name = info["name"]
                nl   = name.lower()
                if info.get("maxInputChannels", 0) >= 1 and name not in seen_in:
                    seen_in[name] = True
                    prefix = "\U0001f3a4 " if any(k in nl for k in NVIDIA_KEYWORDS) else \
                             "\u2605 "      if any(k in nl for k in VB_KEYWORDS)     else ""
                    inputs.append((i, prefix + name))
                if info.get("maxOutputChannels", 0) >= 1 and name not in seen_out:
                    seen_out[name] = True
                    is_fiio = any(k in nl for k in FIIO_KEYWORDS)
                    is_vb   = any(k in nl for k in VB_KEYWORDS)
                    prefix  = "\U0001f3a7 " if is_fiio else \
                              "\u2605 "      if is_vb   else ""
                    outputs.append((i, prefix + name))

        try:
            default_in  = pa.get_default_input_device_info().get("index")
            default_out = pa.get_default_output_device_info().get("index")
        except Exception:
            default_in = default_out = None
        log_event(logging.INFO, "audio_devices_enumerated",
                  input_count=len(inputs), output_count=len(outputs),
                  default_in=default_in, default_out=default_out,
                  preferred_api_order=[api_map.get(i) for i in pref_order[:4]])
    except Exception as e:
        log_exception("audio_device_enumeration_failed", e)
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


def find_device_by_keywords(devices, keywords):
    """Return the first device label whose name (lowercased) contains any keyword."""
    for _idx, label in devices:
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


# ─────────────────────────────────────────────────────────────────────────────
# VB-Audio loopback output thread
# Reads raw PCM from a ring-buffer and writes it to the chosen output device
# so the user hears the captured audio through their headphones (Fiio etc.).
# ─────────────────────────────────────────────────────────────────────────────

class LoopbackOutputThread(threading.Thread):
    def __init__(self, output_device_index, input_sample_rate, status_cb):
        super().__init__(daemon=True, name="loopback-output")
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
        self._stop_evt.set()
        self._buf.put(b"")

    def run(self):
        if not _PYAUDIO_AVAILABLE:
            return
        pa     = pyaudio.PyAudio()
        stream = None
        try:
            # Probe the actual rate and channel count the device accepts.
            # WDM-KS devices are excluded from enumeration, but if we still
            # land on a device that refuses to open, we catch below.
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

# Audio capture + VAD
# ─────────────────────────────────────────────────────────────────────────────

class AudioCaptureThread(threading.Thread):
    def __init__(self, device_index, phrase_queue, status_cb, level_cb,
                 loopback_out: "LoopbackOutputThread | None" = None):
        super().__init__(daemon=True, name="audio-capture")
        self.device_index      = device_index
        self.phrase_queue      = phrase_queue
        self.status_cb         = status_cb
        self.level_cb          = level_cb
        self.loopback_out      = loopback_out
        self._stop_evt         = threading.Event()
        self.energy_threshold  = 300
        self._sensitivity_mult = DEFAULT_SENS

    def set_sensitivity(self, mult: float):
        self._sensitivity_mult = mult

    def stop(self):
        self._stop_evt.set()

    def _flush(self, buf, label="Phrase", resample=None):
        min_c = int(MIN_PHRASE_SEC * SAMPLE_RATE / CHUNK_SAMPLES)
        if len(buf) >= min_c:
            raw     = b"".join(buf)
            payload = resample(raw) if resample else raw
            try:
                self.phrase_queue.put(payload, timeout=0.25)
                backlog = self.phrase_queue.qsize()
                log_event(logging.INFO, "phrase_flushed",
                          reason=label, bytes=len(payload),
                          chunks=len(buf), queue_backlog=backlog)
                if backlog >= PHRASE_QUEUE_BACKLOG_WARN:
                    log_event(logging.WARNING, "phrase_queue_backlog_high", backlog=backlog)
            except queue.Full:
                log_event(logging.WARNING, "phrase_dropped_queue_full",
                          reason=label, bytes=len(payload), chunks=len(buf))

    def run(self):
        pa     = pyaudio.PyAudio()
        stream = None
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
            return

        # ── Calibrate ~1 s ───────────────────────────────────────────────
        self.status_cb("Calibrating…")
        cal = []
        for _ in range(int(open_rate / chunk_sz)):
            if self._stop_evt.is_set():
                stream.close(); return
            try:
                data = stream.read(chunk_sz, exception_on_overflow=False)
                if in_channels > 1:
                    import array
                    arr = array.array('h', data)
                    data = arr[0::in_channels].tobytes()
                cal.append(rms(data))
            except Exception:
                pass
        if cal:
            ambient               = sum(cal) / len(cal)
            self.energy_threshold = max(100, ambient * CAL_MULTIPLIER)
            self.status_cb("Ready.  Ambient " + str(round(ambient)) +
                           "  threshold " + str(round(self.energy_threshold)))
            log_event(logging.INFO, "audio_calibrated",
                      ambient=round(ambient, 2),
                      threshold=round(self.energy_threshold, 2),
                      sensitivity=round(self._sensitivity_mult, 2),
                      open_rate=open_rate)
        else:
            self.status_cb("Ready (default threshold).")
            log_event(logging.WARNING, "audio_calibration_empty",
                      threshold=self.energy_threshold, open_rate=open_rate)

        sil_c    = int(SILENCE_SEC      * open_rate / chunk_sz)
        stream_c = int(STREAM_CHUNK_SEC * open_rate / chunk_sz)
        max_c    = int(MAX_PHRASE_SEC   * open_rate / chunk_sz)

        buf, sil_count, in_speech, chunk_timer = [], 0, False, 0

        # ── Resampler: captured at open_rate; Whisper needs SAMPLE_RATE (16 kHz)
        if open_rate != SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
                from fractions import Fraction
                import numpy as _np_rs
                _frac = Fraction(SAMPLE_RATE, open_rate).limit_denominator(100)
                _up, _down = _frac.numerator, _frac.denominator
                def _resample(raw: bytes) -> bytes:
                    arr = _np_rs.frombuffer(raw, dtype=_np_rs.int16).astype(_np_rs.float32)
                    out = resample_poly(arr, _up, _down)
                    return out.clip(-32768, 32767).astype(_np_rs.int16).tobytes()
                log_event(logging.INFO, "resampler_scipy",
                          open_rate=open_rate, whisper_rate=SAMPLE_RATE, up=_up, down=_down)
            except ImportError:
                import numpy as _np_rs
                _ratio = max(1, open_rate // SAMPLE_RATE)
                def _resample(raw: bytes) -> bytes:
                    arr = _np_rs.frombuffer(raw, dtype=_np_rs.int16)
                    return arr[::_ratio].tobytes()
                log_event(logging.WARNING, "resampler_decimation",
                          open_rate=open_rate, whisper_rate=SAMPLE_RATE, ratio=_ratio)
        else:
            def _resample(raw: bytes) -> bytes:
                return raw

        while not self._stop_evt.is_set():
            try:
                data = stream.read(chunk_sz, exception_on_overflow=False)
                if in_channels > 1:
                    import array
                    arr = array.array('h', data)
                    data = arr[0::in_channels].tobytes()
            except Exception as e:
                log_exception("audio_stream_read_failed", e)
                time.sleep(0.01)
                continue

            # ── Feed raw audio to loopback so headphones receive it ──────
            if self.loopback_out is not None:
                self.loopback_out.push(data)

            level     = rms(data)
            threshold = self.energy_threshold * self._sensitivity_mult
            self.level_cb(level, threshold)

            if level > threshold:
                if not in_speech:
                    in_speech   = True
                    chunk_timer = 0
                    self.status_cb("🎙 Speech…")
                buf.append(data)
                sil_count    = 0
                chunk_timer += 1
                if chunk_timer >= stream_c:
                    self._flush(buf, "mid", _resample)
                    buf, chunk_timer = [], 0
                elif len(buf) >= max_c:
                    self._flush(buf, "max", _resample)
                    buf, chunk_timer = [], 0
            elif in_speech:
                sil_count   += 1
                chunk_timer += 1
                buf.append(data)
                if sil_count >= sil_c:
                    self._flush(buf, "end", _resample)
                    buf, sil_count, chunk_timer, in_speech = [], 0, 0, False

        if buf:
            self._flush(buf, "final", _resample)
        try:
            if stream is not None:
                stream.close()
        finally:
            pa.terminate()
        log_event(logging.INFO, "audio_capture_stopped", open_rate=open_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Whisper model loader — GPU with automatic CPU fallback
# ─────────────────────────────────────────────────────────────────────────────

class WhisperModelLoader(threading.Thread):
    def __init__(self, model_size, status_cb):
        super().__init__(daemon=True, name="whisper-loader")
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
        from faster_whisper import WhisperModel
        label = "GPU" if device == "cuda" else "CPU"
        self.status_cb(f"Loading Whisper '{model_size}' on {label}…")
        t0    = time.perf_counter()
        
        # Prevent openmp thread explosion on CPU by limiting threads per worker
        kwargs = dict(device=device, compute_type=compute_type)
        if device == "cpu":
            import os
            cores = os.cpu_count() or 4
            # If we have 2 workers, give each worker half the cores
            kwargs["cpu_threads"] = max(1, cores // 2)
            
        model = WhisperModel(model_size, **kwargs)
        self.status_cb("Pre-warming model…")
        import numpy as np
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        list(model.transcribe(dummy, language="en", beam_size=1,
                               vad_filter=False, without_timestamps=True)[0])
        return model, round((time.perf_counter() - t0) * 1000, 2)

    def run(self):
        try:
            # ── Primary attempt ───────────────────────────────────────────
            try:
                self.model, ms = self._try_load(self.model_size, WHISPER_DEVICE, COMPUTE_TYPE)
                label = "GPU" if WHISPER_DEVICE == "cuda" else "CPU"
                self.status_cb(f"✔ Whisper '{self.model_size}' ready on {label}.")
                log_event(logging.INFO, "whisper_model_ready",
                          model_size=self.model_size, device=label,
                          compute_type=COMPUTE_TYPE, elapsed_ms=ms, fallback_used=False)
            except Exception as first_err:
                log_exception("whisper_model_primary_load_failed", first_err,
                              model_size=self.model_size, device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE)
                if WHISPER_DEVICE != "cuda":
                    raise first_err
                # ── CPU fallback ──────────────────────────────────────────
                fb_size = MODELS.get(DEFAULT_MODEL_CPU, "tiny")
                self.status_cb("GPU load failed — falling back to CPU (Tiny model)…")
                self.model, ms     = self._try_load(fb_size, "cpu", "int8")
                self.model_size    = fb_size
                self.active_device = "cpu"
                self.active_compute_type = "int8"
                self.worker_count  = 2
                self.default_model_key = DEFAULT_MODEL_CPU
                self.used_fallback = True
                self.status_cb(f"✔ Whisper '{fb_size}' ready on CPU (fallback).")
                log_event(logging.WARNING, "whisper_model_cpu_fallback_ready",
                          original_device=WHISPER_DEVICE, fallback_device="cpu",
                          active_model_size=fb_size, elapsed_ms=ms)
        except Exception as e:
            self.error = str(e)
            self.status_cb(f"ERROR loading model: {e}")
            log_exception("whisper_model_load_failed", e,
                          model_size=self.model_size, device=self.active_device,
                          compute_type=self.active_compute_type)
        finally:
            self.ready.set()


# ─────────────────────────────────────────────────────────────────────────────
# Transcription worker
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionWorker(threading.Thread):
    def __init__(self, wid, model, src_whisper, src_lang_key, tgt_lang_key,
                 phrase_queue, result_queue, status_cb):
        super().__init__(daemon=True, name=f"transcription-worker-{wid}")
        self.wid          = wid
        self.model        = model
        self.model_lock   = INFERENCE_LOCK
        self.src_whisper  = src_whisper
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key
        self.phrase_queue = phrase_queue
        self.result_queue = result_queue
        self.status_cb    = status_cb
        self._stop_evt    = threading.Event()

    def stop(self):
        self._stop_evt.set()
        self.phrase_queue.put(None)

    def run(self):
        import numpy as np
        while not self._stop_evt.is_set():
            try:
                audio_bytes = self.phrase_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if audio_bytes is None:
                break
            t0 = time.perf_counter()
            try:
                audio_np = (np.frombuffer(audio_bytes, dtype=np.int16)
                            .astype(np.float32) / 32768.0)
                lock_t0 = time.perf_counter()
                with self.model_lock:
                    lock_wait_ms = round((time.perf_counter() - lock_t0) * 1000, 2)
                    segments, _ = self.model.transcribe(
                        audio_np,
                        language=self.src_whisper,
                        task="transcribe",
                        beam_size=1, best_of=1, temperature=0.0,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                    )
                text       = " ".join(seg.text for seg in segments).strip()
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                log_event(logging.INFO, "transcription_completed",
                          worker_id=self.wid,
                          input_bytes=len(audio_bytes), text_chars=len(text),
                          elapsed_ms=elapsed_ms, lock_wait_ms=lock_wait_ms,
                          queue_backlog=self.phrase_queue.qsize())
                self.result_queue.put(("metric", ("transcription_ms", elapsed_ms)))
                self.result_queue.put(("metric", ("lock_wait_ms",     lock_wait_ms)))
                if text:
                    self.result_queue.put(("heard",     text))
                    self.result_queue.put(("translate", (text, self.src_lang_key, self.tgt_lang_key)))
            except Exception as e:
                self.result_queue.put(("status", f"Worker {self.wid}: {e}"))
                log_exception("transcription_failed", e,
                              worker_id=self.wid,
                              input_bytes=len(audio_bytes) if audio_bytes else 0)


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class InterpreterApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Voice Interpreter")
        self.geometry("980x800")
        self.minsize(820, 640)

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
        self._capture_thr         = None
        self._loopback_out        = None
        self._workers             = []
        self._translation_workers = []
        self._translation_queue   = queue.Queue(maxsize=TRANSLATION_QUEUE_MAX)
        self._model_loader        = None
        self._whisper_model       = None
        self.is_listening         = False
        self._active_mode         = None
        self._start_retry_after_id = None
        self._tts_lock            = threading.Lock()

        # ── TTS ───────────────────────────────────────────────────────────
        self.tts = None
        if _TTS_AVAILABLE:
            try:
                self.tts = pyttsx3.init()
                log_event(logging.INFO, "tts_initialized")
            except Exception as e:
                log_exception("tts_init_failed", e)

        # ── Audio devices ─────────────────────────────────────────────────
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map  = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        input_labels     = [lbl for _, lbl in self._input_devices]
        output_labels    = [lbl for _, lbl in self._output_devices]

        nvidia_lbl  = find_device_by_keywords(self._input_devices,  NVIDIA_KEYWORDS)
        vb_lbl      = find_best_vb_input(self._input_devices)       # prefers CABLE-A
        fiio_lbl    = find_best_output_device(self._output_devices)  # K7 > Q series etc.
        default_out = output_labels[0] if output_labels else ""

        self._mode_state = [
            [nvidia_lbl or (input_labels[0] if input_labels else ""), "English", "Russian"],
            [vb_lbl     or (input_labels[0] if input_labels else ""), "Russian", "English"],
        ]

        self.device_var     = ctk.StringVar(value=self._mode_state[0][0])
        self.out_device_var = ctk.StringVar(value=fiio_lbl or default_out)
        self.loopback_var   = ctk.BooleanVar(value=bool(vb_lbl))
        self.model_var      = ctk.StringVar(value=_DEFAULT_MODEL)
        self.sensitivity_var = ctk.DoubleVar(value=DEFAULT_SENS)
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
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        log_event(logging.INFO, "app_initialized",
                  session_id=self.session_id,
                  whisper_device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE,
                  num_workers=NUM_WORKERS,
                  input_device_count=len(input_labels),
                  output_device_count=len(output_labels))
        self._preload_model()
        self.after(80,   self._poll_queue)
        self.after(1000, self._refresh_diagnostics)

    # ── Model ─────────────────────────────────────────────────────────────

    def _preload_model(self):
        self._whisper_model = None
        model_size = MODELS[self.model_var.get()]
        log_event(logging.INFO, "whisper_model_preload_started", model_size=model_size)
        self._model_loader = WhisperModelLoader(
            model_size=model_size,
            status_cb=lambda m: self.result_queue.put(("status", m)),
        )
        self._model_loader.start()

    def _reload_model_if_changed(self):
        log_event(logging.INFO, "whisper_model_reload_requested", selected_model=self.model_var.get())
        if self.is_listening:
            self._stop_session()
        self._whisper_model = None
        self._preload_model()

    # ── Header ────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray85", "gray17"))
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(hdr, fg_color="transparent")
        title_row.grid(row=0, column=0, pady=(14, 2))

        ctk.CTkLabel(title_row, text="🎙  Voice Interpreter",
                     font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=0)

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

        ctk.CTkLabel(hdr,
                     text="Powered by Whisper AI  ·  Bidirectional real-time translation",
                     font=ctk.CTkFont(size=12),
                     text_color=("gray50", "gray60")).grid(row=1, column=0, pady=(0, 12))

    # ── Quick switch ──────────────────────────────────────────────────────

    def _build_quick_switch(self):
        qs = ctk.CTkFrame(self, fg_color=("gray88", "gray20"), corner_radius=10)
        qs.grid(row=1, column=0, sticky="ew", padx=20, pady=(14, 0))
        qs.grid_columnconfigure(0, weight=1)
        qs.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(qs, text="QUICK SWITCH",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=("gray50", "gray55")).grid(
            row=0, column=0, columnspan=2, pady=(8, 4))
        self._mode_btns = []
        for col, (label, emoji, src, tgt) in enumerate([
            ("My Voice", "🎤", "English", "Russian"),
            ("Friend",   "🔊", "Russian", "English"),
        ]):
            btn = ctk.CTkButton(
                qs, text=f"{emoji}  {label}\n{src} → {tgt}",
                font=ctk.CTkFont(size=14, weight="bold"),
                height=68, corner_radius=8,
                fg_color=COL_INACTIVE, hover_color=COL_IN_HOVER,
                command=lambda i=col: self._on_mode_btn(i))
            btn.grid(row=1, column=col, sticky="ew",
                     padx=(12, 6) if col == 0 else (6, 12), pady=(0, 12))
            self._mode_btns.append(btn)

    # ── Controls ──────────────────────────────────────────────────────────

    def _build_controls(self, input_labels, output_labels):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 0))
        outer.grid_columnconfigure(1, weight=1)

        # Audio input
        ctk.CTkLabel(outer, text="Audio input:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=0, column=0, sticky="w", pady=2)
        r0 = ctk.CTkFrame(outer, fg_color="transparent")
        r0.grid(row=0, column=1, sticky="w")
        self.device_menu = ctk.CTkOptionMenu(
            r0, variable=self.device_var, values=input_labels, width=320,
            command=self._on_manual_device_change)
        self.device_menu.grid(row=0, column=0)
        ctk.CTkButton(r0, text="⟳", width=34,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._refresh_devices).grid(row=0, column=1, padx=(8, 0))

        # Audio output + loopback toggle
        ctk.CTkLabel(outer, text="Audio output:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=1, column=0, sticky="w", pady=2)
        r1 = ctk.CTkFrame(outer, fg_color="transparent")
        r1.grid(row=1, column=1, sticky="w")
        self.out_device_menu = ctk.CTkOptionMenu(
            r1, variable=self.out_device_var, values=output_labels, width=260)
        self.out_device_menu.grid(row=0, column=0)
        self.loopback_chk = ctk.CTkCheckBox(
            r1, text="Hear input through output  (loopback)",
            variable=self.loopback_var, font=ctk.CTkFont(size=12))
        self.loopback_chk.grid(row=0, column=1, padx=(16, 0))

        # Translate
        ctk.CTkLabel(outer, text="Translate:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=2, column=0, sticky="w", pady=2)
        r2 = ctk.CTkFrame(outer, fg_color="transparent")
        r2.grid(row=2, column=1, sticky="w")
        lang_names = sorted(LANGUAGES.keys())
        self.src_menu = ctk.CTkOptionMenu(
            r2, variable=self.src_var, values=lang_names, width=155,
            command=self._on_manual_lang_change)
        self.src_menu.grid(row=0, column=0)
        ctk.CTkLabel(r2, text="  →  ",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=1)
        self.tgt_menu = ctk.CTkOptionMenu(
            r2, variable=self.tgt_var, values=lang_names, width=155,
            command=self._on_manual_lang_change)
        self.tgt_menu.grid(row=0, column=2)

        # Model + sensitivity
        ctk.CTkLabel(outer, text="Model:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=3, column=0, sticky="w", pady=(6, 2))
        r3 = ctk.CTkFrame(outer, fg_color="transparent")
        r3.grid(row=3, column=1, sticky="ew", pady=(6, 2))
        self.model_menu = ctk.CTkOptionMenu(
            r3, variable=self.model_var, values=list(MODELS.keys()), width=230,
            command=lambda _: self._reload_model_if_changed())
        self.model_menu.grid(row=0, column=0, padx=(0, 20))
        ctk.CTkLabel(r3, text="Sensitivity:",
                     font=ctk.CTkFont(size=13)).grid(row=0, column=1)
        self.sens_slider = ctk.CTkSlider(
            r3, from_=0.1, to=1.5, variable=self.sensitivity_var,
            width=160, command=self._on_sensitivity)
        self.sens_slider.grid(row=0, column=2, padx=(8, 4))
        self.sens_label = ctk.CTkLabel(r3, text=f"{DEFAULT_SENS}×",
                                       font=ctk.CTkFont(size=12), width=36)
        self.sens_label.grid(row=0, column=3)

        # Action buttons
        btns = ctk.CTkFrame(outer, fg_color="transparent")
        btns.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        self.start_btn = ctk.CTkButton(
            btns, text="▶  Start Listening", width=165,
            command=self._toggle_listening)
        self.start_btn.grid(row=0, column=0, padx=(0, 8))
        self.tts_btn = ctk.CTkButton(
            btns, text="🔊  Read Aloud", width=130,
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
            text_color=("gray10", "gray90"),
            command=self._speak_translation)
        self.tts_btn.grid(row=0, column=1, padx=(0, 8))
        ctk.CTkButton(btns, text="Clear", width=70,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._clear).grid(row=0, column=2)

    # ── Input-level meter ─────────────────────────────────────────────────

    def _build_meter(self):
        mf = ctk.CTkFrame(self, fg_color="transparent")
        mf.grid(row=3, column=0, sticky="ew", padx=20, pady=(8, 0))
        mf.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(mf, text="Input level:",
                     font=ctk.CTkFont(size=11),
                     text_color=("gray50", "gray55"),
                     width=80, anchor="w").grid(row=0, column=0, sticky="w")
        self.level_bar = ctk.CTkProgressBar(mf, height=14, corner_radius=4)
        self.level_bar.set(0)
        self.level_bar.grid(row=0, column=1, sticky="ew", padx=(4, 8))
        self.level_text = ctk.CTkLabel(mf, text="—",
                                       font=ctk.CTkFont(size=10),
                                       text_color=("gray50", "gray55"),
                                       width=140, anchor="w")
        self.level_text.grid(row=0, column=2)

    # ── Main tabbed area ──────────────────────────────────────────────────

    def _build_text_area(self):
        tabs = ctk.CTkTabview(self, fg_color="transparent")
        tabs.grid(row=4, column=0, sticky="nsew", padx=20, pady=(10, 0))
        tabs.add("Interpreter")
        tabs.add("Diagnostics")
        self.main_tabs = tabs

        # ─── Interpreter tab ──────────────────────────────────────────────
        itab = tabs.tab("Interpreter")
        itab.grid_columnconfigure(0, weight=1)
        itab.grid_columnconfigure(1, weight=1)
        itab.grid_rowconfigure(1, weight=1)

        self.heard_label = ctk.CTkLabel(itab, text="Heard in English",
                                        font=ctk.CTkFont(size=12, weight="bold"))
        self.heard_label.grid(row=0, column=0, sticky="w", padx=4, pady=(4, 4))
        self.translated_label = ctk.CTkLabel(itab, text="Translation → Russian",
                                             font=ctk.CTkFont(size=12, weight="bold"))
        self.translated_label.grid(row=0, column=1, sticky="w", padx=(12, 4), pady=(4, 4))
        self.heard_box = ctk.CTkTextbox(itab, wrap="word", font=ctk.CTkFont(size=14))
        self.heard_box.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.translated_box = ctk.CTkTextbox(itab, wrap="word", font=ctk.CTkFont(size=14))
        self.translated_box.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        # ─── Diagnostics tab ──────────────────────────────────────────────
        dtab = tabs.tab("Diagnostics")
        dtab.grid_columnconfigure(0, weight=1)
        dtab.grid_rowconfigure(2, weight=1)   # textbox row gets all spare space

        # Health summary (fixed-height, multi-line label replacement)
        self.diag_summary_box = ctk.CTkTextbox(
            dtab, height=90, wrap="word",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("gray92", "gray18"), activate_scrollbars=False)
        self.diag_summary_box.grid(row=0, column=0, sticky="ew", padx=2, pady=(4, 6))
        self.diag_summary_box.configure(state="disabled")

        # Shortcut buttons
        btn_frame = ctk.CTkFrame(dtab, fg_color="transparent")
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
        self.diag_box = ctk.CTkTextbox(dtab, wrap="word", font=ctk.CTkFont(size=12))
        self.diag_box.grid(row=2, column=0, sticky="nsew")

    # ── Status bar ────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=5, column=0, sticky="ew", padx=22, pady=(4, 10))
        bar.grid_columnconfigure(0, weight=1)
        self.status_var = ctk.StringVar(value="Loading Whisper model…")
        ctk.CTkLabel(bar, textvariable=self.status_var,
                     font=ctk.CTkFont(size=11),
                     text_color=("gray45", "gray55"), anchor="w").grid(
            row=0, column=0, sticky="ew")
        ctk.CTkButton(bar, text="Export Log Summary", width=150,
                      fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                      text_color=("gray10", "gray90"),
                      command=self._export_log_summary).grid(row=0, column=1, padx=(10, 0))

    # ── Level meter ───────────────────────────────────────────────────────

    def _on_level(self, level, threshold):
        self.result_queue.put(("level", (level, threshold)))

    def _update_meter(self, level, threshold):
        ceiling  = max(threshold * 4, 800)
        fraction = min(level / ceiling, 1.0)
        self.level_bar.set(fraction)
        active = level > threshold
        self.level_bar.configure(progress_color=COL_OK if active else COL_INFO)
        pct = int(fraction * 100)
        tag = "▐ SPEECH ▌" if active else " quiet  "
        self.level_text.configure(text=f"{pct:3d}%  [{tag}]")

    # ── Mode switching ────────────────────────────────────────────────────

    def _on_mode_btn(self, mode_idx):
        if self._active_mode is not None:
            self._mode_state[self._active_mode] = [
                self.device_var.get(), self.src_var.get(), self.tgt_var.get()]
        self._apply_mode(mode_idx, switch_source=True)

    def _apply_mode(self, mode_idx, switch_source=False):
        was = self.is_listening
        if was:
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
        if was:
            self._start_session()

    def _on_manual_device_change(self, _=None):
        if self.is_listening:
            self._stop_session()

    def _on_manual_lang_change(self, _=None):
        self.heard_label.configure(text=f"Heard in {self.src_var.get()}")
        self.translated_label.configure(text=f"Translation → {self.tgt_var.get()}")
        if self.is_listening:
            self._stop_session()

    def _refresh_devices(self):
        if self.is_listening:
            self._stop_session()
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
        vb_lbl     = find_device_by_keywords(self._input_devices, VB_KEYWORDS)
        if nvidia_lbl: self._mode_state[0][0] = nvidia_lbl
        if vb_lbl:     self._mode_state[1][0] = vb_lbl
        self.status_var.set(f"Refreshed — {len(in_labels)} in / {len(out_labels)} out device(s).")
        log_event(logging.INFO, "audio_devices_refreshed",
                  input_count=len(in_labels), output_count=len(out_labels))
        self._refresh_diagnostics_once()

    def _set_controls_state(self, state):
        for w in (self.device_menu, self.out_device_menu, self.loopback_chk,
                  self.src_menu, self.tgt_menu, self.model_menu, self.sens_slider):
            w.configure(state=state)

    # ── Session start / stop ──────────────────────────────────────────────

    def _toggle_listening(self):
        if self.is_listening:
            self._stop_session()
        else:
            self._start_session()

    def _start_session(self):
        if self.is_listening:
            return
        if self._model_loader and not self._model_loader.ready.is_set():
            self.status_var.set("⏳ Loading Whisper model — please wait…")
            if self._start_retry_after_id is None:
                self._start_retry_after_id = self.after(600, self._retry_start_session)
            return
        self._start_retry_after_id = None
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
        self.start_btn.configure(text="⏹  Stop Listening",
                                 fg_color=COL_STOP, hover_color=COL_STOP_HOV)
        self._set_live_status("LISTENING", COL_OK)
        self._set_controls_state("disabled")

        in_idx = self._input_map.get(self.device_var.get(), -1)
        if in_idx == -1:
            in_idx = None
        log_event(logging.INFO, "audio_device_selected",
                  device_label=self.device_var.get(), device_index=in_idx)

        src_key     = self.src_var.get()
        tgt_key     = self.tgt_var.get()
        src_whisper, _ = LANGUAGES.get(src_key, ("en", "en"))

        worker_count = getattr(self._model_loader, "worker_count", NUM_WORKERS)
        self.phrase_queue = queue.Queue(maxsize=max(worker_count * 4, 8))

        # ── Optional loopback output (VB-Audio → Fiio headphones) ────────
        # Look up the native sample rate of the input device so the loopback
        # output thread opens at the same rate and PortAudio accepts it.
        _pa_tmp = pyaudio.PyAudio() if _PYAUDIO_AVAILABLE else None
        in_native_rate = get_device_native_rate(_pa_tmp, in_idx) if _pa_tmp else SAMPLE_RATE
        if _pa_tmp:
            _pa_tmp.terminate()
        self._loopback_out = None
        if self.loopback_var.get():
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

        self._capture_thr = AudioCaptureThread(
            device_index=in_idx,
            phrase_queue=self.phrase_queue,
            status_cb=lambda m: self.result_queue.put(("status", m)),
            level_cb=self._on_level,
            loopback_out=self._loopback_out,
        )
        self._capture_thr.set_sensitivity(round(self.sensitivity_var.get(), 2))
        self._capture_thr.start()

        # ── Translation workers (TRANSLATION_WORKERS parallel) ────────────
        self._translation_workers = []
        for wid in range(TRANSLATION_WORKERS):
            tw = TranslationWorker(wid, self._translation_queue, self.result_queue)
            tw.start()
            self._translation_workers.append(tw)

        # ── Transcription workers ─────────────────────────────────────────
        self._workers = []
        for wid in range(worker_count):
            w = TranscriptionWorker(
                wid=wid, model=self._whisper_model,
                src_whisper=src_whisper,
                src_lang_key=src_key, tgt_lang_key=tgt_key,
                phrase_queue=self.phrase_queue,
                result_queue=self.result_queue,
                status_cb=lambda m: self.result_queue.put(("status", m)))
            w.start()
            self._workers.append(w)

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
        self._start_retry_after_id = None
        self._start_session()

    def _stop_session(self, keep_mode=False):
        self.is_listening = False
        if self._start_retry_after_id is not None:
            self.after_cancel(self._start_retry_after_id)
            self._start_retry_after_id = None
        if self._capture_thr:
            self._capture_thr.stop()
            self._capture_thr = None
        if self._loopback_out:
            self._loopback_out.stop()
            self._loopback_out = None
        for w in self._workers:
            w.stop()
        self._workers = []
        for tw in self._translation_workers:
            tw.stop()
        self._translation_workers = []
        self.start_btn.configure(text="▶  Start Listening",
                                 fg_color=["#3B8ED0", "#1F6AA5"],
                                 hover_color=["#3281BC", "#1A5C8F"])
        self._set_controls_state("normal")
        self.level_bar.set(0)
        self.level_text.configure(text="—")
        self._set_live_status("IDLE", COL_WARN)
        log_event(logging.INFO, "session_stopped",
                  session_id=self.session_id, keep_mode=keep_mode)
        if not keep_mode:
            self.status_var.set("Stopped.")

    # ── TTS — fixed: lock prevents "run loop already started" ────────────

    def _speak_translation(self):
        text = self.translated_box.get("1.0", "end-1c").strip()
        if not text:
            self.status_var.set("Nothing to read aloud.")
            return
        if not self.tts:
            self.status_var.set("TTS engine not available.")
            log_event(logging.WARNING, "tts_unavailable")
            return
        if not self._tts_lock.acquire(blocking=False):
            self.status_var.set("Read Aloud already running — please wait.")
            return
        self.tts_btn.configure(state="disabled")

        def _run():
            try:
                self.tts.say(text)
                self.tts.runAndWait()
                log_event(logging.INFO, "tts_completed", text_chars=len(text))
            except RuntimeError:
                # Engine was busy; reinitialise for next call
                try:
                    self.tts = pyttsx3.init()
                    log_event(logging.WARNING, "tts_engine_reinitialised")
                except Exception as e2:
                    log_exception("tts_reinit_failed", e2)
            except Exception as e:
                log_exception("tts_run_failed", e, text_chars=len(text))
            finally:
                self._tts_lock.release()
                self.after(0, lambda: self.tts_btn.configure(state="normal"))

        threading.Thread(target=_run, daemon=True, name="tts-worker").start()

    # ── Clear ─────────────────────────────────────────────────────────────

    def _clear(self):
        if self.is_listening:
            self._stop_session()
        self.heard_box.delete("1.0", "end")
        self.translated_box.delete("1.0", "end")
        self.status_var.set("Cleared.")
        log_event(logging.INFO, "ui_cleared")

    # ── Sensitivity ───────────────────────────────────────────────────────

    def _on_sensitivity(self, val):
        v = round(float(val), 2)
        self.sens_label.configure(text=f"{v}×")
        if self._capture_thr:
            self._capture_thr.set_sensitivity(v)
        log_event(logging.INFO, "sensitivity_changed", value=v)

    # ── Queue poll ────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                kind, data = self.result_queue.get_nowait()
                if kind == "heard":
                    self._append(self.heard_box, data)
                    self._trim_textbox(self.heard_box)
                elif kind == "translate":
                    text, sk, tk = data
                    try:
                        self._translation_queue.put((text, sk, tk), timeout=0.1)
                        backlog = self._translation_queue.qsize()
                        log_event(logging.INFO, "translation_queued",
                                  input_chars=len(text), src_lang=sk, tgt_lang=tk,
                                  queue_backlog=backlog)
                        if backlog >= TRANSLATION_QUEUE_BACKLOG_WARN:
                            log_event(logging.WARNING, "translation_queue_backlog_high",
                                      backlog=backlog)
                    except queue.Full:
                        log_event(logging.WARNING, "translation_dropped_queue_full",
                                  input_chars=len(text), src_lang=sk, tgt_lang=tk)
                elif kind == "translated":
                    self._append(self.translated_box, data)
                    self._trim_textbox(self.translated_box)
                    self.status_var.set("Listening…" if self.is_listening else "Done.")
                elif kind == "status":
                    self.status_var.set(data)
                    log_event(logging.INFO, "status_updated", status=data)
                elif kind == "level":
                    self._update_meter(*data)
                elif kind == "metric":
                    self._update_metric(*data)
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _update_metric(self, name, value):
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
        self._refresh_diagnostics_once()
        self.after(1000, self._refresh_diagnostics)

    def _refresh_diagnostics_once(self):
        avg_tr  = self._avg(self._perf_stats["transcription_ms"])
        avg_tl  = self._avg(self._perf_stats["translation_ms"])
        avg_lk  = self._avg(self._perf_stats["lock_wait_ms"])
        pbl     = self.phrase_queue.qsize()
        tbl     = self._translation_queue.qsize()

        if pbl >= PHRASE_QUEUE_BACKLOG_WARN or tbl >= TRANSLATION_QUEUE_BACKLOG_WARN:
            color, health = COL_ERR,  "⚠ Backlog high — translation lagging"
        elif self.is_listening:
            color, health = COL_OK,   "✔ Healthy — listening"
        else:
            color, health = COL_WARN, "● Idle"

        summary = (
            f"{health}\n"
            f"Transcriptions: {self._perf_stats['transcriptions']}   "
            f"Avg transcribe: {avg_tr:.0f} ms   "
            f"Avg translate: {avg_tl:.0f} ms   "
            f"Avg lock wait: {avg_lk:.0f} ms\n"
            f"Phrase Q: {pbl}   Translation Q: {tbl}   "
            f"TW workers: {len(self._translation_workers)}"
        )
        self.diag_summary_box.configure(state="normal", text_color=color)
        self.diag_summary_box.delete("1.0", "end")
        self.diag_summary_box.insert("1.0", summary)
        self.diag_summary_box.configure(state="disabled")

        detail_lines = [
            f"session_id            {self.session_id}",
            f"input device          {self.device_var.get()}",
            f"output device         {self.out_device_var.get()}",
            f"loopback enabled      {self.loopback_var.get()}",
            f"model                 {self.model_var.get()}",
            f"source language       {self.src_var.get()}",
            f"target language       {self.tgt_var.get()}",
            f"listening             {self.is_listening}",
            f"transcription workers {len(self._workers)}",
            f"translation workers   {len(self._translation_workers)}",
            f"whisper device        {WHISPER_DEVICE}",
            f"compute type          {COMPUTE_TYPE}",
            f"phrase queue backlog  {pbl}",
            f"translation Q backlog {tbl}",
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
        self.live_status_badge.configure(text=f" {text} ", fg_color=color)

    def _open_path(self, path: Path):
        try:
            target = path if path.exists() else APP_DIR
            os.startfile(str(target))
            log_event(logging.INFO, "open_path_requested", path=str(target))
        except Exception as e:
            self.status_var.set(f"Open failed: {e}")
            log_exception("open_path_failed", e, path=str(path))

    def _open_app_folder(self):
        self._open_path(APP_DIR)

    def _export_log_summary(self):
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
        return (sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _trim_textbox(box):
        text = box.get("1.0", "end-1c")
        if len(text) > MAX_TEXT_HISTORY_CHARS:
            box.delete("1.0", f"1.0 + {len(text) - MAX_TEXT_HISTORY_CHARS} chars")

    @staticmethod
    def _append(box, text):
        cur = box.get("1.0", "end-1c")
        box.insert("end", ("\n" if cur else "") + text)
        box.see("end")

    def _on_close(self):
        try:
            if self.is_listening:
                self._stop_session()
            log_event(logging.INFO, "app_closing", session_id=self.session_id)
        finally:
            self.destroy()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_event(logging.INFO, "app_main_start")
    app = InterpreterApp()
    app.mainloop()
    log_event(logging.INFO, "app_main_exit")

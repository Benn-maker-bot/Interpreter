"""
Voice Interpreter — optimised for minimum latency.

Speed stack:
  • Auto-detects NVIDIA GPU (CUDA DLLs from the nvidia-* pip packages);
    falls back to CPU int8 if they are missing.
  • Speech is cut at natural pauses (relative to the speaker's level) with a
    5 s cap at the quietest moment, and a live preview is transcribed every
    0.7 s while someone is still talking.
  • To-English translation runs in Whisper itself (task=translate) on the
    GPU; other targets use online services over a keep-alive session.
  • One inference worker using all physical cores (inference is serialised).
  • WASAPI loopback: capture what an output device (e.g. FiiO K7) is playing.
  • Text-to-speech via SAPI on its own thread: toggled on/off, stops
    instantly, and mutes capture while speaking on the captured device.
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


def _add_cuda_dll_dirs():
    """Make the pip-installed NVIDIA CUDA DLLs (nvidia-cublas-cu12,
    nvidia-cudnn-cu12) loadable by ctranslate2 on Windows."""
    if os.name != "nt":
        return []
    roots = [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(getattr(sys, "_MEIPASS", APP_DIR)) / "nvidia")
    added = []
    for root in roots:
        for bin_dir in root.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(bin_dir))
    return added


# Device labels contain emoji; a cp1252 console would raise on every log line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paths / log config ────────────────────────────────────────────────────────
APP_DIR               = Path(__file__).resolve().parent
LOG_PATH              = APP_DIR / "interpreter.log.jsonl"
ERROR_LOG_PATH        = APP_DIR / "interpreter.error.log.jsonl"
LOG_SUMMARY_PATH      = APP_DIR / "interpreter.log.summary.json"
SETTINGS_PATH         = APP_DIR / "interpreter.settings.json"
MAX_LOG_BYTES         = 5 * 1024 * 1024
LOG_BACKUP_COUNT      = 5

# ── Tuning ────────────────────────────────────────────────────────────────────
MAX_TEXT_HISTORY_CHARS         = 12_000
PHRASE_QUEUE_BACKLOG_WARN      = 6
TRANSLATION_QUEUE_BACKLOG_WARN = 16
TRANSLATION_WORKERS            = 1          # >1 lets results arrive out of order
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
SILENCE_SEC      = 0.4     # pause that ends a phrase
MAX_PHRASE_SEC   = 4       # longest phrase before a forced cut...
CUT_SEARCH_SEC   = 1.5     # ...made at the quietest moment in this final window
PAUSE_RATIO      = 0.25    # mid-speech, "quiet" means below this fraction of the
                           # speaker's level (call audio rarely hits true silence)
PARTIAL_INTERVAL_SEC = 0.35 # live preview cadence; the worker skips stale ones,
                            # so this self-limits to what the GPU can keep up with
PARTIAL_MIN_SEC  = 0.8     # less audio than this makes Whisper guess wildly
MIN_PHRASE_SEC   = 0.5     # shorter clips are mostly silence and make Whisper hallucinate
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
        dll_dirs = _add_cuda_dll_dirs()
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        # On Windows, just because get_cuda_device_count() > 0 doesn't mean
        # the cuBLAS/cuDNN libraries are loadable. If they are missing,
        # CTranslate2 throws a RuntimeError at transcribe() time, which can deadlock.
        # We explicitly verify we can load the DLLs before trusting 'cuda'.
        if n > 0 and os.name == "nt":
            import ctypes
            for dll in ("cublas64_12.dll", "cudnn64_9.dll"):
                try:
                    ctypes.cdll.LoadLibrary(dll)
                except Exception:
                    log_event(logging.WARNING, "cuda_detection_dll_missing",
                              dll=dll, searched=dll_dirs)
                    n = 0
                    break

        log_event(logging.INFO, "cuda_detection_result", cuda_device_count=n,
                  ctranslate2_version=ctranslate2.__version__)
        if n > 0:
            return "cuda", "float16", 1, DEFAULT_MODEL_GPU
    except Exception as e:
        log_exception("cuda_detection_failed", e)
    return "cpu", "int8", 1, DEFAULT_MODEL_CPU


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


class RateLimitedError(Exception):
    pass


def _google_translate(text: str, src: str, tgt: str) -> str:
    url    = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
    resp   = _http.get(url, params=params, timeout=6)
    if resp.status_code == 429:
        raise RateLimitedError("Google returned 429 Too Many Requests")
    resp.raise_for_status()
    data = resp.json()
    return "".join(p[0] for p in data[0] if p[0]).strip()


_MYMEMORY_CODES: dict | None = None

def _mymemory_code(code: str) -> str:
    # MyMemory wants region codes ("ru-RU"), not bare ISO codes ("ru").
    global _MYMEMORY_CODES
    if _MYMEMORY_CODES is None:
        from deep_translator import MyMemoryTranslator
        _MYMEMORY_CODES = MyMemoryTranslator(source="en-GB", target="ru-RU") \
            .get_supported_languages(as_dict=True)
    codes = list(_MYMEMORY_CODES.values())
    if code in codes:
        return code
    for c in codes:
        if c.split("-")[0].lower() == code.split("-")[0].lower():
            return c
    raise ValueError(f"MyMemory does not support language '{code}'")


def _mymemory_translate(text: str, src: str, tgt: str) -> str:
    # Called directly over the shared keep-alive session: deep-translator
    # opens a fresh TLS connection per call, adding ~200 ms each time.
    resp = _http.get("https://api.mymemory.translated.net/get",
                     params={"q": text,
                             "langpair": f"{_mymemory_code(src)}|{_mymemory_code(tgt)}"},
                     timeout=6)
    if resp.status_code == 429:
        raise RateLimitedError("MyMemory returned 429 Too Many Requests")
    resp.raise_for_status()
    data = resp.json()
    if data.get("responseStatus") == 429 or "QUOTA" in str(data.get("responseDetails", "")).upper():
        raise RateLimitedError(str(data.get("responseDetails")))
    if data.get("responseStatus") not in (200, "200"):
        raise RuntimeError(f"MyMemory error: {data.get('responseDetails')}")
    return (data["responseData"]["translatedText"] or "").strip()


# Providers are tried in order. A provider that rate-limits us is skipped for
# an increasing cooldown so we stop hammering it (which prolongs the block).
_PROVIDERS = [("google", _google_translate), ("mymemory", _mymemory_translate)]
_PROVIDER_COOLDOWN_BASE = 30.0
_PROVIDER_COOLDOWN_MAX  = 600.0
_provider_state = {name: {"until": 0.0, "cooldown": 0.0} for name, _ in _PROVIDERS}
_provider_lock  = threading.Lock()


@lru_cache(maxsize=512)
def _translate_cached(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    errors = []
    for name, fn in _PROVIDERS:
        with _provider_lock:
            state = _provider_state[name]
            wait  = state["until"] - time.monotonic()
        if wait > 0:
            errors.append(f"{name}: cooling down ({wait:.0f}s)")
            continue
        try:
            t0         = time.perf_counter()
            translated = fn(text, src, tgt)
            with _provider_lock:
                state["cooldown"] = 0.0
            log_event(logging.INFO, "translation_completed",
                      provider=name, src=src, tgt=tgt,
                      input_chars=len(text), output_chars=len(translated),
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
            return translated
        except RateLimitedError as e:
            with _provider_lock:
                state["cooldown"] = min(_PROVIDER_COOLDOWN_MAX,
                                        max(_PROVIDER_COOLDOWN_BASE, state["cooldown"] * 2))
                state["until"]    = time.monotonic() + state["cooldown"]
            log_event(logging.WARNING, "translation_rate_limited",
                      provider=name, cooldown_s=state["cooldown"], error=str(e))
            errors.append(f"{name}: rate limited")
        except Exception as e:
            log_exception("translation_provider_failed", e,
                          provider=name, src=src, tgt=tgt, input_chars=len(text))
            errors.append(f"{name}: {type(e).__name__}")
    log_event(logging.ERROR, "translation_failed",
              src=src, tgt=tgt, input_chars=len(text), errors=errors)
    raise RuntimeError("; ".join(errors))


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
            seq, text, src_lang_key, tgt_lang_key = item
            _, src = LANGUAGES.get(src_lang_key, ("en", "en"))
            _, tgt = LANGUAGES.get(tgt_lang_key, ("ru", "ru"))
            try:
                t0  = time.perf_counter()
                out = _translate_cached(text, src, tgt)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                self.result_queue.put(("translated", (seq, out or text)))
                self.result_queue.put(("metric", ("translation_ms", elapsed_ms)))
            except Exception as e:
                # Show the failure in the translation box, not just the status
                # bar, which "Listening…" overwrites almost immediately.
                self.result_queue.put(("translated", (seq, f"[translation failed: {text}]")))
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
    pa = None
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
                    prefix = "\U0001f50a " if info.get("isLoopbackDevice")           else \
                             "\U0001f3a4 " if any(k in nl for k in NVIDIA_KEYWORDS) else \
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
    finally:
        # A leaked PortAudio instance breaks WASAPI streams opened later
        # from other threads ("Unanticipated host error").
        if pa is not None:
            pa.terminate()
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


def find_best_friend_input(input_devices):
    """Default input for Friend mode: capture what the FiiO is playing (the
    call audio from Teams/Telemost) via WASAPI loopback, else a VB-Audio cable."""
    loopbacks = [(i, l) for i, l in input_devices if "[loopback]" in l.lower()]
    for priority_kw in FIIO_PRIORITY_KEYWORDS:
        for _idx, label in loopbacks:
            if priority_kw in label.lower():
                return label
    return find_best_vb_input(input_devices)


def find_device_by_keywords(devices, keywords):
    """Return the first device label whose name (lowercased) contains any keyword.
    Loopback (output-capture) devices are skipped: "nvidia" would otherwise
    match "NVIDIA High Definition Audio [Loopback]" (a monitor's speakers)."""
    for _idx, label in devices:
        if "[loopback]" in label.lower():
            continue
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


def open_loopback_keepalive(pa, loopback_info, attempts=3):
    for attempt in range(attempts):
        stream = _open_loopback_keepalive(pa, loopback_info, log_failure=attempt == attempts - 1)
        if stream is not None:
            return stream
        time.sleep(0.2)
    return None


def _open_loopback_keepalive(pa, loopback_info, log_failure=True):
    """WASAPI loopback delivers no data while nothing is playing, so a blocking
    read() would hang. Playing silence to the same render device keeps the
    loopback stream ticking. Returns the stream (caller closes it) or None."""
    try:
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        name   = loopback_info["name"].replace(" [Loopback]", "")
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if (d["hostApi"] == wasapi and not d.get("isLoopbackDevice")
                    and d["maxOutputChannels"] > 0 and d["name"] == name):
                ch = min(2, int(d["maxOutputChannels"]))
                silence_cb = lambda _in, frames, _t, _s: (b"\x00" * frames * 2 * ch,
                                                          pyaudio.paContinue)
                stream = pa.open(format=pyaudio.paInt16, channels=ch,
                                 rate=int(d["defaultSampleRate"]), output=True,
                                 output_device_index=i, stream_callback=silence_cb)
                log_event(logging.INFO, "loopback_keepalive_opened",
                          render_device=name, render_index=i)
                return stream
        log_event(logging.WARNING, "loopback_keepalive_device_not_found", name=name)
    except Exception as e:
        if log_failure:
            log_exception("loopback_keepalive_failed", e)
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

class LatestSlot:
    """Single-item mailbox: put() overwrites, take() empties. Live previews
    only matter if current, so a stale one is replaced rather than queued."""
    def __init__(self):
        self._lock = threading.Lock()
        self._item = None

    def put(self, item):
        with self._lock:
            self._item = item

    def take(self):
        with self._lock:
            item, self._item = self._item, None
            return item


class AudioCaptureThread(threading.Thread):
    def __init__(self, device_index, phrase_queue, status_cb, level_cb,
                 loopback_out: "LoopbackOutputThread | None" = None,
                 partial_slot: "LatestSlot | None" = None,
                 mute_evt: "threading.Event | None" = None):
        super().__init__(daemon=True, name="audio-capture")
        self.device_index      = device_index
        self._keepalive        = None
        self.is_loopback       = False
        self.partial_slot      = partial_slot
        self.mute_evt          = mute_evt   # set while our own TTS is audible on this device
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

    def _flush(self, seq, buf, min_c, label="Phrase", resample=None):
        # Always emit, even an empty payload, so the UI can clear the live
        # preview line for this phrase.
        payload = b""
        if len(buf) >= min_c:
            raw     = b"".join(buf)
            payload = resample(raw) if resample else raw
        try:
            self.phrase_queue.put((seq, payload), timeout=0.25)
            backlog = self.phrase_queue.qsize()
            log_event(logging.INFO, "phrase_flushed",
                      reason=label, seq=seq, bytes=len(payload),
                      chunks=len(buf), queue_backlog=backlog)
            if backlog >= PHRASE_QUEUE_BACKLOG_WARN:
                log_event(logging.WARNING, "phrase_queue_backlog_high", backlog=backlog)
        except queue.Full:
            log_event(logging.WARNING, "phrase_dropped_queue_full",
                      reason=label, bytes=len(payload), chunks=len(buf))

    def run(self):
        pa     = pyaudio.PyAudio()
        try:
            self._run(pa)
        finally:
            if self._keepalive is not None:
                try:
                    self._keepalive.close()
                except Exception:
                    pass
            pa.terminate()

    def _run(self, pa):
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
                    data = downmix(data, in_channels)
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

        per_sec   = open_rate / chunk_sz
        sil_c     = max(1, int(SILENCE_SEC * per_sec))
        max_c     = int(MAX_PHRASE_SEC * per_sec)
        cut_win   = int(CUT_SEARCH_SEC * per_sec)
        partial_c = max(1, int(PARTIAL_INTERVAL_SEC * per_sec))
        min_c     = max(1, int(MIN_PHRASE_SEC * per_sec))
        pmin_c    = max(1, int(PARTIAL_MIN_SEC * per_sec))

        buf, levels  = [], []
        sil_count    = 0
        in_speech    = False
        since_prev   = 0
        speech_level = 0.0
        seq          = 0
        silence      = bytes(chunk_sz * SAMPLE_WIDTH)

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
                    data = downmix(data, in_channels)
            except Exception as e:
                log_exception("audio_stream_read_failed", e)
                time.sleep(0.01)
                continue

            # Our own text-to-speech playing on the captured device must not
            # be transcribed as the friend speaking.
            if self.mute_evt is not None and self.mute_evt.is_set():
                data = silence[:len(data)]

            # ── Feed raw audio to loopback so headphones receive it ──────
            if self.loopback_out is not None:
                self.loopback_out.push(data)

            level     = rms(data)
            threshold = self.energy_threshold * self._sensitivity_mult
            self.level_cb(level, threshold)

            if in_speech:
                loud = level > max(threshold, PAUSE_RATIO * speech_level)
            else:
                loud = level > threshold
            if loud:
                if not in_speech:
                    in_speech, speech_level = True, level
                    self.status_cb("🎙 Speech…")
                speech_level = 0.95 * speech_level + 0.05 * level
                sil_count = 0
            elif in_speech:
                sil_count += 1
            if not in_speech:
                continue

            buf.append(data)
            levels.append(level)
            since_prev += 1
            if sil_count >= sil_c:
                self._flush(seq, buf, min_c, "end", _resample)
                seq += 1
                buf, levels, sil_count, since_prev, in_speech = [], [], 0, 0, False
            elif len(buf) >= max_c:
                # Cut at the quietest chunk (likely a word gap) near the end,
                # carrying the remainder into the next phrase.
                start = len(levels) - cut_win
                cut   = start + min(range(cut_win), key=lambda i: levels[start + i])
                self._flush(seq, buf[:cut + 1], min_c, "max", _resample)
                seq += 1
                buf, levels = buf[cut + 1:], levels[cut + 1:]
                since_prev  = len(buf)
            elif (since_prev >= partial_c and len(buf) >= pmin_c
                    and self.partial_slot is not None):
                self.partial_slot.put((seq, _resample(b"".join(buf))))
                since_prev = 0

        if buf:
            self._flush(seq, buf, min_c, "final", _resample)
        if stream is not None:
            stream.close()
        log_event(logging.INFO, "audio_capture_stopped", open_rate=open_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Persisted user settings
# ─────────────────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        log_exception("settings_load_failed", e, path=str(SETTINGS_PATH))
        return {}


def save_settings(**changes):
    settings = load_settings()
    settings.update(changes)
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception as e:
        log_exception("settings_save_failed", e, path=str(SETTINGS_PATH))


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
        
        # INFERENCE_LOCK serialises inference, so a second worker only adds
        # contention. One worker gets every physical core instead (logical
        # count / 2 on SMT CPUs; hyperthreads slow CTranslate2 down). On GPU the
        # CPU threads still run feature extraction and VAD.
        kwargs = dict(device=device, compute_type=compute_type)
        cores = os.cpu_count() or 4
        kwargs["cpu_threads"] = max(1, cores // 2 if cores > 4 else cores)
            
        try:
            from faster_whisper.utils import download_model
            download_model(model_size, local_files_only=True)
        except Exception:
            self.status_cb(f"Downloading Whisper '{model_size}' (first time only, "
                           f"can take a few minutes)…")
            log_event(logging.INFO, "whisper_model_download_started", model_size=model_size)
        model = WhisperModel(model_size, **kwargs)
        self.status_cb("Pre-warming model…")
        import numpy as np
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        # One token is enough to warm the encoder and decoder; letting it
        # decode freely on silence wastes seconds generating junk.
        list(model.transcribe(dummy, language="en", beam_size=1, max_new_tokens=1,
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
                self.worker_count  = 1
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

# Whisper's stock outputs on silence/noise (learned from YouTube captions).
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "please subscribe", "subscribe", "you", "bye",
    "hmm", "um", "uh", "oh", "so",
}


def is_hallucination(text: str) -> bool:
    core = "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()
    return not core or " ".join(core.split()) in HALLUCINATIONS


def plain_device_name(label: str) -> str:
    """'🔊 Speakers (FiiO Q series) [Loopback]' -> 'Speakers (FiiO Q series)'."""
    import re
    return re.sub(r"^[^\w(]+", "", label).replace(" [Loopback]", "").strip()


class Speaker(threading.Thread):
    """Text-to-speech via Windows SAPI, owned by one thread (COM objects are
    apartment-bound). Speech is asynchronous, so stop() purges it instantly,
    unlike pyttsx3's blocking runAndWait(). `speaking` is set while audio is
    playing so the capture thread can ignore our own voice."""
    SVSF_ASYNC, SVSF_PURGE = 1, 2
    HANGOVER_SEC = 0.4   # loopback capture lags playback slightly

    def __init__(self):
        super().__init__(daemon=True, name="tts")
        self._cmds    = queue.Queue()
        self.speaking = threading.Event()
        self.ready    = threading.Event()
        self.error    = None

    def configure(self, output_label: str, language: str):
        self._cmds.put(("config", output_label, language))

    def say(self, text: str):
        self._cmds.put(("say", text))

    def stop_speech(self):
        self._cmds.put(("stop",))

    def shutdown(self):
        self._cmds.put(("quit",))

    def _configure(self, voice, output_label, language):
        want_out = plain_device_name(output_label).lower()
        outputs  = voice.GetAudioOutputs()
        for i in range(outputs.Count):
            tok = outputs.Item(i)
            if tok.GetDescription().lower() == want_out:
                voice.AudioOutput = tok
                break
        voices = voice.GetVoices()
        chosen = None
        for i in range(voices.Count):
            tok = voices.Item(i)
            if language.lower() in tok.GetDescription().lower():
                chosen = tok
                break
        if chosen is not None:
            voice.Voice = chosen
        log_event(logging.INFO, "tts_configured", output=want_out, language=language,
                  voice=chosen.GetDescription() if chosen else None)
        if chosen is None:
            return f"No {language} text-to-speech voice installed (Windows Settings → Speech)."
        return None

    def run(self):
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except Exception as e:
            self.error = str(e)
            log_exception("tts_init_failed", e)
            self.ready.set()
            return
        self.ready.set()
        quiet_since = None
        while True:
            try:
                cmd = self._cmds.get(timeout=0.05)
            except queue.Empty:
                cmd = None
            try:
                if cmd is None:
                    pass
                elif cmd[0] == "quit":
                    voice.Speak("", self.SVSF_ASYNC | self.SVSF_PURGE)
                    break
                elif cmd[0] == "stop":
                    voice.Speak("", self.SVSF_ASYNC | self.SVSF_PURGE)
                elif cmd[0] == "say":
                    voice.Speak(cmd[1], self.SVSF_ASYNC)
                    self.speaking.set()
                elif cmd[0] == "config":
                    self.error = self._configure(voice, cmd[1], cmd[2])
                # RunningState 2 = SRSEIsSpeaking
                if voice.Status.RunningState == 2:
                    self.speaking.set()
                    quiet_since = None
                elif self.speaking.is_set():
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > self.HANGOVER_SEC:
                        self.speaking.clear()
            except Exception as e:
                log_exception("tts_command_failed", e, command=cmd[0] if cmd else None)
        self.speaking.clear()


class TranscriptionWorker(threading.Thread):
    """Finished phrases (phrase_queue) always take priority; live previews
    (partial_slot) only run when the model would otherwise sit idle.

    When the target is English, Whisper translates directly (task="translate")
    on the GPU: no network round trip, no rate limits, and it makes live
    previews of the *translation* cheap enough to show while the friend talks.
    """
    def __init__(self, wid, model, src_whisper, src_lang_key, tgt_lang_key,
                 phrase_queue, result_queue, status_cb,
                 partial_slot: "LatestSlot | None" = None):
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
        self.partial_slot = partial_slot
        self._stop_evt    = threading.Event()
        tgt_code          = LANGUAGES.get(tgt_lang_key, ("en", "en"))[0]
        self.local_translate = tgt_code == "en" and src_whisper != "en"
        self._last_final  = -1
        self._last_preview = None

    def stop(self):
        self._stop_evt.set()
        self.phrase_queue.put(None)

    NO_SPEECH_PROB = 0.6

    def _prepare(self):
        """Prompt tokens and VAD options, built once per worker."""
        from faster_whisper.vad import VadOptions
        tok = self.model.hf_tokenizer
        ids = lambda t: tok.token_to_id(t)
        head = [ids("<|startoftranscript|>"), ids(f"<|{self.src_whisper}|>")]
        tail = [ids("<|notimestamps|>")]
        self._prompts = {"transcribe": head + [ids("<|transcribe|>")] + tail,
                         "translate":  head + [ids("<|translate|>")] + tail}
        self._tokenizer = tok
        self._vad_opts  = VadOptions(min_silence_duration_ms=300, speech_pad_ms=200)
        self._n_frames  = self.model.feature_extractor.nb_max_frames

    def _decode(self, audio_np, tasks):
        """Return one text per task. VAD runs once (CPU); the GPU then encodes
        and decodes all tasks as a single batch, which costs little more than
        one task because Whisper's encoder always processes a padded 30 s
        window."""
        import numpy as np
        import ctranslate2
        from faster_whisper.vad import get_speech_timestamps
        stamps = get_speech_timestamps(audio_np, self._vad_opts)
        if not stamps:
            return ["" for _ in tasks], 0.0
        audio_np = audio_np[stamps[0]["start"]:stamps[-1]["end"]]
        feats = self.model.feature_extractor(audio_np)[:, :self._n_frames]
        if feats.shape[1] < self._n_frames:
            feats = np.pad(feats, ((0, 0), (0, self._n_frames - feats.shape[1])))
        batch = ctranslate2.StorageView.from_array(
            np.ascontiguousarray(np.stack([feats] * len(tasks)), dtype=np.float32))
        lock_t0 = time.perf_counter()
        with self.model_lock:
            lock_wait_ms = round((time.perf_counter() - lock_t0) * 1000, 2)
            enc = self.model.model.encode(batch, to_cpu=False)
            results = self.model.model.generate(
                enc, [self._prompts[t] for t in tasks],
                beam_size=1, max_length=224, return_no_speech_prob=True,
                suppress_blank=True)
        texts = []
        for r in results:
            if r.no_speech_prob > self.NO_SPEECH_PROB:
                texts.append("")
            else:
                texts.append(self._tokenizer.decode(r.sequences_ids[0]).strip())
        return texts, lock_wait_ms

    def run(self):
        import numpy as np
        self._prepare()
        tasks = ["transcribe", "translate"] if self.local_translate else ["transcribe"]
        while not self._stop_evt.is_set():
            try:
                item = self.phrase_queue.get(timeout=0.01)
            except queue.Empty:
                item = False
            if item is None:
                break
            if item is False:
                partial = self.partial_slot.take() if self.partial_slot else None
                if partial and partial[0] > self._last_final:
                    self._process_partial(*partial)
                continue
            seq, audio_bytes = item
            self._last_final = seq
            t0 = time.perf_counter()
            try:
                if not audio_bytes:
                    self.result_queue.put(("final", (seq, "", "")))
                    continue
                audio_np = (np.frombuffer(audio_bytes, dtype=np.int16)
                            .astype(np.float32) / 32768.0)
                texts, lock_wait_ms = self._decode(audio_np, tasks)
                text       = texts[0]
                translated = texts[1] if self.local_translate else None
                if is_hallucination(text):
                    log_event(logging.INFO, "transcription_discarded",
                              worker_id=self.wid, text=text)
                    text = ""
                    translated = "" if self.local_translate else None
                elif self.local_translate and is_hallucination(translated):
                    translated = ""
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                log_event(logging.INFO, "transcription_completed",
                          worker_id=self.wid, seq=seq, tasks=len(tasks),
                          input_bytes=len(audio_bytes), text_chars=len(text),
                          elapsed_ms=elapsed_ms, lock_wait_ms=lock_wait_ms,
                          queue_backlog=self.phrase_queue.qsize())
                self.result_queue.put(("metric", ("transcription_ms", elapsed_ms)))
                self.result_queue.put(("metric", ("lock_wait_ms",     lock_wait_ms)))
                if self._stop_evt.is_set():
                    break   # session restarted: don't leak into the new one
                # translated None => the UI sends it for online translation
                self.result_queue.put(("final", (seq, text, translated)))
            except Exception as e:
                self.result_queue.put(("final", (seq, "", "")))
                self.result_queue.put(("status", f"Worker {self.wid}: {e}"))
                log_exception("transcription_failed", e,
                              worker_id=self.wid,
                              input_bytes=len(audio_bytes) if audio_bytes else 0)

    def _process_partial(self, seq, audio_bytes):
        import numpy as np
        try:
            audio_np = (np.frombuffer(audio_bytes, dtype=np.int16)
                        .astype(np.float32) / 32768.0)
            tasks = ["transcribe", "translate"] if self.local_translate else ["transcribe"]
            t0 = time.perf_counter()
            texts, _ = self._decode(audio_np, tasks)
            heard = texts[0] if not is_hallucination(texts[0]) else ""
            translated = (texts[1] if self.local_translate and heard
                          and not is_hallucination(texts[1]) else None)
            preview = (seq, heard, translated)
            if heard and preview != self._last_preview and not self._stop_evt.is_set():
                self._last_preview = preview
                self.result_queue.put(("partial", preview))
                self.result_queue.put(("metric", ("partial_ms",
                                                  round((time.perf_counter() - t0) * 1000, 2))))
        except Exception as e:
            log_exception("partial_transcription_failed", e, worker_id=self.wid, seq=seq)


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
        self._muted               = False
        self._initialised         = False
        self._active_mode         = None
        self._start_retry_after_id = None
        # ── TTS ───────────────────────────────────────────────────────────
        self._tts_on   = False
        self.speaker   = Speaker()
        self.speaker.start()
        # Live preview state: per text box, where the grey preview line starts
        # and which phrase it belongs to.
        self._live       = {}
        self._mute_evt   = None
        self._partial_slot = None

        # ── Audio devices ─────────────────────────────────────────────────
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map  = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        input_labels     = [lbl for _, lbl in self._input_devices]
        output_labels    = [lbl for _, lbl in self._output_devices]

        nvidia_lbl  = find_device_by_keywords(self._input_devices,  NVIDIA_KEYWORDS)
        vb_lbl      = find_best_vb_input(self._input_devices)       # prefers CABLE-A
        friend_lbl  = find_best_friend_input(self._input_devices)   # FiiO loopback > VB cable
        fiio_lbl    = find_best_output_device(self._output_devices)  # K7 > Q series etc.
        default_out = output_labels[0] if output_labels else ""

        self._mode_state = [
            [nvidia_lbl or (input_labels[0] if input_labels else ""), "English", "Russian"],
            [friend_lbl or (input_labels[0] if input_labels else ""), "Russian", "English"],
        ]
        saved_devices = load_settings().get("mode_devices", [])
        for i, lbl in enumerate(saved_devices[:2]):
            if lbl in self._input_map:
                self._mode_state[i][0] = lbl

        self.device_var     = ctk.StringVar(value=self._mode_state[0][0])
        self.out_device_var = ctk.StringVar(value=fiio_lbl or default_out)
        # Replaying audio is only for VB-Audio cables; with a [Loopback]
        # input the audio is already playing on the device.
        self.loopback_var   = ctk.BooleanVar(
            value=bool(vb_lbl) and "[loopback]" not in (friend_lbl or "").lower())
        saved_model = load_settings().get("model")
        self.model_var      = ctk.StringVar(
            value=saved_model if saved_model in MODELS else _DEFAULT_MODEL)
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
        self._initialised = True
        self._start_session()
        self.after(30,   self._poll_queue)
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
        save_settings(model=self.model_var.get())
        if self.is_listening:
            self._stop_session(keep_mode=True)
        self._whisper_model = None
        self._preload_model()
        self._restart_session()

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
        ctk.CTkLabel(r0, text="🔊 = sound playing on an output device (e.g. FiiO in a call)",
                     font=ctk.CTkFont(size=11), text_color=("gray45", "gray60")
                     ).grid(row=0, column=2, padx=(10, 0))

        # Audio output + loopback toggle
        ctk.CTkLabel(outer, text="Audio output:", width=140, anchor="w",
                     font=ctk.CTkFont(size=13)).grid(row=1, column=0, sticky="w", pady=2)
        r1 = ctk.CTkFrame(outer, fg_color="transparent")
        r1.grid(row=1, column=1, sticky="w")
        self.out_device_menu = ctk.CTkOptionMenu(
            r1, variable=self.out_device_var, values=output_labels, width=260,
            command=self._on_manual_device_change)
        self.out_device_menu.grid(row=0, column=0)
        self.loopback_chk = ctk.CTkCheckBox(
            r1, text="Hear input through output  (VB-Audio cables only)",
            variable=self.loopback_var, font=ctk.CTkFont(size=12),
            command=self._on_manual_device_change)
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
        self.mute_btn = ctk.CTkButton(
            btns, text="🔇  Mute", width=165,
            command=self._toggle_mute)
        self.mute_btn.grid(row=0, column=0, padx=(0, 8))
        self.tts_btn = ctk.CTkButton(
            btns, text="🔇  Speak: Off", width=130,
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
            text_color=("gray10", "gray90"),
            command=self._toggle_tts)
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
        if self.is_listening:
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
        if self._initialised:
            self._restart_session()

    def _on_manual_device_change(self, _=None):
        self._restart_session()

    def _on_manual_lang_change(self, _=None):
        self.heard_label.configure(text=f"Heard in {self.src_var.get()}")
        self.translated_label.configure(text=f"Translation → {self.tgt_var.get()}")
        self._restart_session()

    def _refresh_devices(self):
        if self.is_listening:
            self._stop_session(keep_mode=True)
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
        friend_lbl = find_best_friend_input(self._input_devices)
        if self._mode_state[0][0] not in self._input_map and nvidia_lbl:
            self._mode_state[0][0] = nvidia_lbl
        if self._mode_state[1][0] not in self._input_map and friend_lbl:
            self._mode_state[1][0] = friend_lbl
        self.status_var.set(f"Refreshed — {len(in_labels)} in / {len(out_labels)} out device(s).")
        log_event(logging.INFO, "audio_devices_refreshed",
                  input_count=len(in_labels), output_count=len(out_labels))
        self._refresh_diagnostics_once()
        self._restart_session()

    # ── Session start / stop ──────────────────────────────────────────────
    # The app listens continuously; settings changes restart the session and
    # Mute is the only way to stop it.

    def _restart_session(self):
        if self.is_listening:
            self._stop_session(keep_mode=True)
        if not self._muted:
            self._start_session()

    def _toggle_mute(self):
        self._muted = not self._muted
        if self._muted:
            self._stop_session()
            self.mute_btn.configure(text="🎙  Unmute", fg_color=COL_STOP,
                                    hover_color=COL_STOP_HOV)
            self._set_live_status("MUTED", COL_ERR)
            self.status_var.set("Muted — not listening.")
        else:
            self.mute_btn.configure(text="🔇  Mute", fg_color=["#3B8ED0", "#1F6AA5"],
                                    hover_color=["#3281BC", "#1A5C8F"])
            self._start_session()
        log_event(logging.INFO, "mute_toggled", muted=self._muted)

    def _start_session(self):
        if self.is_listening or self._muted:
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
        self._set_live_status("LISTENING", COL_OK)
        self.status_var.set("Listening…")

        in_idx = self._input_map.get(self.device_var.get(), -1)
        if in_idx == -1:
            in_idx = None
        log_event(logging.INFO, "audio_device_selected",
                  device_label=self.device_var.get(), device_index=in_idx)
        if self._active_mode is not None:
            self._mode_state[self._active_mode][0] = self.device_var.get()
        save_settings(mode_devices=[m[0] for m in self._mode_state])
        capturing_output = "[loopback]" in self.device_var.get().lower()

        src_key     = self.src_var.get()
        tgt_key     = self.tgt_var.get()
        src_whisper, _ = LANGUAGES.get(src_key, ("en", "en"))

        worker_count = getattr(self._model_loader, "worker_count", NUM_WORKERS)
        self.phrase_queue  = queue.Queue(maxsize=max(worker_count * 4, 8))
        self._partial_slot = LatestSlot()
        self._live         = {}
        # Spoken translations played on the device we're capturing would be
        # heard as the friend; mute capture while the speaker is talking.
        same_device = (capturing_output and plain_device_name(self.device_var.get())
                       == plain_device_name(self.out_device_var.get()))
        self._mute_evt = self.speaker.speaking if same_device else None
        if self._tts_on:
            self._configure_speaker()

        # ── Optional loopback output (VB-Audio → Fiio headphones) ────────
        # Look up the native sample rate of the input device so the loopback
        # output thread opens at the same rate and PortAudio accepts it.
        _pa_tmp = pyaudio.PyAudio() if _PYAUDIO_AVAILABLE else None
        in_native_rate = get_device_native_rate(_pa_tmp, in_idx) if _pa_tmp else SAMPLE_RATE
        if _pa_tmp:
            _pa_tmp.terminate()
        self._loopback_out = None
        if self.loopback_var.get() and capturing_output:
            # The audio is already playing on that device; replaying it would
            # double it up (or feed back if it's the same device).
            log_event(logging.INFO, "loopback_skipped_output_capture",
                      device_label=self.device_var.get())
        elif self.loopback_var.get():
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

        # Opening a new PortAudio instance while the previous capture thread is
        # still terminating its own fails with "Unanticipated host error".
        prev = getattr(self, "_stopping_capture", None)
        if prev is not None and prev.is_alive():
            prev.join(timeout=1.0)
        self._stopping_capture = None
        self._capture_thr = AudioCaptureThread(
            device_index=in_idx,
            phrase_queue=self.phrase_queue,
            status_cb=lambda m: self.result_queue.put(("status", m)),
            level_cb=self._on_level,
            loopback_out=self._loopback_out,
            partial_slot=self._partial_slot,
            mute_evt=self._mute_evt,
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
                status_cb=lambda m: self.result_queue.put(("status", m)),
                partial_slot=self._partial_slot)
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
            self._stopping_capture = self._capture_thr
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
        self.level_bar.set(0)
        self.level_text.configure(text="—")
        self._set_live_status("IDLE", COL_WARN)
        log_event(logging.INFO, "session_stopped",
                  session_id=self.session_id, keep_mode=keep_mode)
        if not keep_mode:
            self.status_var.set("Stopped.")

    # ── TTS — fixed: lock prevents "run loop already started" ────────────

    def _toggle_tts(self):
        """On: each new translation is spoken as it arrives, through the
        selected Audio output device. Off: speech stops immediately."""
        if self.speaker.error and not self.speaker.is_alive():
            self.status_var.set(f"Text-to-speech unavailable: {self.speaker.error}")
            return
        self._tts_on = not self._tts_on
        if self._tts_on:
            self._configure_speaker()
            self.tts_btn.configure(text="🔊  Speak: On", fg_color=COL_OK,
                                   hover_color=COL_MY_HOVER, text_color="white")
        else:
            self.speaker.stop_speech()
            self.tts_btn.configure(text="🔇  Speak: Off", fg_color=("gray75", "gray30"),
                                   hover_color=("gray65", "gray40"),
                                   text_color=("gray10", "gray90"))
        log_event(logging.INFO, "tts_toggled", enabled=self._tts_on)

    def _configure_speaker(self):
        self.speaker.configure(self.out_device_var.get(), self.tgt_var.get())
        self.after(300, lambda: self.speaker.error and self.status_var.set(self.speaker.error))

    # ── Clear ─────────────────────────────────────────────────────────────

    def _clear(self):
        self.heard_box.delete("1.0", "end")
        self.translated_box.delete("1.0", "end")
        self._live = {}
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
                if kind == "partial":
                    seq, heard, translated = data
                    self._set_live(self.heard_box, seq, heard)
                    if translated:
                        self._set_live(self.translated_box, seq, translated)
                elif kind == "final":
                    seq, text, translated = data
                    self._commit(self.heard_box, seq, text)
                    if translated is not None:
                        self._commit_translation(seq, translated)
                    elif text:
                        self.result_queue.put(("translate", (seq, text, self.src_var.get(),
                                                             self.tgt_var.get())))
                    else:
                        self._commit(self.translated_box, seq, "")
                elif kind == "translate":
                    seq, text, sk, tk = data
                    try:
                        self._translation_queue.put((seq, text, sk, tk), timeout=0.1)
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
                    seq, text = data
                    self._commit_translation(seq, text)
                elif kind == "status":
                    self.status_var.set(data)
                    log_event(logging.INFO, "status_updated", status=data)
                elif kind == "level":
                    self._update_meter(*data)
                elif kind == "metric":
                    self._update_metric(*data)
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

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

    def _set_live(self, box, seq, text):
        """Show (or replace) the grey in-progress line for phrase `seq`."""
        start, live_seq = self._live.get(box, (None, None))
        if live_seq is not None and seq < live_seq:
            return
        if start is not None:
            box.delete(start, "end")
        else:
            start = box.index("end-1c")
        prefix = "\n" if box.get("1.0", start).strip() else ""
        box.insert(start, prefix + "… " + text, "live")
        box.tag_config("live", foreground="gray55")
        box.see("end")
        self._live[box] = (start, seq)

    def _commit(self, box, seq, text):
        """Replace the live line (if it belongs to `seq` or earlier) with the
        final text."""
        start, live_seq = self._live.get(box, (None, None))
        if start is not None and live_seq <= seq:
            box.delete(start, "end")
            self._live.pop(box, None)
        if text:
            if box in self._live:      # a newer phrase is previewing: insert before it
                start, live_seq = self._live[box]
                prefix = "\n" if box.get("1.0", start).strip() else ""
                box.insert(start, prefix + text)
                self._live[box] = (box.index(f"{start} + {len(prefix + text)} chars"), live_seq)
            else:
                self._append(box, text)
                self._trim_textbox(box)

    def _commit_translation(self, seq, text):
        self._commit(self.translated_box, seq, text)
        if text and self._tts_on and not text.startswith("[translation failed"):
            self.speaker.say(text)
        self.status_var.set("Listening…" if self.is_listening else "Done.")

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
            self.speaker.shutdown()
        finally:
            self.destroy()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_event(logging.INFO, "app_main_start")
    app = InterpreterApp()
    app.mainloop()
    log_event(logging.INFO, "app_main_exit")

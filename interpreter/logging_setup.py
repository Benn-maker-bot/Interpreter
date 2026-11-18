"""Logging: every event the app records, and what happens to errors.

- log_event(level, "event_name", key=value ...) writes one JSON line to
  interpreter.log.jsonl (warnings and errors also to the error log) and a
  readable line to the console. loguru does the writing.
- explain_issue() turns a logged error into plain language for the window.
- log_exception() logs an error with its traceback.
- install_global_exception_hooks() makes sure crashes in any thread are
  logged too.
- build_log_summary() is behind the "Export Log Summary" button.

Importing this module sets up logging straight away (_setup_logging() runs
at import time), so any module that imports log_event can use it at once.
"""

import threading
import json
import logging
import os
import sys
import traceback
import gzip
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .config import (
    ERROR_LOG_PATH, LOG_BACKUP_COUNT, LOG_PATH, LOG_SUMMARY_PATH, MAX_LOG_BYTES)


# Device labels contain emoji; a cp1252 console would raise on every log line.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Structured JSONL logging
#
# Everything the app does is recorded with log_event("event_name", key=value).
# Each entry becomes one line of JSON in interpreter.log.jsonl (warnings and
# errors also go to interpreter.error.log.jsonl) and a readable line in the
# console. log_tools.py reads these files to measure speed and accuracy.
# ─────────────────────────────────────────────────────────────────────────────


def _json_safe(value):
    """Turn values json can't write (numpy numbers/arrays, paths, sets) into
    plain ones, so a log line is never lost because of one field."""
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        return value.item()               # numpy scalar -> Python number/bool
    if hasattr(value, "tolist"):
        return value.tolist()             # numpy array -> list
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


# loguru does the writing: it is thread-safe, rotates and compresses the
# files itself, records where each line was logged from (module, function,
# line) and prints tracebacks with the values of the variables involved.
# log_event() keeps its old signature, so the rest of the app is unchanged.
#
# Set the environment variable INTERPRETER_LOG_LEVEL=DEBUG to also record
# DEBUG events (default INFO).
LOG_LEVEL = os.environ.get("INTERPRETER_LOG_LEVEL", "INFO").upper()

# Fields every JSON line has; the extra fields passed to log_event() follow.
_LOG_BASE_KEYS = ("ts", "level", "logger", "event", "message", "thread",
                  "process", "module", "function", "line")


def _to_json_line(record) -> None:
    """loguru patcher: build the JSON text for one record and keep it in
    record["extra"]["_json"]. Patchers run on the thread that logged, while
    the traceback still exists, so the "exception" field is always complete."""
    extra = record["extra"]
    payload = {
        "ts":       record["time"].isoformat(timespec="milliseconds"),
        "level":    record["level"].name,
        "logger":   extra.get("logger", "interpreter.app"),
        "event":    extra.get("event", record["message"]),
        "message":  record["message"],
        "thread":   record["thread"].name,
        "process":  record["process"].id,
        "module":   record["name"],
        "function": record["function"],
        "line":     record["line"],
    }
    payload.update(extra.get("fields") or {})
    if record["exception"] is not None:
        payload["exception"] = "".join(traceback.format_exception(*record["exception"]))
    # default=_json_safe: numpy numbers (np.int64, np.bool_ …) are not
    # JSON types; without it the whole line was lost.
    extra["_json"] = json.dumps(payload, ensure_ascii=False, default=_json_safe)
    # Short "key=value, …" text for the console line.
    extra["_kv"] = ", ".join(f"{k}={v}" for k, v in (extra.get("fields") or {}).items())


def _json_format(record) -> str:
    """loguru format function for the JSONL files: the ready-made JSON line."""
    return "{extra[_json]}\n"


def _console_format(record) -> str:
    """loguru format function for the console: time, level, thread, event,
    fields, and (added by loguru) a coloured traceback when there is one."""
    kv = " | {extra[_kv]}" if record["extra"].get("_kv") else ""
    return ("<green>{time:HH:mm:ss.SSS}</green> <level>{level: ^8}</level> | "
            "<cyan>{thread.name: <15.15}</cyan> | <level>{extra[event]}</level>" + kv
            + " <dim>({name}:{function}:{line})</dim>\n{exception}")


def _gzip_rotated(path: str, base: Path = None) -> None:
    """loguru compression hook, run after a log file passes MAX_LOG_BYTES.

    loguru renames interpreter.log.jsonl to interpreter.log.<time>.jsonl;
    this gzips it as interpreter.log.jsonl.<time>.gz (the name log_tools.py
    and iter_log_files() look for) and keeps the newest LOG_BACKUP_COUNT."""
    src = Path(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    gz_path = base.with_name(f"{base.name}.{stamp}.gz")
    try:
        with open(src, "rb") as fin, gzip.open(gz_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        src.unlink(missing_ok=True)
    except Exception as e:
        print(f"Log compression failed: {e}", file=sys.stderr)
    old = sorted(base.parent.glob(base.name + ".*.gz"), key=lambda p: p.stat().st_mtime)
    for stale in old[:-LOG_BACKUP_COUNT]:
        stale.unlink(missing_ok=True)


class _InterceptHandler(logging.Handler):
    """
    Sends warnings from libraries that use Python's standard logging
    (faster_whisper, huggingface_hub, urllib3, ctranslate2 …) into loguru,
    so they reach the log files instead of only the console.
    
    This class acts as a bridge between Python's built-in logging system and
    the more advanced loguru library. When external libraries like Whisper or
    Hugging Face generate log messages, this handler catches them and forwards
    them to the loguru system so they're properly recorded in the application's
    log files instead of just appearing on the console.
    """
    def emit(self, record: logging.LogRecord) -> None:
        """
        Called by Python's logging system: forward the record as a log_event.
        
        This method is automatically invoked when any library using standard
        Python logging generates a message. It takes that log record and converts
        it into a format suitable for the application's main logging system.
        """
        if record.name.startswith("interpreter"):
            return
        text = record.getMessage()
        # Public Hub repositories can be downloaded without authentication.
        # This notice only says unauthenticated requests have lower rate limits;
        # treating it as an app warning makes it appear as a program problem.
        hub_auth_notice = (record.name.startswith("huggingface_hub.") and
                           "unauthenticated requests to the hf hub" in text.lower())
        event = ("huggingface_auth_optional" if hub_auth_notice else
                 "library_" + record.name.split(".")[0].replace("-", "_"))
        fields = {"source": record.name, "text": text,
                  "where": f"{record.pathname}:{record.lineno} (in {record.funcName})"}
        logger.opt(exception=record.exc_info or None).bind(event=event, fields=fields) \
              .log("INFO" if hub_auth_notice else
                   (record.levelname if record.levelname in _LOGURU_LEVELS else record.levelno),
                   text)


_LOGURU_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


def _setup_logging() -> None:
    """
    Replace loguru's default console output with the app's sinks:
    console (readable, coloured), the main JSONL log and the error JSONL log.
    
    This function configures the logging system to write to both the console
    and log files in a structured format. It sets up the appropriate formatting
    for different types of output and ensures that all application events are
    properly recorded for debugging and analysis.
    """
    logger.remove()
    logger.configure(patcher=_to_json_line, extra={"event": "", "fields": {}})
    if sys.stdout is not None:   # None in a windowed .exe (no console)
        logger.add(sys.stdout, level=LOG_LEVEL, format=_console_format,
                   colorize=None, backtrace=True, diagnose=True)
    for path, level in ((LOG_PATH, LOG_LEVEL), (ERROR_LOG_PATH, "WARNING")):
        logger.add(path, level=level, format=_json_format, encoding="utf-8",
                   rotation=MAX_LOG_BYTES,
                   compression=lambda p, base=path: _gzip_rotated(p, base),
                   backtrace=True, diagnose=False)
    # Library warnings (and errors) from standard logging go to loguru too.
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.WARNING, force=True)


_setup_logging()


def add_issue_sink(sink: Callable[[Any], None]) -> int:
    """Also send every warning/error to `sink` (a function that gets loguru's
    message; message.record has "level", "extra" and "exception"). Used by
    the Diagnostics page's "Recent issues". Returns an id for remove_log_sink()."""
    return logger.add(sink, level="WARNING", format="{message}", catch=True)


def remove_log_sink(sink_id: int) -> None:
    """Stop a sink added by add_issue_sink() (ignored if already removed)."""
    try:
        logger.remove(sink_id)
    except ValueError:
        pass


def _level_name(level: int | str) -> str | int:
    """logging.INFO (20) -> "INFO", which loguru knows; other numbers as is."""
    if isinstance(level, str):
        return level.upper()
    name = logging.getLevelName(level)
    return name if name in _LOGURU_LEVELS else level


def log_event(level: int, event: str, **fields: Any) -> None:
    """
    Write one event to the log. This is the function the whole app uses.
    
    This is the main logging function used throughout the application to
    record events and activities. It writes structured JSON data to the
    log files that can be analyzed later for performance, debugging, and
    usage statistics.
    
    Args:
        level: The logging level (logging.INFO, logging.WARNING, etc.)
        event: A short name such as "phrase_trace"
        **fields: Any keyword arguments are stored as extra JSON fields
    
    Note:
        Warnings and errors also go to the error log. The line records the
        caller's module, function and line number.
    """
    logger.opt(depth=1).bind(event=event, fields=fields).log(_level_name(level), event)


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language explanations of errors (shown in the window)
#
# The log names an error by its event ("music_processing_failed") and
# Python's message ("name 'np' is not defined"), which means little to
# someone who didn't write the code. explain_issue() turns that into a
# title, what it means, and what to do. Events not listed get a general
# explanation based on the kind of Python error.
# ─────────────────────────────────────────────────────────────────────────────

ISSUE_HELP = {
    # event: (title, what it means, what you can do)
    "music_processing_failed": (
        "Music mode stopped working",
        "The part that detects music and removes instruments from the sound "
        "failed. Speech is still recognised normally, just without the "
        "music adjustments.",
        "Nothing urgent. If it keeps happening, restart the app; if it still "
        "fails, it is a bug - copy the details for the developer."),
    "music_scene_failed": (
        "Music detection failed",
        "The once-per-second check for music could not run.",
        "Speech recognition continues. Restart the app if music mode is needed."),
    "music_scene_unavailable": (
        "Music mode is not available",
        "The music detector could not be started (a package may be missing).",
        "Re-run setup_and_run.bat, or the pip install command in README.txt."),
    "transcription_failed": (
        "A phrase could not be recognised",
        "Whisper failed on one piece of speech; that phrase is missing from "
        "the transcript.",
        "One-off failures are harmless. If many happen, try a smaller Model."),
    "translation_failed": (
        "A translation failed",
        "The text was recognised but could not be translated.",
        "Check the internet connection if the local translator isn't used."),
    "audio_device_error": (
        "Problem with the audio device",
        "The selected microphone or speaker stopped sending sound.",
        "Check it is plugged in; choose it again under Audio input; in Windows "
        "Sound settings untick 'Allow applications to take exclusive control'."),
    "audio_open_failed": (
        "The audio device could not be opened",
        "Windows refused to let the app record from the selected device.",
        "Pick another device, or close apps that may be using it exclusively."),
    "phrase_record_failed": (
        "A recording could not be saved",
        "The audio copy kept for accuracy checks was not written (disk full, "
        "or a file open in another program). Recognition is not affected.",
        "Close programs using files in the recordings folder."),
    "settings_save_failed": (
        "Settings could not be saved",
        "Your choices will not be remembered next time.",
        "Check that interpreter.settings.json is not read-only or open elsewhere."),
    "power_throttling_failed": (
        "Windows power saving could not be turned off",
        "When this window is in the background, Windows may slow the app down.",
        "Usually harmless; keep the window visible during important calls."),
    "resource_warning": (
        "The computer is running short of resources",
        "Free memory, graphics memory or CPU time is low; recognition may "
        "become slow or stutter.",
        "Close unused programs, or choose a smaller Model."),
    "source_silent": (
        "No sound is reaching the selected device",
        "Nothing is playing on it: the app playing the sound stopped, or it is "
        "playing to a different device.",
        "Check the call or video is playing, and that it plays to this device."),
    "music_processing_disabled": (
        "Music mode was switched off",
        "Music processing failed several times in a row, so it was turned off "
        "for this session. Speech recognition works as usual.",
        "Restart the app to try music mode again."),
    "watchdog_restart": (
        "A stuck part of the app was restarted",
        "A background task stopped responding and was restarted automatically.",
        "Nothing to do unless it happens often."),
    "whisper_model_primary_load_failed": (
        "The speech recognition model '{model_size}' could not be loaded",
        "Whisper (the speech recogniser) failed to start with model '{model_size}' on "
        "{device}. Common causes: the model name is wrong, the download was interrupted, "
        "there is no internet for the first download, or the graphics card ran out of memory.",
        "Choose another Model and try again. If it says out of memory, close other programs "
        "that use the graphics card. The app falls back to the small 'tiny' model on the CPU."),
    "whisper_model_cpu_fallback_ready": (
        "Using the backup model '{active_model_size}' on the CPU",
        "The chosen model could not load on the graphics card ({original_device}), so the "
        "much smaller '{active_model_size}' model runs on the processor instead. Recognition "
        "works but is less accurate and slower.",
        "See the error just before this one for why the chosen model failed."),
    "phrase_dropped_queue_full": (
        "A phrase was lost: recognition fell too far behind",
        "{queue_size} finished phrases were already waiting to be transcribed, so phrase "
        "{seq} ({audio_sec} s of audio) had to be dropped. The graphics card is too busy: "
        "another program is using it, or the model is too large for it.",
        "Close other programs using the graphics card (games, AI tools, video editors), or "
        "choose a smaller Model or Previews: Off."),
    "phrase_queue_backlog_high": (
        "Recognition is falling behind",
        "{backlog} finished phrases are waiting to be transcribed, so text appears late.",
        "Close other programs using the graphics card, or choose a smaller Model or "
        "Previews: Off."),
    "early_final_failed": (
        "A phrase could not be pre-transcribed",
        "To save time the app starts transcribing a phrase as soon as a pause begins. That "
        "attempt failed for phrase {seq}; the phrase is transcribed normally instead.",
        "Nothing to do unless it repeats; then copy the details for the developer."),
    "audio_device_enumeration_failed": (
        "The list of sound devices could not be read",
        "Windows' audio system (PortAudio) didn't answer, so device lists may be empty.",
        "Unplug and re-plug USB audio devices, or restart the app."),
}

# General explanations by Python error type, for events not listed above
ERROR_KIND_HELP = {
    "NameError":       "a programming mistake in the app (a name used in the code doesn't exist) - a bug",
    "AttributeError":  "a programming mistake in the app, or an incompatible package version",
    "TypeError":       "a programming mistake in the app (wrong kind of value) - a bug",
    "KeyError":        "missing data the code expected - probably a bug",
    "ImportError":     "a required package is missing or broken",
    "ModuleNotFoundError": "a required package is not installed",
    "FileNotFoundError": "a file or model the app needs was not found",
    "PermissionError": "Windows refused access to a file or device",
    "OSError":         "a problem with a device, file or the operating system",
    "MemoryError":     "the computer ran out of memory",
    "ConnectionError": "no internet connection (or the service is down)",
    "TimeoutError":    "something took too long to respond",
    "RuntimeError":    "a library reported a problem (often the graphics card / CUDA)",
}


class _Blank(dict):
    """
    For str.format_map: a {field} the log line doesn't have shows as '?'.
    
    This is a helper class that ensures when formatting strings with
    placeholders, missing keys are replaced with '?' instead of raising
    a KeyError. This prevents crashes when some log entries might not have
    all the fields that are expected for display.
    """
    def __missing__(self, key):
        return "?"


# Log fields that are noise in an explanation (who logged, internal ids)
_DETAIL_SKIP = {"exception_type", "exception_message", "session_id", "run_id", "process",
                "thread", "logger", "message", "ts", "event", "level",
                "module", "function", "line"}


def _describe_fields(fields, limit=12):
    """The facts logged with a problem, one 'name: value' per line, so the
    explanation shows WHAT was involved (file, device, language …)."""
    lines = []
    for key, value in fields.items():
        if key in _DETAIL_SKIP or value in (None, "", [], {}):
            continue
        text = str(value)
        lines.append(f"{key.replace('_', ' ')}: {text[:200] + '…' if len(text) > 200 else text}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _where(exc_info):
    """Where the error happened: the last line of the app's own code (which
    is what a developer opens first) and, if different, the library line
    that actually raised it."""
    if not exc_info or exc_info[2] is None:
        return ""
    tb = traceback.extract_tb(exc_info[2])
    if not tb:
        return ""
    # "Our" code is the interpreter\ package and the scripts beside it
    # (interpreter_qt.py, session_controller.py ...) in the project folder.
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(pkg_dir)
    ours = [f for f in tb
            if os.path.dirname(os.path.abspath(f.filename)) in (pkg_dir, app_dir)]
    last = tb[-1]
    describe = lambda f: f"{os.path.basename(f.filename)} line {f.lineno} (in {f.name})"
    if ours and ours[-1] is not last:
        lib = os.path.relpath(last.filename, app_dir) if "site-packages" not in last.filename \
              else last.filename.split("site-packages")[-1].strip("\\/")
        return f"{describe(ours[-1])}, inside the library {lib} line {last.lineno}"
    return describe(last)


def explain_issue(event: str, fields: dict[str, Any] | None = None,
                  exc_info: Any = None) -> dict[str, str]:
    """Plain-language description of a logged problem: dict with title,
    meaning, action, technical (the Python error), where (file:line) and
    details (the facts logged with it). ISSUE_HELP texts may name fields in
    braces, e.g. "{device}", which are filled in from the log line."""
    fields = fields or {}
    kind = fields.get("exception_type") or ""
    message = fields.get("exception_message") or fields.get("error") or ""
    if not message and fields.get("problems"):
        message = "; ".join(map(str, fields["problems"]))
    where = _where(exc_info)
    general = ERROR_KIND_HELP.get(kind)
    if event in ISSUE_HELP:
        title, meaning, action = (s.format_map(_Blank(fields)) for s in ISSUE_HELP[event])
        if general and general not in meaning:
            meaning += f" (Python reported {kind}: {general}.)"
    else:
        title = event.replace("_", " ").capitalize()
        part = event.split("_")[0]
        meaning = (f"The '{part}' part of the app reported a problem: {general}." if general
                   else f"The '{part}' part of the app logged '{event}' as a problem. There is "
                        "no specific explanation for this message yet; the details below "
                        "show what was involved.")
        action = ("If it keeps happening, restart the app. Copy the details (Copy button) "
                  "to send to the developer; the log file has the full record.")
    technical = f"{kind}: {message}" if kind else message
    return {"title": title, "meaning": meaning, "action": action,
            "technical": technical.strip(": "), "where": where,
            "details": _describe_fields(fields)}


def log_exception(event: str, exc: Exception, **fields: Any) -> None:
    """Log an error together with its full traceback (the list of calls that
    led to it), so problems can be diagnosed later from the log files.
    """
    merged = dict(fields)
    merged.setdefault("exception_type",    type(exc).__name__)
    merged.setdefault("exception_message", str(exc))
    logger.opt(depth=1, exception=exc).bind(event=event, fields=merged).error(event)


def install_global_exception_hooks():
    """Make sure crashes that nobody caught are still written to the log.

    Python normally just prints such errors to the console (which may not be
    visible). These hooks log them as CRITICAL instead. Only the app itself
    installs them (see the bottom of the file), not scripts that import it.
    """
    def _excepthook(exc_type, exc_value, exc_tb):
        logger.opt(exception=(exc_type, exc_value, exc_tb))               .bind(event="unhandled_exception", fields={}).critical("unhandled_exception")

    def _threading_excepthook(args):
        logger.opt(exception=(args.exc_type, args.exc_value, args.exc_traceback))               .bind(event="unhandled_thread_exception",
                    fields={"thread": args.thread.name if args.thread else None})               .critical("unhandled_thread_exception")

    sys.excepthook      = _excepthook
    threading.excepthook = _threading_excepthook


# ─────────────────────────────────────────────────────────────────────────────
# Log summary helpers
#
# Small helpers behind the "Export Log Summary" button: read the log files
# back and count what happened.
# ─────────────────────────────────────────────────────────────────────────────

def iter_log_files(base_path: Path):
    """List the current log file plus its rotated, gzipped older copies."""
    files = [base_path] if base_path.exists() else []
    files.extend(sorted(base_path.parent.glob(base_path.name + ".*.gz")))
    return files


def read_jsonl_entries(path: Path):
    """Read a .jsonl (or .jsonl.gz) log file and yield one dictionary per line,
    skipping any line that is not valid JSON.
    """
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
    """Count log entries by level and event and write interpreter.log.summary.json.

    Used by the "Export Log Summary" button.
    """
    # Counters for log levels and event names, plus a list of problems.
    # (A Counter is a dict that counts: levels['ERROR'] += 1 starts at 0.)
    levels, events, issues, total = Counter(), Counter(), [], 0
    # Read every entry of every log file, counting as we go and keeping
    # warnings and errors aside.
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
    # Gather the results and save them as an indented, readable JSON file.
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

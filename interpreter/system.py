"""The computer the app runs on: GPU, Windows scheduling, resource use.

- detect_device() runs once at import and decides WHISPER_DEVICE ("cuda" or
  "cpu"), COMPUTE_TYPE and NUM_WORKERS. INFERENCE_LOCK lets only one AI
  model use the GPU at a time.
- disable_power_throttling() / boost_audio_thread() stop Windows slowing
  the app down when its window is in the background.
- gpu_snapshot(), audio_sessions_snapshot() and log_session_context() record
  what the PC was doing, so a log can be read later with that context.
- collect_resource_stats() / resource_warnings() measure CPU, RAM and GPU
  use for the Diagnostics panel and the resource_stats log line.
"""

import threading
import time
import logging
import os
import sys
from pathlib import Path

from .config import (
    APP_DIR, DEFAULT_MODEL_CPU, DEFAULT_MODEL_GPU, HIGH_SYSTEM_CPU_PCT,
    LOW_GPU_MEM_MB, LOW_RAM_MB)
from .logging_setup import log_event, log_exception


def _add_cuda_dll_dirs():
    """
    Make the pip-installed NVIDIA CUDA DLLs (nvidia-cublas-cu12,
    nvidia-cudnn-cu12) loadable by ctranslate2 on Windows.
    
    This function ensures that the necessary CUDA libraries can be found
    and loaded when running on Windows systems with NVIDIA GPUs. It's needed
    because CTranslate2 (the engine used for Whisper models) requires these
    specific DLL files to be accessible.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Windows scheduling: keep the audio smooth while Teams/Telemost run too
#
# When this window is in the background (Teams in front), Windows may put
# the app in "EcoQoS" power-saving mode: slower clocks and the small
# efficiency cores. Audio work must not be throttled (Microsoft: EcoQoS is
# not for performance-critical work), so the process opts out. The audio
# capture thread also registers with MMCSS "Pro Audio" (the Multimedia Class
# Scheduler, used by audio apps) so a CPU burst from another app cannot
# delay it; the logs showed one 364 ms capture stall.
# ─────────────────────────────────────────────────────────────────────────────

def acquire_single_instance():
    """Allow only one copy of the app at a time. Two copies would fight over
    the audio device and the GPU, and both write the same log file at once,
    which corrupted a log line in testing. Returns an open lock file to keep
    for the app's lifetime, or None if another copy already holds it.
    Used by __main__.py and by interpreter_qt.py when that is run directly."""
    import msvcrt
    lock = open(APP_DIR / "interpreter.lock", "a+")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return lock
    except OSError:
        lock.close()
        return None


def disable_power_throttling():
    """Opt the whole process out of EcoQoS throttling (Windows 10 1709+)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        # ControlMask = EXECUTION_SPEED (1) controlled by us; StateMask 0 = off
        state = PowerThrottlingState(1, 1, 0)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   ctypes.c_void_p, wintypes.DWORD]
        ok = kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4,  # ProcessPowerThrottling
                                            ctypes.byref(state), ctypes.sizeof(state))
        log_event(logging.INFO, "power_throttling_disabled", ok=bool(ok))
        return bool(ok)
    except Exception as e:
        log_exception("power_throttling_failed", e)
        return False


def boost_audio_thread():
    """Call from the audio capture thread: MMCSS "Pro Audio", or failing
    that a plain above-normal thread priority. Returns what was applied."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        task_index = wintypes.DWORD(0)
        avrt = ctypes.windll.avrt
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        if avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index)):
            return "mmcss_pro_audio"
    except Exception:
        pass
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        if kernel32.SetThreadPriority(ctypes.c_void_p(kernel32.GetCurrentThread()), 1):  # ABOVE_NORMAL
            return "above_normal"
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
#
# Whisper is much faster on an NVIDIA graphics card (GPU). This checks, once
# at start-up, that the GPU and its CUDA libraries are usable, and picks
# sensible defaults for GPU or CPU.
# ─────────────────────────────────────────────────────────────────────────────

def detect_device():
    """Decide whether Whisper runs on the NVIDIA GPU ("cuda") or the CPU.

    Returns (device, compute_type, worker_count, default_model_label). The GPU
    is only chosen if the CUDA libraries really load; otherwise Whisper would
    crash later in the middle of transcribing.
    """
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


# Decided once at start-up:
#   WHISPER_DEVICE  "cuda" (GPU) or "cpu"
#   COMPUTE_TYPE    number format for the models (float16 on GPU, int8 on CPU)
#   NUM_WORKERS     transcription threads (always 1, see WhisperModelLoader)
#   _DEFAULT_MODEL  the Model-menu entry to use if none was saved
WHISPER_DEVICE, COMPUTE_TYPE, NUM_WORKERS, _DEFAULT_MODEL = detect_device()
# Only one AI model may use the GPU at a time; everything that runs a
# model does it inside `with INFERENCE_LOCK:` (see Lock in the glossary).
INFERENCE_LOCK = threading.Lock()


def audio_sessions_snapshot() -> list:
    """Which apps have sound sessions on the default playback device, and
    whether they are playing right now (Windows Core Audio, via pycaw).
    Logged when the captured device goes silent or comes back, so a source
    app that stopped (e.g. a browser tab pausing) can be told apart from a
    problem in this program. Returns [] if pycaw is missing."""
    try:
        import comtypes
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
        comtypes.CoInitialize()
        out = {}
        for session in AudioUtilities.GetAllSessions():
            name = session.Process.name() if session.Process else "system sounds"
            try:
                peak = session._ctl.QueryInterface(IAudioMeterInformation).GetPeakValue()
            except Exception:
                peak = None
            state = {0: "inactive", 1: "active", 2: "expired"}.get(session.State, session.State)
            prev = out.get(name)
            # One entry per app: the loudest / most active of its sessions
            if prev is None or (peak or 0) > (prev["peak"] or 0) or state == "active":
                out[name] = {"state": state, "peak": round(peak, 3) if peak is not None else None}
        return [dict(app=k, **v) for k, v in out.items()
                if v["state"] == "active" or (v["peak"] or 0) > 0]
    except Exception as e:
        return [{"error": type(e).__name__}]


_APP_VERSION = None


def app_version() -> str:
    """The code version: git commit, plus '+changes' if files were edited
    since (so results can be tied to the exact code that produced them)."""
    global _APP_VERSION
    if _APP_VERSION is None:
        import subprocess
        try:
            run = lambda *a: subprocess.run(["git", *a], cwd=APP_DIR, capture_output=True, text=True,
                                            timeout=3, creationflags=0x08000000).stdout.strip()
            _APP_VERSION = (run("rev-parse", "--short", "HEAD") or "unknown") + \
                           ("+changes" if run("status", "--porcelain", "--untracked-files=no") else "")
        except Exception:
            _APP_VERSION = "unknown"
    return _APP_VERSION


_NVML_LOCK = threading.Lock()
_NVML_MODULE = None
_NVML_HANDLE = None
_NVML_ERROR = None


def _nvml_device():
    """Return the cached NVML module and first device handle, initializing once."""
    global _NVML_MODULE, _NVML_HANDLE, _NVML_ERROR
    if _NVML_HANDLE is not None:
        return _NVML_MODULE, _NVML_HANDLE
    if _NVML_ERROR is not None:
        raise RuntimeError(_NVML_ERROR)
    with _NVML_LOCK:
        if _NVML_HANDLE is not None:
            return _NVML_MODULE, _NVML_HANDLE
        if _NVML_ERROR is not None:
            raise RuntimeError(_NVML_ERROR)
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML_MODULE = pynvml
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            _NVML_ERROR = type(exc).__name__
            raise
    return _NVML_MODULE, _NVML_HANDLE


def gpu_snapshot() -> dict:
    """Graphics card load, memory, and which OTHER programs use it right now.
    Another program on the GPU slows Whisper (31 Oct: encoder 140 -> 450 ms)."""
    out = {}
    try:
        pynvml, h = _nvml_device()
        import psutil
        util, mem = pynvml.nvmlDeviceGetUtilizationRates(h), pynvml.nvmlDeviceGetMemoryInfo(h)
        out.update(gpu_load_pct=util.gpu, gpu_mem_used_mb=mem.used // 2**20,
                   gpu_mem_total_mb=mem.total // 2**20,
                   gpu_mem_free_mb=mem.free // 2**20,
                   gpu_temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                   gpu_sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        try:
            own = [p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                   if p.pid == os.getpid() and p.usedGpuMemory]
            if own:
                out["gpu_mem_app_mb"] = own[0] // 2**20
        except Exception:
            pass
        # Programs actually computing on the GPU (% of its time, recent
        # samples). The plain process lists name every app with a window.
        busy = {}
        try:
            for s in pynvml.nvmlDeviceGetProcessUtilization(h, 0):
                if s.pid != os.getpid() and s.smUtil >= 2:
                    try:
                        name = psutil.Process(s.pid).name()
                    except Exception:
                        name = f"pid {s.pid}"
                    busy[name] = max(busy.get(name, 0), s.smUtil)
        except Exception:
            pass
        out["gpu_other_programs_pct"] = dict(sorted(busy.items(), key=lambda kv: -kv[1])[:8])
    except Exception as e:
        out["gpu_error"] = type(e).__name__
    return out


def log_session_context(session_id, **fields):
    """Log what the session ran in (code version, GPU load and other GPU
    programs, apps playing sound), in the background (git and Windows audio
    queries take ~0.1-0.3 s). Explains results that differ between sessions."""
    def work():
        try:
            log_event(logging.INFO, "session_context", session_id=session_id,
                      app_version=app_version(), python=sys.version.split()[0],
                      playing_apps=[a.get("app") for a in audio_sessions_snapshot()
                                    if (a.get("peak") or 0) > 0.001],
                      **gpu_snapshot(), **fields)
        except Exception as e:
            log_exception("session_context_failed", e)
    threading.Thread(target=work, daemon=True, name="session-context").start()


# ── Resource measurements (logged every STATS_INTERVAL_SEC as resource_stats) ─
def collect_resource_stats(app) -> dict:
    """Measure CPU, RAM and GPU use. CPU figures are "% of one core" (can
    exceed 100) and also "% of the whole machine". `app` keeps the previous
    readings between calls (CPU use is time used since the last reading)."""
    fields = {}
    try:
        import psutil
        if getattr(app, "_psutil_process", None) is None:
            app._psutil_process = psutil.Process()
            app._psutil_process.cpu_percent(None)
            app._thread_times, app._other_times = {}, {}
        proc, logical = app._psutil_process, psutil.cpu_count() or 1
        cpu = proc.cpu_percent(None)
        mem, vm = proc.memory_info(), psutil.virtual_memory()
        fields.update(
            process_cpu_pct=round(cpu, 1),                       # % of one core
            process_cpu_machine_pct=round(cpu / logical, 1),     # % of all cores
            rss_mb=round(mem.rss / 2**20),
            private_mb=round(getattr(mem, "private", 0) / 2**20),
            page_faults=getattr(mem, "num_page_faults", None),
            mem_available_mb=round(vm.available / 2**20),
            mem_used_pct=vm.percent,
            threads=proc.num_threads())
        # Whole-machine CPU from our own previous reading (psutil.cpu_percent
        # keeps one shared "since last call" counter that the Diagnostics page
        # also uses every second, which made this read 0).
        times = psutil.cpu_times()
        busy, total = sum(times) - times.idle, sum(times)
        last = getattr(app, "_sys_cpu_last", None)
        app._sys_cpu_last = (busy, total)
        if last and total > last[1]:
            fields["system_cpu_pct"] = round((busy - last[0]) / (total - last[1]) * 100, 1)
        try:
            freq = psutil.cpu_freq()
            fields["cpu_freq_mhz"] = round(freq.current) if freq else None
        except Exception:
            pass
        # Which of our threads use the CPU (named Python threads; others are
        # library threads: CUDA, CTranslate2, Qt, audio driver).
        names = {t.native_id: t.name for t in threading.enumerate()}
        interval = max(1e-3, time.monotonic() - getattr(app, "_res_last_t", time.monotonic() - 10))
        app._res_last_t = time.monotonic()
        per_thread, now_times = {}, {}
        for th in proc.threads():
            total = th.user_time + th.system_time
            now_times[th.id] = total
            used = total - app._thread_times.get(th.id, total)
            if used > 0:
                key = names.get(th.id, "native")
                per_thread[key] = per_thread.get(key, 0) + used
        app._thread_times = now_times
        fields["thread_cpu_pct"] = {k: round(v / interval * 100, 1) for k, v in
                                    sorted(per_thread.items(), key=lambda kv: -kv[1])[:6]}
        # The busiest other apps: shows whether Teams/Telemost were short of CPU.
        others, now_other = [], {}
        process_probe_started = time.perf_counter()
        for p in psutil.process_iter(["pid", "name", "cpu_times"]):
            ct = p.info.get("cpu_times")
            if ct is None or p.info["pid"] in (0, proc.pid):
                continue
            total = ct.user + ct.system
            now_other[p.info["pid"]] = total
            used = total - app._other_times.get(p.info["pid"], total)
            if used > 0:
                others.append((used, p.info["name"]))
        app._other_times = now_other
        others.sort(reverse=True)
        fields["top_other_apps"] = {n: round(u / interval * 100, 1) for u, n in others[:5]}
        fields["process_inventory_ms"] = round(
            (time.perf_counter() - process_probe_started) * 1000, 1)
    except Exception:
        pass
    if WHISPER_DEVICE == "cuda":
        gpu_probe_started = time.perf_counter()
        try:
            fields.update(gpu_snapshot())
            fields["gpu_util_pct"] = fields.pop("gpu_load_pct")
        except Exception:
            pass
        finally:
            fields["gpu_snapshot_ms"] = round(
                (time.perf_counter() - gpu_probe_started) * 1000, 1)
    return fields


def resource_warnings(fields) -> list:
    """Plain-language problems in a resource_stats reading."""
    problems = []
    if (fields.get("mem_available_mb") or 1e9) < LOW_RAM_MB:
        problems.append(f"low RAM: {fields['mem_available_mb']} MB free")
    if (fields.get("gpu_mem_free_mb") or 1e9) < LOW_GPU_MEM_MB:
        problems.append(f"low GPU memory: {fields['gpu_mem_free_mb']} MB free")
    if (fields.get("system_cpu_pct") or 0) > HIGH_SYSTEM_CPU_PCT:
        problems.append(f"CPU busy: {fields['system_cpu_pct']} % of the whole machine")
    return problems

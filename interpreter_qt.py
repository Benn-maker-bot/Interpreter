"""
Voice Interpreter — Fluent (Windows 11 style) interface
=======================================================

This is a second window for the same program, built with PySide6 (the
official Python version of the Qt toolkit) and PyQt-Fluent-Widgets
(Windows 11 "Fluent" design controls for Qt). Start it with:

    .venv\\Scripts\\python interpreter_qt.py

The original window (interpreter.py, built with customtkinter) still works.
Both share all of the real work: recording, speech detection, Whisper,
translation, logging. Only the interface differs.

HOW THIS FILE REUSES interpreter.py
-----------------------------------
interpreter.py's InterpreterApp mixes two kinds of methods:

  * logic (start/stop a listening session, handle results, the watchdog,
    load models, log timings) — about twenty methods. These are BORROWED
    unchanged: e.g. `_start_session = InterpreterApp._start_session` below.
    In Python a method is just a function whose first argument is `self`,
    so it works on any object that has the attributes it uses.
  * interface (build widgets, draw meters, show text) — these are written
    again here with Qt widgets.

The borrowed logic expects a few customtkinter things, so this file gives
it small stand-ins ("adapters"):

  QVar        acts like a Tk variable (get()/set()) but reads and writes a
              Qt control, e.g. a combo box or slider
  ComboShim   lets borrowed code replace a combo box's items
              (configure(values=[...]))
  after()     Tk's "run this later" timer, implemented with QTimer
  TextShim    a label that accepts configure(text=...)

Qt basics used here: a QWidget is anything on screen; layouts (QVBoxLayout =
stacked vertically, QHBoxLayout = side by side, QGridLayout = rows and
columns) position widgets; "signals" (e.g. a button's `clicked`) are
connected to functions that run when the event happens; QTimer runs a
function later or repeatedly on the main thread.
"""

import logging
import os
import sys
import threading
import time
from collections import deque

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QTextCharFormat, QTextCursor, QGuiApplication
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QSizePolicy, QTableWidgetItem, QVBoxLayout, QWidget,
                               QHeaderView, QAbstractItemView)
from qfluentwidgets import (BodyLabel, CaptionLabel, CheckBox, ComboBox, FluentIcon as FIF,
                            FluentWindow, HeaderCardWidget, InfoBar, InfoBarPosition,
                            MessageBoxBase, PrimaryPushButton, PushButton, SimpleCardWidget,
                            Slider, SmoothScrollArea, StrongBodyLabel, SubtitleLabel,
                            TableWidget, TextEdit, Theme, TitleLabel, TogglePushButton,
                            setCustomStyleSheet, setTheme, setThemeColor)

import interpreter as core
from interpreter import (InterpreterApp, LANGUAGES, continuation_join, trailing_marks_length, MODELS, PREVIEW_CHOICES, Speaker,
                         PerfWindow, get_audio_devices, find_device_by_keywords,
                         find_best_vb_input, find_best_friend_input, find_best_output_device,
                         load_settings, log_event, log_exception, vad_threshold_for,
                         NVIDIA_KEYWORDS, DEFAULT_SENS, TRANSLATION_QUEUE_MAX,
                         MAX_TEXT_HISTORY_CHARS, WATCHDOG_SEC, STATS_INTERVAL_SEC,
                         WHISPER_DEVICE, COMPUTE_TYPE, NUM_WORKERS, _DEFAULT_MODEL)

# Status colours: green = fine, amber = worth a look, red = problem
OK, WARN, BAD, MUTED = "#2ECC71", "#F1C40F", "#E74C3C", "#9AA0A6"


# ─────────────────────────────────────────────────────────────────────────────
# Adapters that let interpreter.py's borrowed logic drive Qt widgets
# ─────────────────────────────────────────────────────────────────────────────

class QVar:
    """Acts like a Tk StringVar/DoubleVar/BooleanVar (get() and set()) but is
    backed by a Qt control. set() never fires the control's change signal,
    matching Tk, where setting a variable doesn't run the widget's command."""
    def __init__(self, getter, setter, widget=None):
        self._get, self._set, self._widget = getter, setter, widget

    def get(self):
        return self._get()

    def set(self, value):
        if self._widget is not None:
            self._widget.blockSignals(True)
        try:
            self._set(value)
        finally:
            if self._widget is not None:
                self._widget.blockSignals(False)


class ComboShim:
    """Lets borrowed code call device_menu.configure(values=[...]) to replace a
    combo box's items, keeping the current choice if it is still there."""
    def __init__(self, combo):
        self.combo = combo

    def configure(self, values=None, **_):
        if values is None:
            return
        current = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(values))
        if current in values:
            self.combo.setCurrentText(current)
        self.combo.blockSignals(False)


class TextShim:
    """A label that accepts configure(text=...) like a customtkinter label."""
    def __init__(self, label):
        self.label = label

    def configure(self, text=None, **_):
        if text is not None:
            self.label.setText(text)


class IssueHandler(logging.Handler):
    """Copies every warning/error written to the error log into a list the
    Diagnostics page shows ("Recent issues"). Handlers run on whichever thread
    logged, so it only appends to a thread-safe deque."""
    def __init__(self, sink):
        super().__init__(level=logging.WARNING)
        self.sink = sink

    def emit(self, record):
        fields = getattr(record, "fields", {}) or {}
        event = getattr(record, "event", record.getMessage())
        try:
            help_ = core.explain_issue(event, fields, record.exc_info)
        except Exception:
            help_ = {"title": event, "meaning": "", "action": "",
                     "technical": str(fields.get("exception_message", "")), "where": "",
                     "details": ", ".join(f"{k}={v}" for k, v in fields.items())[:500]}
        # (time, level, event, short detail, plain-language explanation)
        self.sink.append((time.strftime("%H:%M:%S"), record.levelname, event,
                          help_["technical"][:160], help_))


# ─────────────────────────────────────────────────────────────────────────────
# Meters, painted with QPainter (Qt scales them for the display automatically)
# ─────────────────────────────────────────────────────────────────────────────

class LevelMeter(QWidget):
    """Segmented LED level meter in dBFS (-60..0): green below -18, amber to
    -6, red above; instant rise, 20 dB/s fall, 1.5 s peak hold. Same
    behaviour as DbMeter in interpreter.py, drawn with Qt."""
    FLOOR_DB, SEG_DB, FALL_DB_PER_SEC, HOLD_SEC = -60.0, 1.5, 20.0, 1.5
    TICKS = (-60, -48, -36, -24, -18, -12, -6, -3, 0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._rms = self._peak = self.FLOOR_DB   # level and peak shown, dB
        self._peak_t = 0.0                        # when the peak was reached
        self._last_t = time.monotonic()           # time of the previous update

    def update_level(self, rms_db, peak_db):
        now = time.monotonic()
        fall = self.FALL_DB_PER_SEC * (now - self._last_t)
        self._last_t = now
        self._rms = max(rms_db, self._rms - fall)
        if peak_db >= self._peak:
            self._peak, self._peak_t = peak_db, now
        elif now - self._peak_t > self.HOLD_SEC:
            self._peak = max(peak_db, self._peak - fall)
        self.update()   # ask Qt to repaint (calls paintEvent)

    def reset(self):
        self._rms = self._peak = self.FLOOR_DB
        self.update()

    @staticmethod
    def _colour(db, lit):
        if db < -18:
            return QColor("#2ECC71" if lit else "#123d24")
        if db < -6:
            return QColor("#F1C40F" if lit else "#4a3f0c")
        return QColor("#E74C3C" if lit else "#4a1712")

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        n = int(-self.FLOOR_DB / self.SEG_DB)
        seg_w = (w - 8) / n
        hold = int((self._peak - self.FLOOR_DB) / self.SEG_DB) - 1
        for i in range(n):
            db = self.FLOOR_DB + (i + 1) * self.SEG_DB
            lit = db <= self._rms or i == hold
            p.fillRect(QRectF(4 + i * seg_w, 2, seg_w - 2, 16), self._colour(db - 0.01, lit))
        p.setPen(QColor("#9AA0A6"))
        f = p.font()
        f.setPointSizeF(7.5)
        p.setFont(f)
        for t in self.TICKS:
            x = 4 + (t - self.FLOOR_DB) / -self.FLOOR_DB * (w - 8)
            p.drawText(QRectF(min(max(x - 14, 0), w - 28), 19, 28, 14), Qt.AlignCenter, str(t))


class SpeechMeter(QWidget):
    """How much the sound resembles speech (0-100 %), with a white line at
    the threshold set by Speech sensitivity. Green while it counts as speech.
    Sensitivity moves the line; it never changes the volume."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._shown, self._threshold, self._speaking = 0.0, 0.5, False

    def update_score(self, prob, threshold, speaking):
        self._shown = max(prob, self._shown * 0.8)   # rise at once, fall gently
        self._threshold, self._speaking = threshold, speaking
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        x0, x1 = 4, self.width() - 4
        span = x1 - x0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2a2d2e"))
        p.drawRoundedRect(QRectF(x0, 2, span, 16), 4, 4)
        p.setBrush(QColor(OK if self._speaking else "#2980B9"))
        p.drawRoundedRect(QRectF(x0, 2, span * self._shown, 16), 4, 4)
        xt = x0 + span * self._threshold
        p.setPen(QColor("white"))
        p.drawLine(int(xt), 0, int(xt), 20)
        p.setPen(QColor("#9AA0A6"))
        f = p.font()
        f.setPointSizeF(7.5)
        p.setFont(f)
        for t in (0, 25, 50, 75, 100):
            x = x0 + span * t / 100
            p.drawText(QRectF(min(max(x - 16, 0), self.width() - 32), 19, 32, 14),
                       Qt.AlignCenter, f"{t}%")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MetricCard(HeaderCardWidget):
    """A titled card with rows of "name   value", where each value can be
    coloured green/amber/red to show whether it is healthy."""
    def __init__(self, title, rows, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        self._values = {}                     # row name -> value label
        for r, (key, tip) in enumerate(rows):
            name = CaptionLabel(key)
            name.setToolTip(tip)
            value = BodyLabel("—")
            value.setToolTip(tip)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self._values[key] = value
        grid.setColumnStretch(1, 1)
        self.viewLayout.addLayout(grid)

    def set(self, key, text, colour=None):
        label = self._values[key]
        label.setText(str(text))
        set_colour(label, colour)


def set_colour(label, colour=None):
    """Colour a Fluent label, or restore the theme's normal text colour.
    (Clearing a style sheet instead would also drop the theme colour and
    leave near-black text on the dark background.)"""
    if colour:
        label.setTextColor(QColor(colour), QColor(colour))   # (light theme, dark theme)
    else:
        label.setTextColor(QColor(0, 0, 0), QColor(255, 255, 255))


def judge(value, ok_below, warn_below):
    """Colour for a number where smaller is better."""
    if value is None:
        return MUTED
    return OK if value < ok_below else WARN if value < warn_below else BAD


def pct(values, q):
    v = sorted(x for x in values if x is not None)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


class ReportDialog(MessageBoxBase):
    """A pop-up showing the text of a log_tools.py report."""
    def __init__(self, title, text, parent=None):
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel(title))
        box = TextEdit()
        box.setReadOnly(True)
        box.setFont(QFont("Consolas", 9))
        box.setPlainText(text)
        box.setMinimumSize(900, 520)
        self.viewLayout.addWidget(box)
        self.yesButton.setText("Close")
        self.cancelButton.hide()


# ─────────────────────────────────────────────────────────────────────────────
# The main window
# ─────────────────────────────────────────────────────────────────────────────

class QtInterpreterApp(FluentWindow):
    """The Fluent main window: an Interpreter page and a Diagnostics page in a
    side navigation bar. The session logic is borrowed from InterpreterApp."""

    # ── Borrowed logic (see the file's opening comment) ──────────────────
    _preload_model            = InterpreterApp._preload_model
    _reload_model_if_changed  = InterpreterApp._reload_model_if_changed
    _restart_session          = InterpreterApp._restart_session
    _start_session            = InterpreterApp._start_session
    _retry_start_session      = InterpreterApp._retry_start_session
    _stop_session             = InterpreterApp._stop_session
    _configure_speaker        = InterpreterApp._configure_speaker
    _on_preview_choice        = InterpreterApp._on_preview_choice
    _ensure_preview_model     = InterpreterApp._ensure_preview_model
    _ensure_music_scene       = InterpreterApp._ensure_music_scene
    _translator_for           = InterpreterApp._translator_for
    _preview_model            = InterpreterApp._preview_model
    _watchdog                 = InterpreterApp._watchdog
    _reconnect                = InterpreterApp._reconnect
    _log_resources            = InterpreterApp._log_resources
    _poll_queue               = InterpreterApp._poll_queue
    _update_metric            = InterpreterApp._update_metric
    _commit_translation       = InterpreterApp._commit_translation
    _on_mode_btn              = InterpreterApp._on_mode_btn
    _refresh_devices          = InterpreterApp._refresh_devices
    _on_manual_device_change  = InterpreterApp._on_manual_device_change
    _on_level                 = InterpreterApp._on_level
    _open_path                = InterpreterApp._open_path
    _export_log_summary       = InterpreterApp._export_log_summary
    _avg                      = staticmethod(InterpreterApp._avg)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Interpreter")
        self._timers, self._timer_id = {}, 0     # for after()/after_cancel()

        # ── state (the same attributes InterpreterApp sets up) ───────────
        import uuid, queue
        self.session_id = str(uuid.uuid4())
        self.result_queue = queue.Queue()        # background threads -> window
        self.phrase_queue = queue.Queue()
        self._perf_stats = {"transcriptions": 0, "translations": 0,
                            "transcription_ms": deque(maxlen=200),
                            "translation_ms": deque(maxlen=200),
                            "lock_wait_ms": deque(maxlen=200)}
        self._capture_thread = None
        self._loopback_out = None
        self._workers, self._translation_workers = [], []
        self._translation_queue = queue.Queue(maxsize=TRANSLATION_QUEUE_MAX)
        self._model_loader = self._whisper_model = None
        self.is_listening = self._muted = False
        self._run_id = 0
        self._traces, self._first_preview = {}, {}
        self._ui_stats = PerfWindow()
        self._last_poll_t = self._diag_psutil_process = None
        self._preview_loader, self._preview_enabled = None, False
        self._previews_off = False
        self._mt_loaders = {}
        self._initialised = False
        self._active_mode = None
        self._start_retry_after_id = None
        self._tts_on = False
        self.speaker = Speaker()
        self.speaker.start()
        self._live, self._last_preview_words = {}, {}
        self._mute_evt = self._partial_slot = None
        self._sens_log_after = None

        # ── diagnostics data (new) ───────────────────────────────────────
        self._started_at = time.monotonic()     # when this window opened
        self._listening_since = None
        self._phrases = deque(maxlen=50)        # recent finished phrases (timings, text)
        self._levels = deque(maxlen=320)        # recent block levels (~10 s)
        self._speech_flags = deque(maxlen=320)  # recent speech yes/no per block
        self._peak_recent = -100.0
        self._issues = deque(maxlen=40)         # recent warnings/errors
        self._issue_count_seen = 0
        self._issue_groups = {}                 # event -> count, times, explanation
        self._watchdog_restarts = 0
        self._nvml = None                       # GPU monitoring handle
        core.ERROR_LOG.addHandler(IssueHandler(self._issues))

        # ── audio devices and default mode settings ──────────────────────
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        in_labels = [lbl for _, lbl in self._input_devices]
        out_labels = [lbl for _, lbl in self._output_devices]
        nvidia = find_device_by_keywords(self._input_devices, NVIDIA_KEYWORDS)
        vb = find_best_vb_input(self._input_devices)
        friend = find_best_friend_input(self._input_devices)
        fiio = find_best_output_device(self._output_devices)
        self._mode_state = [[nvidia or (in_labels[0] if in_labels else ""), "English", "Russian"],
                            [friend or (in_labels[0] if in_labels else ""), "Russian", "English"]]
        for i, lbl in enumerate(load_settings().get("mode_devices", [])[:2]):
            if lbl in self._input_map:
                self._mode_state[i][0] = lbl

        # ── pages ────────────────────────────────────────────────────────
        self.interp_page = self._build_interpreter_page(in_labels, out_labels, fiio, vb, friend)
        self.diag_page = self._build_diagnostics_page()
        self.addSubInterface(self.interp_page, FIF.CHAT, "Interpreter")
        self.addSubInterface(self.diag_page, FIF.DEVELOPER_TOOLS, "Diagnostics")
        self._place_window()

        log_event(logging.INFO, "app_initialized", session_id=self.session_id, gui="qt",
                  whisper_device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE,
                  num_workers=NUM_WORKERS, input_device_count=len(in_labels),
                  output_device_count=len(out_labels))
        # Start in the mode used last time (Friend, for most calls)
        self._apply_mode(int(load_settings().get("mode", 0)) % 2, switch_source=True)
        self._preload_model()
        self._ensure_preview_model()
        self._initialised = True
        self._start_session()
        self.after(15, self._poll_queue)
        self.after(WATCHDOG_SEC * 1000, self._watchdog)
        self.after(STATS_INTERVAL_SEC * 1000, self._log_resources)
        self.after(1000, self._refresh_diagnostics)

    # ── Tk-style timer (used by the borrowed logic) ──────────────────────

    def after(self, ms, fn):
        """Run fn once after ms milliseconds, on the main thread. Returns an
        id that after_cancel() accepts."""
        self._timer_id += 1
        tid = self._timer_id
        timer = QTimer(self)
        timer.setSingleShot(True)

        def fire():
            self._timers.pop(tid, None)
            fn()
        timer.timeout.connect(fire)
        self._timers[tid] = timer
        timer.start(int(ms))
        return tid

    def after_cancel(self, tid):
        timer = self._timers.pop(tid, None)
        if timer is not None:
            timer.stop()

    # ── Window placement ─────────────────────────────────────────────────

    def _place_window(self):
        """Size the window to fit the usable screen area (taskbar excluded)
        and centre it. Qt handles display scaling itself, so sizes here are
        in scaled ("logical") pixels."""
        area = QGuiApplication.primaryScreen().availableGeometry()
        w, h = min(1180, int(area.width() * 0.95)), min(880, int(area.height() * 0.92))
        self.resize(w, h)
        self.move(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2)
        log_event(logging.INFO, "window_placed", gui="qt", size_logical=[w, h],
                  screen_area=[area.x(), area.y(), area.width(), area.height()])

    # ── Interpreter page ─────────────────────────────────────────────────

    def _build_interpreter_page(self, in_labels, out_labels, fiio, vb, friend):
        page = QWidget()
        page.setObjectName("interpreterPage")   # FluentWindow needs a unique name
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 16, 24, 12)
        root.setSpacing(10)

        # Header: title, GPU/CPU badge, status pill
        header = QHBoxLayout()
        header.addWidget(TitleLabel("Voice Interpreter"))
        self.hw_badge = self._pill(" GPU " if WHISPER_DEVICE == "cuda" else " CPU ",
                                   OK if WHISPER_DEVICE == "cuda" else "#2980B9")
        self.status_badge = self._pill(" IDLE ", WARN)
        header.addWidget(self.hw_badge)
        header.addWidget(self.status_badge)
        header.addStretch(1)
        header.addWidget(CaptionLabel("Whisper AI · local translation"))
        # Fold the settings away to give the transcript more room (caption
        # apps devote most of the window to the text); remembered.
        self.settings_btn = TogglePushButton()
        self.settings_btn.setText("Settings")
        self.settings_btn.setIcon(FIF.SETTING)
        self.settings_btn.setToolTip("Show or hide the settings (more room for the transcript)")
        self.settings_btn.clicked.connect(self._toggle_settings)
        header.addWidget(self.settings_btn)
        root.addLayout(header)

        # Mode buttons and actions
        actions = QHBoxLayout()
        self._mode_btns = []
        for i, (icon, text) in enumerate([(FIF.MICROPHONE, "My Voice   English → Russian"),
                                          (FIF.HEADPHONE, "Friend   Russian → English")]):
            b = TogglePushButton()
            b.setText(text)
            b.setIcon(icon)
            b.setMinimumHeight(40)
            b.clicked.connect(lambda _=False, m=i: self._on_mode_btn(m))
            actions.addWidget(b, 2)
            self._mode_btns.append(b)
        self.mute_btn = TogglePushButton()
        self.mute_btn.setText("Mute")
        self.mute_btn.setIcon(FIF.MUTE)
        self.mute_btn.setMinimumHeight(40)
        self.mute_btn.clicked.connect(self._toggle_mute)
        self.tts_btn = TogglePushButton()
        self.tts_btn.setText("Speak")
        self.tts_btn.setIcon(FIF.VOLUME)
        self.tts_btn.setMinimumHeight(40)
        self.tts_btn.clicked.connect(self._toggle_tts)
        clear_btn = PushButton()
        clear_btn.setText("Clear")
        clear_btn.setIcon(FIF.BROOM)
        clear_btn.setMinimumHeight(40)
        clear_btn.clicked.connect(self._clear)
        for b in (self.mute_btn, self.tts_btn, clear_btn):
            actions.addWidget(b, 1)
        root.addLayout(actions)

        # Settings card
        card = SimpleCardWidget()
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        def label(text, tip):
            l = BodyLabel(text)
            l.setToolTip(tip)
            return l

        self.device_combo = ComboBox()
        self.device_combo.addItems(in_labels)
        self.device_combo.setToolTip("Where to listen. 🔊 = what an output device "
                                     "(e.g. the FiiO in a call) is playing.")
        refresh = PushButton()
        refresh.setIcon(FIF.SYNC)
        refresh.setToolTip("Re-scan audio devices")
        refresh.clicked.connect(self._refresh_devices)
        self.out_combo = ComboBox()
        self.out_combo.addItems(out_labels)
        self.out_combo.setCurrentText(fiio or (out_labels[0] if out_labels else ""))
        self.replay_check = CheckBox("Hear input through output (VB-Audio cables only)")
        self.replay_check.setChecked(bool(vb) and "[loopback]" not in (friend or "").lower())
        # Source list starts with "Auto-detect" (the language of each phrase
        # is detected); a target must be a real language.
        targets = sorted(core.TARGET_LANGUAGES)
        self.src_combo, self.tgt_combo = ComboBox(), ComboBox()
        self.src_combo.addItems(["Auto-detect"] + targets)
        self.tgt_combo.addItems(targets)
        self.model_combo = ComboBox()
        self.model_combo.addItems(list(MODELS))
        saved = load_settings()
        self.model_combo.setCurrentText(saved.get("model") if saved.get("model") in MODELS
                                        else _DEFAULT_MODEL)
        self.sens_slider = Slider(Qt.Horizontal)
        self.sens_slider.setRange(0, 100)
        self.sens_slider.setValue(int(round(float(saved.get("sensitivity", DEFAULT_SENS)) * 100)))
        self.sens_value = BodyLabel(f"{self.sens_slider.value()}%")
        self.preview_combo = ComboBox()
        self.preview_combo.addItems(list(PREVIEW_CHOICES))
        choice = saved.get("preview_model")
        if choice not in PREVIEW_CHOICES:
            choice = "Tiny (instant)" if saved.get("fast_preview", False) else "Small (balanced)"
        self.preview_combo.setCurrentText(choice)

        grid.addWidget(label("Audio input", "Device to listen to"), 0, 0)
        grid.addWidget(self.device_combo, 0, 1, 1, 3)
        grid.addWidget(refresh, 0, 4)
        grid.addWidget(label("Audio output", "Device for Speak and replay"), 1, 0)
        grid.addWidget(self.out_combo, 1, 1, 1, 2)
        grid.addWidget(self.replay_check, 1, 3, 1, 2)
        grid.addWidget(label("Translate", "Language spoken → language shown"), 2, 0)
        lang_row = QHBoxLayout()
        lang_row.addWidget(self.src_combo)
        lang_row.addWidget(BodyLabel("→"))
        lang_row.addWidget(self.tgt_combo)
        # Automatic language detection (see LANG_SWITCH_PROB in interpreter.py)
        self.auto_lang_check = CheckBox("Detect other languages")
        self.auto_lang_check.setToolTip(
            "If a phrase is clearly in another language (e.g. a German song while\n"
            "listening for Russian), recognise it in that language, label it\n"
            "[German] and translate it. Off: label it but don't translate it.")
        self.auto_lang_check.setChecked(bool(load_settings().get("auto_language", True)))
        self.auto_lang_check.stateChanged.connect(self._on_auto_language)
        lang_row.addSpacing(16)
        lang_row.addWidget(self.auto_lang_check)
        lang_row.addStretch(1)
        grid.addLayout(lang_row, 2, 1, 1, 4)
        grid.addWidget(label("Model", "Whisper model for the final text"), 3, 0)
        grid.addWidget(self.model_combo, 3, 1)
        grid.addWidget(label("Previews", "Model for the live grey text"), 3, 2)
        grid.addWidget(self.preview_combo, 3, 3, 1, 2)
        grid.addWidget(label("Speech sensitivity",
                             "Higher = also catch quieter/less clear speech. Changes what "
                             "counts as speech, not the volume."), 4, 0)
        sens_row = QHBoxLayout()
        sens_row.addWidget(self.sens_slider, 1)
        sens_row.addWidget(self.sens_value)
        grid.addLayout(sens_row, 4, 1, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        root.addWidget(card)
        self.settings_card = card
        show = bool(load_settings().get("show_settings", True))
        card.setVisible(show)
        self.settings_btn.setChecked(show)

        # Meters
        meters = QGridLayout()
        meters.setHorizontalSpacing(10)
        meters.addWidget(CaptionLabel("Input level"), 0, 0)
        self.meter = LevelMeter()
        meters.addWidget(self.meter, 0, 1)
        self.level_label = CaptionLabel("—")
        self.level_label.setFont(QFont("Consolas", 10))
        self.level_label.setMinimumWidth(210)
        meters.addWidget(self.level_label, 0, 2)
        meters.addWidget(CaptionLabel("Speech"), 1, 0)
        self.speech_meter = SpeechMeter()
        meters.addWidget(self.speech_meter, 1, 1)
        self.speech_label = CaptionLabel("—")
        self.speech_label.setFont(QFont("Consolas", 10))
        meters.addWidget(self.speech_label, 1, 2)
        meters.setColumnStretch(1, 1)
        root.addLayout(meters)

        # Text panes
        panes = QGridLayout()
        self.heard_label = StrongBodyLabel("Heard")
        self.translated_label = StrongBodyLabel("Translation")
        # Text size (caption guidance: readers must control the size)
        # The label shares the row with the buttons: when both sat in the same
        # grid cell, the label covered the buttons and swallowed their clicks.
        size_row = QHBoxLayout()
        size_row.addWidget(self.translated_label)
        size_row.addStretch(1)
        size_row.addWidget(CaptionLabel("Text size"))
        for text, step in (("A−", -1), ("A+", 1)):
            b = PushButton()
            b.setText(text)
            b.setFixedWidth(52)
            b.setToolTip("Smaller / larger transcript text")
            b.clicked.connect(lambda _=False, d=step: self._change_text_size(d))
            size_row.addWidget(b)
        panes.addLayout(size_row, 0, 1)
        self.heard_box, self.translated_box = TextEdit(), TextEdit()
        self._text_size = int(load_settings().get("text_size", 13))
        for box in (self.heard_box, self.translated_box):
            box.setReadOnly(True)
        self._apply_text_size()
        panes.addWidget(self.heard_label, 0, 0)
        panes.addWidget(self.heard_box, 1, 0)
        panes.addWidget(self.translated_box, 1, 1)
        root.addLayout(panes, 1)

        self.status_label = CaptionLabel("Loading Whisper model…")
        root.addWidget(self.status_label)

        # ── adapters for the borrowed logic ──────────────────────────────
        self.device_var = QVar(self.device_combo.currentText, self.device_combo.setCurrentText,
                               self.device_combo)
        self.out_device_var = QVar(self.out_combo.currentText, self.out_combo.setCurrentText,
                                   self.out_combo)
        self.loopback_var = QVar(self.replay_check.isChecked, self.replay_check.setChecked,
                                 self.replay_check)
        self.src_var = QVar(self.src_combo.currentText, self.src_combo.setCurrentText,
                            self.src_combo)
        self.tgt_var = QVar(self.tgt_combo.currentText, self.tgt_combo.setCurrentText,
                            self.tgt_combo)
        self.model_var = QVar(self.model_combo.currentText, self.model_combo.setCurrentText,
                              self.model_combo)
        self.preview_choice_var = QVar(self.preview_combo.currentText,
                                       self.preview_combo.setCurrentText, self.preview_combo)
        self.sensitivity_var = QVar(lambda: self.sens_slider.value() / 100,
                                    lambda v: self.sens_slider.setValue(int(round(v * 100))),
                                    self.sens_slider)
        self.status_var = QVar(self.status_label.text, self.status_label.setText)
        self.device_menu = ComboShim(self.device_combo)
        self.out_device_menu = ComboShim(self.out_combo)
        self.level_text = TextShim(self.level_label)

        # ── control events ───────────────────────────────────────────────
        self.device_combo.currentTextChanged.connect(lambda _: self._on_manual_device_change())
        self.out_combo.currentTextChanged.connect(lambda _: self._on_manual_device_change())
        self.replay_check.stateChanged.connect(lambda _: self._on_manual_device_change())
        self.src_combo.currentTextChanged.connect(lambda _: self._on_manual_lang_change())
        self.tgt_combo.currentTextChanged.connect(lambda _: self._on_manual_lang_change())
        self.model_combo.currentTextChanged.connect(lambda _: self._reload_model_if_changed())
        self.preview_combo.currentTextChanged.connect(lambda _: self._on_preview_choice())
        self.sens_slider.valueChanged.connect(lambda v: self._on_sensitivity(v / 100))
        return page

    def _on_auto_language(self, *_):
        """'Detect other languages' ticked/unticked: save and apply now
        (the workers read the flag for each phrase; no restart needed)."""
        on = self.auto_lang_check.isChecked()
        core.save_settings(auto_language=on)
        for worker in getattr(self, "_workers", []):
            worker.auto_language = on
        log_event(logging.INFO, "auto_language_changed", enabled=on)

    def _toggle_settings(self):
        """Settings button: show or hide the settings card (remembered)."""
        show = not self.settings_card.isVisible()
        self.settings_card.setVisible(show)
        self.settings_btn.setChecked(show)
        core.save_settings(show_settings=show)

    def _apply_text_size(self):
        """The Fluent theme's stylesheet fixes the text box font size, which
        overrides setFont(), so the size goes in the widget's own custom
        stylesheet (kept when the theme changes)."""
        qss = f"TextEdit {{ font: {self._text_size}pt 'Segoe UI'; }}"
        for box in (self.heard_box, self.translated_box):
            setCustomStyleSheet(box, qss, qss)

    def _change_text_size(self, step):
        """A−/A+ buttons: change the transcript font size (9-28 pt), saved."""
        self._text_size = max(9, min(28, self._text_size + step))
        self._apply_text_size()
        core.save_settings(text_size=self._text_size)

    @staticmethod
    def _at_bottom(box):
        """True if the box is scrolled to the end. New text only scrolls the
        view if so: someone re-reading earlier text isn't yanked back down."""
        bar = box.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    @staticmethod
    def _follow(box):
        box.verticalScrollBar().setValue(box.verticalScrollBar().maximum())

    @staticmethod
    def _pill(text, colour):
        """A small rounded coloured label (status badge)."""
        l = QLabel(text)
        l.setStyleSheet(f"background:{colour}; color:white; border-radius:6px; "
                        f"padding:3px 8px; font-weight:600;")
        return l

    # ── Interface versions of InterpreterApp's widget methods ────────────

    def _set_live_status(self, text, color):
        self.status_badge.setText(f" {text} ")
        self.status_badge.setStyleSheet(f"background:{color}; color:white; border-radius:6px; "
                                        f"padding:3px 8px; font-weight:600;")
        if text == "LISTENING":
            self._listening_since = time.monotonic()

    def _apply_mode(self, mode_idx, switch_source=False):
        """Switch to mode 0 (My Voice) or 1 (Friend): its device, languages,
        highlighted button, then restart listening."""
        if self.is_listening:
            self._stop_session(keep_mode=True)
        self._active_mode = mode_idx
        core.save_settings(mode=mode_idx)
        device_lbl, src, tgt = self._mode_state[mode_idx]
        if switch_source and device_lbl:
            self.device_var.set(device_lbl)
        self.src_var.set(src)
        self.tgt_var.set(tgt)
        self._update_pane_titles()
        for i, b in enumerate(self._mode_btns):
            b.setChecked(i == mode_idx)
        if self._initialised:
            self._restart_session()

    def _on_manual_lang_change(self):
        self._update_pane_titles()
        self._restart_session()

    def _update_pane_titles(self):
        self.heard_label.setText(f"Heard in {self.src_var.get()}")
        self.translated_label.setText(f"Translation → {self.tgt_var.get()}")

    def _toggle_mute(self):
        self._muted = not self._muted
        self.mute_btn.setChecked(self._muted)
        if self._muted:
            self._stop_session()
            self.mute_btn.setText("Unmute")
            self.mute_btn.setIcon(FIF.MICROPHONE)
            self._set_live_status("MUTED", BAD)
            self.status_var.set("Muted — not listening.")
        else:
            self.mute_btn.setText("Mute")
            self.mute_btn.setIcon(FIF.MUTE)
            self._start_session()
        log_event(logging.INFO, "mute_toggled", muted=self._muted)

    def _toggle_tts(self):
        if self.speaker.error and not self.speaker.is_alive():
            self.status_var.set(f"Text-to-speech unavailable: {self.speaker.error}")
            self.tts_btn.setChecked(False)
            return
        self._tts_on = not self._tts_on
        self.tts_btn.setChecked(self._tts_on)
        self.tts_btn.setText("Speak: On" if self._tts_on else "Speak")
        if self._tts_on:
            self._configure_speaker()
        else:
            self.speaker.stop_speech()
        log_event(logging.INFO, "tts_toggled", enabled=self._tts_on)

    def _on_sensitivity(self, v):
        v = round(float(v), 2)
        self.sens_value.setText(f"{round(v * 100)}%")
        if self._capture_thread:
            self._capture_thread.set_sensitivity(v)
        if self._sens_log_after is not None:
            self.after_cancel(self._sens_log_after)

        def settled():
            self._sens_log_after = None
            core.save_settings(sensitivity=v)
            log_event(logging.INFO, "sensitivity_changed", value=v, run_id=self._run_id,
                      session_id=self.session_id, vad_threshold=round(vad_threshold_for(v), 2),
                      speech_score_now=round(self.speech_meter._shown, 2))
        self._sens_log_after = self.after(500, settled)

    def _update_meter(self, rms_db, peak_db, speaking, prob=0.0, threshold=0.5):
        self.meter.update_level(rms_db, peak_db)
        self.speech_meter.update_score(prob, threshold, speaking)
        self.level_label.setText("  -∞ dBFS  (silent)" if rms_db <= -99 else f"{rms_db:6.1f} dBFS")
        self.speech_label.setText(f"{prob * 100:3.0f}% / needs {threshold * 100:.0f}%  "
                                  f"{'SPEECH' if speaking else 'no speech'}")
        set_colour(self.speech_label, OK if speaking else None)
        if rms_db > -99:
            self._levels.append(rms_db)
            self._peak_recent = max(peak_db, self._peak_recent - 0.05)
        self._speech_flags.append(bool(speaking))

    def _clear(self):
        self.heard_box.clear()
        self.translated_box.clear()
        self._live = {}
        self._last_preview_words = {}
        self.status_var.set("Cleared.")
        log_event(logging.INFO, "ui_cleared")

    # ── Live preview and final text in the panes ─────────────────────────
    # Same behaviour as interpreter.py: the preview line is replaced each
    # time; the words two previews agree on are shown normally, the rest
    # grey; the final text replaces the preview.

    def _fmt(self, grey):
        f = QTextCharFormat()
        f.setForeground(QColor("#8a8f94") if grey else self.heard_box.palette().text().color())
        return f

    def _set_live(self, box, seq, text):
        follow = self._at_bottom(box)
        start, live_seq = self._live.get(box, (None, None))
        if live_seq is not None and seq < live_seq:
            return
        cur = box.textCursor()
        if start is not None:
            cur.setPosition(start)
            cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cur.removeSelectedText()
        else:
            cur.movePosition(QTextCursor.End)
            start = cur.position()
        prev_seq, prev_words = self._last_preview_words.get(box, (None, []))
        words = text.split()
        stable = 0
        if prev_seq == seq:
            while (stable < min(len(words), len(prev_words))
                   and words[stable] == prev_words[stable]):
                stable += 1
        self._last_preview_words[box] = (seq, words)
        prefix = "\n" if start > 0 else ""
        stable_text = " ".join(words[:stable])
        cur.insertText(prefix + "… " + stable_text + (" " if stable_text else ""), self._fmt(False))
        cur.insertText(" ".join(words[stable:]), self._fmt(True))
        self._live[box] = (start, seq)
        if follow:
            self._follow(box)

    def _commit(self, box, seq, text):
        follow = self._at_bottom(box)
        start, live_seq = self._live.get(box, (None, None))
        cur = QTextCursor(box.document())
        if start is not None and live_seq <= seq:
            cur.setPosition(start)
            cur.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            cur.removeSelectedText()
            self._live.pop(box, None)
        if not text:
            return
        # Insert at the end of the finished text: before a newer phrase's
        # preview if one is showing, otherwise at the very end.
        live = self._live.get(box)
        anchor = live[0] if live else box.document().characterCount() - 1
        before = box.document().toPlainText()[:anchor]
        joined, piece = continuation_join(before, text)
        if joined:      # "...они..." + "...включаться" -> one line
            cut, insert = trailing_marks_length(before), " " + piece
        else:
            cut, insert = 0, ("\n" if before.strip() else "") + text
        pos = anchor - cut
        cur.setPosition(pos)
        cur.setPosition(anchor, QTextCursor.KeepAnchor)
        cur.insertText(insert, self._fmt(False))     # replaces the "..." if any
        if live:
            self._live[box] = (pos + len(insert), live[1])
        else:
            self._trim(box)
        if follow:
            self._follow(box)

    @staticmethod
    def _trim(box):
        """Keep at most MAX_TEXT_HISTORY_CHARS characters (oldest removed)."""
        doc = box.document()
        extra = doc.characterCount() - MAX_TEXT_HISTORY_CHARS
        if extra > 0:
            cur = QTextCursor(doc)
            cur.setPosition(0)
            cur.setPosition(extra, QTextCursor.KeepAnchor)
            cur.removeSelectedText()

    def _log_displayed(self, seq, trace, online):
        """Log the screen timings (borrowed), and keep them for Diagnostics."""
        first_preview = self._first_preview.get(seq)
        InterpreterApp._log_displayed(self, seq, trace, online)
        now = time.monotonic()
        self._phrases.append({
            "time": time.strftime("%H:%M:%S"),
            "pause_to_text": (now - trace["t_end"]) * 1000,
            "first_preview": first_preview,
            # recognition time: for an early-transcribed phrase the work was
            # done during the pause (early_ms), so process_ms would read 0
            "process": trace.get("early_ms") if trace.get("early_used") else trace.get("process_ms"),
            "mt": trace.get("mt_ms") or None,
            "early": bool(trace.get("early_used")),
            "model": trace.get("model"),
            "doubtful": bool(trace.get("suspicious")),
            "retry_used": bool((trace.get("retry") or {}).get("used")),
            "context_reset": bool(trace.get("context_reset")),
            "onset": trace.get("onset_delay_ms"),
            "text": trace.get("text") or "",
        })

    # ── Diagnostics page ─────────────────────────────────────────────────

    def _build_diagnostics_page(self):
        page = SmoothScrollArea()
        page.setObjectName("diagnosticsPage")
        page.setWidgetResizable(True)
        page.setStyleSheet("QScrollArea{border:none; background:transparent}")
        body = QWidget()
        body.setStyleSheet("background:transparent")
        page.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(TitleLabel("Diagnostics"))
        top.addStretch(1)
        for icon, text, fn in [(FIF.FOLDER, "App folder", lambda: self._open_path(core.APP_DIR)),
                               (FIF.DOCUMENT, "Main log", lambda: self._open_path(core.LOG_PATH)),
                               (FIF.DOCUMENT, "Error log", lambda: self._open_path(core.ERROR_LOG_PATH)),
                               (FIF.COPY, "Copy", self._copy_diagnostics)]:
            b = PushButton()
            b.setText(text)
            b.setIcon(icon)
            b.clicked.connect(fn)
            top.addWidget(b)
        report = PrimaryPushButton()
        report.setText("Speed report")
        report.setIcon(FIF.SPEED_HIGH)
        report.setToolTip("Run log_tools.py report for this run of the app")
        report.clicked.connect(self._show_speed_report)
        top.addWidget(report)
        lay.addLayout(top)

        self.health_card = MetricCard("Health", [
            ("Overall", "A summary of everything below"),
            ("Listening for", "Time since listening (re)started"),
            ("Main model", "Whisper model for final text, and its number format"),
            ("Preview model", "Model making the live grey previews"),
            ("Translation", "Local Opus-MT translator state for this language pair"),
        ])
        lay.addWidget(self.health_card)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.speed_card = MetricCard("Speed (last 50 phrases: median / slowest 5 %)", [
            ("Pause → final text", "From when the pause is confirmed (0.3 s after the last word) "
                                   "to the final text on screen. Good: < 400 ms"),
            ("Speech → first preview", "Start of speech to the first live text. Good: < 900 ms"),
            ("Recognition time", "GPU time to transcribe a phrase (done during the pause when "
                                 "the early transcription was used)"),
            ("Translation time", "Local Opus-MT time per phrase"),
            ("Early transcription used", "Phrases whose result was ready when the pause ended"),
        ])
        self.audio_card = MetricCard("Audio", [
            ("Level now", "Current input level in dBFS"),
            ("Level (last 10 s)", "Median level; speech is best around -35 to -12 dBFS"),
            ("Speech share (10 s)", "How much of the audio counted as speech"),
            ("Onset delay", "How late speech starts were detected (median). > 150 ms can "
                            "clip first words: raise sensitivity"),
            ("Sensitivity", "Slider value and the speech threshold it sets"),
            ("Loudest peak", "Close to 0 dBFS means the input is clipping"),
        ])
        self.quality_card = MetricCard("Recognition quality (last 50 phrases)", [
            ("Doubtful results", "Results that looked wrong and were re-checked. Good: < 10 %"),
            ("Retries used", "Re-checks that produced a better result"),
            ("Empty phrases", "Phrases with no words (noise or music triggered them)"),
            ("Context resets", "Times the previous-text context was cleared after a pause"),
        ])
        self.resource_card = MetricCard("Computer", [
            ("CPU (this app)", "100 % = one CPU core fully busy"),
            ("Memory (this app)", "RAM used by the app"),
            ("GPU load", "How busy the graphics card is"),
            ("GPU memory", "Graphics memory used (all programs)"),
            ("GPU temperature", "Above 80 °C the card may slow down"),
            ("GPU clock", "Current graphics card speed"),
        ])
        self.pipeline_card = MetricCard("Pipeline", [
            ("Waiting phrases", "Finished phrases not yet transcribed (should be 0)"),
            ("Waiting translations", "Phrases waiting for online translation"),
            ("Threads", "Capture / transcription / speech threads alive"),
            ("Watchdog restarts", "Times listening was restarted after a failure"),
            ("Phrases this run", "Finished phrases since the app opened"),
        ])
        for i, c in enumerate([self.speed_card, self.audio_card, self.quality_card,
                               self.resource_card]):
            grid.addWidget(c, i // 2, i % 2)
        grid.addWidget(self.pipeline_card, 2, 0)
        self.issues_card = HeaderCardWidget()
        self.issues_card.setTitle("Recent issues (warnings and errors)")
        self.issues_text = TextEdit()
        self.issues_text.setReadOnly(True)
        self.issues_text.setFont(QFont("Segoe UI", 10))
        self.issues_text.setMinimumHeight(150)
        self.issues_text.setPlainText("No warnings or errors since the app opened.")
        self.issues_card.viewLayout.addWidget(self.issues_text)
        grid.addWidget(self.issues_card, 2, 1)
        lay.addLayout(grid)

        phrases_card = HeaderCardWidget()
        phrases_card.setTitle("Recent phrases")
        self.phrase_table = TableWidget()
        cols = ["Time", "Model", "Pause→text", "1st preview", "Recognition", "Early", "Text"]
        self.phrase_table.setColumnCount(len(cols))
        self.phrase_table.setHorizontalHeaderLabels(cols)
        self.phrase_table.verticalHeader().hide()
        self.phrase_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.phrase_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.phrase_table.setMinimumHeight(260)
        phrases_card.viewLayout.addWidget(self.phrase_table)
        lay.addWidget(phrases_card)
        return page

    def _refresh_diagnostics(self):
        try:
            self._refresh_diagnostics_once()
        except Exception as e:
            log_exception("diagnostics_refresh_failed", e)
        self.after(1000, self._refresh_diagnostics)

    def _refresh_diagnostics_once(self):
        """Recompute every figure on the Diagnostics page (once a second)."""
        ph = list(self._phrases)
        p2t = [p["pause_to_text"] for p in ph]
        fp = [p["first_preview"] for p in ph]
        rec = [p["process"] for p in ph]
        mt = [p["mt"] for p in ph]
        fmt_ms = lambda a, b: "—" if a is None else f"{a:.0f} ms / {b:.0f} ms"

        s = self.speed_card
        m50, m95 = pct(p2t, .5), pct(p2t, .95)
        s.set("Pause → final text", fmt_ms(m50, m95), judge(m50, 400, 800))
        m50, m95 = pct(fp, .5), pct(fp, .95)
        s.set("Speech → first preview", fmt_ms(m50, m95), judge(m50, 900, 1500))
        m50, m95 = pct(rec, .5), pct(rec, .95)
        s.set("Recognition time", fmt_ms(m50, m95), judge(m50, 300, 600))
        m50, m95 = pct(mt, .5), pct(mt, .95)
        s.set("Translation time", fmt_ms(m50, m95), judge(m50, 100, 300))
        early = 100 * sum(p["early"] for p in ph) / len(ph) if ph else None
        s.set("Early transcription used", "—" if early is None else f"{early:.0f} %",
              MUTED if early is None else OK if early >= 50 else WARN)

        a = self.audio_card
        lv = list(self._levels)
        now_lv = lv[-1] if lv else None
        a.set("Level now", "—" if now_lv is None else f"{now_lv:.1f} dBFS")
        med = pct(lv, .5)
        a.set("Level (last 10 s)", "—" if med is None else f"{med:.1f} dBFS",
              MUTED if med is None else OK if -40 <= med <= -10 else WARN)
        sf = list(self._speech_flags)
        a.set("Speech share (10 s)", f"{100 * sum(sf) / len(sf):.0f} %" if sf else "—")
        onset = pct([p["onset"] for p in ph], .5)
        a.set("Onset delay", "—" if onset is None else f"{onset:.0f} ms", judge(onset, 150, 400))
        sens = self.sensitivity_var.get()
        a.set("Sensitivity", f"{sens * 100:.0f} %  (threshold {vad_threshold_for(sens):.2f})")
        a.set("Loudest peak", f"{self._peak_recent:.1f} dBFS" if self._peak_recent > -99 else "—",
              BAD if self._peak_recent > -1 else OK)

        q = self.quality_card
        n = len(ph)
        share = lambda key: 100 * sum(p[key] for p in ph) / n if n else None
        d = share("doubtful")
        q.set("Doubtful results", "—" if d is None else f"{d:.0f} %", judge(d, 10, 25))
        q.set("Retries used", sum(p["retry_used"] for p in ph))
        empty = 100 * sum(not p["text"] for p in ph) / n if n else None
        q.set("Empty phrases", "—" if empty is None else f"{empty:.0f} %", judge(empty, 10, 30))
        q.set("Context resets", sum(p["context_reset"] for p in ph))

        self._refresh_resources()

        pl = self.pipeline_card
        pq, tq = self.phrase_queue.qsize(), self._translation_queue.qsize()
        pl.set("Waiting phrases", pq, OK if pq < 2 else WARN if pq < 5 else BAD)
        pl.set("Waiting translations", tq, OK if tq < 3 else WARN)
        cap = self._capture_thread
        alive = lambda t: "✔" if t is not None and t.is_alive() else "✖"
        threads = (f"capture {alive(cap)}   transcription "
                   f"{alive(self._workers[0]) if self._workers else '✖'}   speech {alive(self.speaker)}")
        pl.set("Threads", threads, OK if (not self.is_listening or "✖" not in threads) else BAD)
        pl.set("Watchdog restarts", self._watchdog_restarts,
               OK if self._watchdog_restarts == 0 else WARN)
        pl.set("Phrases this run", self._perf_stats["transcriptions"])

        self._refresh_health(m50_ok=pct(p2t, .5))
        self._refresh_issues()
        self._refresh_phrase_table(ph)

    def _refresh_health(self, m50_ok):
        h = self.health_card
        loader = self._model_loader
        if self._muted:
            overall, colour = "Muted", WARN
        elif loader is not None and not loader.ready.is_set():
            overall, colour = "Loading the Whisper model…", WARN
        elif loader is not None and loader.error:
            overall, colour = f"Model error: {loader.error}", BAD
        elif self.phrase_queue.qsize() >= 5:
            overall, colour = "Falling behind: phrases are piling up", BAD
        elif self.is_listening:
            overall, colour = "Healthy — listening", OK
        else:
            overall, colour = "Idle", WARN
        # Count kinds of problem, not repeats (one error on every phrase is one problem)
        problems = [g["help"]["title"] for g in self._issue_groups.values()
                    if g["level"] in ("ERROR", "CRITICAL")]
        if problems and colour == OK:
            overall = (f"Listening — problem: {problems[0]}" if len(problems) == 1
                       else f"Listening — {len(problems)} problems, see Recent issues below")
            colour = WARN
        h.set("Overall", overall, colour)
        since = self._listening_since
        h.set("Listening for", time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - since))
              if since and self.is_listening else "—")
        h.set("Main model", f"{MODELS.get(self.model_var.get())}  "
                            f"({getattr(loader, 'active_compute_type', COMPUTE_TYPE)} on "
                            f"{getattr(loader, 'active_device', WHISPER_DEVICE)})" if loader else "—")
        pl = self._preview_loader
        if not self._preview_enabled:
            h.set("Preview model", "same as main model")
        elif pl is not None and pl.ready.is_set():
            h.set("Preview model", pl.model_size, OK)
        else:
            h.set("Preview model", "loading…", WARN)
        src = LANGUAGES.get(self.src_var.get(), ("", ""))[0]
        tgt = LANGUAGES.get(self.tgt_var.get(), ("", ""))[0]
        mt = self._mt_loaders.get((src, tgt))
        if mt is None:
            h.set("Translation", "online services (no local model for this pair)", WARN)
        elif not mt.ready.is_set():
            h.set("Translation", "local Opus-MT loading…", WARN)
        elif mt.translator is None:
            h.set("Translation", "local Opus-MT failed to load — using fallback", BAD)
        else:
            h.set("Translation", f"local Opus-MT {src}→{tgt}", OK)

    def _refresh_resources(self):
        r = self.resource_card
        try:
            import psutil
            if self._diag_psutil_process is None:
                self._diag_psutil_process = psutil.Process()
                self._diag_psutil_process.cpu_percent(None)
            cpu = self._diag_psutil_process.cpu_percent(None)
            # psutil counts 100 % per core, so a busy app can show over 100 %
            # (up to 100 % x the number of logical cores). Show both forms.
            whole = cpu / (psutil.cpu_count() or 1)
            r.set("CPU (this app)", f"{whole:.1f} % of PC  ({cpu:.0f} % of one core)",
                  judge(whole, 10, 25))
            r.set("Memory (this app)", f"{self._diag_psutil_process.memory_info().rss / 2**20:.0f} MB")
        except Exception:
            r.set("CPU (this app)", "psutil not installed", MUTED)
        if WHISPER_DEVICE != "cuda":
            r.set("GPU load", "not used (CPU mode)", MUTED)
            return
        try:
            import pynvml
            if self._nvml is None:
                pynvml.nvmlInit()
                self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
            h = self._nvml
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, 0)
            r.set("GPU load", f"{util} %", judge(util, 85, 97))
            r.set("GPU memory", f"{mem.used / 2**30:.1f} / {mem.total / 2**30:.1f} GB",
                  judge(mem.used / mem.total * 100, 85, 95))
            r.set("GPU temperature", f"{temp} °C", judge(temp, 80, 88))
            r.set("GPU clock", f"{pynvml.nvmlDeviceGetClockInfo(h, 1)} MHz")
        except Exception:
            r.set("GPU load", "nvidia-ml-py not available", MUTED)

    def _refresh_issues(self):
        """Group the recent problems by kind for the Diagnostics page, and pop
        up a plain-language note the FIRST time each kind of error happens.
        Repeats only raise its count (one error on every phrase used to stack
        five identical pop-ups over the settings)."""
        issues = list(self._issues)
        self._watchdog_restarts = sum(1 for i in issues if i[2] == "watchdog_restart")
        if len(issues) == self._issue_count_seen:
            return
        new = issues[self._issue_count_seen:] if len(issues) > self._issue_count_seen else []
        self._issue_count_seen = len(issues)
        for t, lvl, ev, detail, help_ in new:
            g = self._issue_groups.setdefault(ev, {"first": t, "count": 0, "level": lvl,
                                                   "help": help_})
            g["count"] += 1
            g["last"], g["help"] = t, help_
            if lvl in ("ERROR", "CRITICAL"):
                g["level"] = lvl
        self._render_issue_groups()
        for t, lvl, ev, detail, help_ in new:
            if lvl not in ("ERROR", "CRITICAL") and ev not in self.POPUP_WARNINGS:
                continue
            g = self._issue_groups[ev]
            if g.get("popped"):
                continue            # already shown: the count on Diagnostics goes up
            g["popped"] = True
            text = f"{help_['meaning']}\n\nWhat to do: {help_['action']}"
            bar = (InfoBar.error if lvl in ("ERROR", "CRITICAL") else InfoBar.warning)(
                title=help_["title"], content=text, orient=Qt.Vertical, isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT, duration=15000, parent=self)
            details = PushButton()
            details.setText("Details")
            details.clicked.connect(lambda _=False, b=bar: (self.switchTo(self.diag_page), b.close()))
            bar.addWidget(details)
            bar.setToolTip(f"{help_['technical']}\n{help_['where']}\n"
                           "Full details: Diagnostics → Recent issues")

    # Warnings worth a pop-up too (all errors pop up)
    POPUP_WARNINGS = ("music_processing_disabled", "resource_warning")

    def _render_issue_groups(self):
        """Recent issues card: one entry per kind of problem, newest first,
        with a plain explanation, what to do, and the technical details."""
        if not self._issue_groups:
            self.issues_text.setPlainText("No warnings or errors since the app opened.")
            return
        blocks = []
        for ev, g in sorted(self._issue_groups.items(), key=lambda kv: kv[1]["last"], reverse=True):
            h = g["help"]
            times = (f"{g['count']} times, {g['first']}–{g['last']}" if g["count"] > 1
                     else f"at {g['last']}")
            lines = [f"{'ERROR' if g['level'] in ('ERROR', 'CRITICAL') else 'Warning'}: "
                     f"{h['title']}  ({times})",
                     f"  What it means: {h['meaning']}",
                     f"  What to do:    {h['action']}"]
            if h.get("technical"):
                lines.append(f"  Technical:     {h['technical']}")
            if h.get("details"):
                pad = "\n" + " " * 17
                lines.append("  Details:       " + h["details"].replace("\n", pad))
            if h.get("where"):
                lines.append(f"  Where:         {h['where']}  (log event: {ev})")
            else:
                lines.append(f"  Log event:     {ev}")
            blocks.append("\n".join(lines))
        self.issues_text.setPlainText("\n\n".join(blocks))

    def _refresh_phrase_table(self, ph):
        t = self.phrase_table
        rows = list(reversed(ph))[:20]
        t.setRowCount(len(rows))
        for r, p in enumerate(rows):
            cells = [p["time"], p["model"] or "", f"{p['pause_to_text']:.0f} ms",
                     f"{p['first_preview']:.0f} ms" if p["first_preview"] else "—",
                     f"{p['process']:.0f} ms" if p["process"] is not None else "—",
                     "yes" if p["early"] else "no", p["text"]]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if c == 2:
                    item.setForeground(QColor(judge(p["pause_to_text"], 400, 800)))
                t.setItem(r, c, item)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)

    def _copy_diagnostics(self):
        """Copy every Diagnostics figure as text (e.g. to paste in a message)."""
        lines = [f"Voice Interpreter diagnostics  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                 f"version {core.APP_VERSION}"]
        for card in (self.health_card, self.speed_card, self.audio_card, self.quality_card,
                     self.resource_card, self.pipeline_card):
            lines.append(f"\n[{card.headerLabel.text()}]")
            lines += [f"  {k:<26} {v.text()}" for k, v in card._values.items()]
        lines.append("\n[Recent issues]\n" + self.issues_text.toPlainText())
        QGuiApplication.clipboard().setText("\n".join(lines))
        InfoBar.success(title="Copied", content="Diagnostics copied to the clipboard",
                        orient=Qt.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP_RIGHT, duration=2500, parent=self)

    def _show_speed_report(self):
        """Run log_tools.py's speed report for this run of the app, in a
        background thread so the window doesn't freeze, then show it."""
        started = time.strftime("%Y-%m-%dT%H:%M",
                                time.localtime(time.time() - (time.monotonic() - self._started_at)))
        self.status_var.set("Building speed report…")

        def work():
            try:
                import log_tools
                text = log_tools.build_report(started)
            except Exception as e:
                text = f"Report failed: {e}"
            self.result_queue.put(("status", "Speed report ready."))
            self._pending_report = text
        self._pending_report = None
        threading.Thread(target=work, daemon=True, name="speed-report").start()

        def show_when_ready():
            if self._pending_report is None:
                self.after(200, show_when_ready)
                return
            ReportDialog("Speed report (this run)", self._pending_report, self).exec()
        self.after(200, show_when_ready)

    # ── Closing ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        try:
            if self.is_listening:
                self._stop_session()
            log_event(logging.INFO, "app_closing", session_id=self.session_id)
            self.speaker.shutdown()
        finally:
            super().closeEvent(event)


def main():
    core.install_global_exception_hooks()
    log_event(logging.INFO, "app_main_start", gui="qt")
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    setThemeColor("#0078D4")          # Windows accent blue
    window = QtInterpreterApp()
    window.show()
    app.exec()
    log_event(logging.INFO, "app_main_exit", gui="qt")


if __name__ == "__main__":
    # Running this file directly: same single-copy rule as interpreter.py
    # Keep the lock in a variable: if it were discarded, Python would close
    # the file at once and release the lock.
    instance_lock = core.acquire_single_instance()
    core.disable_power_throttling()
    if instance_lock is None:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, "Voice Interpreter is already running.",
                                         "Voice Interpreter", 0x40)
        sys.exit(0)
    main()

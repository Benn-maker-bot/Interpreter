"""
Simulator window - watch and hear a simulation run (simulate.py) live
=====================================================================

Runs the same simulation as simulate.py through the real pipeline, in a
Fluent window like the interpreter's. Three kinds of test input:

  Synthetic clips   make_testdata.py: one language per clip, exact text
  Recorded clips    fetch_testdata.py: real people (FLEURS) and songs
  Live (Windows)    whatever is playing on an output device right now: a
                    radio stream in the browser, Spotify, a video … (there
                    is no correct text, so only languages and speed are
                    measured)

The window shows:
  * the clip now playing with its correct text, and a progress bar
  * the input level and speech meters
  * the recognised text and its translation, with grey live previews
    (the same display code as the interpreter window)
  * a results table that fills in as each clip (or live phrase) finishes,
    with running totals
  * clips are played to an output device you choose (the FiiO by default),
    in step with what the pipeline hears, each brought to the same level
    first ("Level each clip")

Problems are explained on the Issues page (and the first of each kind pops
up), and everything is logged to the app's normal log files.

Start it with:
    .venv\\Scripts\\python simulator_qt.py

Results are saved to testdata/sim_results/ exactly as from simulate.py.
"""

import argparse
import logging
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import pyaudio as real_pyaudio            # kept: simulate.build() swaps interpreter's for a fake

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QHeaderView, QSizePolicy, QSplitter,
                               QAbstractItemView, QTableWidgetItem, QVBoxLayout, QWidget)
from qfluentwidgets import setThemeColor  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from qfluentwidgets import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
                            BodyLabel, CaptionLabel, CheckBox, ComboBox, FluentIcon as FIF,
                            FluentWindow, LineEdit, PrimaryPushButton, ProgressBar, PushButton,
                            SimpleCardWidget, Slider, SmoothScrollArea, StrongBodyLabel,
                            TableWidget, TextEdit, Theme, TogglePushButton, setTheme)

import interpreter as core
import simulate as S
from interpreter import (LANGUAGES, MODELS, get_audio_devices, find_best_output_device,
                         load_settings, save_settings, log_event, log_exception)
from interpreter_qt import (QtInterpreterApp, IssueHandler, LevelMeter, SpeechMeter, set_colour,
                            OK, WARN, BAD, MUTED)

RATE = S.RATE
STALL_SEC = 15          # no level readings for this long while playing = pipeline stuck
SCORE_DELAY_SEC = 4     # a clip is scored this long after it ends (its last text has arrived)

# Plain-language help for the simulator's own problems (pop-ups, Issues page).
# {name} is filled in from the logged details.
core.ISSUE_HELP.update({
    "simulation_failed": (
        "The simulation stopped with an error",
        "Something went wrong while loading the models or playing the clips, so the run "
        "ended early. The 'Technical' line says what failed and 'Where' the line of code.",
        "If it names a model: choose another Whisper model. If it names a file: run "
        "make_testdata.py / fetch_testdata.py again. If it names a device (live mode): "
        "check the device is still connected. Otherwise copy the details for the developer."),
    "sim_monitor_failed": (
        "Can't play the clips on the chosen output",
        "The simulation carries on, but you won't hear it on this device. Windows refused "
        "to open it for playback (it may be unplugged, disabled, or in exclusive use by "
        "another program).",
        "Check the device is connected and enabled in Windows Sound settings, or choose "
        "another output under 'Listen on'."),
    "sim_pipeline_stalled": (
        "The interpreter stopped responding",
        f"No audio level readings arrived for {STALL_SEC} s while clips were playing (at "
        "clip {clip}). The part that reads the audio (capture thread) may be stuck.",
        "Press Stop and start again. If it repeats, copy the details for the developer."),
    "sim_clip_missing": (
        "Test clips are missing",
        "The test data list names {count} audio files that aren't on disk, e.g. {first}.",
        "Run make_testdata.py (synthetic) or fetch_testdata.py (recorded) again."),
    "sim_wrong_language": (
        "A clip was heard in the wrong language",
        "Clip {clip} ({file}) is in '{lang}' but was recognised as '{detected}'. This is a "
        "test result, not a fault of the app: Whisper confused the language (close "
        "languages such as Russian/Ukrainian, or a clip with music, are typical).",
        "Nothing to fix in the simulator. If it happens often for one language, choose that "
        "language as Source instead of Auto-detect."),
    "sim_online_translation_failed": (
        "Online translation failed for a phrase",
        "The online translator (internet) didn't answer for phrase {seq} in '{lang}'.",
        "Check the internet connection, or untick 'Online translation' to use only the "
        "local translators."),
    "sim_gui_update_failed": (
        "The simulator window failed to update once",
        "Showing a result in the window raised an error; the simulation itself carries on.",
        "If it repeats, copy the details (they include the line of code) for the developer."),
    "sim_wrong_python": (
        "The simulator is running in the wrong Python",
        "It was started with {python} (Python {python_version}) instead of the project's own "
        "environment ({expected}). Package versions there can differ, so results and speed "
        "may not match the interpreter app.",
        "In VS Code use Run and Debug -> 'Simulator (window)' (F5), or choose the .venv "
        "interpreter: Ctrl+Shift+P -> 'Python: Select Interpreter'. From a terminal: "
        ".venv\\Scripts\\python simulator_qt.py"),
    "sim_live_device_failed": (
        "The live audio source could not be opened",
        "Windows refused to record from '{device}'.",
        "Choose another source; make sure something plays on that output device."),
})

DATA_CHOICES = {"Synthetic clips": "synth", "Recorded clips": "real",
                "Your reported phrases": "feedback", "Live (Windows audio)": "live"}
KINDS = {"Speech and songs": "speech,song", "Speech only": "speech", "Songs only": "song"}
PREVIEWS = {"Small": "small", "Tiny": "tiny", "Main model": "main", "Off": "off"}
PER_LANGUAGE = ["All"] + [str(n) for n in range(1, 11)]
NO_OUTPUT = "(don't play)"


# ─────────────────────────────────────────────────────────────────────────────
# Playing the clips to a real output device
# ─────────────────────────────────────────────────────────────────────────────

class AudioMonitor(threading.Thread):
    """Plays the playlist on an output device in step with the fake input:
    sample n is played when the pipeline is fed sample n. If the device
    falls behind (e.g. it was busy) it skips ahead rather than drift."""
    BLOCK = 320                                   # 20 ms at 16 kHz
    AHEAD = int(0.08 * RATE)                      # keep at most 80 ms queued

    def __init__(self, playlist, device_index, volume):
        """playlist: the clips being simulated; device_index: the output to
        play them on; volume: 0..1 (changed live by the slider)."""
        super().__init__(daemon=True, name="sim-monitor")
        self.playlist, self.device_index = playlist, device_index
        self.volume = volume                      # 0..1, changed live by the slider
        self._stop = threading.Event()
        self.error = None
        self.reported = False

    def stop(self):
        self._stop.set()

    def _open(self, pa):
        """16 kHz mono if the device takes it (MME/DirectSound do), otherwise
        its own rate and channel count (WASAPI insists on those)."""
        info = pa.get_device_info_by_index(self.device_index)
        try:
            pa.is_format_supported(RATE, output_device=self.device_index, output_channels=1,
                                   output_format=real_pyaudio.paInt16)
            rate, channels = RATE, 1
        except ValueError:
            rate = int(info["defaultSampleRate"])
            channels = min(2, int(info["maxOutputChannels"]))
        stream = pa.open(format=real_pyaudio.paInt16, channels=channels, rate=rate, output=True,
                         output_device_index=self.device_index,
                         frames_per_buffer=int(self.BLOCK * rate / RATE))
        log_event(logging.INFO, "sim_monitor_opened", device=info["name"], rate=rate,
                  channels=channels)
        return stream, rate, channels

    def run(self):
        """Thread: open the device and keep writing the playlist's audio,
        always at the sample the simulation is up to (resampled and copied
        to stereo if the device needs it). Errors are logged and kept in
        self.error, which the window shows."""
        import soxr
        pa = real_pyaudio.PyAudio()
        stream = None
        try:
            stream, rate, channels = self._open(pa)
            resampler = soxr.ResampleStream(RATE, rate, 1, dtype="float32") if rate != RATE else None
            samples, pos = self.playlist.samples, None
            while not self._stop.is_set():
                due = int((time.monotonic() - self.playlist.t0) * RATE)
                if pos is None or pos < due - 4 * self.AHEAD:
                    pos = max(0, due)                 # start, or catch up after a stall
                if pos >= len(samples):
                    break
                if pos > due + self.AHEAD:
                    time.sleep(0.005)
                    continue
                x = samples[pos:pos + self.BLOCK].astype(np.float32) / 32768 * self.volume
                pos += self.BLOCK
                if resampler is not None:
                    x = resampler.resample_chunk(x)
                if channels > 1:
                    x = np.repeat(x[:, None], channels, axis=1).reshape(-1)
                stream.write((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            log_exception("sim_monitor_failed", e, device_index=self.device_index)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
            pa.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# The window
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorWindow(FluentWindow):
    """Simulator page (settings, now playing, meters, text, results) and an
    Issues page. Everything that can go wrong is logged and explained."""

    # The interpreter window's live-preview / final-text display, borrowed
    # (it needs self.heard_box, self._live and self._last_preview_words)
    _fmt, _set_live, _commit = QtInterpreterApp._fmt, QtInterpreterApp._set_live, QtInterpreterApp._commit
    _at_bottom = staticmethod(QtInterpreterApp._at_bottom)
    _follow = staticmethod(QtInterpreterApp._follow)
    _trim = staticmethod(QtInterpreterApp._trim)
    # Warnings and errors: grouped on the Issues page, first of each kind pops up
    _refresh_issues = QtInterpreterApp._refresh_issues
    _render_issue_groups = QtInterpreterApp._render_issue_groups
    POPUP_WARNINGS = QtInterpreterApp.POPUP_WARNINGS + ("sim_pipeline_stalled", "sim_clip_missing",
                                                        "sim_wrong_python")

    CLIP_COLUMNS = ["#", "Clip", "Language", "Heard as", "Gain", "WER", "Pause → text", "Music",
                    "Recognised"]
    LIVE_COLUMNS = ["#", "Time", "Heard as", "End → text", "Length", "Music", "Translated by",
                    "Recognised"]

    def __init__(self):
        """Set up the state, list the audio devices (before the simulator swaps
        in its fake sound card), build both pages and start the update timers."""
        super().__init__()
        self.setWindowTitle("Interpreter Simulator")
        import uuid
        self.session_id = f"simgui-{uuid.uuid4().hex[:8]}"
        self._issues = deque(maxlen=200)
        self._issue_count_seen, self._issue_groups, self._watchdog_restarts = 0, {}, 0
        self._issue_handler = IssueHandler(self._issues)
        core.ERROR_LOG.addHandler(self._issue_handler)
        self._events = queue.Queue()             # simulation thread -> window
        self._level = None                       # newest meter reading (capture thread)
        self._last_level_t, self._stall_logged = None, False
        self._stop_evt = None
        self._thread = self._monitor = None
        self.playlist = None
        self._running_live = False
        self._live, self._last_preview_words = {}, {}
        self._reset_run_state()
        # Devices are listed now, before simulate.build() replaces pyaudio
        inputs, self._output_list = get_audio_devices()
        self._outputs = {lbl: idx for idx, lbl in self._output_list}
        self._loopbacks = {lbl: idx for idx, lbl in inputs if "[loopback]" in lbl.lower()}
        self._inputs = {lbl: idx for idx, lbl in inputs}
        self.page = self._build_page()
        self.diag_page = self._build_issues_page()
        self.addSubInterface(self.page, FIF.PLAY, "Simulator")
        self.addSubInterface(self.diag_page, FIF.DEVELOPER_TOOLS, "Issues")
        # Keep the side bar narrow: expanded, it took 350 px from the page
        if hasattr(self.navigationInterface, "setMinimumExpandWidth"):
            self.navigationInterface.setMinimumExpandWidth(100000)
        self._place_window()
        self._on_data_change()
        self._render_issue_groups()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)
        self._slow_timer = QTimer(self)
        self._slow_timer.timeout.connect(self._slow_tick)
        self._slow_timer.start(1000)
        log_event(logging.INFO, "sim_gui_opened", session_id=self.session_id,
                  output_devices=len(self._output_list), loopback_devices=len(self._loopbacks),
                  output=self.out_combo.currentText(), python=sys.executable,
                  python_version=sys.version.split()[0])
        self._check_environment()

    def _check_environment(self):
        """Warn if this isn't the project's .venv Python: another Python may
        have different (or missing) package versions, so results and speed
        can differ from the app's."""
        from pathlib import Path
        venv = (Path(__file__).resolve().parent / ".venv").resolve()
        if Path(sys.prefix).resolve() != venv:
            log_event(logging.WARNING, "sim_wrong_python", python=sys.executable,
                      python_version=sys.version.split()[0], expected=str(venv / "Scripts" / "python.exe"))
            self.status_label.setText(f"⚠ Running in {sys.executable}, not the project's .venv "
                                      "(see Issues).")
            set_colour(self.status_label, WARN)

    def _reset_run_state(self):
        """Forget the previous run's phrases, previews and scores."""
        self._finals, self._previews, self._scored = [], [], 0
        self._shown_clip = -1
        self._run_started = None

    def _place_window(self):
        """Fit the usable screen area (taskbar excluded). On a short screen
        the page scrolls rather than squeezing widgets over each other."""
        area = QGuiApplication.primaryScreen().availableGeometry()
        w, h = min(1500, area.width() - 40), min(1000, area.height() - 40)
        self.resize(w, h)
        self.move(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2)
        log_event(logging.INFO, "window_placed", gui="simulator", size_logical=[w, h],
                  screen_area=[area.x(), area.y(), area.width(), area.height()])

    # ── Layout ───────────────────────────────────────────────────────────

    @staticmethod
    def _combo(items, current=None, min_width=None):
        """A drop-down list with `items`, `current` selected, at least min_width wide."""
        c = ComboBox()
        c.addItems(list(items))
        if current in items:
            c.setCurrentText(current)
        if min_width:
            c.setMinimumWidth(min_width)
        return c

    @staticmethod
    def _field(name, widget, tip=None):
        """A caption above a control, as one small block."""
        box = QVBoxLayout()
        box.setSpacing(4)
        label = CaptionLabel(name)
        box.addWidget(label)
        box.addWidget(widget)
        if tip:
            label.setToolTip(tip)
            widget.setToolTip(tip)
        return box

    def _build_page(self):
        # The page scrolls if the screen is too short for it
        """Build the Simulator page, top to bottom: test settings card, listen
        row, now-playing card, meters, then text panes and results table
        (split by a draggable bar) and the status line."""
        scroll = SmoothScrollArea()
        scroll.setObjectName("simulatorPage")
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent}")
        page = QWidget()
        page.setStyleSheet("background:transparent")
        page.setMinimumHeight(680)               # below this the page scrolls
        scroll.setWidget(page)
        root = QVBoxLayout(page)
        root.setContentsMargins(20, 12, 20, 12)
        root.setSpacing(10)
        settings = load_settings()

        # ── Test settings card ───────────────────────────────────────────
        card = SimpleCardWidget()
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        self.data_combo = self._combo(DATA_CHOICES, settings.get("sim_data", "Synthetic clips"), 190)
        self.data_combo.currentTextChanged.connect(self._on_data_change)
        self.kinds_combo = self._combo(KINDS, min_width=170)
        self.langs_edit = LineEdit()
        self.langs_edit.setPlaceholderText("all  (or e.g. de,fr,ru)")
        self.limit_combo = self._combo(PER_LANGUAGE, "All", 90)
        self.src_combo = self._combo(LANGUAGES, "Auto-detect", 170)
        self.tgt_combo = self._combo([k for k in LANGUAGES if k != "Auto-detect"], "English", 170)
        self._models = dict(MODELS)
        self.model_combo = self._combo(self._models, next(k for k, v in self._models.items()
                                                          if v == "large-v3-turbo"), 230)
        self.preview_combo = self._combo(PREVIEWS, "Small", 110)
        grid.addLayout(self._field("Test data", self.data_combo,
                                   "Synthetic: computer voices, one language per clip, exact text.\n"
                                   "Recorded: real people and songs (FLEURS, Jamendo).\n"
                                   "Live: whatever Windows is playing on an output device."), 0, 0)
        grid.addLayout(self._field("Clips", self.kinds_combo), 0, 1)
        grid.addLayout(self._field("Languages", self.langs_edit,
                                   "Language codes separated by commas; empty = all"), 0, 2)
        grid.addLayout(self._field("Per language", self.limit_combo,
                                   "How many clips of each language and kind"), 0, 3)
        grid.addLayout(self._field("Source", self.src_combo), 1, 0)
        grid.addLayout(self._field("Target", self.tgt_combo), 1, 1)
        grid.addLayout(self._field("Whisper model", self.model_combo), 1, 2)
        grid.addLayout(self._field("Previews", self.preview_combo), 1, 3)
        # Live source (shown instead of Clips / Languages / Per language)
        self.live_combo = self._combo(self._loopbacks or {"(no loopback devices found)": -1}, None, 360)
        self.live_combo.setToolTip("Records everything Windows plays on this output. To test one "
                                   "app only, send it to its own output in Windows Settings → Sound "
                                   "→ Volume mixer.")
        want_live = settings.get("sim_live_source")
        if want_live not in self._loopbacks:
            want_live = self._default_output_loopback()
        if want_live in self._loopbacks:
            self.live_combo.setCurrentText(want_live)
        self.live_apps = CaptionLabel("")
        self.live_apps.setWordWrap(True)
        self.live_box = QWidget()
        lb = QVBoxLayout(self.live_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(4)
        lb.addWidget(CaptionLabel("Live source (what Windows plays on)"))
        lb.addWidget(self.live_combo)
        lb.addWidget(self.live_apps)
        grid.addWidget(self.live_box, 0, 1, 1, 3)
        # Options and buttons
        self.music_check = CheckBox("Music mode")
        self.music_check.setChecked(True)
        self.online_check = CheckBox("Online translation")
        self.online_check.setChecked(bool(settings.get("sim_online", True)))
        self.level_check = CheckBox("Level each clip")
        self.level_check.setChecked(True)
        self.level_check.setToolTip("Automatic gain: bring every clip to the same loudness "
                                    f"({S.LEVEL_TARGET_DBFS} dBFS) before the interpreter hears it")
        opts = QVBoxLayout()
        opts.setSpacing(6)
        for c in (self.music_check, self.online_check, self.level_check):
            opts.addWidget(c)
        grid.addLayout(opts, 0, 4)
        self.start_btn = PrimaryPushButton(FIF.PLAY, "Start")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = PushButton(FIF.PAUSE, "Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        grid.addLayout(buttons, 1, 4, Qt.AlignBottom)
        grid.setColumnStretch(2, 1)
        root.addWidget(card)

        # ── Listening: output device and volume ──────────────────────────
        self.listen_row = QWidget()
        listen = QHBoxLayout(self.listen_row)
        listen.setContentsMargins(0, 0, 0, 0)
        listen.addWidget(StrongBodyLabel("Listen on"))
        outs = [NO_OUTPUT] + list(self._outputs)
        want = settings.get("sim_output") or find_best_output_device(self._output_list)
        self.out_combo = self._combo(outs, want if want in outs else NO_OUTPUT, 340)
        self.out_combo.currentTextChanged.connect(self._on_output_change)
        listen.addWidget(self.out_combo)
        listen.addSpacing(16)
        listen.addWidget(CaptionLabel("Volume"))
        self.volume_slider = Slider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(settings.get("sim_volume", 70)))
        self.volume_slider.setFixedWidth(200)
        self.volume_slider.valueChanged.connect(self._on_volume)
        listen.addWidget(self.volume_slider)
        self.mute_btn = TogglePushButton(FIF.MUTE, "Mute")
        self.mute_btn.toggled.connect(lambda _: self._on_volume())
        listen.addWidget(self.mute_btn)
        listen.addStretch(1)
        root.addWidget(self.listen_row)

        # ── Now playing + progress ───────────────────────────────────────
        now = SimpleCardWidget()
        nv = QVBoxLayout(now)
        nv.setContentsMargins(16, 10, 16, 10)
        top = QHBoxLayout()
        self.clip_title = StrongBodyLabel("Not running")
        top.addWidget(self.clip_title, 1)
        self.time_label = CaptionLabel("")
        top.addWidget(self.time_label)
        nv.addLayout(top)
        self.clip_text = BodyLabel("Press Start to play the test clips through the interpreter.")
        self.clip_text.setWordWrap(True)
        nv.addWidget(self.clip_text)
        self.progress = ProgressBar()
        self.progress.setRange(0, 1000)
        nv.addWidget(self.progress)
        root.addWidget(now)

        # ── Meters ───────────────────────────────────────────────────────
        meters = QGridLayout()
        meters.setVerticalSpacing(4)
        self.meter, self.speech_meter = LevelMeter(), SpeechMeter()
        self.level_label, self.speech_label = CaptionLabel(""), CaptionLabel("")
        for lbl in (self.level_label, self.speech_label):
            lbl.setFont(QFont("Consolas", 10))
            lbl.setFixedWidth(250)
        meters.addWidget(CaptionLabel("Input level"), 0, 0)
        meters.addWidget(self.meter, 0, 1)
        meters.addWidget(self.level_label, 0, 2)
        meters.addWidget(CaptionLabel("Speech"), 1, 0)
        meters.addWidget(self.speech_meter, 1, 1)
        meters.addWidget(self.speech_label, 1, 2)
        meters.setColumnStretch(1, 1)
        root.addLayout(meters)

        # ── Text panes and results, split by a draggable bar ─────────────
        texts = QWidget()
        tg = QGridLayout(texts)
        tg.setContentsMargins(0, 0, 0, 0)
        tg.addWidget(StrongBodyLabel("Recognised"), 0, 0)
        self.translated_label = StrongBodyLabel("Translation")
        tg.addWidget(self.translated_label, 0, 1)
        self.heard_box, self.translated_box = TextEdit(), TextEdit()
        for box in (self.heard_box, self.translated_box):
            box.setReadOnly(True)
            box.setMinimumHeight(90)
        tg.addWidget(self.heard_box, 1, 0)
        tg.addWidget(self.translated_box, 1, 1)
        tg.setRowStretch(1, 1)

        results = QWidget()
        rv = QVBoxLayout(results)
        rv.setContentsMargins(0, 6, 0, 0)
        head = QHBoxLayout()
        head.addWidget(StrongBodyLabel("Results"))
        head.addStretch(1)
        self.totals_label = BodyLabel("")
        head.addWidget(self.totals_label)
        rv.addLayout(head)
        self.table = TableWidget()
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setWordWrap(False)
        self.table.setMinimumHeight(90)
        rv.addWidget(self.table, 1)

        # Text boxes and table take whatever height is left: by default they ask
        # for ~400 px each, which made the page taller than the window
        for w in (self.heard_box, self.translated_box, self.table):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        split = QSplitter(Qt.Vertical)
        split.addWidget(texts)
        split.addWidget(results)
        split.setChildrenCollapsible(False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)             # text and results share the height equally
        root.addWidget(split, 1)

        self.status_label = CaptionLabel("Ready.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        return scroll

    def _default_output_loopback(self):
        """The loopback of Windows' default output: where Spotify, browsers etc.
        play unless told otherwise, so the most likely live source."""
        pa = None
        try:
            pa = real_pyaudio.PyAudio()
            wasapi = pa.get_host_api_info_by_type(real_pyaudio.paWASAPI)
            name = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
            return next((l for l in self._loopbacks if name in l), None)
        except Exception as e:
            log_exception("sim_default_output_unknown", e)
            return None
        finally:
            if pa is not None:
                pa.terminate()

    def _set_columns(self, live):
        """Empty the results table and give it the clip or live-phrase columns."""
        cols = self.LIVE_COLUMNS if live else self.CLIP_COLUMNS
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        header = self.table.horizontalHeader()
        for c in range(len(cols) - 1):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(len(cols) - 1, QHeaderView.Stretch)

    def _build_issues_page(self):
        """Build the Issues page: grouped warnings/errors with explanations,
        and a Copy button."""
        page = QWidget()
        page.setObjectName("issuesPage")
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 12, 20, 12)
        top = QHBoxLayout()
        top.addWidget(StrongBodyLabel("Recent warnings and errors"))
        top.addStretch(1)
        copy = PushButton(FIF.COPY, "Copy")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.issues_text.toPlainText()))
        top.addWidget(copy)
        v.addLayout(top)
        v.addWidget(CaptionLabel(
            "Each problem: what it means in plain words, what to do, the technical error, where in "
            "the code it happened, and the details logged with it (file, device, language …)."))
        self.issues_text = TextEdit()
        self.issues_text.setReadOnly(True)
        self.issues_text.setFont(QFont("Consolas", 10))
        v.addWidget(self.issues_text, 1)
        v.addWidget(CaptionLabel(f"Everything is also in the app's logs: {core.LOG_PATH.name} "
                                 f"and {core.ERROR_LOG_PATH.name} (simulation session ids start "
                                 "with \"sim-\")."))
        return page

    def _on_data_change(self, *_):
        """Show the controls that apply to the chosen test data."""
        live = DATA_CHOICES[self.data_combo.currentText()] == "live"
        for w in (self.kinds_combo, self.langs_edit, self.limit_combo, self.level_check):
            w.setVisible(not live)
        self.live_box.setVisible(live)
        self.listen_row.setVisible(not live)
        self._set_field_labels_visible(not live)
        if live:
            self._refresh_live_apps()
        save_settings(sim_data=self.data_combo.currentText())
        if not (self._thread and self._thread.is_alive()):
            self._set_columns(live)
            self.totals_label.setText("")
            self.clip_title.setText("Not running")
            self.clip_text.setText(
                "Press Start to transcribe what Windows plays on the live source." if live else
                "Press Start to play the test clips through the interpreter.")

    def _set_field_labels_visible(self, visible):
        """The captions above Clips / Languages / Per language."""
        for w in (self.kinds_combo, self.langs_edit, self.limit_combo):
            layout = self._layout_of(w)
            if layout is not None:
                layout.itemAt(0).widget().setVisible(visible)

    def _layout_of(self, widget):
        """The small caption+control layout in the settings card holding widget."""
        grid = self.page.widget().layout().itemAt(0).widget().layout()
        for i in range(grid.count()):
            item = grid.itemAt(i)
            lay = item.layout()
            if lay is not None and lay.indexOf(widget) >= 0:
                return lay
        return None

    def _refresh_live_apps(self):
        """Which apps are playing sound now (shown under the live source)."""
        try:
            apps = core.audio_sessions_snapshot()
        except Exception as e:
            log_exception("sim_audio_sessions_failed", e)
            apps = []
        playing = [a["app"].removesuffix(".exe") for a in apps
                   if "app" in a and (a.get("peak") or 0) > 0.001]
        idle = [a["app"].removesuffix(".exe") for a in apps
                if "app" in a and not (a.get("peak") or 0) > 0.001]
        self.live_apps.setText(
            ("Playing now: " + ", ".join(playing) if playing else "Nothing is playing sound right now.")
            + (f"   (open but silent: {', '.join(idle)})" if idle else ""))

    # ── Starting and stopping ────────────────────────────────────────────

    def _args(self):
        """Turn the window's settings into the options simulate.run() expects
        (the same ones as simulate.py's command line)."""
        data = DATA_CHOICES[self.data_combo.currentText()]
        limit = self.limit_combo.currentText()
        args = argparse.Namespace(
            src=self.src_combo.currentText(), tgt=self.tgt_combo.currentText(),
            kinds=KINDS[self.kinds_combo.currentText()],
            langs=self.langs_edit.text().replace(" ", "").lower(),
            limit=0 if limit == "All" else int(limit),
            gap=1.5, preview=PREVIEWS[self.preview_combo.currentText()],
            model=self._models[self.model_combo.currentText()], sensitivity=0.5,
            no_music=not self.music_check.isChecked(),
            offline=not self.online_check.isChecked(), data=data,
            level_clips=self.level_check.isChecked())
        if data == "live":
            args.device_name = self.live_combo.currentText()
            args.device = self._loopbacks.get(args.device_name, -1)
        return args

    def _start(self):
        """Start button: check the settings, clear the previous results, and run
        the simulation in a background thread that reports back via events."""
        args = self._args()
        live = args.data == "live"
        if not live:
            S.DATA_DIR = S.DATA_SETS[args.data]
        problem = self._check_args(args)
        if problem:
            self.status_label.setText(problem)
            set_colour(self.status_label, BAD)
            log_event(logging.INFO, "sim_gui_start_refused", reason=problem, **vars(args))
            return
        set_colour(self.status_label, None)
        save_settings(sim_online=self.online_check.isChecked(),
                      **({"sim_live_source": args.device_name} if live else {}))
        self._reset_run_state()
        self._running_live = live
        self._last_level_t, self._stall_logged = time.monotonic(), False
        self.heard_box.clear(); self.translated_box.clear()
        self._live, self._last_preview_words = {}, {}
        self._set_columns(live)
        self.totals_label.setText("")
        self.progress.setValue(0)
        self.translated_label.setText(f"Translation → {args.tgt}")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.clip_title.setText("Loading the models…")
        self.clip_text.setText("The first start after opening the window takes longest.")
        self._stop_evt = threading.Event()

        def level_cb(*reading):
            self._level = reading

        def work():
            """Background thread: run the whole simulation and report the result."""
            try:
                rows, path, wall = S.run(args, lambda k, d: self._events.put((k, d)),
                                         self._stop_evt, level_cb)
                self._events.put(("done", (rows, path, wall)))
            except BaseException as e:            # SystemExit from a library too
                log_exception("simulation_failed", e, data=args.data, model=args.model,
                              device=getattr(args, "device_name", None))
                self._events.put(("failed", f"{type(e).__name__}: {e}"))
        self._thread = threading.Thread(target=work, daemon=True, name="simulation")
        self._thread.start()
        log_event(logging.INFO, "sim_gui_started", **vars(args))

    def _check_args(self, args):
        """A reason the run can't start, or None. Checked here, so a mistake
        is a clear message instead of an error in the background thread."""
        import json
        if args.src == args.tgt:
            return f"Source and Target are both {args.src}: choose a different Target."
        if args.data == "live":
            if args.device < 0:
                return ("No live source: Windows lists no loopback devices. Check that an "
                        "output device is enabled in Windows Sound settings.")
            return None
        maker = "make_testdata.py" if args.data == "synth" else "fetch_testdata.py"
        manifest = S.DATA_DIR / "manifest.jsonl"
        if not manifest.exists():
            if args.data == "feedback":
                return ("No reported phrases yet: in the interpreter window, use 'Report a "
                        "mistake' on phrases it got wrong; they appear here.")
            return f"No test data yet: run  .venv\\Scripts\\python {maker}  first."
        try:
            rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
        except (OSError, ValueError) as e:
            log_exception("sim_manifest_unreadable", e, path=str(manifest))
            return f"The list of test clips can't be read ({e}). Run {maker} again."
        kinds = args.kinds.split(",")
        have = {r["lang"] for r in rows if r["kind"] in kinds}
        wanted = {l for l in args.langs.split(",") if l}
        if wanted - have:
            return (f"No {self.kinds_combo.currentText().lower()} clips in "
                    f"{', '.join(sorted(wanted - have))}. Available: {', '.join(sorted(have))}")
        chosen = [r for r in rows if r["kind"] in kinds and (not wanted or r["lang"] in wanted)
                  and r.get("file")]
        if not chosen:
            return "No clips match these settings."
        missing = [r["file"] for r in chosen if not (S.DATA_DIR / r["file"]).exists()]
        if missing:
            log_event(logging.ERROR, "sim_clip_missing", count=len(missing), first=missing[:5],
                      data=args.data)
            return f"{len(missing)} test clips are missing (e.g. {missing[0]}). Run {maker} again."
        return None

    def _stop(self):
        """Stop button: ask the simulation to finish; it scores and saves what
        has played so far."""
        if self._stop_evt is not None:
            log_event(logging.INFO, "sim_gui_stop_pressed", clips_scored=self._scored)
            self._stop_evt.set()
            self.status_label.setText("Stopping… (finishing the phrase in progress and saving)")
            self.stop_btn.setEnabled(False)

    # ── Listening ────────────────────────────────────────────────────────

    def _volume(self):
        return 0.0 if self.mute_btn.isChecked() else self.volume_slider.value() / 100

    def _on_volume(self, *_):
        """Volume slider or Mute changed: apply at once and remember it."""
        if self._monitor is not None:
            self._monitor.volume = self._volume()
        save_settings(sim_volume=self.volume_slider.value())

    def _on_output_change(self, label):
        """A different 'Listen on' device: remember it, and switch the
        playback to it at once if a simulation is running."""
        save_settings(sim_output=label)
        log_event(logging.INFO, "sim_output_changed", output=label)
        if self.playlist is not None and self._thread is not None and self._thread.is_alive():
            self._start_monitor()

    def _start_monitor(self):
        """(Re)start playing the clips on the chosen output device."""
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        label = self.out_combo.currentText()
        if label == NO_OUTPUT or self.playlist is None:
            return
        self._monitor = AudioMonitor(self.playlist, self._outputs[label], self._volume())
        self._monitor.start()

    # ── Updates from the simulation (every 30 ms, on the main thread) ────

    def _tick(self):
        """One failed update must not stop the timer (the window would freeze
        its display), so errors are logged and the next tick carries on."""
        try:
            self._tick_once()
        except Exception as e:
            log_exception("sim_gui_update_failed", e)

    def _slow_tick(self):
        """Once a second: issues, the live 'playing now' list, the stall check."""
        try:
            self._refresh_issues()
            if self.live_box.isVisible():
                self._refresh_live_apps()
            running = self._thread is not None and self._thread.is_alive() and \
                self._run_started is not None
            if running and not self._stall_logged and \
                    time.monotonic() - self._last_level_t > STALL_SEC:
                self._stall_logged = True
                log_event(logging.WARNING, "sim_pipeline_stalled", seconds=STALL_SEC,
                          clip=self._shown_clip + 1)
        except Exception as e:
            log_exception("sim_gui_update_failed", e)

    def _tick_once(self):
        """Handle everything the simulation has sent since the last tick (status,
        previews, finals, translations, done), update the meters, progress
        and scores."""
        for _ in range(200):                     # bounded: keep the window responsive
            try:
                kind, data = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_label.setText(str(data))
            elif kind == "started":
                self._started(data)
            elif kind == "partial":
                seq, heard, translated = data[0], data[1], data[2]
                self._previews.append((time.monotonic(), heard))
                self._set_live(self.heard_box, seq, heard)
                if translated:
                    self._set_live(self.translated_box, seq, translated)
            elif kind == "final":
                self._finals.append(data)
                self._commit(self.heard_box, data["seq"], data["text"])
                self._commit(self.translated_box, data["seq"], data["translated"] or "")
                if self._running_live and data["text"]:
                    self._add_live_row(S.live_row(data))
            elif kind == "translation":
                self._commit(self.translated_box, data["seq"], data["translated"] or "")
            elif kind == "done":
                self._finished(*data)
            elif kind == "failed":
                self._finished(None, None, None, error=data)
        if self._monitor is not None and self._monitor.error and not self._monitor.reported:
            self._monitor.reported = True
            self.status_label.setText(f"Can't play on {self.out_combo.currentText()}: "
                                      f"{self._monitor.error}  (see Issues)")
            set_colour(self.status_label, WARN)
        if self._level is not None:
            rms_db, peak_db, speaking, prob, threshold = self._level
            self._level = None
            self._last_level_t = time.monotonic()
            self.meter.update_level(rms_db, peak_db)
            self.speech_meter.update_score(prob, threshold, speaking)
            self.level_label.setText("  -∞ dBFS  (silent)" if rms_db <= -99 else f"{rms_db:6.1f} dBFS")
            self.speech_label.setText(f"{prob * 100:3.0f}% / needs {threshold * 100:.0f}%  "
                                      f"{'SPEECH' if speaking else 'no speech'}")
            set_colour(self.speech_label, OK if speaking else None)
        if self._thread is not None and self._thread.is_alive() and self._run_started is not None:
            if self._running_live:
                self._show_live_progress()
            elif self.playlist is not None:
                self._show_progress()
                self._score_finished_clips()

    def _started(self, playlist):
        """The clips have started playing (or live recording has begun)."""
        self._run_started = time.monotonic()
        self._last_level_t = self._run_started
        self.playlist = playlist
        if playlist is None:                     # live
            self.clip_title.setText(f"Live: {self.live_combo.currentText()}")
            self.clip_text.setText("Listening to what Windows plays on this output. There is no "
                                   "correct text to compare with, so the table shows each phrase, "
                                   "its language and how fast its text arrived. Press Stop to end.")
            self.progress.setRange(0, 0)         # moving bar: no end
            return
        self.progress.setRange(0, 1000)
        self._start_monitor()
        self.status_label.setText(f"Playing {len(playlist.clips)} clips "
                                  f"({playlist.duration / 60:.1f} min).")

    def _show_live_progress(self):
        """Live mode: elapsed time and number of phrases so far."""
        t = time.monotonic() - self._run_started
        phrases = sum(1 for f in self._finals if f["text"])
        self.time_label.setText(f"{int(t // 60)}:{int(t % 60):02d}  ·  {phrases} phrases")

    def _show_progress(self):
        """Progress bar and time, and which clip is playing now (shown with its
        correct text, and logged, when it changes)."""
        p = self.playlist
        t = time.monotonic() - p.t0
        self.progress.setValue(int(1000 * min(1.0, t / p.duration)))
        self.time_label.setText(f"{int(t // 60)}:{int(t % 60):02d} / "
                                f"{int(p.duration // 60)}:{int(p.duration % 60):02d}")
        pos = t * RATE
        current = -1
        for i, (clip, (start, _)) in enumerate(zip(p.clips, p.spans)):
            if pos >= start - clip.get("vocal_start", 0) * RATE:
                current = i
        if current != self._shown_clip and current >= 0:
            self._shown_clip = current
            clip = p.clips[current]
            extra = [clip.get("voice"),
                     f"vocals {clip['snr_db']} dB above the music" if "snr_db" in clip else None,
                     f"gain {clip['gain_db']:+.1f} dB" if "gain_db" in clip else None]
            self.clip_title.setText("  ·  ".join(
                [f"Clip {current + 1} of {len(p.clips)}", clip["kind"], clip["lang"], clip["file"]]
                + [x for x in extra if x]))
            self.clip_text.setText(clip["text"])
            log_event(logging.INFO, "sim_clip_playing", clip=current + 1, of=len(p.clips),
                      file=clip["file"], kind=clip["kind"], lang=clip["lang"],
                      gain_db=clip.get("gain_db"))

    def _score_finished_clips(self):
        """Score each clip once its last phrase has had time to arrive."""
        p, now = self.playlist, time.monotonic()
        done = sum(p.time_of(b) < now - SCORE_DELAY_SEC for _, b in p.spans)
        if done > self._scored:
            rows = S.score(p, self._finals, self._previews)
            for r in rows[self._scored:done]:
                self._add_clip_row(r)
            self._scored = done
            self._show_totals(rows[:done])

    # ── Results table ────────────────────────────────────────────────────

    def _put_row(self, cells, tooltip):
        """Add one row to the results table: cells are (text, colour or None)."""
        n = self.table.rowCount()
        self.table.insertRow(n)
        for c, (text, colour) in enumerate(cells):
            item = QTableWidgetItem(text)
            if colour:
                item.setForeground(QColor(colour))
            item.setToolTip(tooltip)
            self.table.setItem(n, c, item)
        self.table.scrollToBottom()
        return n

    @staticmethod
    def _latency_cell(ms):
        """A pause -> text time as a table cell, coloured green/amber/red."""
        if ms is None:
            return "-", MUTED
        return f"{ms} ms", OK if ms < 800 else WARN if ms < 1500 else BAD

    def _add_clip_row(self, r):
        """Add a scored clip to the table (and the log): detected language,
        gain, word error rate, delay, music mode, recognised text."""
        n = self.table.rowCount()
        wer = min(r["errors"], r["ref_words"]) / r["ref_words"] if r["ref_words"] else 0
        clip = self.playlist.clips[n] if self.playlist and n < len(self.playlist.clips) else {}
        gain = clip.get("gain_db")
        log_event(logging.INFO, "sim_clip_scored", clip=n + 1, file=r["file"], lang=r["lang"],
                  detected=r["detected"], wer=round(wer, 3), latency_ms=r["latency_ms"],
                  music_mode=r["music_mode"], phrases=r["phrases"], gain_db=gain)
        if r["detected"] and r["detected"] != r["lang"]:
            log_event(logging.WARNING, "sim_wrong_language", clip=n + 1, file=r["file"],
                      lang=r["lang"], detected=r["detected"], heard=r["hyp"][:120])
        cells = [(str(n + 1), None), (r["file"], None), (r["lang"], None),
                 (r["detected"] or "-", OK if r["detected"] == r["lang"] else BAD),
                 (f"{gain:+.0f} dB" if gain is not None else "-", None),
                 (f"{wer:.0%}", OK if wer < 0.1 else WARN if wer < 0.25 else BAD) if r["ref_words"]
                 else ("-", MUTED),
                 self._latency_cell(r["latency_ms"]),
                 ("on" if r["music_mode"] else "off", None), (r["hyp"] or "(nothing)", None)]
        self._put_row(cells, f"Correct: {r['ref']}\nHeard: {r['hyp']}\nTranslation: {r['translation']}")

    def _add_live_row(self, r):
        """Add a live phrase to the table and update the live totals."""
        n = self.table.rowCount()
        cells = [(str(n + 1), None), (time.strftime("%H:%M:%S"), None), (r["detected"] or "-", None),
                 self._latency_cell(r["latency_ms"]),
                 (f"{r['audio_sec']:.1f} s" if r["audio_sec"] else "-", None),
                 ("on" if r["music_mode"] else "off", None), (r["engine"] or "-", None),
                 (r["hyp"], None)]
        self._put_row(cells, f"Heard: {r['hyp']}\nTranslation: {r['translation']}")
        rows = S.live_rows(self._finals)
        lat = [x["latency_ms"] for x in rows if x["latency_ms"] is not None]
        langs = sorted({x["detected"] for x in rows if x["detected"]})
        self.totals_label.setText(f"{len(rows)} phrases  ·  languages heard: {', '.join(langs) or '-'}"
                                  f"  ·  end → text median {int(np.median(lat)) if lat else '-'} ms")

    def _show_totals(self, rows):
        """The totals line above the table for the clips scored so far."""
        if not rows:
            return
        # Reported phrases may have no correct text (then no word error rate)
        words = sum(r["ref_words"] for r in rows)
        ok = sum(r["detected"] == r["lang"] for r in rows)
        errs = sum(min(r["errors"], r["ref_words"]) for r in rows)
        lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        self.totals_label.setText(
            f"{len(rows)} clips  ·  language right {ok}/{len(rows)}  ·  "
            f"WER {f'{errs / words:.0%}' if words else '-'}  ·  "
            f"pause → text median {int(np.median(lat)) if lat else '-'} ms")

    def _finished(self, rows, path, wall, error=None):
        """The simulation ended (finished, stopped or failed): stop playback,
        add any remaining rows, show where the results were saved."""
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        self.progress.setRange(0, 1000)
        stopped = self._stop_evt is not None and self._stop_evt.is_set()
        if rows is not None:
            if not self._running_live and self.playlist is not None:
                for r in rows[self._scored:]:
                    self._add_clip_row(r)
                self._scored = len(rows)
                self._show_totals(rows)
            self.progress.setValue(1000)
            what = "phrases" if self._running_live else "clips"
            self.clip_title.setText(f"{'Stopped' if stopped else 'Finished'}: {len(rows)} {what} "
                                    f"in {wall / 60:.1f} min")
            self.status_label.setText(f"Details saved to {path}")
            log_event(logging.INFO, "sim_gui_finished", stopped=stopped, rows=len(rows),
                      live=self._running_live, minutes=round(wall / 60, 2), results=str(path),
                      totals=self.totals_label.text())
        else:
            self.progress.setValue(0)
            self.clip_title.setText("Simulation failed")
            self.clip_text.setText(f"{error}\n\nThe Issues page (left) explains what this means "
                                   "and what to do.")
            self.status_label.setText("Simulation failed (see Issues).")
            set_colour(self.status_label, BAD)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.playlist = None
        self._run_started = None

    def closeEvent(self, event):
        """Window closing: stop the simulation and playback, give the run up to
        5 s to save its results."""
        try:
            running = self._thread is not None and self._thread.is_alive()
            log_event(logging.INFO, "sim_gui_closing", session_id=self.session_id, running=running)
            if self._stop_evt is not None:
                self._stop_evt.set()
            if self._monitor is not None:
                self._monitor.stop()
            if running:
                self._thread.join(timeout=5)        # let it stop the pipeline and save results
            core.ERROR_LOG.removeHandler(self._issue_handler)
        finally:
            super().closeEvent(event)


def main():
    """Start the simulator window: catch crashes in the log, dark theme,
    run until the window closes."""
    core.install_global_exception_hooks()
    log_event(logging.INFO, "app_main_start", gui="simulator")
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)
    setThemeColor("#0078D4")
    window = SimulatorWindow()
    window.show()
    app.exec()
    log_event(logging.INFO, "app_main_exit", gui="simulator")


if __name__ == "__main__":
    main()

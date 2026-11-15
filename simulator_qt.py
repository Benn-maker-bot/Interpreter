"""
simulator_qt.py - the simulator in a window: watch and hear a test run live
===========================================================================

WHAT THIS FILE IS FOR
---------------------
simulate.py tests the interpreter by playing recorded clips through it and
scoring the results - but in a terminal you only see the numbers at the end.
This file wraps the same simulation in a WINDOW, so you can watch and hear
what happens while it runs:

  * which clip is playing now, with its correct text, and a progress bar
  * the input level and speech meters (the same as the interpreter's)
  * the recognised text and its translation, with the grey live previews
    (the same display code as the interpreter window)
  * a results table that fills in as each clip finishes, with running
    totals (language right, word error rate, delay)
  * the clips themselves, played on an output device of your choice (the
    FiiO by default), in step with what the interpreter is hearing

Four kinds of test input ("Test data"):
  Synthetic clips        made by make_testdata.py: computer voices, one
                         language per clip, exact text
  Recorded clips         downloaded by fetch_testdata.py: real people and songs
  Your reported phrases  phrases you marked with "Report a mistake"
  Live (Windows audio)   whatever is playing on an output device right now:
                         a radio stream, Spotify, a video … (no correct text,
                         so only language and speed are measured)

Problems are explained in plain words on the Issues page (the first of
each kind also pops up), and everything is written to the app's log files.
Results are saved to testdata/sim_results/ exactly as from simulate.py.

HOW TO START IT
---------------
  In VS Code: Run and Debug (Ctrl+Shift+D) -> "Simulator (window)" -> F5.
  Or in the terminal:  .venv\\Scripts\\python simulator_qt.py

HOW A QT WINDOW WORKS (the basics used in this file)
----------------------------------------------------
The window is built with Qt (a professional toolkit for windows, through
its Python version, PySide6) and "Fluent" widgets (qfluentwidgets: Qt
controls styled like Windows 11).

  Widget        anything on screen: a button, a label, a text box, a whole
                page. Widgets can contain other widgets.
                  button = PushButton("Start")
  Layout        arranges the widgets inside another widget:
                  QVBoxLayout  one above the other (Vertical box)
                  QHBoxLayout  side by side (Horizontal box)
                  QGridLayout  in rows and columns, like a table
                  layout.addWidget(button)   puts the button in the layout
  Signal/slot   widgets announce events through "signals", and you
                "connect" a function (the "slot") that runs when it happens:
                  button.clicked.connect(self._start)
                  -> clicking the button calls self._start()
  QTimer        runs a function again and again (or once, later):
                  timer.timeout.connect(self._tick); timer.start(30)
                  -> self._tick() runs every 30 ms
  Main thread   Qt only allows the MAIN thread (the one that started the
                program) to change widgets. Background threads must not
                touch the window; they send messages through a queue,
                and a timer on the main thread picks them up.
  Event loop    app.exec() at the bottom keeps the program running: it
                waits for clicks, key presses, timers and redraws, and
                calls the connected functions - until the window closes.

HOW THE PARTS TALK TO EACH OTHER
--------------------------------

   Start button
        |
        v
   _start() ---- starts a background thread ----> simulate.run(...)
                                                   (plays clips through the
                                                    real interpreter code)
                                                        |
        +--------- self._events (a queue) <-------------+  "status", "partial",
        |                                                  "final", "done" …
        v
   _tick() every 30 ms on the main thread: takes the messages out of the
   queue and updates the text boxes, meters, progress bar and table.

   AudioMonitor (another thread) plays the same clips on your FiiO.

WHAT IS IN THIS FILE
--------------------
  Settings and ISSUE_HELP   explanations of the simulator's own problems.
  AudioMonitor              plays the clips on an output device.
  SimulatorWindow           the window:
    __init__, _build_page, _build_issues_page   building it
    _on_data_change, _refresh_live_apps          reacting to the settings
    _args, _start, _check_args, _stop            starting and stopping
    _on_volume, _on_output_change, _start_monitor   listening
    _tick, _slow_tick, _tick_once, _started,      updating it while it runs
    _show_progress, _score_finished_clips
    _put_row, _add_clip_row, _add_live_row,       the results table
    _show_totals, _finished
    closeEvent                                    closing
  main()                    starts the program.

Python ideas new to this file are explained where they appear ("PYTHON:").
The basics are explained at the top of music.py.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse                 # Namespace: the bundle of settings simulate.run() expects
import logging                  # log levels: logging.INFO, logging.WARNING, logging.ERROR
import queue                    # thread-safe mailbox between the simulation and the window
import sys                      # sys.argv, sys.executable (the Python program running this)
import threading                # background threads and Events (stop flags)
import time                     # clocks and sleeping
from collections import deque   # a list with a maximum length

import numpy as np              # arrays of numbers (audio, medians)
# The real sound-card library, under its own name: simulate.build() swaps the
# interpreter's "pyaudio" for a fake one, but AudioMonitor needs the real one.
import pyaudio as real_pyaudio

# Qt: the window toolkit (PySide6 is its official Python version).
#   QtCore    non-visual basics: Qt (constants), QTimer
#   QtGui     colours, fonts, the screen
#   QtWidgets the standard controls and layouts
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QHeaderView, QSizePolicy, QSplitter,
                               QAbstractItemView, QTableWidgetItem, QVBoxLayout, QWidget)
# Fluent widgets: Windows 11-style versions of the controls. FluentIcon is
# imported "as FIF" (a short name): FIF.PLAY is the play icon, etc.
# The "# pyright: ignore[...]" notes only silence VS Code's type checker for
# this library (it has no type information); they don't change anything.
from qfluentwidgets import setThemeColor  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from qfluentwidgets import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
                            BodyLabel, CaptionLabel, CheckBox, ComboBox, FluentIcon as FIF,
                            FluentWindow, LineEdit, PrimaryPushButton, ProgressBar, PushButton,
                            SimpleCardWidget, Slider, SmoothScrollArea, StrongBodyLabel,
                            TableWidget, TextEdit, Theme, TogglePushButton, setTheme)

# The rest of this project:
import interpreter as core      # the app's logic ("core"), e.g. core.ISSUE_HELP
import simulate as S            # the simulation engine ("S"), e.g. S.run(...)
from interpreter import (LANGUAGES, MODELS, get_audio_devices, find_best_output_device,
                         load_settings, save_settings, log_event, log_exception)
# From the interpreter's Fluent window: the issue collector, the two meters,
# the label-colouring helper and the four status colours.
from interpreter_qt import (QtInterpreterApp, PaneText, IssueHandler, LevelMeter, SpeechMeter,
                            set_colour, OK, WARN, BAD, MUTED)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
RATE = S.RATE           # int 16000: samples per second of the clips
STALL_SEC = 15          # int: no level readings for this long while playing = pipeline stuck
SCORE_DELAY_SEC = 4     # int: a clip is scored this long after it ends (its last text has arrived)

# Plain-language help for the simulator's own problems, shown on the Issues
# page and in pop-ups. core.ISSUE_HELP is a dict in interpreter.py:
#     event name -> (title, what it means, what to do)
# .update({...}) adds these entries to it. "{name}" in the text is filled in
# from the details logged with the problem, e.g. {clip} -> 12.
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

# The drop-down lists' choices: dict, text shown -> value used by simulate.py.
DATA_CHOICES = {"Synthetic clips": "synth", "Synthetic clips (Piper)": "synth_piper",
                "Recorded clips": "real",
                "Your reported phrases": "feedback", "Live (Windows audio)": "live"}
KINDS = {"Speech and songs": "speech,song", "Speech only": "speech", "Songs only": "song"}
PREVIEWS = {"Small": "small", "Tiny": "tiny", "Main model": "main", "Off": "off"}
# ["All", "1", "2", … "10"]. PYTHON: str(n) turns a number into text.
PER_LANGUAGE = ["All"] + [str(n) for n in range(1, 11)]
NO_OUTPUT = "(don't play)"      # str: the "Listen on" choice for no playback


# ─────────────────────────────────────────────────────────────────────────────
# Playing the clips on a real output device (so you can hear the test)
# ─────────────────────────────────────────────────────────────────────────────

class AudioMonitor(threading.Thread):
    """Plays the test clips on an output device, in step with the simulation.

    The rule: sample number n is played at the moment the fake microphone
    hands sample n to the interpreter. So what you hear is exactly what the
    interpreter is hearing, at the same time.

    If the device falls behind (for example Windows was busy for a moment),
    it skips ahead to "now" rather than playing late, so it never drifts out
    of step.

    PYTHON: "class AudioMonitor(threading.Thread)" means AudioMonitor is a
    SUBCLASS of Thread: it is a kind of thread, and gets all of Thread's
    abilities (start(), is_alive() …). Its run() method is the code that
    runs in the background after .start().

    Class attributes:
        BLOCK  int 320: samples written per step (20 ms at 16 kHz).
        AHEAD  int 1280: at most this many samples (80 ms) may be queued
               ahead of "now" in the device. More would add delay; less
               could make the sound stutter.

    Attributes:
        playlist      the Playlist being simulated (its samples and t0).
        device_index  int: which output device (PortAudio's number for it).
        volume        float 0.0-1.0: playback volume; the window changes it
                      while it plays when you move the slider.
        error         str or None: what went wrong, if anything; the window
                      shows it.
        reported      bool: has the window already shown the error?
        _stop         threading.Event: set it to make the thread finish.

    Example:
        monitor = AudioMonitor(playlist, device_index=25, volume=0.7)
        monitor.start()        # starts playing in the background
        monitor.volume = 0.3   # quieter, immediately
        monitor.stop()         # finish
    """
    BLOCK = 320                                   # 20 ms at 16 kHz
    AHEAD = int(0.08 * RATE)                      # keep at most 80 ms queued

    def __init__(self, playlist, device_index, volume):
        """Prepare the player (it starts playing when .start() is called).

        Inputs:
            playlist      the Playlist being simulated.
            device_index  int: the output device to play on.
            volume        float 0.0-1.0.

        PYTHON: super().__init__(...) runs the PARENT class's __init__ (here
        threading.Thread's), which sets the thread up. daemon=True means the
        thread won't keep the program open after the window closes.
        """
        super().__init__(daemon=True, name="sim-monitor")
        self.playlist, self.device_index = playlist, device_index
        self.volume = volume                      # 0..1, changed live by the slider
        # PYTHON: a threading.Event is a flag shared between threads:
        # .set() raises it, .is_set() checks it.
        self._stop = threading.Event()
        self.error = None
        self.reported = False

    def stop(self):
        """Ask the thread to stop playing (it finishes within ~20 ms)."""
        self._stop.set()

    def _open(self, pa):
        """Open the output device for playing.

        Inputs:
            pa  a started PortAudio (real_pyaudio.PyAudio()).

        Returns (three values):
            stream    the open output stream: stream.write(bytes) plays them.
            rate      int: the sample rate the device was opened at.
            channels  int: 1 (mono) or 2 (stereo).

        Why the format varies:
            The clips are 16 kHz mono. Windows' older audio systems (MME,
            DirectSound) accept that and convert it themselves. The modern
            one (WASAPI) insists on the device's own format, e.g. 44.1 kHz
            stereo for the FiiO - then run() converts the audio itself.
        """
        # info: dict describing the device, e.g. {"name": "Speakers (FiiO Q series)",
        # "defaultSampleRate": 44100.0, "maxOutputChannels": 2, …}
        info = pa.get_device_info_by_index(self.device_index)
        try:
            # Raises ValueError if the device can't play 16 kHz mono 16-bit.
            pa.is_format_supported(RATE, output_device=self.device_index, output_channels=1,
                                   output_format=real_pyaudio.paInt16)
            rate, channels = RATE, 1
        except ValueError:
            rate = int(info["defaultSampleRate"])                  # e.g. 44100
            channels = min(2, int(info["maxOutputChannels"]))      # at most stereo
        # frames_per_buffer: how much audio the device takes per write;
        # 20 ms worth at the device's rate (e.g. 320 x 44100 / 16000 = 882).
        stream = pa.open(format=real_pyaudio.paInt16, channels=channels, rate=rate, output=True,
                         output_device_index=self.device_index,
                         frames_per_buffer=int(self.BLOCK * rate / RATE))
        log_event(logging.INFO, "sim_monitor_opened", device=info["name"], rate=rate,
                  channels=channels)
        return stream, rate, channels

    def run(self):
        """The thread's work: keep writing the playlist's audio to the device.

        Each round of the loop:
          1. due = which sample the simulation is at right now.
          2. If we're just starting, or more than 4 x AHEAD (320 ms) behind,
             jump to `due` (skip what was missed).
          3. If we're more than AHEAD (80 ms) in front, wait 5 ms.
          4. Otherwise take the next 20 ms of audio, apply the volume,
             convert it to the device's rate and channels if needed, and
             write it to the device.
        It stops when the playlist ends or stop() was called. Any error is
        logged and kept in self.error (the window shows it); the simulation
        itself carries on regardless.

        PYTHON: "finally:" runs whether the loop ended normally or with an
        error, so the device is always closed and released.
        """
        import soxr                               # sample-rate conversion
        pa = real_pyaudio.PyAudio()               # start PortAudio
        stream = None
        try:
            stream, rate, channels = self._open(pa)
            # A ResampleStream converts audio piece by piece and remembers the
            # end of each piece, so the joins are seamless (no clicks). Only
            # needed if the device doesn't run at 16 kHz.
            resampler = soxr.ResampleStream(RATE, rate, 1, dtype="float32") if rate != RATE else None
            # samples: the whole recording; pos: next sample to play (None = not started).
            samples, pos = self.playlist.samples, None
            while not self._stop.is_set():
                # Step 1. Seconds since the start x 16000 = the sample due now.
                due = int((time.monotonic() - self.playlist.t0) * RATE)
                if pos is None or pos < due - 4 * self.AHEAD:          # step 2
                    pos = max(0, due)                 # start, or catch up after a stall
                if pos >= len(samples):
                    break                             # the end of the playlist
                if pos > due + self.AHEAD:                             # step 3
                    time.sleep(0.005)
                    continue
                # Step 4. Next 20 ms, as decimals (-1..1), times the volume.
                x = samples[pos:pos + self.BLOCK].astype(np.float32) / 32768 * self.volume
                pos += self.BLOCK
                if resampler is not None:
                    x = resampler.resample_chunk(x)   # 16 kHz -> the device's rate
                if channels > 1:
                    # Mono -> stereo: repeat each sample for left and right.
                    # x[:, None] makes a column; np.repeat(…, 2, axis=1) copies it
                    # into 2 columns; .reshape(-1) flattens to L, R, L, R, …
                    x = np.repeat(x[:, None], channels, axis=1).reshape(-1)
                stream.write((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        except Exception as e:
            # type(e).__name__ is the error's kind, e.g. "OSError".
            self.error = f"{type(e).__name__}: {e}"
            log_exception("sim_monitor_failed", e, device_index=self.device_index)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream(); stream.close()
                except Exception:
                    pass
            pa.terminate()                        # release PortAudio


# ─────────────────────────────────────────────────────────────────────────────
# The window
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorWindow(FluentWindow):
    """The simulator window: a Simulator page and an Issues page.

    PYTHON: SimulatorWindow is a SUBCLASS of FluentWindow (a Windows
    11-style main window with a side bar for switching pages). It inherits
    everything a window can do (show, resize, close …) and adds its own
    pages and behaviour.

    The Simulator page, top to bottom:
        settings card   Test data, Clips, Languages, Per language, Source,
                        Target, Whisper model, Previews, three options and
                        Start/Stop (or the Live source list in live mode)
        Listen on       output device, volume and mute (hidden in live mode)
        now playing     clip title, its correct text, the progress bar
        meters          input level and speech
        text panes      Recognised | Translation
        results         the table and its totals line
        status line     messages, e.g. "Details saved to …"

    Main attributes (besides the widgets):
        session_id     str: this window's id in the log, e.g. "simgui-1e438f09".
        _events        queue.Queue: messages from the simulation thread.
        _level         tuple or None: the newest meter reading.
        _thread        the simulation thread, or None.
        _monitor       the AudioMonitor, or None.
        _stop_evt      threading.Event that tells the simulation to stop.
        playlist       the Playlist being played (None when idle or live).
        _running_live  bool: is the current run in live mode?
        _finals        list of finished phrases this run.
        _previews      list of (time, text) previews this run.
        _scored        int: how many clips have been added to the table.
        _shown_clip    int: the clip shown in the "now playing" card (-1 = none).
        _run_started   float or None: when playback started (clock time).
        _issues        deque of recent warnings/errors (for the Issues page).
        _outputs       dict: output device name -> device number.
        _loopbacks     dict: loopback (live source) name -> device number.

    Borrowed methods (see below) need these too: heard_box,
    translated_box, _live, _last_preview_words.
    """

    # ── Borrowed from the interpreter window ────────────────────────────
    # PYTHON: a method is just a function stored in a class, so it can be
    # copied into another class. These come from interpreter_qt.py and work
    # here because this class has the attributes they use.
    # The live-preview / final-text display (PaneText; needs self.heard_box,
    # self._live and self._last_preview_words):
    _fmt, _set_live, _commit = PaneText._fmt, PaneText._set_live, PaneText._commit
    # Static methods must stay static when copied, so they are wrapped in staticmethod().
    _at_bottom = staticmethod(PaneText._at_bottom)
    _follow = staticmethod(PaneText._follow)
    _trim = staticmethod(PaneText._trim)
    # Warnings and errors: grouped on the Issues page, first of each kind pops up.
    _refresh_issues = QtInterpreterApp._refresh_issues
    _render_issue_groups = QtInterpreterApp._render_issue_groups
    # Which warnings (not only errors) get a pop-up. PYTHON: "+" joins two tuples.
    POPUP_WARNINGS = QtInterpreterApp.POPUP_WARNINGS + ("sim_pipeline_stalled", "sim_clip_missing",
                                                        "sim_wrong_python")

    # The results table's columns, for clips and for live phrases.
    CLIP_COLUMNS = ["#", "Clip", "Language", "Heard as", "Gain", "WER", "Pause → text", "Music",
                    "Recognised"]
    LIVE_COLUMNS = ["#", "Time", "Heard as", "End → text", "Length", "Music", "Translated by",
                    "Recognised"]

    def __init__(self):
        """Build the window and get it ready.

        Steps:
          1. Set up the state (empty queues, nothing running).
          2. Start collecting warnings/errors for the Issues page: an
             IssueHandler is added as a loguru sink (WARNING and up), so every
             warning or error written anywhere lands in self._issues.
          3. List the audio devices NOW, before simulate.build() replaces
             the sound-card library with its fake one.
          4. Build the two pages and add them to the side bar.
          5. Size and place the window; show the right controls for the
             saved Test data choice.
          6. Start two timers: _tick every 30 ms (results, meters) and
             _slow_tick every second (issues, stall check).
          7. Log that the window opened, and check which Python runs it.
        """
        super().__init__()                       # set up the FluentWindow itself
        self.setWindowTitle("Interpreter Simulator")
        import uuid
        # Step 1. uuid4().hex is 32 random characters; [:8] keeps 8: "simgui-1e438f09".
        self.session_id = f"simgui-{uuid.uuid4().hex[:8]}"
        # Step 2
        self._issues = deque(maxlen=200)
        self._issue_count_seen, self._issue_groups, self._watchdog_restarts = 0, {}, 0
        self._issue_handler = IssueHandler(self._issues)
        self._issue_sink_id = core.add_issue_sink(self._issue_handler)
        self._events = queue.Queue()             # simulation thread -> window
        self._level = None                       # newest meter reading (capture thread)
        self._last_level_t, self._stall_logged = None, False
        self._stop_evt = None
        self._thread = self._monitor = None
        self.playlist = None
        self._running_live = False
        self._live, self._last_preview_words = {}, {}
        self._reset_run_state()
        # Step 3. Devices are listed now, before simulate.build() replaces pyaudio.
        # get_audio_devices() returns two lists of (number, name) pairs.
        inputs, self._output_list = get_audio_devices()
        # Dict comprehensions: name -> number, e.g. {"🎧 Speakers (FiiO Q series)": 25, …}
        self._outputs = {lbl: idx for idx, lbl in self._output_list}
        # Loopback devices record what an OUTPUT is playing: the live sources.
        self._loopbacks = {lbl: idx for idx, lbl in inputs if "[loopback]" in lbl.lower()}
        self._inputs = {lbl: idx for idx, lbl in inputs}
        # Step 4
        self.page = self._build_page()
        self.diag_page = self._build_issues_page()
        # addSubInterface(page, icon, text) adds a page and its side-bar button.
        self.addSubInterface(self.page, FIF.PLAY, "Simulator")
        self.addSubInterface(self.diag_page, FIF.DEVELOPER_TOOLS, "Issues")
        # Keep the side bar narrow: expanded, it took 350 px from the page.
        # (hasattr checks the method exists, in case of an older library version.)
        if hasattr(self.navigationInterface, "setMinimumExpandWidth"):
            self.navigationInterface.setMinimumExpandWidth(100000)
        # Step 5
        self._place_window()
        self._on_data_change()
        self._render_issue_groups()
        # Step 6. QTimer(self): the timer belongs to the window and stops with it.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)                    # every 30 ms
        self._slow_timer = QTimer(self)
        self._slow_timer.timeout.connect(self._slow_tick)
        self._slow_timer.start(1000)             # every second
        # Step 7. sys.executable: the Python program running this;
        # sys.version.split()[0]: just the version number, e.g. "3.13.16".
        log_event(logging.INFO, "sim_gui_opened", session_id=self.session_id,
                  output_devices=len(self._output_list), loopback_devices=len(self._loopbacks),
                  output=self.out_combo.currentText(), python=sys.executable,
                  python_version=sys.version.split()[0])
        self._check_environment()

    def _check_environment(self):
        """Warn if this isn't running in the project's own Python (.venv).

        Why: another Python may have different (or missing) package
        versions, so results and speed could differ from the app's.

        How: sys.prefix is the folder of the Python environment in use. For
        the project's environment it is ...\\Interpreter\\.venv. (The
        program file itself may be C:\\Program Files\\Python313\\python.exe
        even then: on Windows a .venv's python.exe starts the main Python
        with the .venv's packages - so the FOLDER is compared, not the file.)
        """
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
        """Size the window to fit the screen (minus the taskbar) and centre it.

        availableGeometry() is the part of the screen not covered by the
        taskbar. Sizes are "logical" pixels: Qt scales them for the display
        (on a 4K screen at 250 % scaling, 1536 x 776 logical pixels fill it).

        Example: available area 1536 x 776 -> window 1496 x 736, with 20 px
        of space all round. On a short screen the page scrolls rather than
        squeezing widgets over each other.
        """
        area = QGuiApplication.primaryScreen().availableGeometry()
        w, h = min(1500, area.width() - 40), min(1000, area.height() - 40)
        self.resize(w, h)
        # Centre: left edge = area's left + half the spare width (// = whole-number division).
        self.move(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2)
        log_event(logging.INFO, "window_placed", gui="simulator", size_logical=[w, h],
                  screen_area=[area.x(), area.y(), area.width(), area.height()])

    # ── Building the page ────────────────────────────────────────────────

    @staticmethod
    def _combo(items, current=None, min_width=None):
        """Make a drop-down list ("combo box").

        Inputs:
            items      the choices (a list, or a dict - then its keys).
            current    the choice to select (if it is one of the items).
            min_width  int or None: the minimum width in pixels.

        Returns:
            the ComboBox.

        Example:
            self._combo(["Small", "Tiny"], "Tiny", 110)
            -> a drop-down showing "Tiny", at least 110 px wide
        """
        c = ComboBox()
        c.addItems(list(items))             # list(a_dict) gives its keys
        if current in items:
            c.setCurrentText(current)
        if min_width:
            c.setMinimumWidth(min_width)
        return c

    @staticmethod
    def _field(name, widget, tip=None):
        """A caption above a control, as one small layout block:

              Languages               <- the caption (CaptionLabel)
            [ all  (or e.g. de,fr) ]  <- the control

        Inputs:
            name    str: the caption.
            widget  the control.
            tip     str or None: a hover explanation for both.

        Returns:
            the QVBoxLayout holding them (to be put into the settings grid).
        """
        box = QVBoxLayout()
        box.setSpacing(4)                   # 4 px between caption and control
        label = CaptionLabel(name)
        box.addWidget(label)
        box.addWidget(widget)
        if tip:
            label.setToolTip(tip)
            widget.setToolTip(tip)
        return box

    def _build_page(self):
        # The page scrolls if the screen is too short for it
        """Build the Simulator page and all its controls.

        Returns:
            the page (a scroll area containing everything).

        Layout, top to bottom (each part is explained where it is built):
          1. settings card (a grid of captioned controls)
          2. "Listen on" row
          3. "now playing" card
          4. meters
          5. text panes and results table, with a draggable bar between them
          6. status line

        The settings are pre-filled from interpreter.settings.json (your
        last choices, saved with keys starting "sim_").
        """
        # A scroll area shows a scroll bar when its content is taller than the
        # window. setWidgetResizable(True): the content widens with the window.
        scroll = SmoothScrollArea()
        scroll.setObjectName("simulatorPage")    # FluentWindow needs each page to have a unique name
        scroll.setWidgetResizable(True)
        # A "style sheet" (Qt's version of CSS) sets looks: no border, see-through.
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent}")
        page = QWidget()
        page.setStyleSheet("background:transparent")
        page.setMinimumHeight(680)               # below this the page scrolls
        scroll.setWidget(page)
        # root: the page's main layout - everything below is stacked in it.
        root = QVBoxLayout(page)
        root.setContentsMargins(20, 12, 20, 12)  # space at left, top, right, bottom
        root.setSpacing(10)                      # space between the parts
        settings = load_settings()               # dict: the saved choices

        # ── 1. Test settings card ─────────────────────────────────────────
        # A card is a rounded panel. Its grid has 2 rows and 5 columns:
        #   row 0:  Test data | Clips   | Languages     | Per language | options
        #   row 1:  Source    | Target  | Whisper model | Previews     | Start Stop
        card = SimpleCardWidget()
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        # settings.get("sim_data", "Synthetic clips"): the saved choice, or the default.
        self.data_combo = self._combo(DATA_CHOICES, settings.get("sim_data", "Synthetic clips"), 190)
        # When the choice changes, show the matching controls (_on_data_change).
        self.data_combo.currentTextChanged.connect(self._on_data_change)
        self.kinds_combo = self._combo(KINDS, min_width=170)
        self.langs_edit = LineEdit()             # a one-line text box
        self.langs_edit.setPlaceholderText("all  (or e.g. de,fr,ru)")   # grey hint when empty
        self.limit_combo = self._combo(PER_LANGUAGE, "All", 90)
        self.src_combo = self._combo(LANGUAGES, "Auto-detect", 170)
        self.tgt_combo = self._combo([k for k in LANGUAGES if k != "Auto-detect"], "English", 170)
        # _models: dict, label -> Whisper size, e.g. "Turbo - best accuracy ★" -> "large-v3-turbo".
        self._models = dict(MODELS)
        # next(k for … if …): the first label whose value is "large-v3-turbo".
        self.model_combo = self._combo(self._models, next(k for k, v in self._models.items()
                                                          if v == "large-v3-turbo"), 230)
        self.preview_combo = self._combo(PREVIEWS, "Small", 110)
        # grid.addLayout(block, row, column) places a captioned control in the grid.
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
        # Live source (shown instead of Clips / Languages / Per language).
        # If no loopback devices exist, the list shows a single explanation.
        self.live_combo = self._combo(self._loopbacks or {"(no loopback devices found)": -1}, None, 360)
        self.live_combo.setToolTip("Records everything Windows plays on this output. To test one "
                                   "app only, send it to its own output in Windows Settings → Sound "
                                   "→ Volume mixer.")
        # The saved live source, else Windows' default output, else the first.
        want_live = settings.get("sim_live_source")
        if want_live not in self._loopbacks:
            want_live = self._default_output_loopback()
        if want_live in self._loopbacks:
            self.live_combo.setCurrentText(want_live)
        self.live_apps = CaptionLabel("")         # "Playing now: Spotify, firefox"
        self.live_apps.setWordWrap(True)          # long text wraps onto more lines
        # live_box: the caption, the list and the "playing now" line, together.
        self.live_box = QWidget()
        lb = QVBoxLayout(self.live_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(4)
        lb.addWidget(CaptionLabel("Live source (what Windows plays on)"))
        lb.addWidget(self.live_combo)
        lb.addWidget(self.live_apps)
        # addWidget(widget, row, column, rows spanned, columns spanned):
        # row 0, column 1, 1 row high, 3 columns wide - over Clips..Per language.
        grid.addWidget(self.live_box, 0, 1, 1, 3)
        # Options and buttons.
        self.music_check = CheckBox("Music mode")
        self.music_check.setChecked(True)
        self.online_check = CheckBox("Online translation")
        self.online_check.setChecked(bool(settings.get("sim_online", True)))
        self.level_check = CheckBox("Level each clip")
        self.level_check.setChecked(True)
        self.level_check.setToolTip("Automatic gain: bring every clip to the same loudness "
                                    f"({S.LEVEL_TARGET_DBFS} dBFS) before the interpreter hears it")
        opts = QVBoxLayout()                      # the three tick boxes, stacked
        opts.setSpacing(6)
        for c in (self.music_check, self.online_check, self.level_check):
            opts.addWidget(c)
        grid.addLayout(opts, 0, 4)
        # PrimaryPushButton: the highlighted (blue) main button.
        self.start_btn = PrimaryPushButton(FIF.PLAY, "Start")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = PushButton(FIF.PAUSE, "Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)           # greyed out until a run starts
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        grid.addLayout(buttons, 1, 4, Qt.AlignBottom)   # aligned to the bottom of its cell
        grid.setColumnStretch(2, 1)               # column 2 (Languages) takes any spare width
        root.addWidget(card)

        # ── 2. Listening: output device and volume ─────────────────────────
        self.listen_row = QWidget()
        listen = QHBoxLayout(self.listen_row)
        listen.setContentsMargins(0, 0, 0, 0)
        listen.addWidget(StrongBodyLabel("Listen on"))
        # outs: "(don't play)" followed by every output device's name.
        outs = [NO_OUTPUT] + list(self._outputs)
        # The saved device, or the best FiiO found, or "(don't play)".
        want = settings.get("sim_output") or find_best_output_device(self._output_list)
        self.out_combo = self._combo(outs, want if want in outs else NO_OUTPUT, 340)
        self.out_combo.currentTextChanged.connect(self._on_output_change)
        listen.addWidget(self.out_combo)
        listen.addSpacing(16)                     # 16 px of empty space
        listen.addWidget(CaptionLabel("Volume"))
        self.volume_slider = Slider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(settings.get("sim_volume", 70)))
        self.volume_slider.setFixedWidth(200)
        self.volume_slider.valueChanged.connect(self._on_volume)
        listen.addWidget(self.volume_slider)
        # A TogglePushButton stays pressed in until clicked again.
        self.mute_btn = TogglePushButton(FIF.MUTE, "Mute")
        # toggled sends True/False; "lambda _:" ignores it and calls _on_volume().
        self.mute_btn.toggled.connect(lambda _: self._on_volume())
        listen.addWidget(self.mute_btn)
        listen.addStretch(1)                      # push everything to the left
        root.addWidget(self.listen_row)

        # ── 3. Now playing + progress ──────────────────────────────────────
        now = SimpleCardWidget()
        nv = QVBoxLayout(now)
        nv.setContentsMargins(16, 10, 16, 10)
        top = QHBoxLayout()
        self.clip_title = StrongBodyLabel("Not running")   # e.g. "Clip 4 of 80 · speech · de · …"
        top.addWidget(self.clip_title, 1)         # the 1 = "takes the spare width"
        self.time_label = CaptionLabel("")        # e.g. "3:12 / 17:30"
        top.addWidget(self.time_label)
        nv.addLayout(top)
        self.clip_text = BodyLabel("Press Start to play the test clips through the interpreter.")
        self.clip_text.setWordWrap(True)
        nv.addWidget(self.clip_text)
        self.progress = ProgressBar()
        # 0..1000 rather than 0..100: smoother movement on a long run.
        self.progress.setRange(0, 1000)
        nv.addWidget(self.progress)
        root.addWidget(now)

        # ── 4. Meters ─────────────────────────────────────────────────────
        # A 2 x 3 grid:  caption | meter | reading
        meters = QGridLayout()
        meters.setVerticalSpacing(4)
        self.meter, self.speech_meter = LevelMeter(), SpeechMeter()
        self.level_label, self.speech_label = CaptionLabel(""), CaptionLabel("")
        for lbl in (self.level_label, self.speech_label):
            lbl.setFont(QFont("Consolas", 10))    # a fixed-width font: numbers don't jump about
            lbl.setFixedWidth(250)
        meters.addWidget(CaptionLabel("Input level"), 0, 0)
        meters.addWidget(self.meter, 0, 1)
        meters.addWidget(self.level_label, 0, 2)
        meters.addWidget(CaptionLabel("Speech"), 1, 0)
        meters.addWidget(self.speech_meter, 1, 1)
        meters.addWidget(self.speech_label, 1, 2)
        meters.setColumnStretch(1, 1)             # the meters take the spare width
        root.addLayout(meters)

        # ── 5. Text panes and results, split by a draggable bar ────────────
        # texts: the two text boxes side by side, each with its title above.
        texts = QWidget()
        tg = QGridLayout(texts)
        tg.setContentsMargins(0, 0, 0, 0)
        tg.addWidget(StrongBodyLabel("Recognised"), 0, 0)
        self.translated_label = StrongBodyLabel("Translation")
        tg.addWidget(self.translated_label, 0, 1)
        self.heard_box, self.translated_box = TextEdit(), TextEdit()
        for box in (self.heard_box, self.translated_box):
            box.setReadOnly(True)                 # you can select and copy, not type
            box.setMinimumHeight(90)
        tg.addWidget(self.heard_box, 1, 0)
        tg.addWidget(self.translated_box, 1, 1)
        tg.setRowStretch(1, 1)                    # the boxes take the spare height

        # results: the "Results" title with the totals line, then the table.
        results = QWidget()
        rv = QVBoxLayout(results)
        rv.setContentsMargins(0, 6, 0, 0)
        head = QHBoxLayout()
        head.addWidget(StrongBodyLabel("Results"))
        head.addStretch(1)
        self.totals_label = BodyLabel("")         # e.g. "36 clips · language right 34/36 · …"
        head.addWidget(self.totals_label)
        rv.addLayout(head)
        self.table = TableWidget()
        self.table.verticalHeader().hide()        # no row numbers on the left (column "#" has them)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)   # read-only
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)  # clicking selects a whole row
        self.table.setWordWrap(False)
        self.table.setMinimumHeight(90)
        rv.addWidget(self.table, 1)

        # Text boxes and table take whatever height is left: by default they ask
        # for ~400 px each, which made the page taller than the window.
        # QSizePolicy(horizontal, vertical): Expanding = grow to fill the width;
        # Ignored = don't ask for any particular height.
        for w in (self.heard_box, self.translated_box, self.table):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        # A splitter holds widgets with a bar between them you can drag.
        split = QSplitter(Qt.Vertical)            # one above the other
        split.addWidget(texts)
        split.addWidget(results)
        split.setChildrenCollapsible(False)       # neither part can be dragged to nothing
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)             # text and results share the height equally
        root.addWidget(split, 1)

        # ── 6. Status line ──────────────────────────────────────────────────
        self.status_label = CaptionLabel("Ready.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        return scroll

    def _default_output_loopback(self):
        """The live source most likely to be playing: Windows' default output.

        Spotify, browsers and most apps play on the default output unless
        told otherwise, so its loopback is the best first guess.

        Returns:
            str: the matching loopback's name, e.g.
            "🔊 Speakers (FiiO Q series) [Loopback]", or None if it can't
            be found (the error is logged).

        How: ask PortAudio's WASAPI system for its default output device's
        name, then find the loopback whose name contains it.
        """
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
        """Empty the results table and give it the right columns.

        Inputs:
            live  bool: True = the live-phrase columns, False = the clip columns.

        The last column ("Recognised") stretches to fill the width; the
        others are just as wide as their contents.
        """
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
        """Build the Issues page: every warning and error, grouped by kind,
        each with a plain-language explanation, and a Copy button (to paste
        the details into a message).

        Returns:
            the page widget. self.issues_text is the text box the borrowed
            _render_issue_groups() fills in.
        """
        page = QWidget()
        page.setObjectName("issuesPage")
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 12, 20, 12)
        top = QHBoxLayout()
        top.addWidget(StrongBodyLabel("Recent warnings and errors"))
        top.addStretch(1)
        copy = PushButton(FIF.COPY, "Copy")
        # Clicking Copy puts the whole text on the clipboard.
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

    # ── Reacting to the settings ─────────────────────────────────────────

    def _on_data_change(self, *_):
        """The Test data choice changed: show the controls that apply to it.

        Live mode hides Clips, Languages, Per language, "Level each clip"
        and the "Listen on" row (you already hear the live audio), and
        shows the Live source list instead. The choice is saved, and (if
        nothing is running) the table gets the matching columns.

        PYTHON: "*_" accepts and ignores any arguments (the signal passes
        the new text, which isn't needed here).
        """
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
        """Show or hide the captions above Clips / Languages / Per language.
        (Each caption is the first item, itemAt(0), of its small layout.)"""
        for w in (self.kinds_combo, self.langs_edit, self.limit_combo):
            layout = self._layout_of(w)
            if layout is not None:
                layout.itemAt(0).widget().setVisible(visible)

    def _layout_of(self, widget):
        """Find the small caption+control layout in the settings card that
        holds `widget` (made by _field). Returns it, or None.

        The path to the settings grid: the scroll area's page (.widget()),
        its main layout (.layout()), the first item in it (the card,
        .itemAt(0).widget()), and the card's layout (the grid).
        """
        grid = self.page.widget().layout().itemAt(0).widget().layout()
        for i in range(grid.count()):
            item = grid.itemAt(i)
            lay = item.layout()
            # indexOf returns the widget's position in the layout, or -1 if absent.
            if lay is not None and lay.indexOf(widget) >= 0:
                return lay
        return None

    def _refresh_live_apps(self):
        """Show which apps are playing sound now, under the Live source list.

        Example: "Playing now: Spotify, firefox   (open but silent: Teams)"
        An app counts as playing if its current sound level ("peak") is
        above 0.001. ".exe" is removed from the names for readability.
        """
        try:
            apps = core.audio_sessions_snapshot()   # list of dicts: {"app": "Spotify.exe", "peak": 0.48, …}
        except Exception as e:
            log_exception("sim_audio_sessions_failed", e)
            apps = []
        # PYTHON: "text".removesuffix(".exe") removes ".exe" from the end, if present.
        playing = [a["app"].removesuffix(".exe") for a in apps
                   if "app" in a and (a.get("peak") or 0) > 0.001]
        idle = [a["app"].removesuffix(".exe") for a in apps
                if "app" in a and not (a.get("peak") or 0) > 0.001]
        self.live_apps.setText(
            ("Playing now: " + ", ".join(playing) if playing else "Nothing is playing sound right now.")
            + (f"   (open but silent: {', '.join(idle)})" if idle else ""))

    # ── Starting and stopping ────────────────────────────────────────────

    def _args(self):
        """Collect the window's settings into the bundle simulate.run() expects.

        Returns:
            an argparse.Namespace - an object whose attributes are the
            settings, the same ones simulate.py's command line produces:
              args.src = "Auto-detect", args.tgt = "English",
              args.kinds = "speech,song", args.langs = "de,fr",
              args.limit = 0 (All), args.model = "large-v3-turbo", …
            In live mode also args.device_name and args.device (its number).

        Example: Per language "2", Languages "DE, fr"
            -> args.limit = 2, args.langs = "de,fr"  (spaces removed, lower case)
        """
        data = DATA_CHOICES[self.data_combo.currentText()]
        limit = self.limit_combo.currentText()
        # PYTHON: argparse.Namespace(a=1, b=2) makes an object with .a == 1 and .b == 2.
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
            # .get(name, -1): the device number, or -1 if not found.
            args.device = self._loopbacks.get(args.device_name, -1)
        return args

    def _start(self):
        """The Start button: check the settings, clear the last results, and
        run the simulation in a background thread.

        Steps:
          1. Collect the settings; point the simulator at the chosen data set.
          2. Check them (_check_args). A problem is shown in red in the
             status line, and nothing starts.
          3. Save some choices, clear the text, table, totals and progress.
          4. Swap the buttons (Start greyed out, Stop enabled).
          5. Start a background thread running simulate.run(). It reports
             back through self._events, which _tick() reads every 30 ms.

        Why a background thread: a simulation takes minutes. If it ran on
        the main thread, the window couldn't redraw or react to clicks
        until it finished - it would look frozen.
        """
        args = self._args()                                          # step 1
        live = args.data == "live"
        if not live:
            S.DATA_DIR = S.DATA_SETS[args.data]
        problem = self._check_args(args)                             # step 2
        if problem:
            self.status_label.setText(problem)
            set_colour(self.status_label, BAD)
            log_event(logging.INFO, "sim_gui_start_refused", reason=problem, **vars(args))
            return
        set_colour(self.status_label, None)                          # step 3
        # PYTHON: **({...} if live else {}) adds the extra named argument only in
        # live mode (unpacking an empty dict adds nothing).
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
        self.start_btn.setEnabled(False)                             # step 4
        self.stop_btn.setEnabled(True)
        self.clip_title.setText("Loading the models…")
        self.clip_text.setText("The first start after opening the window takes longest.")
        self._stop_evt = threading.Event()                           # step 5

        def level_cb(*reading):
            """Called by the capture thread for every audio block with
            (level_db, peak_db, speaking, speech_prob, threshold). It only
            stores the newest reading; _tick() shows it. (Updating the
            meters here would touch widgets from the wrong thread.)
            PYTHON: *reading collects all the arguments into one tuple."""
            self._level = reading

        def work():
            """The background thread: run the whole simulation, then report
            ("done", results) or ("failed", error text) through the queue.

            PYTHON: "except BaseException" also catches SystemExit, which
            some libraries raise on fatal errors - the window must survive
            and explain it rather than vanish.
            """
            try:
                # The lambda passes every progress message into the queue as (kind, data).
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
        """Check the settings before starting.

        Returns:
            None if all is well, or a message (str) saying what's wrong
            and what to do, e.g.
              "Source and Target are both English: choose a different Target."
              "No speech only clips in xx. Available: bg, de, en, …"
              "3 test clips are missing (e.g. speech\\de_00.wav). Run make_testdata.py again."

        Why check here: a mistake found now is a clear message in the
        window. Found later, inside the background thread, it would be a
        technical error in the middle of a run.
        """
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
        # PYTHON: "except (A, B) as e" catches either kind of error:
        # OSError (can't read the file) or ValueError (not valid JSON).
        except (OSError, ValueError) as e:
            log_exception("sim_manifest_unreadable", e, path=str(manifest))
            return f"The list of test clips can't be read ({e}). Run {maker} again."
        kinds = args.kinds.split(",")
        # have: set of languages available for these kinds; wanted: set asked for.
        have = {r["lang"] for r in rows if r["kind"] in kinds}
        wanted = {l for l in args.langs.split(",") if l}
        # PYTHON: set1 - set2 = the items of set1 that aren't in set2.
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
        """The Stop button: ask the simulation to finish early.

        It doesn't stop instantly: the background thread notices the flag
        within ~0.05 s, finishes the phrase in progress, scores the clips
        played so far and saves them. _finished() then updates the window.
        """
        if self._stop_evt is not None:
            log_event(logging.INFO, "sim_gui_stop_pressed", clips_scored=self._scored)
            self._stop_evt.set()
            self.status_label.setText("Stopping… (finishing the phrase in progress and saving)")
            self.stop_btn.setEnabled(False)

    # ── Listening ────────────────────────────────────────────────────────

    def _volume(self):
        """The playback volume, 0.0 to 1.0: 0 when muted, otherwise the
        slider's value / 100 (e.g. 70 -> 0.7)."""
        return 0.0 if self.mute_btn.isChecked() else self.volume_slider.value() / 100

    def _on_volume(self, *_):
        """Volume slider moved or Mute clicked: apply it at once (the
        AudioMonitor reads .volume for every 20 ms block) and remember it."""
        if self._monitor is not None:
            self._monitor.volume = self._volume()
        save_settings(sim_volume=self.volume_slider.value())

    def _on_output_change(self, label):
        """A different "Listen on" device was chosen: remember it, and if a
        simulation is running, switch the playback to it straight away.

        Inputs:
            label  str: the device's name (the signal passes the new text).
        """
        save_settings(sim_output=label)
        log_event(logging.INFO, "sim_output_changed", output=label)
        if self.playlist is not None and self._thread is not None and self._thread.is_alive():
            self._start_monitor()

    def _start_monitor(self):
        """(Re)start playing the clips on the chosen output device: stop any
        current player, then start a new AudioMonitor (unless "(don't play)"
        is chosen or nothing is playing)."""
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        label = self.out_combo.currentText()
        if label == NO_OUTPUT or self.playlist is None:
            return
        self._monitor = AudioMonitor(self.playlist, self._outputs[label], self._volume())
        self._monitor.start()

    # ── Updating the window while it runs (on the main thread) ──────────

    def _tick(self):
        """Runs every 30 ms (the fast timer): update the window.

        It only calls _tick_once() inside try/except. If one update fails,
        the error is logged (and explained on the Issues page), and the
        next tick carries on. Without this, one error would stop the timer
        and the window would silently stop updating.
        """
        try:
            self._tick_once()
        except Exception as e:
            log_exception("sim_gui_update_failed", e)

    def _slow_tick(self):
        """Runs every second (the slow timer).

          * shows new warnings/errors (Issues page, pop-ups)
          * refreshes the "Playing now" list in live mode
          * the STALL CHECK: if a run is going but no meter reading has
            arrived for STALL_SEC (15) seconds, the capture thread is
            probably stuck - log a warning once (it pops up and explains).
        """
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
        """One update of the window.

        Steps:
          1. Take the messages the simulation sent since the last tick (at
             most 200, so a flood can't freeze the window) and act on each:
               "status"       show it in the status line
               "started"      playback began (_started)
               "partial"      show a grey preview in the text panes
               "final"        replace the preview with the final text (and
                              in live mode add a table row)
               "translation"  an online translation arrived: show it
               "done"/"failed"  the run ended (_finished)
          2. If the audio player hit an error, show it once (amber).
          3. Show the newest meter reading.
          4. Update the progress, the "now playing" card and the scores.

        Example message from the queue:
            ("final", {"seq": 5, "text": "Guten Morgen", "translated": "Good morning",
                       "trace": {...}, "t": 1003.4})
        """
        for _ in range(200):                     # bounded: keep the window responsive
            try:
                # get_nowait(): take a message if there is one; never wait.
                kind, data = self._events.get_nowait()
            except queue.Empty:
                break                            # no more messages for now
            if kind == "status":
                self.status_label.setText(str(data))
            elif kind == "started":
                self._started(data)
            elif kind == "partial":
                # data = (seq, heard, translated, t_start, t); we need the first three.
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
                # PYTHON: f(*data) unpacks the tuple (rows, path, wall) into three arguments.
                self._finished(*data)
            elif kind == "failed":
                self._finished(None, None, None, error=data)
        # Step 2
        if self._monitor is not None and self._monitor.error and not self._monitor.reported:
            self._monitor.reported = True
            self.status_label.setText(f"Can't play on {self.out_combo.currentText()}: "
                                      f"{self._monitor.error}  (see Issues)")
            set_colour(self.status_label, WARN)
        # Step 3
        if self._level is not None:
            # Unpack the 5 values the capture thread stored.
            rms_db, peak_db, speaking, prob, threshold = self._level
            self._level = None
            self._last_level_t = time.monotonic()    # for the stall check
            self.meter.update_level(rms_db, peak_db)
            self.speech_meter.update_score(prob, threshold, speaking)
            # {rms_db:6.1f}: 1 decimal, 6 characters wide, e.g. " -23.5".
            self.level_label.setText("  -∞ dBFS  (silent)" if rms_db <= -99 else f"{rms_db:6.1f} dBFS")
            self.speech_label.setText(f"{prob * 100:3.0f}% / needs {threshold * 100:.0f}%  "
                                      f"{'SPEECH' if speaking else 'no speech'}")
            set_colour(self.speech_label, OK if speaking else None)
        # Step 4
        if self._thread is not None and self._thread.is_alive() and self._run_started is not None:
            if self._running_live:
                self._show_live_progress()
            elif self.playlist is not None:
                self._show_progress()
                self._score_finished_clips()

    def _started(self, playlist):
        """The clips have started playing (or live recording has begun).

        Inputs:
            playlist  the Playlist, or None in live mode.

        Clips: start the audio player and switch the progress bar to 0-1000.
        Live: there's no end, so the progress bar shows a moving "busy"
        stripe (a range of 0 to 0 means "unknown length").
        """
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
        """Live mode: elapsed time and phrases so far, e.g. "3:07  ·  41 phrases".

        {int(t % 60):02d}: the seconds part, always 2 digits ("07").
        """
        t = time.monotonic() - self._run_started
        phrases = sum(1 for f in self._finals if f["text"])
        self.time_label.setText(f"{int(t // 60)}:{int(t % 60):02d}  ·  {phrases} phrases")

    def _show_progress(self):
        """Update the progress bar, the time ("3:12 / 17:30"), and the
        "now playing" card.

        Which clip is playing: the last clip whose start (its voice start
        minus any music intro) has been reached. When it changes, the card
        shows its details and correct text, and the change is logged.

        Example card title:
            "Clip 38 of 80  ·  song  ·  bg  ·  songs\\bg_01.wav  ·  bg-BG-KalinaNeural
             ·  vocals 8.6 dB above the music  ·  gain +3.0 dB"
        """
        p = self.playlist
        t = time.monotonic() - p.t0                  # seconds since playback started
        # min(1.0, …) stops the bar overshooting during the 4 s tail.
        self.progress.setValue(int(1000 * min(1.0, t / p.duration)))
        self.time_label.setText(f"{int(t // 60)}:{int(t % 60):02d} / "
                                f"{int(p.duration // 60)}:{int(p.duration % 60):02d}")
        pos = t * RATE                               # the sample position now
        current = -1
        for i, (clip, (start, _)) in enumerate(zip(p.clips, p.spans)):
            if pos >= start - clip.get("vocal_start", 0) * RATE:
                current = i
        if current != self._shown_clip and current >= 0:
            self._shown_clip = current
            clip = p.clips[current]
            # Optional details; the ones a clip doesn't have are None and left out.
            # {x:+.1f} always shows the sign: +3.0 or -2.5.
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
        """Add clips to the results table as they finish.

        A clip is scored SCORE_DELAY_SEC (4 s) after its voice ended, so its
        last phrase has had time to be recognised. The whole run is scored
        again each time (simulate.score is quick) and only the new rows are
        added. Example: 12 clips done, 10 already in the table -> add rows
        11 and 12, update the totals.
        """
        p, now = self.playlist, time.monotonic()
        # Count the clips whose voice ended more than 4 s ago (True counts as 1).
        done = sum(p.time_of(b) < now - SCORE_DELAY_SEC for _, b in p.spans)
        if done > self._scored:
            rows = S.score(p, self._finals, self._previews)
            for r in rows[self._scored:done]:
                self._add_clip_row(r)
            self._scored = done
            self._show_totals(rows[:done])

    # ── The results table ────────────────────────────────────────────────

    def _put_row(self, cells, tooltip):
        """Add one row at the bottom of the results table and scroll to it.

        Inputs:
            cells    list of (text, colour or None), one per column, e.g.
                     [("1", None), ("speech\\de_00.wav", None), ("de", "#2ECC71"), …]
            tooltip  str: shown when the mouse hovers over the row.

        Returns:
            int: the new row's number (0 = first).
        """
        n = self.table.rowCount()
        self.table.insertRow(n)
        for c, (text, colour) in enumerate(cells):
            item = QTableWidgetItem(text)            # one cell
            if colour:
                item.setForeground(QColor(colour))   # the text colour
            item.setToolTip(tooltip)
            self.table.setItem(n, c, item)
        self.table.scrollToBottom()
        return n

    @staticmethod
    def _latency_cell(ms):
        """A "pause -> text" time as a table cell, with a traffic-light colour.

        Returns (text, colour):
            None   -> ("-", grey)
            372    -> ("372 ms", green)    under 0.8 s: good
            1063   -> ("1063 ms", amber)   0.8 - 1.5 s: noticeable
            2424   -> ("2424 ms", red)     1.5 s or more: slow
        """
        if ms is None:
            return "-", MUTED
        return f"{ms} ms", OK if ms < 800 else WARN if ms < 1500 else BAD

    def _add_clip_row(self, r):
        """Add one scored clip to the table, and log its score.

        Inputs:
            r  dict: the clip's result from simulate.score(), e.g.
               {"file": "speech\\de_00.wav", "lang": "de", "detected": "de",
                "errors": 1, "ref_words": 14, "latency_ms": 372, …}

        Colours: the "Heard as" cell is green if the language was right,
        red if not; WER is green under 10 %, amber under 25 %, red above.
        A wrong language is also logged as a warning (sim_wrong_language),
        which the Issues page explains.
        """
        n = self.table.rowCount()
        # Word error rate as a fraction: 1 error / 14 words = 0.071 (7 %).
        wer = min(r["errors"], r["ref_words"]) / r["ref_words"] if r["ref_words"] else 0
        # The matching playlist clip (for its gain), or {} if there isn't one.
        clip = self.playlist.clips[n] if self.playlist and n < len(self.playlist.clips) else {}
        gain = clip.get("gain_db")
        log_event(logging.INFO, "sim_clip_scored", clip=n + 1, file=r["file"], lang=r["lang"],
                  detected=r["detected"], wer=round(wer, 3), latency_ms=r["latency_ms"],
                  music_mode=r["music_mode"], phrases=r["phrases"], gain_db=gain)
        if r["detected"] and r["detected"] != r["lang"]:
            log_event(logging.WARNING, "sim_wrong_language", clip=n + 1, file=r["file"],
                      lang=r["lang"], detected=r["detected"], heard=r["hyp"][:120])
        # One (text, colour) pair per column; {gain:+.0f} shows "+3 dB" / "-2 dB".
        cells = [(str(n + 1), None), (r["file"], None), (r["lang"], None),
                 (r["detected"] or "-", OK if r["detected"] == r["lang"] else BAD),
                 (f"{gain:+.0f} dB" if gain is not None else "-", None),
                 # A clip with no correct text (some reported phrases) has no WER.
                 (f"{wer:.0%}", OK if wer < 0.1 else WARN if wer < 0.25 else BAD) if r["ref_words"]
                 else ("-", MUTED),
                 self._latency_cell(r["latency_ms"]),
                 ("on" if r["music_mode"] else "off", None), (r["hyp"] or "(nothing)", None)]
        self._put_row(cells, f"Correct: {r['ref']}\nHeard: {r['hyp']}\nTranslation: {r['translation']}")

    def _add_live_row(self, r):
        """Add one live phrase to the table, and update the live totals line:
        e.g. "41 phrases  ·  languages heard: bg, en, ru  ·  end → text median 232 ms".

        Inputs:
            r  dict from simulate.live_row().
        """
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
        """The totals line above the table, for the clips scored so far, e.g.
        "76 clips  ·  language right 74/76  ·  WER 5%  ·  pause → text median 580 ms".

        WER here is all errors / all correct words (not an average of the
        clips' percentages), so long clips count for more.
        """
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
            f"pause → text median {int(np.median(lat)) if lat else '-'} ms"
            # chrF (translation quality, 0-100) exists only once the run has
            # finished: simulate.run() adds it just before saving.
            + (f"  ·  translation chrF {S.TS.summary(rows).split()[0]}"
               if any(r.get("chrf") is not None for r in rows) else ""))

    def _finished(self, rows, path, wall, error=None):
        """The run ended - finished, stopped, or failed.

        Inputs:
            rows   list of result dicts, or None if it failed.
            path   Path of the saved results file (None if it failed).
            wall   float: how long it took, in seconds (None if it failed).
            error  str: what went wrong (only when it failed).

        Finished or stopped: add any rows not yet in the table, fill the
        progress bar, say where the results were saved, and log a summary.
        Failed: show the error in the "now playing" card, pointing to the
        Issues page, which explains it.
        Either way: stop the audio player and swap the buttons back.
        """
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        self.progress.setRange(0, 1000)          # back from the moving "busy" bar
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
        """The window is closing (Qt calls this when you click the X).

        Stops the simulation and the audio player, and gives a running
        simulation up to 5 seconds to stop the pipeline and save its
        results. Then detaches the issue collector from the error log.

        PYTHON: this method has a special name Qt looks for ("override"):
        defining closeEvent in our class replaces the parent's version;
        super().closeEvent(event) then runs the parent's version too, which
        actually closes the window. "finally" makes sure that always happens.
        """
        try:
            running = self._thread is not None and self._thread.is_alive()
            log_event(logging.INFO, "sim_gui_closing", session_id=self.session_id, running=running)
            if self._stop_evt is not None:
                self._stop_evt.set()
            if self._monitor is not None:
                self._monitor.stop()
            if running:
                # .join(timeout=5): wait for the thread to finish, at most 5 s.
                self._thread.join(timeout=5)        # let it stop the pipeline and save results
            core.remove_log_sink(self._issue_sink_id)
        finally:
            super().closeEvent(event)


def main():
    """Start the simulator.

    Steps:
      1. Catch any crash anywhere and write it to the log (so it can be
         explained later instead of the window just vanishing).
      2. Create the Qt application - every Qt program needs exactly one.
      3. Dark theme with the Windows accent blue.
      4. Build and show the window.
      5. app.exec(): run the "event loop" - wait for clicks, timers and
         redraws, and call the connected functions - until the window
         closes.
    """
    core.install_global_exception_hooks()                    # step 1
    log_event(logging.INFO, "app_main_start", gui="simulator")
    app = QApplication(sys.argv)                             # step 2
    setTheme(Theme.DARK)                                     # step 3
    setThemeColor("#0078D4")                                 # "#0078D4" = a colour as red-green-blue hex
    window = SimulatorWindow()                               # step 4
    window.show()
    app.exec()                                               # step 5
    log_event(logging.INFO, "app_main_exit", gui="simulator")


# Start the window only when this file is run directly (not when imported).
if __name__ == "__main__":
    main()

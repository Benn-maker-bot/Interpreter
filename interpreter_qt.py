"""
Voice Interpreter - Fluent (Windows 11 style) interface
=======================================================

This is a second window for the same program, built with PySide6 (the
official Python version of the Qt toolkit) and PyQt-Fluent-Widgets
(Windows 11 "Fluent" design controls for Qt). Start it with:

    .venv\\Scripts\\python interpreter_qt.py

TWO CHANNELS AT ONCE
--------------------
This window listens to two audio devices at the same time, e.g. your
microphone ("Me", English → Russian) and what the call plays in your
headphones ("Friend", Russian → English). Each is a CHANNEL (class Channel)
with its own pipeline: capture thread → transcription worker → translation,
built by its own SessionController. Both channels share one Whisper model on
the GPU (they take turns). Their results arrive in one queue, each message
labelled with its channel (session_controller.TaggedQueue), and appear in one
conversation (ConversationView), in the order the phrases were spoken.

The original window (interpreter.py, built with customtkinter) still works,
with one device at a time and the My Voice / Friend mode buttons. Both share
all of the real work: recording, speech detection, Whisper, translation,
logging.

BORROWED LOGIC AND ADAPTERS
---------------------------
A few methods that only deal with the (shared) model settings are borrowed
from the classic window's InterpreterApp class. They expect Tk-style helpers,
so this window provides:

  QVar        acts like a Tk variable (get()/set()) but reads and writes a
              Qt control, e.g. a combo box
  after()     Tk's "run this later" timer, implemented with QTimer

Qt basics used here: a QWidget is anything on screen; layouts (QVBoxLayout =
stacked vertically, QHBoxLayout = side by side, QGridLayout = rows and
columns) position widgets; "signals" (e.g. a button's `clicked`) are
connected to functions that run when the event happens; QTimer runs a
function later or repeatedly on the main thread.
"""

import logging
import queue
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from typing import Any, Callable, Hashable, Sequence, cast

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import (QCloseEvent, QColor, QFont, QFontMetricsF, QPaintEvent, QPainter,
                           QTextBlock, QTextBlockFormat, QTextCharFormat, QTextCursor,
                           QTextOption, QGuiApplication)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QSizePolicy, QTableWidgetItem, QVBoxLayout, QWidget,
                               QHeaderView, QAbstractItemView, QComboBox)
from qfluentwidgets import setThemeColor  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from qfluentwidgets import (  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
                            BodyLabel, CaptionLabel, CheckBox, ComboBox, FluentIcon as FIF,
                            FluentWindow, HeaderCardWidget, InfoBar, InfoBarPosition, LineEdit,
                            MessageBoxBase, NavigationItemPosition, PrimaryPushButton,
                            PushButton, SimpleCardWidget, Slider, SmoothScrollArea,
                            StrongBodyLabel, SubtitleLabel, TableWidget, TextEdit, Theme,
                            TitleLabel, TogglePushButton, TransparentToolButton,
                            setCustomStyleSheet, setTheme)

import interpreter as core
from interpreter import (InterpreterApp, LANGUAGES, continuation_join, trailing_marks_length, MODELS, Speaker,
                         PerfWindow, get_audio_devices, find_device_by_keywords,
                         find_best_vb_input, find_best_friend_input, find_best_output_device,
                         load_settings, log_event, log_exception, vad_threshold_for, plain_device_name,
                         NVIDIA_KEYWORDS, DEFAULT_SENS,
                         MAX_TEXT_HISTORY_CHARS, WATCHDOG_SEC, STATS_INTERVAL_SEC,
                         WHISPER_DEVICE, COMPUTE_TYPE, NUM_WORKERS,
                         DEFAULT_MODEL_CPU, DEFAULT_MODEL_GPU)
from session_controller import AnyEvent, SessionConfig, SessionController

PREVIEW_CHOICES: dict[str, str | None] = core.PREVIEW_CHOICES

# Status colours: green = fine, amber = worth a look, red = problem
OK, WARN, BAD, MUTED = "#2ECC71", "#F1C40F", "#E74C3C", "#9AA0A6"


# ─────────────────────────────────────────────────────────────────────────────
# Adapters that let interpreter.py's borrowed logic drive Qt widgets
# ─────────────────────────────────────────────────────────────────────────────

class QVar:
    """Acts like a Tk StringVar/DoubleVar/BooleanVar (get() and set()) but is
    backed by a Qt control. set() never fires the control's change signal,
    matching Tk, where setting a variable doesn't run the widget's command."""
    def __init__(self, getter: Callable[[], Any], setter: Callable[[Any], None],
                 widget: QWidget | None = None) -> None:
        self._get, self._set, self._widget = getter, setter, widget

    def get(self) -> Any:
        return self._get()

    def set(self, value: Any) -> None:
        """Change the control's value without firing its change signal (so the
        borrowed logic doesn't trigger itself, as with Tk variables)."""
        if self._widget is not None:
            self._widget.blockSignals(True)
        try:
            self._set(value)
        finally:
            if self._widget is not None:
                self._widget.blockSignals(False)


def _set_combo_text(combo: ComboBox, text: str) -> None:
    """Set text through Qt's base combo-box API.

    Fluent's ComboBox has a narrower type signature, but at runtime it is a
    QComboBox. This adapter keeps the borrowed Tk-style logic independent of
    Qt's widget classes.
    """
    cast(QComboBox, combo).setCurrentText(text)


def _ignore_signal_value(*_: object, callback: Callable[[], None]) -> None:
    """Adapt a Qt signal to a no-argument callback, whatever the signal sends.

    A combo box sends its new text, a check box its state, but a Fluent
    button's `clicked` sends NOTHING (Qt only passes what the connected
    function accepts). `*_` accepts any number of values (also none) and
    throws them away. It used to require exactly one, so clicking a Fluent
    button crashed with "missing 1 required positional argument" (logged
    2026-11-09 for the old mode buttons, and 2026-11-15 for Speak/Mute).
    `callback` must be given by name, e.g.
    partial(_ignore_signal_value, callback=self._clear).
    """
    callback()


class IssueHandler:
    """Copies every warning/error written to the log into a list the
    Diagnostics page shows ("Recent issues"). It is a loguru sink (added with
    core.add_issue_sink) and runs on whichever thread logged, so it only
    appends to a thread-safe deque."""
    def __init__(self, sink: deque[tuple[str, str, str, str, dict[str, str]]]) -> None:
        self.sink = sink

    def __call__(self, message: Any) -> None:
        """Called by loguru for every warning/error: turn it into a
        plain-language explanation and add it to the Diagnostics list."""
        record = message.record
        fields: dict[str, Any] = record["extra"].get("fields") or {}
        event: str = record["extra"].get("event") or record["message"]
        exc_info = tuple(record["exception"]) if record["exception"] else None
        try:
            help_: dict[str, Any] = core.explain_issue(event, fields, exc_info)
        except Exception:
            help_ = {"title": event, "meaning": "", "action": "",
                     "technical": str(fields.get("exception_message", "")), "where": "",
                     "details": ", ".join(f"{k}={v}" for k, v in fields.items())[:500]}
        # (time, level, event, short detail, plain-language explanation)
        self.sink.append((time.strftime("%H:%M:%S"), record["level"].name, event,
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

    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty meter (everything at the floor, -60 dBFS)."""
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._rms = self._peak = self.FLOOR_DB   # level and peak shown, dB
        self._peak_t = 0.0                        # when the peak was reached
        self._last_t = time.monotonic()           # time of the previous update

    def update_level(self, rms_db: float, peak_db: float) -> None:
        """Show a new reading: the bar jumps up at once but falls slowly
        (FALL_DB_PER_SEC); the peak mark stays for HOLD_SEC, then falls too."""
        now = time.monotonic()
        fall = self.FALL_DB_PER_SEC * (now - self._last_t)
        self._last_t = now
        self._rms = max(rms_db, self._rms - fall)
        if peak_db >= self._peak:
            self._peak, self._peak_t = peak_db, now
        elif now - self._peak_t > self.HOLD_SEC:
            self._peak = max(peak_db, self._peak - fall)
        self.update()   # ask Qt to repaint (calls paintEvent)

    def reset(self) -> None:
        self._rms = self._peak = self.FLOOR_DB
        self.update()

    @staticmethod
    def _colour(db: float, lit: bool) -> QColor:
        """Green below -18 dBFS, amber up to -6, red above; dim if not lit."""
        if db < -18:
            return QColor("#2ECC71" if lit else "#123d24")
        if db < -6:
            return QColor("#F1C40F" if lit else "#4a3f0c")
        return QColor("#E74C3C" if lit else "#4a1712")

    def paintEvent(self, _: QPaintEvent) -> None:
        """Qt calls this to draw the meter: one coloured segment per 1.5 dB,
        the peak-hold segment, and the dB scale underneath."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
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
            p.drawText(QRectF(min(max(x - 14, 0), w - 28), 19, 28, 14),
                       Qt.AlignmentFlag.AlignCenter, str(t))


class SpeechMeter(QWidget):
    """How much the sound resembles speech (0-100 %), with a white line at
    the threshold set by Speech sensitivity. Green while it counts as speech.
    Sensitivity moves the line; it never changes the volume."""
    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty speech meter (0 %, threshold line at 50 %)."""
        super().__init__(parent)
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._shown, self._threshold, self._speaking = 0.0, 0.5, False

    def update_score(self, prob: float, threshold: float, speaking: bool) -> None:
        """Show a new speech score (0-1), the threshold it must reach, and
        whether it currently counts as speech (bar turns green)."""
        self._shown = max(prob, self._shown * 0.8)   # rise at once, fall gently
        self._threshold, self._speaking = threshold, speaking
        self.update()

    @property
    def shown(self) -> float:
        return self._shown

    def paintEvent(self, _: QPaintEvent) -> None:
        """Qt calls this to draw the bar, the white threshold line and the
        0-100 % scale."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, x1 = 4, self.width() - 4
        span = x1 - x0
        p.setPen(Qt.PenStyle.NoPen)
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
                       Qt.AlignmentFlag.AlignCenter, f"{t}%")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MetricCard(HeaderCardWidget):
    """A titled card with rows of "name   value", where each value can be
    coloured green/amber/red to show whether it is healthy."""
    def __init__(self, title: str, rows: list[tuple[str, str]],  # pyright: ignore[reportIncompatibleVariableOverride]
                 parent: QWidget | None = None) -> None:
        """Build the card: one 'name   value' row per (name, tooltip) in rows."""
        super().__init__(parent)  # pyright: ignore[reportUnknownMemberType]
        self.setTitle(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        self._values: dict[str, BodyLabel] = {}  # row name -> value label
        for r, (key, tip) in enumerate(rows):
            name = CaptionLabel(key)
            name.setToolTip(tip)
            value = BodyLabel("-")
            value.setToolTip(tip)
            grid.addWidget(name, r, 0)
            grid.addWidget(value, r, 1)
            self._values[key] = value
        grid.setColumnStretch(1, 1)
        self.viewLayout.addLayout(grid)

    @property
    def values(self) -> dict[str, BodyLabel]:
        return self._values

    def set(self, key: str, text: object, colour: str | None = None) -> None:
        """Show `text` in the row `key`, optionally coloured (OK/WARN/BAD)."""
        label = self._values[key]
        label.setText(str(text))
        set_colour(label, colour)


def set_colour(label: BodyLabel | CaptionLabel, colour: str | None = None) -> None:
    """Colour a Fluent label, or restore the theme's normal text colour.
    (Clearing a style sheet instead would also drop the theme colour and
    leave near-black text on the dark background.)"""
    if colour:
        label.setTextColor(QColor(colour), QColor(colour))   # (light theme, dark theme)
    else:
        label.setTextColor(QColor(0, 0, 0), QColor(255, 255, 255))


def judge(value: float | None, ok_below: float, warn_below: float) -> str:
    """Colour for a number where smaller is better."""
    if value is None:
        return MUTED
    return OK if value < ok_below else WARN if value < warn_below else BAD


def pct(values: Sequence[float | None], q: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    return v[min(len(v) - 1, int(len(v) * q))] if v else None


class ReportDialog(MessageBoxBase):
    """A pop-up showing the text of a log_tools.py report."""
    def __init__(self, title: str, text: str, parent: QWidget | None = None) -> None:
        """A read-only box in a fixed-width font, with a Close button."""
        super().__init__(parent)  # pyright: ignore[reportUnknownMemberType]
        self.viewLayout.addWidget(SubtitleLabel(title))
        box = TextEdit()
        box.setReadOnly(True)
        box.setFont(QFont("Consolas", 9))
        box.setPlainText(text)
        box.setMinimumSize(900, 520)
        self.viewLayout.addWidget(box)
        self.yesButton.setText("Close")
        self.cancelButton.hide()


class ReportDialogMistake(MessageBoxBase):
    """"Report a mistake": which phrase (newest first), what was wrong, and
    optionally what was really said and in which language."""
    def __init__(self, recent: list[tuple[Hashable, dict[str, Any]]],
                 parent: QWidget | None = None) -> None:
        """Build the dialog: the phrase list (`recent`: (key, details) pairs,
        newest first), one tick box per kind of problem, the correct text,
        the real language and a note."""
        super().__init__(parent)  # pyright: ignore[reportUnknownMemberType]
        self.viewLayout.addWidget(SubtitleLabel("Report a mistake"))
        self.viewLayout.addWidget(CaptionLabel("Which phrase?"))
        self.phrase_combo = ComboBox()
        self._keys = [key for key, _ in recent]
        for _, e in recent:
            who = f"{e['who']}: " if e.get("who") else ""
            cast(QComboBox, self.phrase_combo).addItem(
                f"{e['time']}  {who}{e['text'][:90] or '(empty)'}"
            )
        self.phrase_combo.setMinimumWidth(640)
        self.viewLayout.addWidget(self.phrase_combo)
        self.viewLayout.addWidget(CaptionLabel("What was wrong? (tick all that apply)"))
        self.checks: dict[str, CheckBox] = {}
        for key, label in core.FEEDBACK_PROBLEMS.items():
            c = CheckBox(label)
            self.checks[key] = c
            self.viewLayout.addWidget(c)
        self.viewLayout.addWidget(CaptionLabel("What was really said (optional, but it makes "
                                               "the report most useful)"))
        self.correct_edit = LineEdit()
        self.correct_edit.setPlaceholderText("The correct words, in the language spoken")
        self.viewLayout.addWidget(self.correct_edit)
        self.viewLayout.addWidget(CaptionLabel("Language actually spoken"))
        self.lang_combo = ComboBox()
        self._langs: list[tuple[str, str] | tuple[str, None]] = [("(as detected)", None)] + sorted(
            ((name, code) for code, name in core.LANGUAGE_NAMES.items()), key=lambda x: x[0])
        self.lang_combo.addItems([n for n, _ in self._langs])
        self.viewLayout.addWidget(self.lang_combo)
        self.note_edit = LineEdit()
        self.note_edit.setPlaceholderText("Anything else (optional)")
        self.viewLayout.addWidget(self.note_edit)
        self.yesButton.setText("Save report")
        self.cancelButton.setText("Cancel")

    def validate(self) -> bool:
        return any(c.isChecked() for c in self.checks.values())

    def result_values(self) -> tuple[Hashable, list[str], str, str | None, str]:
        """What the user entered: (phrase key, problem keys ticked, correct
        text, language code or None, note)."""
        return (self._keys[self.phrase_combo.currentIndex()],
                [k for k, c in self.checks.items() if c.isChecked()],
                self.correct_edit.text(), self._langs[self.lang_combo.currentIndex()][1],
                self.note_edit.text())


# ─────────────────────────────────────────────────────────────────────────────
# Two-pane live text (used by the simulator window)
# ─────────────────────────────────────────────────────────────────────────────

class PaneText:
    """Live preview and final text in a pair of "Heard" / "Translation" boxes.

    The interpreter window used to show its text this way; it now shows one
    conversation (ConversationView). The simulator window (simulator_qt.py)
    still uses two boxes and borrows these methods. The class that uses them
    needs `heard_box`, `_live` and `_last_preview_words`.

    The preview line is replaced each time; the words two previews agree on
    are shown normally, the rest grey; the final text replaces the preview.
    """
    heard_box: TextEdit
    _live: dict[TextEdit, tuple[int, int]]
    _last_preview_words: dict[TextEdit, tuple[int | None, list[str]]]

    @staticmethod
    def _at_bottom(box: TextEdit) -> bool:
        """True if the box is scrolled to the end. New text only scrolls the
        view if so: someone re-reading earlier text isn't yanked back down."""
        bar = box.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    @staticmethod
    def _follow(box: TextEdit) -> None:
        box.verticalScrollBar().setValue(box.verticalScrollBar().maximum())

    def _fmt(self, grey: bool) -> QTextCharFormat:
        """Text format for the panes: grey for still-changing preview words,
        the normal text colour otherwise."""
        f = QTextCharFormat()
        f.setForeground(QColor("#8a8f94") if grey else self.heard_box.palette().text().color())
        return f

    def _set_live(self, box: TextEdit, seq: int, text: str) -> None:
        """Show (or replace) phrase seq's preview at the end of a pane. Words
        the last two previews agree on are shown normally, the rest grey."""
        follow = self._at_bottom(box)
        start, live_seq = self._live.get(box, (None, None))
        if live_seq is not None and seq < live_seq:
            return
        cur = box.textCursor()
        if start is not None:
            cur.setPosition(start)
            cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            cur.removeSelectedText()
        else:
            cur.movePosition(QTextCursor.MoveOperation.End)
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

    def _commit(self, box: TextEdit, seq: int, text: str) -> None:
        """Replace phrase seq's preview with its final text. A phrase that
        continues the previous sentence is joined onto the same line."""
        follow = self._at_bottom(box)
        start, live_seq = self._live.get(box, (None, None))
        cur = QTextCursor(box.document())
        if start is not None and live_seq is not None and live_seq <= seq:
            cur.setPosition(start)
            cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
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
        cur.setPosition(anchor, QTextCursor.MoveMode.KeepAnchor)
        cur.insertText(insert, self._fmt(False))     # replaces the "..." if any
        if live:
            self._live[box] = (pos + len(insert), live[1])
        else:
            self._trim(box)
        if follow:
            self._follow(box)

    @staticmethod
    def _trim(box: TextEdit) -> None:
        """Keep at most MAX_TEXT_HISTORY_CHARS characters (oldest removed)."""
        doc = box.document()
        extra = doc.characterCount() - MAX_TEXT_HISTORY_CHARS
        if extra > 0:
            cur = QTextCursor(doc)
            cur.setPosition(0)
            cur.setPosition(extra, QTextCursor.MoveMode.KeepAnchor)
            cur.removeSelectedText()


# ─────────────────────────────────────────────────────────────────────────────
# Channels: the audio sources the interpreter listens to at the same time
# ─────────────────────────────────────────────────────────────────────────────

# One colour per channel, used for its name in the conversation and its strip
CHANNEL_COLOURS = ("#4FC3F7", "#FFB74D")      # light blue, amber


@dataclass
class ChannelSettings:
    """One channel's choices, saved in interpreter.settings.json under
    "channels" (a list with one dict per channel).

    name         shown in the conversation, e.g. "Me" or "Friend"
    device       input device label, e.g. "Microphone (RODECaster Duo …)" or
                 "🔊 Speakers (FiiO Q series) [Loopback]" (what an output plays)
    src, tgt     language names: spoken ("Auto-detect" allowed) -> shown
    sensitivity  0..1, how readily sound counts as speech
    muted        True = this channel is switched off
    speak        True = read this channel's translations aloud
    output       output device label for Speak and replay
    replay       True = play the input through `output` (VB-Audio cables)
    """
    name: str
    device: str
    src: str
    tgt: str
    sensitivity: float = DEFAULT_SENS
    muted: bool = False
    speak: bool = False
    output: str = ""
    replay: bool = False


def load_channel_settings(input_devices: list[tuple[int, str]],
                          output_devices: list[tuple[int, str]]) -> list[ChannelSettings]:
    """The two channels' settings: the saved ones, or sensible defaults.

    The defaults follow the old two-mode design: channel 1 ("Me") is your
    microphone, English → Russian; channel 2 ("Friend") is the call audio,
    Russian → English. Devices saved by the old version ("mode_devices")
    are reused, so an upgrade keeps your choices.
    """
    saved = load_settings()
    in_labels = [lbl for _, lbl in input_devices]
    out_labels = [lbl for _, lbl in output_devices]
    first_in = in_labels[0] if in_labels else ""
    old_devices = [d for d in saved.get("mode_devices", []) if isinstance(d, str)]
    me_device = ((old_devices[0] if len(old_devices) > 0 else None)
                 or find_device_by_keywords(input_devices, NVIDIA_KEYWORDS) or first_in)
    friend_device = ((old_devices[1] if len(old_devices) > 1 else None)
                     or find_best_friend_input(input_devices) or first_in)
    headphones = find_best_output_device(output_devices) or (out_labels[0] if out_labels else "")
    vb_cable = find_best_vb_input(input_devices)
    sensitivity = float(saved.get("sensitivity", DEFAULT_SENS))
    channels = [
        ChannelSettings("Me", me_device, "English", "Russian", sensitivity, output=headphones),
        ChannelSettings("Friend", friend_device, "Russian", "English", sensitivity,
                        output=headphones, replay=bool(vb_cable) and friend_device == vb_cable),
    ]
    known = {f.name for f in fields(ChannelSettings)}
    for i, stored in enumerate(saved.get("channels", [])[:len(channels)]):
        if not isinstance(stored, dict):
            continue
        # PYTHON: dataclasses.replace() makes a copy with some fields changed.
        # Only known keys are used, so an old or hand-edited file can't crash it.
        channel = replace(channels[i], **{k: v for k, v in stored.items() if k in known})
        if channel.src not in LANGUAGES:
            channel.src = channels[i].src
        if channel.tgt not in core.TARGET_LANGUAGES:
            channel.tgt = channels[i].tgt
        channels[i] = channel
    return channels


class Channel:
    """One audio source the interpreter listens to, and everything that
    belongs to it while the app runs.

    Each channel has its own pipeline (a SessionController: capture thread,
    transcription worker, translation), its own text-to-speech Speaker and
    its own music detector. The Whisper model is shared: both workers take
    turns on the GPU (INFERENCE_LOCK in interpreter.py).

    run_id changes every time the channel (re)starts. Messages from the
    channel's threads arrive labelled (index, run_id), so a late message
    from a pipeline that has just been stopped can be recognised.
    Phrases are identified by the key (index, run_id, seq).
    """

    def __init__(self, index: int, settings: ChannelSettings, result_queue: queue.Queue[Any]) -> None:
        self.index = index
        self.settings = settings
        self.colour = CHANNEL_COLOURS[index % len(CHANNEL_COLOURS)]
        self.controller = SessionController(result_queue, core)
        self.speaker = Speaker()
        self.speaker.start()
        self.listening = False
        self.run_id = 0
        self.music_scene: Any = None
        self.problem: str | None = None    # why it isn't listening (shown on its strip)
        self.failures = 0                  # restarts in a row (the watchdog waits longer each time)
        self.retry_at = 0.0                # time.monotonic() of the next start attempt
        self.needs_rescan = False          # re-scan devices before the next attempt
        self.missing_checks = 0            # looks for a missing device (they get rarer)
        self.first_preview: dict[int, float] = {}          # seq -> ms from speech to first preview
        self.pending_translation: dict[int, dict[str, Any]] = {}   # seq -> trace (online)
        self.levels: deque[float] = deque(maxlen=320)      # recent block levels (~10 s)
        self.speech_flags: deque[bool] = deque(maxlen=320)  # recent speech yes/no per block
        self.peak_recent = -100.0
        self.sens_log_after: int | None = None
        self.strip: ChannelStrip       # set when the page is built
        self.controls: ChannelControls

    @property
    def capture(self) -> core.AudioCaptureThread | None:
        return self.controller.capture_thread

    @property
    def workers(self) -> list[core.TranscriptionWorker]:
        return self.controller.workers


@dataclass
class ChannelControls:
    """One channel's card on the Settings page and its controls."""
    card: HeaderCardWidget
    name_edit: LineEdit
    device_combo: ComboBox
    src_combo: ComboBox
    tgt_combo: ComboBox
    sens_slider: Slider
    sens_value: BodyLabel
    out_combo: ComboBox
    replay_check: CheckBox


# Added to a device menu entry whose device isn't connected now
NOT_CONNECTED = "  (not connected)"


def _fill_combo(combo: ComboBox, labels: list[str], current: str) -> None:
    """Replace a combo box's items and select `current`. A device that is
    unplugged right now stays in the list, marked "(not connected)", so the
    box still shows the channel's real choice (it reconnects when the device
    comes back) without looking like a device that works.
    Signals are blocked: refilling is not the user choosing something."""
    combo.blockSignals(True)
    combo.clear()
    shown = current if current in labels or not current else current + NOT_CONNECTED
    combo.addItems(labels + ([shown] if shown != current else []))
    _set_combo_text(combo, shown)
    combo.blockSignals(False)


def _combo_device(combo: ComboBox) -> str:
    """The device label chosen in a device menu (without "(not connected)")."""
    return combo.currentText().removesuffix(NOT_CONNECTED)


def similar_devices(label: str, labels: Sequence[str]) -> list[str]:
    """Connected devices that look like the same hardware as `label`: they
    share a distinctive word from its brackets, e.g. "Realtek" in
    "Digital Output (Realtek USB2.0 Audio)" and "SPDIF Interface (Realtek
    USB Audio)". Windows renames devices after driver updates or when they
    are plugged into another port, so this helps to find the new name."""
    common = {"audio", "usb", "device", "high", "definition", "stereo", "main"}
    inside = label[label.find("(") + 1:label.rfind(")")] if "(" in label else label
    words = {w for w in inside.lower().replace(".", " ").split() if len(w) > 2} - common
    same_kind = "[loopback]" in label.lower()
    return [l for l in labels if l != label and ("[loopback]" in l.lower()) == same_kind
            and any(w in l.lower() for w in words)]


class ChannelStrip(SimpleCardWidget):
    """The always-visible summary of one channel above the conversation:
    its name in its colour, device and languages, the level and speech
    meters, what it is doing now, and its Speak and Mute switches."""

    def __init__(self, colour: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)  # pyright: ignore[reportUnknownMemberType]
        grid = QGridLayout(self)
        grid.setContentsMargins(12, 8, 12, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        bar = QFrame()                      # the coloured stripe on the left
        bar.setFixedWidth(4)
        bar.setStyleSheet(f"background:{colour}; border-radius:2px;")
        self.name_label = StrongBodyLabel("")
        self.name_label.setTextColor(QColor(colour), QColor(colour))
        self.detail_label = CaptionLabel("")
        self.speak_btn = TogglePushButton()
        self.speak_btn.setText("Speak")
        self.speak_btn.setIcon(FIF.VOLUME)
        self.speak_btn.setToolTip("Read this channel's translations aloud\n"
                                  "(output device: Settings → Speak and replay to)")
        self.mute_btn = TogglePushButton()
        self.mute_btn.setText("Mute")
        self.mute_btn.setIcon(FIF.MUTE)
        self.mute_btn.setToolTip("Stop listening to this channel (releases the device)")
        self.settings_btn = TransparentToolButton(FIF.SETTING)
        self.settings_btn.setToolTip("Device, languages and sensitivity (the Settings page)")
        # Top row: name and buttons; the device and languages get a line of
        # their own below (in one row they ran into the Speak button)
        top = QHBoxLayout()
        top.addWidget(self.name_label)
        top.addStretch(1)
        top.addWidget(self.speak_btn)
        top.addWidget(self.mute_btn)
        top.addWidget(self.settings_btn)
        self.meter = LevelMeter()
        self.speech_meter = SpeechMeter()
        self.level_label = CaptionLabel("-")
        self.speech_label = CaptionLabel("-")
        for label in (self.level_label, self.speech_label):
            label.setFont(QFont("Consolas", 9))
            label.setMinimumWidth(120)
        self.status_label = CaptionLabel("Starting…")
        self.status_label.setWordWrap(True)          # long problem messages wrap
        grid.addWidget(bar, 0, 0, 5, 1)
        grid.addLayout(top, 0, 1, 1, 2)
        grid.addWidget(self.detail_label, 1, 1, 1, 2)
        grid.addWidget(self.meter, 2, 1)
        grid.addWidget(self.level_label, 2, 2)
        grid.addWidget(self.speech_meter, 3, 1)
        grid.addWidget(self.speech_label, 3, 2)
        grid.addWidget(self.status_label, 4, 1, 1, 2)
        grid.setColumnStretch(1, 1)

    def show_settings(self, s: ChannelSettings) -> None:
        """Show the channel's name, device and languages."""
        self.name_label.setText(s.name)
        device = ("🔊 " if "[loopback]" in s.device.lower() else "🎙 ") + plain_device_name(s.device)
        short = device if len(device) <= 60 else device[:58] + "…"   # its own line now
        self.detail_label.setText(f"{short}  ·  {s.src} → {s.tgt}")
        self.detail_label.setToolTip(f"{s.device}\n{s.src} → {s.tgt}")
        self.mute_btn.setChecked(s.muted)
        self.mute_btn.setText("Unmute" if s.muted else "Mute")
        self.mute_btn.setIcon(FIF.MICROPHONE if s.muted else FIF.MUTE)
        self.speak_btn.setChecked(s.speak)
        self.speak_btn.setText("Speak: On" if s.speak else "Speak")

    def set_status(self, text: str, colour: str | None = None) -> None:
        self.status_label.setText(text)
        set_colour(self.status_label, colour)

    def update_levels(self, rms_db: float, peak_db: float, speaking: bool,
                      prob: float, threshold: float) -> None:
        """Show one audio block's level and speech score."""
        self.meter.update_level(rms_db, peak_db)
        self.speech_meter.update_score(prob, threshold, speaking)
        self.level_label.setText("  -∞ dBFS" if rms_db <= -99 else f"{rms_db:6.1f} dBFS")
        self.speech_label.setText(f"{prob * 100:3.0f}% / {threshold * 100:.0f}% "
                                  f"{'SPEECH' if speaking else ''}")
        set_colour(self.speech_label, OK if speaking else None)

    def reset_meters(self) -> None:
        self.meter.reset()
        self.speech_meter.update_score(0.0, 0.5, False)
        self.level_label.setText("-")
        self.speech_label.setText("-")


# ─────────────────────────────────────────────────────────────────────────────
# The conversation: both channels' phrases in one transcript
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Entry:
    """One phrase in the conversation (see ConversationView).

    text_block / translation_block  its two paragraphs in the document
    heard        what was heard (the preview, then the final text)
    stable       preview only: how many leading words two previews agreed on
    final        True once the final text has replaced the preview
    translation  None while there is none (yet); "" if none was possible
    show_label   False when the line above is the same speaker (no repeat)
    """
    channel: int
    time: str
    t: float
    text_block: QTextBlock
    translation_block: QTextBlock
    heard: str = ""
    stable: int = 0
    final: bool = False
    translation: str | None = None
    translation_final: bool = False
    show_label: bool = True


class ConversationView(TextEdit):
    """The conversation: every phrase from both channels, in the order the
    phrases started. Each phrase is an ENTRY of two lines (paragraphs):

        10:02  Friend   Да, слышу хорошо.
                        → Yes, I hear you well.

    The name is left out when the same person carries on. While someone is
    still talking their entry shows the live preview in grey; the final text
    then replaces it IN PLACE. So when both people talk at once, each keeps
    their own entry and nothing gets mixed up.

    Qt notes: a QTextDocument is made of paragraphs called BLOCKS. A
    QTextBlock is a handle to one: it stays valid while other blocks are
    added or removed, and block.position() always gives its current place.
    That is how an entry can still be found and changed later, e.g. when
    its translation arrives from the online service a second afterwards.
    The document always starts and ends with an empty block ("sentinels"):
    entries are only ever added or removed as whole blocks BETWEEN other
    blocks, so an entry's block is never merged into a neighbour.
    The hanging indent (names on the left, wrapped text lined up to the
    right of them) is a tab stop plus a negative first-line indent.
    """
    GREY = "#8a8f94"
    MAX_ENTRIES = 300          # older phrases are removed from the top
    SAME_TURN_SEC = 60         # same speaker within this gap: no repeated name

    def __init__(self, names: Sequence[str], colours: Sequence[str],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.names, self.colours = list(names), list(colours)
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()   # in document order
        self._newest_preview: dict[int, Hashable] = {}                  # channel -> key
        self._indent = 120.0
        self._text_pt = 13
        self.reset()

    # ── public interface (called by the window) ──────────────────────────

    def reset(self) -> None:
        """Remove everything (the Clear button)."""
        self._entries.clear()
        self._newest_preview.clear()
        self.document().clear()
        cursor = QTextCursor(self.document())
        cursor.insertBlock()                        # [top sentinel][bottom sentinel]
        self._bottom = self.document().lastBlock()

    def show_preview(self, key: Hashable, channel: int, heard: str,
                     translated: str | None) -> None:
        """Show (or update) the live preview of a phrase still being spoken."""
        newest = self._newest_preview.get(channel)
        if newest is not None and key < newest:     # type: ignore[operator]
            return                                  # older than a preview already shown
        entry = self._entries.get(key)
        if entry is not None and entry.final:
            return                                  # its final text is already shown
        if entry is None and not heard.strip():
            return
        follow = self._at_bottom()
        if entry is None:
            entry = self._new_entry(key, channel)
        words, old = heard.split(), entry.heard.split()
        stable = 0
        while stable < min(len(words), len(old)) and words[stable] == old[stable]:
            stable += 1
        entry.heard, entry.stable = heard, stable
        if translated:
            entry.translation = translated
        self._newest_preview[channel] = key
        self._render(entry)
        if follow:
            self._follow()

    def commit(self, key: Hashable, channel: int, heard: str, translation: str | None) -> None:
        """Show a phrase's final text. translation None = still to come
        (online); an empty `heard` (noise, not speech) removes the entry."""
        entry = self._entries.get(key)
        if not heard:
            if entry is not None:
                self._remove([key])
            return
        follow = self._at_bottom()
        if entry is None:
            entry = self._new_entry(key, channel)
        entry.heard, entry.final = heard, True
        entry.translation, entry.translation_final = translation, translation is not None
        self._render(entry)
        self._trim()
        if follow:
            self._follow()

    def set_translation(self, key: Hashable, text: str) -> None:
        """A translation arrived later (online service)."""
        entry = self._entries.get(key)
        if entry is None:
            return
        follow = self._at_bottom()
        entry.translation, entry.translation_final = text, True
        self._render(entry)
        if follow:
            self._follow()

    def drop_previews(self, channel: int) -> None:
        """Remove a channel's unfinished previews (the channel stopped, so
        their final text will never come)."""
        self._remove([k for k, e in self._entries.items() if e.channel == channel and not e.final])
        self._newest_preview.pop(channel, None)

    def set_names(self, names: Sequence[str]) -> None:
        """A channel was renamed: redraw every entry with the new names."""
        self.names = list(names)
        self._layout_changed()

    def set_text_size(self, points: int) -> None:
        """The Fluent theme's stylesheet fixes the font size, which overrides
        setFont(), so the size goes in the widget's own custom stylesheet."""
        self._text_pt = points
        qss = f"TextEdit {{ font: {points}pt 'Segoe UI'; }}"
        setCustomStyleSheet(self, qss, qss)
        self._layout_changed()

    def plain_entries(self) -> list[tuple[int, str, str | None, bool]]:
        """(channel, heard, translation, final) for each entry; for tests."""
        return [(e.channel, e.heard, e.translation, e.final) for e in self._entries.values()]

    # ── drawing ──────────────────────────────────────────────────────────

    def _layout_changed(self) -> None:
        """Recalculate the name column's width and redraw every entry."""
        bold = QFont("Segoe UI", self._text_pt)
        bold.setBold(True)
        metrics = QFontMetricsF(bold, self.viewport())   # measured for this screen's DPI
        widest = max((metrics.horizontalAdvance("00:00  " + n) for n in self.names), default=80.0)
        self._indent = widest + metrics.horizontalAdvance("  ")
        for entry in self._entries.values():
            self._render(entry)

    def _char(self, colour: str | None = None, bold: bool = False) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setForeground(QColor(colour) if colour else self.palette().text().color())
        f.setFontWeight(QFont.Weight.Bold if bold else QFont.Weight.Normal)
        return f

    def _block_format(self, top_margin: float = 0.0) -> QTextBlockFormat:
        """Hanging indent: the first line starts at the left edge, a tab
        jumps to the text column, wrapped lines start in the text column."""
        f = QTextBlockFormat()
        f.setLeftMargin(self._indent)
        f.setTextIndent(-self._indent)
        f.setTabPositions([QTextOption.Tab(self._indent, QTextOption.TabType.LeftTab)])
        f.setTopMargin(top_margin)
        return f

    @staticmethod
    def _write(block: QTextBlock, pieces: list[tuple[str, QTextCharFormat]],
               fmt: QTextBlockFormat) -> None:
        """Replace one block's text with `pieces` (text, format) and set its format."""
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.setBlockFormat(fmt)
        for text, char_format in pieces:
            cursor.insertText(text, char_format)

    def _render(self, entry: _Entry) -> None:
        """Draw an entry's two lines from its fields."""
        colour = self.colours[entry.channel % len(self.colours)]
        name = self.names[entry.channel] if entry.channel < len(self.names) else "?"
        line: list[tuple[str, QTextCharFormat]] = []
        if entry.show_label:
            line += [(entry.time + "  ", self._char(self.GREY)), (name, self._char(colour, bold=True))]
        line.append(("\t", self._char()))
        if entry.final:
            line.append((entry.heard, self._char()))
        else:
            words = entry.heard.split()
            stable = " ".join(words[:entry.stable])
            line += [("… " + stable + (" " if stable else ""), self._char()),
                     (" ".join(words[entry.stable:]), self._char(self.GREY))]
        self._write(entry.text_block, line, self._block_format(8.0 if entry.show_label else 0.0))
        if entry.translation is None:
            translation = [("\t→ …", self._char(self.GREY))]
        else:
            translation = [("\t", self._char()), ("→ ", self._char(colour)),
                           (entry.translation or "(no translation)",
                            self._char(None if entry.translation_final else self.GREY))]
        self._write(entry.translation_block, translation, self._block_format())

    # ── adding and removing entries ──────────────────────────────────────

    def _new_entry(self, key: Hashable, channel: int) -> _Entry:
        """Add an empty entry at the end (just above the bottom sentinel)."""
        cursor = QTextCursor(self._bottom.previous())
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        cursor.insertBlock()
        text_block = cursor.block()
        cursor.insertBlock()
        entry = _Entry(channel, time.strftime("%H:%M"), time.monotonic(),
                       text_block, cursor.block())
        previous = next(reversed(self._entries.values()), None)
        entry.show_label = self._needs_label(previous, entry)
        self._entries[key] = entry
        return entry

    def _needs_label(self, previous: _Entry | None, entry: _Entry) -> bool:
        return (previous is None or previous.channel != entry.channel
                or entry.t - previous.t > self.SAME_TURN_SEC)

    def _remove(self, keys: list[Hashable]) -> None:
        """Remove entries (both their blocks), then fix the name labels of
        the entries that now follow a different speaker."""
        if not keys:
            return
        for key in keys:
            entry = self._entries.pop(key)
            cursor = QTextCursor(entry.text_block)
            cursor.setPosition(entry.translation_block.next().position(),
                               QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
        previous = None
        for entry in self._entries.values():
            label = self._needs_label(previous, entry)
            if label != entry.show_label:
                entry.show_label = label
                self._render(entry)
            previous = entry

    def _trim(self) -> None:
        """Keep at most MAX_ENTRIES entries; the oldest finished ones go."""
        extra = len(self._entries) - self.MAX_ENTRIES
        if extra > 0:
            self._remove([k for k, e in self._entries.items() if e.final][:extra])

    def _at_bottom(self) -> bool:
        """True if scrolled to the end. New text only scrolls the view if so:
        someone re-reading earlier text isn't yanked back down."""
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def _follow(self) -> None:
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# ─────────────────────────────────────────────────────────────────────────────
# The main window
# ─────────────────────────────────────────────────────────────────────────────

class QtInterpreterApp(FluentWindow):
    """The Fluent main window: an Interpreter page and a Diagnostics page in a
    side navigation bar.

    It listens to two audio devices at the same time ("channels", see
    Channel). Each channel has its own SessionController pipeline; both
    share one Whisper model. Their results arrive in one queue, labelled
    with the channel, and appear in one conversation.
    """

    # ── Borrowed logic (see the file's opening comment) ──────────────────
    # These only use the model settings, which both channels share.
    _preload_model            = InterpreterApp._preload_model  # pyright: ignore[reportPrivateUsage]
    _on_preview_choice        = InterpreterApp._on_preview_choice  # pyright: ignore[reportPrivateUsage]
    _ensure_preview_model     = InterpreterApp._ensure_preview_model  # pyright: ignore[reportPrivateUsage]
    _translator_for           = InterpreterApp._translator_for  # pyright: ignore[reportPrivateUsage]
    _preview_model            = InterpreterApp._preview_model  # pyright: ignore[reportPrivateUsage]
    _update_metric            = InterpreterApp._update_metric  # pyright: ignore[reportPrivateUsage]
    _open_path                = InterpreterApp._open_path  # pyright: ignore[reportPrivateUsage]

    def __init__(self) -> None:
        """Set up the state, find the audio devices, build both pages, place
        the window and start listening on both channels."""
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.setWindowTitle("Voice Interpreter")
        self._timers: dict[int, QTimer] = {}
        self._timer_id = 0                       # for after()/after_cancel()

        # ── state ────────────────────────────────────────────────────────
        self.session_id = str(uuid.uuid4())
        self.result_queue: queue.Queue[Any] = queue.Queue()  # background threads -> window
        self._perf_stats: dict[str, Any] = {
            "transcriptions": 0, "translations": 0,
            "transcription_ms": deque(maxlen=200),
            "translation_ms": deque(maxlen=200),
            "lock_wait_ms": deque(maxlen=200),
        }
        self._model_loader: core.WhisperModelLoader | None = None
        self._whisper_model: Any = None
        self._run_id = 0                         # counts every channel start (unique run ids)
        self._ui_stats = PerfWindow()
        self._last_poll_t: float | None = None
        self._diag_psutil_process: Any = None
        self._preview_loader: core.WhisperModelLoader | None = None
        self._preview_enabled = False
        self._previews_off = False
        self._mt_loaders: dict[tuple[str, str], core.TranslatorLoader] = {}
        self._initialised = False
        self._closing = False                    # set when the window closes
        self._start_retry_after_id: int | None = None

        # ── diagnostics data ─────────────────────────────────────────────
        self._started_at = time.monotonic()     # when this window opened
        self._listening_since: float | None = None
        self._phrases: deque[dict[str, Any]] = deque(maxlen=50)   # recent finished phrases (timings, text)
        self._recent_finals: OrderedDict[Hashable, dict[str, Any]] = OrderedDict()  # "Report a mistake"
        # recent warnings/errors: (time, level, event, detail, explanation)
        self._issues: deque[tuple[str, str, str, str, dict[str, Any]]] = deque(maxlen=40)
        self._issue_count_seen = 0
        self._issue_groups: dict[str, dict[str, Any]] = {}  # event -> count, times, explanation
        self._watchdog_restarts = 0
        self._nvml: Any = None                  # GPU monitoring handle
        core.add_issue_sink(IssueHandler(self._issues))

        # ── audio devices and the two channels ───────────────────────────
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map: dict[str, int] = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map: dict[str, int] = {lbl: idx for idx, lbl in self._output_devices}
        self.channels = [Channel(i, s, self.result_queue) for i, s in
                         enumerate(load_channel_settings(self._input_devices, self._output_devices))]

        # ── pages ────────────────────────────────────────────────────────
        self.interp_page = self._build_interpreter_page()
        self.settings_page = self._build_settings_page()
        self.diag_page = self._build_diagnostics_page()
        self.addSubInterface(self.interp_page, FIF.CHAT, "Interpreter")  # pyright: ignore[reportUnknownMemberType]
        self.addSubInterface(self.diag_page, FIF.DEVELOPER_TOOLS, "Diagnostics")  # pyright: ignore[reportUnknownMemberType]
        # Settings at the bottom of the side bar, as in Windows 11's own apps
        self.addSubInterface(self.settings_page, FIF.SETTING, "Settings",  # pyright: ignore[reportUnknownMemberType]
                             NavigationItemPosition.BOTTOM)
        self._place_window()

        log_event(logging.INFO, "app_initialized", session_id=self.session_id, gui="qt",
                  whisper_device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE,
                  num_workers=NUM_WORKERS, input_device_count=len(self._input_devices),
                  output_device_count=len(self._output_devices),
                  channels=[asdict(c.settings) for c in self.channels])
        self._preload_model()
        self._ensure_preview_model()
        self._initialised = True
        self._start_all()
        self.after(15, self._poll_queue)
        self.after(WATCHDOG_SEC * 1000, self._watchdog)
        self.after(STATS_INTERVAL_SEC * 1000, self._log_resources)
        self.after(1000, self._refresh_diagnostics)

    @property
    def is_listening(self) -> bool:
        """True while at least one channel is listening."""
        return any(c.listening for c in self.channels)

    # ── Tk-style timer (used by the borrowed logic) ──────────────────────

    def after(self, ms: int, fn: Callable[[], None]) -> int:
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

    def after_cancel(self, tid: int) -> None:
        """Cancel a timer started with after() (if it hasn't fired yet)."""
        timer = self._timers.pop(tid, None)
        if timer is not None:
            timer.stop()

    # ── Window placement ─────────────────────────────────────────────────

    def _place_window(self) -> None:
        """Size the window to fit the usable screen area (taskbar excluded)
        and centre it. Qt handles display scaling itself, so sizes here are
        in scaled ("logical") pixels."""
        area = QGuiApplication.primaryScreen().availableGeometry()
        w, h = min(1180, int(area.width() * 0.95)), min(920, int(area.height() * 0.92))
        self.resize(w, h)
        self.move(area.x() + (area.width() - w) // 2, area.y() + (area.height() - h) // 2)
        log_event(logging.INFO, "window_placed", gui="qt", size_logical=[w, h],
                  screen_area=[area.x(), area.y(), area.width(), area.height()])

    # ── Interpreter page ─────────────────────────────────────────────────

    def _build_interpreter_page(self) -> QWidget:
        """Build the Interpreter page, top to bottom: the header with the
        action buttons, one strip per channel, the conversation, and the
        status line. The settings are on their own page (_build_settings_page)."""
        page = QWidget()
        page.setObjectName("interpreterPage")   # FluentWindow needs a unique name
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 16, 24, 12)
        root.setSpacing(10)

        # Header: title, GPU/CPU badge, status pill, actions
        header = QHBoxLayout()
        header.addWidget(TitleLabel("Voice Interpreter"))
        self.hw_badge = self._pill(" GPU " if WHISPER_DEVICE == "cuda" else " CPU ",
                                   OK if WHISPER_DEVICE == "cuda" else "#2980B9")
        self.status_badge = self._pill(" IDLE ", WARN)
        header.addWidget(self.hw_badge)
        header.addWidget(self.status_badge)
        header.addStretch(1)
        clear_btn = PushButton()
        clear_btn.setText("Clear")
        clear_btn.setIcon(FIF.BROOM)
        clear_btn.clicked.connect(self._clear)
        report_btn = PushButton()
        report_btn.setText("Report a mistake")
        report_btn.setIcon(FIF.FEEDBACK)
        report_btn.setToolTip("Mark a recent phrase as wrong. Its audio and what went wrong\n"
                              "are saved, so the problem can be found and fixed, and the\n"
                              "simulator can replay it to check the fix.")
        report_btn.clicked.connect(self._report_mistake)
        for b in (clear_btn, report_btn):
            header.addWidget(b)
        root.addLayout(header)

        # One strip per channel, side by side. (The settings have their own
        # page, the gear at the bottom of the side bar, so the conversation
        # and the meters get the room.)
        strips = QHBoxLayout()
        strips.setSpacing(10)
        for ch in self.channels:
            ch.strip = ChannelStrip(ch.colour)
            ch.strip.show_settings(ch.settings)
            ch.strip.mute_btn.clicked.connect(partial(_ignore_signal_value,
                                                      callback=partial(self._toggle_channel_mute, ch)))
            ch.strip.speak_btn.clicked.connect(partial(_ignore_signal_value,
                                                       callback=partial(self._toggle_channel_speak, ch)))
            ch.strip.settings_btn.clicked.connect(partial(_ignore_signal_value,
                                                          callback=self._open_settings))
            strips.addWidget(ch.strip, 1)
        root.addLayout(strips)

        # The conversation, with text size buttons (caption guidance:
        # readers must control the size)
        title_row = QHBoxLayout()
        title_row.addWidget(StrongBodyLabel("Conversation"))
        title_row.addStretch(1)
        title_row.addWidget(CaptionLabel("Text size"))
        for text, step in (("A−", -1), ("A+", 1)):
            b = PushButton()
            b.setText(text)
            b.setFixedWidth(52)
            b.setToolTip("Smaller / larger conversation text")
            b.clicked.connect(lambda _=False, d=step: self._change_text_size(d))
            title_row.addWidget(b)
        root.addLayout(title_row)
        self.conversation = ConversationView([c.settings.name for c in self.channels],
                                             [c.colour for c in self.channels])
        self._text_size = int(load_settings().get("text_size", 13))
        self.conversation.set_text_size(self._text_size)
        root.addWidget(self.conversation, 1)

        self.status_label = CaptionLabel("Loading Whisper model…")
        root.addWidget(self.status_label)
        self.status_var = QVar(self.status_label.text, self.status_label.setText)
        return page

    def _build_settings_page(self) -> SmoothScrollArea:
        """The Settings page (gear icon at the bottom of the side bar): one
        card per channel, side by side (name, device, languages, sensitivity,
        output, replay), and a card for what both channels share (model,
        previews, other languages). Changes apply at once; there is no Save
        button. The page scrolls when the window is small."""
        page = SmoothScrollArea()
        page.setObjectName("settingsPage")      # FluentWindow needs a unique name
        page.setWidgetResizable(True)
        page.setStyleSheet("QScrollArea{border:none; background:transparent}")
        body = QWidget()
        body.setStyleSheet("background:transparent")
        page.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(TitleLabel("Settings"))
        top.addStretch(1)
        refresh = PushButton()
        refresh.setText("Re-scan devices")
        refresh.setIcon(FIF.SYNC)
        refresh.setToolTip("Find audio devices again (e.g. after plugging one in)")
        refresh.clicked.connect(partial(_ignore_signal_value, callback=self._refresh_devices))
        top.addWidget(refresh)
        lay.addLayout(top)
        lay.addWidget(CaptionLabel("Changes apply at once. A channel restarts when its device, "
                                   "languages or output change."))

        def label(text: str, tip: str) -> BodyLabel:
            """A settings label with a hover explanation."""
            l = BodyLabel(text)
            l.setToolTip(tip)
            return l

        in_labels = [lbl for _, lbl in self._input_devices]
        out_labels = [lbl for _, lbl in self._output_devices]
        targets = sorted(core.TARGET_LANGUAGES)
        cards = QHBoxLayout()
        cards.setSpacing(12)
        for ch in self.channels:
            s = ch.settings
            card = HeaderCardWidget()
            card.headerLabel.setStyleSheet(f"color: {ch.colour};")   # the channel's colour
            name_edit = LineEdit()
            name_edit.setText(s.name)
            name_edit.setMaxLength(16)
            device_combo = ComboBox()
            _fill_combo(device_combo, in_labels, s.device)
            src_combo, tgt_combo = ComboBox(), ComboBox()
            # The source list starts with "Auto-detect" (the language of each
            # phrase is detected); a target must be a real language.
            src_combo.addItems(["Auto-detect"] + targets)
            tgt_combo.addItems(targets)
            _set_combo_text(src_combo, s.src)
            _set_combo_text(tgt_combo, s.tgt)
            sens_slider = Slider(Qt.Orientation.Horizontal)
            sens_slider.setRange(0, 100)
            sens_slider.setValue(int(round(s.sensitivity * 100)))
            sens_value = BodyLabel(f"{sens_slider.value()}%")
            out_combo = ComboBox()
            _fill_combo(out_combo, out_labels, s.output)
            replay_check = CheckBox("Hear input through output (VB-Audio cables only)")
            replay_check.setChecked(s.replay)
            ch.controls = ChannelControls(card, name_edit, device_combo, src_combo, tgt_combo,
                                          sens_slider, sens_value, out_combo, replay_check)
            self._set_card_title(ch)

            lang_row = QHBoxLayout()
            lang_row.addWidget(src_combo, 1)
            lang_row.addWidget(BodyLabel("→"))
            lang_row.addWidget(tgt_combo, 1)
            sens_row = QHBoxLayout()
            sens_row.addWidget(sens_slider, 1)
            sens_row.addWidget(sens_value)
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(10)
            # One row per setting: (label, hover explanation, control)
            rows: list[tuple[str, str, QWidget | QHBoxLayout]] = [
                ("Name", "Shown in the conversation", name_edit),
                ("Listen to", "Device to listen to. 🔊 = what an output device (e.g. the "
                              "headphones in a call) is playing.", device_combo),
                ("Translate", "Language spoken → language shown", lang_row),
                ("Speech sensitivity", "Higher = also catch quieter/less clear speech. Changes "
                                       "what counts as speech, not the volume.", sens_row),
                ("Speak and replay to", "Output device for Speak (this channel's translations "
                                        "read aloud) and for replay", out_combo),
                ("", "", replay_check),
            ]
            for r, (text, tip, control) in enumerate(rows):
                if text:
                    grid.addWidget(label(text, tip), r, 0)
                if isinstance(control, QHBoxLayout):
                    grid.addLayout(control, r, 1)
                else:
                    grid.addWidget(control, r, 1)
            grid.setColumnStretch(1, 1)
            card.viewLayout.addLayout(grid)
            cards.addWidget(card, 1)

            # ── control events (partial() fixes which channel they are for)
            name_edit.editingFinished.connect(partial(self._on_channel_name, ch))
            device_combo.currentTextChanged.connect(
                partial(_ignore_signal_value, callback=partial(self._on_channel_device, ch)))
            src_combo.currentTextChanged.connect(
                partial(_ignore_signal_value, callback=partial(self._on_channel_languages, ch)))
            tgt_combo.currentTextChanged.connect(
                partial(_ignore_signal_value, callback=partial(self._on_channel_languages, ch)))
            sens_slider.valueChanged.connect(partial(self._on_channel_sensitivity, ch))
            out_combo.currentTextChanged.connect(
                partial(_ignore_signal_value, callback=partial(self._on_channel_output, ch)))
            replay_check.stateChanged.connect(
                partial(_ignore_signal_value, callback=partial(self._on_channel_replay, ch)))
        lay.addLayout(cards)

        # Shared by both channels: one Whisper model serves both
        saved = load_settings()
        self.model_combo = ComboBox()
        self.model_combo.addItems(list(MODELS))
        saved_model = saved.get("model")
        if isinstance(saved_model, str):
            saved_model = saved_model.replace(chr(0x2014), "-")
        if not isinstance(saved_model, str) or saved_model not in MODELS:
            saved_model = DEFAULT_MODEL_GPU if WHISPER_DEVICE == "cuda" else DEFAULT_MODEL_CPU
        _set_combo_text(self.model_combo, saved_model)
        self.preview_combo = ComboBox()
        self.preview_combo.addItems(list(PREVIEW_CHOICES))
        choice = saved.get("preview_model")
        if choice not in PREVIEW_CHOICES:
            choice = "Tiny (instant)" if saved.get("fast_preview", False) else "Small (balanced)"
        _set_combo_text(self.preview_combo, choice)
        # Automatic language detection (see LANG_SWITCH_PROB in interpreter.py)
        self.auto_lang_check = CheckBox("Detect other languages")
        self.auto_lang_check.setToolTip(
            "If a phrase is clearly in another language (e.g. a German song while\n"
            "listening for Russian), recognise it in that language, label it\n"
            "[German] and translate it. Off: label it but don't translate it.")
        self.auto_lang_check.setChecked(bool(saved.get("auto_language", True)))
        shared_card = HeaderCardWidget()
        shared_card.setTitle("Both channels")
        shared = QGridLayout()
        shared.setHorizontalSpacing(12)
        shared.setVerticalSpacing(10)
        shared.addWidget(label("Model", "Whisper model for the final text"), 0, 0)
        shared.addWidget(self.model_combo, 0, 1)
        shared.addWidget(label("Previews", "Model for the live grey text"), 1, 0)
        shared.addWidget(self.preview_combo, 1, 1)
        shared.addWidget(self.auto_lang_check, 2, 1)
        shared.setColumnStretch(2, 1)            # keep the menus a sensible width
        self.model_combo.setMinimumWidth(320)
        self.preview_combo.setMinimumWidth(320)
        shared_card.viewLayout.addLayout(shared)
        lay.addWidget(shared_card)
        lay.addStretch(1)

        # Adapters for the borrowed logic (it calls .get() and .set())
        self.model_var = QVar(self.model_combo.currentText,
                              partial(_set_combo_text, self.model_combo), self.model_combo)
        self.preview_choice_var = QVar(self.preview_combo.currentText,
                                       partial(_set_combo_text, self.preview_combo),
                                       self.preview_combo)
        self.model_combo.currentTextChanged.connect(
            partial(_ignore_signal_value, callback=self._on_model_change))
        self.preview_combo.currentTextChanged.connect(
            partial(_ignore_signal_value, callback=self._on_preview_choice))
        self.auto_lang_check.stateChanged.connect(self._on_auto_language)
        return page

    @staticmethod
    def _pill(text: str, colour: str) -> QLabel:
        """A small rounded coloured label (status badge)."""
        l = QLabel(text)
        l.setStyleSheet(f"background:{colour}; color:white; border-radius:6px; "
                        f"padding:3px 8px; font-weight:600;")
        return l

    def _set_live_status(self, text: str, color: str) -> None:
        """Show a status word (LISTENING, MUTED…) in the coloured badge."""
        self.status_badge.setText(f" {text} ")
        self.status_badge.setStyleSheet(f"background:{color}; color:white; border-radius:6px; "
                                        f"padding:3px 8px; font-weight:600;")

    def _update_live_status(self) -> None:
        """The badge sums up both channels: LISTENING if either listens."""
        if self.is_listening:
            if self._listening_since is None:
                self._listening_since = time.monotonic()
            self._set_live_status("LISTENING", OK)
            return
        self._listening_since = None
        if all(c.settings.muted for c in self.channels):
            self._set_live_status("MUTED", BAD)
        else:
            self._set_live_status("IDLE", WARN)

    # ── Settings changes ─────────────────────────────────────────────────

    def _save_channels(self) -> None:
        core.save_settings(channels=[asdict(c.settings) for c in self.channels])

    def _on_channel_name(self, ch: Channel) -> None:
        """Name edited: used for new log lines and redrawn everywhere."""
        name = ch.controls.name_edit.text().strip() or f"Channel {ch.index + 1}"
        if name == ch.settings.name:
            return
        ch.settings.name = name
        self._save_channels()
        ch.strip.show_settings(ch.settings)
        self._set_card_title(ch)
        self.conversation.set_names([c.settings.name for c in self.channels])
        log_event(logging.INFO, "channel_renamed", channel=ch.index, name=name)

    def _on_channel_device(self, ch: Channel) -> None:
        """A different device chosen: listen to it instead."""
        ch.settings.device = _combo_device(ch.controls.device_combo)
        # Refill: a "(not connected)" entry that is no longer chosen disappears
        _fill_combo(ch.controls.device_combo, [lbl for _, lbl in self._input_devices],
                    ch.settings.device)
        self._save_channels()
        ch.strip.show_settings(ch.settings)
        ch.failures, ch.retry_at, ch.missing_checks = 0, 0.0, 0
        self._restart_channel(ch)

    def _on_channel_languages(self, ch: Channel) -> None:
        """Spoken or shown language changed: restart this channel."""
        ch.settings.src = ch.controls.src_combo.currentText()
        ch.settings.tgt = ch.controls.tgt_combo.currentText()
        self._save_channels()
        ch.strip.show_settings(ch.settings)
        if MODELS[self.model_var.get()] == "tiny" and ch.settings.src != "English":
            self.status_var.set("Note: Tiny makes many mistakes in non-English speech; "
                                "use Turbo or Medium.")
        self._restart_channel(ch)

    def _on_channel_sensitivity(self, ch: Channel, value: int) -> None:
        """Sensitivity slider moved: apply it to the capture thread at once;
        save and log it only once the slider has settled for 0.5 s."""
        v = round(value / 100, 2)
        ch.settings.sensitivity = v
        ch.controls.sens_value.setText(f"{value}%")
        if ch.capture is not None:
            ch.capture.set_sensitivity(v)
        if ch.sens_log_after is not None:
            self.after_cancel(ch.sens_log_after)

        def settled() -> None:
            """Runs once the slider has been still for 0.5 s: save and log."""
            ch.sens_log_after = None
            self._save_channels()
            log_event(logging.INFO, "sensitivity_changed", channel=ch.settings.name, value=v,
                      run_id=ch.run_id, session_id=self.session_id,
                      vad_threshold=round(vad_threshold_for(v), 2),
                      speech_score_now=round(ch.strip.speech_meter.shown, 2))
        ch.sens_log_after = self.after(500, settled)

    def _on_channel_output(self, ch: Channel) -> None:
        """Output device changed. It also decides which channel must ignore
        this speaker's voice, so both channels restart."""
        ch.settings.output = _combo_device(ch.controls.out_combo)
        _fill_combo(ch.controls.out_combo, [lbl for _, lbl in self._output_devices],
                    ch.settings.output)
        self._save_channels()
        if ch.settings.speak:
            self._configure_speaker(ch)
        self._restart_session()

    def _on_channel_replay(self, ch: Channel) -> None:
        ch.settings.replay = ch.controls.replay_check.isChecked()
        self._save_channels()
        self._restart_channel(ch)

    def _on_model_change(self) -> None:
        """Model menu changed: save, load the new model, restart both channels."""
        log_event(logging.INFO, "whisper_model_reload_requested", selected_model=self.model_var.get())
        core.save_settings(model=self.model_var.get())
        if (MODELS[self.model_var.get()] == "tiny"
                and any(c.settings.src != "English" for c in self.channels)):
            self.after(1500, lambda: self.status_var.set(
                "Note: Tiny makes many mistakes in non-English speech; use Turbo or Medium."))
        self._stop_all()
        self._whisper_model = None
        self._preload_model()
        self._ensure_preview_model()
        self._start_all()

    def _on_auto_language(self, *_: object) -> None:
        """'Detect other languages' ticked/unticked: save and apply now
        (the workers read the flag for each phrase; no restart needed)."""
        on = self.auto_lang_check.isChecked()
        core.save_settings(auto_language=on)
        for ch in self.channels:
            for worker in ch.workers:
                worker.auto_language = on
        log_event(logging.INFO, "auto_language_changed", enabled=on)

    def _open_settings(self) -> None:
        """A strip's gear button: show the Settings page."""
        self.switchTo(self.settings_page)

    @staticmethod
    def _set_card_title(ch: Channel) -> None:
        """Title of a channel's settings card, e.g. "Channel 2 · Friend"."""
        ch.controls.card.setTitle(f"Channel {ch.index + 1}  ·  {ch.settings.name}")

    def _change_text_size(self, step: int) -> None:
        """A−/A+ buttons: change the conversation font size (9-28 pt), saved."""
        self._text_size = max(9, min(28, self._text_size + step))
        self.conversation.set_text_size(self._text_size)
        core.save_settings(text_size=self._text_size)

    def _toggle_channel_mute(self, ch: Channel) -> None:
        """A strip's Mute button: stop listening to that channel (and release
        its device), or start again."""
        ch.settings.muted = not ch.settings.muted
        self._save_channels()
        ch.strip.show_settings(ch.settings)
        if ch.settings.muted:
            self._stop_channel(ch)
            ch.strip.set_status("Muted - not listening.", BAD)
        else:
            ch.failures, ch.retry_at = 0, 0.0
            self._start_channel(ch)
        self._update_live_status()
        log_event(logging.INFO, "mute_toggled", channel=ch.settings.name, muted=ch.settings.muted)

    def _toggle_channel_speak(self, ch: Channel) -> None:
        """A strip's Speak button: read this channel's translations aloud
        through its output device, or stop doing so."""
        if ch.speaker.error and not ch.speaker.is_alive():
            self.status_var.set(f"Text-to-speech unavailable: {ch.speaker.error}")
            ch.strip.speak_btn.setChecked(False)
            return
        ch.settings.speak = not ch.settings.speak
        self._save_channels()
        ch.strip.show_settings(ch.settings)
        if ch.settings.speak:
            self._configure_speaker(ch)
        else:
            ch.speaker.stop_speech()
        log_event(logging.INFO, "tts_toggled", channel=ch.settings.name, enabled=ch.settings.speak)

    def _configure_speaker(self, ch: Channel) -> None:
        """Point the channel's voice at its output device and target language."""
        ch.speaker.configure(ch.settings.output, ch.settings.tgt)

        def report_error() -> None:
            """0.3 s later: show the problem if the voice couldn't be set up."""
            if ch.speaker.error:
                self.status_var.set(f"{ch.settings.name}: {ch.speaker.error}")
        self.after(300, report_error)

    def _clear(self) -> None:
        """Clear button: empty the conversation (listening carries on)."""
        self.conversation.reset()
        self.status_var.set("Cleared.")
        log_event(logging.INFO, "ui_cleared")

    def _refresh_devices(self) -> None:
        """The ⟳ button: re-scan audio devices, then restart both channels."""
        self._stop_all()
        self._rescan_devices()
        self.status_var.set(f"Refreshed - {len(self._input_devices)} in / "
                            f"{len(self._output_devices)} out device(s).")
        log_event(logging.INFO, "audio_devices_refreshed",
                  input_count=len(self._input_devices), output_count=len(self._output_devices))
        for ch in self.channels:
            ch.failures, ch.retry_at = 0, 0.0
        self._start_all()

    def _rescan_devices(self) -> None:
        """Find the audio devices again (their numbers change when a USB
        device is re-plugged) and refill the device menus."""
        self._input_devices, self._output_devices = get_audio_devices()
        self._input_map = {lbl: idx for idx, lbl in self._input_devices}
        self._output_map = {lbl: idx for idx, lbl in self._output_devices}
        in_labels = [lbl for _, lbl in self._input_devices]
        out_labels = [lbl for _, lbl in self._output_devices]
        for ch in self.channels:
            _fill_combo(ch.controls.device_combo, in_labels, ch.settings.device)
            _fill_combo(ch.controls.out_combo, out_labels, ch.settings.output)
            ch.needs_rescan = False

    # ── Starting and stopping the channels ───────────────────────────────
    # The app listens continuously; a settings change restarts the channel
    # concerned, and a channel's Mute button is the only way to stop it.

    def _model_ready(self) -> bool:
        """True when the Whisper model can be used. While it is still
        loading, says so and tries starting again shortly."""
        loader = self._model_loader
        if loader is not None and not loader.ready.is_set():
            self.status_var.set("⏳ Loading Whisper model - please wait…")
            if self._start_retry_after_id is None:
                self._start_retry_after_id = self.after(600, self._retry_start)
            return False
        if self._whisper_model is None and loader is not None:
            if loader.error:
                self.status_var.set(f"Cannot start: {loader.error}")
                self._set_live_status("MODEL ERROR", BAD)
                log_event(logging.ERROR, "session_start_blocked_model_error", error=loader.error)
                return False
            self._whisper_model = loader.model
            if getattr(loader, "used_fallback", False):
                self._set_live_status("CPU FALLBACK", WARN)
                self.model_var.set(loader.default_model_key)
        if self._whisper_model is None:
            self.status_var.set("Whisper model unavailable.")
            log_event(logging.ERROR, "session_start_blocked_model_unavailable")
            return False
        return True

    def _retry_start(self) -> None:
        """Timer callback: try starting again (the model was still loading)."""
        self._start_retry_after_id = None
        self._start_all()

    def _start_all(self) -> None:
        for ch in self.channels:
            self._start_channel(ch)

    def _stop_all(self) -> None:
        if self._start_retry_after_id is not None:
            self.after_cancel(self._start_retry_after_id)
            self._start_retry_after_id = None
        for ch in self.channels:
            self._stop_channel(ch)

    def _restart_session(self) -> None:
        """Restart both channels (e.g. the Previews choice changed). The
        borrowed _on_preview_choice calls this."""
        self._stop_all()
        self._start_all()

    def _restart_channel(self, ch: Channel) -> None:
        self._stop_channel(ch)
        self._start_channel(ch)

    def _channel_problem(self, ch: Channel, text: str, colour: str = WARN,
                         **details: Any) -> None:
        """Show why a channel isn't listening on its strip (logged once,
        with any `details` that help to diagnose it from the log)."""
        if text != ch.problem:
            log_event(logging.WARNING, "channel_not_listening", channel=ch.settings.name,
                      device=ch.settings.device, problem=text, **details)
        ch.problem = text
        ch.strip.set_status(text, colour)
        ch.strip.status_label.setToolTip(text)

    def _device_missing(self, ch: Channel) -> None:
        """The channel's device isn't in Windows' device list: unplugged,
        switched off, disabled, or renamed (a driver update or another USB
        port can give it a new name). Say so, suggest a connected device that
        looks similar, but never switch silently: a wrong guess would
        translate the wrong person."""
        name = plain_device_name(ch.settings.device) or "the device"
        labels = [lbl for _, lbl in self._input_devices]
        similar = similar_devices(ch.settings.device, labels)
        hint = (f" Similar device connected: {plain_device_name(similar[0])}."
                if similar else "")
        self._channel_problem(
            ch, f"⚠ {name} is not connected. Plug it in, or choose another device "
                f"in Settings → Listen to.{hint}",
            similar=similar, available=labels)

    def _start_channel(self, ch: Channel) -> None:
        """Start one channel's pipeline with its current settings (unless it
        is muted, already listening, or its device isn't there)."""
        if ch.listening or ch.settings.muted or not self._initialised or self._closing:
            return
        if not self._model_ready():
            return
        s = ch.settings
        if s.device not in self._input_map:
            self._device_missing(ch)
            return
        twin = next((c for c in self.channels
                     if c is not ch and c.listening and c.settings.device == s.device), None)
        if twin is not None:
            self._channel_problem(ch, f"⚠ Same device as {twin.settings.name}: choose another "
                                      f"in Settings.")
            return
        self._run_id += 1
        ch.run_id = self._run_id
        ch.first_preview, ch.pending_translation = {}, {}
        if s.speak:
            self._configure_speaker(ch)
        config = SessionConfig(
            session_id=self.session_id,
            run_id=ch.run_id,
            device_label=s.device,
            device_index=self._input_map[s.device],
            output_label=s.output,
            output_index=self._output_map.get(s.output),
            loopback=s.replay,
            src_key=s.src,
            tgt_key=s.tgt,
            model_label=self.model_var.get(),
            sensitivity=s.sensitivity,
            preview_choice=self.preview_choice_var.get(),
            preview_enabled=self._preview_enabled,
            previews_off=self._previews_off,
            tts_on=s.speak,
            record_audio=bool(load_settings().get("record_audio", True)),
            auto_language=self.auto_lang_check.isChecked(),
            # Every language either channel uses: candidates for a short,
            # unclear phrase (see TranscriptionWorker._recent_langs)
            mode_state=tuple(("", c.settings.src, c.settings.tgt) for c in self.channels),
            channel=s.name,
            tag=(ch.index, ch.run_id),
        )
        log_event(logging.INFO, "audio_device_selected", channel=s.name,
                  device_label=s.device, device_index=config.device_index)
        try:
            ch.controller.start(
                config,
                model=self._whisper_model,
                model_loader=self._model_loader,
                speaker=None,
                on_level=None,
                preview_model=self._preview_model,
                translator_for=self._translator_for,
                music_scene=self._music_scene_for(ch),
                mute_event=self._mute_event_for(ch),
            )
        except Exception as exc:
            ch.controller.stop_workers()
            ch.failures += 1
            ch.retry_at = time.monotonic() + min(30, 2 ** ch.failures)
            self._channel_problem(ch, f"⚠ Could not start: {exc}", BAD)
            log_exception("session_start_failed", exc, channel=s.name, device_label=s.device,
                          model=config.model_label, run_id=config.run_id)
            return
        ch.listening = True
        ch.problem = None
        ch.strip.set_status("Listening…")
        self.status_var.set("Listening…")
        self._update_live_status()

    def _stop_channel(self, ch: Channel) -> None:
        """Stop one channel's pipeline and clear its meters and previews."""
        was_listening = ch.listening
        ch.listening = False
        ch.controller.stop_workers()
        ch.strip.reset_meters()
        self.conversation.drop_previews(ch.index)
        if was_listening:
            ch.strip.set_status("Stopped.")
            log_event(logging.INFO, "session_stopped", session_id=self.session_id,
                      channel=ch.settings.name, run_id=ch.run_id)
        self._update_live_status()

    def _mute_event_for(self, ch: Channel) -> AnyEvent | None:
        """When a channel listens to what an output device plays, it must
        ignore our own text-to-speech on that device (or it would translate
        it again). Returns a flag that is set while any speaker that uses
        this device is talking, or None if none does."""
        if "[loopback]" not in ch.settings.device.lower():
            return None
        device = plain_device_name(ch.settings.device)
        events = [c.speaker.speaking for c in self.channels
                  if plain_device_name(c.settings.output) == device]
        return AnyEvent(events) if events else None

    def _music_scene_for(self, ch: Channel) -> Any:
        """The channel's music detector (music.py), created once and kept
        across restarts. The second channel's detector shares the first
        one's models. "music_mode": false in interpreter.settings.json turns
        it off; "vocal_separation": false keeps the detection but skips
        removing the instruments."""
        if ch.music_scene is None:
            settings = load_settings()
            shared = next((c.music_scene for c in self.channels if c.music_scene is not None), None)
            name = ch.settings.name

            def log(event: str, **fields: Any) -> None:
                log_event(logging.INFO, event, channel=name, **fields)
            try:
                from music import MusicScene
                ch.music_scene = MusicScene(enabled=bool(settings.get("music_mode", True)),
                                            separate=bool(settings.get("vocal_separation", True)),
                                            log=log, models_from=shared)
            except Exception as e:
                log_exception("music_scene_unavailable", e, channel=name)
                return None
        return ch.music_scene if ch.music_scene.enabled else None

    # ── Results from the channels ────────────────────────────────────────

    def _poll_queue(self) -> None:
        """Runs every 15 ms on the main thread: handle every message the
        background threads have sent, and update the window. This is the
        only place results reach the screen."""
        now = time.monotonic()
        if self._last_poll_t is not None:
            # >0 means the window was busy and results waited
            self._ui_stats.add("poll_lag_ms", (now - self._last_poll_t) * 1000 - 15)
        self._last_poll_t = now
        handled = 0
        try:
            while True:
                message = self.result_queue.get_nowait()
                handled += 1
                try:
                    self._handle_message(message)
                except Exception as e:      # one bad message must not stop the window
                    log_exception("ui_message_failed", e)
        except queue.Empty:
            pass
        if handled:
            self._ui_stats.add("msgs_per_poll", handled)
        self.after(15, self._poll_queue)

    def _handle_message(self, message: tuple[Any, Any]) -> None:
        """Act on one message from the result queue.

        A channel's messages arrive as ((channel number, run id), (kind,
        data)) (see TaggedQueue); the window's own as (kind, data), e.g.
        ("status", "Loading…") from the model loader.

        kinds:  partial     live preview of a phrase still being spoken
                final       a phrase's final text (+ translation, or None)
                translated  an online translation arrived
                status      a message for the status line
                level       one audio block's level, for the meters
                metric      a timing figure for Diagnostics
        """
        if isinstance(message[0], str):          # the window's own message
            kind, data = message
            if kind == "status":
                self.status_var.set(data)
                log_event(logging.INFO, "status_updated", status=data)
            return
        (index, run_id), (kind, data) = message
        ch = self.channels[index]
        if kind == "translated":
            # Online translations can arrive after a restart: the phrase is
            # still on screen, so show it anyway.
            seq, text = data
            self._show_translation(ch, (index, run_id, seq), text)
            return
        if run_id != ch.run_id or not ch.listening:
            return                               # from a pipeline that was stopped
        if kind == "partial":
            seq, heard, translated, t_start, t_ready = data
            self.conversation.show_preview((index, run_id, seq), index, heard, translated)
            now = time.monotonic()
            self._ui_stats.add("preview_ui_ms", (now - t_ready) * 1000)
            if seq not in ch.first_preview and t_start:
                ch.first_preview[seq] = round((now - t_start) * 1000, 1)
        elif kind == "final":
            self._show_final(ch, (index, run_id, data[0]), *data[1:])
        elif kind == "status":
            ch.strip.set_status(data, BAD if data.startswith(("ERROR", "⚠")) else None)
            self.status_var.set(f"{ch.settings.name}: {data}")
            log_event(logging.INFO, "status_updated", channel=ch.settings.name, status=data)
        elif kind == "level":
            self._update_meter(ch, *data)
        elif kind == "metric":
            self._update_metric(*data)

    def _show_final(self, ch: Channel, key: tuple[int, int, int], text: str,
                    translated: str | None, trace: dict[str, Any] | None) -> None:
        """A phrase is finished: show it, speak its translation, or send it
        to the online translator if no local translation was possible."""
        self.conversation.commit(key, ch.index, text, translated)
        if not text:
            return
        if trace:
            self._log_displayed(ch, key, trace, online=translated is None)
        if translated is not None:
            self._show_translation(ch, key, translated)
            return
        seq = key[2]
        request = (seq, text, ch.settings.src, ch.settings.tgt)
        try:
            ch.controller.translation_queue.put(request, timeout=0.1)
            backlog = ch.controller.translation_queue.qsize()
            log_event(logging.INFO, "translation_queued", channel=ch.settings.name,
                      input_chars=len(text), src_lang=ch.settings.src, tgt_lang=ch.settings.tgt,
                      queue_backlog=backlog)
            if backlog >= core.TRANSLATION_QUEUE_BACKLOG_WARN:
                log_event(logging.WARNING, "translation_queue_backlog_high", backlog=backlog)
        except queue.Full:
            self.conversation.set_translation(key, "")
            log_event(logging.WARNING, "translation_dropped_queue_full", channel=ch.settings.name,
                      input_chars=len(text), src_lang=ch.settings.src, tgt_lang=ch.settings.tgt)

    def _show_translation(self, ch: Channel, key: tuple[int, int, int], text: str) -> None:
        """Show a translation, remember it for 'Report a mistake', and speak
        it if the channel's Speak is on."""
        self.conversation.set_translation(key, text)
        if key in self._recent_finals:
            self._recent_finals[key]["translation"] = text
        if ch.settings.speak and text:
            ch.speaker.say(text)
        trace = ch.pending_translation.pop(key[2], None) if key[1] == ch.run_id else None
        if trace:
            log_event(logging.INFO, "translation_displayed", channel=ch.settings.name,
                      run_id=trace["run_id"], seq=key[2],
                      speech_end_to_translation_ms=round((time.monotonic() - trace["t_end"]) * 1000, 1))

    def _update_meter(self, ch: Channel, rms_db: float, peak_db: float, speaking: bool,
                      prob: float = 0.0, threshold: float = 0.5) -> None:
        """Called for every audio block (via the result queue): update the
        channel's meters, and keep the last ~10 s for Diagnostics."""
        ch.strip.update_levels(rms_db, peak_db, speaking, prob, threshold)
        if rms_db > -99:
            ch.levels.append(rms_db)
            ch.peak_recent = max(peak_db, ch.peak_recent - 0.05)
        ch.speech_flags.append(bool(speaking))

    def _log_displayed(self, ch: Channel, key: tuple[int, int, int], trace: dict[str, Any],
                       online: bool) -> None:
        """Log how long a finished phrase took to reach the screen
        ("phrase_displayed"), and keep it for Diagnostics and 'Report a
        mistake'."""
        seq = key[2]
        now = time.monotonic()
        first_preview = ch.first_preview.pop(seq, None)
        log_event(logging.INFO, "phrase_displayed", session_id=self.session_id,
                  channel=ch.settings.name, run_id=trace["run_id"], seq=seq,
                  ui_ms=round((now - trace["t_result"]) * 1000, 1),
                  speech_end_to_text_ms=round((now - trace["t_end"]) * 1000, 1),
                  first_preview_ms=first_preview, online_translation_pending=online)
        if online:
            ch.pending_translation[seq] = trace
        self._recent_finals[key] = {"time": time.strftime("%H:%M:%S"), "who": ch.settings.name,
                                    "trace": trace, "text": trace.get("text") or "",
                                    "translation": trace.get("translation") or ""}
        while len(self._recent_finals) > 30:
            self._recent_finals.popitem(last=False)
        retry = trace.get("retry")
        retry_data = cast(dict[str, Any], retry) if isinstance(retry, dict) else {}
        self._phrases.append({
            "time": time.strftime("%H:%M:%S"),
            "who": ch.settings.name,
            "channel": ch.index,
            "pause_to_text": (now - trace["t_end"]) * 1000,
            "first_preview": first_preview,
            # recognition time: for an early-transcribed phrase the work was
            # done during the pause (early_ms), so process_ms would read 0
            "process": trace.get("early_ms") if trace.get("early_used") else trace.get("process_ms"),
            "mt": trace.get("mt_ms") or None,
            "early": bool(trace.get("early_used")),
            "model": trace.get("model"),
            "doubtful": bool(trace.get("suspicious")),
            "retry_used": bool(retry_data.get("used")),
            "context_reset": bool(trace.get("context_reset")),
            "onset": trace.get("onset_delay_ms"),
            "text": trace.get("text") or "",
        })

    # ── Keeping both channels alive ──────────────────────────────────────

    def _watchdog(self) -> None:
        """Every WATCHDOG_SEC: restart a channel whose capture thread died or
        gave up (device unplugged, driver reset) or whose worker exited,
        instead of showing it as listening while nothing is heard. Waits
        longer after each failure (2, 4, 8 … up to 30 s). A channel waiting
        for its device is retried until the device is back; it never
        silently switches to another device."""
        if self._closing:
            return
        try:
            now = time.monotonic()
            for ch in self.channels:
                if ch.settings.muted:
                    continue
                if ch.listening:
                    capture = ch.capture
                    problem = None
                    if capture is not None and not capture.is_alive():
                        problem = capture.failed or "capture thread stopped"
                    elif any(not w.is_alive() for w in ch.workers):
                        problem = "transcription worker stopped"
                    if problem is None:
                        ch.failures = 0
                        continue
                    ch.failures += 1
                    wait = min(30, 2 ** ch.failures)
                    log_event(logging.WARNING, "watchdog_restart", channel=ch.settings.name,
                              problem=problem, attempt=ch.failures, next_check_sec=wait)
                    self._stop_channel(ch)
                    self._channel_problem(ch, f"⚠ {problem} - restarting…")
                    ch.retry_at, ch.needs_rescan = now + wait, True
                elif now >= ch.retry_at and self._whisper_model is not None:
                    if ch.needs_rescan or ch.settings.device not in self._input_map:
                        self._rescan_devices()
                    self._start_channel(ch)
                    if ch.listening:
                        ch.missing_checks = 0
                        log_event(logging.INFO, "watchdog_reconnected", channel=ch.settings.name,
                                  device=ch.settings.device)
                    else:
                        # Still not there: look again, less often each time
                        # (3, 6, 12, 24, then every 30 s). Every look re-scans
                        # all devices; every 4 s filled the log and restarted
                        # PortAudio while the other channel was recording.
                        ch.missing_checks += 1
                        ch.retry_at = now + min(30, 3 * 2 ** (ch.missing_checks - 1))
        except Exception as e:
            log_exception("watchdog_failed", e)
        self.after(WATCHDOG_SEC * 1000, self._watchdog)

    def _log_resources(self) -> None:
        """Every STATS_INTERVAL_SEC: CPU, memory, GPU load and the window's
        responsiveness, so slowdowns can be tied to resource limits."""
        base = dict(listening=self.is_listening, run_id=self._run_id,
                    result_queue=self.result_queue.qsize(),
                    phrase_queue=sum(c.controller.phrase_queue.qsize() for c in self.channels),
                    translation_queue=sum(c.controller.translation_queue.qsize()
                                          for c in self.channels),
                    **self._ui_stats.summary())

        def measure() -> None:
            """Background thread: read CPU, memory and GPU figures (slow
            Windows calls), log them, and warn if anything is running low."""
            t0 = time.perf_counter()
            fields = core.collect_resource_stats(self)
            fields["probe_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            log_event(logging.INFO, "resource_stats", **base, **fields)
            warnings = core.resource_warnings(fields)
            if warnings:
                log_event(logging.WARNING, "resource_warning", problems=warnings,
                          **{k: fields.get(k) for k in ("system_cpu_pct", "mem_available_mb",
                                                         "gpu_mem_free_mb", "gpu_util_pct",
                                                         "top_other_apps")})
        threading.Thread(target=measure, name="resource-probe", daemon=True).start()
        self.after(STATS_INTERVAL_SEC * 1000, self._log_resources)

    # ── Report a mistake ─────────────────────────────────────────────────

    def _report_mistake(self) -> None:
        """Choose a recent phrase, say what was wrong (and optionally the
        correct text), save it with its audio for analysis and replay."""
        if not self._recent_finals:
            InfoBar.info(title="Nothing to report yet", content="No phrases have been shown yet.",  # pyright: ignore[reportUnknownMemberType]
                         orient=Qt.Orientation.Horizontal, isClosable=True,
                         position=InfoBarPosition.TOP_RIGHT, duration=3000, parent=self)
            return
        dialog = ReportDialogMistake(list(reversed(self._recent_finals.items())), self)
        if not dialog.exec():
            return
        key, problems, correct, lang, note = dialog.result_values()
        entry = self._recent_finals[key]
        index, run_id, seq = cast(tuple[int, int, int], key)
        ch = self.channels[index]
        audio: bytes | None = None
        if run_id == ch.run_id:                  # older runs' audio has gone with their worker
            for worker in ch.workers:
                if seq in worker.recent_audio:
                    audio = worker.recent_audio[seq]
                    break
        try:
            path = core.save_feedback(seq, audio, entry["trace"], entry["text"], entry["translation"],
                                      problems, correct, lang, note)
        except Exception as e:
            log_exception("report_mistake_failed", e, seq=seq, channel=ch.settings.name)
            return
        InfoBar.success(title="Thank you, saved",  # pyright: ignore[reportUnknownMemberType]
                        content=("Saved with its audio" if path else "Saved (audio no longer in memory)")
                        + " to testdata/feedback. The simulator can replay it.",
                        orient=Qt.Orientation.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP_RIGHT, duration=4000, parent=self)

    # ── Diagnostics page ─────────────────────────────────────────────────

    def _build_diagnostics_page(self) -> SmoothScrollArea:
        """Build the Diagnostics page: the health, speed, audio, recognition,
        computer and pipeline cards, recent issues, and the phrase table."""
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
        cols = ["Time", "Who", "Model", "Pause→text", "1st preview", "Recognition", "Early", "Text"]
        self.phrase_table.setColumnCount(len(cols))
        self.phrase_table.setHorizontalHeaderLabels(cols)
        self.phrase_table.verticalHeader().hide()
        self.phrase_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.phrase_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.phrase_table.setMinimumHeight(260)
        phrases_card.viewLayout.addWidget(self.phrase_table)
        lay.addWidget(phrases_card)
        return page

    def _refresh_diagnostics(self) -> None:
        """Every second: update the Diagnostics page, and schedule the next
        update even if this one failed (an error is logged, not shown)."""
        try:
            self._refresh_diagnostics_once()
        except Exception as e:
            log_exception("diagnostics_refresh_failed", e)
        self.after(1000, self._refresh_diagnostics)

    def _refresh_diagnostics_once(self) -> None:
        """Recompute every figure on the Diagnostics page (once a second)."""
        ph = list(self._phrases)
        p2t = [p["pause_to_text"] for p in ph]
        fp = [p["first_preview"] for p in ph]
        rec = [p["process"] for p in ph]
        mt = [p["mt"] for p in ph]
        def fmt_ms(a: float | None, b: float | None) -> str:
            return "-" if a is None else f"{a:.0f} ms / {b:.0f} ms"

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
        s.set("Early transcription used", "-" if early is None else f"{early:.0f} %",
              MUTED if early is None else OK if early >= 50 else WARN)

        # Audio: one value per channel, e.g. "Me -32.1 dBFS · Friend -40.5 dBFS".
        # The colour is the worst of the channels.
        rank = {OK: 0, MUTED: 0, WARN: 1, BAD: 2}

        def per_channel(row: str, values: list[tuple[str, str | None]]) -> None:
            """Show one (text, colour) per channel in a row of the Audio card."""
            text = "  ·  ".join(f"{c.settings.name} {v}" for c, (v, _) in zip(self.channels, values))
            colours = [col for _, col in values if col]
            self.audio_card.set(row, text, max(colours, key=lambda col: rank.get(col, 0))
                                if colours else None)

        def level_now(c: Channel) -> tuple[str, str | None]:
            return ("-", None) if not c.levels else (f"{c.levels[-1]:.1f} dBFS", None)

        def level_median(c: Channel) -> tuple[str, str | None]:
            med = pct(list(c.levels), .5)
            return (("-", None) if med is None
                    else (f"{med:.1f} dBFS", OK if -40 <= med <= -10 else WARN))

        def speech_share(c: Channel) -> tuple[str, str | None]:
            sf = list(c.speech_flags)
            return (f"{100 * sum(sf) / len(sf):.0f} %" if sf else "-", None)

        def onset(c: Channel) -> tuple[str, str | None]:
            value = pct([p["onset"] for p in ph if p["channel"] == c.index], .5)
            return ("-" if value is None else f"{value:.0f} ms", judge(value, 150, 400))

        def sensitivity(c: Channel) -> tuple[str, str | None]:
            sens = c.settings.sensitivity
            return (f"{sens * 100:.0f} % (threshold {vad_threshold_for(sens):.2f})", None)

        def peak(c: Channel) -> tuple[str, str | None]:
            return ((f"{c.peak_recent:.1f} dBFS", BAD if c.peak_recent > -1 else OK)
                    if c.peak_recent > -99 else ("-", None))

        for row, measure in (("Level now", level_now), ("Level (last 10 s)", level_median),
                             ("Speech share (10 s)", speech_share), ("Onset delay", onset),
                             ("Sensitivity", sensitivity), ("Loudest peak", peak)):
            per_channel(row, [measure(c) for c in self.channels])

        q = self.quality_card
        n = len(ph)
        def share(key: str) -> float | None:
            return 100 * sum(p[key] for p in ph) / n if n else None
        d = share("doubtful")
        q.set("Doubtful results", "-" if d is None else f"{d:.0f} %", judge(d, 10, 25))
        q.set("Retries used", sum(p["retry_used"] for p in ph))
        empty = 100 * sum(not p["text"] for p in ph) / n if n else None
        q.set("Empty phrases", "-" if empty is None else f"{empty:.0f} %", judge(empty, 10, 30))
        q.set("Context resets", sum(p["context_reset"] for p in ph))

        self._refresh_resources()

        pl = self.pipeline_card
        pq = self._waiting_phrases()
        tq = sum(c.controller.translation_queue.qsize() for c in self.channels)
        pl.set("Waiting phrases", pq, OK if pq < 2 else WARN if pq < 5 else BAD)
        pl.set("Waiting translations", tq, OK if tq < 3 else WARN)
        def alive(thread: threading.Thread | None) -> str:
            return "✔" if thread is not None and thread.is_alive() else "✖"
        parts, healthy = [], True
        for c in self.channels:
            text = (f"{c.settings.name}: capture {alive(c.capture)} transcription "
                    f"{alive(c.workers[0]) if c.workers else '✖'} speech {alive(c.speaker)}")
            parts.append(text)
            healthy = healthy and (not c.listening or "✖" not in text)
        pl.set("Threads", "   ".join(parts), OK if healthy else BAD)
        pl.set("Watchdog restarts", self._watchdog_restarts,
               OK if self._watchdog_restarts == 0 else WARN)
        pl.set("Phrases this run", self._perf_stats["transcriptions"])

        self._refresh_health()
        self._refresh_issues()
        self._refresh_phrase_table(ph)

    def _refresh_health(self) -> None:
        """The Health card: overall state (listening, problems found), how long
        it has been listening, and which model and translator are in use."""
        h = self.health_card
        loader = self._model_loader
        listening = [c for c in self.channels if c.listening]
        stuck = [c for c in self.channels if not c.settings.muted and not c.listening]
        if all(c.settings.muted for c in self.channels):
            overall, colour = "Muted", WARN
        elif loader is not None and not loader.ready.is_set():
            overall, colour = "Loading the Whisper model…", WARN
        elif loader is not None and loader.error:
            overall, colour = f"Model error: {loader.error}", BAD
        elif self._waiting_phrases() >= 5:
            overall, colour = "Falling behind: phrases are piling up", BAD
        elif listening and stuck:
            overall, colour = (f"Listening to {', '.join(c.settings.name for c in listening)}; "
                               f"{stuck[0].settings.name} is not: {stuck[0].problem or 'stopped'}"), WARN
        elif listening:
            overall, colour = "Healthy - listening", OK
        else:
            overall, colour = "Idle", WARN
        # Count kinds of problem, not repeats (one error on every phrase is one problem)
        problems = [g["help"]["title"] for g in self._issue_groups.values()
                    if g["level"] in ("ERROR", "CRITICAL")]
        if problems and colour == OK:
            overall = (f"Listening - problem: {problems[0]}" if len(problems) == 1
                       else f"Listening - {len(problems)} problems, see Recent issues below")
            colour = WARN
        h.set("Overall", overall, colour)
        since = self._listening_since
        h.set("Listening for", time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - since))
              if since and self.is_listening else "-")
        h.set("Main model", f"{MODELS.get(self.model_var.get())}  "
                            f"({getattr(loader, 'active_compute_type', COMPUTE_TYPE)} on "
                            f"{getattr(loader, 'active_device', WHISPER_DEVICE)})" if loader else "-")
        pl = self._preview_loader
        if not self._preview_enabled:
            h.set("Preview model", "same as main model")
        elif pl is not None and pl.ready.is_set():
            h.set("Preview model", pl.model_size, OK)
        else:
            h.set("Preview model", "loading…", WARN)
        # Translation: one entry per channel, coloured by the worst
        texts, colours = [], []
        for c in self.channels:
            src = LANGUAGES.get(c.settings.src, ("", ""))[0]
            tgt = LANGUAGES.get(c.settings.tgt, ("", ""))[0]
            if src == "auto":
                texts.append(f"{c.settings.name}: Opus-MT per detected language → {tgt}")
                colours.append(OK)
                continue
            mt = self._mt_loaders.get((src, tgt))
            if mt is None:
                texts.append(f"{c.settings.name}: online (no local model for {src}→{tgt})")
                colours.append(WARN)
            elif not mt.ready.is_set():
                texts.append(f"{c.settings.name}: Opus-MT {src}→{tgt} loading…")
                colours.append(WARN)
            elif mt.translator is None:
                texts.append(f"{c.settings.name}: Opus-MT {src}→{tgt} failed - using fallback")
                colours.append(BAD)
            else:
                texts.append(f"{c.settings.name}: Opus-MT {src}→{tgt}")
                colours.append(OK)
        worst = BAD if BAD in colours else WARN if WARN in colours else OK
        h.set("Translation", "   ".join(texts), worst)

    def _waiting_phrases(self) -> int:
        """Finished phrases not yet transcribed, both channels together."""
        return sum(c.controller.phrase_queue.qsize() for c in self.channels)

    def _refresh_resources(self) -> None:
        """The Computer card: this app's CPU and memory, and the graphics card's
        load, memory, temperature and clock speed."""
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
            import pynvml  # pyright: ignore[reportMissingTypeStubs]
            if self._nvml is None:
                pynvml.nvmlInit()
                self._nvml = pynvml.nvmlDeviceGetHandleByIndex(0)  # pyright: ignore[reportUnknownMemberType]
            h = self._nvml
            util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)  # pyright: ignore[reportUnknownMemberType]
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)  # pyright: ignore[reportUnknownMemberType]
            memory_used = int(mem.used)
            memory_total = int(mem.total)
            temp = pynvml.nvmlDeviceGetTemperature(h, 0)  # pyright: ignore[reportUnknownMemberType]
            r.set("GPU load", f"{util} %", judge(util, 85, 97))
            r.set("GPU memory", f"{memory_used / 2**30:.1f} / {memory_total / 2**30:.1f} GB",
                  judge(memory_used / memory_total * 100, 85, 95))
            r.set("GPU temperature", f"{temp} °C", judge(temp, 80, 88))
            r.set("GPU clock", f"{pynvml.nvmlDeviceGetClockInfo(h, 1)} MHz")  # pyright: ignore[reportUnknownMemberType]
        except Exception:
            r.set("GPU load", "nvidia-ml-py not available", MUTED)

    def _refresh_issues(self) -> None:
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
        for t, lvl, ev, _, help_ in new:
            g = self._issue_groups.setdefault(ev, {"first": t, "count": 0, "level": lvl,
                                                   "help": help_})
            g["count"] += 1
            g["last"], g["help"] = t, help_
            if lvl in ("ERROR", "CRITICAL"):
                g["level"] = lvl
        self._render_issue_groups()
        for t, lvl, ev, _, help_ in new:
            if lvl not in ("ERROR", "CRITICAL") and ev not in self.POPUP_WARNINGS:
                continue
            g = self._issue_groups[ev]
            if g.get("popped"):
                continue            # already shown: the count on Diagnostics goes up
            g["popped"] = True
            text = f"{help_['meaning']}\n\nWhat to do: {help_['action']}"
            bar = (InfoBar.error if lvl in ("ERROR", "CRITICAL") else InfoBar.warning)(  # pyright: ignore[reportUnknownMemberType]
                title=help_["title"], content=text, orient=Qt.Orientation.Vertical, isClosable=True,
                position=InfoBarPosition.BOTTOM_RIGHT, duration=15000, parent=self)
            details = PushButton()
            details.setText("Details")
            details.clicked.connect(lambda _=False, b=bar: (self.switchTo(self.diag_page), b.close()))
            bar.addWidget(details)
            bar.setToolTip(f"{help_['technical']}\n{help_['where']}\n"
                           "Full details: Diagnostics → Recent issues")

    # Warnings worth a pop-up too (all errors pop up)
    POPUP_WARNINGS = ("music_processing_disabled", "resource_warning")

    def _render_issue_groups(self) -> None:
        """Recent issues card: one entry per kind of problem, newest first,
        with a plain explanation, what to do, and the technical details."""
        if not self._issue_groups:
            self.issues_text.setPlainText("No warnings or errors since the app opened.")
            return
        blocks: list[str] = []
        for ev, g in sorted(self._issue_groups.items(), key=lambda kv: kv[1]["last"], reverse=True):
            h = g["help"]
            times = (f"{g['count']} times, {g['first']}-{g['last']}" if g["count"] > 1
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

    def _refresh_phrase_table(self, ph: list[dict[str, Any]]) -> None:
        """The Recent phrases table: the newest 20 phrases with their timings."""
        t = self.phrase_table
        rows = list(reversed(ph))[:20]
        t.setRowCount(len(rows))
        for r, p in enumerate(rows):
            cells: list[str] = [p["time"], p["who"], p["model"] or "",
                     f"{p['pause_to_text']:.0f} ms",
                     f"{p['first_preview']:.0f} ms" if p["first_preview"] else "-",
                     f"{p['process']:.0f} ms" if p["process"] is not None else "-",
                     "yes" if p["early"] else "no", p["text"]]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if c == 1:
                    item.setForeground(QColor(CHANNEL_COLOURS[p["channel"] % len(CHANNEL_COLOURS)]))
                elif c == 3:
                    item.setForeground(QColor(judge(p["pause_to_text"], 400, 800)))
                t.setItem(r, c, item)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)

    def _copy_diagnostics(self) -> None:
        """Copy every Diagnostics figure as text (e.g. to paste in a message)."""
        lines = [f"Voice Interpreter diagnostics  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                 f"version {core.APP_VERSION}"]
        for card in (self.health_card, self.speed_card, self.audio_card, self.quality_card,
                     self.resource_card, self.pipeline_card):
            lines.append(f"\n[{card.headerLabel.text()}]")
            lines += [f"  {k:<26} {v.text()}" for k, v in card.values.items()]
        lines.append("\n[Recent issues]\n" + self.issues_text.toPlainText())
        QGuiApplication.clipboard().setText("\n".join(lines))
        InfoBar.success(title="Copied", content="Diagnostics copied to the clipboard",  # pyright: ignore[reportUnknownMemberType]
                        orient=Qt.Orientation.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP_RIGHT, duration=2500, parent=self)

    def _show_speed_report(self) -> None:
        """Run log_tools.py's speed report for this run of the app, in a
        background thread so the window doesn't freeze, then show it."""
        started = time.strftime("%Y-%m-%dT%H:%M",
                                time.localtime(time.time() - (time.monotonic() - self._started_at)))
        self.status_var.set("Building speed report…")

        def work() -> None:
            """Background thread: build the speed report (can take seconds)."""
            try:
                import log_tools
                text = log_tools.build_report(started)
            except Exception as e:
                text = f"Report failed: {e}"
            self.result_queue.put(("status", "Speed report ready."))
            self._pending_report = text
        self._pending_report = None
        threading.Thread(target=work, daemon=True, name="speed-report").start()

        def show_when_ready() -> None:
            """Check every 0.2 s whether the report is ready, then show it."""
            if self._pending_report is None:
                self.after(200, show_when_ready)
                return
            ReportDialog("Speed report (this run)", self._pending_report, self).exec()
        self.after(200, show_when_ready)

    # ── Closing ──────────────────────────────────────────────────────────

    def closeEvent(self, e: QCloseEvent) -> None:
        """The window is closing: stop listening, log it, stop the speakers.
        _closing stops the watchdog (or a pending timer) restarting a channel."""
        try:
            self._closing = True
            for timer in self._timers.values():
                timer.stop()
            self._timers.clear()
            self._stop_all()
            log_event(logging.INFO, "app_closing", session_id=self.session_id)
            for ch in self.channels:
                ch.speaker.shutdown()
        finally:
            super().closeEvent(e)  # pyright: ignore[reportUnknownMemberType]


def main() -> None:
    """Start the Fluent window: catch crashes in the log, use the dark theme
    and Windows accent colour, then run until the window closes."""
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

"""Settings that are fixed in the code: paths, tuning values, menus, colours.

Every value in CAPITALS here is a "tuning knob": change it to change how
the app behaves. Each has a comment saying what it does and, where it was
measured, why it has that value. Values that change while the app runs
(locks, the detected GPU, the user's saved choices) live with the code
that uses them, not here.

This module imports nothing from the rest of the package, so any module
can import from it without a circular import.

PYTHON: other modules import what they need by name, e.g.
    from .config import SAMPLE_RATE, SILENCE_SEC
The leading dot means "from this package" (a relative import).
"""

import subprocess
from pathlib import Path


# ── Paths / log config ────────────────────────────────────────────────────────
# Where the app keeps its files. APP_DIR is the project folder: this file
# is in interpreter\, so .parent is that folder and .parent.parent the one
# above it. Logs and settings always sit next to the program.
APP_DIR               = Path(__file__).resolve().parent.parent
# LOG_PATH: the main log (every event); ERROR_LOG_PATH: warnings/errors only;
# LOG_SUMMARY_PATH: written by Export Log Summary; SETTINGS_PATH: saved choices.
LOG_PATH              = APP_DIR / "interpreter.log.jsonl"
ERROR_LOG_PATH        = APP_DIR / "interpreter.error.log.jsonl"
LOG_SUMMARY_PATH      = APP_DIR / "interpreter.log.summary.json"
SETTINGS_PATH         = APP_DIR / "interpreter.settings.json"


def _app_version() -> str:
    """The git commit this copy of the program comes from (e.g. "366520e"),
    logged with each session so results can be tied to a code version."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=APP_DIR,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


APP_VERSION = _app_version()
# Start a new log file after 5 MB, keeping LOG_BACKUP_COUNT older ones.
MAX_LOG_BYTES         = 5 * 1024 * 1024
LOG_BACKUP_COUNT      = 5

# ── Tuning ────────────────────────────────────────────────────────────────────
# General limits. A *backlog* is work waiting in a queue; if it grows, the
# app is falling behind, so a warning is logged.
MAX_TEXT_HISTORY_CHARS         = 12_000
# Warn when this many phrases / translations are waiting to be processed.
PHRASE_QUEUE_BACKLOG_WARN      = 6
TRANSLATION_QUEUE_BACKLOG_WARN = 16
GPU_CPU_THREADS                = 4          # CPU threads per model when it runs on the GPU
TRANSLATION_WORKERS            = 1          # >1 lets results arrive out of order
TRANSLATION_QUEUE_MAX          = 64


# ── Audio / VAD ────────────────────────────────────────────────────────────────
# Settings for recording and for deciding where phrases start and end.
# Whisper expects 16 kHz mono 16-bit audio (SAMPLE_RATE / CHANNELS /
# SAMPLE_WIDTH); devices usually record at 44.1 or 48 kHz, so audio is
# resampled. Times ending in _SEC are in seconds.
SAMPLE_RATE      = 16_000
# Whisper's audio format: 16 kHz (SAMPLE_RATE), mono (CHANNELS = 1),
# 16-bit samples (SAMPLE_WIDTH = 2 bytes). CHUNK_SAMPLES = 512 samples is
# one 32 ms block at 16 kHz; blocks at other rates last the same 32 ms.
CHUNK_SAMPLES    = 512
CHANNELS         = 1
SAMPLE_WIDTH     = 2
SILENCE_SEC      = 0.3     # pause that ends a phrase (sentence gaps are often ~0.25-0.4 s)
MAX_PHRASE_SEC   = 10      # longest phrase before a forced cut. Previews show text
                           # live, so a longer phrase costs no visible delay but
                           # means fewer cuts through words (the main error source)
CUT_SEARCH_SEC   = 2.5     # ...made at the quietest word gap in this final window
PARTIAL_INTERVAL_SEC = 0.2 # live preview cadence; the worker skips stale ones,
                           # so this self-limits to what the GPU can keep up with
PARTIAL_MIN_SEC  = 0.5     # less audio than this makes Whisper guess wildly
MAIN_MODEL_PREVIEW_INTERVAL_SEC = 0.45   # preview cadence when a large main model
                                         # makes the previews (see _start_session)
# Music mode (see music.py): sung notes have no clear pauses, so the speech
# detector chopped songs into 1 s pieces Whisper could not understand. While
# music plays, a phrase needs a longer pause to end and may run longer, and
# the speech threshold is lower (singing scores lower than speech).
MUSIC_SILENCE_SEC    = 0.9
MUSIC_MAX_PHRASE_SEC = 15
# ...but once a sung phrase is this long, a shorter gap (a breath between
# lines) ends it. Without this a quarter of sung phrases ran to the 15 s
# limit and were cut mid-word, and a forced cut can't use the early final:
# measured ~1.4 s from the end of singing to the text.
MUSIC_SOFT_PHRASE_SEC  = 6
MUSIC_SOFT_SILENCE_SEC = 0.4
# Same idea for speech: 15% of phrases on real calls ran to MAX_PHRASE_SEC
# (people who barely pause). Past this length a short gap between words ends
# the phrase, so it isn't cut mid-word and can use the early final.
SPEECH_SOFT_PHRASE_SEC  = 7
SPEECH_SOFT_SILENCE_SEC = 0.2
MUSIC_VAD_DROP       = 0.15
VOICE_GAP_DB         = 12     # separated voice this close to the mix = singing
MIN_PHRASE_SEC   = 0.5     # shorter clips are mostly silence and make Whisper hallucinate
MAX_READ_FAILURES = 20     # consecutive audio read errors (~1 s) before capture gives up
WATCHDOG_SEC      = 2      # how often the UI checks that capture/workers are alive
VAD_WINDOW_SEC   = 0.25    # audio context given to Silero VAD per decision
VAD_HYSTERESIS   = 0.15    # once in speech, stay until probability drops this far below
PREROLL_SEC      = 0.6     # audio kept from before VAD fired, so first words aren't clipped
                           # when the detector reacts late (more likely at low sensitivity).
                           # RealtimeSTT keeps 1 s. Leading silence is trimmed off again
                           # before Whisper, so a longer pre-roll costs almost nothing.
EARLY_FINAL_SEC  = 0.1     # "early transcription on silence" (idea from RealtimeSTT): this far
                           # into a pause, start the final transcription speculatively. If the
                           # pause reaches SILENCE_SEC the result is used at once (saving the
                           # ~0.2 s transcription time); if speech resumes it is thrown away.
SILENCE_FLOOR_DBFS = -65   # below this it's digital silence: skip VAD entirely
SOURCE_SILENT_SEC  = 5     # this long with no sound at all -> "source_silent" warning
MIC_SILENT_SEC     = 60    # the same for a microphone (its noise gate is silent in pauses)
ONSET_PROB       = 0.2     # a speech score this high counts as "possibly speech": the time
                           # from then until the detector actually starts a phrase is logged
                           # as onset_delay_ms (how late, at this sensitivity, speech is noticed)
DEFAULT_SENS     = 0.5     # speech sensitivity 0..1; higher catches quieter/less clear speech
AGC_TARGET_DBFS  = -20     # clips are normalised to this RMS before Whisper...
AGC_MAX_GAIN_DB  = 30      # ...boosting quiet audio by at most this much
AGC_PEAK_DBFS    = -1      # and never letting peaks exceed this
NO_SPEECH_PROB   = 0.6     # Whisper's rule: discard only if no-speech prob is above
LOGPROB_FLOOR    = -1.0    # this AND the average token log-prob is below this
CONTEXT_TOKENS   = 96      # previous text fed back as context (Whisper's default)
CONTEXT_RESET_SEC = 6      # forget the context after this long a *silence* between
                           # phrases (timing it from the last finished phrase reset it
                           # mid-speech, since phrases last up to 7 s: that caused the
                           # "Спасибо." failures seen in the logs)
RECORD_DIR       = APP_DIR / "recordings"
RECORD_KEEP      = 400     # newest phrase recordings kept for accuracy checks

# VB-Audio virtual cables appear as "CABLE-A Output", "CABLE Output", etc.
# Voicemeeter mix buses are deliberately excluded so we capture the raw cable.
VB_KEYWORDS     = ["cable output", "cable-a output", "cable-b output",
                   "vb-audio virtual", "virtual cable"]
# Words that identify an NVIDIA Broadcast (noise-removing) microphone,
# preferred as the default for My Voice mode.
NVIDIA_KEYWORDS = ["nvidia", "broadcast", "rtx voice"]
# Fiio output devices: FiiO K7, K3, K5, K9, Q3, Q5, Q7, Q series, KA1, KA3, KA5
FIIO_KEYWORDS   = ["fiio", "fio k", "fio q", "fio dac",
                   "k3)", "k5)", "k7)", "k9)",
                   "q3)", "q5)", "q7)",
                   "ka1", "ka3", "ka5",
                   "btr", "realtek usb audio"]

# ── Whisper models ────────────────────────────────────────────────────────────
# MODELS maps the names shown in the Model menu to Whisper model sizes.
# Bigger models are more accurate but slower. LANGUAGES maps menu names to
# (Whisper language code, translation-service language code).
MODELS = {
    "Tiny  - fastest":        "tiny",
    "Small - recommended ✓":  "small",
    "Medium - most accurate": "medium",
    # Measured on real Russian call audio vs large-v3: 7.5% word errors at
    # ~225 ms, against Medium 9.9% at ~276 ms. It cannot translate (trained
    # for transcription only), so translation comes from Opus-MT/online.
    "Turbo - best accuracy ★": "large-v3-turbo",
}
# Model chosen on first start: Tiny without a GPU (anything bigger is too
# slow on a CPU), Turbo with one.
DEFAULT_MODEL_CPU = "Tiny  - fastest"
DEFAULT_MODEL_GPU = "Turbo - best accuracy ★"

# Model for live previews (finished phrases always use the main model).
# Measured preview cost on the RTX 4060 Ti: Tiny ~40 ms, Small ~90 ms,
# Turbo ~190 ms (which saturated the GPU and delayed finished phrases).
# "Off": no grey live text at all; only finished phrases are shown. Leaves
# the GPU free for finished phrases, so it is the baseline to compare the
# preview settings against (log_tools.py compare groups by this).
PREVIEW_OFF = "off"
PREVIEW_CHOICES: dict[str, str | None] = {
    "Off (final text only)": PREVIEW_OFF,
    "Tiny (instant)": "tiny",
    "Small (balanced)": "small",
    "Main model": None,
}

# Local neural translation (Opus-MT via CTranslate2, ~30-55 ms on GPU).
# Replaces the online services for these pairs: no network, no rate limits.
# Local Opus-MT translation models (University of Helsinki), converted for
# CTranslate2 and published on Hugging Face. Each is ~150-300 MB, downloaded
# the first time that language pair is needed. ooeoeo's float16 builds are
# used where they exist, gaudi's for a few more languages into English.
_OOEOEO_PAIRS = (
    # into English
    "ar-en bg-en cs-en da-en de-en es-en fr-en it-en ja-en ko-en ru-en tr-en vi-en zh-en "
    # from English
    "en-ar en-de en-es en-fr en-it en-ru en-zh "
    # into Russian
    "ar-ru bg-ru da-ru es-ru fr-ru ja-ru ko-ru vi-ru "
    # between other languages
    "de-es de-fr ar-de ar-es ar-fr ar-it ar-tr").split()
OPUS_MT_REPOS = {tuple(pair.split("-")): f"ooeoeo/opus-mt-{pair}-ct2-float16"
                 for pair in _OOEOEO_PAIRS}
OPUS_MT_REPOS.update({(src, "en"): f"gaudi/opus-mt-{src}-en-ctranslate2"
                      for src in ("uk", "pl", "nl", "hi")})
# Many languages -> English in one model: used when no direct model exists
OPUS_MT_MUL_EN = "gaudi/opus-mt-mul-en-ctranslate2"

# Menu name -> (Whisper language code, online-translation language code).
# They differ only for Chinese (Google needs "zh-CN").
LANGUAGES = {
    # Source only: Whisper detects the language of every phrase
    "Auto-detect":        ("auto", "auto"),
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
    "Bulgarian":          ("bg", "bg"),
    "Polish":             ("pl", "pl"),
    "Czech":              ("cs", "cs"),
}

# ── Automatic language detection ─────────────────────────────────────────
# Whisper guesses the language of each finished phrase from the audio
# encoding it already made (~4 ms). A German song or a Bulgarian advert in
# "Russian" mode used to come out as German/Bulgarian text "translated" by
# the Russian->English model (logs). Now such a phrase is transcribed in its
# own language, labelled, and translated via English. The switch needs a
# clear verdict, because short or sung Russian is sometimes misjudged (a
# 0.9 s clip scored Spanish 0.38 in the tests):
LANG_SWITCH_PROB  = 0.70   # the other language must score at least this
LANG_SOURCE_MAX   = 0.15   # ...and the expected language at most this
LANG_MIN_SPEECH_SEC = 1.5  # ...on at least this much speech
AUTO_LANG_PROB    = 0.50   # Auto-detect: below this on a short clip, keep the last language
# Auto-detect: a language NOT heard recently in this session needs this much
# certainty before the app switches to it. Live test (Russian songs, Bulgarian
# adverts): real switches scored 0.95-0.99; wrong ones on short sung fragments
# (Polish, Spanish, Portuguese, Turkish in a Russian song) scored 0.31-0.80.
NEW_LANG_PROB       = 0.85
NEW_LANG_MUSIC_PROB = 0.90  # while music plays (singing fools the detector more)
# Names for the label shown before a phrase in another language
LANGUAGE_NAMES = {code: name for name, (code, _) in LANGUAGES.items() if code != "auto"}
TARGET_LANGUAGES = [name for name in LANGUAGES if LANGUAGES[name][0] != "auto"]
LANGUAGE_NAMES.update({"bg": "Bulgarian", "be": "Belarusian", "sr": "Serbian", "pl": "Polish",
                       "cs": "Czech", "sv": "Swedish", "fi": "Finnish", "kk": "Kazakh",
                       "he": "Hebrew", "hu": "Hungarian", "ro": "Romanian", "da": "Danish",
                       "no": "Norwegian", "lv": "Latvian", "lt": "Lithuanian", "et": "Estonian"})

# ── UI colours ────────────────────────────────────────────────────────────────
# Colours used by the window. A pair means (light theme, dark theme).
COL_MY_VOICE = ("#1A6B3A", "#1A6B3A")
# COL_MY_* / COL_FR_*: My Voice (green) / Friend (blue) button colours,
# normal and when the mouse is over them; COL_INACTIVE: unselected mode;
# COL_STOP: the red Unmute button; COL_OK/WARN/ERR/INFO: status colours.
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

# ── Device matching ───────────────────────────────────────────────────────────
# Words that pick the default devices from their names.
# Ordered priority list for VB-Audio input auto-selection.
# CABLE-A is the conventional "loopback capture" cable in most setups;
# prefer it over CABLE-B or the un-lettered CABLE.
VB_INPUT_PRIORITY_KEYWORDS = [
    "cable-a output",          # VB-Audio Virtual Cable A
    "cable output (vb-audio",  # CABLE Output (VB-Audio Virtual Cable)
    "cable-b output",          # last resort: Cable B
    "voicemeeter out",         # Voicemeeter mix buses
]


MME_NAME_MAX = 31     # Windows' old MME interface cuts device names to 31 characters


# Ordered priority list for output device auto-selection: the first keyword
# that matches any device name wins.  This lets K7 beat Q series.
FIIO_PRIORITY_KEYWORDS = [
    "fiio k7", "fiio k9", "fiio k5", "fiio k3",  # desktop DAC/amps first
    "fiio q", "fiio ka",                           # portable
    "fiio",                                        # any other Fiio
    "realtek usb audio",                           # fallback (Fiio uses Realtek USB chip)
]


KEEPALIVE_ERROR_AFTER_TRIES = 15   # 1 s retries before a failing keep-alive is an error

# ── Performance and resources ─────────────────────────────────────────────────
STATS_INTERVAL_SEC = 10   # how often each component logs its perf summary
# Below these, other apps (or this one) may start to stutter.
LOW_RAM_MB, LOW_GPU_MEM_MB, HIGH_SYSTEM_CPU_PCT = 2048, 1024, 85

# ── Transcription checks ──────────────────────────────────────────────────────
# Spotting text Whisper invented (hallucinations) or results that look wrong.
# Whisper's stock outputs on silence/noise (learned from YouTube captions).
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "please subscribe", "subscribe", "you", "bye",
    "hmm", "um", "uh", "oh", "so",
    # Russian ones seen in the logs: dropped only when they are the WHOLE text
    "продолжение следует", "спасибо за просмотр", "подписывайтесь на канал",
}


# Short stock phrases Whisper emits in place of real speech it failed to
# decode (seen in the logs: "Спасибо." for 11-12 spoken words). They are real
# words, so they are only treated as suspect when far too short for the audio.
SUSPECT_PHRASES = {"спасибо", "спасибо за внимание", "продолжение следует",
                   "субтитры сделал dimatorzok", "редактор субтитров а синецкая",
                   "thank you", "thanks for watching"}
MIN_CHARS_PER_SEC = 5.0   # good phrases ran 9-17 chars/s in the logs; at 5 the check
                          # caught both real losses in 52 phrases (2 false alarms)
MAX_COMPRESSION   = 2.4   # Whisper's own repetition-loop check
RETRY_USELESS = {"stock_phrase"}                 # see _retry_if_suspicious
RECENT_AUDIO_PHRASES = 40                         # kept in memory for "Report a mistake"
FEEDBACK_DIR = APP_DIR / "testdata" / "feedback"  # reported phrases (simulator test set)


FEEDBACK_PROBLEMS = {
    "words":       "Wrong or missing words",
    "language":    "Wrong language",
    "translation": "Bad translation",
    "cut":         "Cut in the middle of a sentence",
    "invented":    "Text that was never said",
    "missed":      "Speech that was not shown at all",
}


# Fragments that only ever come from Whisper's training captions (YouTube
# subtitle credits), never from real speech; any text containing one is dropped.
HALLUCINATION_FRAGMENTS = ("субтитр", "dimatorzok", "редактор субтитров", "корректор",
                           "amara.org", "subtitles by",
                           "ご視聴ありがとうございました")   # "thanks for watching", invented on music
# Whisper marks a sentence cut between two phrases with these (see continuation_join).
CONTINUATION_MARKS = ("...", "…")


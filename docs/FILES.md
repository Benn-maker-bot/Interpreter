# What each file is for

---

## The table

| File | What it is | Open it when… |
| --- | --- | --- |
| `run.py` | Starts the app (`--classic` for the old window) | you want to run it (F5 in VS Code does this) |
| `interpreter\` | The application: audio, models, workers, logs, settings and the classic window (see below) | you want to change app services or the classic UI |
| `session_controller.py` | Shared session configuration and worker lifecycle | you want to change how capture/transcription/translation workers start or stop |
| `interpreter_qt.py` | **The Fluent window** you normally use | you want to change what you see or click |
| `music.py` | Music detection and vocal separation | songs are recognised badly |
| `simulator_qt.py` | **The simulator window** | you want to test with recordings or live audio |
| `simulate.py` | The simulator engine (also runs without a window) | you want to change how tests are scored |
| `log_tools.py` | Analyses the log files | you want reports: speed, accuracy, trends |
| `make_testdata.py` | Builds synthetic test clips (computer voices + music) | you want new or more test clips |
| `fetch_testdata.py` | Downloads real recorded test clips | first-time setup of "Recorded clips" |
| `README.txt` | Installing and using the app | setting up a new PC |
| `notes.md` | **Your** brain-dump file | an idea pops up mid-task |
| `docs/` | These guides | you want to understand something |
| [LEARNING.md](LEARNING.md) | A guided code tour and small experiments | you want to trace a phrase or learn by changing one behaviour |
| [FLOWCHARTS.md](FLOWCHARTS.md) | Expanded Mermaid diagrams for data flow, UI design, and testing | a picture would help explain a system boundary |

Files the program **writes** (don't edit by hand):

| File / folder | Contents |
| --- | --- |
| `interpreter.settings.json` | your choices (model, devices, sensitivity…) |
| `interpreter.log.jsonl` | everything that happened, one line per event |
| `interpreter.error.log.jsonl` | warnings and errors only |
| `recordings/` | audio of recent phrases (for accuracy checks) |
| `testdata/` | test clips, simulator results, your reported phrases |

---

## Inside the `interpreter\` package

The application code is split into modules, one per job. Each module's
opening docstring says what it does. Press **Ctrl+T** in VS Code to jump to
any class or function in the project by name.

| Module | Main parts | In one sentence |
| --- | --- | --- |
| `__init__.py` | — | Plain-language explanation of everything. **Read this first.** |
| `config.py` | `SILENCE_SEC`, `MODELS`, `LANGUAGES`… | The tuning numbers and menus (see [TUNING.md](TUNING.md)). |
| `logging_setup.py` | `log_event`, `explain_issue` | Writes every event to the log files; turns errors into plain words. |
| `system.py` | `detect_device`, `gpu_snapshot` | Finds the GPU, keeps Windows from slowing the app, measures resources. |
| `audio.py` | `AudioCaptureThread`, `get_audio_devices` | Lists sound devices, records sound, detects speech, cuts phrases. |
| `text_filters.py` | `is_hallucination`, `continuation_join` | Spots text Whisper invented and tidies the results. |
| `translation.py` | `LocalTranslator`, `TranslationWorker` | Translates on this PC, or via the internet when no local model exists. |
| `speech.py` | `Speaker` | Reads the translation aloud. |
| `settings.py` | `load_settings`, `save_settings` | Remembers your choices between runs. |
| `transcription.py` | `WhisperModelLoader`, `TranscriptionWorker` | Whisper, language choice, music, translation. |
| `gui_tk.py` | `InterpreterApp` | The classic window and compatibility callbacks for the Fluent window. |
| `__main__.py` | `main` | Starts the app (`python -m interpreter`; `run.py` calls it too). |

Each module imports only from the ones above it in this table, so there are
no circular imports.

---

## Inside `interpreter_qt.py`

| Part | In one sentence |
| --- | --- |
| `QVar`, `ComboShim`, `TextShim` | Adapters for settings and result handling shared with the classic window. |
| `LevelMeter`, `SpeechMeter` | The input level and speech meters. |
| `ReportDialogMistake` | The "Report a mistake" pop-up. |
| `QtInterpreterApp` | The window: Interpreter page and Diagnostics page. |

---

## Inside `simulator_qt.py`

| Part | In one sentence |
| --- | --- |
| `ISSUE_HELP.update(...)` | Plain-language explanations of simulator problems. |
| `AudioMonitor` | Plays the test clips on your FiiO, in step with the test. |
| `SimulatorWindow` | The window: settings, now playing, meters, text, results, Issues. |

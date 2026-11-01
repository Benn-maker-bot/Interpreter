# What each file is for

---

## The table

| File | What it is | Open it when… |
| --- | --- | --- |
| `interpreter.py` | **The whole app's logic** (and the old "classic" window) | you want to change how the app behaves |
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

Files the program **writes** (don't edit by hand):

| File / folder | Contents |
| --- | --- |
| `interpreter.settings.json` | your choices (model, devices, sensitivity…) |
| `interpreter.log.jsonl` | everything that happened, one line per event |
| `interpreter.error.log.jsonl` | warnings and errors only |
| `recordings/` | audio of recent phrases (for accuracy checks) |
| `testdata/` | test clips, simulator results, your reported phrases |

---

## Inside `interpreter.py` (top to bottom)

The file is long. It is split into **sections** marked with a line of `─` characters.
Use **Ctrl+Shift+O** in VS Code to jump to any class or function by name.

| Section | Main parts | In one sentence |
| --- | --- | --- |
| Opening comment | — | Plain-language explanation of everything. **Read this first.** |
| Logging | `log_event`, `log_exception` | Writes every event to the log files. |
| Error explanations | `ISSUE_HELP`, `explain_issue` | Turns errors into plain words for the Issues page. |
| Settings / constants | `SILENCE_SEC`, `MAX_PHRASE_SEC`… | The tuning numbers (see [TUNING.md](TUNING.md)). |
| Online translation | `TranslationWorker` | Translates via the internet when no local model exists. |
| Audio helpers | `get_audio_devices`… | Lists sound devices, finds the FiiO. |
| Audio capture | `AudioCaptureThread` | Records sound, detects speech, cuts phrases. |
| Model loading | `WhisperModelLoader` | Loads Whisper onto the graphics card. |
| Hallucination filters | `is_hallucination`, `suspicious_reasons` | Spots text Whisper invented. |
| Feedback | `save_feedback` | Saves "Report a mistake" reports. |
| Session context | `log_session_context`, `gpu_snapshot` | Logs the code version and GPU sharing. |
| Transcription | `TranscriptionWorker` | Whisper, language choice, music, translation. |
| Main window | `InterpreterApp` | The classic window; its logic is reused by the Fluent window. |

---

## Inside `interpreter_qt.py`

| Part | In one sentence |
| --- | --- |
| `QVar`, `ComboShim`, `TextShim` | Small adapters so the logic in `interpreter.py` can drive Qt widgets. |
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

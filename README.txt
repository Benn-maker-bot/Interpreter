Voice Interpreter
=================
Listens to speech, shows what was said, and translates it, live, on your own
PC. Speech recognition uses OpenAI's Whisper models and translation uses
Opus-MT models, both running locally on your NVIDIA graphics card (GPU).
Nothing is sent to the internet except for language pairs that have no local
translation model.

Typical use: in a Teams or Telemost call, your friend's Russian voice plays
through your FiiO headphones. In "Friend" mode the app listens to the FiiO,
shows the Russian text as it is spoken and the English translation beside it.
"My Voice" mode does the reverse with your microphone (English -> Russian).


QUICK START
-----------
  1. Install Python 3.13 from https://python.org (tick "Add Python to PATH").
  2. In this folder, create the project's virtual environment and install
     the packages (a virtual environment keeps them separate from other
     Python projects):
         py -3.13 -m venv .venv
         .venv\Scripts\python -m pip install -r requirements.txt
  3. Start the app (or double-click setup_and_run.bat):
         .venv\Scripts\python interpreter.py              Fluent window (default)
         .venv\Scripts\python interpreter.py --classic    old classic window
     Only one copy can run at a time (two copies would fight over the
     audio device and the log files).
  4. The first start downloads the AI models once (Turbo ~1.6 GB, Small
     ~480 MB, Opus-MT ~300 MB per direction). Later starts take seconds.

  The app starts listening by itself, in the mode you used last. In VS Code,
  press F5 and choose "Interpreter" or "Interpreter (classic window)".
  The Settings button (top right) hides the settings to give the
  transcript more room; A-/A+ change the text size. Both are remembered.


DIAGNOSTICS PAGE (Fluent window)
--------------------------------
  Click the tools icon in the left bar. Values are coloured green (fine),
  amber (worth a look) or red (problem); hover over a name for an
  explanation.
    Health        overall state, models and translator in use
    Speed         pause -> final text, first preview, recognition and
                  translation time, early transcription (last 50 phrases)
    Audio         level, speech share, onset delay, sensitivity, peaks
    Recognition   doubtful results, retries, empty phrases, context resets
    Computer      CPU, memory, GPU load, memory, temperature and clock
    Pipeline      waiting work, threads alive, watchdog restarts
    Recent issues every warning/error (errors also pop up at the top right)
    Recent phrases a table of the last 20 phrases with their timings
  Buttons: open the app folder and logs, Copy (all figures as text, e.g.
  to paste into a message) and Speed report (log_tools.py report for this
  run, shown in the window).


USING THE WINDOW
----------------
  My Voice / Friend   switch mode. Each mode remembers its own audio device.
                      Devices marked with a speaker icon record what an
                      OUTPUT device is playing (e.g. the FiiO during a call).
  Mute                stop listening and release the device; click again to
                      resume. Changing any setting restarts listening.
  Speak: On/Off       read each new translation aloud through "Audio output".
                      (Speaking Russian needs a Russian voice installed in
                      Windows Settings > Time & language > Speech.)
  Clear               empty the text boxes.
  Model               Whisper model for the final text:
                        Turbo   most accurate, fast on the GPU (recommended)
                        Medium  accurate, slower
                        Small   fast, less accurate
                        Tiny    fastest, many mistakes in Russian
  Speech sensitivity  higher = also catch quieter or less clear speech;
                      lower = ignore more background sound.
  Previews            model for the grey "live" text shown while someone is
                      still talking: Tiny (instant, rough), Small (balanced,
                      recommended) or Main model (best, but loads the GPU).
  Input level         a broadcast-style meter in dBFS: speech should sit
                      around -30 to -12 dBFS; red near 0 means too loud.


FILES IN THIS FOLDER
--------------------
  interpreter.py               the application (start it with this file)
                               and the classic window
                               (start reading at the top: it explains how
                               everything works)
  interpreter_qt.py            the Fluent window (PySide6); reuses all the
                               logic in interpreter.py
  log_tools.py                 analysis tool for the logs (see below)
  requirements.txt             the Python packages needed, with comments
  setup_and_run.bat            one-click install and start
  build_exe.bat                build a standalone VoiceInterpreter.exe
  interpreter.settings.json    your saved choices (created automatically)
  interpreter.log.jsonl        detailed log, one JSON object per line
  interpreter.error.log.jsonl  warnings and errors only
  recordings/                  audio of the newest 400 phrases, kept for
                               accuracy checks. Contains other people's
                               voices: add "record_audio": false to
                               interpreter.settings.json to turn it off.


CHECKING SPEED AND ACCURACY
---------------------------
  After a session, from this folder:
      .venv\Scripts\python log_tools.py report
          time taken by every stage (speech detection, GPU, screen…),
          resource use, and a list of problems found
      .venv\Scripts\python log_tools.py accuracy
          re-checks the saved recordings with a larger model (large-v3,
          ~3 GB download once) and lists words the app got wrong
  Add a start time to look at one session only, e.g.
      .venv\Scripts\python log_tools.py report 2026-10-31T14:00


TROUBLESHOOTING
---------------
  The header shows CPU instead of GPU
      The NVIDIA libraries did not load. Re-run the pip install command
      above; check interpreter.error.log.jsonl for "cuda_detection".

  "Waiting for … to reconnect"
      The audio device stopped responding (unplugged, driver reset). The
      app resumes by itself when the same device is back.

  Words are missing or wrong
      Use the Turbo model. Check the level meter (speech around -30 to
      -12 dBFS). Run "log_tools.py accuracy" to see exactly what was missed.

  Text appears slowly
      Set Previews to Small or Tiny. "log_tools.py report" shows which
      stage is slow.

  "Audio device error"
      Another app may have exclusive control of the device. In Windows
      Sound settings > device Properties > Advanced, untick "Allow
      applications to take exclusive control".

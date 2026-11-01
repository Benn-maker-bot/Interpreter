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
  Report a mistake    mark a recent phrase as wrong (wrong words, wrong
                      language, bad translation, cut mid-sentence, invented
                      or missed speech), optionally with what was really
                      said. Its audio and details are saved in
                      testdata\feedback\, where the simulator can replay
                      them ("Your reported phrases") to check a fix.
  Model               Whisper model for the final text:
                        Turbo   most accurate, fast on the GPU (recommended)
                        Medium  accurate, slower
                        Small   fast, less accurate
                        Tiny    fastest, many mistakes in Russian
  Speech sensitivity  higher = also catch quieter or less clear speech;
                      lower = ignore more background sound.
  Previews            model for the grey "live" text shown while someone is
                      still talking: Off (no live text; only finished
                      phrases, GPU free for them), Tiny (instant, rough),
                      Small (balanced, recommended) or Main model (best, but
                      loads the GPU). Compare them afterwards with
                      "log_tools.py compare" (see below).
  Input level         a broadcast-style meter in dBFS: speech should sit
                      around -30 to -12 dBFS; red near 0 means too loud.


OTHER LANGUAGES
---------------
  Translate: choose "Auto-detect" as the spoken language and the app
  works out the language of every phrase itself (Whisper's language
  detection), labels it when it changes ("[German] ...") and translates it
  into the chosen target. Local translation models (Opus-MT, ~150-300 MB
  each, downloaded the first time a language is heard) cover German,
  French, Spanish, Italian, Russian, Ukrainian, Polish, Bulgarian, Czech,
  Dutch, Turkish, Arabic, Chinese, Japanese, Korean and more; pairs with no
  direct model go via English, and the rest use the online translator.

  "Detect other languages" (next to Translate) lets the app notice when a
  phrase is clearly in a different language than expected, e.g. a German
  song or a Bulgarian advert while listening for Russian. Such a phrase is
  recognised in its own language, labelled ("[German] ...") and translated
  online from that language. If it is already in the target language it is
  shown as it is. Short or unclear phrases stay in the expected language
  (Whisper sometimes misjudges a second of singing). Untick the box to
  only label foreign phrases. "log_tools.py report" lists the languages
  detected.


MUSIC AND SINGING
-----------------
  The app notices when music is playing (YAMNet sound classifier) and
  switches to music mode by itself; the status line says "Music detected".
  In music mode:
    - a phrase needs a longer pause to end (0.9 s; 0.4 s once it is
      longer than 6 s) and may last 15 s, so sung lines are not chopped
      into single syllables
    - the voice is separated from the music in 6 s chunks while the
      singing is still going, so little is left to do when it ends
    - the speech detector is more sensitive (singing scores lower)
    - the instruments are removed (UVR vocal separation, on the GPU) so
      Whisper hears only the voice; instrumental passages are skipped
    - earlier sentences are not used as context for lyrics
  It returns to normal when the music stops. To turn it off, add
  "music_mode": false (or just "vocal_separation": false) to
  interpreter.settings.json. "log_tools.py report" has a "Music and
  singing" section.


UNDERSTANDING THE PROJECT
-------------------------
  docs\START_HERE.md is the place to begin: short guides with diagrams
  (MAP.md), what each file is for (FILES.md), step-by-step recipes
  (HOW_TO.md), the tuning numbers (TUNING.md), a glossary, and tips for
  a calm VS Code setup (VSCODE.md). In VS Code, press Ctrl+Shift+V on a
  .md file to read it formatted. notes.md is a scratch pad for ideas.


FILES IN THIS FOLDER
--------------------
  interpreter.py               the application (start it with this file)
                               and the classic window
                               (start reading at the top: it explains how
                               everything works)
  interpreter_qt.py            the Fluent window (PySide6); reuses all the
                               logic in interpreter.py
  music.py                     music/singing detection and vocal separation
  log_tools.py                 analysis tool for the logs (see below)
  simulate.py                  tests the app with recorded audio (see below)
  simulator_qt.py              the simulator in a window: watch and hear it
  make_testdata.py             builds the synthetic test clips
  fetch_testdata.py            downloads the simulator's test recordings
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


TESTING WITH RECORDED AUDIO (SIMULATOR)
---------------------------------------
  The simulator plays recordings through the real pipeline (speech
  detection, music mode, Whisper, language detection, translation) instead
  of a sound card, and scores the results against the correct text.
      .venv\Scripts\python simulator_qt.py
          the simulator in a window (in VS Code: Run and Debug ->
          "Simulator (window)", F5). Choose the test data: synthetic clips
          (one language each, exact text), recorded clips, your reported
          phrases, or live audio playing in Windows (a radio stream,
          Spotify …). Shows the clip, meters, live text and results, and
          plays the clips on an output device of your choice.
      .venv\Scripts\python make_testdata.py
          builds the synthetic clips once (needs internet): computer
          voices in 10 languages, and the same over generated music
          downloads ~30 MB of test clips once: read speech in 10 languages
          (Google FLEURS) and sung lines with music in 4 (JamendoLyrics)
      .venv\Scripts\python simulate.py
          plays them all (switching language every clip), then prints per
          language: language detected correctly, word error rate (WER),
          translated, pause -> text and first preview times
      .venv\Scripts\python simulate.py --src Russian --tgt English --kinds song
          other settings; see the top of simulate.py for all options
  Results are saved in testdata\sim_results\.


CHECKING SPEED AND ACCURACY
---------------------------
  After a session, from this folder:
      .venv\Scripts\python log_tools.py report
          time taken by every stage (speech detection, GPU, screen…),
          resource use, and a list of problems found
      .venv\Scripts\python log_tools.py accuracy
          re-checks the saved recordings with a larger model (large-v3,
          ~3 GB download once) and lists words the app got wrong
      .venv\Scripts\python log_tools.py compare
          one row per combination of model, Previews and sensitivity used:
          speed (pause -> text, first text) and, with --ref, accuracy
      .venv\Scripts\python log_tools.py trend
          headline figures per day, real use and simulations side by
          side: empty and retried phrases, forced cuts, end of speech ->
          text, errors, mistakes reported, GPU shared with other programs
  Every phrase's log line also says WHY things happened: which pause
  rule ended it, why the language was kept or changed, why text was
  discarded or retried. Each session logs its code version and which
  other programs were using the graphics card.
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

Voice Interpreter  (Whisper edition)
=====================================
Real-time speech recognition powered by OpenAI Whisper (local AI model).
Dramatically better accuracy for Russian and all other supported languages.


WHAT'S NEW vs THE PREVIOUS VERSION
------------------------------------
  - Switched from Google Web Speech API to faster-whisper (local Whisper AI)
  - Custom VAD (voice activity detection) pipeline — ~32 ms audio chunks,
    auto-calibrated ambient noise threshold, responsive phrase detection
  - Whisper model selector: Tiny / Small / Medium
  - Sensitivity slider to tune how quiet a sound triggers recording
  - Language forced at the Whisper level (no language-detection overhead)


QUICK START
-----------
  1. Install Python 3.9+ from https://python.org
     (tick "Add Python to PATH")
  2. Double-click setup_and_run.bat
  3. On first "Start Listening", the Whisper model downloads once (~450 MB
     for Small) and is cached for all future sessions.


WHISPER MODEL GUIDE
-------------------
  Tiny   ~150 MB  Fastest. Good for clear audio, simple sentences.
  Small  ~450 MB  Best balance. Recommended for Russian. (default)
  Medium ~1.5 GB  Most accurate. Use if Small misses words. Slower.


SENSITIVITY SLIDER
------------------
  The slider adjusts how loud audio needs to be before it is treated as
  speech.  Default (1.0×) is calibrated automatically at startup.

  Lower value (e.g. 0.5×) — picks up quieter / more distant audio
  Higher value (e.g. 2.0×) — only reacts to louder sounds; ignores
                              background noise better


VIRTUAL AUDIO CABLE (capture Teams, Zoom, etc.)
-------------------------------------------------
  1. In Teams/Zoom: Settings → Speaker → CABLE Input (VB-Audio Virtual Cable)
  2. In Voice Interpreter: Audio source → ★ CABLE Output (auto-selected)
  3. Press Start Listening

  To still hear audio yourself while capturing:
  Recording tab → CABLE Output → Properties → Listen tab →
  "Listen to this device" → playback through your Fiio DAC


TROUBLESHOOTING
---------------
  "Whisper load error":
    Run: pip install faster-whisper --upgrade

  Transcription is slow:
    Switch to Tiny model. On CPU, Small takes ~1-2 s per phrase.

  Missing words / poor accuracy:
    Try Medium model. Lower the Sensitivity slider slightly.
    Ensure audio source is clean (not too compressed).

  "Audio device error":
    Another app may have exclusive control. Check Windows Sound settings
    → set device to Shared mode.

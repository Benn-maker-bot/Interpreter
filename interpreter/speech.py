"""Speaking the translation aloud (text-to-speech through Windows SAPI).

Speaker is a thread that owns the Windows speech object, so speech never
blocks the window and stop() silences it at once.
"""

import queue
import threading
import time
import logging

from .logging_setup import log_event, log_exception
from .audio import plain_device_name


class Speaker(threading.Thread):
    """
    Text-to-speech via Windows SAPI, owned by one thread (COM objects are
    apartment-bound). Speech is asynchronous, so stop() purges it instantly,
    unlike pyttsx3's blocking runAndWait(). `speaking` is set while audio is
    playing so the capture thread can ignore our own voice.
    
    This class handles text-to-speech functionality using Windows' built-in
    SAPI (Speech Application Programming Interface) system. It ensures that
    speech output is handled by a single dedicated thread to prevent conflicts
    with COM objects and provides methods for speaking text asynchronously.
    """
    SVSF_ASYNC, SVSF_PURGE = 1, 2
    HANGOVER_SEC = 0.4   # loopback capture lags playback slightly

    def __init__(self):
        """Set up the command queue; the SAPI voice is created inside the thread."""
        super().__init__(daemon=True, name="tts")
        # _commands: queue of commands for the speech thread; speaking: set while
        # audio is playing; ready: set once the voice is created; error: last
        # problem (e.g. no voice for the language), shown in the status bar.
        self._commands    = queue.Queue()
        self.speaking = threading.Event()
        self.ready    = threading.Event()
        self.error    = None

    def configure(self, output_label: str, language: str):
        """Choose the output device and a voice for `language` (e.g. "English")."""
        self._commands.put(("config", output_label, language))

    def say(self, text: str):
        """Queue text to be spoken (does not wait for it to finish)."""
        self._commands.put(("say", text))

    def stop_speech(self):
        """Stop speaking immediately and drop anything queued."""
        self._commands.put(("stop",))

    def shutdown(self):
        """Stop speaking and end the thread (when the app closes)."""
        self._commands.put(("quit",))

    def _configure(self, voice, output_label, language):
        """Apply an output device and language to the SAPI voice.

        Returns an error message if no voice for the language is installed.
        """
        want_out = plain_device_name(output_label).lower()
        outputs  = voice.GetAudioOutputs()
        for i in range(outputs.Count):
            tok = outputs.Item(i)
            if tok.GetDescription().lower() == want_out:
                voice.AudioOutput = tok
                break
        voices = voice.GetVoices()
        chosen = None
        for i in range(voices.Count):
            tok = voices.Item(i)
            if language.lower() in tok.GetDescription().lower():
                chosen = tok
                break
        if chosen is not None:
            voice.Voice = chosen
        log_event(logging.INFO, "tts_configured", output=want_out, language=language,
                  voice=chosen.GetDescription() if chosen else None)
        if chosen is None:
            return f"No {language} text-to-speech voice installed (Windows Settings → Speech)."
        return None

    def run(self):
        """Thread loop: carry out queued commands and keep `speaking` up to date."""
        try:
            # Create the Windows speech engine (a SAPI 'SpVoice' COM object).
            # CoInitialize prepares this thread to use COM; COM objects must be
            # used from the thread that created them, which is why all speaking
            # happens on this one thread, driven by commands from a queue.
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except Exception as e:
            self.error = str(e)
            log_exception("tts_init_failed", e)
            self.ready.set()
            return
        self.ready.set()
        # Main loop: wait up to 50 ms for a command, then carry it out.
        # Commands are tuples: ('say', text), ('stop',), ('config', device,
        # language) or ('quit',).
        quiet_since = None
        while True:
            try:
                cmd = self._commands.get(timeout=0.05)
            except queue.Empty:
                cmd = None
            try:
                if cmd is None:
                    pass
                # SVSF_ASYNC = start speaking and return at once (don't wait);
                # SVSF_PURGE = throw away anything still being said or queued.
                # Speaking an empty string with PURGE is SAPI's way to stop instantly.
                elif cmd[0] == "quit":
                    voice.Speak("", self.SVSF_ASYNC | self.SVSF_PURGE)
                    break
                elif cmd[0] == "stop":
                    voice.Speak("", self.SVSF_ASYNC | self.SVSF_PURGE)
                elif cmd[0] == "say":
                    voice.Speak(cmd[1], self.SVSF_ASYNC)
                    self.speaking.set()
                elif cmd[0] == "config":
                    self.error = self._configure(voice, cmd[1], cmd[2])
                # Keep the `speaking` flag up to date. The capture thread reads it to
                # ignore our own voice when it plays on the device being captured. It
                # stays set for HANGOVER_SEC after speech ends, because the recorded
                # audio lags slightly behind what is played.
                # RunningState 2 = SRSEIsSpeaking
                if voice.Status.RunningState == 2:
                    self.speaking.set()
                    quiet_since = None
                elif self.speaking.is_set():
                    quiet_since = quiet_since or time.monotonic()
                    if time.monotonic() - quiet_since > self.HANGOVER_SEC:
                        self.speaking.clear()
            except Exception as e:
                log_exception("tts_command_failed", e, command=cmd[0] if cmd else None)
        self.speaking.clear()

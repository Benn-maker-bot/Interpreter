"""simulate.py swaps in a fake sound card by setting interpreter.audio.pyaudio.

This checks the swap really reaches the audio code. It would fail if the
audio functions moved to another module, or if they got pyaudio some other
way, and the simulator would then quietly use the real sound card.
"""
import interpreter


class FakePyAudioModule:
    """Stands in for the pyaudio module: just enough for portaudio_start()."""
    class PyAudio:
        pass


def test_swapping_audio_pyaudio_reaches_the_audio_code(monkeypatch):
    monkeypatch.setattr(interpreter.audio, "pyaudio", FakePyAudioModule)
    assert isinstance(interpreter.portaudio_start(), FakePyAudioModule.PyAudio)


def test_swapping_the_package_copy_does_not(monkeypatch):
    # The trap this guards against: interpreter.pyaudio is only the package's
    # copy of the name. The audio code never looks at it.
    monkeypatch.setattr(interpreter, "pyaudio", FakePyAudioModule)
    assert interpreter.audio.pyaudio is not FakePyAudioModule

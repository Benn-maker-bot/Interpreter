"""Start the Voice Interpreter.

    python -m interpreter            the Fluent window (interpreter_qt.py)
    python -m interpreter --classic  the original customtkinter window

PYTHON: `python -m interpreter` runs this file. Python first imports the
package (interpreter/__init__.py, which loads every module), then runs this
file with __name__ == "__main__". run.py in the project folder does the same
for VS Code's F5, setup_and_run.bat and the PyInstaller build.
"""
import logging
import sys

from . import (APP_DIR, disable_power_throttling, install_global_exception_hooks,
               log_event, log_exception, run_classic)


def acquire_single_instance():
    """Allow only one copy of the app at a time. Two copies would fight over
    the audio device and the GPU, and both write the same log file at once,
    which corrupted a log line in testing. Returns an open lock file to keep
    for the app's lifetime, or None if another copy already holds it."""
    import msvcrt
    lock = open(APP_DIR / "interpreter.lock", "a+")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return lock
    except OSError:
        lock.close()
        return None


def main():
    """Check for a second copy, then open the Fluent or the classic window."""
    install_global_exception_hooks()
    # instance_lock must stay open while the app runs. It does: this function
    # doesn't return until the window closes, and the name keeps it alive.
    instance_lock = acquire_single_instance()
    if instance_lock is not None:
        disable_power_throttling()
    if instance_lock is None:
        log_event(logging.WARNING, "second_instance_refused")
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None, "Voice Interpreter is already running.", "Voice Interpreter", 0x40)
        sys.exit(0)
    if "--classic" in sys.argv:
        run_classic()
    else:
        # interpreter_qt does `import interpreter`. The package is already
        # loaded (Python imported it before running this file), so it gets
        # the same module and nothing is initialised twice.
        try:
            import interpreter_qt
        except ImportError as e:
            log_exception("fluent_window_unavailable", e)
            run_classic()     # PySide6 not installed: fall back to the classic window
        else:
            interpreter_qt.main()


if __name__ == "__main__":
    main()

"""The app's files must sit in the project folder, not inside interpreter\\.

The code lives in the interpreter\\ package, one folder below the project.
A path built with Path(__file__).parent in a package module points at
interpreter\\, which would quietly start new, empty logs and settings there.
"""
from pathlib import Path

import interpreter

PROJECT = Path(__file__).resolve().parent.parent     # tests\ -> project folder


def test_app_dir_is_the_project_folder():
    assert interpreter.APP_DIR == PROJECT


def test_logs_and_settings_are_in_the_project_folder():
    for path in (interpreter.LOG_PATH, interpreter.ERROR_LOG_PATH,
                 interpreter.LOG_SUMMARY_PATH, interpreter.SETTINGS_PATH):
        assert path.parent == PROJECT, path


def test_recordings_and_feedback_are_in_the_project_folder():
    assert interpreter.RECORD_DIR == PROJECT / "recordings"
    assert interpreter.FEEDBACK_DIR == PROJECT / "testdata" / "feedback"

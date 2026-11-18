"""Start the Voice Interpreter: `python run.py` (add --classic for the old window).

The app itself lives in the interpreter\\ package; this file only starts it,
so VS Code's F5, setup_and_run.bat and PyInstaller have a script to point at.
`python -m interpreter` does the same thing.
"""
from interpreter.__main__ import main

if __name__ == "__main__":
    main()

"""The Fluent window must not load the classic window's GUI toolkit.

`import interpreter` should not import customtkinter (or Tk); only using a
classic-window name such as interpreter.InterpreterApp should. Each check
runs in a fresh Python, because this test session may already have loaded
customtkinter for another test.
"""
import subprocess
import sys

CHECK = """
import sys, interpreter
print("RESULT", "customtkinter" in sys.modules, "tkinter" in sys.modules)
{extra}
"""


def run(extra=""):
    out = subprocess.run([sys.executable, "-c", CHECK.format(extra=extra)],
                         capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    # The app also logs to the console, so keep only our "RESULT ..." lines.
    lines = [line.split()[1:] for line in out.stdout.splitlines() if line.startswith("RESULT")]
    return [word for line in lines for word in line]


def test_importing_the_package_does_not_load_tk():
    assert run() == ["False", "False"]


def test_the_classic_window_still_loads_on_demand():
    result = run("interpreter.InterpreterApp\nprint('RESULT', 'customtkinter' in sys.modules)")
    assert result == ["False", "False", "True"]

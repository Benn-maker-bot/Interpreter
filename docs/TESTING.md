# Testing with pytest

Automated tests check that small pieces of the program still do what they
should. Run them before and after every change: if a test fails, the change
broke something.

## One-time set-up

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
```

## Run the tests

From the project folder:

```powershell
.venv\Scripts\python -m pytest          # all tests
.venv\Scripts\python -m pytest -v       # one line per test
.venv\Scripts\python -m pytest tests\test_text_helpers.py   # one file
.venv\Scripts\python -m pytest -k rms   # only tests with "rms" in the name
```

Or in VS Code: open the **Testing** panel (the flask icon in the left bar),
then click **Run Tests**. Click the green triangle beside one test to run
just that test, or right-click it and choose **Debug Test** to stop at your
breakpoints.

## Files

| File | Purpose |
|---|---|
| `pytest.ini` | pytest settings: where tests live, and that `import interpreter` works |
| `requirements-dev.txt` | packages needed for development only (pytest) |
| `tests\conftest.py` | shared set-up; stops tests writing to the real logs |
| `tests\test_audio_levels.py` | `rms()` and `levels_dbfs()` |
| `tests\test_text_helpers.py` | `continuation_join()`, `trailing_marks_length()`, `clean_translation()`, `collapse_repetitions()` |

## Writing a test

1. Create or open a file in `tests\` whose name starts with `test_`.
2. Write a function whose name starts with `test_`.
3. Inside it: build the input, call the function, then `assert` the result.

```python
from interpreter import rms

def test_rms_of_silence_is_zero():
    assert rms(b"\x00\x00" * 100) == 0.0
```

Useful tools:

- `pytest.approx(x)` compares floating-point numbers with a small tolerance.
- `@pytest.mark.parametrize` runs one test with several sets of inputs.
- `with pytest.raises(ValueError):` checks that code raises an error.

## What makes a good test

- Tests **one behaviour**, and its name says which.
- Needs no microphone, GPU, network or window, so it runs in a second.
- Gives the same result every time.

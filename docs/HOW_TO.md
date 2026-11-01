# How to…

Short recipes. Each one is a small task you can do in one sitting.
Commands go in the VS Code terminal (**Ctrl+`**), in the project folder.

---

## Run the interpreter

1. **Ctrl+Shift+D** (Run and Debug).
2. Choose **Interpreter** at the top.
3. Press **F5**. (**Shift+F5** stops it.)

---

## Run the simulator window

1. **Ctrl+Shift+D** → **Simulator (window)** → **F5**.
2. Choose the **Test data** and press **Start**.
3. Watch the Results table fill in. Hover over a row to see the correct text.

Tip: "Per language: 1" gives a quick 5-minute test.

---

## Report a mistake the app made

1. In the interpreter window, press **Report a mistake**.
2. Pick the phrase, tick what was wrong.
3. If you can, type what was really said.
4. **Save report.**

Later, the simulator can replay all your reports: Test data → **Your reported phrases**.

---

## Check a change (before / after)

1. Before changing anything: simulator, **Per language: 1**, Start. Note the totals line.
2. Make **one** change.
3. Run the same test again.
4. Compare: language right, WER (lower is better), pause → text (lower is better).
5. Better or the same? Commit. Worse? Undo it (**Ctrl+Z**, or `git checkout -- filename`).

---

## See how things went (reports)

| Command | Shows |
| --- | --- |
| `.venv\Scripts\python log_tools.py trend` | one line per day: phrases, errors, speed |
| `.venv\Scripts\python log_tools.py report` | time taken by each step, and problems found |
| `.venv\Scripts\python log_tools.py accuracy` | words the app got wrong (slow: uses a big model) |

---

## Understand an error

1. Look at the **Issues** page (simulator) or **Diagnostics → Recent issues** (interpreter).
2. Each problem says: what it means, what to do, the error, **where** in the code, and the details.
3. "Where" gives a file and line number: **Ctrl+P**, type the file name, then `:` and the line number (for example `interpreter.py:2410`).

---

## Change a tuning number

1. Open [TUNING.md](TUNING.md) and find the setting.
2. In `interpreter.py`, press **Ctrl+F** and search for its name (for example `SILENCE_SEC`).
3. Change the number. Keep the comment next to it up to date.
4. Do "Check a change" above.

---

## Save your work (git)

1. **Ctrl+Shift+G** opens Source Control.
2. Type a short message saying what changed.
3. Press **Commit**.

Or ask Claude: "commit the changes".

---

## Get back after a break

1. Open `notes.md` – what were you doing?
2. **Ctrl+Shift+G** – which files did you change?
3. Pick **one** small next step.

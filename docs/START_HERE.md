# Start here

A short guide to the Voice Interpreter project.
Read one section at a time. Each one stands on its own.

---

## What the program does (1 minute)

- It **listens** to sound: a call in your headphones, or your microphone.
- It **writes down** what was said (speech recognition, "Whisper").
- It **translates** it (for example Russian → English).
- It shows both side by side, **live**, while people are still talking.

---

## The other guides

Open them in VS Code and press **Ctrl+Shift+V** to see them nicely formatted.

| Guide | Read it when you want to… |
| --- | --- |
| [MAP.md](MAP.md) | see **pictures** of how everything fits together |
| [FILES.md](FILES.md) | know **what each file is for** |
| [HOW_TO.md](HOW_TO.md) | **do** something: run, test, check the logs, change a setting |
| [TUNING.md](TUNING.md) | understand the **numbers** that control the behaviour |
| [GLOSSARY.md](GLOSSARY.md) | look up a **word** you don't know |
| [VSCODE.md](VSCODE.md) | make VS Code **calmer and easier** to work in |
| [../notes.md](../notes.md) | jot down an idea quickly and get back to your task |

---

## The three things you will use most

1. **The interpreter window** – the real app.
   In VS Code: Run and Debug (**Ctrl+Shift+D**) → **Interpreter** → **F5**.
2. **The simulator window** – tests the app with recordings, and shows you what it hears.
   Run and Debug → **Simulator (window)** → **F5**.
3. **The trend report** – how well things went, day by day.
   Terminal: `.venv\Scripts\python log_tools.py trend`

---

## A safe reading order (one 15–20 minute block each)

Stop after any block. Nothing breaks if you only read part of it.

1. This page.
2. [MAP.md](MAP.md) – the first picture only ("The big picture").
3. [FILES.md](FILES.md) – just the table at the top.
4. The opening comment of `interpreter.py` (lines 1–250). It explains the
   whole app in plain words.
5. [GLOSSARY.md](GLOSSARY.md) – skim; come back when a word is unclear.
6. One class at a time, in the order in [MAP.md](MAP.md) ("A phrase's journey").

---

## Ground rules for changing code

- **Change one thing at a time.**
- **Test it** with the simulator before and after (see HOW_TO.md → "Check a change").
- **Commit** when it works, so you can always go back.
- If something goes wrong, the **Issues / Diagnostics page** explains the error in plain words.

---

## Help from Claude

Highlight confusing code and ask, for example:

- "Explain this code like I am a beginner, in short bullet points."
- "What does this function receive and return?"
- "Draw a diagram of what happens when this runs."

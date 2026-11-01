# Making VS Code calmer and easier

Most of this is already set up in the project (`.vscode/settings.json`).

---

## Already done for you

| Setting | What it does |
| --- | --- |
| Dim unfocused panels | Panels you aren't typing in are dimmed, so your eyes stay on one place. |
| Calmer error checking | Pylance "basic" mode: real mistakes are underlined, not hundreds of "type unknown" notes. |
| Calm cursor | The cursor doesn't blink. |
| No minimap | The small code overview on the right is hidden (less clutter). |
| Sticky scroll | The name of the function you are in stays at the top while you scroll. |
| Bracket colours and guides | Matching brackets have the same colour, so nesting is easier to see. |
| Spell check words | Project words (FiiO, recognise…) aren't underlined. |
| Run configurations | F5 runs the interpreter or simulator with the right Python. |

---

## Extensions to install

When you open the project, VS Code offers **"Install recommended extensions"**. Say yes.
Or: **Ctrl+Shift+X**, type `@recommended`.

| Extension | Why |
| --- | --- |
| Python + Pylance | Shows mistakes **as you type**, and explains functions when you hover over them. |
| autoDocstring | Type `"""` under a function and press Enter: a comment template appears. |
| Markdown Preview Mermaid Support | Draws the diagrams in [MAP.md](MAP.md). |
| Code Spell Checker | Underlines typos in comments. |

---

## Shortcuts worth learning (one at a time)

| Keys | Does |
| --- | --- |
| **Ctrl+K Z** | **Zen mode**: hides everything except the code. Press Esc twice to leave. |
| **Ctrl+Shift+O** | Jump to a function or class in this file. |
| **Ctrl+P** | Open a file by typing part of its name. |
| **F12** | Go to where a function is defined. **Alt+←** goes back. |
| **Ctrl+K Ctrl+0** | Fold all code to just the outline; **Ctrl+K Ctrl+J** unfolds. |
| **Ctrl+Shift+V** | Show a Markdown file (like this one) formatted. |
| **Ctrl+`** | Open or close the terminal. |

---

## Choosing a calm colour theme

**Ctrl+K Ctrl+T**, then try **GitHub Dark Default**, **Solarized Dark** or **Solarized Light**.
Use the arrow keys to preview them without committing.

---

## Why no automatic code formatter (Black)?

Black reformats a whole file when you save it. The project's files were never
formatted with Black, so the first save would change thousands of lines at once
and hide your real change. If you want it later, ask Claude to format the
project in one separate commit first.

---

## Pacing tips

- Work in **15–20 minute blocks**. Stop at the end of a block, even mid-task.
- Before stopping, write **one line** in `notes.md`: what you were doing and the next step.
- Open **only the files you need**: **Ctrl+K W** closes all open files, then open just the one for this task.

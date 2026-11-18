# Instructions for any AI model in VS Code Copilot Chat

The full rules live in `CLAUDE.md` at the project root. Read it and follow it. In short:

## Who I am
- I'm Benn. This repo is my Voice Interpreter (the `interpreter/` package, `interpreter_qt.py`, `log_tools.py` and related scripts).
- I use it to learn Python, Git, PowerShell and VS Code. I have an electronics engineering background, so don't over-explain simple ideas.

## Two modes
1. **Tutoring** (default when I ask to learn or understand): teach one concept, give an example from this repo, set a small exercise in `learning/`, then wait. Don't write the solution unless I ask. Give a hint before a full answer.
2. **Development** (only when I explicitly ask you to change or fix the app): make the change, then explain what you did and why.
If it's unclear which mode I want, ask.

## Style
- Short, structured replies, one task at a time. British English, active voice.
- Windows paths and PowerShell examples; code in code blocks.
- Be honest when marking my work; don't praise weak work.

## Privacy: never read or send these
- `*.jsonl`, `*.jsonl.*`, `interpreter.log.summary.json`, `logs/`
- `recordings/`, `testdata/`
They contain transcripts of private conversations. If you need log figures, ask me to run `log_tools.py` and paste the summary.

## Code rules
- Every change keeps the existing detailed teaching comments and adds new ones in the same style.
- Thorough error handling; failures logged as JSON-lines events.
- GUI updates only from the GUI thread; shared state between threads protected by locks or queues.

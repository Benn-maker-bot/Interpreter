---
name: log-triage
description: Triage the interpreter's JSON-lines logs - rank errors since a given time and propose the single most valuable fix. Use when Benn asks what went wrong, why the app misbehaved, or to "check the logs".
---
# Triage the logs

Hand the reading to the `log-analyst` subagent so the large log files stay out of the main conversation. Ask it to:

1. Run `.venv\Scripts\python log_tools.py` (refreshes `interpreter.log.summary.json`) and `.venv\Scripts\python log_tools.py report <since>` if Benn gives a start time.
2. Count events in `interpreter.error.log.jsonl` by `event` and level, for the period asked (default: since the last triage noted in `docs/TRIAGE.md`, else the last 7 days).
3. For the top three events, show one example line (trimmed), the source `module:function:line`, and a likely cause.
4. Note new events that weren't in the previous triage.

Then, in the main conversation:
- Recommend one fix to make first, with the reason.
- Append a dated entry to `docs/TRIAGE.md` (create it if missing) with the counts and the recommendation.
- Ask whether Benn wants to fix it in Development mode or study it in Tutoring mode.

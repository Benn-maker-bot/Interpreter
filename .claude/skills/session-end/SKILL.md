---
name: session-end
description: End a tutoring or development session - update learning/progress.md and offer a Git commit as practice. Use when Benn says "stop", "end session", "that's enough for today".
---
# End a session

1. Summarise in three lines what Benn covered, what went well and what needs another look.
2. Update `learning/progress.md`:
   - Replace "Last session" with today's date and topics.
   - Add new weak spots; remove ones Benn has now answered correctly twice.
   - Rewrite "Next time" with the very next step.
   - Add one line at the top of "Session log".
3. Update "Current position" in `CLAUDE.md` to match (three lines at most).
4. Run `git status`. If app files changed, suggest a commit message and let Benn type the `git add` / `git commit` commands himself. Remind him that `learning/` is git-ignored.

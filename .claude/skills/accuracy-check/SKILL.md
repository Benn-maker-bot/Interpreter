---
name: accuracy-check
description: Measure transcription/translation speed and accuracy with the simulator and compare with the previous run. Use before and after a change to the audio, Whisper or translation code, or when Benn asks "did that help?".
---
# Accuracy check

1. Ask which change is being tested, and note the current `git rev-parse --short HEAD` plus whether the tree is dirty.
2. Run a short simulation (adjust the limit if Benn wants a fuller run):
   ```powershell
   .venv\Scripts\python simulate.py --data synth --limit 20 --offline
   ```
3. Run the reports on the period the simulation covered:
   ```powershell
   .venv\Scripts\python log_tools.py accuracy <start-time>
   .venv\Scripts\python log_tools.py report <start-time>
   .venv\Scripts\python log_tools.py trend
   ```
4. Present a small before/after table: word error rate, median and 95th-percentile latency, dropped or failed phrases.
5. Append the results with the commit, date and change description to `docs/ACCURACY_LOG.md` (create it if missing).
6. Say plainly whether the change helped, hurt or made no measurable difference. Small runs are noisy; say so when the difference is within the noise.

# Learn the project by following one phrase

This guide is a practical route through the code. You do not need to read the
whole project first. Start with one sound block, follow the messages it creates,
and use the logs and simulator to check your understanding.

## The main idea: data moves through stages

The app turns a continuous stream of audio into small, labelled pieces of work:

```text
audio samples
    -> speech decisions and phrase boundaries
    -> preview or finished-phrase message
    -> recognised words and language
    -> translation
    -> window update and log event
```

Each stage has a clear input and output. When learning a function, write down
those two things first. Then ask what state it reads or changes, and what it
does when the input is empty, late, or invalid.

## A guided code tour

Open `interpreter.py` and use **Ctrl+Shift+O** to jump to these names:

1. **`AudioCaptureThread`** — find `run`. Follow how audio blocks are read,
   scored by voice activity detection, accumulated into a phrase, and sent for
   preview or final transcription. The nearby constants explain timing choices.
2. **`LatestSlot` and the queues** — compare the preview slot with
   `phrase_queue`. A preview is replaceable: a newer one makes an old pending
   preview stale. A finished phrase is important work and goes through a queue.
   This distinction prevents slow previews from building a backlog ahead of
   final results.
3. **`TranscriptionWorker`** — follow its worker loop and the method that
   handles a phrase. Look for the order of optional vocal separation,
   transcription, language selection, filtering/retry, and local translation.
   If local translation is unavailable, the result can be sent to the window
   for the online fallback.
4. **`InterpreterApp`** — find `_poll_queue` (or jump to the result queue
   name). The window checks for messages and updates widgets on its own thread.
   Background workers send messages; they do not draw into the window.
5. **`log_event` and `phrase_trace`** — inspect what facts are recorded about a
   phrase. A trace helps answer *why* a result happened, not just what appeared.

Names and exact method boundaries can evolve. If a name has moved, search for
the queue or event name shown in the surrounding code and follow where it is
created and consumed.

## Make a prediction before running code

Pick one setting in [TUNING.md](TUNING.md), such as `SILENCE_SEC`. Before
changing it, predict the effect in plain language. For example: “A longer
silence threshold should join more words into one phrase, but final text may
arrive later.” Then:

1. Run the same small simulator selection before and after the change.
2. Compare phrase boundaries, word error rate, and pause-to-text time.
3. Read the phrase trace for a clip where your prediction did not hold.
4. Restore the value if the result is worse, or keep the change and update its
   nearby explanation if it is useful.

The simulator and log commands are in [HOW_TO.md](HOW_TO.md). Keep the model,
clips, and other settings the same during a comparison; otherwise you cannot
tell which change caused the difference. Simulated timings are comparisons on
the same machine, not a promise about every live audio device.

## Learn from an individual result

For one phrase, compare these three views:

| View | What it teaches |
| --- | --- |
| Simulator result | What the app recognised and how it scored against known text |
| `phrase_trace` log entry | Which boundaries, language decisions, retries, and timings led to the result |
| Audio clip | What the models actually received, including pauses, music, and clipping |

If the simulator's reference text is known, word error rate (WER) measures
inserted, missing, and substituted words. A low WER alone does not explain a
mistake, so compare the recognised text with the audio and trace as well. The
larger-model check in `log_tools.py accuracy` is an estimate for real recordings,
not ground truth; its reference is another model's transcription.

## A few concepts that make the code easier to read

- **A thread** is an independent path of work. The capture thread must keep
  reading audio while the GPU is transcribing older audio.
- **A queue** is a thread-safe hand-off. The sender puts a message in; the
  receiver takes it later. Search both the `.put(...)` and `.get...(...)` sides
  to understand a message's path.
- **A callback** is code supplied to another part to be called later. For
  example, workers use callbacks to report status or results without owning
  the window.
- **A constant** gives a setting a name. If a timing appears as a constant,
  follow its uses before changing it; similar-looking constants may apply only
  to music, previews, or long phrases.
- **A trace/log event** is evidence from a run. Prefer it to guessing about
  behaviour from a setting's name.

## Good next stops

- [MAP.md](MAP.md) for the system diagrams.
- [FILES.md](FILES.md) for module responsibilities and where to continue.
- [GLOSSARY.md](GLOSSARY.md) for audio, model, and measurement terms.
- [HOW_TO.md](HOW_TO.md) for repeatable run and report commands.

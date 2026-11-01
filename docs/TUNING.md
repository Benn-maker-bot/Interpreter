# The tuning numbers

These values (in CAPITALS, near the top of `interpreter.py`) control how the app behaves.
Search for the name with **Ctrl+F** to find it.
Change **one** at a time, and test before and after ([HOW_TO.md](HOW_TO.md) → "Check a change").

---

## When does a phrase end?

| Name | Value | What it does |
| --- | --- | --- |
| `SILENCE_SEC` | 0.3 s | A pause this long ends a spoken phrase. |
| `SPEECH_SOFT_PHRASE_SEC` | 7 s | After this long… |
| `SPEECH_SOFT_SILENCE_SEC` | 0.2 s | …a shorter pause is enough (fewer forced cuts). |
| `MAX_PHRASE_SEC` | 10 s | Longest spoken phrase; then a forced cut. |
| `CUT_SEARCH_SEC` | 2.5 s | A forced cut happens at the quietest moment in the last 2.5 s. |
| `MIN_PHRASE_SEC` | 0.5 s | Shorter sounds are ignored (they make Whisper invent text). |
| `PREROLL_SEC` | 0.6 s | Audio kept from just before speech started, so the first word isn't lost. |

## Music and singing

| Name | Value | What it does |
| --- | --- | --- |
| `MUSIC_SILENCE_SEC` | 0.9 s | During music, the pause that ends a phrase. |
| `MUSIC_SOFT_PHRASE_SEC` | 6 s | After this long… |
| `MUSIC_SOFT_SILENCE_SEC` | 0.4 s | …a breath between lines is enough. |
| `MUSIC_MAX_PHRASE_SEC` | 15 s | Longest sung phrase. |
| `MUSIC_VAD_DROP` | 0.15 | Speech detection is this much more sensitive during music. |
| `VOICE_GAP_DB` | 12 dB | Decides if there is singing or only instruments. |

## Speed

| Name | Value | What it does |
| --- | --- | --- |
| `PARTIAL_INTERVAL_SEC` | 0.2 s | How often the grey live preview updates. |
| `EARLY_FINAL_SEC` | 0.1 s | Start transcribing this soon into a pause (saves ~0.3 s). |

## Speech detection and loudness

| Name | Value | What it does |
| --- | --- | --- |
| `DEFAULT_SENS` | 0.5 | Speech sensitivity when nothing is saved (the slider in the window). |
| `AGC_TARGET_DBFS` | -20 dBFS | Quiet speech is made louder to this level before Whisper. |

## Throwing away bad text

| Name | Value | What it does |
| --- | --- | --- |
| `NO_SPEECH_PROB` | 0.6 | With `LOGPROB_FLOOR`: when Whisper thinks there was no speech. |
| `LOGPROB_FLOOR` | -1.0 | Below this confidence, a result is suspicious. |
| `CONTEXT_RESET_SEC` | 6 s | After a silence this long, earlier sentences are no longer used as context. |

## Languages

| Name | Value | What it does |
| --- | --- | --- |
| `AUTO_LANG_PROB` | 0.50 | Auto-detect: below this certainty on a short phrase, keep the last language. |
| `NEW_LANG_PROB` | 0.85 | A language not heard recently needs this certainty. |
| `NEW_LANG_MUSIC_PROB` | 0.90 | The same, during music. |
| `LANG_SWITCH_PROB` | 0.70 | Fixed source language: certainty needed to switch to another one. |
| `LANG_MIN_SPEECH_SEC` | 1.5 s | …and at least this much speech. |

---

## Where the values came from

Most of them were **measured**: the comment next to each value in the code says how.
For example, `NEW_LANG_PROB` came from a live test: real language changes scored
0.95–0.99, wrong ones on short sung fragments 0.31–0.80.

# Glossary

Words used in this project, in plain language.
(The opening comment of `interpreter.py` has more, about Python and audio.)

---

## Sound

| Word | Meaning |
| --- | --- |
| **dBFS** | Loudness of digital sound. 0 is the loudest possible; speech is usually -30 to -12. |
| **Sample rate** | Sound measurements per second. Sound cards: 48000; Whisper: 16000. |
| **Loopback** | Recording what an output device (your FiiO) is playing. |
| **Phrase** | One piece of speech between pauses. The app's unit of work. |
| **Pause** | Quiet time after speech. A long enough pause ends a phrase. |
| **Forced cut** | Ending a phrase because it got too long, not because of a pause. |
| **Gain / levelling** | Making audio louder or quieter. The simulator levels each clip to -20 dBFS. |

## Recognising speech

| Word | Meaning |
| --- | --- |
| **Whisper** | The AI model that turns speech into text. "Turbo" is the version used. |
| **VAD** | Voice Activity Detection: scoring each moment of sound for "is this speech?". |
| **Preview** | The grey text shown while someone is still talking. It may still change. |
| **Final** | The finished text of a phrase, after it ended. |
| **Early final** | Starting the final transcription as soon as a pause begins, to save time. |
| **Hallucination** | Text Whisper invents that was never said ("Thank you.", subtitle credits). |
| **Retry** | Transcribing a phrase a second time because the first result looked wrong. |
| **Context** | Earlier sentences given to Whisper so it understands the next one better. |
| **Language detection** | Whisper guessing which language a phrase is in. |

## Music

| Word | Meaning |
| --- | --- |
| **Music mode** | Settings for songs: longer phrases, more sensitive, instruments removed. |
| **YAMNet** | A small AI model that recognises sounds (speech, music, singing). |
| **Vocal separation (UVR)** | An AI model that removes instruments so only the voice is left. |

## Translating

| Word | Meaning |
| --- | --- |
| **Opus-MT** | Small translation models that run on your PC, one per language pair. |
| **Online translation** | Translation via the internet, used when no local model exists. |

## Testing and measuring

| Word | Meaning |
| --- | --- |
| **Simulator** | Plays test recordings through the real app instead of a sound card. |
| **Synthetic clips** | Test clips made with computer voices: one language each, exact text. |
| **WER** | Word Error Rate: % of words wrong, missing or extra. Lower is better. |
| **Latency / pause → text** | How long after someone stops talking the text appears. |
| **Median (p50)** | The middle value: half are faster, half slower. |
| **p90** | 90 % of cases are faster than this. Shows the slow cases. |
| **A/B test** | Running the same test with and without a change, then comparing. |
| **Trend** | Figures per day, to see if things get better over time. |

## Computer

| Word | Meaning |
| --- | --- |
| **GPU** | The graphics card. Whisper runs on it (much faster than the processor). |
| **CUDA** | NVIDIA's way of running programs on the graphics card. |
| **Thread** | Part of a program that runs at the same time as the rest. |
| **Queue** | A mailbox between threads: one puts work in, another takes it out. |
| **.venv** | The project's own Python, with exactly the packages this project needs. |
| **Commit** | A saved snapshot of the code in git. You can always go back to it. |

# Map of the program

Pictures of how the parts fit together.
To see the diagrams: open this file and press **Ctrl+Shift+V**
(needs the "Markdown Preview Mermaid Support" extension, see [VSCODE.md](VSCODE.md)).

---

## 1. The big picture

```mermaid
flowchart LR
    A[Sound<br/>FiiO headphones<br/>or microphone] --> B[AudioCaptureThread<br/>records, finds speech,<br/>cuts it into phrases]
    B -- phrase audio --> C[TranscriptionWorker<br/>Whisper: audio → text<br/>Opus-MT: translation]
    C -- text + translation --> D[Window<br/>shows text,<br/>can speak it]
    D --> E[(Log files<br/>interpreter.log.jsonl)]
    C --> E
    B --> E
```

- **Left to right** is the path of the sound.
- Each box is a **thread**: it runs at the same time as the others.
- They pass work along through **queues** (like mailboxes).

---

## 2. A phrase's journey

What happens to one sentence, step by step.

```mermaid
sequenceDiagram
    participant S as Sound
    participant C as AudioCaptureThread
    participant W as TranscriptionWorker
    participant UI as Window
    S->>C: sound, every 20-30 ms
    C->>C: Silero VAD: is this speech?
    loop while the person talks (every 0.2 s)
        C->>W: unfinished phrase (preview)
        W->>UI: grey preview text
    end
    C->>W: pause starts (0.1 s): early transcription
    C->>W: pause lasts (0.3 s): phrase finished
    W->>W: Whisper (uses the early result if ready)
    W->>W: translate (Opus-MT)
    W->>UI: final text + translation
    UI->>UI: replace grey preview, log timings
```

---

## 3. Where a phrase ends

```mermaid
flowchart TD
    A[New 20-30 ms block] --> B{Speech?}
    B -- yes --> A
    B -- no: pause grows --> C{Music playing?}
    C -- no --> D{Phrase longer than 7 s?}
    D -- no --> E[End after 0.3 s pause]
    D -- yes --> F[End after 0.2 s pause]
    C -- yes --> G{Phrase longer than 6 s?}
    G -- no --> H[End after 0.9 s pause]
    G -- yes --> I[End after 0.4 s pause]
    A --> J{Phrase too long?<br/>10 s speech / 15 s music}
    J -- yes --> K[Forced cut at the quietest<br/>moment of the last 2.5 s]
```

The log line of every phrase says which rule ended it (`end_rule`).

---

## 4. Choosing the language (Auto-detect)

```mermaid
flowchart TD
    A[Whisper guesses the language<br/>with a certainty 0-1] --> B{Certain enough?<br/>≥ 0.5, or ≥ 3 s of speech}
    B -- no --> C{A recently heard language<br/>is possible?}
    C -- yes --> D[Use that language]
    C -- no --> E[Keep the current language,<br/>unless it is very unlikely]
    B -- yes --> F{Is it a language<br/>heard recently?}
    F -- yes --> G[Use it]
    F -- no --> H{Certainty ≥ 0.85<br/>≥ 0.90 during music}
    H -- yes --> G
    H -- no --> I[Keep the current language]
```

The log line says why (`lang_reason`).

---

## 5. Music mode

```mermaid
flowchart LR
    A[Every second:<br/>YAMNet sound classifier] --> B{Music?}
    B -- yes --> C[Music mode on]
    C --> D[Longer pauses end a phrase]
    C --> E[UVR vocal separation:<br/>remove instruments,<br/>keep the voice]
    E --> F[Whisper hears only the voice]
    B -- no --> G[Normal speech mode]
```

The voice is separated in 6-second chunks **while the singing is still going**, so little is left to do when the phrase ends.

---

## 6. The threads

```mermaid
flowchart TB
    subgraph Main thread
        UI[Window: draws everything,<br/>checks the result queue every 15 ms]
    end
    subgraph Background threads
        CAP[audio-capture]
        TW[transcription-worker]
        TR[translation-worker<br/>online fallback]
        WL[whisper-loader / mt-loader<br/>load AI models]
        TTS[tts: speaks text]
    end
    CAP -- phrase_queue --> TW
    TW -- result_queue --> UI
    UI -- translation queue --> TR
    TR -- result_queue --> UI
    UI --> TTS
```

**Rule:** only the main thread may touch the window. Everything else sends messages.

---

## 7. The simulator

```mermaid
flowchart LR
    A[Test data<br/>synthetic, recorded,<br/>your reported phrases,<br/>or live Windows audio] --> B[Fake sound card<br/>FakePyAudio]
    B --> C[The REAL pipeline<br/>capture + worker]
    C --> D[Scoring<br/>language right? word errors?<br/>how fast?]
    D --> E[Results table +<br/>testdata/sim_results/*.json]
    A --> F[Your FiiO<br/>so you can listen]
```

The simulator uses the **same code** as the app; only the sound card is replaced.

---

## 8. Which file uses which

```mermaid
flowchart TB
    IQ[interpreter_qt.py<br/>Fluent window] --> I[interpreter.py<br/>all the logic]
    I --> M[music.py<br/>music detection,<br/>vocal separation]
    SQ[simulator_qt.py<br/>simulator window] --> S[simulate.py<br/>simulator engine]
    SQ --> IQ
    S --> I
    S --> LT[log_tools.py<br/>log analysis]
    MT[make_testdata.py] --> FT[fetch_testdata.py]
```

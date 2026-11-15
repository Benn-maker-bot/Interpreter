# Map of the program

Quick overview diagrams. For more detailed pictures of module responsibilities,
thread hand-offs, phrase lifecycle, interface reuse, and testing, see
[FLOWCHARTS.md](FLOWCHARTS.md). To view Mermaid diagrams, open a Markdown file
and press **Ctrl+Shift+V** (see [VSCODE.md](VSCODE.md)).

---

## 1. The big picture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart LR
    A[Sound<br/>FiiO headphones<br/>or microphone] --> B[AudioCaptureThread<br/>records, finds speech,<br/>cuts it into phrases]
    B -- phrase audio --> C[TranscriptionWorker<br/>Whisper: audio → text<br/>Opus-MT: translation]
    C -- text + translation --> D[Window<br/>shows text,<br/>can speak it]
    D --> E[(Log files<br/>interpreter.log.jsonl)]
    C --> E
    B --> E
```

- **Left to right** is the path of the sound.
- The capture and transcription boxes are background threads; the window runs
  on the main thread. Sound and logs are data sources and storage, not threads.
- The Fluent window runs this path **twice at once**, one per channel (see
  [Two channels](#1a-two-channels-fluent-window)). The classic window runs it
  once, for the device of the selected mode.

## 1a. Two channels (Fluent window)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart LR
    M[Channel 1 'Me'<br/>microphone<br/>English → Russian] --> C1[SessionController 1<br/>capture → worker<br/>→ translation]
    F[Channel 2 'Friend'<br/>call audio 🔊 loopback<br/>Russian → English] --> C2[SessionController 2<br/>capture → worker<br/>→ translation]
    W[(One Whisper model<br/>on the GPU<br/>INFERENCE_LOCK)] -.- C1
    W -.- C2
    C1 -- "TaggedQueue<br/>(1, run)" --> Q[result_queue]
    C2 -- "TaggedQueue<br/>(2, run)" --> Q
    Q --> UI[Window: one conversation,<br/>a strip per channel]
```

- Each channel has its own device, languages, sensitivity, Mute, Speak and
  output device. The model and preview model are shared.
- Every message from a channel arrives labelled `(channel, run id)`. The run id
  changes when a channel restarts, so late messages from a stopped pipeline are
  recognised and ignored.
- PortAudio may only be started or stopped by one thread at a time
  (`portaudio_start()` / `portaudio_stop()`, `PORTAUDIO_LOCK`): with two
  channels, overlapping starts crashed the app.
- Work passes between threads through **queues** (like mailboxes). The detailed
  [message flow](FLOWCHARTS.md#2-message-flow-threads-and-hand-offs) shows which
  messages use a replaceable preview slot and which use queues.

---

## 2. A phrase's journey

What happens to one sentence, step by step.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "sequence": {"actorFontSize": 18, "messageFontSize": 18, "noteFontSize": 16, "actorMargin": 48, "messageMargin": 32}}}%%
sequenceDiagram
    participant S as Sound
    participant C as AudioCaptureThread
    participant W as TranscriptionWorker
    participant UI as Window
    S->>C: audio blocks; VAD scores 32 ms frames at 16 kHz
    C->>C: Silero VAD: is this speech?
    loop while the person talks (every 0.2 s)
        C->>W: unfinished phrase (preview)
        W->>UI: grey preview text
    end
    C->>W: pause starts (0.1 s): offer early audio in a replaceable slot
    C->>W: pause lasts (0.3 s): finished phrase enters phrase_queue
    W->>W: Whisper (reuse early result if still valid, otherwise transcribe)
    W->>W: translate (Opus-MT)
    W->>UI: final text + translation
    UI->>UI: replace grey preview, log timings
```

---

## 3. Where a phrase ends

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
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
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
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
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
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
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart TB
    subgraph Main thread
        UI[Window: draws everything,<br/>checks the result queue every 15 ms]
    end
    subgraph Background threads, one set per channel
        CAP[audio-capture-Me / -Friend]
        TW[transcription-worker-Me / -Friend]
        TR[translation-worker<br/>online fallback]
        TTS[tts: speaks text]
    end
    WL[whisper-loader / mt-loader<br/>load AI models, shared]
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
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
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
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart TB
    IQ[interpreter_qt.py<br/>Fluent window] --> SC[session_controller.py<br/>shared worker lifecycle]
    CI[interpreter.py<br/>classic window] --> SC
    SC --> I[interpreter.py<br/>audio and model services]
    I --> M[music.py<br/>music detection,<br/>vocal separation]
    SQ[simulator_qt.py<br/>simulator window] --> S[simulate.py<br/>simulator engine]
    SQ --> IQ
    S --> I
    S --> LT[log_tools.py<br/>log analysis]
    MT[make_testdata.py] --> FT[fetch_testdata.py]
```

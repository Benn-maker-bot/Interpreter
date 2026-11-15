# Visual guide to the project

These diagrams answer different questions, so start with the one you need:

| Question | Diagram |
| --- | --- |
| Which part owns each responsibility? | Project structure |
| How do background threads communicate? | Message flow |
| How does audio become a phrase? | Phrase lifecycle |
| How does the Qt window reuse the app logic? | Two interfaces, shared core |
| How does the simulator test the real pipeline? | Test design |

In VS Code, open this file and press **Ctrl+Shift+V**. Mermaid diagrams render
with the recommended Markdown Mermaid extension ([VSCODE.md](VSCODE.md)). The
source is plain text between the `mermaid` fences, so it remains readable in
any editor.

---

## 1. Project structure: who owns what?

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart LR
    subgraph interfaces["User interfaces"]
        direction TB
        QI["interpreter_qt.py<br/>Fluent interpreter window"]
        CI["interpreter.py<br/>classic interpreter window"]
        SQ["simulator_qt.py<br/>simulator window"]
    end
    subgraph application["Shared application and analysis"]
        direction TB
        CORE["interpreter.py<br/>audio capture, workers, models, settings, logs"]
        SESSION["session_controller.py<br/>session configuration and worker lifecycle"]
        MUSIC["music.py<br/>music detection and vocal separation"]
        SIM["simulate.py<br/>test playback and scoring"]
        LOG["log_tools.py<br/>read and analyse logs"]
    end
    subgraph preparation["Test data preparation"]
        direction TB
        SYN["make_testdata.py<br/>generate speech and music clips"]
        REAL["fetch_testdata.py<br/>download recorded clips"]
        DATA[("testdata manifests and audio")]
    end
    QI -->|settings snapshot and callbacks| SESSION
    CI -->|settings snapshot and callbacks| SESSION
    SESSION --> CORE
    CORE --> MUSIC
    SQ --> SIM
    SIM -->|uses real capture and transcription| CORE
    SIM --> LOG
    SYN --> DATA
    REAL --> DATA
    SIM --> DATA
    classDef interface fill:#dceeff,stroke:#2878a8,stroke-width:2px,color:#142d3d
    classDef core fill:#e8e2ff,stroke:#6750a4,stroke-width:2px,color:#211b36
    classDef data fill:#e1f3e8,stroke:#3a8554,stroke-width:2px,color:#163322
    class QI,CI,SQ interface
    class CORE,MUSIC,SIM,LOG core
    class SYN,REAL,DATA data
```

**How to read it:** an arrow means “uses” or “hands work to,” as labelled.
The two interpreter windows share the same application logic. The simulator
also uses that logic, with test audio supplied in place of a physical input.

**Code trail:** start at `QtInterpreterApp` or `InterpreterApp`, follow its
settings snapshot into `SessionController.start()` in `session_controller.py`,
then follow the worker constructors in `interpreter.py`. For tests, start at
`run()` in `simulate.py` and follow `build()`.

---

## 2. Message flow: threads and hand-offs

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart LR
    subgraph Audio thread
        CAP[AudioCaptureThread<br/>reads blocks and detects speech]
    end
    subgraph Model worker
        WORK[TranscriptionWorker<br/>transcribes and translates locally]
    end
    subgraph Window thread
        UI[Interpreter window<br/>updates widgets and starts speech]
        ONLINE[TranslationWorker<br/>online translation fallback]
    end
    CAP -->|LatestSlot: newest preview only| WORK
    CAP -->|phrase_queue: every finished phrase| WORK
    CAP -->|status and audio levels| UI
    WORK -->|result_queue: preview, final, status| UI
    UI -->|translation queue when needed| ONLINE
    ONLINE -->|translated result| UI
    UI -->|speak requested text| TTS[Speaker thread]
    CAP -.->|logs events| LOG[(log files)]
    WORK -.->|logs decisions and timings| LOG
    UI -.->|logs display and issues| LOG
    classDef thread fill:#dceeff,stroke:#2878a8,color:#142d3d
    classDef main fill:#e8e2ff,stroke:#6750a4,color:#211b36
    classDef store fill:#e1f3e8,stroke:#3a8554,color:#163322
    class CAP,WORK,ONLINE,TTS thread
    class UI main
    class LOG store
```

The **window thread owns the widgets**. Background workers send messages for
the window to handle. A `LatestSlot` is a one-item mailbox: a newer preview
replaces a stale pending preview. The `phrase_queue` keeps finished phrases,
because each one must still be processed. The window polls its result queue and
updates the display on the UI thread.

**Code trail:** search for `LatestSlot`, `phrase_queue`, `result_queue`, and
`_poll_queue` in `interpreter.py`. Follow each message from its `.put(...)` to
the matching `.get...(...)`.

---

## 3. Phrase lifecycle: audio blocks to final text

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart TD
    A[Read audio block] --> B[Measure level and score speech with VAD]
    B --> C{Speech threshold reached?}
    C -->|No| A
    C -->|Yes| D[Start or continue phrase<br/>include pre-roll audio]
    D --> E{Speech in this block?}
    E -->|Yes| P[Reset pause counter;<br/>offer preview when due]
    P --> D
    E -->|No| G[Count another quiet block]
    G --> H{Enough pause to end phrase?}
    H -->|Yes| I[Send finished phrase in phrase_queue]
    H -->|No| F{Early-final threshold reached?}
    F -->|Yes| K[Offer current audio in early_slot]
    F -->|Not yet| D
    K --> D
    D --> J{Maximum phrase length reached?}
    J -->|Yes| CUT[Cut near a quiet point and carry remainder forward]
    J -->|No| D
    CUT --> I
    I --> ASR[Worker uses a valid early result or runs Whisper]
    ASR --> M[Choose language, filter or retry if needed]
    M --> N{Local translation ready?}
    N -->|Yes| O[Translate locally]
    N -->|No or unavailable| R[Ask window for online fallback]
    O --> S[Send final result to window]
    R --> S
    S --> T[Replace live preview, display text, record trace]
    classDef audio fill:#dceeff,stroke:#2878a8,color:#142d3d
    classDef decision fill:#fff1ce,stroke:#ad7612,color:#35270b
    classDef model fill:#e8e2ff,stroke:#6750a4,color:#211b36
    classDef finish fill:#e1f3e8,stroke:#3a8554,color:#163322
    class A,B,D,P,G,I,K,CUT audio
    class C,E,F,H,J,N decision
    class ASR,M,O,R model
    class S,T finish
```

Music mode changes the pause and maximum-length rules and can separate the
voice from instruments before recognition. The early slot is speculative: if
speech resumes, that early audio no longer represents the finished phrase.
The final queue remains the authoritative hand-off for a completed phrase.

**Code trail:** `AudioCaptureThread.run()` handles blocks and boundaries;
`_flush()` sends a finished phrase; `TranscriptionWorker.run()` consumes work.
Search for `_detect_auto()` to study language selection.

---

## 4. Two interfaces, one shared core

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart LR
    subgraph Classic interface
        TK[InterpreterApp<br/>customtkinter widgets + app logic]
    end
    subgraph Fluent interface
        QT[QtInterpreterApp<br/>PySide6 widgets]
        ADAPT[QVar, ComboShim,<br/>TextShim, after timer]
    end
    SESSION[SessionController<br/>config, queues,<br/>worker lifecycle]
    SHARED[Shared callbacks<br/>watchdog, queue polling,<br/>logging, model services]
    TK --> SESSION
    TK --> SHARED
    QT --> ADAPT
    QT --> SESSION
    ADAPT --> SHARED
    SESSION --> CAP[AudioCaptureThread]
    SESSION --> WORK[TranscriptionWorker]
    classDef interface fill:#dceeff,stroke:#2878a8,color:#142d3d
    classDef shared fill:#e8e2ff,stroke:#6750a4,color:#211b36
    class TK,QT,ADAPT interface
    class SESSION,SHARED,CAP,WORK shared
```

Both interfaces pass a settings snapshot and UI callbacks to the controller.
The controller owns the worker queues and runtime threads; the views own their
widgets and display updates. A few older callbacks remain shared during the
ongoing UI separation.

**Code trail:** read the module guide at the top of `interpreter_qt.py`, then
read `SessionController` in `session_controller.py`, then follow the view's
callback bindings in each UI class.

---

## 5. Test design: repeatable input, real processing

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Segoe UI, Arial, sans-serif", "lineColor": "#536273", "primaryTextColor": "#17212b"}, "flowchart": {"nodeSpacing": 52, "rankSpacing": 64, "htmlLabels": true, "curve": "basis"}}}%%
flowchart LR
    SYN[Generated clips] --> PLAY[Playlist and fake audio device]
    REC[Recorded clips] --> PLAY
    FB[Reported mistake clips] --> PLAY
    PLAY --> REALPIPE[Real capture and transcription pipeline]
    REALPIPE --> RESULTS[Recognised phrases and trace logs]
    RESULTS --> SCORE[Match phrase times to clips<br/>compare words and language]
    SCORE --> TABLE[Per-clip and per-language results]
    LIVE[Live Windows audio] --> LIVEPIPE[Real audio device and same pipeline]
    LIVEPIPE --> NOGT[Measure language and speed<br/>no known-text word score]
    classDef input fill:#dceeff,stroke:#2878a8,color:#142d3d
    classDef process fill:#e8e2ff,stroke:#6750a4,color:#211b36
    classDef output fill:#e1f3e8,stroke:#3a8554,color:#163322
    class SYN,REC,FB,PLAY,LIVE input
    class REALPIPE,SCORE,LIVEPIPE process
    class RESULTS,TABLE,NOGT output
```

Known-text clips let the simulator measure word error rate. Live audio has no
reference transcript, so it can report timing and detected language, but not
whether every word was correct. This is why a controlled before-and-after run
is more informative than judging one result in isolation.

**Code trail:** `build()` connects the real workers to a fake or live audio
source; `score()` associates phrase timings with clips; `log_tools.py` analyses
the resulting traces.

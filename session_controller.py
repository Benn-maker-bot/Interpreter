"""Own the shared audio/model worker lifecycle for either desktop UI.

The view supplies a settings snapshot and callbacks; this controller builds
and stops the capture, transcription, and translation pipeline without
creating or updating widgets.

One SessionController runs ONE pipeline: one audio device -> one capture
thread -> one transcription worker -> translation. The Fluent window listens
to two devices at once, so it owns two controllers ("channels"). Both
send their results into the same window queue; a TaggedQueue labels each
message with the channel (and the run) it came from, so the window knows
whose words they are.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable

if TYPE_CHECKING:
    import interpreter as core


@dataclass(frozen=True)
class SessionConfig:
    """Immutable settings captured by one listening session.

    channel  name of the channel ("Me", "Friend") for thread names and logs;
             "" for the single-device classic window.
    tag      any label (e.g. (channel number, run id)). When set, every
             message this session sends to the window arrives as
             (tag, (kind, data)) instead of (kind, data).
    """

    session_id: str
    run_id: int
    device_label: str
    device_index: int | None
    output_label: str
    output_index: int | None
    loopback: bool
    src_key: str
    tgt_key: str
    model_label: str
    sensitivity: float
    preview_choice: str
    preview_enabled: bool
    previews_off: bool
    tts_on: bool
    record_audio: bool
    auto_language: bool
    mode_state: tuple[tuple[str, ...], ...]
    channel: str = ""
    tag: Hashable | None = None


class TaggedQueue:
    """Looks like a queue to the background threads, but labels everything
    they put in it before passing it on to the window's shared queue.

    Example: with tag (1, 7), a worker's put(("final", data)) arrives in the
    shared queue as ((1, 7), ("final", data)): "channel 1, run 7".

    The threads only ever call put() and qsize(), so those are all it needs.
    (This is the ADAPTER pattern: an object that offers the interface the
    callers expect and translates calls to the real object behind it.)
    """

    def __init__(self, target: queue.Queue, tag: Hashable) -> None:
        self.target, self.tag = target, tag

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        self.target.put((self.tag, item), block, timeout)

    def qsize(self) -> int:
        return self.target.qsize()


class AnyEvent:
    """Acts like a threading.Event that is set while ANY of several events
    is set. The capture thread checks mute_evt.is_set() to ignore our own
    text-to-speech; with one speaker per channel, either may be audible on
    the device being captured."""

    def __init__(self, events: Iterable[threading.Event]) -> None:
        self.events = list(events)

    def is_set(self) -> bool:
        return any(event.is_set() for event in self.events)


class SessionController:
    """Manage the threads and queues for one shared interpreter session.

    UI references can read these attributes for diagnostics, but only this
    class creates, replaces, and stops the runtime workers.
    """

    def __init__(self, result_queue: queue.Queue, core_module: Any) -> None:
        self.core = core_module
        self.result_queue = result_queue
        self.phrase_queue: queue.Queue = queue.Queue()
        self.translation_queue: queue.Queue = queue.Queue(
            maxsize=self.core.TRANSLATION_QUEUE_MAX)
        self.capture_thread: Any = None
        self.loopback_out: Any = None
        self.workers: list[Any] = []
        self.translation_workers: list[Any] = []
        self.partial_slot = None
        self.early_slot = None
        self.stopping_capture: Any = None

    def start(
        self,
        config: SessionConfig,
        *,
        model: Any,
        model_loader: core.WhisperModelLoader | None,
        speaker: core.Speaker | None,
        on_level: Callable[..., None] | None,
        preview_model: Callable[[], Any],
        translator_for: Callable[..., Any],
        music_scene: Any,
        mute_event: Any = None,
    ) -> None:
        """Start the pipeline from a UI settings snapshot and callbacks.

        speaker     the classic window's single Speaker: capture is muted
                    while it speaks on the captured device. None if the
                    caller passes `mute_event` itself instead.
        on_level    receives each audio block's level; None = send it to
                    the window as a ("level", (...)) message.
        mute_event  anything with is_set(); while set, the captured audio
                    is replaced by silence (our own speech).
        """
        core = self.core
        self.stop_workers()
        previous = self.stopping_capture
        if previous is not None and previous.is_alive():
            previous.join(timeout=1.0)
        self.stopping_capture = None

        # Where this session's messages go: straight to the window's queue,
        # or through a TaggedQueue that labels them with the channel.
        out: Any = (TaggedQueue(self.result_queue, config.tag) if config.tag is not None
                    else self.result_queue)
        if on_level is None:
            def on_level(*levels: Any) -> None:
                out.put(("level", levels))

        def status(message: str) -> None:
            out.put(("status", message))

        worker_count = getattr(model_loader, "worker_count", core.NUM_WORKERS)
        self.phrase_queue = queue.Queue(maxsize=max(worker_count * 16, 32))
        self.translation_queue = queue.Queue(maxsize=core.TRANSLATION_QUEUE_MAX)
        self.partial_slot = None if config.previews_off else core.LatestSlot()
        self.early_slot = core.LatestSlot()

        src_whisper, _ = core.LANGUAGES.get(config.src_key, ("en", "en"))
        tgt_lang = core.LANGUAGES.get(config.tgt_key, ("ru", "ru"))[0]
        session_config = {
            "model": core.MODELS.get(config.model_label),
            "compute_type": getattr(model_loader, "active_compute_type", core.COMPUTE_TYPE),
            "device": getattr(model_loader, "active_device", core.WHISPER_DEVICE),
            "preview_choice": config.preview_choice,
            "sensitivity": round(config.sensitivity, 2),
            "src_lang": config.src_key,
            "tgt_lang": config.tgt_key,
            "speak": config.tts_on,
        }

        if (mute_event is None and speaker is not None
                and "[loopback]" in config.device_label.lower()
                and core.plain_device_name(config.device_label)
                == core.plain_device_name(config.output_label)):
            mute_event = speaker.speaking

        pa = core.portaudio_start() if core._PYAUDIO_AVAILABLE else None
        native_rate = core.get_device_native_rate(pa, config.device_index) if pa else core.SAMPLE_RATE
        if pa:
            core.portaudio_stop(pa)

        self.loopback_out = None
        if config.loopback and "[loopback]" in config.device_label.lower():
            core.log_event(core.logging.INFO, "loopback_skipped_output_capture",
                           channel=config.channel, device_label=config.device_label)
        elif config.loopback:
            self.loopback_out = core.LoopbackOutputThread(
                output_device_index=config.output_index,
                input_sample_rate=native_rate,
                status_cb=status,
            )
            self.loopback_out.start()
            core.log_event(core.logging.INFO, "loopback_started", channel=config.channel,
                           output_label=config.output_label, output_index=config.output_index,
                           input_native_rate=native_rate)

        self.capture_thread = core.AudioCaptureThread(
            device_index=config.device_index,
            phrase_queue=self.phrase_queue,
            status_cb=status,
            level_cb=on_level,
            loopback_out=self.loopback_out,
            partial_slot=self.partial_slot,
            mute_evt=mute_event,
            early_slot=self.early_slot,
            channel=config.channel,
        )
        self.capture_thread.set_sensitivity(round(config.sensitivity, 2))
        if (not config.preview_enabled and not config.previews_off
                and core.MODELS.get(config.model_label) in ("large-v3-turbo", "medium")):
            self.capture_thread.preview_interval_sec = core.MAIN_MODEL_PREVIEW_INTERVAL_SEC
        self.capture_thread.music_scene = music_scene
        self.capture_thread.start()

        self.translation_workers = []
        for worker_id in range(core.TRANSLATION_WORKERS):
            worker = core.TranslationWorker(worker_id, self.translation_queue, out)
            worker.start()
            self.translation_workers.append(worker)

        self.workers = []
        for worker_id in range(worker_count):
            worker = core.TranscriptionWorker(
                wid=worker_id,
                model=model,
                src_whisper=src_whisper,
                src_lang_key=config.src_key,
                tgt_lang_key=config.tgt_key,
                phrase_queue=self.phrase_queue,
                result_queue=out,
                status_cb=status,
                partial_slot=self.partial_slot,
                preview_model=preview_model,
                run_id=config.run_id,
                record_audio=config.record_audio,
                translator=translator_for(src_whisper, tgt_lang),
                early_slot=self.early_slot,
                session_id=config.session_id,
                config=session_config,
                channel=config.channel,
            )
            worker.music_scene = music_scene
            worker.auto_language = config.auto_language
            worker.translator_for = translator_for
            for mode in config.mode_state:
                for language_name in mode[1:]:
                    code = core.LANGUAGES.get(language_name, ("auto",))[0]
                    if code != "auto" and code not in worker._recent_langs:
                        worker._recent_langs.append(code)
            worker.start()
            self.workers.append(worker)

        core.log_event(
            core.logging.INFO,
            "session_config",
            run_id=config.run_id,
            session_id=config.session_id,
            channel=config.channel,
            device_label=config.device_label,
            output_label=config.output_label,
            **session_config,
            tuning={
                "silence_sec": core.SILENCE_SEC,
                "max_phrase_sec": core.MAX_PHRASE_SEC,
                "preview_interval_sec": core.PARTIAL_INTERVAL_SEC,
                "preroll_sec": core.PREROLL_SEC,
                "early_final_sec": core.EARLY_FINAL_SEC,
                "context_tokens": core.CONTEXT_TOKENS,
            },
            app_version=core.APP_VERSION,
        )
        core.log_event(
            core.logging.INFO,
            "session_started",
            session_id=config.session_id,
            channel=config.channel,
            device_label=config.device_label,
            device_index=config.device_index,
            src_lang=config.src_key,
            tgt_lang=config.tgt_key,
            model=config.model_label,
            sensitivity=round(config.sensitivity, 2),
            worker_count=len(self.workers),
            translation_workers=len(self.translation_workers),
            loopback=config.loopback,
            model_device=getattr(model_loader, "active_device", core.WHISPER_DEVICE),
            model_compute_type=getattr(model_loader, "active_compute_type", core.COMPUTE_TYPE),
        )
        core.log_session_context(config.session_id, preview=config.preview_choice,
                                 auto_language=config.auto_language)

    def stop_workers(self) -> None:
        """Request a clean stop and retain the capture thread for safe restart."""
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.stopping_capture = self.capture_thread
            self.capture_thread = None
        if self.loopback_out is not None:
            self.loopback_out.stop()
            self.loopback_out = None
        for worker in self.workers:
            worker.stop()
        self.workers = []
        for worker in self.translation_workers:
            worker.stop()
        self.translation_workers = []

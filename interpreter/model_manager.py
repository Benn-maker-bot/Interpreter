"""The AI models the app keeps loaded: Whisper (main and preview) and Opus-MT.

ModelManager has no window code at all. The Fluent window, the classic
window and the simulator each own one and call it the same way:

    models = ModelManager()
    models.load_main("Turbo (best)", status_cb=...)   # starts loading in the background
    models.choose_preview("Small (balanced)", "Turbo (best)")
    ...
    worker = TranscriptionWorker(..., preview_model=models.preview_model,
                                 translator_for=models.translator_for)

Before this module existed, these methods lived in the classic window
(InterpreterApp) and the Fluent window and simulator borrowed them, which
tied both to the customtkinter code. A window now only passes in what the
user chose in its menus (the menu text, e.g. "Turbo (best)").

THREADS: load_main() and choose_preview() run on the window's thread.
preview_model() and translator_for() are also called by the transcription
worker thread, so they only read plain attributes, and translator_for()
takes _MT_LOADERS_LOCK before it changes mt_loaders.
"""
import logging
from typing import Callable

from .config import MODELS, PREVIEW_CHOICES, PREVIEW_OFF
from .logging_setup import log_event
from .transcription import WhisperModelLoader
from .translation import LocalTranslator, _MT_LOADERS_LOCK, opus_repo, TranslatorLoader


class ModelManager:
    """Loads and holds the Whisper and Opus-MT models.

    Attributes:
        model_loader     WhisperModelLoader for the main model (None at first).
        whisper_model    the main model once a session has taken it from the
                         loader; None until then, and after a model change.
        preview_loader   WhisperModelLoader for the preview model, or None.
        preview_enabled  True when a separate preview model is in use.
        previews_off     True when Previews is set to "Off".
        mt_loaders       (src, tgt) -> TranslatorLoader, one per language pair.
    """

    def __init__(self):
        """Start with nothing loaded."""
        self.model_loader: WhisperModelLoader | None = None
        self.whisper_model = None
        self.preview_loader: WhisperModelLoader | None = None
        self.preview_enabled = False
        self.previews_off = False
        self.mt_loaders: dict[tuple[str, str], TranslatorLoader] = {}

    def load_main(self, model_key: str, status_cb: Callable[[str], None]) -> None:
        """Start loading the main Whisper model chosen in the Model menu.

        model_key  the menu text, e.g. "Turbo (best)" (a key of MODELS).
        status_cb  called with progress messages ("Loading …") for the window.
        """
        self.whisper_model = None
        model_size = MODELS[model_key]
        log_event(logging.INFO, "whisper_model_preload_started", model_size=model_size)
        self.model_loader = WhisperModelLoader(model_size=model_size, status_cb=status_cb)
        self.model_loader.start()

    def choose_preview(self, preview_choice: str, model_key: str) -> None:
        """Load the chosen preview model alongside the main model. Not needed
        when it is the same as the main model.

        preview_choice  the Previews menu text (a key of PREVIEW_CHOICES).
        model_key       the Model menu text, to spot "same as the main model".
        """
        size = PREVIEW_CHOICES[preview_choice]
        # Plain attributes, not the window's menu variables: preview_model()
        # runs on the worker thread, which must not touch window widgets.
        self.previews_off = size == PREVIEW_OFF
        self.preview_enabled = (size not in (None, PREVIEW_OFF)
                                and size != MODELS[model_key])
        if not self.preview_enabled:
            return
        if self.preview_loader is None or self.preview_loader.model_size != size:
            self.preview_loader = WhisperModelLoader(model_size=size, status_cb=lambda m: None)
            self.preview_loader.start()
            log_event(logging.INFO, "preview_model_load_started", model_size=size)

    def preview_model(self):
        """The preview model, or None while it loads (previews then use the
        main model). Called from the transcription worker thread."""
        if not self.preview_enabled:
            return None
        loader = self.preview_loader
        if loader is not None and loader.ready.is_set() and loader.model is not None:
            return loader.model
        return None

    def translator_for(self, src: str, tgt: str) -> Callable[[], LocalTranslator | None] | None:
        """Start loading the Opus-MT model for this pair (first use downloads
        ~300 MB) and return a callable the worker polls; until it is ready,
        translation falls back to Whisper/online. None if no local model
        exists for the pair."""
        if opus_repo(src, tgt) is None:
            return None
        # Workers call this too (a phrase in a new language): one loader per pair
        with _MT_LOADERS_LOCK:
            loader = self.mt_loaders.get((src, tgt))
            if loader is None:
                loader = self.mt_loaders[(src, tgt)] = TranslatorLoader(src, tgt)
                loader.start()
                log_event(logging.INFO, "local_translator_load_started", pair=f"{src}-{tgt}",
                          repo=opus_repo(src, tgt))
        return lambda l=loader: l.translator if l.ready.is_set() else None

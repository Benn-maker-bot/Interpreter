"""Translation: local models first, online services as a last resort.

- LocalTranslator runs an Opus-MT model (University of Helsinki, converted
  for CTranslate2) on this PC: ~30-55 ms on the GPU, no network, no rate
  limits. TranslatorLoader downloads and loads one in the background.
  opus_repo() / local_route() decide which model(s) a language pair needs,
  going via English when there is no direct model.
- _translate_cached() asks the online services (Google, then MyMemory) for
  pairs with no local model, backing off from one that rate-limits us.
- TranslationWorker is a thread that takes phrases from a queue and puts
  the translations on the window's result queue.
"""

import queue
import threading
import time
import logging
import os
from functools import lru_cache

import requests

from .config import LANGUAGES, OPUS_MT_MUL_EN, OPUS_MT_REPOS
from .logging_setup import log_event, log_exception
from .system import INFERENCE_LOCK, WHISPER_DEVICE


_MT_LOADERS_LOCK = threading.Lock()


def opus_repo(src, tgt):
    """Hugging Face repository of the local model for src -> tgt, or None."""
    if (src, tgt) in OPUS_MT_REPOS:
        return OPUS_MT_REPOS[(src, tgt)]
    if tgt == "en" and src not in ("en", "auto"):
        return OPUS_MT_MUL_EN
    return None


def local_route(src, tgt):
    """Pairs to translate through locally: [(src, tgt)] if a model exists,
    [(src, "en"), ("en", tgt)] via English, or [] (use the online service)."""
    if src == tgt or "auto" in (src, tgt):
        return []
    if opus_repo(src, tgt):
        return [(src, tgt)]
    if tgt != "en" and opus_repo(src, "en") and opus_repo("en", tgt):
        return [(src, "en"), ("en", tgt)]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Online translation (fallback)
#
# Normally translation is done locally by Opus-MT (see LocalTranslator) or by
# Whisper itself. These functions are the last resort, for language pairs
# with no local model: they call free web services (Google, then MyMemory)
# over one shared, kept-open HTTPS connection (_http).
# ─────────────────────────────────────────────────────────────────────────────

# One shared web connection for the online translators. Re-using it saves
# ~200 ms per request (no new secure-connection handshake each time).
# _adapter sets how many connections may be open and turns off automatic
# retries (a refused request should go to the next provider instead).
_http = requests.Session()
_http.headers.update({"User-Agent": "Mozilla/5.0"})
_http.adapters.clear()
_adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
_http.mount("http://",  _adapter)
_http.mount("https://", _adapter)


class RateLimitedError(Exception):
    """Raised when an online translation service says "too many requests" (HTTP 429)."""
    pass


def _google_translate(text: str, src: str, tgt: str) -> str:
    """Translate text with Google's free web endpoint (online fallback only).

    Google often blocks this endpoint (HTTP 429); the caller then pauses it.
    """
    url    = "https://translate.googleapis.com/translate_a/single"
    params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
    response   = _http.get(url, params=params, timeout=6)
    if response.status_code == 429:
        raise RateLimitedError("Google returned 429 Too Many Requests")
    response.raise_for_status()
    data = response.json()
    return "".join(p[0] for p in data[0] if p[0]).strip()


# MyMemory's list of supported languages, fetched the first time it's needed.
_MYMEMORY_CODES: dict | None = None

def _mymemory_code(code: str) -> str:
    """Convert a short language code ("ru") into the regional code MyMemory
    expects ("ru-RU"), using MyMemory's own list of supported languages.
    """
    # MyMemory wants region codes ("ru-RU"), not bare ISO codes ("ru").
    global _MYMEMORY_CODES
    if _MYMEMORY_CODES is None:
        from deep_translator import MyMemoryTranslator
        _MYMEMORY_CODES = MyMemoryTranslator(source="en-GB", target="ru-RU") \
            .get_supported_languages(as_dict=True)
    codes = list(_MYMEMORY_CODES.values())
    if code in codes:
        return code
    for c in codes:
        if c.split("-")[0].lower() == code.split("-")[0].lower():
            return c
    raise ValueError(f"MyMemory does not support language '{code}'")


def _mymemory_translate(text: str, src: str, tgt: str) -> str:
    """Translate text with the MyMemory online service (second online fallback)."""
    # Called directly over the shared keep-alive session: deep-translator
    # opens a fresh TLS connection per call, adding ~200 ms each time.
    response = _http.get("https://api.mymemory.translated.net/get",
                     params={"q": text,
                             "langpair": f"{_mymemory_code(src)}|{_mymemory_code(tgt)}"},
                     timeout=6)
    if response.status_code == 429:
        raise RateLimitedError("MyMemory returned 429 Too Many Requests")
    response.raise_for_status()
    data = response.json()
    if data.get("responseStatus") == 429 or "QUOTA" in str(data.get("responseDetails", "")).upper():
        raise RateLimitedError(str(data.get("responseDetails")))
    if data.get("responseStatus") not in (200, "200"):
        raise RuntimeError(f"MyMemory error: {data.get('responseDetails')}")
    return (data["responseData"]["translatedText"] or "").strip()


# Providers are tried in order. A provider that rate-limits us is skipped for
# an increasing cooldown so we stop hammering it (which prolongs the block).
_PROVIDERS = [("google", _google_translate), ("mymemory", _mymemory_translate)]
# How long to stop using a provider that said 'too many requests': 30 s
# the first time, doubling each time up to 600 s (10 min).
# _provider_state: per provider, 'until' = when it may be used again and
# 'cooldown' = the current pause length. _provider_lock protects it
# because more than one thread may translate at once.
_PROVIDER_COOLDOWN_BASE = 30.0
_PROVIDER_COOLDOWN_MAX  = 600.0
_provider_state = {name: {"until": 0.0, "cooldown": 0.0} for name, _ in _PROVIDERS}
_provider_lock  = threading.Lock()


@lru_cache(maxsize=512)
def _translate_cached(text: str, src: str, tgt: str) -> str:
    """Translate text online, trying each provider in turn (Google, MyMemory).

    Results are cached (lru_cache) so repeated phrases are instant. A provider
    that rate-limits us is skipped for a cooldown that doubles each time.
    Only used when no local Opus-MT model exists for the language pair.
    """
    if src == tgt:
        return text
    # Try each online provider in order (_PROVIDERS: Google, then MyMemory)
    # until one succeeds. `errors` collects why each one failed, for the log.
    errors = []
    for name, translate_fn in _PROVIDERS:
        # Is this provider 'cooling down' after refusing us recently? If so,
        # skip it. (_provider_lock: several threads may read/change the state.)
        with _provider_lock:
            state = _provider_state[name]
            wait  = state["until"] - time.monotonic()
        if wait > 0:
            errors.append(f"{name}: cooling down ({wait:.0f}s)")
            continue
        try:
            # Call the provider. On success, reset its cooldown and return the text.
            t0         = time.perf_counter()
            translated = translate_fn(text, src, tgt)
            with _provider_lock:
                state["cooldown"] = 0.0
            log_event(logging.INFO, "translation_completed",
                      provider=name, src=src, tgt=tgt,
                      input_chars=len(text), output_chars=len(translated),
                      elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))
            return translated
        # The provider said 'too many requests' (HTTP 429): don't ask it again
        # for a while. The pause doubles each time (30 s, 60 s, 120 s …, up to
        # 10 min) - called 'exponential backoff' - so we stop pestering it.
        except RateLimitedError as e:
            with _provider_lock:
                state["cooldown"] = min(_PROVIDER_COOLDOWN_MAX,
                                        max(_PROVIDER_COOLDOWN_BASE, state["cooldown"] * 2))
                state["until"]    = time.monotonic() + state["cooldown"]
            # INFO: a blocked service is retried after each cooldown by design
            log_event(logging.INFO, "translation_rate_limited",
                      provider=name, cooldown_s=state["cooldown"], error=str(e))
            errors.append(f"{name}: rate limited")
        except Exception as e:
            log_exception("translation_provider_failed", e,
                          provider=name, src=src, tgt=tgt, input_chars=len(text))
            errors.append(f"{name}: {type(e).__name__}")
    # Every provider failed: log it and raise an error; the caller then shows
    # '[translation failed: …]' in the translation box.
    log_event(logging.ERROR, "translation_failed",
              src=src, tgt=tgt, input_chars=len(text), errors=errors)
    raise RuntimeError("; ".join(errors))


class TranslationWorker(threading.Thread):
    """Background thread that does *online* translation, one phrase at a time.

    The window puts (seq, text, source, target) items on a queue; this thread
    translates them and puts the result on result_queue. Kept to one thread so
    results come back in the same order the phrases were spoken.
    """
    def __init__(self, wid: int, request_queue: queue.Queue, result_queue: queue.Queue):
        """Set up the worker. `wid` is just a number used in the thread's name."""
        super().__init__(daemon=True, name=f"translation-worker-{wid}")
        # wid: this worker's number (for its thread name); request_queue: phrases
        # to translate; result_queue: where results go; _stop_evt: set to stop.
        self.wid           = wid
        self.request_queue = request_queue
        self.result_queue  = result_queue
        self._stop_evt     = threading.Event()

    def stop(self):
        """Ask the thread to finish (the None item wakes it up if it is waiting)."""
        self._stop_evt.set()
        self.request_queue.put(None)

    def run(self):
        """Main loop: take a phrase from the queue, translate it, send the result."""
        while not self._stop_evt.is_set():
            item = self.request_queue.get()
            if item is None:
                break
            seq, text, src_lang_key, tgt_lang_key = item
            _, src = LANGUAGES.get(src_lang_key, ("en", "en"))
            _, tgt = LANGUAGES.get(tgt_lang_key, ("ru", "ru"))
            try:
                t0  = time.perf_counter()
                out = _translate_cached(text, src, tgt)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                self.result_queue.put(("translated", (seq, out or text)))
                self.result_queue.put(("metric", ("translation_ms", elapsed_ms)))
            except Exception as e:
                # Show the failure in the translation box, not just the status
                # bar, which "Listening…" overwrites almost immediately.
                self.result_queue.put(("translated", (seq, f"[translation failed: {text}]")))
                self.result_queue.put(("status", f"Translation error: {e}"))


class LocalTranslator:
    """
    Opus-MT translation model for one language pair. Text is split into
    sentences (Opus-MT is trained on sentences; fragments translate badly)
    and translated as one batch.
    
    This class manages the local translation models that provide fast, offline
    translation between different languages without requiring internet connectivity.
    It handles downloading, loading, and using the translation models.
    """
    def __init__(self, src, tgt):
        """
        Download (first time) and load the Opus-MT model for (src, tgt).
        
        This method downloads and loads a specific translation model from
        Hugging Face. The models are trained on sentence-level data, so text
        is split into sentences before processing.
        
        `sentencepiece` splits text into the sub-word pieces the model was trained
        on; there is one tokenizer for the source and one for the target language.
        
        Args:
            src: Source language code (e.g. "ru")
            tgt: Target language code (e.g. "en")
        """
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download
        # pair: (source, target) language codes, e.g. ("ru", "en").
        # _translator: the Opus-MT model running on CTranslate2.
        # _src_tokenizer / _tgt_tokenizer: split text into, and join it back from,
        # the sub-word pieces the model uses (sentencepiece).
        self.pair = (src, tgt)
        path = snapshot_download(opus_repo(src, tgt))
        device = "cuda" if WHISPER_DEVICE == "cuda" else "cpu"
        self._translator = ctranslate2.Translator(
            path, device=device, compute_type="float16" if device == "cuda" else "int8",
            intra_threads=2 if device == "cuda" else 4)
        self._src_tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(path, "source.spm"))
        self._tgt_tokenizer = spm.SentencePieceProcessor(model_file=os.path.join(path, "target.spm"))

    def translate(self, text: str) -> str:
        """Translate `text`: split into sentences, translate them as one batch on the
        GPU, and join the results.
        """
        import re
        sentences = [x for x in re.split(r"(?<=[.!?…])\s+", text.strip()) if x]
        if not sentences:
            return ""
        batch = [self._src_tokenizer.encode(x, out_type=str) + ["</s>"] for x in sentences]
        with INFERENCE_LOCK:
            results = self._translator.translate_batch(batch, beam_size=2, max_decoding_length=256)
        import html
        # Opus-MT models were trained on text with HTML-escaped punctuation, so
        # they sometimes output "&apos;" for ' or "&quot;" for ": unescape it.
        return " ".join(html.unescape(self._tgt_tokenizer.decode(
            [t for t in r.hypotheses[0] if t != "</s>"]))
                        for r in results).strip()


class TranslatorLoader(threading.Thread):
    """Loads a LocalTranslator on a background thread (first use downloads ~300 MB).

    Until it is ready the app uses Whisper or online translation instead.
    """
    def __init__(self, src, tgt):
        """Prepare to load the translator for the language pair (src, tgt)."""
        super().__init__(daemon=True, name=f"mt-loader-{src}-{tgt}")
        # src/tgt: language pair; translator: the loaded LocalTranslator (or None
        # if loading failed); ready: set when loading finished either way.
        self.src, self.tgt = src, tgt
        self.translator = None
        self.ready = threading.Event()

    def run(self):
        """Load the model, translate one word to warm it up, then signal `ready`."""
        t0 = time.perf_counter()
        try:
            self.translator = LocalTranslator(self.src, self.tgt)
            self.translator.translate("Hello." if self.src == "en" else "Привет.")  # warm-up
            log_event(logging.INFO, "local_translator_ready", pair=f"{self.src}-{self.tgt}",
                      elapsed_ms=round((time.perf_counter() - t0) * 1000))
        except Exception as e:
            log_exception("local_translator_failed", e, pair=f"{self.src}-{self.tgt}")
        finally:
            self.ready.set()

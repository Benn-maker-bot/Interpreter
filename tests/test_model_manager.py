"""Tests for ModelManager's decisions that need no model to be loaded.

Loading a real Whisper or Opus-MT model takes seconds (and a download the
first time), so these tests only use the choices that load nothing: previews
off, previews from the main model, and language pairs with no local model.
"""
import interpreter
from interpreter import ModelManager

TURBO = interpreter.DEFAULT_MODEL_GPU          # "Turbo - best accuracy ★"


def test_new_manager_has_nothing_loaded():
    m = ModelManager()
    assert m.model_loader is None
    assert m.whisper_model is None
    assert m.mt_loaders == {}
    assert m.preview_model() is None


def test_previews_off():
    m = ModelManager()
    m.choose_preview("Off (final text only)", TURBO)
    assert m.previews_off
    assert not m.preview_enabled
    assert m.preview_loader is None


def test_previews_from_the_main_model_load_nothing_extra():
    m = ModelManager()
    m.choose_preview("Main model", TURBO)
    assert not m.previews_off
    assert not m.preview_enabled
    assert m.preview_loader is None
    assert m.preview_model() is None      # the worker then uses the main model


def test_preview_same_size_as_main_model_loads_nothing_extra():
    # "Small" previews while the main model is Small: one model does both.
    m = ModelManager()
    m.choose_preview("Small (balanced)", "Small - recommended ✓")
    assert not m.preview_enabled
    assert m.preview_loader is None


def test_no_local_translator_for_a_pair_without_a_model():
    m = ModelManager()
    assert m.translator_for("en", "en") is None
    assert m.mt_loaders == {}             # nothing started loading

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Load the UPSTREAM Qwen3-TTS reference implementation for the golden dumpers.

Qwen3-TTS is not part of `transformers` at any released version: its model
type (`qwen3_tts_tokenizer_12hz`, architecture `Qwen3TTSTokenizerV2Model`) has
no entry in the auto classes, and the published checkpoints carry no remote
modelling code either. The reference lives only in the `qwen-tts` PyPI package
(GitHub `QwenLM/Qwen3-TTS`), which is what the model card tells a user to
install. So a golden for `crates/mimi` or `crates/ecapatdnn` can only come from
that package - dumping from brain's own port would prove nothing.

Three things make loading it awkward, and this module owns all of them so the
dumpers stay readable:

1. `qwen-tts` pins `transformers==4.57.3`, and the modelling code really does
   depend on that API (`check_model_inputs` changed from a decorator factory to
   a plain decorator in transformers 5). Installing that pin into the ambient
   environment would downgrade a shared interpreter, so the pinned pair is
   installed into a private directory and put on `sys.path` for this process
   only.
2. Importing `qwen_tts.<anything>` executes `qwen_tts/__init__.py`, which pulls
   the 25 Hz tokenizer, `sox` and `torchaudio.compliance.kaldi` - none of which
   the 12 Hz codec or the speaker encoder touch. Registering `qwen_tts` and
   `qwen_tts.core` as bare namespace packages skips those `__init__` bodies
   while leaving every module we actually import completely unmodified. The
   reference code is never patched; that is the whole point of using it.
3. `modeling_qwen3_tts` imports `librosa` and `soundfile` at module scope, for
   the mel front end and for reading clips off disk. A tensor-only dumper (the
   Talker decoder) reaches neither, so `_stub_absent` publishes raising
   placeholders when those libraries are not installed - see its docstring.

The reference tree defaults to `resources/qwen3-tts/` (repo-relative; that
directory is gitignored, like every other downloaded reference source in this
repo) and is fetched on demand. Override with `$QWEN_TTS_REF_DIR`.
"""

import importlib
import os
import subprocess
import sys
import types
import zipfile

__all__ = ["bootstrap", "load_codec", "load_speaker", "load_talker"]

WHEEL = "qwen_tts-0.1.1-py3-none-any.whl"
PACKAGE = "qwen-tts==0.1.1"
# What `qwen-tts` 0.1.1 pins, plus the two libraries whose major versions
# transformers 4.57 refuses to run against.
PINS = ["transformers==4.57.3", "huggingface_hub==0.35.3", "tokenizers==0.22.1"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))


def ref_dir():
    return os.environ.get("QWEN_TTS_REF_DIR") or os.path.join(_REPO, "resources", "qwen3-tts")


def _pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "--disable-pip-version-check", *args])


def _ensure_source(root):
    """Unpacked `qwen-tts` wheel: the reference implementation itself."""
    src = os.path.join(root, "src")
    if os.path.isdir(os.path.join(src, "qwen_tts")):
        return src
    os.makedirs(root, exist_ok=True)
    wheel = os.path.join(root, WHEEL)
    if not os.path.isfile(wheel):
        print(f"[qwen3tts_ref] fetching {PACKAGE} into {root}", flush=True)
        _pip("download", PACKAGE, "--no-deps", "-d", root)
    os.makedirs(src, exist_ok=True)
    with zipfile.ZipFile(wheel) as z:
        z.extractall(src)
    return src


def _ensure_pins(root):
    """The `transformers` version the reference was written against, private to
    this process. Returns the directory to prepend to `sys.path`."""
    site = os.path.join(root, "site")
    marker = os.path.join(site, "transformers", "__init__.py")
    if not os.path.isfile(marker):
        print(f"[qwen3tts_ref] installing {' '.join(PINS)} into {site}", flush=True)
        _pip("install", "--no-deps", "--target", site, *PINS)
    return site


def bootstrap():
    """Put the reference implementation and its pinned `transformers` on the
    path, and neutralise the two `__init__` bodies that pull unrelated deps.

    Idempotent. Must run before `transformers` is first imported, which is why
    every dumper here calls it at the top of `main()` rather than at import
    time next to its own `import torch`."""
    root = ref_dir()
    src = _ensure_source(root)
    site = _ensure_pins(root)
    # The repo's librosa-backed stand-in for a torchaudio whose compiled
    # extension will not load against this torch.
    shim = os.path.join(_HERE, "torchaudio_shim")
    for p in (src, shim, site):
        if p not in sys.path:
            sys.path.insert(0, p)

    import transformers

    if not transformers.__version__.startswith("4.57"):
        raise SystemExit(
            f"transformers {transformers.__version__} is on the path ahead of {site}; "
            "the Qwen3-TTS reference needs the 4.57 API. Run this dumper in a "
            "process that has not already imported transformers."
        )

    for name, sub in (("qwen_tts", ""), ("qwen_tts.core", "core")):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(src, "qwen_tts", sub)]
        sys.modules[name] = mod
    return src


def load_codec():
    """`(config_class, model_class)` for the 12 Hz speech tokenizer."""
    bootstrap()
    cfg = importlib.import_module(
        "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2"
    )
    mod = importlib.import_module(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
    )
    return cfg.Qwen3TTSTokenizerV2Config, mod.Qwen3TTSTokenizerV2Model


def _stub_absent(name, attrs=()):
    """Register a placeholder for an audio-IO module that is NOT installed.

    `qwen_tts.core.models.modeling_qwen3_tts` imports `librosa.filters.mel` at
    module scope (for `mel_spectrogram`) and, through the inference-side
    tokenizer wrapper, `soundfile` (for reading clips off disk). Neither is
    reachable from a pure-tensor forward pass such as the Talker decoder, so a
    dumper that only needs tensors would otherwise have to drag in librosa's
    whole numba/soxr stack to satisfy two import statements.

    The placeholder raises on every name it publishes, so nothing can silently
    degrade: a dumper that really does need mel features fails at the call with
    a `NotImplementedError` naming the module, instead of being handed wrong
    data. It is installed only when the real module is genuinely absent, so an
    environment that has librosa/soundfile behaves exactly as before. The
    reference code itself is still never patched - same principle as the
    namespace-package trick in `bootstrap`.

    The stub publishes ONLY the listed attributes and leaves module dunders
    alone; a catch-all `__getattr__` would break `transformers`' lazy-module
    machinery, which probes attributes to decide what a submodule exports.
    """
    try:
        importlib.import_module(name)
        return
    except ModuleNotFoundError:
        pass
    mod = types.ModuleType(name)
    for attr in attrs:
        def raiser(*_a, _who=f"{name}.{attr}", **_kw):
            raise NotImplementedError(
                f"{_who} is a placeholder: this environment has no {name.split('.')[0]}"
            )
        setattr(mod, attr, raiser)
    sys.modules[name] = mod
    if "." in name:
        parent, leaf = name.rsplit(".", 1)
        setattr(sys.modules[parent], leaf, mod)


def _load_models_module():
    """Import `qwen_tts.core.models.modeling_qwen3_tts`, wiring up the two
    tokenizer names its import chain expects and standing in for absent audio
    IO. Returns the `(configuration, modeling)` module pair."""
    bootstrap()
    _stub_absent("librosa")
    _stub_absent("librosa.filters", ("mel",))
    _stub_absent("soundfile", ("read", "write"))
    # `qwen_tts.core.models.modeling_qwen3_tts` imports the inference-side
    # tokenizer wrapper, which in turn does `from ..core import <4 names>`.
    # Three of those belong to the 25 Hz tokenizer, which needs `sox` and
    # `torchaudio.compliance`; the wrapper only ever uses them for isinstance
    # dispatch on a model we do not build here. Publishing the 12 Hz pair for
    # real and the 25 Hz pair as absent lets the import complete without
    # touching a line of the reference.
    core = sys.modules["qwen_tts.core"]
    v2cfg, v2model = load_codec()
    core.Qwen3TTSTokenizerV2Config = v2cfg
    core.Qwen3TTSTokenizerV2Model = v2model
    core.Qwen3TTSTokenizerV1Config = None
    core.Qwen3TTSTokenizerV1Model = None

    cfg = importlib.import_module("qwen_tts.core.models.configuration_qwen3_tts")
    mod = importlib.import_module("qwen_tts.core.models.modeling_qwen3_tts")
    return cfg, mod


def load_speaker():
    """`(config_class, encoder_class, mel_spectrogram)` for the ECAPA-TDNN
    speaker encoder and the exact mel front end the reference feeds it."""
    cfg, mod = _load_models_module()
    return cfg.Qwen3TTSSpeakerEncoderConfig, mod.Qwen3TTSSpeakerEncoder, mod.mel_spectrogram


def load_talker():
    """`(config_class, model_class, modeling_module)` for the Talker decoder.

    `Qwen3TTSTalkerModel` is the 28-layer decoder stack plus the codec/text
    embedding tables - deliberately NOT `Qwen3TTSTalkerForConditionalGeneration`,
    which also builds the 5-layer MTP code predictor. The modelling module comes
    back too so a dumper can reach `apply_multimodal_rotary_pos_emb` and
    `rotate_half` to check the RoPE convention against the reference's own
    functions rather than a re-implementation."""
    cfg, mod = _load_models_module()
    return cfg.Qwen3TTSTalkerConfig, mod.Qwen3TTSTalkerModel, mod

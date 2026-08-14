# Models & weights

Every servable model has a name, and getting its weights onto disk works one
of two ways: some models fetch and convert themselves automatically the
first time you use them, others need you to point brain at a checkpoint you
already have. This page covers both - how model ids are structured, and how
to get weights in place either way.

## Model naming

Every model brain can serve is named `<vendor>/<repo>[-<QUANT>]`, matching its
upstream URL exactly - case included. This is the same string a browser would
resolve at `https://huggingface.co/<vendor>/<repo>`, plus an optional GGUF
quant suffix. There is exactly one name per servable checkpoint; nothing gets
a second, shorter alias going forward.

```
Qwen/Qwen3-0.6B                    -- the base (safetensors) checkpoint
Qwen/Qwen3-0.6B-Q4_K_M              -- a GGUF quant of the same repo
nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-0.6B-BF16-Q8_0
                                     -- repo "…-BF16", quant "Q8_0"
```

### The grammar

- Exactly one `/`. Byte-exact, case-sensitive comparison.
- A trailing `-<QUANT>` is stripped from `repo` into a separate `quant` field
  when the suffix is one of the closed set of GGUF quant tokens
  (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q2_K`, …) matched in their exact
  canonical uppercase form. `BF16`/`F16`/`F32`/`FP8` are deliberately **not**
  quant tokens - they are base-repo dtype markers, which is what makes
  `…-BF16-Q8_0` parse as repo `…-BF16` + quant `Q8_0` rather than ambiguously.
- Two quant suffixes in a row (`…-Q8_0-Q4_0`) is a parse error with a message
  explaining why.

Parsing and validating a name does zero filesystem or network I/O - it's
always safe and instant, which is what lets discovery endpoints (`GET
/models`) stay offline.

### Reserved vendors

Three vendor prefixes are reserved and are **never** fetched from the
network:

| vendor | meaning |
|---|---|
| `brain/` | built-ins shipped inside the `brain` binary itself (no upstream repo) - `brain/mock`, `brain/demo`, `brain/imageops`, `brain/yolov8`, `brain/zipdepth`, `brain/qwen3tts`, `brain/s3dit`, `brain/flux2-klein`, `brain/chronos2`, `brain/fincast`, `brain/kronos`, `brain/gpt`, `brain/glm`, and the `brain/qwen3`/`brain/lfm2`/`brain/nemotronasr`/`brain/qwen3asr` fallback used when an env-loaded checkpoint carries no upstream vendor/repo provenance |
| `local/` | a file a user dropped into the model store by hand, with no upstream origin to record |
| `test/` | reserved for test-mode mocks |

### Legacy names

Every built-in used to be named by its bare short form (`mock`, `yolo`,
`qwen`, …). Those names still work as **deprecations, not a second id** - an
old client sending `"model":"mock"` still dispatches, resolved to its
canonical `brain/<name>` form behind the scenes. `GET /models` never lists a
legacy name, only the canonical id a model's manifest actually carries.

### On disk

Once fetched or imported, a model's files live under
`<models-dir>/<vendor>/<base-repo>/` - the quant suffix is stripped, because
a quant shares its base repo's tokenizer and config and belongs in the same
directory:

```
<models-dir>/Qwen/Qwen3-0.6B/
    config.json  tokenizer.json  tokenizer_config.json
    model.safetensors           # upstream, as downloaded
    model.brain.safetensors     # brain-format conversion (what actually serves)
    Q8_0.gguf                   # a quant, downloaded or locally produced
```

`<models-dir>` defaults to `$XDG_DATA_HOME/brain/models` and is overridden by
`BRAIN_MODELS_DIR` or `--models-dir` - see
[Configuration](configuration.md#paths).

## Getting the weights

There are two ways a model's weights end up in place, and the model catalog
tells you which applies (look for the **⤓** marker):

- **Auto-fetch (⤓)** - the model id names a real Hugging Face repo (e.g.
  `Qwen/Qwen3-0.6B`, `Ultralytics/YOLOv8`, `LiquidAI/LFM2.5-350M`), and brain
  fetches and converts it itself the first time it's needed - no manual
  export, no extra setup. This happens on the serving surfaces (`brain
  serve`'s HTTP and D-Bus transports, which resolve a named model on demand);
  `brain do` does not auto-fetch - it only reaches models already registered
  locally by name. The first request against a not-yet-fetched model pays a
  one-time download-and-convert cost; every request after that just loads the
  cached, already-converted checkpoint.
- **A local checkpoint** - everything else needs you to point brain at a
  checkpoint on disk, via a `BRAIN_*_WEIGHTS`/`_CKPT`/`_DIR` environment
  variable named on that model's own page (e.g. `BRAIN_YOLOV8`,
  `BRAIN_ZIPDEPTH_WEIGHTS`, `BRAIN_QWEN3TTS_WEIGHTS` + `BRAIN_QWEN3TTS_CKPT`). Unset ⇒ the
  model simply isn't served, with no error. The full list of these variables
  is in [Configuration](configuration.md#model-weights--gating).

## Importing your own checkpoint

For model families that support it, you can also import a Hugging Face
checkpoint you already have into brain's own format, rather than relying on
auto-fetch or a raw upstream file. Qwen3 is the reference case:

```bash
brain qwen import --hf /path/to/Qwen3-0.6B --out qwen.safetensors
```

This converts an HF-layout checkpoint directory into a `.safetensors` file
in brain's own format, which you can then point a `BRAIN_*_WEIGHTS` variable
at (to serve it) or pass directly via `--weights` (for one-off CLI use). Each
model's own page under `docs/models/` documents whether it supports an
`import` step and what its checkpoint layout looks like.

### Importing a GGUF

A quantized GGUF checkpoint is converted by one generic command that picks the
right importer from the file's own `general.architecture` metadata:

```bash
brain import-gguf /path/to/Model-Q4_K_M.gguf        # -> Model-Q4_K_M.brain.safetensors
brain import-gguf FILE --out PATH --id VENDOR/REPO  # explicit output / catalog id
brain import-gguf --list                            # registered architectures
```

By default the conversion is written next to the source as
`<stem>.brain.safetensors`, so if the GGUF already lives in the models
directory the next model-dir scan discovers and serves the result with no
further configuration.

This step is deliberately **explicit and one-time**, not something the
model-dir scan does for you. brain's engine is fp32 (dequantize-on-load), so
converting a quantized GGUF materializes a much larger file - a 22 GB Q4_K_M
of a 35B model becomes roughly 140 GB of fp32 safetensors. Silently writing
that during a server-startup directory scan would be a surprising use of disk
and an unbounded startup delay, so the scan instead logs the exact
`brain import-gguf` command to run.

Not every GGUF architecture needs this. Architectures brain serves directly
from GGUF (e.g. `qwen3`) are picked up by the model-dir scan as they are; the
import path is for architectures whose tensor layout has to be translated
first. `brain import-gguf --list` shows which those are.

## See also

- [Configuration](configuration.md) - every `BRAIN_*` environment variable,
  including every model's weights variable.
- [Serving](serving.md) - how auto-fetch and model resolution work when
  models are served over HTTP/D-Bus.
- [The CLI](cli.md) - `brain caps` and `brain do`, the uniform ways to reach
  any model.

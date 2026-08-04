# Model naming

Every model brain can serve is named `<vendor>/<repo>[-<QUANT>]`, matching its
upstream URL exactly — case included. This is the same string a browser would
resolve at `https://huggingface.co/<vendor>/<repo>`, plus an optional GGUF
quant suffix. There is exactly one name per servable checkpoint; nothing gets
a second, shorter alias going forward.

```
Qwen/Qwen3-0.6B                    -- the base (safetensors) checkpoint
Qwen/Qwen3-0.6B-Q4_K_M              -- a GGUF quant of the same repo
nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-0.6B-BF16-Q8_0
                                     -- repo "…-BF16", quant "Q8_0"
```

## The grammar (`crates/modelref`)

`brain_modelref::ModelRef` is the parser and the type every serving path
carries a model identity as: `{ vendor, repo, quant: Option<Quant> }`.

- Exactly one `/`. Byte-exact, case-sensitive comparison.
- A trailing `-<QUANT>` is stripped from `repo` into a separate `quant` field
  when the suffix is one of the closed set of GGUF quant tokens
  (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `Q2_K`, …) matched in their exact
  canonical uppercase form. `BF16`/`F16`/`F32`/`FP8` are deliberately **not**
  quant tokens — they are base-repo dtype markers, which is what makes
  `…-BF16-Q8_0` parse as repo `…-BF16` + quant `Q8_0` rather than ambiguously.
- Two quant suffixes in a row (`…-Q8_0-Q4_0`) is a parse error with a message
  explaining why.

`ModelRef::parse` does zero filesystem or network I/O — parsing and validating
a name is always safe and instant, which is what lets discovery endpoints
(`GET /models`) and the reserved-vendor check below stay offline.

## Reserved vendors

Three vendor prefixes are reserved and are **never** fetched from the network:

| vendor | meaning |
|---|---|
| `brain/` | built-ins shipped inside the `brain` binary itself (no upstream repo) — `brain/mock`, `brain/demo`, `brain/imageops`, `brain/yolo`, `brain/depth`, `brain/tts`, `brain/z-image`, `brain/flux2-klein`, `brain/chronos2`, `brain/fincast`, `brain/kronos`, `brain/gpt`, `brain/glm`, and the `brain/qwen`/`brain/lfm`/`brain/nemotron`/`brain/qwen-asr` fallback used when an env-loaded checkpoint carries no upstream vendor/repo provenance |
| `local/` | a file a user dropped into the model store by hand, with no upstream origin to record |
| `test/` | reserved for test-mode mocks (not yet used) |

`ModelRef::is_reserved()` is the check: a reserved-vendor ref resolves purely
from the grammar and local disk, never touching a hub.

## Legacy names (`modelref::alias`)

Every built-in used to be named by its bare short form (`mock`, `yolo`,
`qwen`, …). Those names still work — an old client sending `"model":"mock"`
still dispatches — but they are **deprecations, not a second id**:

- `modelref::alias::canonical(name)` maps a known legacy short name to its
  canonical `brain/<name>` form, or returns `None` (already canonical, or
  simply unknown — both are the caller's problem to handle normally).
- It is consulted in exactly two places: `apiserve::catalog::candidates` (every
  HTTP surface) and the D-Bus/`brain do` model-argument resolution. Nowhere
  else.
- `GET /models` never lists a legacy name — only the canonical id a manifest
  actually carries. A `capability::Manifest.model` field is **always**
  canonical; the alias table exists purely at the two dispatch seams above.

Adding a model does not mean adding a short alias. If a model genuinely
shipped under a short name before this convention existed, add one row to
`modelref::alias::ROWS` alongside it — don't invent a new short form.

## The model store (`crates/modelstore`)

On-disk models live under `<models-dir>/<vendor>/<base-repo>/` — the quant
suffix is stripped, because a quant shares its base repo's tokenizer and
config and belongs in the same directory:

```
<models-dir>/Qwen/Qwen3-0.6B/
    config.json  tokenizer.json  tokenizer_config.json
    model.safetensors           # upstream, as downloaded
    model.brain.safetensors     # brain-format conversion (what actually serves)
    Q8_0.gguf                   # a quant, downloaded or locally produced
```

`brain_modelstore::Store` finds what's already on disk (`Store::local`,
`Store::scan`); `brain_modelstore::plan` is the pure resolution ladder that
decides what would need to happen for a ref that isn't there yet (on disk →
serve; an existing upstream quantized artifact → download; otherwise the base
checkpoint plus a local quantize step). `crates/cli/src/model_dir.rs`'s
catalog scan prefers this layout, falling back to the original flat
single-level directory for back-compat (with a one-time warning) so older
model directories keep working.

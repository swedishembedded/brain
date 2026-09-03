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

`<models-dir>` defaults to `$XDG_DATA_HOME/brain/models` (in practice, absent
`XDG_DATA_HOME`, `$HOME/.local/share/brain/models` - **not**
`~/.local/brain/models`) and is overridden by `--models-dir`, the global
`--brain-data-dir <root>` (models land in `<root>/models`), or
`BRAIN_MODELS_DIR` - see [Configuration](configuration.md#paths) for the full
precedence ladder.

## Seeing what you have

`brain models list` is the store's own view of itself: architecture →
provider repo → quantization, joining what is actually on disk against each
architecture's declared official quantizations (`brain_arch::Arch::variants`
- a real, verified-against-upstream list, not a live probe) and a cached
per-model cost.

```
qwen3  Qwen3 dense decoder  (6 repos, 1 local)
├─ Qwen/Qwen3-0.6B  0.60B params  local
│  ├─ qwen3 Qwen/Qwen3-0.6B          base    local  1.9 GiB  gpu0 fits  exact 1.2 GFLOP (100% covered)
│  └─ qwen3 Qwen/Qwen3-0.6B-Q8_0     Q8_0    not pulled
└─ Qwen/Qwen3-8B  8.20B params  not pulled
   ├─ qwen3 Qwen/Qwen3-8B-Q4_K_M     Q4_K_M  not pulled
   └─ qwen3 Qwen/Qwen3-8B-Q8_0       Q8_0    not pulled
```

On a terminal this renders as an interactive tree (arrow keys / `j`/`k` /
`PageUp`/`PageDown` to move, `Enter`/`Space` to expand or collapse a branch -
or, on a pulled model's own row, open its tensor detail view, the same tree
`brain models info` prints, with its own arrows/pgup/pgdn scrolling - `Esc` to
leave a detail view or quit at the top level, `/` to filter, `q` to quit);
piped or redirected it renders as the plain lines above, and every LEAF line
carries the full canonical id, so `brain models list | grep Q4_K_M` returns
complete, self-explanatory lines rather than fragments that only make sense
next to a parent row. `--json` emits the same tree as data; `--arch <id>`
filters to one architecture; `--local` drops every declared-but-not-pulled
row.

The cost column is a **cache read** - `brain models list` never opens a
device or builds a model by itself, so it stays fast regardless of how many
models are in the store. A model that has never been priced on this machine
reads `not profiled`, not a guess. Three ways to fill it in:

- `brain models list --reprofile` re-measures this device's hardware
  roofline and re-prices **every** local model, at the cheap bandwidth-only
  tier (tensor byte sizes, no real weight buffer materialized) - safe to run
  unattended regardless of how large a local checkpoint is.
- `brain models profile <model>` prices **one** already-pulled model now, at
  the exact tier if its architecture has one registered, and errors (never
  fetches) if it is not pulled.
- `brain flops --model ... --weights <path>` (see [The
  CLI](cli.md#infrastructure-verbs-unchanged-not-per-architecture)) writes
  into this SAME cache when it prices a real on-disk model - `brain flops`
  and `brain models list` share one pricing engine (`crates/modelcost`), so
  running one primes the other.

An exact-tier price needs a real (if weight-free) model build and is only
registered for a few architectures so far (`qwen3`, `gpt2`, `lfm2`); every
architecture with a shape manifest gets the bandwidth tier at least - a real
weight size and a memory-bandwidth-bound lower bound, never a fabricated FLOP
count (GGUF quantization changes bytes moved, not floating-point operation
count, in brain's dequantize-on-load engine).

`brain models list-adapters` is the same idea for LoRA adapters: architecture
→ base variant → adapter, with rank/alpha/targets/dataset from the adapter's
own card. `brain models info <model>` prints one checkpoint's real tensor
tree - every tensor's own name, dtype and shape (a GGUF routinely mixes
precision per tensor) - with any pulled adapter's tensors merged in at the
node they target, marked with a leading `+`.

### Real, timed measurement: `brain models profile --measure`

Everything above is a *dry* cost - shape-derived, no device, no execution.
`brain models profile <model> --measure [--reps N]` is the other kind: it
builds the model for real and runs it, reporting

- **load** time (weight upload + pipeline build) separately from a forward
  pass's own time - folding the two together would answer a different
  question than either "how long does inference take" or "how long before
  the first request";
- **cold** (the first pass, pipeline specialisation still pending) separately
  from **hot** (the best of `--reps` passes after that, default 5) - a
  single flattering "average" would hide the real one-time cost a cold start
  actually pays;
- achieved FLOP/s at both cold and hot, from the wall-clock measurement
  against the same FLOP total the dry tier already computes;
- a per-layer FLOP figure - for a uniform stack (`qwen3`, `gpt2`) this is
  DERIVED from dry probes at 0/1/2 layers and verified affine at the
  point outside that basis, exactly the way this workspace already prices
  flux2/ltxv's block-depth-affine graphs; for a hybrid stack whose layers
  are not interchangeable (`lfm2`'s per-layer choice of gated short-conv vs
  attention) it is the plain average (`total ÷ layer count`) instead, and is
  never presented as more precise than that.

Never cached - a timing is a fact about this machine right now, not about the
model, unlike the dry tiers above.

## Getting the weights

There are two ways a model's weights end up in place, and the model catalog
tells you which applies (look for the **⤓** marker):

- **`brain pull <model>` / auto-fetch (⤓)** - the same operation, explicit or
  opt-in. `brain pull Qwen/Qwen3-0.6B` (or the HuggingFace URL) fetches and
  converts a model up front, with a progress bar. Auto-fetch is the same
  fetch run implicitly by a first inference or serve request, but only when
  you ask for it: pass the global `--autofetch` flag (or export
  `BRAIN_AUTO_FETCH=1`). With it off - the default - a run whose weights are
  not pulled prints an error naming what is missing instead of downloading
  anything. Use `brain pull` when you want the download to happen now, on a
  connection you are watching, rather than in the middle of a first
  inference. See [The CLI](cli.md#pulling-weights).
- **Auto-fetch (⤓, opt-in)** - the model id names a real Hugging Face repo
  (e.g. `Qwen/Qwen3-0.6B`, `Ultralytics/YOLOv8`, `LiquidAI/LFM2.5-350M`), and
  brain fetches and converts it itself the first time it's needed - no manual
  export, no extra setup. This happens only with `--autofetch` /
  `BRAIN_AUTO_FETCH=1`, on the CLI and on the serving surfaces (`brain
  serve`'s HTTP and D-Bus transports, which resolve a named model on demand);
  `brain do` never auto-fetches - it only reaches models already registered
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
brain qwen3 import --hf /path/to/Qwen3-0.6B --out qwen.safetensors
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
brain import /path/to/Model-Q4_K_M.gguf        # -> Model-Q4_K_M.brain.safetensors
brain import FILE --out PATH --id VENDOR/REPO  # explicit output / catalog id
brain import --list                            # registered architectures
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
`brain import` command to run.

Not every GGUF architecture needs this. Architectures brain serves directly
from GGUF (e.g. `qwen3`) are picked up by the model-dir scan as they are; the
import path is for architectures whose tensor layout has to be translated
first. `brain import --list` shows which those are.

### Native K-quant GPU execution

Even a GGUF architecture that loads directly used to pay a hidden cost on
every run: `checkpoint::gguf` dequantized every block format to fp32 the
moment a tensor was read, so a quantized tier's only advantage was a smaller
download, never a smaller device footprint. The engine (`crates/gguf`,
`crates/model`, `crates/kernels`) now has a lossless, on-device execution
path for six GGML block formats instead - Q4_K and Q5_K through two new
affine K-quant kernels, and Q6_K/Q5_0/Q4_0/Q8_0 by reusing the existing
symmetric int8 GEMM kernels through template knobs (group size and code
width are compile-time constants those kernels already took; no new kernel
files). None of the six needs a host-side fp32 dequantize to be usable on
the device this way - each block's raw codes and per-group scale/min are
relaid out once, on the host, into one shared device-native layout, and
reconstructing them from that layout reproduces the fp32 dequantize path bit
for bit (checked by an `assert_eq!`, never a tolerance).

That shared layout has a real, measured byte cost, because giving all six
formats one shape means the three legacy formats - which have no
super-block grouping coarser than their own 32-element block - can't
amortize the packed scale plane the way the K-quant formats' 256-element
super-block can:

| type | device bytes/param | GGUF bytes/param | ratio |
|---|---|---|---|
| Q4_K | 0.578125 | 0.5625     | 1.0278x |
| Q5_K | 1.078125 | 0.6875     | 1.5682x |
| Q6_K | 1.140625 | 0.8203125  | 1.3905x |
| Q5_0 | 1.187500 | 0.6875     | 1.7273x |
| Q4_0 | 0.687500 | 0.5625     | 1.2222x |
| Q8_0 | 1.187500 | 1.0625     | 1.1176x |

Q8_0's row is this shared path's theoretical cost; production Q8_0 weights
never actually take it - they reach the device through the older, separate,
already-shipped `gguf::int8_direct` byte-exact path this engine has used for
a while (`flux2`, `s3dit`, `ltxv`, `gemma4`, `wan`, and `qwen3`'s own int8
tier all go through it today).

This is engine-level capability, not yet a served model's default. Every
model crate that reads a K-quant GGUF today still dequantizes those tensors
to fp32 on load, same as before this landed - none has wired its own loader
onto the new device path yet. The kernels, the host relayout, and the device
dispatch are implemented and gated by real device-level tests; a model crate
actually reaching for a K-quant tier this way is the remaining step.

### Quantizing a checkpoint

The opposite direction - a full-precision checkpoint to a quantized GGUF -
is one generic command that needs no per-architecture code:

```bash
brain quantize model-bf16.safetensors --out model-Q8_0.gguf --arch <name>
brain quantize HF_DIR --out model-Q8_0.gguf          # a directory of shards
brain quantize model-bf16.safetensors --plan          # decide, print, write nothing
```

The source may be a `.safetensors` file, a HuggingFace-style directory of
them, or an existing `.gguf`. A tensor is quantized when it is a rank-2
matrix whose fastest-varying dimension is a whole number of blocks; every
other tensor is written through unchanged as F32. Those two rules are
structural - a quantized GEMM operand is a matrix, and a block carries one
scale for its own contiguous elements, so a row length that is not a block
multiple cannot be encoded at all.

What is NOT inferred is which named tensors an architecture must keep at
full precision regardless of shape - modulation tables, conditioning
projections, anything whose numeric scale the rest of the graph depends on.
Pass those with `--keep SUBSTR` (repeatable, or comma-separated);
`--min-elems N` additionally keeps anything smaller than `N` elements.

Every source tensor is accounted for in the output, and `--plan` prints the
decision and the reason for each one before anything is written.

## See also

- [Configuration](configuration.md) - every `BRAIN_*` environment variable,
  including every model's weights variable.
- [Serving](serving.md) - how auto-fetch and model resolution work when
  models are served over HTTP/D-Bus.
- [The CLI](cli.md) - `brain caps` and `brain do`, the uniform ways to reach
  any model.

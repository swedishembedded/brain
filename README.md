# brain

A small, dependency-light framework for **training and evaluating neural networks
from scratch on the GPU** — **pure Rust + raw WGSL**, fp32-only so the same kernels
run on old desktop GPUs, on modern integrated GPUs, and in the browser via WebGPU.
It is a self-contained Cargo workspace (`crates/`) with **no Python in the build or
test path**; backprop correctness is gated by an in-repo finite-difference gradient
checker instead of a PyTorch oracle.

One engine, **three runtime backends**, and a growing family of real models — from a
nanoGPT-parity decoder to a from-scratch, checkpoint-compatible **Qwen3-TTS** voice
cloner.

- Architecture & crate graph: [`docs/architecture.md`](docs/architecture.md)
- Routing guide for contributors: [`AGENTS.md`](AGENTS.md)
- Testing strategy & the gradient-check gate: [`docs/testing.md`](docs/testing.md)
- Performance notes: [`docs/performance/overview.md`](docs/performance/overview.md)
- **Scaling across GPUs (data / pipeline / tensor parallelism):** [`docs/scaling/overview.md`](docs/scaling/overview.md)
  → [`docs/scaling/data-parallel.md`](docs/scaling/data-parallel.md), [`docs/scaling/pipeline-sharding.md`](docs/scaling/pipeline-sharding.md), [`docs/scaling/tensor-parallel.md`](docs/scaling/tensor-parallel.md)
- Per-area deep dives: `docs/models/yolo/`, `docs/models/tts/`, `docs/federated.md`, `docs/engine/`

---

## Quick start

```bash
make release                          # build the optimized ./target/release/brain
make test                             # full cargo test suite
make gradcheck                        # backprop correctness gate (finite differences)

# Train + evaluate the GPT baseline end to end:
make data/calculator                  # generate a dataset
make train/gpt/calculator             # -> out/gpt-calculator.safetensors
make eval/gpt/calculator              # validation perplexity + task exact-match
```

Every model is one `brain <model> <verb>` subcommand — see the
[Model support](#model-support) table below for the full, current list of models and
what each supports; the frequently-used ones:

```
brain data        dataset generation + tokenizers
brain devices     canonical GPU table (index, PCI bus, UUID, VRAM) + ambient selection
brain gpt         GPT decoder: train | gen | eval
brain qwen        Qwen3 LLM: import | infer | export | precompile | train | finetune
brain glm         GLM-5.2 decoder: train | finetune | infer | eval | import | export
brain lfm         LFM2.5-Encoder: import | fill-mask | embed | data | finetune | eval
brain tts         Qwen3-TTS: import | clone | synth | design | serve | sim | finetune
brain yolo        YOLOv8 detector: train | fine-tune | eval | detect
brain depth       ZipDepth monocular depth: image | camera | calib | train
brain flux2       FLUX.2 Klein text-to-image + editing: generate
brain mirror      WorldMirror-2 3D reconstruction: import | infer | demo
brain splat       3D Gaussian Splatting: info | render | view | fit
brain wm          playable world models (DIAMOND): play | replay | bench | finetune
brain forecast    Chronos-2 / Kronos / FinCast forecasting: compare | serve | import | finetune
brain npu         OpenVINO/NPU: export | quantize | check | run | bench | sim
brain federated   sharded MoE: split | verify | merge | assemble | train-expert
brain pid         PID control transformer
brain bench       architecture-evaluation harness (+ eval | scale | advise | compare)
brain perf        performance benchmarking (latency/throughput/serve/sweep, vs a baseline)
brain flops       offline/online FLOP and int-OPS accounting for a forward/backward
brain caps        every model's action manifest; `brain do <model> <action>` runs one
brain run         event-driven streaming controller (HFSM over JSONL); alias: serve
brain gradcheck   finite-difference backprop correctness gate
```

`brain serve [--openai [PORT]] [--anthropic [PORT]] [--openrouter [PORT]] [--dbus]`
serves the same models over localhost HTTP and/or D-Bus — see `brain serve --help` and
the Model support table for which surface each model actually answers on.

## Backends — CPU, GPU, NPU

The same WGSL kernels run on three backends, selected at runtime:

```bash
brain gpt gen --weights out/gpt-calculator.safetensors --device gpu   # wgpu (default)
brain gpt gen --weights out/gpt-calculator.safetensors --device cpu   # WGSL -> Cranelift JIT, all cores
BRAIN_DEVICE=cpu make test                                        # whole suite, no GPU needed
```

- **GPU (wgpu / WebGPU)** — the default; runs on desktop GPUs, integrated GPUs, and in
  the browser.
- **CPU** — the exact same WGSL, JIT-compiled to native code via Cranelift and run
  across all cores with rayon. No GPU required.
- **NPU (Intel, via OpenVINO)** — a separate whole-graph export → compile → run path
  (`brain npu …`), loaded at runtime so the default build stays free of OpenVINO.

---

## Model support

Every model `brain` can run today, grouped by task. **Model id** is the exact string
you pass to `brain`/`brain do`/D-Bus/HTTP; **✓** means the capability is reachable
through some real command, documented on the linked page — nothing here is
aspirational. Architecture ports with no serving surface yet (parity-gated components,
not runnable models — T5-XXL encoder, SDXL UNet, ControlNet, FLUX.1/Kontext, PuLID,
plus the forward-only `qwenvl`/`moondream` VLMs) are listed below the table instead of
given a row.

| Model id | Solves | Infer | Train | LoRA | QLoRA |
|---|---|---|---|---|---|
| **text** | | | | | |
| [`Qwen/Qwen3-0.6B`](docs/models/qwen/readme.md) ⤓ | instruct/chat LLM, tool calls, paged-KV serving | ✓ | ✓ | ✓ | |
| [`brain/gpt`](docs/models/gpt/readme.md) | dense nanoGPT-parity baseline | ✓ | ✓ | | |
| [`brain/glm`](docs/models/glm/readme.md) | GLM-5.2 decoder (MLA + sigmoid MoE + DSA + MTP) | ✓ | ✓ | | |
| [`brain train`](docs/models/moe/readme.md) | sparse top-2-of-4 MoE toy next-token rule | ✓ | ✓ | | |
| **embed** | | | | | |
| [`LiquidAI/LFM2.5-350M`](docs/models/lfm/readme.md) ⤓ | text embeddings + fill-mask, 8k context | ✓ | ✓ | | |
| [`brain/clip`](docs/models/clip/readme.md) | CLIP-L / OpenCLIP-bigG / EVA-CLIP embeddings | ✓ | | | |
| [`brain/facenet`](docs/models/face/readme.md) | face detection + ArcFace identity embedding | ✓ | | | |
| **vlm** | | | | | |
| [`brain/fastvlm`](docs/models/vlm/readme.md) | image captioning | ✓ | | | |
| **asr** | | | | | |
| [`brain/nemotron`](docs/models/asr/readme.md) | streaming speech-to-text | ✓ | | | |
| [`brain/qwen-asr`](docs/models/asr/readme.md) | offline speech-to-text | ✓ | | | |
| **tts** | | | | | |
| [`brain/tts`](docs/models/tts/readme.md) | voice cloning / speech synthesis | ✓ | | ✓ | |
| **image gen** | | | | | |
| [`Tongyi-MAI/Z-Image-Turbo`](docs/models/zimage/readme.md) ⤓ | text-to-image (S³-DiT diffusion) | ✓ | | ✓ | |
| [`brain/flux2-klein`](docs/models/flux2/readme.md) | text-to-image + reference-image editing | ✓ | | ✓ | |
| **image edit** | | | | | |
| [`brain/restore`](docs/models/restore/readme.md) | blind face restoration (CodeFormer) | ✓ | | | |
| [`brain/upscale`](docs/models/upscale/readme.md) | 4x super-resolution (Real-ESRGAN) | ✓ | | | |
| [`brain/vqgan`](docs/models/vqgan/readme.md) | image ↔ codebook indices (VQ encode/decode) | ✓ | | | |
| [`brain/imgpipe`](docs/models/imgpipe/readme.md) | composed segment → restore → upscale, one call | ✓ | | | |
| **vision** | | | | | |
| [`Ultralytics/YOLOv8`](docs/models/yolo/readme.md) ⤓ | anchor-free object detection (YOLOv8-style) | ✓ | ✓ | | |
| [`brain/depth`](docs/models/depth/readme.md) | monocular relative depth (ZipDepth) | ✓ | ✓ | | |
| [`brain/sam2`](docs/models/sam2/readme.md) | promptable image segmentation (SAM 2.1) | ✓ | | | |
| **3d** | | | | | |
| [`brain mirror`](docs/models/mirror/readme.md) | multi-view photos → 3D Gaussian Splatting scene | ✓ | | | |
| [`brain splat`](docs/models/splat/readme.md) | 3DGS render / fly-through / scene fit | ✓ | ✓ | | |
| **forecast** | | | | | |
| [`brain/chronos2`](docs/models/chronos2/readme.md) | probabilistic time-series forecasting | ✓ | | | |
| [`brain/fincast`](docs/models/fincast/readme.md) | probabilistic time-series forecasting | ✓ | | | |
| [`brain/kronos`](docs/models/kronos/readme.md) | OHLCV bar forecasting | ✓ | ✓ | ✓ | |
| **world model** | | | | | |
| [`brain wm`](docs/models/world-models/readme.md) | playable action-conditioned video (DIAMOND) | ✓ | ✓ | | |
| [`brain pid`](docs/models/pid/readme.md) | control policy over CBOR records (PID imitation) | ✓ | ✓ | | |

**⤓** = `brain` fetches and converts the weights itself on first use
(`crates/modelstore`, via a per-family recipe — `crates/modelstore/src/recipe.rs`)
— everything else needs a local checkpoint, pointed at by a `BRAIN_*` env var named
on the model's own page. Three recipes exist today: HF `transformers`-shaped repos
(`config.json` + a recognized architecture — `qwen`, `glm`, or `lfm`; GLM has no
known public checkpoint today so it stays a `brain/` id despite the code path
existing), Z-Image's diffusers-pipeline shape, and YOLO's flat GitHub-release
shape — the same downloader underneath every one of them, so a future source
(e.g. p2p) only has to implement one small `Hub` trait, not a fourth copy of the
fetch/store/single-flight machinery.
**QLoRA is not implemented anywhere in brain** — the INT8/GGUF paths are inference
tiers only (`crates/qwen/src/model.rs` asserts the int8 path is inference-only); the
column is here so the gap stays visible rather than silently absent.
Which of CLI / D-Bus / HTTP a model answers on is stated at the top of its own page —
it varies per model and is derived from each manifest's action shape
(`crates/apiserve/src/catalog.rs`), not configured by hand.

## Models — a closer look

### GPT decoder — the dense baseline

nanoGPT-parity: token + learned positional embeddings, pre-LN, causal MHA, GELU MLP,
untied head, masked cross-entropy.

```bash
make data/calculator                  # or: reverser wordcalc timeseries shakespeare_char gpt
brain gpt train data/calculator --out out/gpt.safetensors --steps 2000 --batch 32 --block 64
brain gpt eval  --weights out/gpt.safetensors --data data/calculator
brain gpt gen   --weights out/gpt.safetensors --prompt "12+7=" --max-new 8
```

### Qwen3 LLM — real 0.6B, on CPU/GPU/NPU

A real, HF-parity-exact Qwen3 dense decoder (RMSNorm, GQA + per-head QK-norm,
half-split RoPE, SwiGLU, tied head), with safetensors import, LoRA, and ONNX/OpenVINO
export.

```bash
brain qwen import --hf <hf_dir> --out qwen.safetensors        # import HF safetensors
brain qwen infer  --weights qwen.safetensors --tokenizer tokenizer.json --prompt "The capital of France is"
brain qwen finetune data/mydata --weights qwen.safetensors --out qwen-ft.safetensors   # full or LoRA
brain qwen export --weights qwen.safetensors --out qwen.onnx --seq 16               # -> ONNX (NPU)
brain qwen precompile --weights qwen.safetensors --seq 16 --npu-cache out/npu-cache # warm NPU blob cache
brain qwen infer --weights qwen.safetensors --device npu --seq 16 --npu-cache out/npu-cache --prompt "…"
```

### Qwen3-TTS — from-scratch, checkpoint-compatible voice cloning

A complete Qwen3-TTS stack built from scratch on the same engine, parity-verified
against the official reference: a Mimi-style 12 Hz neural **codec** (max-abs 3.7e-2
vs reference), an **ECAPA-TDNN speaker encoder** (cosine 1.000), and a Qwen3 **Talker**
+ 5-layer **MTP** code predictor (top-1 logits exact). End-to-end voice clone reaches
**0.96 speaker-similarity** to the reference voice — matching the official model's own
baseline. See [`docs/models/tts/readme.md`](docs/models/tts/readme.md).

```bash
# import the four components (Talker, MTP, codec, speaker) from the HF checkpoints:
brain tts import --ckpt <Qwen3-TTS-12Hz-0.6B-Base> --codec-ckpt <Qwen3-TTS-Tokenizer-12Hz> --out-dir out/tts

# voice clone: synthesize new text in the timbre of a reference voice
brain tts clone --weights-dir out/tts --ckpt <hf_dir> \
                --text "Hello from brain." --ref voice.wav --ref-text "transcript of voice.wav" \
                --lang english --out clone.wav

# speaker-free text-to-speech
brain tts synth --weights-dir out/tts --ckpt <hf_dir> --text "Hello from brain." --out tts.wav

# single-speaker LoRA fine-tune
brain tts finetune <data_dir> --weights-dir out/tts --out out/tts-ft
```

Codec decode also runs on the Intel NPU (OpenVINO); see `docs/models/tts/readme.md` for the
export/run path and the streaming `audio_chunk` serving seam.

### YOLOv8 detector — from-scratch object detection

Anchor-free CSP backbone → PAN-FPN neck → decoupled DFL head, with the assigner +
BCE/CIoU/DFL loss and NMS decode. Byte-compatible with canonical `yolov8n` for weight
import.

```bash
make data/detect                      # synthetic RGB-shapes detection dataset
brain yolo train data/detect --out out/yolo.safetensors --steps 500 --batch 16
brain yolo eval   --weights out/yolo.safetensors --data data/detect      # mAP@0.5 + P/R
brain yolo detect --weights out/yolo.safetensors --image sample.ppm      # JSON boxes
```

### Sparse MoE Transformer (+ federated experts)

RMSNorm/RoPE, top-2-of-4 routed SwiGLU experts; a toy 64-symbol next-token rule for
studying memorization vs. generalization, with vertical expert sharding.

```bash
brain train data/moe --out out/moe.safetensors            # MoE train
make federated-demo                                   # train -> split -> verify -> merge
brain federated split out/moe.safetensors out/shards/
brain federated verify out/shards/
brain federated merge  out/shards/ --out out/merged.safetensors
```

### PID control Transformer

A control policy over CBOR records that imitates a PID oracle — the model behind the
browser demo (`make web/dev`).

```bash
brain pid …                           # see `brain pid` for the subcommands
```

### The rest of the catalog

The six above get a full walkthrough because they're the models most people start
with; brain has grown a much larger family since, all sharing this same engine and
gradient-check discipline. Full detail — architecture notes, parity numbers,
serving status — lives in [`AGENTS.md`](AGENTS.md)'s "Models (today)" section and
each model's own `docs/models/<name>/` (or `docs/imaging/plan.md` for the imaging
pipeline models below).

| Model | Crate | What it is |
|---|---|---|
| GLM-5.2 | `glm` | MLA + MoE + DSA indexer + MTP decoder |
| Qwen3-Omni-30B Thinker | `omni` | dense-then-MoE Qwen3-based omni-modal decoder, served over multi-GPU residency |
| Qwen3-VL-4B | `qwenvl` | Qwen3 decoder + vision encoder, image-conditioned generation |
| FastVLM-0.5B | `fastvlm` | Apple's FastViTHD vision encoder + LLaVA-style splice into a Qwen2 decoder |
| Moondream 3 | `moondream` | vision encoder + parallel-block decoder VLM |
| Seq2seq | `seq2seq` | encoder-decoder Transformer |
| LFM2.5-Encoder | `lfm` | LiquidAI's bidirectional conv/attention hybrid, MLM |
| Bottleneck autoencoder | `autoencoder` | sequence → single-vector bottleneck → sequence |
| ASR (streaming) | `qwen-asr` | Whisper-style + Nemotron FastConformer streaming transducer |
| Z-Image | `zimage` | Tongyi S³-DiT text-to-image |
| FLUX.1 / Kontext | `flux1` | BFL's 12B MMDiT text-to-image + edit path |
| FLUX.2 Klein | `flux2` | BFL's 4B/9B MMDiT text-to-image |
| T5-XXL encoder | `t5` | text conditioning encoder for FLUX.1 |
| SDXL UNet2DConditionModel | `unet` | SDXL diffusion UNet + discrete samplers |
| ControlNet | `controlnet` | backbone-agnostic spatial control seam for SDXL |
| PuLID-FLUX | `pulid` | ArcFace-conditioned identity injection into FLUX.1 |
| VQGAN / CodeFormer autoencoder | `vqgan` | VQ autoencoder shared by CodeFormer and the imaging pipeline |
| CodeFormer restoration | `restore` | code-Transformer face restoration over `vqgan` |
| Real-ESRGAN | `upscale` | RRDBNet super-resolution, the imaging pipeline's upscale tail |
| CLIP | `clip` | CLIP-L / OpenCLIP-bigG / EVA-CLIP text + image towers |
| SAM 2.1 | `sam2` | promptable segmentation (image path: Hiera trunk, FPN neck, two-way mask decoder) |
| ZipDepth | `depth` | 6.1M-param monocular depth network |
| Face recognition | `facenet` | SCRFD detector + alignment + ArcFace embedding |
| WorldMirror-2 | `mirror` | multi-view 3D reconstruction |
| 3D Gaussian Splatting | `splat` | from-scratch tiled 3DGS rasterizer + fit + viewer |
| Chronos-2 | `chronos2` | encoder-only T5-style patch transformer forecaster |
| Kronos | `kronos` | BSQ-tokenized OHLCV forecaster |
| FinCast | `fincast` | TimesFM-style patched decoder forecaster |
| DIAMOND | `wm-diamond` | EDM diffusion world model (Atari-100k) |
| GenieRedux-G | `wm-genie` | CoinRun ST-transformer world model |

---

## Architecture-evaluation harness (`brain bench`)

A model-agnostic battery for answering *"does this architecture actually learn task
X?"* — each benchmark owns its dataset and scoring; the harness runs it the same way
across architectures.

```bash
brain bench                           # run every registered benchmark, one table
brain bench mqar                      # run a single benchmark
brain bench eval    --arch gpt        # whole battery vs one arch -> results/<arch>-<seed>.json
brain bench scale   --arch gpt        # predictive per-capability scaling (score@2x/@4x)
brain bench advise  results/gpt-0.json  # ranked tuning recommendations
brain bench compare results/*.json    # side-by-side leaderboard
```

Registered benchmarks include **mqar** (multi-query associative recall), the **MAD**
family (recall / fuzzy / noisy / selective-copy / memorize), **toolcall**, and the
algorithmic state-tracking probes **parity**, **mod_add** (grokking), and **dyck**.
Capability axes (recall / copying / memory / state-tracking / compression / arithmetic)
aggregate them into a comparable profile.

---

## Datasets, tokenizers, training

```bash
make data/<name>                      # calculator | reverser | wordcalc | timeseries
                                      #   shakespeare_char | gpt | detect | tts
brain data gen <name> --out data/<name> --n 10000 --seed 0
```

Tokenizers: char-level, GPT-2 BPE, and the Qwen BPE (parses `tokenizer.json`). Datasets
are a simple `train.bin`/`val.bin` (u16 or u32) + `meta.json` layout. Training is shared
across models (`fit`): AdamW + grad clip, cosine LR with warmup, grad accumulation, and
LoRA (frozen base + trainable adapters) for parameter-efficient fine-tuning.

## NPU export (Intel, OpenVINO)

```bash
brain npu export   --weights out/yolo.safetensors --out yolo.onnx
brain npu quantize --weights out/yolo.safetensors --calib data/detect --out yolo.int8.onnx
brain npu check    --onnx yolo.onnx --device NPU
brain npu run      --onnx yolo.onnx --image sample.ppm --device NPU
brain npu bench    --onnx yolo.onnx --device NPU --iters 100
```

`make requirements` installs the Python tooling (OpenVINO, etc.); the Rust engine needs
none of it. OpenVINO is loaded at run time, so `make build`/`make test` stay green
without it installed.

## Streaming runtime + web demo

```bash
# event-driven controller: reads JSONL events on stdin, emits JSONL on stdout
printf '{"event":"user_text","text":"hi"}\n' | brain run --gpt out/gpt.safetensors
printf '{"event":"camera_frame","format":"rgb8","w":128,"h":128,"data":"…"}\n' | brain run --yolo out/yolo.safetensors

make web/dev                          # WebGPU browser demo (Node 18+ and a WebGPU browser)
```

`brain run` is an HFSM controller: `user_text → brain_text_chunk` (streamed one token
per tick), `camera_frame → object_detected`, and `user_synth_request → audio_chunk`.

## Testing & correctness

```bash
make test                             # full suite (BRAIN_DEVICE=cpu to skip GPU; MOE_SKIP_GPU_TESTS=1 too)
make gradcheck                        # finite-difference backprop gate
```

Every model's analytic WGSL gradients are checked against finite differences of its own
forward pass; every external-weight import is gated by a parity test against a reference
dump. The whole suite runs on CPU with no GPU required.

## Invariants

**WGSL is the source of truth** (kernels live only in `crates/kernels/wgsl/`); the engine
is **fp32-only, core-compute-only** (single bind group, `@workgroup_size(64)`, no
atomics/subgroups/f16) so it stays portable to old GPUs and WebGPU; **two backends, one
API** (no per-backend model code — the CPU backend JIT-compiles the same WGSL); and
**backprop is gated by the gradient checker**.

## Kernel catalogue

<!-- BEGIN KERNEL TABLE (generated by scripts/build/gen-kernel-table.py) -->

**367 kernels.** Every column is a field the kernel DECLARES in its own header
(`@what` / `@how` / `@opt` / `@cpu` / `@gpu` / `@npu` / `@quant`) — nothing here is
inferred. `make kernels-table` regenerates; `make kernels-table/check` fails when a
kernel is missing a field, when a declaration contradicts the code, or when this
table has drifted. Edit the kernel's header, never the row.

**opt** is a *structural* rubric — how the work is parallelised, not how fast it
measured. Measured numbers live in [`docs/performance/`](docs/performance/overview.md);
a kernel can be 5/5 structurally and still be a defect at a given shape.

* **5/5** — register-tiled / DP4A / split-K — a thread owns a register block of the output (14 kernels)
* **4/5** — workgroup-cooperative — threads share a reduction through workgroup memory (25 kernels)
* **3/5** — coalesced elementwise or gather — one thread per output, no serial reduction (173 kernels)
* **2/5** — one thread per output with a serial reduction over a Params-bounded axis (109 kernels)
* **1/5** — one thread walking a DEEP (3+ nested) serial reduction — the `conv2d_dx` shape (46 kernels)

**cpu** — `backend-cpu`'s Cranelift JIT splits a kernel body at ONE top-level
`workgroupBarrier()` and no more; with two or more it does not fail cleanly, it
**corrupts memory** ([`docs/lessons.md`](docs/lessons.md) #26), so those are `✗`
(9 kernels) — cross-checked against the barrier count on every run.
`native` marks the 32 kernels with a hand-written AVX2 path that runs instead
of the JIT; `native only` means that path is the *only* way it works there, because
its WGSL has >1 barrier — under `BRAIN_NO_FASTCONV=1` or on a non-AVX2 host it would
fall back to the JIT and corrupt memory. A `✓` says the JIT *can* run it, not that the
selector will choose it: `DeviceCaps::workgroup_reductions` is `false` there, so
cooperative variants are deliberately not selected.

**gpu** — every kernel; `✓ (256)` needs a device whose queried
`DeviceCaps::max_workgroup_size` allows it (256 is the WebGPU floor).
**npu** — the Intel NPU never runs WGSL: it is a whole-graph OpenVINO path fed by an
exported ONNX graph, so `✓` means `crates/npu`'s topology DSL can emit an equivalent
op (153 kernels), not that this file runs there.
**quant** — part of the INT8 path (17 kernels).

| kernel | what it does | how | opt | cpu | gpu | npu | quant |
|---|---|---|---|---|---|---|---|
| [`adamw`](crates/kernels/wgsl/adamw.wgsl) | AdamW update (decoupled weight decay), matching torch.optim.AdamW | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`adaptive_avgpool2d`](crates/kernels/wgsl/adaptive_avgpool2d.wgsl) | torch adaptive_avg_pool2d (forward) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`adaptive_avgpool2d_dx`](crates/kernels/wgsl/adaptive_avgpool2d_dx.wgsl) | adaptive_avg_pool2d backward | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`add`](crates/kernels/wgsl/add.wgsl) | Residual add:  dst[i] += src[i] | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`add2`](crates/kernels/wgsl/add2.wgsl) | Out-of-place add:  out[i] = a[i] + b[i] | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`add_chan_bcast`](crates/kernels/wgsl/add_chan_bcast.wgsl) | Add a per-(image, channel) scalar to a full map, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`add_chan_bcast_dv`](crates/kernels/wgsl/add_chan_bcast_dv.wgsl) | Gradient of add_chan_bcast wrt the per-(image, channel) scalar | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`add_chan_inplace`](crates/kernels/wgsl/add_chan_inplace.wgsl) | In-place per-channel bias over NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`add_index_mask`](crates/kernels/wgsl/add_index_mask.wgsl) | Add the DSA per-(query,key) sparse mask into the MLA attention scores before softmax | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`add_inplace`](crates/kernels/wgsl/add_inplace.wgsl) | In-place elementwise add: out += a (single read_write binding — wgpu usage-scope safe) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`arcface_margin`](crates/kernels/wgsl/arcface_margin.wgsl) | ArcFace additive ANGULAR margin, applied to a cosine-similarity logit table | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`arcface_margin_bwd`](crates/kernels/wgsl/arcface_margin_bwd.wgsl) | Backward of arcface_margin.wgsl w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`argmax_final`](crates/kernels/wgsl/argmax_final.wgsl) | Row-wise argmax, FINAL stage | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`argmax_part`](crates/kernels/wgsl/argmax_part.wgsl) | Row-wise argmax, PARTIAL stage | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`argmax_row`](crates/kernels/wgsl/argmax_row.wgsl) | Row-wise argmax:  out[m] = f32(argmax_n x[m, n])   (tie -> LOWEST n) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attention`](crates/kernels/wgsl/attention.wgsl) | Causal multi-head attention with online (numerically stable) softmax | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`attn_apply`](crates/kernels/wgsl/attn_apply.wgsl) | Attention output: out[b,i,h,d] = sum_{j<=i} probs[b,h,i,j] * v[b,j,h,d] | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`attn_apply_bidir`](crates/kernels/wgsl/attn_apply_bidir.wgsl) | Bidirectional attention output | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_apply_cross`](crates/kernels/wgsl/attn_apply_cross.wgsl) | Cross-attention output: out[b,i,h,d] = sum_{j<T_enc} probs[b,h,i,j] * v[b,j,h,d] | one thread per output element, serial inner reduction | 2/5 | native | ✓ | — | — |
| [`attn_apply_full`](crates/kernels/wgsl/attn_apply_full.wgsl) | Attention output over ALL keys (non-causal), reading v from a SEPARATE value buffer (not a fused qkv) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dbias`](crates/kernels/wgsl/attn_bwd_dbias.wgsl) | Backward w.r.t. the additive score bias for attn_scores_{bidir,causal}_bias | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dk`](crates/kernels/wgsl/attn_bwd_dk.wgsl) | Attention backward, step 4 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dk_bias`](crates/kernels/wgsl/attn_bwd_dk_bias.wgsl) | Backward w.r.t. k for the biased/configurable-scale scores kernels (attn_scores_{bidir,causal}_bias) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dk_bidir`](crates/kernels/wgsl/attn_bwd_dk_bidir.wgsl) | Bidirectional attention backward, step 4 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dk_cross`](crates/kernels/wgsl/attn_bwd_dk_cross.wgsl) | Cross-attention backward, step 4 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dk_cross_acc`](crates/kernels/wgsl/attn_bwd_dk_cross_acc.wgsl) | Accumulating twin of attn_bwd_dk_cross for QUERY-CHUNKED backward | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dq`](crates/kernels/wgsl/attn_bwd_dq.wgsl) | Attention backward, step 3 — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dq_bias`](crates/kernels/wgsl/attn_bwd_dq_bias.wgsl) | Backward w.r.t. q for the biased/configurable-scale scores kernels (attn_scores_{bidir,causal}_bias) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dq_bidir`](crates/kernels/wgsl/attn_bwd_dq_bidir.wgsl) | Bidirectional attention backward, step 3 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dq_cross`](crates/kernels/wgsl/attn_bwd_dq_cross.wgsl) | Cross-attention backward, step 3 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dscores`](crates/kernels/wgsl/attn_bwd_dscores.wgsl) | Attention backward, step 1 — gradient through (probs @ v) and the softmax | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dscores_bidir`](crates/kernels/wgsl/attn_bwd_dscores_bidir.wgsl) | Bidirectional attention backward, step 1 — gradient through (probs @ v) and the softmax | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dscores_cross`](crates/kernels/wgsl/attn_bwd_dscores_cross.wgsl) | Cross-attention backward, step 1 — gradient through (probs @ v) and the softmax | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dv`](crates/kernels/wgsl/attn_bwd_dv.wgsl) | Attention backward, step 2 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dv_bidir`](crates/kernels/wgsl/attn_bwd_dv_bidir.wgsl) | Bidirectional attention backward, step 2 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dv_cross`](crates/kernels/wgsl/attn_bwd_dv_cross.wgsl) | Cross-attention backward, step 2 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_bwd_dv_cross_acc`](crates/kernels/wgsl/attn_bwd_dv_cross_acc.wgsl) | Accumulating twin of attn_bwd_dv_cross for QUERY-CHUNKED backward | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_decode_apply`](crates/kernels/wgsl/attn_decode_apply.wgsl) | Decode-step attention apply | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_decode_scores`](crates/kernels/wgsl/attn_decode_scores.wgsl) | Decode-step attention scores | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_decode_scores_win`](crates/kernels/wgsl/attn_decode_scores_win.wgsl) | Windowed decode-step attention scores | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_prefix_mask`](crates/kernels/wgsl/attn_prefix_mask.wgsl) | Moondream prefix-LM attention mask, added into the scores before softmax | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`attn_scores`](crates/kernels/wgsl/attn_scores.wgsl) | Attention scores (materialised, for training) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_scores_bidir`](crates/kernels/wgsl/attn_scores_bidir.wgsl) | Bidirectional (encoder self-attention) scores (materialised, for training) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_scores_bidir_bias`](crates/kernels/wgsl/attn_scores_bidir_bias.wgsl) | Bidirectional (non-causal) attention scores with an additive per-head bias and a CONFIGURABLE scalar scale — the spatial-attention primitive for GenieRedux's ST transformer | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_scores_causal_bias`](crates/kernels/wgsl/attn_scores_causal_bias.wgsl) | Causal attention scores with an additive per-head bias and a CONFIGURABLE scalar scale — the temporal-attention primitive for GenieRedux's ST transformer | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_scores_cross`](crates/kernels/wgsl/attn_scores_cross.wgsl) | Cross-attention scores (materialised, for training) | one thread per output element, serial inner reduction | 2/5 | native | ✓ | — | — |
| [`attn_scores_full`](crates/kernels/wgsl/attn_scores_full.wgsl) | Full (bidirectional, NON-causal) attention scores with an additive key mask and NO 1/sqrt(head_dim) scaling — the Chronos-2 encoder contract | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_scores_masked`](crates/kernels/wgsl/attn_scores_masked.wgsl) | Attention scores with causal mask AND key-padding mask (no RoPE) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_scores_qk`](crates/kernels/wgsl/attn_scores_qk.wgsl) | Attention scores from SEPARATE q,k buffers, with a configurable scale and an optional causal mask — covers Kronos's two attention modes | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`attn_softmax`](crates/kernels/wgsl/attn_softmax.wgsl) | Row-wise causal softmax over the key axis | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`attn_softmax_bidir`](crates/kernels/wgsl/attn_softmax_bidir.wgsl) | Row-wise bidirectional softmax over the full key axis | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`attn_softmax_cross`](crates/kernels/wgsl/attn_softmax_cross.wgsl) | Row-wise cross-attention softmax over the encoder key axis | one thread per output element, 3 nested serial reductions | 1/5 | native | ✓ | ✓ | — |
| [`attn_softmax_full`](crates/kernels/wgsl/attn_softmax_full.wgsl) | Row-wise FULL (non-causal) softmax over the key axis, padding-safe | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`attn_softmax_masked`](crates/kernels/wgsl/attn_softmax_masked.wgsl) | Row-wise causal softmax over the key axis, padding-safe | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`avgpool2d`](crates/kernels/wgsl/avgpool2d.wgsl) | Adaptive/box average-pool forward, NCHW, arbitrary output size | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`avgpool2d_dx`](crates/kernels/wgsl/avgpool2d_dx.wgsl) | Adaptive/box average-pool INPUT gradient, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`axpy`](crates/kernels/wgsl/axpy.wgsl) | Scaled accumulate:  out[i] = out[i] + s * in[i] | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`bce_logits`](crates/kernels/wgsl/bce_logits.wgsl) | Binary cross-entropy with logits, per (anchor, class), against a soft target t in [0,1] | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`bce_logits_grad`](crates/kernels/wgsl/bce_logits_grad.wgsl) | Gradient of bce_logits w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`bias_add`](crates/kernels/wgsl/bias_add.wgsl) | Add a per-output-feature bias in place | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`bias_grad`](crates/kernels/wgsl/bias_grad.wgsl) | Bias gradient:  dbias[n] += sum_m dy[m,n] | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`bn_dbeta`](crates/kernels/wgsl/bn_dbeta.wgsl) | BatchNorm backward w.r.t. beta | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`bn_dgamma`](crates/kernels/wgsl/bn_dgamma.wgsl) | BatchNorm backward w.r.t. gamma | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`bn_dstats`](crates/kernels/wgsl/bn_dstats.wgsl) | BatchNorm backward: per-channel reduced sums for the input-grad formula | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`bn_dx`](crates/kernels/wgsl/bn_dx.wgsl) | BatchNorm backward w.r.t. x | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`bn_eval`](crates/kernels/wgsl/bn_eval.wgsl) | BatchNorm forward for INFERENCE using RUNNING statistics, NCHW x[N,C,H,W] | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`bn_running`](crates/kernels/wgsl/bn_running.wgsl) | BatchNorm momentum update of running statistics | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`bn_stats`](crates/kernels/wgsl/bn_stats.wgsl) | BatchNorm training batch statistics for an NCHW tensor x[N,C,H,W] | one thread per output element, 6 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`bn_train`](crates/kernels/wgsl/bn_train.wgsl) | BatchNorm forward using BATCH statistics, NCHW tensor x[N,C,H,W] | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`broadcast_add_hw`](crates/kernels/wgsl/broadcast_add_hw.wgsl) | Broadcast-add a row-strip and a column-strip into a full map, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`broadcast_add_hw_da`](crates/kernels/wgsl/broadcast_add_hw_da.wgsl) | Gradient of broadcast_add_hw wrt ONE strip | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`bsq_quantize`](crates/kernels/wgsl/bsq_quantize.wgsl) | Binary Spherical Quantization (Kronos), inference form | one thread per output element | 3/5 | ✓ | ✓ | — | int8 |
| [`ce_grad`](crates/kernels/wgsl/ce_grad.wgsl) | Cross-entropy gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`ce_grad_masked`](crates/kernels/wgsl/ce_grad_masked.wgsl) | Cross-entropy gradient over U_BINS with ignore_index, normalised by the count of non-ignored positions (passed in as a float) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`ce_grad_stats`](crates/kernels/wgsl/ce_grad_stats.wgsl) | Cross-entropy gradient using precomputed per-row softmax stats (see ce_stats) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`ce_stats`](crates/kernels/wgsl/ce_stats.wgsl) | Per-row softmax statistics for the cross-entropy backward | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`ce_value`](crates/kernels/wgsl/ce_value.wgsl) | Per-position cross-entropy loss (for logging) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`ce_value_masked`](crates/kernels/wgsl/ce_value_masked.wgsl) | Per-position cross-entropy over U_BINS, with ignore_index (sentinel IGNORE) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`chan_place`](crates/kernels/wgsl/chan_place.wgsl) | Place a tensor into a contiguous channel range of a larger NCHW tensor | one thread per output element | 3/5 | native | ✓ | — | — |
| [`ciou`](crates/kernels/wgsl/ciou.wgsl) | CIoU loss value per assigned anchor | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`ciou_grad`](crates/kernels/wgsl/ciou_grad.wgsl) | Gradient of the CIoU loss L = 1 - CIoU w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`clip_coef`](crates/kernels/wgsl/clip_coef.wgsl) | Global grad-norm clip coefficient, computed on-device (no host round-trip) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`clip_coef_wg`](crates/kernels/wgsl/clip_coef_wg.wgsl) | Global grad-norm clip coefficient from `gradnorm_part`'s partial sums — the cooperative counterpart of `clip_coef.wgsl` | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | — | — |
| [`col2im`](crates/kernels/wgsl/col2im.wgsl) | col2im as a GATHER — the input gradient of a conv, given the gradient of its im2col matrix | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`concat2`](crates/kernels/wgsl/concat2.wgsl) | Concatenate two NCHW tensors along the channel axis | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`concat_split`](crates/kernels/wgsl/concat_split.wgsl) | Concat backward / channel-slice copy | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`conv1d`](crates/kernels/wgsl/conv1d.wgsl) | 1D convolution forward (bias-free), NCL layout, with grouping + dilation | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`conv1d_dw`](crates/kernels/wgsl/conv1d_dw.wgsl) | 1D convolution weight gradient | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`conv1d_dx`](crates/kernels/wgsl/conv1d_dx.wgsl) | 1D convolution input gradient (gather form, no scatter / no atomics) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`conv2d`](crates/kernels/wgsl/conv2d.wgsl) | 2D convolution forward (bias-free), NCHW layout, square KxK kernel | one thread per output element, 3 nested serial reductions | 1/5 | native | ✓ | ✓ | — |
| [`conv2d_dw`](crates/kernels/wgsl/conv2d_dw.wgsl) | 2D convolution weight gradient | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`conv2d_dx`](crates/kernels/wgsl/conv2d_dx.wgsl) | 2D convolution input gradient (transposed-conv GATHER form, no scatter) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`conv2d_gd`](crates/kernels/wgsl/conv2d_gd.wgsl) | 2D convolution forward (bias-free), NCHW, square KxK, WITH grouping + dilation | one thread per output element, 3 nested serial reductions | 1/5 | native | ✓ | ✓ | — |
| [`conv2d_gd_dw`](crates/kernels/wgsl/conv2d_gd_dw.wgsl) | Grouped+dilated 2D convolution WEIGHT gradient | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`conv2d_gd_dx`](crates/kernels/wgsl/conv2d_gd_dx.wgsl) | Grouped+dilated 2D convolution INPUT gradient (transposed-conv GATHER form) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`conv2d_gd_reg`](crates/kernels/wgsl/conv2d_gd_reg.wgsl) | Register-tiled GROUPED/DILATED conv2d forward (bias-free) — conv2d_gd's math with conv_act_reg's 8x4 register tile | register block per thread | 5/5 | native | ✓ | ✓ | — |
| [`conv2d_tiled`](crates/kernels/wgsl/conv2d_tiled.wgsl) | Weight-staged conv2d: identical math to conv2d.wgsl, but one workgroup loads its output channel's weights into WORKGROUP (on-chip) memory once and reuses them across a block of output spatial positions — so each weight is read from global memory once per block instead of once per output pixel (the H*W weight re-read that makes the naive kernel memory-bound on a GPU) | 64-thread workgroup tile, 2 barriers | 4/5 | native | ✓ | ✓ | — |
| [`conv_act`](crates/kernels/wgsl/conv_act.wgsl) | Fused conv2d -> per-channel affine (BatchNorm-eval collapsed) -> activation | one thread per output element, 3 nested serial reductions | 1/5 | native | ✓ | ✓ | — |
| [`conv_act_reg`](crates/kernels/wgsl/conv_act_reg.wgsl) | Register-tiled fused conv -> per-channel affine -> activation | register block per thread | 5/5 | native | ✓ | ✓ | — |
| [`conv_act_tiled`](crates/kernels/wgsl/conv_act_tiled.wgsl) | Weight-staged fused conv -> per-channel affine -> activation | 64-thread workgroup tile, 1 barrier | 4/5 | native | ✓ | ✓ | — |
| [`conv_bias`](crates/kernels/wgsl/conv_bias.wgsl) | Fused conv2d + per-output-channel bias | one thread per output element, 3 nested serial reductions | 1/5 | native | ✓ | ✓ | — |
| [`conv_bias_reg`](crates/kernels/wgsl/conv_bias_reg.wgsl) | Register-tiled conv + per-output-channel bias — conv_act_reg's 8x4 tile (8 output channels x 4 coalesced spatial positions in scalar registers, (kh,kw)-outer/ci-inner loop) with a PLAIN BIAS epilogue instead of the BN-affine+SiLU one | register block per thread | 5/5 | native | ✓ | ✓ | — |
| [`conv_epilogue`](crates/kernels/wgsl/conv_epilogue.wgsl) | Conv-as-GEMM epilogue: per-channel affine (BN-eval collapsed) + activation | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`convex_upsample`](crates/kernels/wgsl/convex_upsample.wgsl) | Convex 3x3 upsample forward (ZipDepth's FastConvexUpsample, unfold path) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`convex_upsample_dd`](crates/kernels/wgsl/convex_upsample_dd.wgsl) | Convex 3x3 upsample: gradient wrt the half-res DEPTH map | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`convex_upsample_dmask`](crates/kernels/wgsl/convex_upsample_dmask.wgsl) | Convex 3x3 upsample: gradient wrt the MASK | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`convtr1d`](crates/kernels/wgsl/convtr1d.wgsl) | Transposed 1D convolution forward (ConvTranspose1d), NCL layout, grouping + dilation | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`convtr1d_dw`](crates/kernels/wgsl/convtr1d_dw.wgsl) | Transposed 1D convolution weight gradient | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`convtr1d_dx`](crates/kernels/wgsl/convtr1d_dx.wgsl) | Transposed 1D convolution input gradient (gather form) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`convtr2d`](crates/kernels/wgsl/convtr2d.wgsl) | Transposed 2D convolution forward (ConvTranspose2d, bias-free), NCHW, square KxK, WITH grouping + dilation | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`convtr2d_dw`](crates/kernels/wgsl/convtr2d_dw.wgsl) | Transposed 2D convolution WEIGHT gradient | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`convtr2d_dx`](crates/kernels/wgsl/convtr2d_dx.wgsl) | Transposed 2D convolution INPUT gradient (gather form, no scatter/atomics) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`crop2d`](crates/kernels/wgsl/crop2d.wgsl) | Asymmetric crop on NCHW (gather) — the exact adjoint of pad2d | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`decode_advance`](crates/kernels/wgsl/decode_advance.wgsl) | Advance the batched-decode metadata one sub-step, on the device (A4) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`decode_feed`](crates/kernels/wgsl/decode_feed.wgsl) | Feed the greedy head's output back as the next decode step's input, on the device (A4) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`decode_softmax`](crates/kernels/wgsl/decode_softmax.wgsl) | Decode-step softmax: max-subtracted softmax over the `t` cached scores of each query head, in place per row of a [n_heads, cap]-strided buffer | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`decode_softmax_batched`](crates/kernels/wgsl/decode_softmax_batched.wgsl) | Batched decode softmax: per (sequence b, head h), max-subtracted softmax over its seq_lens[b] scores in a [batch, n_heads, cap]-strided buffer | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`dfl_decode`](crates/kernels/wgsl/dfl_decode.wgsl) | DFL decode: for each (anchor, side) softmax over `reg_max` logits then take the expectation E = sum_i i * p_i | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`dfl_grad`](crates/kernels/wgsl/dfl_grad.wgsl) | DFL decode gradient. Given upstream dE[A,4] = dL/dE for each expected distance E, produce logit grads | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`dfl_loss`](crates/kernels/wgsl/dfl_loss.wgsl) | Distribution Focal Loss value per assigned anchor | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`dfl_loss_grad`](crates/kernels/wgsl/dfl_loss_grad.wgsl) | Gradient of dfl_loss w.r.t | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`dw_splitk_reduce`](crates/kernels/wgsl/dw_splitk_reduce.wgsl) | Fold `matmul_dw_reg_splitk`'s per-slice partials into the weight gradient | one thread per output element, serial inner reduction | 5/5 | ✓ | ✓ | — | — |
| [`dwconv3d`](crates/kernels/wgsl/dwconv3d.wgsl) | Depthwise 3D convolution (forward) — the PEG (position-encoding generator) of ST-ViViT tokenizers | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`dwconv3d_dw`](crates/kernels/wgsl/dwconv3d_dw.wgsl) | Depthwise 3D convolution, WEIGHT gradient | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`dwconv3d_dx`](crates/kernels/wgsl/dwconv3d_dx.wgsl) | Depthwise 3D convolution, INPUT gradient (adjoint of dwconv3d) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`edm_mix`](crates/kernels/wgsl/edm_mix.wgsl) | EDM output mix D = c_skip*x + c_out*F — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`edm_wrap`](crates/kernels/wgsl/edm_wrap.wgsl) | EDM output wrap for the DIAMOND sampler (denoiser.py::wrap_model_output) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`emb_bwd`](crates/kernels/wgsl/emb_bwd.wgsl) | Embedding backward (also the tied lm_head's weight) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`embed`](crates/kernels/wgsl/embed.wgsl) | Embedding gather: x[t, c] = emb[token[t], c] | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`embed_tile`](crates/kernels/wgsl/embed_tile.wgsl) | Embedding gather over a VOCAB TILE | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`expert_counts`](crates/kernels/wgsl/expert_counts.wgsl) | Load-balancing fractions used by the aux-loss gradient | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`film_chan`](crates/kernels/wgsl/film_chan.wgsl) | FiLM per-channel modulation (forward) for NCHW — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`film_chan_dsb`](crates/kernels/wgsl/film_chan_dsb.wgsl) | FiLM per-channel modulation, scale/shift gradient — spec | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`film_chan_dx`](crates/kernels/wgsl/film_chan_dx.wgsl) | FiLM per-channel modulation, input gradient — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`film_row`](crates/kernels/wgsl/film_row.wgsl) | FiLM per-row-group modulation (forward) for [R,D] rows — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`film_row_dsb`](crates/kernels/wgsl/film_row_dsb.wgsl) | FiLM per-row-group modulation, scale/shift gradient — spec | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`film_row_dx`](crates/kernels/wgsl/film_row_dx.wgsl) | FiLM per-row-group modulation, input gradient — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`flash_attn_bidir`](crates/kernels/wgsl/flash_attn_bidir.wgsl) | Flash attention (bidirectional self-attention), TILED with shared-memory K/V reuse + online softmax — Pascal-friendly (sm_61 | 64-thread workgroup tile, 3 barriers | 4/5 | ✗ | ✓ | — | — |
| [`flash_attn_bidir_split`](crates/kernels/wgsl/flash_attn_bidir_split.wgsl) | Flash attention (bidirectional self-attention), LANE-SPLIT across head_dim | 256-thread workgroup tile, 3 barriers | 4/5 | ✗ | ✓ (256) | — | — |
| [`focal_dice_grad`](crates/kernels/wgsl/focal_dice_grad.wgsl) | Gradient of the SAM-style segmentation objective w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`focal_dice_stats`](crates/kernels/wgsl/focal_dice_stats.wgsl) | Per-mask reductions for the SAM-style segmentation objective (sigmoid focal loss + dice loss), over `n_masks` masks of `hw` pixels each (row-major [n_masks, hw]) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`gate_row`](crates/kernels/wgsl/gate_row.wgsl) | adaLN gated residual merge (forward) for [R,D] rows — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`gate_row_dg`](crates/kernels/wgsl/gate_row_dg.wgsl) | adaLN gated residual, gate gradient — spec | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`gate_row_dh`](crates/kernels/wgsl/gate_row_dh.wgsl) | adaLN gated residual, branch gradient — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`geglu_shift`](crates/kernels/wgsl/geglu_shift.wgsl) | Moondream MoE expert activation — GeGLU with a +1 shift | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`geglu_shift_da`](crates/kernels/wgsl/geglu_shift_da.wgsl) | geglu_shift backward w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`geglu_shift_db`](crates/kernels/wgsl/geglu_shift_db.wgsl) | geglu_shift backward w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`gelu`](crates/kernels/wgsl/gelu.wgsl) | GELU activation (tanh approximation, as used by GPT-2-style MLPs) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`gelu_bwd`](crates/kernels/wgsl/gelu_bwd.wgsl) | GELU backward (tanh approximation) — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`gelu_erf`](crates/kernels/wgsl/gelu_erf.wgsl) | Exact (erf-based) GELU, matching torch's default `F.gelu` | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`gelu_erf_bwd`](crates/kernels/wgsl/gelu_erf_bwd.wgsl) | Exact (erf-based) GELU backward — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`glu`](crates/kernels/wgsl/glu.wgsl) | Gated Linear Unit over the middle dim, matching torch `F.glu(x, dim=1)` | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`glu_bwd`](crates/kernels/wgsl/glu_bwd.wgsl) | Backward of glu.wgsl (F.glu over the middle dim) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`gn_apply`](crates/kernels/wgsl/gn_apply.wgsl) | GroupNorm forward apply, NCHW x[N,C,H,W] — spec | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`gn_dbeta`](crates/kernels/wgsl/gn_dbeta.wgsl) | GroupNorm backward w.r.t. beta, NCHW — spec | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`gn_dgamma`](crates/kernels/wgsl/gn_dgamma.wgsl) | GroupNorm backward w.r.t. gamma, NCHW — spec | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`gn_dgb2`](crates/kernels/wgsl/gn_dgb2.wgsl) | GroupNorm affine gradients, STAGE 2 of 2 — fold the partials and ACCUMULATE | one thread per output element, serial inner reduction | 4/5 | ✓ | ✓ | ✓ | — |
| [`gn_dgb_part`](crates/kernels/wgsl/gn_dgb_part.wgsl) | GroupNorm affine gradients, STAGE 1 of 2 — partial sums for BOTH dgamma and dbeta in one pass over `dy` | one thread per output element, serial inner reduction | 4/5 | ✓ | ✓ | ✓ | — |
| [`gn_dsum`](crates/kernels/wgsl/gn_dsum.wgsl) | GroupNorm backward per-group reductions, NCHW — spec | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`gn_dsum2`](crates/kernels/wgsl/gn_dsum2.wgsl) | GroupNorm backward per-group reductions, STAGE 2 of 2 — fold the partials | one thread per output element, serial inner reduction | 4/5 | ✓ | ✓ | ✓ | — |
| [`gn_dsum_part`](crates/kernels/wgsl/gn_dsum_part.wgsl) | GroupNorm backward per-group reductions, STAGE 1 of 2 — partial sums | 1 barrier | 3/5 | ✓ | ✓ | ✓ | — |
| [`gn_dx`](crates/kernels/wgsl/gn_dx.wgsl) | GroupNorm backward w.r.t. x, NCHW — spec | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`gn_part`](crates/kernels/wgsl/gn_part.wgsl) | GroupNorm partial reduction (stage 1 of 2) — the parallel replacement for gn_stats' serial per-group loop on wide GPUs | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`gn_stats`](crates/kernels/wgsl/gn_stats.wgsl) | GroupNorm statistics for an NCHW tensor x[N,C,H,W] — spec | one thread per output element, 4 nested serial reductions | 1/5 | native | ✓ | ✓ | — |
| [`gn_stats2`](crates/kernels/wgsl/gn_stats2.wgsl) | GroupNorm statistics combine (stage 2 of 2, after gn_part) | one thread per output element, serial inner reduction | 4/5 | native | ✓ | ✓ | — |
| [`gn_stats_wg`](crates/kernels/wgsl/gn_stats_wg.wgsl) | GroupNorm statistics, one WORKGROUP per (n,g) group — the parallel, COALESCED twin of gn_stats.wgsl | 256-thread workgroup tile, 3 barriers | 4/5 | ✗ | ✓ (256) | ✓ | — |
| [`gqa_apply`](crates/kernels/wgsl/gqa_apply.wgsl) | GQA attention output, separate v buffer | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`gqa_bwd_dk`](crates/kernels/wgsl/gqa_bwd_dk.wgsl) | GQA attention backward, step 4 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`gqa_bwd_dq`](crates/kernels/wgsl/gqa_bwd_dq.wgsl) | GQA attention backward, step 3 — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`gqa_bwd_dscores`](crates/kernels/wgsl/gqa_bwd_dscores.wgsl) | GQA attention backward, step 1 — gradient through (probs @ v) and softmax | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`gqa_bwd_dv`](crates/kernels/wgsl/gqa_bwd_dv.wgsl) | GQA attention backward, step 2 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`gqa_scores`](crates/kernels/wgsl/gqa_scores.wgsl) | GQA attention scores (materialised, for training), separate q/k buffers | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`gqa_scores_kmask`](crates/kernels/wgsl/gqa_scores_kmask.wgsl) | GQA attention scores with an additive per-key mask — `gqa_scores` plus `kmask[j]` added to every finite score | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`grad_scale`](crates/kernels/wgsl/grad_scale.wgsl) | Scale a gradient buffer in place | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`grad_scale_buf`](crates/kernels/wgsl/grad_scale_buf.wgsl) | Scale a gradient buffer in place by a coefficient that lives in a GPU buffer (written by clip_coef) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`gradnorm_part`](crates/kernels/wgsl/gradnorm_part.wgsl) | Per-parameter sum of squares of its gradient, as a COOPERATIVE tree reduction | 64-thread workgroup tile, 2 barriers | 4/5 | ✓ | ✓ | — | — |
| [`gradnorm_sq`](crates/kernels/wgsl/gradnorm_sq.wgsl) | Per-parameter sum of squares of its gradient, written to norms[slot] | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`grid_sample`](crates/kernels/wgsl/grid_sample.wgsl) | Bilinear grid sample forward — `torch.nn.functional.grid_sample(mode='bilinear', padding_mode='zeros')` | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`grid_sample_dgrid`](crates/kernels/wgsl/grid_sample_dgrid.wgsl) | Bilinear grid sample GRID gradient — the other half of grid_sample.wgsl's backward, w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`grid_sample_dx`](crates/kernels/wgsl/grid_sample_dx.wgsl) | Bilinear grid sample INPUT gradient — the adjoint of grid_sample.wgsl w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`head_pack`](crates/kernels/wgsl/head_pack.wgsl) | Pack one attention operand head-major-contiguous for GEMM attention | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`head_pack_t`](crates/kernels/wgsl/head_pack_t.wgsl) | As head_pack but TRANSPOSED per head | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`head_unpack`](crates/kernels/wgsl/head_unpack.wgsl) | Inverse of head_pack: scatter per-head [rows, hd] context blocks back into the row-major [rows, d_model] stream the output projection consumes | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`im2col`](crates/kernels/wgsl/im2col.wgsl) | im2col: lower a conv input into a GEMM operand | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`im2col_at`](crates/kernels/wgsl/im2col_at.wgsl) | im2col over a RANGE of output positions — `im2col.wgsl` with a `[pos0, pos0+cnt)` window, so a conv can be lowered to a GEMM in spatial chunks | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`kv_append`](crates/kernels/wgsl/kv_append.wgsl) | Append one token's projected K (or V) into a KV cache | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`kv_expand`](crates/kernels/wgsl/kv_expand.wgsl) | GQA head expansion into a fused attention buffer (LFM2.5 bidirectional path) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`kv_expand_bwd`](crates/kernels/wgsl/kv_expand_bwd.wgsl) | Backward of kv_expand — the adjoint of head replication is a group-sum | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`l2norm_scale`](crates/kernels/wgsl/l2norm_scale.wgsl) | Per-row L2 normalization with a learnable per-dim scale — the QK-norm used by GenieRedux attention (applied to each head slice of q and k, over head_dim, before the scores kernel; the scores kernel then uses a constant scale of 8) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`l2norm_scale_dg`](crates/kernels/wgsl/l2norm_scale_dg.wgsl) | Backward w.r.t. the per-dim scale g for l2norm_scale | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`l2norm_scale_dx`](crates/kernels/wgsl/l2norm_scale_dx.wgsl) | Backward w.r.t. x for l2norm_scale | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`layernorm`](crates/kernels/wgsl/layernorm.wgsl) | LayerNorm forward (matches torch.nn.LayerNorm) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`layernorm2d`](crates/kernels/wgsl/layernorm2d.wgsl) | Channels-first LayerNorm (ConvNeXt / SAM 2's `LayerNorm2d`), FUSED — the normalisation runs in NCHW, with no permute either side | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`layernorm_dbeta`](crates/kernels/wgsl/layernorm_dbeta.wgsl) | LayerNorm backward w.r.t. beta | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`layernorm_dgamma`](crates/kernels/wgsl/layernorm_dgamma.wgsl) | LayerNorm backward w.r.t. gamma | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`layernorm_dx`](crates/kernels/wgsl/layernorm_dx.wgsl) | LayerNorm backward w.r.t. x | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`layernorm_dx_rows`](crates/kernels/wgsl/layernorm_dx_rows.wgsl) | LayerNorm backward w.r.t. x, one WORKGROUP per row — the coalesced variant | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | ✓ | — |
| [`layernorm_rows`](crates/kernels/wgsl/layernorm_rows.wgsl) | LayerNorm forward, one WORKGROUP per row — the coalesced variant | 64-thread workgroup tile, 2 barriers | 4/5 | ✓ | ✓ | ✓ | — |
| [`leaky_relu`](crates/kernels/wgsl/leaky_relu.wgsl) | Leaky ReLU forward:  y = x        if x >= 0 y = slope*x  otherwise | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`leaky_relu_bwd`](crates/kernels/wgsl/leaky_relu_bwd.wgsl) | Leaky ReLU backward — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`ln_head`](crates/kernels/wgsl/ln_head.wgsl) | Strided per-head LayerNorm (QK-norm) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`ln_head_dgb`](crates/kernels/wgsl/ln_head_dgb.wgsl) | Strided per-head LayerNorm backward (parameter grads) | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`ln_head_dx`](crates/kernels/wgsl/ln_head_dx.wgsl) | Strided per-head LayerNorm backward (input grad), the ln_head companion | one thread per output element, 4 nested serial reductions | 1/5 | ✓ | ✓ | — | — |
| [`ln_stats`](crates/kernels/wgsl/ln_stats.wgsl) | LayerNorm helper: per-row mean and inverse-std | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`ln_stats_rows`](crates/kernels/wgsl/ln_stats_rows.wgsl) | LayerNorm per-row mean + inverse-std, one WORKGROUP per row | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | — | — |
| [`lstm_gates`](crates/kernels/wgsl/lstm_gates.wgsl) | Fused LSTM cell gate activation (PyTorch nn.LSTM layout) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`lstm_gates_bwd`](crates/kernels/wgsl/lstm_gates_bwd.wgsl) | Backward of lstm_gates.wgsl | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`masked_l1`](crates/kernels/wgsl/masked_l1.wgsl) | Masked L1, per element:  out = /pred - tgt/ * mask | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`masked_l1_grad`](crates/kernels/wgsl/masked_l1_grad.wgsl) | Masked L1 gradient:  d_pred = sign(pred - tgt) * mask * scale | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`matmul`](crates/kernels/wgsl/matmul.wgsl) | Generic matmul matching PyTorch nn.Linear (no bias) | one thread per output element, serial inner reduction | 2/5 | native | ✓ | ✓ | — |
| [`matmul_dw`](crates/kernels/wgsl/matmul_dw.wgsl) | Backward of  out = x @ W^T  w.r.t | one thread per output element, serial inner reduction | 2/5 | native | ✓ | ✓ | — |
| [`matmul_dw_reg`](crates/kernels/wgsl/matmul_dw_reg.wgsl) | Backward of out = x @ W^T w.r.t | register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | native only | ✓ (256) | ✓ | — |
| [`matmul_dw_reg_splitk`](crates/kernels/wgsl/matmul_dw_reg_splitk.wgsl) | Split-K backward of out = x @ W^T w.r.t | register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | ✗ | ✓ (256) | ✓ | — |
| [`matmul_dw_reg_tn`](crates/kernels/wgsl/matmul_dw_reg_tn.wgsl) | Backward of out = x @ W^T w.r.t | register block per thread, 256-thread workgroup tile, 3 barriers | 4/5 | ✗ | ✓ (256) | ✓ | — |
| [`matmul_dx`](crates/kernels/wgsl/matmul_dx.wgsl) | Backward of  out = x @ W^T  w.r.t | one thread per output element, serial inner reduction | 2/5 | native | ✓ | ✓ | — |
| [`matmul_dx_reg`](crates/kernels/wgsl/matmul_dx_reg.wgsl) | Backward of out = x @ W^T w.r.t | register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | native only | ✓ (256) | ✓ | — |
| [`matmul_gemv`](crates/kernels/wgsl/matmul_gemv.wgsl) | Skinny-M matmul (out = x @ W^T), one WORKGROUP per output COLUMN — the decode-regime GEMM | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | ✓ | — |
| [`matmul_i8`](crates/kernels/wgsl/matmul_i8.wgsl) | INT8 register-tiled + software-pipelined GEMM via DP4A (dot4I8Packed) | DP4A packed int8, register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | ✗ | ✓ (256) | ✓ | int8 |
| [`matmul_i8_dyn`](crates/kernels/wgsl/matmul_i8_dyn.wgsl) | matmul_i8 with a DYNAMIC per-tensor activation scale (sx from a buffer, sw a | DP4A packed int8, register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | ✗ | ✓ (256) | ✓ | int8 |
| [`matmul_i8_gemv`](crates/kernels/wgsl/matmul_i8_gemv.wgsl) | Skinny-M INT8 matmul (out = dequant(x_q @ W_qᵀ)), one WORKGROUP per output COLUMN — the decode-regime int8 GEMM | DP4A packed int8, 64-thread workgroup tile, 1 barrier | 5/5 | ✓ | ✓ | ✓ | int8 |
| [`matmul_reg`](crates/kernels/wgsl/matmul_reg.wgsl) | Register-tiled matmul — same math as matmul.wgsl (out = x @ Wᵀ), sized for a COMPUTE-bound discrete GPU | register block per thread, 256-thread workgroup tile, 2 barriers | 5/5 | native only | ✓ (256) | ✓ | — |
| [`matmul_reg2`](crates/kernels/wgsl/matmul_reg2.wgsl) | Software-pipelined register-tiled matmul (out = x @ Wᵀ) | register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | native only | ✓ (256) | ✓ | — |
| [`matmul_reg3`](crates/kernels/wgsl/matmul_reg3.wgsl) | Register-tiled matmul (out = x @ Wᵀ), matmul_reg2's tiling with its two shared-memory bank-conflict patterns removed | register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | native only | ✓ (256) | ✓ | — |
| [`matmul_reg3_splitk`](crates/kernels/wgsl/matmul_reg3_splitk.wgsl) | Split-K register-tiled matmul — matmul_reg3 for skinny-M shapes | register block per thread, 256-thread workgroup tile, 3 barriers | 5/5 | ✗ | ✓ (256) | ✓ | — |
| [`matmul_rows`](crates/kernels/wgsl/matmul_rows.wgsl) | Row-blocked matmul, same contract as matmul.wgsl | one thread per output element, serial inner reduction | 4/5 | ✓ | ✓ | ✓ | — |
| [`matmul_tile`](crates/kernels/wgsl/matmul_tile.wgsl) | Matmul into a COLUMN TILE of the output | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`matmul_tiled`](crates/kernels/wgsl/matmul_tiled.wgsl) | Tiled matmul — same math as matmul.wgsl (out = x @ W^T) but GPU-parallelised | 64-thread workgroup tile, 2 barriers | 4/5 | native only | ✓ | ✓ | — |
| [`max_abs_final`](crates/kernels/wgsl/max_abs_final.wgsl) | Reduce the partial maxima from max_abs_part into the int8 quantization scale | one thread per output element | 3/5 | ✓ | ✓ | — | int8 |
| [`max_abs_part`](crates/kernels/wgsl/max_abs_part.wgsl) | Partial per-tensor max/x/ for dynamic int8 quantization | one thread per output element | 3/5 | ✓ | ✓ | — | int8 |
| [`max_abs_row`](crates/kernels/wgsl/max_abs_row.wgsl) | Per-ROW (per-token) max/x/ → int8 scale, for outlier-robust activation quantization | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`max_abs_rows`](crates/kernels/wgsl/max_abs_rows.wgsl) | Per-ROW (per-token) max/x/ -> int8 scale, one WORKGROUP per row — the cooperative form of `max_abs_row.wgsl` | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | — | int8 |
| [`maxpool2d`](crates/kernels/wgsl/maxpool2d.wgsl) | Generic KxK max-pool forward, NCHW, arbitrary STRIDE + symmetric zero-pad | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`maxpool2d_dx`](crates/kernels/wgsl/maxpool2d_dx.wgsl) | Generic KxK max-pool INPUT gradient, NCHW, arbitrary STRIDE + symmetric pad | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`mla_bwd_dk_pass`](crates/kernels/wgsl/mla_bwd_dk_pass.wgsl) | MLA backward — grad w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`mla_bwd_dk_rope`](crates/kernels/wgsl/mla_bwd_dk_rope.wgsl) | MLA backward — grad w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`mla_bwd_dq_pass`](crates/kernels/wgsl/mla_bwd_dq_pass.wgsl) | MLA backward — grad w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`mla_bwd_dq_rope`](crates/kernels/wgsl/mla_bwd_dq_rope.wgsl) | MLA backward — grad w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`mla_index_scores`](crates/kernels/wgsl/mla_index_scores.wgsl) | DSA indexer scores (forward, detached) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`mla_scores`](crates/kernels/wgsl/mla_scores.wgsl) | MLA (Multi-head Latent Attention) scores (forward), for GLM-5.2 | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`moe_linear_gated`](crates/kernels/wgsl/moe_linear_gated.wgsl) | Sparse-MoE expert linear: matmul.wgsl, but skips non-routed rows | one thread per output element, serial inner reduction, early exit | 2/5 | ✓ | ✓ | — | — |
| [`moe_linear_gated_dw`](crates/kernels/wgsl/moe_linear_gated_dw.wgsl) | Sparse-MoE expert linear backward w.r.t. W: matmul_dw.wgsl, gated | one thread per output element, in-loop skip on the gate | 2/5 | ✓ | ✓ | — | — |
| [`moe_linear_gated_dx`](crates/kernels/wgsl/moe_linear_gated_dx.wgsl) | Sparse-MoE expert linear backward w.r.t. x: matmul_dx.wgsl, gated | one thread per output element, pre-reduction early exit on the gate | 2/5 | ✓ | ✓ | — | — |
| [`moe_linear_gated_i8`](crates/kernels/wgsl/moe_linear_gated_i8.wgsl) | Sparse-MoE expert linear, int8 (DP4A): moe_linear_gated.wgsl's row skip, packed weights | DP4A packed int8, one thread per output element, serial inner reduction, early exit | 2/5 | ✓ | ✓ | — | int8 |
| [`moe_scatter_scaled_add`](crates/kernels/wgsl/moe_scatter_scaled_add.wgsl) | Scatter a compacted expert's output back into the dense MoE accumulator, gate-scaled | one thread per output element, scatter-add (no atomics: rows are unique per call) | 3/5 | ✓ | ✓ | — | — |
| [`mse_grad`](crates/kernels/wgsl/mse_grad.wgsl) | Mean-squared-error gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`mse_grad_w`](crates/kernels/wgsl/mse_grad_w.wgsl) | Per-sample weighted MSE gradient — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`mse_value`](crates/kernels/wgsl/mse_value.wgsl) | Mean-squared-error loss value (the Regression-head analogue of ce_value) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`mse_value_w`](crates/kernels/wgsl/mse_value_w.wgsl) | Per-sample weighted MSE partial sums — spec | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`mul`](crates/kernels/wgsl/mul.wgsl) | Elementwise Hadamard product — spec | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`nchw_nlc`](crates/kernels/wgsl/nchw_nlc.wgsl) | Layout permutation NCHW -> NLC [N, L=H*W, C] (gather) — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`nlc_bias_nchw`](crates/kernels/wgsl/nlc_bias_nchw.wgsl) | NLC -> NCHW with a per-channel bias — the epilogue of a conv lowered to a row-major GEMM | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | — | — |
| [`nlc_nchw`](crates/kernels/wgsl/nlc_nchw.wgsl) | Layout permutation NLC [N, L=H*W, C] -> NCHW (gather) — exact inverse AND adjoint of nchw_nlc | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`pack_qkv`](crates/kernels/wgsl/pack_qkv.wgsl) | Pack three separate [seq, d_model] projections (q, k, v) into one fused [seq, 3*d_model] buffer laid out per token as [ q(d) / k(d) / v(d) ] — the layout the bidirectional attention trio (attn_scores_bidir / _softmax_ / _apply_) reads via q_off=0, k_off=d, v_off=2d | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`pad2d`](crates/kernels/wgsl/pad2d.wgsl) | Asymmetric zero-pad on NCHW (gather) — spec | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`paged_decode_apply`](crates/kernels/wgsl/paged_decode_apply.wgsl) | Paged decode-step attention apply | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`paged_decode_apply_batched`](crates/kernels/wgsl/paged_decode_apply_batched.wgsl) | Batched paged decode apply | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`paged_decode_apply_i8_batched`](crates/kernels/wgsl/paged_decode_apply_i8_batched.wgsl) | Batched paged decode apply over an INT8 pool | one thread per output element | 3/5 | ✓ | ✓ | — | int8 |
| [`paged_decode_scores`](crates/kernels/wgsl/paged_decode_scores.wgsl) | Paged decode-step attention scores | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`paged_decode_scores_batched`](crates/kernels/wgsl/paged_decode_scores_batched.wgsl) | Batched paged decode scores | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`paged_decode_scores_i8_batched`](crates/kernels/wgsl/paged_decode_scores_i8_batched.wgsl) | Batched paged decode scores over an INT8 pool | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`paged_decode_scores_wg`](crates/kernels/wgsl/paged_decode_scores_wg.wgsl) | Batched paged decode scores, one WORKGROUP per score — the coalesced variant | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | — | — |
| [`paged_kv_append`](crates/kernels/wgsl/paged_kv_append.wgsl) | Write one token's projected K (or V) into a paged KV block pool at a physical block + offset | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`paged_kv_append_batched`](crates/kernels/wgsl/paged_kv_append_batched.wgsl) | Append a batch of new tokens' K (or V) into the paged pool | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`paged_kv_append_i8_batched`](crates/kernels/wgsl/paged_kv_append_i8_batched.wgsl) | Append a batch of new tokens' K (or V) into an INT8 paged pool | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`paged_kv_append_i8_clipped_batched`](crates/kernels/wgsl/paged_kv_append_i8_clipped_batched.wgsl) | Calibrated twin of paged_kv_append_i8_batched | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`pixel_shuffle`](crates/kernels/wgsl/pixel_shuffle.wgsl) | Pixel shuffle (depth-to-space) forward, NCHW | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`pixel_shuffle_dx`](crates/kernels/wgsl/pixel_shuffle_dx.wgsl) | Pixel shuffle INPUT gradient, NCHW | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`pos_add`](crates/kernels/wgsl/pos_add.wgsl) | Add learned absolute positional embeddings in place | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`pos_bwd`](crates/kernels/wgsl/pos_bwd.wgsl) | Positional-embedding backward (scatter) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`prelu`](crates/kernels/wgsl/prelu.wgsl) | PReLU forward with a LEARNED slope, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`prelu_bwd`](crates/kernels/wgsl/prelu_bwd.wgsl) | PReLU backward, NCHW — produces BOTH gradients in one pass | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | ✓ | — |
| [`prelu_bwd_wg`](crates/kernels/wgsl/prelu_bwd_wg.wgsl) | PReLU backward, one WORKGROUP per CHANNEL — the cooperative, COALESCED twin of prelu_bwd.wgsl | 64-thread workgroup tile, 3 barriers | 4/5 | ✓ | ✓ | ✓ | — |
| [`quant_pack`](crates/kernels/wgsl/quant_pack.wgsl) | Quantize + pack an [M, K] f32 activation into [M, K/4] u32 (4 int8 per u32, little-endian along K) using a dynamic per-tensor scale sx (from a buffer) | one thread per output element | 3/5 | ✓ | ✓ | — | int8 |
| [`quick_gelu`](crates/kernels/wgsl/quick_gelu.wgsl) | QuickGELU activation (OpenAI CLIP's sigmoid approximation) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`quick_gelu_bwd`](crates/kernels/wgsl/quick_gelu_bwd.wgsl) | QuickGELU backward — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`region_copy`](crates/kernels/wgsl/region_copy.wgsl) | Strided-region copy between same-layout buffers | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rel_shift`](crates/kernels/wgsl/rel_shift.wgsl) | Transformer-XL relative-position shift (NeMo / Conformer rel-pos attention) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rel_shift_bwd`](crates/kernels/wgsl/rel_shift_bwd.wgsl) | Backward (transpose) of rel_shift.wgsl | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`relu_inplace`](crates/kernels/wgsl/relu_inplace.wgsl) | In-place ReLU (single read_write binding — wgpu usage-scope safe) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`resize_bicubic`](crates/kernels/wgsl/resize_bicubic.wgsl) | Bicubic resize forward (Catmull-Rom / cubic convolution, a = -0.75), NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`resize_bicubic_dx`](crates/kernels/wgsl/resize_bicubic_dx.wgsl) | Bicubic resize INPUT gradient, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`resize_bilinear`](crates/kernels/wgsl/resize_bilinear.wgsl) | Bilinear resize forward, NCHW, arbitrary output size | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`resize_bilinear_dx`](crates/kernels/wgsl/resize_bilinear_dx.wgsl) | Bilinear resize INPUT gradient, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`resize_nearest`](crates/kernels/wgsl/resize_nearest.wgsl) | Nearest-neighbour resize, NCHW, ARBITRARY output size | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`resize_nearest_dx`](crates/kernels/wgsl/resize_nearest_dx.wgsl) | Nearest-neighbour resize INPUT gradient, NCHW | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`rms_inv`](crates/kernels/wgsl/rms_inv.wgsl) | Helper: per-row inverse RMS,  inv[n] = 1/sqrt(mean_c(x[n,c]^2) + eps) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`rms_inv_eps`](crates/kernels/wgsl/rms_inv_eps.wgsl) | Per-row inverse RMS with a RUNTIME epsilon | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`rmsnorm`](crates/kernels/wgsl/rmsnorm.wgsl) | RMSNorm:  out[t, c] = weight[c] * x[t, c] / sqrt(mean_c(x[t, c]^2) + eps) One invocation per row (token) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`rmsnorm_dw`](crates/kernels/wgsl/rmsnorm_dw.wgsl) | RMSNorm backward w.r.t. the gain weight | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`rmsnorm_dx`](crates/kernels/wgsl/rmsnorm_dx.wgsl) | RMSNorm backward w.r.t. the input x | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`rmsnorm_dx_eps`](crates/kernels/wgsl/rmsnorm_dx_eps.wgsl) | RMSNorm backward w.r.t. x, with a RUNTIME epsilon (eps-parameterized twin of rmsnorm_dx, which hardcodes 1e-6) | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`rmsnorm_eps`](crates/kernels/wgsl/rmsnorm_eps.wgsl) | RMSNorm with a RUNTIME epsilon | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`rmsnorm_rows`](crates/kernels/wgsl/rmsnorm_rows.wgsl) | RMSNorm, one WORKGROUP per row — the decode-regime variant | 64-thread workgroup tile, 1 barrier | 4/5 | ✓ | ✓ | ✓ | — |
| [`roof_dp4a`](crates/kernels/wgsl/roof_dp4a.wgsl) | Peak packed-int8 (DP4A) rate probe — the INT8 half of the device roofline | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`roof_fma`](crates/kernels/wgsl/roof_fma.wgsl) | Peak fp32 FMA-rate probe — the COMPUTE half of the device roofline | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`rope`](crates/kernels/wgsl/rope.wgsl) | Rotary position embedding, applied in place to either the q or k region of the fused qkv buffer (select via base_off) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope2d`](crates/kernels/wgsl/rope2d.wgsl) | Table-driven 2D RoPE (DINOv3/WorldMirror "normalized" variant), in place on the q or k region of a fused [rows, row_stride] buffer | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope2d_partial`](crates/kernels/wgsl/rope2d_partial.wgsl) | Table-driven interleaved M-RoPE, PARTIAL: rotate only the first `2*half` channels of each `head_dim`-wide head, in place on the q or k region of a fused [rows, row_stride] buffer | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_at`](crates/kernels/wgsl/rope_at.wgsl) | RoPE (forward) at an EXPLICIT absolute position — the decode-step twin of rope_base | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_base`](crates/kernels/wgsl/rope_base.wgsl) | Batched RoPE (forward), HF/Qwen "half-split" (GPT-NeoX) convention, with a configurable base theta | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_base_bwd`](crates/kernels/wgsl/rope_base_bwd.wgsl) | RoPE backward for the half-split (HF/Qwen) convention (see `rope_base.wgsl`) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_interleave_table`](crates/kernels/wgsl/rope_interleave_table.wgsl) | Table-driven interleaved rotary position embedding (Z-Image / multi-axis) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_neox`](crates/kernels/wgsl/rope_neox.wgsl) | Rotary position embedding, NeoX / half-split style (Chronos-2, GPT-NeoX, Llama-HF) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_paged`](crates/kernels/wgsl/rope_paged.wgsl) | RoPE (half-split, base theta) applied to a batch of single-token rows, each at its OWN absolute position `positions[row]` — the batched-decode twin of rope_at (which assumes pos_base+row) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_partial`](crates/kernels/wgsl/rope_partial.wgsl) | Moondream partial RoPE (forward) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_partial_bwd`](crates/kernels/wgsl/rope_partial_bwd.wgsl) | Moondream partial RoPE (backward) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_sub`](crates/kernels/wgsl/rope_sub.wgsl) | Interleaved RoPE (forward, in place) on the FIRST `rope_dim` channels of each head (a sub-slice of a `head_dim`-wide head), for the DSA indexer where each head is laid out [rope / pass] | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`rope_train`](crates/kernels/wgsl/rope_train.wgsl) | Batched RoPE (forward). Rows are flattened [B*T, ...]; the rotation angle uses the WITHIN-sequence position = row % T | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`rope_train_bwd`](crates/kernels/wgsl/rope_train_bwd.wgsl) | RoPE backward: gradient is the inverse (transpose) rotation, i.e | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`router_bwd`](crates/kernels/wgsl/router_bwd.wgsl) | Router backward: gradient w.r.t | one thread per output element, 5 nested serial reductions, array-free | 1/5 | ✓ | ✓ | — | — |
| [`router_bwd_sigmoid`](crates/kernels/wgsl/router_bwd_sigmoid.wgsl) | GLM/DeepSeek-V3 "noaux_tc" MoE router (backward) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`router_gate`](crates/kernels/wgsl/router_gate.wgsl) | Router gating: softmax over experts -> keep top_k -> renormalise | one thread per token, array-free (no expert-count cap) | 1/5 | ✓ | ✓ | — | — |
| [`router_gate_sigmoid`](crates/kernels/wgsl/router_gate_sigmoid.wgsl) | GLM/DeepSeek-V3 "noaux_tc" MoE router (forward) | one thread per output element, 6 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`router_gate_train`](crates/kernels/wgsl/router_gate_train.wgsl) | Router gating (training variant): softmax + full probs + top_k gate | one thread per token, array-free (no expert-count cap) | 1/5 | ✓ | ✓ | — | — |
| [`row_scatter`](crates/kernels/wgsl/row_scatter.wgsl) | Row scatter by index — the inverse of the `embed` row-gather for UNIQUE indices | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`scale_add`](crates/kernels/wgsl/scale_add.wgsl) | MoE combine for one expert | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`scale_add_dexp`](crates/kernels/wgsl/scale_add_dexp.wgsl) | MoE combine backward, part 1 — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`scale_add_dgate`](crates/kernels/wgsl/scale_add_dgate.wgsl) | MoE combine backward, part 2 — gradient w.r.t | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`scale_chan`](crates/kernels/wgsl/scale_chan.wgsl) | Per-channel scale (forward) — the codec decoder's LayerScale and any elementwise per-channel gain | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`scale_chan_dg`](crates/kernels/wgsl/scale_chan_dg.wgsl) | Per-channel scale backward (gain grad), the scale_chan companion | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`scale_row`](crates/kernels/wgsl/scale_row.wgsl) | Per-row (per-sample) scalar scale on a row-major [N, M] tensor — spec | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`scan_add`](crates/kernels/wgsl/scan_add.wgsl) | Exclusive prefix scan, stage 2 | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`scan_block`](crates/kernels/wgsl/scan_block.wgsl) | Exclusive prefix scan, stage 1 of the generic multi-pass scan | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`sigmoid`](crates/kernels/wgsl/sigmoid.wgsl) | Sigmoid activation:  y = 1 / (1 + exp(-x)) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`sigmoid_bwd`](crates/kernels/wgsl/sigmoid_bwd.wgsl) | Sigmoid backward:  dx = dy * s * (1 - s),  s = sigmoid(x) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`silu`](crates/kernels/wgsl/silu.wgsl) | SiLU (a.k.a. swish) activation | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`silu_bwd`](crates/kernels/wgsl/silu_bwd.wgsl) | SiLU backward — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`silu_bwd_da`](crates/kernels/wgsl/silu_bwd_da.wgsl) | SwiGLU backward, part 1 — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`silu_bwd_db`](crates/kernels/wgsl/silu_bwd_db.wgsl) | SwiGLU backward, part 2 — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`silu_gate`](crates/kernels/wgsl/silu_gate.wgsl) | SwiGLU activation (Kronos FFN) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`silu_mul`](crates/kernels/wgsl/silu_mul.wgsl) | SwiGLU activation core:  out[i] = SiLU(a[i]) * b[i],  SiLU(x) = x * sigmoid(x) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`snake_beta`](crates/kernels/wgsl/snake_beta.wgsl) | SnakeBeta activation (forward) — the periodic activation in the codec SEANet decoder / BigVGAN-style vocoder | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`softmax_k`](crates/kernels/wgsl/softmax_k.wgsl) | Softmax over a STRIDED axis of length K, NCHW-flattened | one thread per output element, 3 nested serial reductions | 1/5 | ✓ | ✓ | ✓ | — |
| [`softmax_k_dx`](crates/kernels/wgsl/softmax_k_dx.wgsl) | Softmax-over-strided-K backward | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | ✓ | — |
| [`softmax_rows`](crates/kernels/wgsl/softmax_rows.wgsl) | Row softmax, one WORKGROUP per row — the long-context attention variant | 64-thread workgroup tile, 3 barriers | 4/5 | ✗ | ✓ | ✓ | — |
| [`sort_hist`](crates/kernels/wgsl/sort_hist.wgsl) | LSD radix sort, stage 1: per-chunk 256-bin digit histogram | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`sort_scatter`](crates/kernels/wgsl/sort_scatter.wgsl) | LSD radix sort, stage 2: stable per-chunk scatter | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_bwd_count`](crates/kernels/wgsl/splat_bwd_count.wgsl) | 3DGS backward, stage 1: per-pixel count of gradient-contributing gaussians (same walk as the forward compositing | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_bwd_emit`](crates/kernels/wgsl/splat_bwd_emit.wgsl) | 3DGS backward, stage 2: replay each pixel's compositing walk and emit one gradient record per contributing gaussian (gsplat blend-backward math) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_bwd_keys`](crates/kernels/wgsl/splat_bwd_keys.wgsl) | 3DGS backward, stage 3 prep | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_emit`](crates/kernels/wgsl/splat_emit.wgsl) | Tiled 3DGS, stage 2: expand each visible gaussian into per-tile sort instances at its scanned offset | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_grad_reduce`](crates/kernels/wgsl/splat_grad_reduce.wgsl) | 3DGS backward, stage 4: per-gaussian segmented reduction over the id-sorted gradient records | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_naive`](crates/kernels/wgsl/splat_naive.wgsl) | Naive 3DGS compositing — the correctness oracle and tiny-scene path | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`splat_pack_rgba8`](crates/kernels/wgsl/splat_pack_rgba8.wgsl) | Pack the RGBA f32 framebuffer into one u32 (r / g<<8 / b<<16 / a<<24) per pixel — quarters the demo readback bandwidth | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_project`](crates/kernels/wgsl/splat_project.wgsl) | 3DGS projection (gsplat-parity EWA) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_project_bwd`](crates/kernels/wgsl/splat_project_bwd.wgsl) | 3DGS backward, stage 5: EWA projection VJP | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_rasterize`](crates/kernels/wgsl/splat_rasterize.wgsl) | Tiled 3DGS, stage 5: front-to-back compositing | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_tile_count`](crates/kernels/wgsl/splat_tile_count.wgsl) | Tiled 3DGS, stage 1: per-gaussian overlapped-tile count | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_tile_ranges`](crates/kernels/wgsl/splat_tile_ranges.wgsl) | Tiled 3DGS, stage 4: per-tile [start, end) ranges over the SORTED keys, by neighbor comparison (disjoint writes, no atomics) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splat_unpack`](crates/kernels/wgsl/splat_unpack.wgsl) | Unpack the fit-time packed gaussian geometry [N*10] = {mean(3), scale(3), quat(4)} into the separate forward-kernel buffers | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splice`](crates/kernels/wgsl/splice.wgsl) | Residual splice (forward) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`splice_add`](crates/kernels/wgsl/splice_add.wgsl) | Residual DeepStack add: accumulate a compact `[n]` source block into `dst` starting at flat element offset `base` | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`splice_bwd`](crates/kernels/wgsl/splice_bwd.wgsl) | Residual splice (backward) | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`tanh_act`](crates/kernels/wgsl/tanh_act.wgsl) | Tanh forward:  y = tanh(x) | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`tanh_act_bwd`](crates/kernels/wgsl/tanh_act_bwd.wgsl) | Tanh backward — gradient w.r.t | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`tau_scale`](crates/kernels/wgsl/tau_scale.wgsl) | Moondream per-(head, token) attention-temperature scale, broadcast over head_dim | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`tau_scale_ds`](crates/kernels/wgsl/tau_scale_ds.wgsl) | tau_scale backward w.r.t. the per-(head,token) scale `s` | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`topk_extract_step`](crates/kernels/wgsl/topk_extract_step.wgsl) | One step of iterative top-K extraction | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`topk_mask`](crates/kernels/wgsl/topk_mask.wgsl) | DSA top-k selection mask (forward) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`unpack_qkv`](crates/kernels/wgsl/unpack_qkv.wgsl) | Inverse of pack_qkv: split one fused [seq, 3*d_model] gradient buffer (laid out per token as [ q(d) / k(d) / v(d) ]) back into three contiguous [seq, d_model] grad buffers | one thread per output element | 3/5 | ✓ | ✓ | — | — |
| [`upsample2`](crates/kernels/wgsl/upsample2.wgsl) | Nearest-neighbour x2 upsample | one thread per output element | 3/5 | native | ✓ | ✓ | — |
| [`upsample2_dx`](crates/kernels/wgsl/upsample2_dx.wgsl) | Nearest-neighbour x2 upsample backward, GATHER form | one thread per output element | 3/5 | ✓ | ✓ | ✓ | — |
| [`vq_argmax_dot`](crates/kernels/wgsl/vq_argmax_dot.wgsl) | Vector-quantization nearest-codebook assignment (COSINE similarity), used by GenieRedux-style tokenizers with `use_cosine_sim=True` | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`vq_argmin`](crates/kernels/wgsl/vq_argmin.wgsl) | Vector-quantization nearest-codebook assignment (Euclidean) | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | int8 |
| [`weighted_gap`](crates/kernels/wgsl/weighted_gap.wgsl) | Weighted global average pool | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`weighted_gap_dm`](crates/kernels/wgsl/weighted_gap_dm.wgsl) | weighted_gap gradient wrt the WEIGHT MAP | one thread per output element, serial inner reduction | 2/5 | ✓ | ✓ | — | — |
| [`weighted_gap_dx`](crates/kernels/wgsl/weighted_gap_dx.wgsl) | weighted_gap gradient wrt the FEATURE MAP | one thread per output element | 3/5 | ✓ | ✓ | — | — |

<!-- END KERNEL TABLE -->

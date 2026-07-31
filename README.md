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

Every model is one `brain <model> <verb>` subcommand:

```
brain data        dataset generation + tokenizers
brain gpt         GPT decoder: train | gen | eval
brain qwen        Qwen3 LLM: import | infer | export | precompile | train | finetune
brain tts         Qwen3-TTS: import | clone | synth | finetune
brain yolo        YOLOv8 detector: train | fine-tune | eval | detect
brain npu         OpenVINO/NPU: export | quantize | check | run | bench | sim
brain federated   sharded MoE: split | verify | merge | assemble | train-expert
brain pid         PID control transformer
brain bench       architecture-evaluation harness (+ eval | scale | advise | compare)
brain run         event-driven streaming controller (HFSM over JSONL)
brain gradcheck   run the gradient checks
```

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

## Models

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

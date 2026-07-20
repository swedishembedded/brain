# AGENTS.md — brain (edge-AI model training framework)

Routing guide for this repo. **brain** trains and evaluates **neural networks
from scratch on the CPU, NPU and GPU**, **pure Rust + raw kernels**. Full
validated parity between all supported accelerator backends.

Brain provides both training and inference with quantization and with highly
efficient use of accelerator hardware.

It is a self-contained Cargo **workspace** under `crates/` — no Python in the build/test
path; backprop correctness is gated by an in-repo finite-difference gradient
checker (`crates/gradcheck`), not a PyTorch oracle.

The engine is **architecture-agnostic**: the WGSL kernels (`crates/kernels`) are
reusable building blocks, not a fixed model. New architectures are composed from
them, keeping the gradient-check discipline.

## Models (today)

1. **GPT decoder** (`crates/gpt`) — dense nanogpt-parity baseline: token+learned
   positional embeddings, pre-LN, causal MHA, GELU MLP, untied `lm_head`, masked
   CE. Train/sample/eval via `brain gpt …`.
2. **Sparse MoE Transformer** (`crates/moe`) — RMSNorm/RoPE, top-k experts; with
   **federated/sharded** expert training (`crates/federated`).
3. **PID event/effect Transformer** (`crates/pid`) — LayerNorm, learned
   positions, biased linears; backs the WebGPU demo (`crates/web`).
4. **GLM-5.2 decoder** (`crates/glm`) — `glm_moe_dsa`: **MLA** (low-rank q/kv with
   a decoupled nope/rope head split + interleaved RoPE), a **sigmoid `noaux_tc`
   MoE** (per-expert selection bias, shared expert, `first_k_dense_replace`
   dense→MoE schedule), untied `lm_head`. Phase 1 = the dense MLA-MoE core
   (indexer is a no-op while `index_topk >= block_size`); the DSA sparse indexer,
   MTP, and NPU export are later phases. Gradient-checked (`gradcheck::check_glm`)
   + learnability tests. Train/eval/infer/finetune/import via `brain glm …`;
   HF import (single/sharded safetensors). Bench arch `glm`.
5. **YOLOv8-style detector** (`crates/yolo`) — from-scratch anchor-free object
   detector: CSP backbone → PAN-FPN neck → decoupled DFL head, with the
   assigner + BCE/CIoU/DFL detection loss and NMS box decode. Trains on the
   synthetic detection dataset and runs `detect` (boxes in pixel coords); CPU
   backend only. Byte-compatible with canonical `yolov8n` for weight import.
   Train/eval/detect/fine-tune via `brain yolo …`.
6. **WorldMirror-2 multi-view 3D reconstruction** (`crates/mirror`) — the
   HY-World 2.0 1.26B feed-forward model: per-frame DINOv2 ViT-L/14 encoding,
   24 alternating frame/global attention levels (QK-norm + normalized 2D
   RoPE), DPT heads (depth/points/normals/gaussians) + iterative camera head;
   photos → a navigable 3DGS scene. Imported exactly from the reference
   checkpoint; parity-gated per stage vs PyTorch goldens. `brain mirror
   {import,infer,demo,export-npu}`; docs `docs/mirror/`.
7. **3D Gaussian Splatting** (`crates/splat`) — from-scratch tiled 3DGS
   rasterizer (atomic-free/barrier-free WGSL: generic scan + radix sort →
   per-tile compositing) with forward AND backward (autograd-verified),
   Inria PLY IO, interactive WASD+mouse viewer, and `splat fit` scene
   optimization. `brain splat {info,render,view,fit}`; docs `docs/splat/`.
8. **ZipDepth monocular depth** (`crates/depth`) — the 6.1M pure-conv depth
   net (QARep/RepVGG blocks, SE/strip/global-context attention, convex
   upsampling), exact vs the reference PyTorch on the released checkpoints.
   Realtime demo (`brain depth --image|--camera`, SDL views incl.
   autostereograms) on CPU/GPU/Vulkan and the Intel NPU (ONNX/OpenVINO,
   cosine 0.99998). Gradient-checked end to end. See `docs/depth/README.md` +
   `docs/depth/STATUS.md`.

## Workspace layout (`crates/`)

| Crate | Responsibility |
|---|---|
| `kernels` | all WGSL kernels (the source of truth) as consts + `src()` |
| `gpu-core` | the accelerator seam: one `Gpu`/`DeviceBuffer`/`Step` API over **two backends** — wgpu and the native CPU backend — chosen at runtime |
| `wgsl-cpu` | the CPU backend's compiler: WGSL → naga IR → **Cranelift JIT** → native code run across cores with rayon |
| `paramstore` / `optim` | param/grad/Adam buffers; AdamW + grad clip |
| `checkpoint` | `.weights` container + manifest/SHA-256 |
| `data` | char + GPT-2 **BPE** tokenizers, dataset generators, loaders (masking/alignment), normalization |
| `gpt` | GPT model + training loop + sampling |
| `moe` / `pid` | the MoE and PID models (fwd/bwd) |
| `glm` | GLM-5.2 decoder: MLA + sigmoid `noaux_tc` MoE (+ shared expert, dense→MoE schedule); HF import (single/sharded) |
| `yolo` | YOLOv8-style detector: backbone/neck/head, DFL decode, assigner + detection loss, NMS, `detect` inference, canonical `yolov8n` weight import |
| `vision` | shared conv-net blocks (spec-driven `Conv` incl. the fused/register-tiled eval paths, `BatchNorm`, `SPPF`, bottlenecks), name-resolved kernel ids, `fold_bn` |
| `depth` | ZipDepth: model/blocks/import/fuse, `Predictor` (reference-exact preprocessing), viz/stereo/effects, INT8 calib report |
| `mirror` | WorldMirror-2: config/param_list, strict import, DINOv2+trunk (via `model::vit`), DPT/camera heads, PIL-exact preprocessing, gaussian assembly |
| `splat` | 3DGS: generic device scan/radix-sort (brain's first), tiled rasterizer fwd+bwd, PLY IO, `fit` optimizer, viewer plumbing |
| `capture` | V4L2 webcam (hand-rolled ioctl FFI, YUYV→RGB, latest-frame slot) |
| `onnx` | pure-Rust ONNX graph model + serializer (export only; vendored `prost` bindings, no `protoc` in the build) |
| `npu` | YOLO→ONNX export + BN fold + brain-native INT8 PTQ + fake-quant simulator + OpenVINO **Intel NPU** runtime (default dep on x86_64 linux/windows; `runtime-linking`) |
| `federated` | vertical expert split/assemble, hash-verified manifests |
| `eval` | perplexity + task exact-match (LM) and detection metrics (mAP@0.5/precision/recall) |
| `gradcheck` | finite-difference backprop correctness gate |
| `events` | JSONL event protocol (`user_text`/`camera_frame`/`brain_text_chunk`/`object_detected`) + base64/PPM frame decode |
| `hfsm` | generic hierarchical state-machine engine (RTC dispatch, LCA entry/exit) |
| `runtime` | event-driven HSM controller wiring loaded models to the event protocol (`InferModel`/`DetectModel` seams) |
| `cli` | the `brain` binary (aggregates everything) |
| `web` | wasm32/WebGPU PID demo; optional `vulkan` (coopmat) is non-default |


## Task → where to look

| Task | Where |
|---|---|
| MoE toy task / honest eval methodology | `README.md` |
| GLM-5.2 (MLA + MoE + DSA indexer + MTP): arch, status, CLI, NPU | `docs/glm/README.md`, `docs/glm/NPU.md`; `crates/glm`, `crates/cli/src/glm_cli.rs` |
| Architecture & crate graph | `docs/ARCHITECTURE.md` |
| Federated MoE pipeline (done vs remaining) | `docs/FEDERATED.md` |
| Testing strategy + gradient-check gate | `docs/TESTING.md` |
| Performance: CPU/GPU inference optimizations (what sped things up + why) | `docs/PERFORMANCE.md` |
| YOLOv8 detector training + inference (end-to-end guide) | `docs/yolo/README.md` |
| Engine internals | `docs/engine-README.md`, `engine-TRAINING.md`, `engine-README_VULKAN.md`, `engine-README_WEB.md` |
| Add/adjust a WGSL kernel | `crates/kernels/wgsl/*.wgsl` (regenerate the const list if you add files) |
| GPT model / training / sampling | `crates/gpt/src/{model,train,sample,init}.rs` |
| YOLO model / loss / inference | `crates/yolo/src/{model,head,blocks,loss,assign,infer,nms,config}.rs` |
| YOLO train / eval / detect / fine-tune (CLI) | `crates/cli/src/yolo_cli.rs` |
| YOLO → Intel NPU: export / quantize / run / bench (OpenVINO) | `crates/npu`, `crates/onnx`, `crates/cli/src/npu_cli.rs`, `docs/yolo/NPU.md` |
| ZipDepth: guide / workstream ledger (incl. GPU perf root causes) | `docs/depth/README.md`, `docs/depth/STATUS.md` |
| WorldMirror-2 (photos → 3DGS scene): guide / ledger / parity gates | `docs/mirror/README.md`, `docs/mirror/STATUS.md`; `crates/mirror`, `crates/cli/src/mirror_cli.rs` |
| 3D Gaussian Splatting rasterizer + viewer + fit | `docs/splat/README.md`, `docs/splat/STATUS.md`; `crates/splat`, `crates/cli/src/splat_cli.rs` |
| Shared ViT block builder (DINOv2/trunk/camera-head blocks) | `crates/model/src/vit.rs` |
| ZipDepth model / import / predictor / demo views | `crates/depth/src/{model,blocks,import,predict,viz,stereo,effects}.rs`, `crates/cli/src/depth_cli.rs` |
| ZipDepth → Intel NPU (fp32 ONNX, exact parity) | `npu::depth_topology`, `crates/depth/src/fuse.rs` |
| Fused conv eval paths (act selector, register tiling, grouped) | `crates/vision/src/blocks.rs`, `crates/kernels/wgsl/conv_act*.wgsl`, `conv2d_gd_reg.wgsl`, `crates/backend-cpu/src/fast_conv.rs` |
| Detection metrics (mAP/precision/recall) | `crates/eval/src/detection.rs` |
| Synthetic detection dataset (RGB shapes + GT boxes) | `crates/data/src/gen_detect.rs` |
| Event/HFSM controller (`brain run`): `camera_frame`→`object_detected`, `user_text`→`brain_text_chunk` | `crates/runtime/src/{lib,pump}.rs`, `crates/cli/src/run_cli.rs`, `crates/events/src/lib.rs` |
| Datasets & tokenizers | `crates/data/src/{prepare,gen_*,tokenizer,bpe,loader,binio,rng}.rs` |
| Federated shard/assemble | `crates/federated/src/{shard,sha256}.rs` |
| CLI subcommands | `crates/cli/src/{main,gpt_cli,yolo_cli,data_cli,federated_cli,pid_cli,run_cli}.rs` |
| Porting source-of-truth (read-only) | `scratchpad/reference/{nanogpt,sharded_moe_example,pytorch}/` |


## Essential commands

**Always build through the Makefile, never `cargo` directly:** use `make build`
for the debug build and `make release` for the optimized build (and `make test`
for the suite). They wrap cargo with the project's expected flags/targets; calling
`cargo build`/`cargo build --release` by hand is not supported.

```bash
make build                           # debug build (wraps cargo build)
make release && make test            # optimized build + full suite (MOE_SKIP_GPU_TESTS=1 to skip GPU)
make gradcheck                       # backprop correctness gate
make data/<name>                     # calculator|reverser|wordcalc|timeseries|shakespeare_char|gpt
make train/gpt/<name>                # train GPT -> out/gpt-<name>.weights
make eval/gpt/<name>                 # perplexity + exact-match
make data/detect                     # synthetic object-detection dataset -> data/detect
make train/yolo                      # train tiny YOLO -> out/yolo.weights
make eval/yolo                       # mAP@0.5 + precision/recall
make detect/yolo                     # run detection on a sample image (JSON boxes)
make bench                           # GPT baseline on shared char datasets
make federated-demo                  # MoE train -> split -> verify -> merge
make web/dev                         # WebGPU demo (delegates to crates/web)

# direct binary
./target/release/brain {data|gpt|yolo|federated|gradcheck|pid|train|eval|generate} …

# event/stdio controller: an HFSM (crates/runtime) reads JSONL events on stdin
# and emits JSONL on stdout — user_text -> brain_text_chunk (streamed, one token
# per tick) and camera_frame -> object_detected. --gpt/--yolo load real models
# (or BRAIN_GPT/BRAIN_YOLO); with neither, fake echo/detector models keep the
# loop usable. printf '{"event":"camera_frame","format":"rgb8","w":128,"h":128,
# "data":"…"}\n' | brain run --yolo out/yolo.weights

# CPU-only (no GPU): add --device cpu to any command, or set BRAIN_DEVICE=cpu.
# Same WGSL kernels, JIT-compiled to native code across all cores.
./target/release/brain gpt train data/calculator --device cpu --out out/gpt.weights
BRAIN_DEVICE=cpu make test            # run the whole suite on CPU, no GPU needed
```


## Benchmark suite (`crates/bench`)

`brain-bench` (lib `bench`) is a **model-agnostic** architecture-evaluation
layer: each benchmark owns its *dataset* and its *scoring*, the harness owns
running it. Use it to answer "does this architecture actually learn task X?"
the same way across many tasks. The pattern is built to be copied — sibling
work adds MAD, formal-language, and scaling-sweep benchmarks alongside the
reference MQAR.

**Run** (no usable GPU here — always select CPU):

```bash
BRAIN_DEVICE=cpu make bench          # run every registered benchmark, one table
BRAIN_DEVICE=cpu make bench/mqar     # run a single benchmark
./target/release/brain bench [--device cpu] [<name>] [--seed S]
```

The runner prints one comparison table: `benchmark | score | (fields) |
threshold | pass/fail`. `make bench/char` keeps the legacy GPT-on-char-datasets
sweep.

**Add a benchmark:**

1. New module `crates/bench/src/<name>.rs` with a type implementing the
   `Benchmark` trait (`name`/`description`/`prepare`/`evaluate`/`threshold`).
   `prepare` writes brain's `train.bin`/`val.bin`/`meta.json` token layout;
   `evaluate` trains (today via `gpt::train`, behind a `// TODO(model-trait)`
   seam) and returns `Metrics` (CE nats/bits, bits-per-byte, exact-match,
   associative-recall, distinct-n, repetition-rate — all in `metrics.rs`).
2. Register it in `bench::registry()` (`crates/bench/src/lib.rs`). The generic
   `make bench/%` rule and `brain bench <name>` then pick it up with no further
   wiring.
3. Add a learnability test in `crates/bench/tests/`, gated by
   `MOE_SKIP_GPU_TESTS`, asserting the score clears a **measured** threshold.

The reference benchmark is **MQAR** (multi-query associative recall): per
sequence, several `key→value` bindings then queried keys whose bound values the
model must recall in-context; loss is masked to the answer region and windows
are line-aligned. Registered benchmarks: **mqar**, **toolcall** (tool-calling:
map a user intent to one structured tool call `TOOL_k args…`, masking the prompt
and training/scoring exact-match only on the assistant tool-call span), plus the
**mad_\*** family (recall, fuzzy/noisy recall, selective-copy, memorize). See
`crates/bench/README.md` for the full design.

Registered benchmarks: `mqar`, the MAD family (`mad_recall`, `mad_fuzzy_recall`,
`mad_noisy_recall`, `mad_selective_copy`, `mad_memorize`), and the
formal-language / algorithmic state-tracking probes `parity` (running-parity bit
state), `mod_add` (`a+b=c (mod p)`, the grokking task), and `dyck` (Dyck-k
balanced brackets, hierarchical state).

### Evaluating a new architecture (turn-key harness)

The whole battery is architecture-agnostic via the `DecoderLm` seam, so the same
benchmarks score *any* architecture and the results are directly comparable. The
3-step recipe:

1. **Implement `DecoderLm`** for the model (`train_decoder` + `load_scorer`, plus
   a `Scorer`). No benchmark changes — `GptDecoder` is the reference impl.
2. **Add one line to `arch_registry()`** in `crates/bench/src/arch.rs` (name +
   `Size` descriptor + a `factory`). Registered today: `gpt`, `gpt-small`,
   `gpt-wide`.
3. **Run + compare**:
   ```bash
   BRAIN_DEVICE=cpu make bench/eval ARCH=<name>   # whole battery -> results/<arch>-<seed>.json
   BRAIN_DEVICE=cpu make bench/compare            # leaderboard over all results/*.json
   ```
   (direct: `brain bench eval --arch <name> [--seed S --out F --smoke]`;
   `brain bench compare a.json b.json …`).

**Capability axes** (`crates/bench/src/axes.rs`) group benchmarks into a small
profile — `recall` (mqar + mad recall/fuzzy/noisy), `copying` (selective-copy,
toolcall), `memory` (memorize), `state_tracking` (parity, dyck), `compression`
(mad_compress), `arithmetic` (mod_add, *informational*) — each scored as the mean
of its benchmarks. `eval` writes a JSON artifact (arch, size, param count, commit,
seed, timestamp, per-benchmark `{score, threshold, passed, informational,
metrics}`, per-axis aggregates, gating pass-rate); `compare` diffs ≥2 of them
side-by-side. `results/` is git-ignored. This is the foundation the next
predictive-scaling + tuning-advisor layer builds on.

> Non-GPT caveat: `mad_compress` is a bottleneck autoencoder (MSE head), not a
> next-token decoder, so it ignores the supplied `DecoderLm` — its `compression`
> score does not yet reflect a candidate architecture.

### Predictive per-capability scaling + tuning advisor

`eval` says where an arch stands; **`scale`** predicts how each capability
improves as the model grows, and **`advise`** says what to tune.

- **`brain bench scale --arch <name>`** (`crates/bench/src/capscale.rs`): sweeps a
  small SIZE grid (`L1xD32xH2 → L2xD64xH4 → L3xD96xH6`, increasing params via
  `ScaledGpt`) and, per capability axis, trains+scores *one representative
  benchmark* (cheapest informative: mqar/mad_selective_copy/mad_memorize/parity/
  mad_compress/mod_add) at each size. Fits a **saturating trend**
  `score(N) ≈ ceil − A·N^(−β)` (gap-to-ceiling power law, reuses `scaling::ols`),
  records the **slope per doubling**, **β**, **R²**, **predicted score@2x/@4x**,
  and a **verdict** ∈ {improving, saturating, flat}. Writes
  `results/scale-<arch>-<seed>.json`. Smoke budget (~few min CPU); the *shape* of
  the curve + extrapolation is the deliverable, not absolute scores.
- **Experts knob (future MoE):** the sweep dimension is a generic `Knob` enum.
  Only `Knob::Size` is wired; a MoE arch sweeps `Knob::Experts` the same way —
  register the arch + fill the `// TODO(experts)` branch in `capscale::grid_for`;
  the fit/advisor are dimension-agnostic and need no change. MoE *scoring* is not
  implemented yet (no MoE arch registered).
- **`brain bench advise <eval.json> [<scale.json>]`** (`crates/bench/src/advisor.rs`):
  ranked, concrete recommendations. Lever = **headroom (1−score, gated axes) ×
  size-slope**; per-axis signal → action (rising slope → *increase size*; flat
  slope → *change the mechanism* = architecture-bound; low `train_ce` + low eval →
  *more data/reg/steps*; ≈ceiling → *deprioritize*); each rec carries
  score-per-Mparam (compute-efficiency). **`brain bench eval` prints the top-3 as
  a footer**, so the eval output itself carries the tuning breakdown.
  `make bench/scale ARCH=<name>` + `make bench/advise ARCH=<name>`.

## Conventions & invariants

- **WGSL is the source of truth.** Kernels live only in `crates/kernels/wgsl/`,
  embedded as consts; no kernel text is duplicated. Adding a `.wgsl` means
  regenerating the const list in `crates/kernels/src/lib.rs`.
- **fp32 only, core compute only** — single bind group, ≤8 storage
  buffers/kernel (the WebGPU guarantee; `router_bwd`/`mla_scores`/
  `layernorm_dgamma` bind 5), `@workgroup_size(64)`, no atomics/subgroups/f16.
  This is what keeps it portable to old GPUs and WebGPU.
- **Two backends, one build, one API.** `gpu-core` exposes a single
  `Gpu`/`DeviceBuffer`/`Step` surface; every model (gpt/moe/pid) is written once
  against it. The accelerator is the *only* thing abstracted — there is no
  per-backend model code. Both backends compile into every native build and are
  selected at runtime (`--device cpu|gpu` / `BRAIN_DEVICE`); wgpu is the default.
  The CPU backend reuses the **same WGSL** via the `wgsl-cpu` Cranelift JIT, so
  WGSL stays the single source of truth. On wasm only the wgpu/WebGPU backend
  exists. `crates/vulkan` (coopmat) is excluded from `default-members`; the `web`
  crate is empty off wasm32.
- **The Intel NPU is NOT a `gpu-core` backend.** OpenVINO is a *whole-graph*
  compiler, so `--device npu` is a separate export→quantize→compile→run path
  (`crates/npu`), not a per-op `Gpu`/`Step` backend. `crates/yolo` and the default
  build stay free of OpenVINO at the source level; the OpenVINO runtime is loaded
  at run time (`runtime-linking`), so `make build`/`make test` stay green with no
  OpenVINO installed. See `docs/yolo/NPU.md`.
- **Backprop is gated by `gradcheck`** (finite differences) — run it after any
  fwd/bwd math change. SSA-style forward (each stage writes a fresh buffer that
  doubles as the backprop activation cache) — preserve it when adding stages.
- **Evaluate honestly.** Hold the input distribution fixed; separate the metric
  (perplexity) from the task (exact-match on held-out data); see `README.md` §3.
- **`scratchpad/` is gitignored** — scratch weights, images, and the read-only
  Python porting references. Generated `data/` and `out/` are gitignored too.

# AGENTS.md — brain (edge-AI model training framework)

Routing guide for this repo. **brain** trains and evaluates **neural networks
from scratch on the GPU**, **pure Rust + raw WGSL**, fp32-only so the same
kernels run on old desktop GPUs and on WebGPU in the browser. It is a
self-contained Cargo **workspace** under `crates/` — no Python in the build/test
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
| `federated` | vertical expert split/assemble, hash-verified manifests |
| `eval` | perplexity + task exact-match (same-input model comparison) |
| `gradcheck` | finite-difference backprop correctness gate |
| `cli` | the `brain` binary (aggregates everything) |
| `web` | wasm32/WebGPU PID demo; optional `vulkan` (coopmat) is non-default |

## Task → where to look

| Task | Where |
|---|---|
| MoE toy task / honest eval methodology | `README.md` |
| Architecture & crate graph | `docs/ARCHITECTURE.md` |
| Federated MoE pipeline (done vs remaining) | `docs/FEDERATED.md` |
| Testing strategy + gradient-check gate | `docs/TESTING.md` |
| Engine internals | `docs/engine-README.md`, `engine-TRAINING.md`, `engine-README_VULKAN.md`, `engine-README_WEB.md` |
| Add/adjust a WGSL kernel | `crates/kernels/wgsl/*.wgsl` (regenerate the const list if you add files) |
| GPT model / training / sampling | `crates/gpt/src/{model,train,sample,init}.rs` |
| Datasets & tokenizers | `crates/data/src/{prepare,gen_*,tokenizer,bpe,loader,binio,rng}.rs` |
| Federated shard/assemble | `crates/federated/src/{shard,sha256}.rs` |
| CLI subcommands | `crates/cli/src/{main,gpt_cli,data_cli,federated_cli,pid_cli}.rs` |
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
make bench                           # GPT baseline on shared char datasets
make federated-demo                  # MoE train -> split -> verify -> merge
make web/dev                         # WebGPU demo (delegates to crates/web)

# direct binary
./target/release/brain {data|gpt|federated|gradcheck|pid|train|eval|generate} …

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

## Conventions & invariants

- **WGSL is the source of truth.** Kernels live only in `crates/kernels/wgsl/`,
  embedded as consts; no kernel text is duplicated. Adding a `.wgsl` means
  regenerating the const list in `crates/kernels/src/lib.rs`.
- **fp32 only, core compute only** — single bind group, ≤4 storage buffers/kernel,
  `@workgroup_size(64)`, no atomics/subgroups/f16. This is what keeps it portable
  to old GPUs and WebGPU.
- **Two backends, one build, one API.** `gpu-core` exposes a single
  `Gpu`/`DeviceBuffer`/`Step` surface; every model (gpt/moe/pid) is written once
  against it. The accelerator is the *only* thing abstracted — there is no
  per-backend model code. Both backends compile into every native build and are
  selected at runtime (`--device cpu|gpu` / `BRAIN_DEVICE`); wgpu is the default.
  The CPU backend reuses the **same WGSL** via the `wgsl-cpu` Cranelift JIT, so
  WGSL stays the single source of truth. On wasm only the wgpu/WebGPU backend
  exists. `crates/vulkan` (coopmat) is excluded from `default-members`; the `web`
  crate is empty off wasm32.
- **Backprop is gated by `gradcheck`** (finite differences) — run it after any
  fwd/bwd math change. SSA-style forward (each stage writes a fresh buffer that
  doubles as the backprop activation cache) — preserve it when adding stages.
- **Evaluate honestly.** Hold the input distribution fixed; separate the metric
  (perplexity) from the task (exact-match on held-out data); see `README.md` §3.
- **`scratchpad/` is gitignored** — scratch weights, images, and the read-only
  Python porting references. Generated `data/` and `out/` are gitignored too.

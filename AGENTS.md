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
| `gpu-core` | device init, dispatch, buffers, readback |
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

```bash
make release && make test            # build + full suite (MOE_SKIP_GPU_TESTS=1 to skip GPU)
make gradcheck                       # backprop correctness gate
make data/<name>                     # calculator|reverser|wordcalc|timeseries|shakespeare_char|gpt
make train/gpt/<name>                # train GPT -> out/gpt-<name>.weights
make eval/gpt/<name>                 # perplexity + exact-match
make bench                           # GPT baseline on shared char datasets
make federated-demo                  # MoE train -> split -> verify -> merge
make web/dev                         # WebGPU demo (delegates to crates/web)

# direct binary
./target/release/brain {data|gpt|federated|gradcheck|pid|train|eval|generate} …
```

## Conventions & invariants

- **WGSL is the source of truth.** Kernels live only in `crates/kernels/wgsl/`,
  embedded as consts; no kernel text is duplicated. Adding a `.wgsl` means
  regenerating the const list in `crates/kernels/src/lib.rs`.
- **fp32 only, core compute only** — single bind group, ≤4 storage buffers/kernel,
  `@workgroup_size(64)`, no atomics/subgroups/f16. This is what keeps it portable
  to old GPUs and WebGPU.
- **Default build is pure wgpu.** `crates/vulkan` (coopmat) is excluded from
  `default-members`; the `web` crate is empty off wasm32.
- **Backprop is gated by `gradcheck`** (finite differences) — run it after any
  fwd/bwd math change. SSA-style forward (each stage writes a fresh buffer that
  doubles as the backprop activation cache) — preserve it when adding stages.
- **Evaluate honestly.** Hold the input distribution fixed; separate the metric
  (perplexity) from the task (exact-match on held-out data); see `README.md` §3.
- **`scratchpad/` is gitignored** — scratch weights, images, and the read-only
  Python porting references. Generated `data/` and `out/` are gitignored too.

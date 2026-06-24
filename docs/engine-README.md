# tiny-sparse-moe-wgsl

> For the task definition, what the model learns, how to evaluate it
> (incl. out-of-distribution testing), and how/why the MoE structure maps onto
> the problem, see the top-level [`../README.md`](../README.md). Training docs
> are in [`TRAINING.md`](TRAINING.md).

Inference + generation for the `tiny_sparse_moe.py` model, reimplemented in Rust
as a **raw-WGSL `wgpu` compute pipeline**. Every forward stage is its own hand-
written compute kernel; the host code just orchestrates dispatches and samples.

It is deliberately **fp32-only** and uses only core compute features (a single
bind group, ≤4 storage buffers per kernel, `@workgroup_size(64)`, no atomics, no
subgroup ops, no f16). That makes it run on essentially any compute device,
including old **sm_61 (Pascal)** via Vulkan, as well as Metal/DX12/GL.

## Pipeline (one kernel per stage)

| WGSL kernel        | Stage                                                        |
|--------------------|-------------------------------------------------------------|
| `embed.wgsl`       | token embedding gather                                      |
| `rmsnorm.wgsl`     | RMSNorm                                                     |
| `matmul.wgsl`      | every `nn.Linear` (qkv, attn-out, router, expert g/u/d, lm_head); `out = x @ Wᵀ` |
| `rope.wgsl`        | rotary position embedding (applied to q and k in place)     |
| `attention.wgsl`   | causal MHA with online (stable) softmax                     |
| `router_gate.wgsl` | router softmax → top-k → renormalised dense gate            |
| `silu_mul.wgsl`    | SwiGLU activation core `SiLU(a)·b`                          |
| `scale_add.wgsl`   | MoE combine: `acc += gate·expert_out` (expert 0 sets)       |
| `add.wgsl`         | residual add                                                |

The whole forward pass is recorded into a **single compute pass**; `wgpu`
inserts the inter-dispatch memory barriers automatically. Sampling is done on
the CPU after reading back the last position's logits.

### A note on "sparse"

Training drops tokens that exceed each expert's capacity (a memory bound).
Inference has no such pressure, so this engine evaluates **all** experts and
masks each by the renormalised top-k gate. That is numerically identical to
sparse top-k dispatch without capacity dropping — and for `n_experts=4,
top_k=2` it is only ~2× the expert FLOPs of true sparse, which is negligible
here.

## Build & run

```bash
# 1. train + export weights from the Python model (repo root)
python tiny_sparse_moe.py train --steps 1500 --device cpu
python tiny_sparse_moe.py export --ckpt moe.pt --out moe.weights

# 2. build (this environment needs a writable CARGO_HOME)
cd moe-rs
CARGO_HOME=/tmp/cargo-moe cargo build --release

# 3. generate on the GPU
./target/release/moe --weights ../moe.weights --prompt 1,2,3,4 --max-new 64
```

Flags: `--weights`, `--prompt` (comma/space ints in `[0,vocab)`), `--max-new`,
`--temperature`, `--top-k`, `--seed`. `--top-k 1` gives deterministic greedy
decoding.

`PowerPreference::HighPerformance` selects a discrete/integrated GPU (e.g. Intel
Arc) when present; on a headless box it falls back to whatever Vulkan adapter
exists (e.g. `llvmpipe`).

## Correctness

Verified to match the PyTorch reference exactly. With greedy decoding both
produce the same sequence:

```
prompt 1,2,3,4  ->  ...,0,54,51,3,40,30,48,37,23,28,2,62
```

(`./target/release/moe ... --top-k 1`  vs  PyTorch `argmax` decoding.)

## Optimisation notes

Kernels are correctness-first and tuned for *universal compatibility*, not peak
throughput on any one GPU: `matmul` is one thread per output element with a
plain fp32 K-loop. For this tiny model (d_model=64, d_ff=128) that is already
far faster than needed. The obvious next steps for larger models —
shared-memory tiling in `matmul`, a KV cache to avoid recomputing the full
context each token, and batching the per-expert matmuls — are intentionally left
out to keep the kernels portable and easy to read.

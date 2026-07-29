# FLOP/OPS accounting: offline calculation + online counters

Every brain model emits device work as `Step`s through **one seam** —
`gpu_core::Gpu::step / step_sliced / step_buf`, submitted via `Gpu::submit`.
The FLOP/OPS accounting hangs on that seam, not on any model: one registry of
per-kernel cost formulas (`crates/gpu-core/src/cost.rs`, keyed by the kernel's
registered NAME) serves every model on every backend (wgpu / Vulkan / CPU).

Two numbers, same formulas:

* **OFFLINE** — `Gpu::cost_of(&steps)` walks a recorded step list without
  executing anything. Models expose their recorded graphs as
  `cost_fwd()` / `cost_bwd()` (qwen, gpt, lfm).
* **ONLINE** — `Gpu::submit` folds every submitted step into that handle's
  counters; read with `Gpu::ops_counters()`, clear with
  `reset_ops_counters()`. Always on (a few string-keyed adds per dispatch —
  noise next to the dispatch itself). Because the counters record what was
  actually dispatched, the runtime number reflects the kernel variants that
  really ran: the int8 path (`matmul_i8_dyn`, `matmul_i8_gemv`) counts
  **integer OPS**, distinct from fp32 **FLOPs**; a decode step counts the GEMV
  variants it used.

**Per device**: one `Gpu` handle is one device context (`share`/`new_like`
start fresh counters), and a sharded pipeline holds one handle per stage — so
per-handle counters are per-device numbers with no extra machinery. On a fully
covered model, offline == online exactly (asserted by tests, see below).

## Conventions

* 1 multiply-accumulate = 2 ops; transcendentals/div/sqrt = 1.
* Loop trip counts are exact for the recorded shape: causal attention costs
  `t(t+1)/2` pairs, bidirectional `t²`, cross `t_dec·t_enc`.
* Workgroup-cooperative variants (`rmsnorm_rows`, `matmul_gemv`) count the
  row's math once — the same model math costs the same regardless of which
  execution variant ran; *which* variant ran is still visible in the
  per-kernel breakdown.
* `bytes` is a streaming estimate (each logical operand read/written once) —
  a roofline denominator, not a cache model.
* Template-specialised variants (`base#K=V`) cost as their base kernel.

## Honesty semantics of coverage

An unknown kernel is **UNCOVERED**, never zero: it is excluded from the totals
and listed by name with its call count, and `CostReport::coverage()` states how
much of the run the totals describe. Unmeasured is null — never
0-pretending-complete. The qwen/gpt/lfm pipelines are pinned at 100% coverage
by unit tests (`pipelines_fully_costed` in each model's `model.rs`): adding a
kernel to a PIPELINES list without a cost formula fails the build's tests
rather than silently shrinking coverage.

## CLI

```
brain flops --model qwen|gpt|lfm [--weights F] [--batch B] [--block T]
            [--train] [--i8] [--stages N] [--run]
```

* No `--weights`: the model's tiny config with synthetic init (shape-only —
  cost depends on shapes, not values).
* `--train`: also record + cost the backward graph.
* `--i8` (qwen): the int8 inference build — its linears report `int_ops`.
* `--stages N` (qwen/gpt): even pipeline split; per-stage = per-device reports.
* `--run`: execute one pass on a synthetic batch and print the online counters
  beside the offline calculation (they agree exactly at full coverage).

Example — the real LFM2.5-Encoder-230M, one 512-token forward (CPU backend):

```
$ BRAIN_DEVICE=cpu brain flops --model lfm --weights out/lfm-230m.weights --block 512 --run
...
matmul_reg2                    98  166.430 G        0      1.181 G
matmul_tile                     3   68.719 G        0    408.945 M
...
TOTAL (covered)               276  242.146 G        0      3.060 G
coverage: 276/276 steps (100.0%)
lfm: online == offline
```

(Sanity: ~2·params·tokens = 2 · 230 M · 512 ≈ 235 G, plus attention.)

## Tests

* `gpu-core` unit tests: hand-computed formula expectations (GEMM, int8,
  causal/bidir/cross attention, norms, conv1d), report accounting, coverage
  probes.
* `crates/{qwen,lfm}/tests/flops.rs`: offline-vs-online exact agreement over a
  real forward+backward on the CPU backend, and the int8 build reporting
  `int_ops` where fp32 reports `flops`.
* `pipelines_fully_costed` in qwen/gpt/lfm pins 100% kernel coverage.

## Adding a kernel

Register the formula in `gpu_core::cost::kernel_cost` under the kernel's
registered name, reading the param layout from the kernel's WGSL header
(`crates/kernels/wgsl/<name>.wgsl`) — the uniform layout comment is the
contract. If the kernel joins a model's PIPELINES, the coverage test will
demand the formula.

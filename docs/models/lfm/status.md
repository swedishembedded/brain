# LFM2.5-Encoder — status ledger

Chronological, measured-only. Two boxes: the **training/GPU box** — 2× Tesla P40
(24 GB, Pascal, `max_storage_buffer_binding_size` 2047 MiB) + Xeon E5-2690 v3
(Haswell, 48 threads), **no NPU** — and the **Intel "AI Boost" (Core Ultra) NPU
box** (OpenVINO 2026.2, kernel 6.17) where the NPU numbers below were measured
(2026-07-29, see "NPU measured — Intel AI Boost").

## Done

- **P0 — import + tokenizer** (2026-07-29). `brain lfm import` maps both
  released checkpoints 1:1 (132 / 148 tensors, strict coverage, no transposes);
  FFN auto-adjust rule validated against real 350M shapes (6656→4608).
  Tokenizer: `QwenBpe` extended — digit-run width auto-detected from the file's
  own pre_tokenizer (`\p{N}{1,3}` vs Qwen's `\p{N}`), `template_prefix()`
  (BOS), `special_id()`; pinned golden vectors incl. digit runs, Arabic-Indic
  digits, CJK, specials. Core evolution: none needed.
- **P1 — forward + parity** (2026-07-29). Staged parity vs the HF reference
  (fixed token ids, fp32): **cosine 1.000000 / rel_l2 ≤ 1e-5 at every stage**
  — post-embed, all 14/16 layer outputs, final hidden, 3 logit probe rows —
  both models, CPU (wgsl-cpu JIT) and GPU (P40) backends. Fill-mask top-1
  " Paris" matches the reference. Core evolution: `model::block` gained
  `bidir_fwd/bwd` (seq2seq migrated onto them, its gradcheck green),
  `kv_expand_{fwd,bwd}` (+2 new WGSL kernels — the only new kernels needed),
  `rmsnorm_eps_{fwd,bwd}`, `pick_gemm` (qwen's 3 pickers now delegate),
  `vocab_tiles` (hoisted from qwen).
- **P2 — 8k chunked inference + capability + event API** (2026-07-29).
  `model::block::chunked_bidir_fwd` hoisted from `model::vit` (vit delegates);
  chunked regime (shared scratch, ping-pong residuals, probe-row head via the
  `embed` row-gather) is bit-exact vs materialized (`chunked_equiv.rs`,
  max|Δ| = 0 across chunk edges) and passes the reference goldens. Capability
  provider (`fill_mask`, `embed` — one-shot, not streaming) registered in
  `brain caps`/`brain do`/Controller event API; JSONL `action_request`
  round-trips. **Found + fixed:** bucket-padding poisoned bidirectional
  attention (unmasked pads are unsound) → exact-length builds; batched padding
  deferred until zeroed-pad-states + kmask land.

### Benchmark results — 8192-token context, fp32, `brain perf` (2026-07-29)

Measured through the **residency executor** (`--target lfm:<weights>:<tok>`,
scheduler + budgets + device lanes — the same path D-Bus serving uses), idle
GPUs, smoke-sized runs (warmups excluded, `smoke: true` in the artifacts).
Artifact unit: `sequence` (one-shot ⇒ ttfa == e2e, tpoa null by design).

| model | device | scenario | result |
|---|---|---|---|
| 230M | P40 (gpu0) | latency, concurrency 1, batch 1 | **e2e p50 22.32 s** (matches the direct forward — executor overhead ≈ 0) |
| 230M | P40 (gpu0) | sweep, batch 4 | conc 1: ttfa p99 21.2 s; conc 2: 45.1 s (**linear** — compute-bound at 8k, equal-length batching neither helps nor hurts throughput on one card) |
| 350M | P40 (gpu1) | latency, concurrency 1, batch 1 | **e2e p50 20.94 s** |
| 230M | Xeon E5-2690v3 (cpu) | single forward (wall) | **463 s** (perf artifact skipped — a warmed scenario run would take hours on this 2015 Haswell; the wall measure stands) |
| both | Intel NPU (AI Boost) | 8192 latency, batch 1 | **measured** (2026-07-29) — 230M **15.09 s** / 350M **15.83 s** f16 p50 e2e; int8/int4 within ~7%. See "NPU measured — Intel AI Boost" below |

Notes (honesty ledger):
- The `chat` workload SLO (ttfa 2 s) is not meaningful for an 8k encoder on
  this hardware — goodput 0 is the workload label's artifact, not a failure.
- **Found + fixed during measurement**: (1) budgeting non-schedulable GPUs let
  placement pick an adapter the process couldn't see; (2) the resident padded
  partial groups to the full batch (repeat-padding), inflating single-request
  service ~4× at batch 4 — the model now prebuilds one forward per group size
  and every group runs at its true size (`fwd_variants`); (3) `latency`
  defaults to concurrency 8 — the floor is measured at `--concurrency 1`.
- Optimization evidence for the deferred items: at 8k the forward is
  compute-bound (GPU 100%, linear concurrency scaling) — `flash_attn_bidir`
  autoselect and kernel-level attention work are where wins would come from,
  not batching policy. *(That prediction held: the flash autoselect landed on
  2026-08-04 at ~4× the GEMM attention — 3.97–4.35× over three independent
  runs; see the A/B section below.)*

### First honest 8k wall-times (single sequence, chunk 1024, fp32)

| model | device | load | forward (8192 tok) |
|---|---|---|---|
| 230M | P40 (gpu0) | 3.0 s | **22.0 s** |
| 350M | P40 (gpu1) | 6.2 s | **20.6 s** |
| 230M | Xeon E5-2690v3 (cpu) | 2.1 s | **463 s** |

Not tuned: attention kernels are the naive materialized trio per chunk;
`flash_attn_bidir` autoselect and an NLC-native depthwise conv are deferred
behind perf evidence (this table is that evidence's baseline). The blog's
"28 s at 8k on CPU" was a modern CPU — this Haswell box is ~16× slower than
that; report what is measured.

- **P3 gate PASSED** (2026-07-29): `make gradcheck` green incl. `check_lfm`;
  finetune (60 steps, seq 1024, b=2, gpu0, ~35 min wall incl. GPU contention)
  improved held-out metrics on the target corpus: pseudo-ppl **13.85 → 9.93**,
  masked-accuracy **64.75% → 68.03%** (identical val batches, seed-fixed).
- **P5 (partial, 2026-07-29)**: `LfmResident` (tokenize-on-dispatcher
  pipelining, exact-length instance keys, TRUE equal-length batched forward
  with YOLO-style repeat padding) registered in `build_executor`;
  `lfm::caps::manifest_resident()` (weights/tokenizer are service config).
  **D-Bus gate passed**: `examples/embedding/embed_document.py` round-tripped
  a 3547-token document as a memfd → `[3547,1024]` f32 states back as a memfd;
  4 concurrent requests completed simultaneously in 2 batched groups (b=2) —
  batching observed, results identical. (Timing polluted by the concurrent
  finetune; honest numbers come from the perf ladder.)
  **perf**: `ExecutorTarget` added to `crates/perf` (any resident model gets
  honest concurrency benchmarks through the real scheduler);
  `--target lfm:<weights>:<tokenizer>` arm; `make perf/lfm` standalone target
  (LFM_WEIGHTS/LFM_TOKENIZER/LFM_INPUT selectable). Smoke passed on gpu0.
  Remaining in P5: `residency::Device::Npu` arm; kmask+zeroed-pads batching.

- **P6 — NPU export** (2026-07-29): `crates/npu/src/lfm_topology.rs` (kmask
  graph input, in-graph statically-unrolled query-chunked attention above
  S=2048, depthwise symmetric Conv, GQA expand, RoPE tables, per-head QK-norm)
  + `lfm_export.rs` + `LfmSession` (ids+kmask→hidden; real + stub) +
  `brain npu lfm --weights F --seq S --out M [--int8]`. Both models exported at
  1024 AND 8192 (chunked emission verified structurally); op histogram clean
  (667 nodes, all NPU-friendly). **OpenVINO-CPU parity: cosine 1.000000,
  max_abs 0.00000** (pip `openvino` wheel; needs unversioned `.so` symlinks in
  `site-packages/openvino/libs`). NPU-device compile/parity/bench: Core Ultra
  box only. Core evolution: `linear_quant` hoisted into `TopoBase` (chronos2
  migrated onto it; qwen/glm/fincast/kronos copies remain — follow-up).
- **P4 — 8k training** (2026-07-29): `block::chunked_bidir_bwd` (per-chunk
  score/softmax recompute; new `attn_bwd_{dk,dv}_cross_acc` kernels with an
  `acc_flag` — chunk 0 assigns, no clears drift) + gathered supervised-row MLM
  head (`row_scatter` kernel with sentinel-skip padding; head dW from gathered
  rows; d_xn zeroed via submit clears). `Lfm::new_train_chunked` /
  `load_train_chunked`; `brain lfm finetune --seq >2048` auto-selects it.
  **Equivalence gate green**: chunked-training loss identical and all 29
  parameter grads match the materialized regime (4 chunks + tail, partial
  supervision). Found+fixed: scatter pad slots colliding on row 0 (sentinel
  skip); sentinel breaking the forward gather on the JIT backend (split
  gather/scatter index buffers). **8k gate PASSED**: 4 MLM training steps at
  seq 8192, b=1 on one P40 — 16.8 GB resident (the memory design's estimate
  held), loss 2.29 → val 2.24, checkpoint saved. ~3.5 min/step at 8k on a
  contended card (clean per-step timing comes with the benchmark pass).
- **P5 finish** (2026-07-29): `residency::Device::Npu(u32)` + `Budgets::npus()`
  + NPU placement arm (RAM-costed compiled blobs; no lane unless a budget is
  set) — all 21 residency tests green.

## Done (continued)

- **P3 — backward + gradcheck + MLM** (2026-07-29): backward emitted entirely
  from shared builders (`bidir_bwd`, `kv_expand_bwd` group-sum, conv1d dx/dw +
  adjoint permutes, `rmsnorm_eps_bwd`, tied-head + `emb_bwd` accumulation,
  `ce_stats` big-vocab CE). **`gradcheck::check_lfm` green on first run**
  (conv+attn+conv tiny stack, MLM-style IGNORE labels, 4e-3/8e-2). `data::mlm`
  (30% / 80-10-10, deterministic, unshifted) with rate/determinism/specials
  tests. `lfm::train::finetune` loop (cosine+warmup, masked-acc + pseudo-ppl
  eval), `eval::mlm::lfm_mlm_eval`, CLI `brain lfm {data,finetune,eval}`.
  `Model`/`ModelConfig` impls → generic trainer + blanket gradcheck.
  Finetune gate (seq 1024, b=2, gpu0) running.

### Optimization pass (2026-07-29, post-benchmark)

Profile-driven (BRAIN_PROFILE: attn_apply_cross 51% + attn_scores_cross 24% of
the CPU forward; GPU scaling fits isolated the same t² term):

| model | device | 8k forward | before | speedup |
|---|---|---|---|---|
| 230M | Tesla P40 | **7.5 s** | 22.1 s | **3.0×** |
| 230M | Xeon E5-2690v3 | **23.2 s** | 463 s | **20×** — beats the blog's 28 s modern-CPU figure on a 2015 Haswell |

- CPU: backend-cpu native fast paths route the cross-attention trio through
  the AVX2 `matmul_abt` (per-head packs) + rayon softmax.
- GPU: `block::gemm_bidir_fwd` — per-head packed operands through
  `matmul_reg2` (GQA replication + 1/√hd folded into `head_pack`; `kv_expand`
  eliminated on this path) and `softmax_rows` (workgroup-per-row); gated on
  `DeviceCaps::workgroup_reductions`. **Superseded on 2026-08-04 by fused flash
  attention — see the A/B below.**
- All parity/equivalence gates re-run green after each step (cosine 1.000000).
- FLOP accounting landed (`brain flops`, `Gpu::cost_of`/`ops_counters`): 100%
  pipeline coverage for qwen/gpt/lfm at merge; the GEMM-attention pack kernels
  added after the registry report as UNCOVERED in GPU online counts until
  costed (honest by construction) — follow-up.
- Remaining headroom: ~9% of P40 peak now — next candidates are reg2 tuning
  for K=64 shapes and conv/permute fusion. 350M re-measure and a chunked-regime
  builder for `brain flops` (the materialized build OOMs at --block 8192) are
  follow-ups.

### Attention A/B: flash vs GEMM at 8k (2026-08-04) — **flash wins ~4×**

The open item "`flash_attn_bidir` autoselect" is **closed, measured**. It had
been left open behind a WRONG conclusion: the earlier "flash measured ≈ naive
here" was taken against `flash_attn_bidir`, the **baseline** kernel, whose
per-thread `var<function> q[128]`/`o[128]` arrays fall out of registers and run
at local-memory bandwidth. `flash_attn_bidir_split` computes the same thing to
cosine 1.0000000000 and is a different kernel entirely at `head_dim = 64` —
which is exactly lfm's config.

Harness: `crates/lfm/src/bin/lfm_attn_ab.rs`
(`cargo run --release -p brain-lfm --bin lfm_attn_ab -- [T] [reps]`). It drives
`model::block` directly at the real shape (`lfm25_encoder_350m`: d_model 1024,
16 heads, 8 kv heads, head_dim 64, chunk 1024 — the chunk `caps::SLAB_BUDGET`
picks at T=8192), best-of-5, fp32, single Tesla P40. The flash rows **include**
the three `kv_expand` dispatches that build the fused `[q | k_exp | v_exp]`
slab, which the GEMM path does not need — so the comparison is like-for-like
and not flattered.

| path | T=8192 | GFLOP/s | % P40 peak | vs GEMM | T=4096 | vs GEMM |
|---|---|---|---|---|---|---|
| `gemm_bidir_fwd` (was) | 1185.10 ms | 232 | 1.97 % | 1.00× | 258.63 ms | 1.00× |
| `kv_expand` + `flash_bidir_split` | **274.17 ms** | **1003** | **8.53 %** | **4.32×** | **68.35 ms** | **3.78×** |
| `kv_expand` + `flash_bidir` (baseline) | 3722.48 ms | 74 | 0.63 % | 0.32× | 949.81 ms | 0.27× |
| `kv_expand` alone | 1.07 ms | — | — | — | 0.86 ms | — |

At T=4096 the split kernel is 3.78× and at T=2048 3.88× (65.88 ms → 16.99 ms);
its GFLOP/s is flat at ~1005 across all three lengths while the GEMM path's
falls with T, which is why the gap widens with context.

Agreement on the same inputs: GEMM vs split **cosine 1.0000000000**, max_abs
1.9e-6; baseline vs split cosine 1.0000000000, max_abs 2.3e-6.

**Independently re-measured three times, and the ratio is not stable — the two
absolute numbers are.** Same box, same binary, `BRAIN_DEVICE=gpu1`, best-of-5:

| run | `gemm_bidir_fwd` | `flash_bidir_split` | ratio |
|---|---|---|---|
| the port | 1185.10 ms | 274.17 ms | 4.32× |
| adversarial re-run | 1194.77 ms | 274.73 ms | 4.35× |
| integration re-run | **1090.09 ms** | **274.68 ms** | **3.97×** |

The split kernel reproduces to **0.2 %** across all three; the GEMM baseline
swings **9 %**, and the whole spread in the headline ratio comes from it. So the
defensible claim is **"about 4× (measured 3.97–4.35×)"**, not "4.32×", and the
`flash_bidir_split` wall-time of **~274.7 ms at T=8192** is the number to quote.
The direction is never in doubt: the smallest observed win is still 3.97×, and
the baseline flash kernel is 0.29–0.32× — i.e. 3.1–3.4× *slower* than GEMM — in
every run.

Note the bottom row: the **baseline** flash kernel is 3.1× SLOWER than the GEMM
path. That is the number the old ledger entry was really reporting, and it is
why the wiring gate in `model.rs` is *"`flash_bidir_variant` resolves to the
SPLIT kernel"* and not *"the device is cooperative"* — on a device that can only
run the 64-thread baseline, `gemm_bidir_fwd` is still the right choice and is
kept.

Wired: `crates/lfm/src/model.rs` registers `flash_attn_bidir_split` (slot 56),
`Lfm::flash_selectable` is the gate, and the chunked forward re-enables the
three `kv_expand` dispatches (`emit_attn_qkv(build_fused = true)`) that the GEMM
path skipped. Follow-up not taken here: with flash selected, the chunked
regime's `[H, chunk, T]` `scores`/`probs` slabs (536 MiB **each** at 8k) are
allocated and never written — freeing ~1 GiB of VRAM needs a change to
`Regime::Chunked`'s allocation, which the probe/`fill-mask` paths also read.

### NPU measured — Intel "AI Boost" (Core Ultra) via OpenVINO (2026-07-29)

The `Intel NPU` row is now measured, on a **different box** from the P40/Haswell
machine: **Intel(R) AI Boost** NPU (`/dev/accel/accel0`), OpenVINO **2026.2.0**,
level-zero NPU driver 1.35 / loader 1.28, kernel 6.17. Tool: **`brain npu
lfm-bench`** — export the fixed-shape graph, compile on the NPU with
`allow_fallback = false`, **assert `device() == "NPU"`** (a CPU/GPU fallback is
never reported as an NPU number), then 5 warmup (excluded) + **20 timed `run()`**
calls → p50/p99/mean. Compile time is measured **separately** — one-time, never
folded into inference. Reference for `--compare` parity is brain's own chunked
fp32 forward on identical token ids.

The Intel NPU is **fp16-native**, so the dense fp32 graph executes in fp16 — the
`f16` column. `int8`/`int4` are per-output-channel **weight-only** quantization
(activations stay fp16). All three compile and run on this NPU.

**8192-token forward — the comparable stat (p50 e2e per inference):**

| model | f16 | int8-w | int4-w | (P40 f16) | (Haswell f16) |
|---|---|---|---|---|---|
| 230M | **15.09 s** | 14.22 s | 14.00 s | 7.5 s | 23.2 s |
| 350M | **15.83 s** | 14.94 s | 14.41 s | 20.6 s | — |

- **230M @ 8k on the NPU sits between the P40 (~2× faster) and the 2015 Haswell
  (~1.5× slower than the NPU)** — a ~40 W accelerator between a 250 W GPU and a
  48-thread server CPU. The NPU's argument is perf-per-watt + host offload, not
  peak throughput.
- **350M ≈ 230M at 8k** (15.8 vs 15.1 s) despite being larger: both have exactly
  **6 attention layers**, and at 8192 the O(S²) bidirectional attention dominates
  — 350M's extra conv layers + wider FFN add only ~5%.
- **Weight quant barely helps at 8k** (int8 −6%, int4 −7% vs f16): this encoder is
  **compute-bound on fp16 attention**, not weight-bandwidth-bound like AR decode.
  Quant's payoff here is footprint (≈4×/8× smaller weights), not latency.

**230M f16 scaling ladder (p50):**

| seq | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| p50 | 0.60 s | 1.76 s | 4.98 s | 15.09 s |
| compile (1-time) | 21.7 s | 47.7 s | 162.8 s | 331.4 s |

Near-quadratic in S — the same compute-bound shape the GPU/CPU ladders show.

**1024-token quant anchor (p50):** 230M f16 0.604 / int8 0.497 / int4 0.544 s;
350M f16 1.120 s. (At short S int4 > int8 — dequant overhead, not yet amortized.)

**Parity — NPU fp16 vs brain fp32 (`--compare`, cosine):** 230M @1024 **0.999905**,
@8192 **0.992796**; 350M @1024 **0.999925**. Cosine ≥ 0.9999 at 1024 (passes the
strict gate); **0.9928 at 8192** is honest fp16 accumulation over 8k
bidirectional-attention keys (>99% aligned — not a defect; for exact-reference
work use the fp32 GPU/CPU path).

**Compile is heavy + one-time** (separate from latency; cache with `--cache-dir`):
230M f16 8192 331 s, int8 263 s, int4 255 s; 350M f16 8192 363 s, int8 303 s,
int4 263 s. Small buckets compile in seconds (1024 ~13–36 s, 4096 ~163 s).

## Remaining

- P4: 8k training — accumulating `attn_bwd_{dk,dv}` variants + per-chunk
  recompute backward; masked-row gather before the head (logits `[8192,65536]`
  f32 = 2.147 GB exceeds the 2 GiB binding limit).
- P5 (expanded per user direction): residency (`LfmResident`, `run_batch`
  length-bucketed batching) + registration in `resident.rs::build_executor` so
  the **generic D-Bus interface** serves lfm (`Run`/`Subscribe`, fd-passing);
  **staged pipelining** (FastVLM two-lock pattern: tokenize/CPU → encode/GPU →
  head so request N+1 overlaps N); `residency::Device::Npu`; batched padding
  (zeroed pad states + additive kmask); **`examples/embedding/` Python D-Bus
  client** that sends a long context as a file descriptor (memfd) and reads
  the embedding result back via fd.
- P6: NPU export (`lfm_topology.rs`, kmask input, in-graph chunked attention,
  bucket ladder {1024, 2048, 4096, 8192} + precompile); OpenVINO-CPU parity
  here. **DONE** — NPU parity + bench measured on the Intel AI Boost box via
  `brain npu lfm-bench` (f16/int8/int4, both models); see "NPU measured" above.
- P7 (expanded per user direction): `brain perf` integration — `--target lfm:`
  CapabilityTarget arm + a **resident-backed target driving the residency
  executor** (scheduler + budgets + lanes) for honest concurrent-request
  numbers; the benchmark **selectable per model and runnable standalone**
  (`brain perf run serve --target lfm:…` / `make perf/lfm`) inside the brain
  perf suite; concurrency ladder on this hardware, then optimize from the
  evidence (flash_attn_bidir autoselect — **done 2026-08-04, ~4×**; batching
  policy); NPU PerfTarget
  prepared for the Ultra box. Deliverable: model × device × concurrency table
  at `--input 8192`.

## Known gaps / caveats

- ~~`tests/chunked_equiv.rs::chunked_matches_materialized` FAILS on a real GPU
  (P40, wgpu): a bind-group buffer offset of 3200 B violates the device's
  `min_storage_buffer_offset_alignment` of 256.~~ **FIXED as a side effect of
  wiring flash attention (2026-08-04).** The offending mid-buffer binds were
  `gemm_bidir_fwd`'s per-head/per-chunk pack slices; `flash_bidir_fwd` binds only
  at span starts (`row0 · stride`), which are whole-sample offsets and therefore
  256-aligned by construction. Measured on gpu1: `hidden max|Δ| = 7.15e-7`,
  `test chunked_matches_materialized ... ok`.
  **The underlying defect is NOT fixed** — it is only no longer on the selected
  path. A device that can run only the 64-thread baseline flash kernel still
  falls back to `gemm_bidir_fwd` and will still hit it. Aligning the chunk-row
  offsets (pad or copy) remains owed.

- Unmasked padding is unsound (bidirectional) — exact-length builds only.
- Ragged batching not yet wired (conv mixer needs equal-length rows; spans
  exist for attention; YOLO-style length bucketing planned at the resident).
- `bench::DecoderLm` seam is causal-only; MLM encoder not in `arch_registry()`.

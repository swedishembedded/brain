# FLUX.2 Klein — status ledger

Chronological, measured-only. Reference material + the phased plan live in
`/…/resources/flux/` (outside the repo); goldens in `testdata/flux2/`.

## P0–P1 (2026-07-30) — goldens, import

- `tools/flux2_dump_reference.py` dumps stage goldens from the diffusers
  pipeline (fp32 CPU): schedule sigmas, text hidden taps + concat, VAE
  moments/packed/decoded, transformer I/O captured by forward hooks (t2i and
  one-ref editing), per-step e2e latents + final image. All f32.
- Import: diffusers 4B layout (169 tensors) → 149 BFL-canonical fused tensors,
  full two-way coverage validation; BFL single-file and BF16 GGUF (9B) paths.
  Real-checkpoint test green. `qwen3_8b()` preset added.
- klein-9B weights were fetched, verified byte-complete, then removed to
  reclaim disk (user request); re-fetch URLs recorded in resources README.

## P2 (2026-07-30) — conditioning parity

- VAE: quant/post-quant convs + `latent::pack/unpack` (2×2 unshuffle order
  `c·4+pi·2+pj`, eval-BN eps 1e-4). **Encode cosine 1.000000, pack max_abs
  0, decode cosine 1.000000** vs goldens (CPU backend). zimage suites
  unaffected (22 suites green).
- Text: `encode_hiddens_padded` — one forward, taps [9,18,27], pad keys
  masked via the new `gqa_scores_kmask` kernel. Measured motivation: with
  unmasked pads, pad-row features diverge up to max|Δ|≈6.4e3 while content
  rows are bit-identical; the DiT consumes all 512 rows, so masking is
  parity-critical. **All-row cosine ≥ 0.9999 per tap and for the 7680-wide
  concat**; tokenizer + `<think>`-suffix template reproduce reference ids
  exactly.

## P3 (2026-07-30) — DiT forward

- `flux2::Flux2Model`: joint slab layout, `step_sliced` row-range ops
  (offsets are **f32 elements**, not bytes — found by SIGSEGV, fixed),
  global modulation folded into LN gamma/beta (6 pairs + 5 gates per
  forward), fused weights split at import.
- Found + fixed: `silu_mul` takes a single `total` param — passing `[n, mlp]`
  computed only the first `n` elements (parity cosine 0.504 → 1.000000).
- **Forward parity: cosine 1.000000 (max_abs 1e-4) on t2i (1536 joint
  tokens) and 1.000000 (2e-4) on the editing case (2560 tokens)**, replaying
  the reference's own captured inputs/ids. Position-id builder matches the
  reference layout exactly (incl. ref t-offsets). CPU backend, fp32.

## P4 (2026-07-30) — sampling pipeline

- `diffusion::scheduler::{empirical_mu, time_shift_exponential, klein_sigmas}`:
  **exact to <2e-6** against reference sigma vectors for (512²,4), (1024²,4),
  (1024²,50), (768×1360,4) — including the step-count-dependent branch.
- `flux2::Pipeline` (env-only paths), `brain flux2 generate` CLI (PPM in/out,
  `--ref` editing, `--variant`, CFG for base), Makefile targets.
- **Composed-loop parity: replaying from the reference's post-step-0 latent,
  per-step latent cosine 1.000000 (steps 1–3) and decoded-image cosine
  1.000000 (max_abs 2e-4)** — text→DiT→Euler→decode proven as a system.
- `model::hostmath::randn` hoisted from zimage (shared noise source). Host
  RNG is xorshift+Box-Muller — statistically equivalent to torch Philox,
  not bit-identical (documented).

## P5 (2026-07-30) — serving (contract sign-off, `docs/serving-contract.md`)

- **Capability ✓** — `flux2::caps` (`crates/flux2/src/caps.rs`), model id
  `flux2-klein`: `text2image` / `edit` / `lora_train`, all streaming, all
  polling `inv.cancel` per denoise/training step; images via the shared
  `capability::blob` codec; `lora_train` returns the adapter as an Outcome
  blob. Static weight-free manifest (`brain caps flux2-klein` works with no
  weights; unit test `manifest_declares_the_full_surface`). `adapter` param
  wired through `Pipeline::build_adapted` (LoRA fold-in at build). 9B
  variants gated on `BRAIN_FLUX2_ALLOW_NC=1` (FLUX.2 [Non-Commercial]
  License; attribution notice printed once when enabled).
- **Residency ✓** — `crates/cli/src/resident_flux2.rs`, registered in
  `resident::build_executor`, env-gated on `BRAIN_FLUX2_DIT` / `_VAE` / `_TE`
  / `_TOKENIZER`. Instance key `{variant}:{w}x{h}:{nref}[:{adapter}]`
  (generation) / `train:{variant}`; `activate` pins `BRAIN_GPU_INDEX` and
  builds the `Pipeline` once per instance. Estimates are measured-single-run
  placeholders (4B fp32 ≈ 18 GiB VRAM + 2 GiB RAM; 9B ≈ 36 GiB; train
  20 GiB RAM — host f32 trainer) — re-measure via `brain perf` when the perf
  target lands.
- **Batching = documented-sequential** (the contract's explicit-reason
  path) — `run_batch` is a sequential loop with the reason in the comment: a
  true batched MMDiT forward needs the joint-sequence device graphs rebuilt
  for N latents and per-request seeds/steps/CFG diverge the trajectories; the
  scheduler already groups same-key jobs, so the follow-up touches only
  `Flux2Instance::run_batch`.
- **D-Bus + examples ✓** — reachable over the existing
  `Run`/`Subscribe`/`Cancel` surface (no surface change needed);
  `examples/imagegen/`: `generate.py` (streaming t2i → PPM), `edit_image.py`
  (memfd input blob), `lora_finetune.py` (live loss + adapter fd + adapted
  generation), `cancel_generation.py` (Cancel after the 2nd progress frame →
  `cancelled` error frame). `brain_py.dbus` gained `subscribe_with_job` so
  clients can cancel.
- Verified without weights: `cargo test -p brain-flux2 --lib` green;
  `brain caps` lists `flux2-klein` with all three actions (static manifest).
  End-to-end weighted runs of the Python examples pending a weights + bus
  session.

## P6 (2026-07-30) — `brain perf` target + first measured numbers (CPU)

- **Perf target ✓** — `--target flux2[:<W>x<H>x<steps>]` (default 512x512x4;
  `crates/cli/src/perf_cli.rs::build_flux2`), **resident-backed**: an
  `ExecutorTarget` over a `residency::Executor` running `Flux2Resident`
  (scheduler + budgets + device lanes — the D-Bus serving path), never a bare
  provider, so concurrency>1 measures the real scheduler. New
  `ExecutorTarget::new_streaming` (`crates/perf/src/targets.rs`) timestamps
  every in-flight `"denoising"` `Progress` as one artifact
  (`artifact_unit = "denoise_step"` — the `CapabilityTarget` contract brought
  to the executor seam; the plain `ExecutorTarget` is one-shot and only marks
  `Admitted` on first progress); "encoding"/"decoding" callbacks are
  bookkeeping, not output units (unit test
  `streaming_executor_target_counts_only_accepted_progress_as_artifacts`).
  `Pipeline::generate` reports progress at step **start**, so TTFA = queue +
  prompt encode, each IAL gap = one denoise step, and the final step + VAE
  decode land between the last artifact and `Done` (inside e2e). Budgets
  cover ONLY schedulable devices (`--device`/`BRAIN_DEVICE` narrowing — the
  guard the LFM ledger's 6× llvmpipe regression mandated). `make perf/flux2`
  (`FLUX2_SIZE`/`FLUX2_REQUESTS`/`FLUX2_WARMUP`).
- **Measured** — device named exactly: **CPU backend (`BRAIN_DEVICE=cpu`,
  `cpu[48 core(s)]`, brain-wgsl-cpu Cranelift JIT + AVX2 fast paths) on an
  Intel Xeon E5-2690 v3, 184 GB RAM** — the parity-proven backend. klein-4b
  fp32, 256×256, 4 steps, txt_len 512 (joint seq 768 tokens), scenario
  `latency`, concurrency 1, 1 warmup (absorbs the pipeline build/weight load)
  + 2 measured requests, release build, seed 1234
  (`results/perf-latency-flux2-cpu-256-clean.json`):

  | metric | p50 | worst (n) |
  |---|---:|---:|
  | TTFA (queue + prompt encode → first step start) | 48.2 s | 53.9 s (n=2) |
  | per denoise step (IAL) | 54.3 s | 64.6 s (n=6) |
  | TPOA | 54.1 s | 60.7 s (n=2) |
  | e2e — one 256² image | 266.4 s | 292.1 s (n=2) |

  ≈ 0.014 denoise_steps/s, i.e. **~4.4–4.9 min per 256² image** on this CPU.
  Peak RSS ~83 GB (everything fp32 — DiT + Qwen3-4B TE + VAE + import
  scratch), so the `estimate()` placeholders remain to be re-measured per
  variant (sweep still pending).
- Honesty notes: **CPU-only — the GPU (P40) path is unmeasured for flux2**;
  no GPU numbers exist and none may be extrapolated from this table. No
  fidelity gate is wired for this target, so the artifact reports
  `correctness.passed: null` ("unverified") — numeric parity is gated
  separately (forward cosine 1.000000, P3/P4). The `chat` workload SLO labels
  are meaningless at this scale — `goodput 0` is the label's artifact, not a
  failure. A first run overlapped a concurrent generation on this box: its
  slower request (e2e 349.5 s, IAL max 90.6 s vs 55 s clean) is contention,
  kept only as `perf-latency-flux2-cpu-256.json`; the quiet-box rerun above
  is the record.

## P7 (2026-07-30) — GPU path + canonical device placement

- Found + fixed: wgpu accrues a staging copy per `write` until a submit-side
  reclaim; on the non-ReBAR P40 the 15.5 GiB DiT upload OOM'd at ~22 GiB.
  Periodic 1-element readback flush (now shared via `paramstore`) caps VRAM at
  the true weight footprint.
- **GPU parity (P40, flash + reg2 fast path): cosine 1.000000 / max_abs 1e-4
  (t2i) and 1.000000 / 2e-4 (edit)** — identical to the CPU backend.
- Two-card layout via the canonical device registry (`docs/engine/devices.md`):
  DiT on gpu0 (14.7 GiB steady), truncated fp32 TE (layers 0..=27) on gpu1
  (16.3 GiB steady; above the naive estimate — non-ReBAR allocator overhead).
- **First GPU generation: 512², 4 steps, 59.8 s** (789.7 s on CPU — 13.2×);
  120 s total process incl. weight load/build. Next levers: resident instance
  (amortizes the build), int8 DiT (zimage `int8.rs` pattern, single-card
  DiT+TE), recorded phase graphs.

## P8 (2026-07-30) — int8 (DP4A) path: a capacity win, NOT a speed win

Quantizer hoisted to `model::int8` — zimage and qwen now delegate to the one
engine-wide implementation (net −48 lines, two duplicates retired).

**Parity** (P40, replayed golden inputs, `tests/int8_parity.rs`):
int8 vs fp32 **cosine 0.998950**, max_abs 0.81; int8 vs the diffusers golden
identical to 6 dp. Generated image vs fp32 at the same seed: mean pixel
delta 4 %, visually indistinguishable. Three-and-a-half tensor families stay
fp32, each chosen by a measured bisection (`int8_bisect_keep_f32_families`,
`#[ignore]`d as a tool): `txt_in` (raw Qwen3 hidden-state outliers crush a
per-token scale: 0.995 → 0.984 if quantized), `img_in` + `final_layer.linear`
(3 MB of boundary insurance), double-block `*_mlp.2` (0.9965 → 0.9989 for
~850 MB).

**Speed — the headline negative result:**

| config | single forward @1536 tok | 512² 4-step image | VRAM |
|---|---|---|---|
| fp32, two cards (DiT gpu0 + TE gpu1) | 12.873 s | 59.8 s | 14.7 + 16.3 GiB |
| int8 DiT + int8 TE, **one card** | 11.654 s | **54.5 s** | **5.6 GiB** |
| speedup | **1.10×** | 1.10× | 5.5× smaller |

> **Superseded by P9.** Every timing in this section predates the attention
> fix; the fp32 forward is now 2.757 s and the fp32 512² image 18.3 s. The
> *analysis* below stands and is what P9 acted on — read it as the diagnosis
> that led there, not as current numbers.

DP4A promises ~4×; we measured 1.10×. Cause: **neither path is
arithmetic-bound.** The forward is 10.17 TFLOP; fp32 achieves 0.79 TFLOPS =
**6.7 % of the P40's 11.8 TFLOPS**, int8 0.87 TOPS = **1.9 % of its 47 TOPS**.
Quadrupling arithmetic throughput cannot help when 93 % of the arithmetic
already sits idle. (This corrects an earlier roofline note in this ledger that
called the batch-1 forward compute-bound: arithmetic *intensity* is high, but
the achieved rate says the limiter is elsewhere.)

So int8's real value is **capacity**: the whole model on one 24 GB card,
freeing the second P40 for a parallel replica (≈2× throughput even at
unchanged latency). Speed must come from kernel efficiency instead.

First hypothesis for the profiling work: `matmul_reg2`'s 128×128×BK8 tiling
gives an arithmetic intensity ≈32 FLOP/byte against the P40's ridge point
≈33 — right at the bandwidth boundary, so a deeper K-block should move it into
compute-bound territory. But 6.7 % is far below even the bandwidth-bound
ceiling, so a per-kernel profile must name the dominant cost before tuning.

## P9 (2026-07-30) — profiling the DiT forward: the limiter was ATTENTION, not the GEMMs

P8 left the forward at **6.7 % of the P40's fp32 peak** and named
`matmul_reg2`'s tiling as the first hypothesis. A per-kernel profile killed
that hypothesis and found a different, much larger one.

### The harness

`crates/flux2/src/bin/flux2_bench.rs` (modelled on `zimage_bench train`): the
DiT graph's cost depends only on *shape*, so the bench replays the exact
dispatch sequence of `Flux2Model::forward` over correctly-shaped scratch — no
15.5 GiB checkpoint, profile in seconds, and each shape class timeable in
isolation. Modes: `mm` / `mm3` (standalone GEMM per shape), `floor`
(per-dispatch cost), `norm`, `flash`, `replay` (the whole graph, per group).
Its analytic FLOP total reproduces the real graph's **10 173 GFLOP**, and its
single-submit wall time reproduces the real forward to 1 % (12.766 s replay vs
12.873 s measured), so the replay is a faithful stand-in.

Per-group numbers drain the queue between groups; the drain costs one queue
round-trip each (0.073 ms), and the 284 groups sum to 12.742 s against 12.766 s
for the same steps in one submit — i.e. **the instrumentation overhead is 0.2 %**
and the per-group split is trustworthy.

### Profile BEFORE (klein-4B, 1536 joint tokens = 512 txt + 1024 img, 1×P40)

| kind | dispatches | ms | % of forward |
|---|---:|---:|---:|
| **`flash_attn_bidir`** | 25 | **10 352.8** | **81.2 %** |
| `matmul_reg2` | 213 | 2 075.4 | 16.3 % |
| `rmsnorm_eps` (QK-norm) | 60 | 187.8 | 1.5 % |
| `layernorm` | 41 | 58.1 | 0.5 % |
| rope + `pack_qkv` | 75 | 25.1 | 0.2 % |
| `silu_mul` | 25 | 22.3 | 0.2 % |
| `gate_row` | 60 | 20.7 | 0.2 % |
| **whole graph, one submit** | **499** | **12 766** | 798 GFLOP/s = **6.8 % of peak** |

Attention is 725 GFLOP of the 10 173, and it was taking **10.35 s → 70 GFLOP/s
= 0.6 % of peak**. The GEMMs — the thing P8 suspected — were already at
**35.8 % of peak** and accounted for only 16 % of the time.

### Diagnosis

`flash_attn_bidir` gives every thread one query row and holds that row's
`q[128]` and output accumulator `o[128]` in `var<function>` arrays. 256 f32 per
thread cannot fit Pascal's 255-register file, and both loops are bounded by the
**runtime** `p.head_dim`, so they cannot be unrolled either — naga/the driver
place both arrays in **local memory, which is global-memory backed**. The inner
loop then performs ~3 local-memory accesses (`q[d]`, `o[d]` read, `o[d]` write)
per 2 FLOP = 6 bytes/FLOP. At the P40's 346 GB/s that is a **58 GFLOP/s roof**,
and the kernel measured 70 GFLOP/s — it was running at local-memory bandwidth,
exactly as the arithmetic predicts. This is also why int8 bought only 1.10× in
P8: 81 % of the forward was a kernel no GEMM precision can touch.

### Hypotheses tested and REJECTED (with the numbers that killed them)

1. **`matmul_reg2` tiling / arithmetic intensity (P8's first hypothesis).**
   Wrong on its own terms: AI = `2·BM·BN·BK / ((BM+BN)·BK·4)` — **BK cancels**,
   so AI is 32 FLOP/byte for *any* K-block depth and deeper K-blocking cannot
   change it. And 32 FLOP/byte × 346 GB/s = an 11.1 TFLOP/s roof ≈ the card's
   peak, while the kernel achieves 4.2 TFLOP/s using only ~131 GB/s = 38 % of
   the card's bandwidth. The GEMM was never bandwidth-bound.
2. **Dispatch/submit overhead.** Measured floor: 0.0065 ms per dispatch
   (500 in one submit). The forward's 499 dispatches cost **3.3 ms = 0.03 %**.
3. **Sliced views (`step_sliced` txt-rows-then-img-rows = 2 dispatches).**
   The double-block pair m=512 (2.414 ms) + m=1024 (4.666 ms) = 7.08 ms; the
   merged m=1536 GEMM is 7.016 ms. Merging would save **0.9 %** of the double
   blocks' GEMM time, i.e. ~0.06 ms per pair.
4. **Occupancy on small-m dispatches.** m=512 reaches 34.0 % of peak vs 35.1 %
   at m=1536 — a 3 % effect, not the missing 93 %.

### Fixes

**1. `flash_attn_bidir_split.wgsl` — lane-split flash attention (new kernel).**
Splits `head_dim` across LANES=4 threads so each thread owns 32 channels
indexed by a *compile-time* trip count → real registers. The per-key dot
product becomes a partial per lane, summed through a small shared buffer once
per key tile; the `o` rescale moves out of the per-key loop to once per tile.
Shared layouts are bank-conflict-free by construction (lanes own *interleaved*
channels, so the 4 lanes of a row touch 4 consecutive banks; `part` is indexed
`[j][row][lane]` so a warp spans all 32 banks exactly once). `@workgroup_size(256)`
= BR 64 × LANES 4, gated on the device's **queried** `max_workgroup_size`.

| head_dim | `flash_attn_bidir` | split | speedup |
|---:|---:|---:|---:|
| 128 | 411.7 ms | **14.2 ms** | **29.0×** |
| 96 | 406.6 ms | 18.6 ms | 21.9× |
| 64 | 400.7 ms | 27.8 ms | 14.4× |
| 32 | 235.1 ms | 54.0 ms | 4.4× |

Agreement with the baseline: **cosine 1.00000000, max_abs 1.3e-6**. It wins at
every head_dim, so it is a general replacement, not a wide-head special case;
selection lives in `model::block::{FlashIds, flash_bidir_step}` and both
`flux2` and `zimage` now go through it. **Forward 12.766 s → 2.699 s.**

**2. QK-norm: the existing `rmsnorm_rows`, not `rmsnorm_eps`.** No new kernel —
a dispatch-choice bug. `rmsnorm_eps` gives thread *t* row *t*, so a warp's 32
loads are `head_dim`=128 floats apart and each 32-byte sector fetched serves
one useful float. At the QK-norm shape (36 864 rows × 128): **3.85 ms →
0.20 ms, 10 GB/s → 190 GB/s, 19.4×** (max_abs 8.3e-7 — reduction order only).
`rmsnorm_rows` gained the runtime `eps` its `rmsnorm_eps` twin already had so
there is still ONE workgroup-per-row RMSNorm; its two qwen callers pass 1e-6
explicitly. **QK-norm 187.8 ms → 12.5 ms.**

**3. `matmul_reg3.wgsl` — `matmul_reg2` with its bank conflicts removed.**
Two layout-only changes: threads own *interleaved* rows/columns (stride 16)
instead of contiguous 8-blocks, which turns the B-tile read from a 4-way
conflict into a conflict-free access and makes the epilogue's stores
coalesced; and the tile stride is padded 128→129 so the staging store's bank
index is `(kk+r) mod 32` rather than `r mod 32` (8-way → ~3-way). The K
accumulation order is untouched, so the output is **bit-identical** — measured
max_abs **0.0** across all 12 of the graph's shapes. Added alongside;
`matmul_reg2` remains the default for every other model.

| shape (m×k×n) | count | reg2 | reg3 | speedup |
|---|---:|---:|---:|---:|
| 1536×3072×3072 (sgl qkv/wo_a) | 80 | 7.016 ms | 6.535 ms | 1.07× |
| 1536×3072×9216 (sgl w1/w3) | 40 | 19.774 ms | 19.880 ms | 0.99× |
| 1536×9216×3072 (sgl wo_b) | 20 | 17.341 ms | 15.745 ms | 1.10× |
| 1024×3072×3072 (dbl img) | 20 | 4.666 ms | 3.950 ms | 1.18× |
| 1024×9216×3072 (dbl img w2) | 5 | 13.653 ms | 11.643 ms | 1.17× |
| 512×3072×3072 (dbl txt) | 20 | 2.414 ms | 2.170 ms | 1.11× |
| 512×7680×3072 (txt_in) | 1 | 5.827 ms | 5.308 ms | 1.10× |
| 1024×128×3072 (img_in) | 1 | 0.369 ms | 0.248 ms | 1.49× |
| **whole forward's GEMMs** | 213 | **2151 ms (37.4 %)** | **2050 ms (39.2 %)** | **1.05×** |

The n=9216 shapes (the bulk) do not move — at that width the kernel is limited
by shared-memory *throughput*, not conflicts. Going past ~40 % needs vec4
shared loads or a bigger register block; that is the next GEMM step, not this
one.

### Profile AFTER

| kind | dispatches | ms | % of forward |
|---|---:|---:|---:|
| `matmul_reg3` | 213 | 1 987.3 | 80.0 % |
| `flash_attn_bidir_split` | 25 | 359.2 | 14.5 % |
| `layernorm` | 41 | 57.5 | 2.3 % |
| rope + `pack_qkv` | 75 | 24.6 | 1.0 % |
| `silu_mul` | 25 | 21.9 | 0.9 % |
| `gate_row` | 60 | 20.2 | 0.8 % |
| `rmsnorm_rows` (QK-norm) | 60 | 12.5 | 0.5 % |
| **whole graph, one submit** | **499** | **2 421** | 4202 GFLOP/s = **35.7 % of peak** |

### Measured on the real model (not the replay)

| | before | after | speedup |
|---|---:|---:|---:|
| fp32 forward @1536 joint tokens | 12.873 s | **2.757 s** | **4.67×** |
| int8 (DP4A) forward | 11.654 s | **1.532 s** | **7.61×** |
| fp32 → int8 ratio | 1.10× | **1.80×** | — |
| fraction of fp32 peak | 6.7 % | **35.7 %** | 5.3× |

The int8 ratio moving from 1.10× to 1.80× is the diagnosis confirming itself:
DP4A could not show up while a memory-bound attention kernel owned 81 % of the
forward. int8 is now a speed win as well as the capacity win P8 documented.

### End to end

`brain flux2 generate --width 512 --height 512` (klein-4B, 4 steps, fp32 DiT on
gpu0 + fp32 Qwen3-4B text encoder on gpu1 — the same two-card placement the
59.8 s baseline used):

| | before | after |
|---|---:|---:|
| 512² 4-step image, wall clock | 59.8 s | **18.3 s** (3.27×) |
| ...of which the 4 DiT forwards | 51.5 s | **11.0 s** |
| ...of which text encode + VAE decode | ~8.3 s | ~7.3 s (untouched) |

The DiT is no longer the majority of the wall clock. Reaching the 8-12 s target
now requires profiling the **text-encoder + VAE** half, which this pass did not
touch: at 7.3 s it is 40 % of the remaining time. (A cheap-looking lead was
ruled out by estimate rather than left implied — qwen's prefill forward still
uses the per-element RMSNorm that P9 replaced in the DiT, but at the TE's
shapes that is worth only ~145 ms of the 7.3 s, so the cost is elsewhere and
the TE needs its own profile.)

Gates: `dit_parity` **cosine 1.000000** (t2i and edit fixtures, unchanged);
`int8_parity` cosine 0.998232 vs both fp32 and the golden (gate ≥ 0.998; was
0.998950 — the split kernel's reduction order differs, fp32 parity is
unaffected at 1.000000, and the shift is int8 quantisation noise landing
differently, so the margin narrowed from 9.5e-4 to 2.3e-4 while staying green);
`make gradcheck` OK (29 tensors); the crates this change touches
(`kernels`, `backend-api`, `gpu-core`, `model`, `qwen`, `zimage`, `flux2`)
all green — including zimage's block/model/dev/real parity tests, which is the
regression evidence for the shared `flash_bidir_step` seam. (Those tests run at
toy dims where zimage takes the materialised-trio branch, so the equivalence
evidence for the flash path itself is the kernel-level A/B above: cosine
1.00000000 at four head_dims.)

**Pre-existing failure found, not introduced here:**
`flux2 --test host_forward_parity` panics on a GPU backend with
`Buffer offset 192 does not respect min_storage_buffer_offset_alignment 256`.
Verified identical on the parent commit with these changes stashed; it passes
on `BRAIN_DEVICE=cpu` (no offset rule there), which is how it has been running.
It is the fp32 face of the constraint `new_with` already asserts for int8
("every sliced row offset is 0 or txt_len rows, so widths and txt_len must be
multiples of 64") — the test's toy dims put a `step_sliced` row offset at 48
floats. The fix is to give the test int8-legal dims, or to assert the
alignment in `mm_rows` so it fails with a readable message; tracked below.

## P10 (2026-07-30) — profiling the OTHER half: the VAE decode was 88 % of it

P9 left the 512² generation at 18.3 s with the DiT at 11.0 s and "text encoder +
VAE ≈ 7.3 s, entirely unprofiled". This pass profiled both. The split was not
even: **the VAE decode is 6.47 s of that 7.3 s; the text encoder is 1.23 s.**

### The harness

`flux2_bench` gained two modes beside `replay`:

* **`te` / `tei8`** — replays `Qwen::forward_steps` for the FLUX.2 text encoder
  (the layer-27-truncated Qwen3-4B, ONE 512-token masked-pad prefill, no head)
  over shape-correct scratch, fp32 or the DP4A shard. Same no-weights trick the
  DiT replay uses. `BRAIN_FLUX2_BENCH_BASELINE=1` selects the pre-fix kernel set,
  so before and after come from one binary on one device in one run.
* **`vae`** — profiles the REAL decode graph (`BRAIN_FLUX2_VAE`; the graph is
  built from the checkpoint's tensors, but its cost is still shape-only).

Both use a new generic profiler: every `Step` built through the `gpu_core`
facade carries a `StepMeta` naming its kernel slot, so the per-kind table is
derived from the graph itself rather than hand-annotated, and each kind is
timed by submitting only its own steps. Sum-of-kinds vs the whole graph in one
submit agreed to **0.2–0.6 %** in every run below, which is the instrumentation
bound. A second table splits the dominant kind by uniform params, i.e. by shape.

Validation that the replays are faithful: the TE replay's 1.230 s vs the CLI's
measured 1.33 s `encoding prompt` (the difference is the three tap readbacks),
and the VAE profile's 6.466 s vs the CLI's 6.85 s `decoding` (the difference is
graph construction + weight upload, which `decode_tokens` redoes per call).

### Profile BEFORE (1×P40)

**VAE decode, 32×64×64 latent → 3×512×512** — 150 dispatches, **6.466 s**

| kernel | disp | ms | % of decode |
|---|---:|---:|---:|
| **`conv_bias_reg`** | 38 | **3 546.6** | **54.7 %** |
| **`gn_stats`** | 30 | **2 262.2** | **34.9 %** |
| **`attn_scores_bidir`** | 1 | **564.4** | **8.7 %** |
| `attn_apply_bidir` | 1 | 59.9 | 0.9 % |
| `gn_apply` | 30 | 17.2 | 0.3 % |
| `silu` | 29 | 15.0 | 0.2 % |
| `add2` | 15 | 8.9 | 0.1 % |
| `upsample2` | 3 | 3.3 | 0.1 % |
| `attn_softmax_bidir` | 1 | 2.9 | 0.0 % |
| `nchw_nlc` / `nlc_nchw` | 2 | 0.8 | 0.0 % |

**Text encoder, fp32, 28 layers × 512 tokens** — 532 dispatches, **1.230 s**,
2 954 GFLOP → 2 402 GFLOP/s = **20.4 % of peak**

| kernel | disp | ms | % of prefill |
|---|---:|---:|---:|
| `matmul_reg2` | 196 | 771.1 | 62.7 % |
| `gqa_scores_kmask` | 28 | 275.3 | 22.4 % |
| `rmsnorm` | 112 | 71.8 | 5.8 % |
| `gqa_apply` | 28 | 61.6 | 5.0 % |
| `attn_softmax` | 28 | 34.6 | 2.8 % |
| `silu_mul` | 28 | 8.4 | 0.7 % |
| `add2` | 56 | 3.6 | 0.3 % |
| `rope_base` | 56 | 2.6 | 0.2 % |

**Text encoder, INT8 (DP4A)** — 756 dispatches, **0.770 s**

| kernel | disp | ms | % of prefill |
|---|---:|---:|---:|
| `gqa_scores_kmask` | 28 | 276.1 | 36.0 % |
| `matmul_i8_dyn` | 196 | 260.2 | 33.9 % |
| `rmsnorm` | 112 | 72.1 | 9.4 % |
| `gqa_apply` | 28 | 61.4 | 8.0 % |
| `max_abs_row` | 112 | 43.6 | 5.7 % |
| `attn_softmax` | 28 | 33.9 | 4.4 % |
| `silu_mul` | 28 | 8.2 | 1.1 % |
| `quant_pack` | 112 | 5.2 | 0.7 % |
| `add2` | 56 | 3.4 | 0.4 % |
| `rope_base` | 56 | 2.5 | 0.3 % |

### Diagnosis

Both of P9's lenses fired, in the VAE, at the same time:

1. **One thread per row → `gn_stats`.** The VAE dispatches `G` = **32
   invocations** for a GroupNorm over up to 33 M elements, each serially walking
   its group's ~1 M contiguous floats — 1/120th of the card's lanes, and a
   warp's 32 lanes sit ~1 M floats apart so every 32-byte sector fetched serves
   one useful float. This is *literally* the DIAMOND `gn_stats` bug
   (`docs/performance/overview.md`, "gn_stats was the GPU", 77.6 % of a frame):
   the fix landed for `wm-diamond` in 2025 and the VAE, written later against
   the same kernels, never picked it up.
2. **One thread per (i, j) → `attn_scores_bidir`.** The mid-block attention is
   T = 4096, C = 512, and each thread's `k` reads are a whole 1536-float row
   apart: 17.2 GFLOP in 564 ms = **30 GFLOP/s, 0.26 % of peak**.
3. **`conv_bias_reg` is at its structural ceiling, not a bug.** Measured across
   all 15 distinct conv shapes it is a flat **~700 GFLOP/s (6.0 % of peak)** —
   665 GF/s on the worst, 863 on the best. Its 8×4 register tile does 12 global
   loads per 32 FMAs = 0.75 byte/FLOP, a 461 GFLOP/s roofline that caching
   stretches to ~700. No `var<function>` array, no coalescing fault: to go
   faster the *algorithm* has to change, which is exactly the im2col + tiled
   GEMM trade `docs/performance/p40.md` already took for YOLO on this card.

The text encoder is a different picture: it was already at 20 % of peak with
63 % of its time in the GEMM, i.e. broadly healthy. Its two coalescing faults
(`rmsnorm`, `attn_softmax`) are only 8.6 % of it.

### Fixes

**1. `gn_stats_wg.wgsl` — workgroup-per-group GroupNorm statistics (new).**
256 threads stride one group's elements together (coalesced by construction),
`@workgroup_size(256)` so a group has 8 warps to hide a pure streaming read's
latency, two barriers. Deliberately the SAME two-pass formulation as `gn_stats`
(mean, then squared deviations) rather than `gn_part`'s one-pass
`E[x²] − mean²`, because at 1 M elements per group the cancellation term is not
something to spend on a 0.4 %-of-decode kernel. Selected on
`DeviceCaps::workgroup_reductions`; the CPU JIT keeps `gn_stats`.
**2 262.2 ms → 14.2 ms (159×).**

**2. Mid-block attention as two GEMMs.** The qkv 1×1 conv already emits
**channel-major** `[3C, T]`, which *is* qᵀ/kᵀ/vᵀ — so `v` needs no transpose at
all (it is directly the `[n, k]` operand of the apply GEMM) and q/k need one
cheap `nchw_nlc` each. `scores[T,T] = q·kᵀ` and `ctx[T,C] = probs·v` then run on
`matmul_reg3`. The 1/√C scale, which lived in the per-element kernel's epilogue,
is folded into the `to_q` weight and bias at build time (exact up to fp32
rounding order). No new kernel.
**`attn_scores_bidir` + `attn_apply_bidir` 624.3 ms → 7.8 ms (80×).**

**3. Conv as GEMM — `im2col_at.wgsl` + `matmul_reg3` + `nlc_bias_nchw.wgsl`
(two new kernels).** `y[HW, Cout] = col[HW, CinKK] · Wᵀ`. The **transposed**
orientation (positions as GEMM *rows*, not columns) is the load-bearing choice:
the un-windowed operand for a 512² conv with Cin=256 is `[262144, 2304]` f32 =
**2.4 GB**, over this card's 2047 MiB `max_storage_buffer_binding_size`, so the
whole-image im2col is not even bindable — but with positions as rows a spatial
chunk is a contiguous row range of *both* `col` and the output, so both bindings
are plain sub-ranges and one bounded scratch (512 MiB, `BRAIN_VAE_COL_MIB`)
serves every conv. `nlc_bias_nchw` permutes back and adds the bias in one pass.
Convs with `Cout < 128` stay on the direct kernel (`conv_out`, Cout = 3, would
pay for a full 128-wide tile 42× over).
**`conv_bias_reg` 3 546.6 ms → 930 ms total (3.8×)**: `matmul_reg3` 484.4 ms
(2 481 GFLOP → **5 126 GFLOP/s = 43.6 % of peak**, better than the DiT's 39 %),
`im2col_at` 274.7, `nlc_bias_nchw` 36.3, the three unlowered convs 15.4.

**4. Text encoder: `rmsnorm_rows` and `softmax_rows`, selected on device caps.**
No new kernels — dispatch choices, in `qwen::model` so the blast radius is one
crate. `rmsnorm` **71.8 → 6.3 ms (11.4×)**; `attn_softmax` **34.6 → 8.6 ms
(4.0×)**. `softmax_rows` normalises the whole row while `attn_softmax`
normalises `j <= i`, and here they are identical because `gqa_scores_kmask`
already writes `-3.4e38` into every `j > i` slot (those exponentiate to 0 and
cannot move the max or the sum); no row is ever fully masked, since a query at
position `i` always sees the content keys at `j <= i`, pad queries included.
Applied to the masked-pad path only — that path *is* the FLUX.2 text encoder.

**5. `matmul_reg3` for qwen's forward linears.** Bit-identical to `matmul_reg2`
by construction (P9 measured max_abs 0.0 across 12 shapes), so it is a pure
speed swap: **771.1 → 698.0 ms (1.11×)** on the TE's shapes.
`backend-cpu` gained `matmul_reg3` to its native-GEMM equivalence class, so the
CPU path is unchanged (the one-graph rule).

### Profile AFTER

**VAE decode** — 275 dispatches, **0.875 s (7.4×)**

| kernel | disp | ms | % of decode |
|---|---:|---:|---:|
| `matmul_reg3` | 64 | 484.4 | 55.5 % |
| `im2col_at` | 62 | 274.7 | 31.5 % |
| `nlc_bias_nchw` | 35 | 36.3 | 4.2 % |
| `gn_apply` | 30 | 17.2 | 2.0 % |
| `conv_bias_reg` | 3 | 15.4 | 1.8 % |
| `silu` | 29 | 15.1 | 1.7 % |
| `gn_stats_wg` | 30 | 14.1 | 1.6 % |
| `add2` | 15 | 8.9 | 1.0 % |
| `upsample2` | 3 | 3.3 | 0.4 % |
| `attn_softmax_bidir` | 1 | 2.9 | 0.3 % |
| `nchw_nlc` / `nlc_nchw` | 3 | 0.6 | 0.1 % |

**Text encoder, fp32** — **1.059 s (1.16×)**, 2 790 GFLOP/s = **23.7 % of peak**

| kernel | disp | ms | % |
|---|---:|---:|---:|
| `matmul_reg3` | 196 | 698.0 | 65.8 % |
| `gqa_scores_kmask` | 28 | 272.1 | 25.7 % |
| `gqa_apply` | 28 | 61.2 | 5.8 % |
| `softmax_rows` | 28 | 8.6 | 0.8 % |
| `silu_mul` | 28 | 8.3 | 0.8 % |
| `rmsnorm_rows` | 112 | 6.3 | 0.6 % |
| `add2` | 56 | 3.6 | 0.3 % |
| `rope_base` | 56 | 2.6 | 0.2 % |

**Text encoder, INT8** — **0.668 s (1.15×)**

| kernel | disp | ms | % |
|---|---:|---:|---:|
| `gqa_scores_kmask` | 28 | 267.4 | 40.0 % |
| `matmul_i8_dyn` | 196 | 259.2 | 38.8 % |
| `gqa_apply` | 28 | 61.3 | 9.2 % |
| `max_abs_row` | 112 | 43.7 | 6.5 % |
| `softmax_rows` | 28 | 8.6 | 1.3 % |
| `silu_mul` | 28 | 8.4 | 1.3 % |
| `rmsnorm_rows` | 112 | 7.0 | 1.0 % |
| `quant_pack` | 112 | 6.0 | 0.9 % |
| `add2` | 56 | 3.6 | 0.5 % |
| `rope_base` | 56 | 2.6 | 0.4 % |

### Hypotheses tested and REJECTED (with the numbers that killed them)

1. **"The text encoder is the unprofiled half."** It is 1.23 s of the 7.3 s;
   the VAE decode is 6.47 s. Profiling the TE first would have chased 17 % of
   the problem. (This is why the brief's "MEASURE FIRST" is in the brief.)
2. **`conv_bias_reg` has a coalescing or spill bug.** It does not: no
   `var<function>` array, coalesced 4-position tiles, and a flat ~700 GFLOP/s
   across all 15 shapes — a *structural* 0.75 byte/FLOP ceiling, not a fault.
   The fix had to be algorithmic (im2col + GEMM), not a rewrite of the kernel.
3. **A workgroup-staged tile will speed up `im2col_at` the way it did
   `nlc_bias_nchw`.** Measured **273 → 311 ms — slower.** The transpose's
   uncoalesced side is 8× amplified, so 16.6 KB of workgroup memory buys more
   than the occupancy it costs (158 → 36 ms); im2col's is only ~2.7× amplified
   (3 consecutive taps land in one sector), and there the same trade loses.
   Reverted to element-indexed; the comment in the kernel records the number.
4. **`gqa_scores_kmask` (272 ms, the TE's #2) wants the same GEMM treatment as
   the VAE's attention.** Its shape is too small for the tile: `matmul_reg3` at
   the per-head 512×128×512 runs at **743 GFLOP/s = 6.3 % of peak** (vs 5 126
   GF/s at the VAE's shapes), so 32 heads × 28 layers of it is ~81 ms — a 3.2×
   local win for per-head packing, 896 extra dispatches and a separate
   mask+scale pass. And the *apply* side is worse as a GEMM than it is today:
   512×512×128 measures **317 GFLOP/s**, i.e. 190 ms against the 61 ms
   `gqa_apply` already costs. Left alone; see the remaining list.
5. **Transposing `k` would fix `gqa_scores_kmask` cheaply.** Coalescing its
   `k` reads only moves it to its bandwidth floor: 2.15 GB/layer at 346 GB/s is
   6.2 ms/layer = 174 ms, against 272 ms today — 1.5×, not the 10× the
   coalescing lens usually pays. The kernel needs arithmetic intensity (tiling),
   not just coalescing, and (4) says the tile does not fit this shape.

### End to end

`brain flux2 generate --width 512 --height 512` (klein-4B, 4 steps), same
placements as P9, warm and idle:

| | before | after | |
|---|---:|---:|---|
| **fp32, two cards** (DiT gpu0 + TE gpu1) | **18.4 s** | **12.7 s** | **1.45×** |
| ...text encode | 1.33 s | 1.19 s | |
| ...4 DiT forwards | 10.17 s | 10.20 s | untouched, as intended |
| ...VAE decode | 6.85 s | **1.34 s** | **5.1×** |
| **int8, ONE card** (`--precision int8`, TE `gpu0:i8`) | **13.4 s** | **7.6 s** | **1.76×** |
| ...text encode | 1.04 s | 0.83 s | |
| ...4 DiT forwards | 5.43 s | 5.39 s | |
| ...VAE decode | 6.91 s | **1.33 s** | **5.2×** |

**The text-encoder + VAE half went 7.3 s → 2.5 s (2.9×)**, and the int8 path is
now inside the 8–12 s target the ledger has been carrying since P8. The decoded
images are visually indistinguishable run-to-run: the fp32 image before vs after
is **cosine 0.9999997, max |Δ| = 2/255** over the whole 512×512×3 (the residue
of the reduction-order changes); the int8 image is cosine 0.979 — same scene and
composition, with int8 activation-scale noise landing differently because the
per-token max-abs is taken over a slightly differently-rounded activation.

The CLI now reports the three phases (`encoding prompt` / `denoising` /
`decoding`) after the total, so this table is reproducible from one run.

### Gates

| gate | result |
|---|---|
| `brain-qwen --test flux2_text_parity` (GPU, `BRAIN_QWEN_TE_SHARD=1`) | layers 9/18/27 **cosine 1.000000** all / content / pad; ctx concat 1.000000 |
| `brain-qwen --test flux2_text_parity` (CPU) | same, 1.000000 |
| `brain-vae --test flux2_parity` (`BRAIN_VAE_DEVICE=gpu`) | encode **1.000000**, pack max_abs 0.0, decode **1.000000** |
| `brain-vae --test flux2_parity` (CPU) | same, 1.000000 |
| `brain-flux2 --test dit_parity` (GPU) | **cosine 1.000000** on both fixtures (unchanged) |
| `brain-flux2 --test int8_parity` (GPU) | cosine 0.998130 ≥ 0.998; fp32 2.481 s vs int8 1.352 s (1.83×) |
| `BRAIN_DEVICE=cpu make gradcheck` | OK (29 tensors) |
| `make build` | OK |
| qwen / vae / zimage / flux2 / model / gpu-core / backend-cpu / kernels suites | green |

`flux2_text_parity` grew a `BRAIN_QWEN_TE_SHARD=1` mode that builds the
**layer-truncated shard the pipeline actually runs** (layers 0..=27, no head).
The whole fp32 4B model is ~16 GB of weights, which with Pascal's non-ReBAR
resident overhead OOMs a 24 GB P40 — so the shard is the only way to gate the
GPU kernel selection against the golden at all. (That OOM is pre-existing and
is why the test has been a CPU test.)

`flux2 --test host_forward_parity` still fails on a GPU backend with
`Buffer offset 192 does not respect min_storage_buffer_offset_alignment 256` —
the pre-existing P9 finding, unchanged, green on `BRAIN_DEVICE=cpu`.

## Known gaps / remaining

- Serving landed (P5), perf target + first CPU measurement landed (P6 above);
  training completion tracked as P7–P8 of the execution plan. True batched
  `run_batch` is a follow-up (see P5 batching bullet). A GPU perf run and a
  `sweep` ladder (concurrency behaviour under the sequential-`run_batch`
  scheduler) wait on the GPU path being measured at all.
- **Kernel efficiency** (was the open problem at P8) is now 35.7 % of fp32
  peak after P9. What remains, in profile order:
  - the GEMMs are 80 % of the forward at 39 % of peak; the ceiling for the
    current shared-memory scheme is ~50 % (one shared word per FMA), so the
    next step is vec4 shared loads or a larger register block, not tiling depth;
  - `layernorm` (57.5 ms, 2.3 %) still has the one-thread-per-row coalescing
    problem P9 fixed for RMSNorm — there is no workgroup-per-row LayerNorm
    kernel yet, and adding one is the same 4-20× shaped win;
  - `flash_attn_bidir_split` is at 17 % of peak, itself shared-memory bound at
    one shared word per FMA; two query rows per thread would double its roof.
- True batched `run_batch` still to come; the kernels already carry the hooks
  (attention `bsz`, `film_row`/`gate_row` `rows_per_cond` groups), so
  mixed-progress continuous batching needs no new WGSL.
- Klein-9B-KV cached-ref attention variant: out of scope (needs per-token
  modulation blend, breaks the LN fold).
- `tests/host_forward_parity.rs` cannot run on a GPU backend: its toy dims put a
  `step_sliced` row offset at 48 floats, under the 256-byte
  `min_storage_buffer_offset_alignment`. Pre-existing (see P9); pick dims that
  are multiples of 64, and assert the alignment in `Flux2Model::mm_rows` so the
  next occurrence names itself instead of surfacing as a wgpu validation error.
- The **text encoder + VAE** half was 40 % of a 512² generation and is now
  2.5 s of 12.7 s (fp32) / 2.2 s of 7.6 s (int8) after P10. What remains there,
  in profile order:
  - `im2col_at` is **274.7 ms = 31 % of the VAE decode**, ~4.8× its 57 ms write
    floor (19.6 GB of `col` at 346 GB/s). The element-indexed gather is the
    cost and a workgroup tile made it worse (P10, rejected hypothesis 3); the
    real answer is an **implicit-GEMM conv** that never materialises `col`, i.e.
    a `matmul_reg3` variant whose B-tile load applies the im2col index map.
    That also removes the 512 MiB scratch and `nlc_bias_nchw` (36.3 ms).
  - `gqa_scores_kmask` is **272 ms = 26 % of the fp32 TE and 40 % of the int8
    TE**, at ~110 GFLOP/s. Neither of the cheap fixes pays (P10, rejected
    hypotheses 4 and 5): it needs a *fused, tiled* causal+kmask attention —
    the `flash_attn_bidir_split` treatment with a causal mask and an additive
    key mask — not a repack.
  - `max_abs_row` is **43.7 ms = 6.5 % of the int8 TE**: one thread per row over
    `[512, d_ff]`, the same coalescing shape `rmsnorm_rows` fixes. A
    workgroup-per-row twin is a ~5-line kernel and the last easy int8 win.
  - the VAE's `matmul_reg3` is at 43.6 % of peak, i.e. already past the DiT's
    39 % and near the ~50 % ceiling of one-shared-word-per-FMA.
- `decode_tokens` rebuilds the whole VAE decode graph and re-uploads every
  decoder weight on **every call** (~0.4 s of the CLI's `decoding` phase, and
  it now exceeds a third of the decode itself). Caching the built `VaeDecoder`
  per (lh, lw) on the pipeline is the obvious next win outside the kernels.

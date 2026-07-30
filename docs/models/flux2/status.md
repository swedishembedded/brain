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

## Known gaps / remaining

- Serving landed (P5), perf target + first CPU measurement landed (P6 above);
  training completion tracked as P7–P8 of the execution plan. True batched
  `run_batch` is a follow-up (see P5 batching bullet). A GPU perf run and a
  `sweep` ladder (concurrency behaviour under the sequential-`run_batch`
  scheduler) wait on the GPU path being measured at all.
- **Kernel efficiency is the open problem**: 6.7 % of fp32 peak / 1.9 % of
  int8 peak (P8 above). Per-kernel profiling + autotune at Klein shapes is the
  critical path to the 8–12 s (fp32) / 3–5 s (int8) realistic floors.
- True batched `run_batch` still to come; the kernels already carry the hooks
  (attention `bsz`, `film_row`/`gate_row` `rows_per_cond` groups), so
  mixed-progress continuous batching needs no new WGSL.
- Klein-9B-KV cached-ref attention variant: out of scope (needs per-token
  modulation blend, breaks the LN fold).

# SDXL UNet2DConditionModel — ledger

`crates/unet` + the discrete schedulers in `crates/diffusion`
(`docs/imaging/plan.md` phase **4b**).

**Scope of what is claimed: goldens → import → FORWARD parity, and the
schedulers.** Backward/`check_unet`, the serving contract (a
`capability::Provider`, a residency adapter, a real `run_batch`, D-Bus, an
`examples/` client), the VAE and text-encoder glue, a sampling CLI, batch > 1
and INT8 are **deferred** and are not claimed by any number below.

---

## What was built

| Piece | Where |
|---|---|
| Config + canonical tensor manifest | `crates/unet/src/config.rs` |
| diffusers import, two-way covered | `crates/unet/src/import.rs` |
| Sinusoidal timestep embedding + SDXL added conditioning (host) | `crates/unet/src/hostemb.rs` |
| Forward graph | `crates/unet/src/model.rs` |
| Synthetic weights for the smoke test | `crates/unet/src/init.rs` |
| DDIM / Euler / Euler-ancestral / DPM-Solver++(2M), ε and v-pred | `crates/diffusion/src/discrete.rs` |
| Reference goldens | `tools/goldens/sdxl_dump_reference.py` → `testdata/sdxl/` |

**Zero new kernels and zero new blocks.** Everything convolutional is
`vae::blocks::Builder`; everything transformer is `model::block`
(`layernorm_fwd`, `pick_gemm`, `flash_bidir_fwd`, `bidir_fwd`) over kernels that
already existed (`gelu_erf`, `mul`, `bias_add`, `add_chan_bcast`, `concat2`,
`attn_*_cross`). That was the measured prediction in `docs/imaging/plan.md` §2
that put the UNet family ahead of a bigger DiT; it held.

## Forward parity — what was actually run

One Tesla P40 (`BRAIN_DEVICE=gpu1`, wgpu/Vulkan), `--release`,
`cargo test --release -p brain-unet` (never a workspace `--tests --examples`
build). Weights: the released `stable-diffusion-xl-base-1.0` `unet/`
fp16 variant, read exactly and expanded to fp32.

Import: **1680 source tensors → 1610 brain tensors**, two-way covered (the
delta is exactly the 70 `BasicTransformerBlock`s' three host-side
fusions/splits — asserted in `config::tests::sdxl_manifest_matches_the_checkpoint_count`).

Graph: **2158 steps** at a 32×32 latent.

| gate | result |
|---|---|
| `sdxl_unet_forward_matches_diffusers` | **165 comparisons, 0 failed** (162 device taps + 2 host + `out.sample`), **worst cosine 0.9999999999** (`up1.attn1.proj_out`, max_abs 3.728e-4, rel_l2 1.539e-5) |
| `out.sample` | cosine **1.0000000000**, max_abs **1.705e-5**, rel_l2 **3.258e-6** |
| host conditioning | `time_proj` 1.0000000000 / 3.188e-5, `add_time_proj` 1.0000000000 / 4.911e-5 |
| `conv_in` | 1.0000000000, max_abs 3.576e-7 |
| `mid.attn0` (10 transformer blocks deep) | 1.0000000000, max_abs 2.890e-4 |

Every stage is at cosine 1.0000000000 except twelve in the up path, all at
0.9999999999 — the accumulation of ~60 transformer blocks of fp32 reduction-order
difference, not a mechanism difference (`rel_l2` stays ≤1.54e-5 throughout).

### Scheduler parity

`cargo test -p brain-diffusion`, host math, no device.

| gate | result |
|---|---|
| `discrete_parity::discrete_schedulers_match_diffusers` | **58 checks, 0 failed, worst max_rel 7.510e-6** (`ddim.epsilon.20.traj`) |
| `discrete::tests` (7 property/unit) | 0 failed |

Covered per family × {epsilon, v_prediction} × {4, 20} steps: the discrete
`timesteps` vector (**exact**, 0.000e0 everywhere), the `sigmas` table
including the terminal entry, `scale_model_input` at every step, and the full
`step()` trajectory (every intermediate latent, not just the last).

### VRAM / residency

`native_resolution_fits_one_card` (`#[ignore]`d — it is a residency
measurement, not a parity gate), one P40, fp32, no INT8, no sharding:

* **2 567 463 684 parameters = 10.27 GB fp32**;
* the production graph (taps off, activation pool live) at a **128×128 latent**
  — SDXL's native 1024×1024 — is **2198 dispatches** and runs in
  **4.08 s per forward**;
* so **yes, the full-resolution fp32 UNet forward runs on one P40.** The
  ~14 GB figure in `docs/imaging/plan.md` is UNet + both text encoders; the
  UNet alone is 10.27 GB and the encoders were not resident in this run.

No profile was taken and no speed claim beyond that wall-clock number is made.

## Conventions pinned (each verified against diffusers, not assumed)

1. **The timestep is ADDED, not scale-shifted.** SDXL ships
   `resnet_time_scale_shift: "default"`, i.e.
   `conv1(...) + time_emb_proj(silu(temb))[:, :, None, None]` *before* `norm2`
   — a per-channel broadcast add (`add_chan_bcast`), **not** `film_chan`. The
   crate's own original description said "scale-shift" and was wrong.
2. **`attention_head_dim: [5, 10, 20]` is a head COUNT**, not a head dim
   (`num_attention_heads = num_attention_heads or attention_head_dim`, and SDXL
   ships `num_attention_heads: null`). The real head dim is 64 at every level.
   Reading it literally yields a forward with correct shapes and wrong results.
3. **The added conditioning is `concat([pooled_text, time_ids_sinusoids])`** —
   pooled FIRST, then the six micro-conditioning values in
   `(orig_h, orig_w, crop_top, crop_left, target_h, target_w)` order. Gated by
   comparing the `[6, 256]` sinusoid block *as a slice* of the 2816-wide vector,
   which pins the order rather than just the width.
4. **Two GroupNorm epsilons in one graph**: `1e-5` in the resnets and
   `conv_norm_out` (the config's `norm_eps`), `1e-6` inside every
   `Transformer2DModel` (hardcoded in diffusers' `_init_continuous_input`).
5. **GEGLU is `hidden * gelu_erf(gate)`** with `hidden` the FIRST half of each
   projection ROW. `chunk(2, dim=-1)` interleaves the halves, so the fused
   `[2I, C]` projection is SPLIT at import into two `[I, C]` weights and the
   activation is `gelu_erf` + `mul` — the composition `mul.wgsl`'s own header
   prescribes. `geglu_shift` is a different function (`gelu(h)·(g+1)`).
6. **`Downsample2D` uses `padding=1`** (symmetric), not the VAE's asymmetric
   `F.pad(x,(0,1,0,1))` + `padding=0`. `Builder::conv_down` implements the
   latter and is deliberately not used here.
7. **DPM-Solver++ is variance-PRESERVING**; the Euler family is
   variance-exploding. DPM converts with `x0 = (x - σ_t·ε)/α_t`
   (`α_t = 1/√(σ²+1)`) and its `scale_model_input` is the identity; Euler uses
   `x0 = x - σ·ε` and MUST pre-divide by `√(σ²+1)`.

## Found and fixed, with the numbers that proved it

| defect | how it showed | fix |
|---|---|---|
| Up-block resnet input width inverted (`prev_output_channel` applied to the LAST resnet instead of the first) | import error: `up_blocks.1.resnets.0.norm1.weight shape [1920], expected [1280]`. Invisible at `up_blocks.0`, where the two widths are both 1280 | `config.rs`; pinned by `sdxl_up_resnet_input_widths_match_the_checkpoint`, which asserts all nine checkpoint widths |
| DPM-Solver++ fed the variance-exploding `x0` conversion | `max_rel 1.115e0` vs diffusers (a different trajectory, not rounding); `scale_model_input` alone was `max_rel 0.81` at step 0 | `x0_from_vp` + identity `scale_model_input`; now `max_rel ≤ 6.02e-6` |
| Sequential f32 `cumprod` for `alphas_cumprod` | `max |Δ|/ᾱ = 9.45e-7` vs torch's parallel scan (terminal 0.0046600914 vs 0.004660095) | f64 accumulator, rounded once — **bit-identical to torch on all 1000 entries** when fed torch's own betas |
| `vae::blocks::Builder` uploaded weights with `storage_init` (mapped-at-creation) and never drained wgpu staging | **wgpu Out of Memory** on a P40 with ~20 GB free, uploading 10.27 GB | `Builder::upload`: `storage()` + `write()` + `poll_wait()` per tensor, plus a 1-element readback every ~1 GiB — the `paramstore` / `zimage::BlockWeights::upload` pattern. Bit-identical data; purely how the memory is obtained |

## Found by the adversarial review of this track, and fixed

Everything above reproduced exactly on a re-run (165/0 with `out.sample` at
cosine 1.0000000000 / rel_l2 3.258e-6; 58/0 worst 7.510e-6; 2 567 463 684
params, 2198 dispatches, 4.06 s at a 128×128 latent). These are the defects the
re-run turned up on top of it.

| defect | how it showed | fix |
|---|---|---|
| **`Sigmas::init_noise_sigma` had its two branches INVERTED** — it returned `σ_max` for `Leading` and `√(σ_max²+1)` for `Linspace`/`Trailing`; diffusers is the other way round (`EulerDiscreteScheduler.init_noise_sigma`: *`if timestep_spacing in ["linspace","trailing"]: return sigmas.max()`*). SDXL ships `leading`, so **every** SDXL sampling run would have started from the wrong noise scale | Nothing gated it — the golden never dumped the value. Measured against diffusers on the SDXL 20-step schedule: reference **11.073580741882324**, brain returned **11.028335571289062** (0.41 % low) | `discrete.rs`; the dumper now stores `{fam}.{pred}.{n}.init_noise_sigma` and `discrete_parity` checks it — the suite goes 58 → **66 checks**, the eight new ones at max_rel ≤ 4.502e-7 |
| Parity ladder gated **cosine only** (`≥ 0.9999`). Cosine is scale-invariant, so `got = 1.05·want` scores 1.0 — a dropped `output_scale_factor`, a doubled attention scale or a mis-read GroupNorm gain would all have passed | `rel_l2` was computed and printed at every stage but never asserted | `parity.rs`: `REL_GATE = 1e-3` on `rel_l2` as well. Worst measured stage is 1.539e-5, so ~65× headroom |
| Self-attention slab sized over **all** levels instead of the levels that record a transformer. SDXL's level 0 is `DownBlock2D`/`UpBlock2D` — no attention — but has 16× the tokens of level 1 | 8× over-allocation: at a 128×128 latent the non-cooperative path reserved 2 × 5·(H·W)² = **10.7 GB** of never-bound buffer instead of 1.3 GB | `model.rs` skips levels with no transformer; pinned by `slab_is_sized_over_attention_levels_only` |
| `import::check_shape` compared **element counts, not shapes**. Every square weight in SDXL (`to_q/to_k/to_v/to_out`, `proj_in`, `proj_out`, `time_embedding.linear_2`) would pass transposed, and SD 1.5's `[C,C,1,1]` conv `proj_in` would pass as SDXL's `[C,C]` linear — the exact variant-checkpoint confusion the two-way coverage exists to reject | Not observed; the shapes happen to be right | `import.rs`: exact shape equality, plus per-piece checks on the three `attn1` and two `attn2` sources and on the GEGLU `ff.net.0.proj` before it is halved |
| `model.rs` hardcoded `scores: 8, softmax: 9, apply: 10` into `BidirIds` — private slot numbers of `vae::blocks::KERNELS`. Adding one entry mid-list would silently dispatch the wrong pipeline, and only on non-cooperative devices | Not observed | `vae::blocks::ATTN_BIDIR_SLOTS` exported and used; `the_bidir_attention_trio_is_at_the_exported_slots` and `appended_slots_hold_their_named_kernels` pin the whole set |
| `manifest.json` recorded **only** `unet/stages.safetensors`. The dumper rebuilt the manifest from scratch on every run, so the two-run workflow (`--skip-unet`, then `--skip-schedulers`) threw away the first run's `files` entry, its sha256 and the `scheduler_config` | `files` had one key; `params` had no `scheduler_config`/`sched_steps` | The dumper now MERGES. Re-running `--skip-unet` reproduced `steps.safetensors` **byte-identically** (sha256 `d45ff4fb…`) and restored both entries |
| `tools/goldens/sdxl_dump_reference.py` defined `_fit` and `_CLASSES` twice (identical copy-paste) | F811 redefinition; harmless but the first copy is dead | Removed |
| `crates/unet/Cargo.toml` listed four dependencies it never used (`brain-vision`, `brain-imaging`, `brain-clip`, `brain-diffusion` — the last two appear only inside doc-comment code spans), directly contradicting its own comment about cargo linking every entry into each test binary | grep for each lib name in `src/` + `tests/` | Removed, with a per-dependency justification comment |

### Corrected claim

The track report said the shared-`vae::blocks` regression was run on "**real
weights, not skipped**" for `brain-vqgan` (15 + 9) and `brain-restore` (17 + 4).
Re-run without the weight env vars, `brain-vqgan`'s 9 parity tests pass in
**0.69 s with 7 of the 9 printing `SKIP: set BRAIN_VQGAN_WEIGHTS …`** — the
green count was mostly skips. The regression was re-run properly with
`BRAIN_VQGAN_WEIGHTS` / `BRAIN_RESTORE_WEIGHTS` / `BRAIN_FLUX2_VAE` pointed at
`/data/workspace/resources`, and it is genuinely clean: vqgan 15 + **9 in
10.63 s**, restore 17 + **4 in 20.25 s** (worst cosine 1.000000000, worst
rel_l2 1.207e-5 at `fuse.256.scale`), `flux2_vae_parity` 1 in 3.00 s. The two
`brain-vae` Z-Image tests still skip (`BRAIN_ZIMAGE_VAE` unset) and were never
claimed.

### Open, not fixed

`docs/imaging/plan.md` §2 asserts "brain today has **no UNet diffusion model at
all** (only DiT: `dit`, `zimage`, `flux2`)". That is false: `crates/wm-diamond`
is an EDM diffusion world model built on a `UNet2DModel`-shaped graph
(`DiamondUNet`, down/mid/up + skips + resnets + self-attention, 663 lines in
`model.rs`) — and it records that graph by hand rather than through
`vae::blocks::Builder`, so brain now has **two** independent UNet graph
recorders. `crates/unet` is the better citizen of the two (it reuses the shared
builder and adds no kernel), so the follow-up is to migrate `wm-diamond` onto
`vae::blocks`, not to undo anything here. Filed rather than done: it is a
refactor of another model's crate.

## Not measured by this gate, and therefore not claimed

* **The 128×128 latent is not parity-gated** — the golden is dumped at 32×32
  (a 256×256 image). The graph is resolution-independent, so what 32×32 gates
  is the composition; the native size is only shown to *run* and to fit.
* **Batch > 1** is not implemented (`Unet::new` records a batch-1 graph) and so
  neither is classifier-free guidance as a single batched forward.
* **INT8** is untouched. It is not needed for residency here (10.27 GB fits),
  but it is the obvious lever for keeping the UNet *and* both encoders *and* a
  ControlNet resident together.
* **The VAE and the text encoders are not wired.** `crates/vae` is
  config-driven and SDXL's is a 4-latent-channel `AutoencoderKL`, and
  `crates/clip` is already parity-gated for CLIP-L + bigG — but no code here
  composes them. *(The CLIP BPE tokenizer, listed here as missing while this
  track ran, landed in parallel as `data::clip_bpe::ClipBpe`; it is still not
  wired to `crates/clip` or to anything in this crate.)* So "SDXL works" is
  **not** a claim this ledger supports; what it supports is that one UNet
  evaluation and four schedulers reproduce diffusers.
* **A run-to-run determinism finding on the CPU-JIT leg is contested** — see
  below; it did not reproduce on the re-run. No `BRAIN_DEVICE=cpu` number is
  quoted anywhere either way.
* Anything about **speed** beyond the single 4.08 s wall-clock above. No
  profile was taken.

## Contested finding: `backend-cpu` run-to-run determinism on this graph

**Two runs disagree and both are recorded, because neither can be dismissed.**
The port reported the divergence below. The adversarial re-run of this track
probed the *exact* configuration it named (`[1,2]`, `layers=1`, 377 steps),
plus `layers=2` (555 steps), both `taps` modes, 8 submits each, on the same
48-thread `BRAIN_DEVICE=cpu`, and got **0 of 1024 outputs differing in all five
configurations**. The integration pass re-ran `crates/unet`'s three smoke tests
on `BRAIN_DEVICE=cpu` and they pass with **no skip**.

So the honest statement is: *this box does not currently reproduce it, and the
original observation is not explained.* A race whose visibility depends on
allocation layout (which is what the evidence below points at) is exactly the
kind of defect that goes quiet without being fixed, so nothing here is closed.
`crates/unet/tests/smoke.rs` now **measures** determinism (submit the same
recorded graph twice, compare bits) instead of gating on a
`caps().workgroup_reductions` proxy — the proxy also skipped `cpu0`, which the
original report called deterministic, and would have kept skipping forever
after a fix.

The original evidence, verbatim, and **not** in `crates/unet`'s scope to fix:

* the SAME recorded graph, submitted twice with the same inputs on
  `BRAIN_DEVICE=cpu` (48 threads), produces different outputs — 1024 of 1024
  values differ;
* it is **deterministic on `BRAIN_DEVICE=cpu0`** (one core, so process affinity
  pinned) and on both P40s (`0 of 1024` differ across a 5-config sweep);
* the first divergent tap is a conv whose own input tap is bit-identical, so
  the inputs are not the cause;
* isolating that conv (and a `concat2` + conv pair) at the same dims and
  dispatch geometry over 8 iterations reproduces **nothing**;
* whether it fires depends on the graph shape, not the step count: at
  `transformer_layers_per_block = [1,2], layers_per_block = 2` (555 steps) it is
  deterministic, at `[1,2], layers=1` (377 steps) it is not, and merely changing
  which buffers are allocated (per-block cross-attention slabs instead of one
  shared pair) moved which configuration fails — the signature of a race whose
  visibility depends on allocation layout;
* `BRAIN_NO_FASTCONV=1` does not fix it.

`crates/unet/tests/smoke.rs`'s two comparison-based tests skip themselves —
with a printed note — only on a backend that **fails the direct measurement**,
rather than asserting something a genuinely racy one cannot provide. On this
box that measurement passes, so all three run.

## What the follow-up needs

1. **`check_unet`** — the backward. The forward is SSA (every stage writes a
   fresh buffer), and `vae::blocks::grad` already differentiates the
   conv/GN/SiLU/add half, so the new adjoints are the transformer half plus
   `add_chan_bcast` and `concat2` (whose backward is the existing
   `concat_split`).
2. **The serving contract** (`docs/serving-contract.md`): `Provider`,
   `resident_unet.rs`, a real batched `run_batch`, D-Bus, `examples/`.
3. **Batch > 1**, which CFG wants: every kernel already takes a `bsz`/`N`
   param; the graph records `N = 1` and the skip stack would need to carry it.
4. **Phase 4c ControlNet** plugs in at named injection points: the down-block
   residual pushes (`skips`) and the mid-block output are exactly those points,
   and `model.rs` already materialises both as a list.
5. **The pipeline**: `crates/vae` with a 4-channel `VaeConfig`, `crates/clip`'s
   penultimate + pooled pair (`ClipTextConfig::penultimate_layer`),
   `data::clip_bpe::ClipBpe` (which now exists and is id-exact vs HF on both
   SDXL tokenizers, but has no `encode_prompt`-shaped caller yet), and a sampler
   loop over `diffusion::discrete::EulerScheduler` (SDXL's shipped default).

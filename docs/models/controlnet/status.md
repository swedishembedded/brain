# ControlNet — status ledger

`crates/controlnet`. Phase 4c of `docs/imaging/plan.md`.

**What is claimed:** the backbone-agnostic `ControlAdapter` seam, and an SDXL
`ControlNetModel` whose **per-injection-point residuals reproduce diffusers**.

**What is not claimed, and is not measured anywhere below:** `check_controlnet`
and any backward at all, LoRA/finetuning, INT8, batch > 1, a sampler loop, a
CLI, and the entire serving contract (no `capability::Provider`, no
`resident_controlnet.rs`, no `run_batch`, no D-Bus, no example). "InstantID
works" is *not* supported by anything here — what is supported is that one
ControlNet evaluation reproduces diffusers and that a brain UNet consumes its
residuals.

---

## 1. The design, and why it is the deliverable

`docs/imaging/plan.md` §2 asks for a **seam**, not an SDXL crate. A ControlNet
is a trainable copy of a backbone's early blocks whose zero-conv outputs are
added as residuals at *named injection points*; FLUX ControlNets inject into
double-stream blocks the same way. So the crate exports the contract first:

| piece | what it is |
|---|---|
| `adapter::InjectionPoint` | a `name` + a `Layout` — `Spatial{c,h,w}` (UNet) or `Tokens{t,d}` (a DiT stream) |
| `adapter::ControlAdapter` | a **backbone**: declares its points in its own control-input order, and whether the recorded graph reads them |
| `adapter::ControlSource` | a **control model**: declares the points it produces |
| `adapter::Residuals` | the bundle, keyed by name, with `scaled()` |
| `check_compatible` / `order_for` | the only places the two meet — **by name and element count** |

`unet::Unet` is the first `ControlAdapter` (the impl lives in
`crates/controlnet` because the trait does; `unet` must not depend on this crate
since this crate composes `unet::model::Rec`). The FLUX DiTs are the intended
second: nothing in `adapter.rs` mentions convolutions, resolutions or SDXL.

**Why by-name matching earns its keep.** diffusers passes
`down_block_additional_residuals` as a bare tuple zipped against the UNet's own
`down_block_res_samples`. SDXL has **four** 320-channel injection points and
**three** 640-channel ones, so a permutation among them type-checks, runs, and
produces a plausible image. `crates/controlnet/tests/smoke.rs::
order_for_reorders_equal_sized_points` is that case; `a_mismatched_backbone_is_
rejected_by_name` is the sibling where every *name* matches and only the element
counts differ (an 8×8 ControlNet against a 16×16 UNet).

## 2. One implementation — what was reused, and what was hoisted

The ControlNet's down and mid blocks **are** the UNet's, so they are recorded by
the UNet's own recorder. Three shared changes were made in `crates/unet` rather
than copying ~250 lines of model math (flagged as shared changes; see §6):

* `unet::model::Rec` is now **public**, with `Rec::new`, `Rec::conditioning`,
  `Rec::down_path`, `Rec::mid_block` and public block methods. `Unet::new` was
  migrated onto them in the same change, so there is one down path, not two.
* `unet::model::attn_slab_words(cfg, h, w, with_up)` — the non-cooperative
  attention-slab sizing, previously inline in `Unet::new`. `with_up = false` is
  the ControlNet shape and is why it is a parameter: a ControlNet has no up path
  and `cfg.up_block_types` must not even be indexed.
* `unet::import::remap_manifest(who, raw, manifest)` — the manifest-driven remap.
  A ControlNet checkpoint carries the **same three fused leaves** (`attn1`'s
  q/k/v, `attn2`'s k/v, the split GEGLU `ff.net.0.proj`) under the same module
  names; a second copy would be a second place those can be got wrong with
  nothing comparing the two.

`ControlNetConfig` **holds** a `UNetConfig` rather than restating it, and its
tensor manifest is literally a *filter* of the backbone's manifest plus the
conditioning embedder and the zero-convs. A ControlNet whose
`transformer_layers_per_block` disagreed with its UNet's would otherwise import
cleanly and be wrong below level 1.

**Adds no block, and one kernel *slot* rather than one kernel.**
`controlnet::model::KERNELS` is `unet::model::KERNELS` **verbatim** plus
`scale_chan`. Being a strict prefix-extension is what lets ONE device drive both
models (exactly as `unet::model::KERNELS` extends `vae::blocks::KERNELS`), and
`tests/smoke.rs` builds a UNet and a ControlNet on one `Gpu` to prove it.
`conditioning_scale` is `scale_chan` with `c = 1, inner = 1`, i.e.
`out[i] = x[i] · scale[0]` from a **one-element device buffer** — the
`restore` fidelity-dial pattern, so changing the scale is a write, not a graph
rebuild, and it is out-of-place so the pre-scale zero-conv output stays a tap.

## 3. The four conventions this graph pins

Each was read off `diffusers/models/controlnets/controlnet.py` and each is
gated by a golden.

1. **The conditioning embedder's SiLU is after every conv EXCEPT `conv_out`.**
   `silu(conv_in(x))`, then `silu(block(·))` ×6, then a bare `conv_out`.
2. **The conditioning embedding is added to `conv_in`'s output** — in latent
   space, after the input convolution. Not concatenated, and not added to the
   latent before `conv_in`. Golden tap `sample_cond`.
3. **`conditioning_scale` multiplies the zero-conv OUTPUT and nothing else.**
   The dumper asserts this inside itself (a second diffusers forward at 0.75 is
   checked to be exactly 0.75× the first) and the Rust test compares against
   that second forward, so it gates the *place* the scale is applied, not just
   self-consistency.
4. **The residual list is the backbone's skip stack, in the same order**:
   `controlnet_down_blocks.k` is fed `skip_stack()[k]`. The golden therefore
   dumps **both halves** of every injection point — `zero{k}.in` (the trainable
   copy's output) and `zero{k}` (after the zero-conv) — and the graph taps both.
   A tap on the conv output alone cannot see a permuted feed.

Also pinned: the conditioning image is **`[0, 1]` CHW at pixel resolution**, not
the `[-1, 1]` the VAE's image input uses — diffusers builds it with
`VaeImageProcessor(do_normalize=False)`. `crates/controlnet/src/cond.rs` is that
conversion, with the asymmetry stated in the module header; the cast and the
HWC→CHW permutation themselves are `imaging::pixels::{u8_to_unit, hwc_to_chw}`,
not a local loop — only the `swap_rb` channel order is this crate's.

## 4. Measured — phase 4c gate

Reference: `/…/instantid/ControlNetModel` (a real, **trained** SDXL ControlNet —
its zero-convs are not zero, so the residuals are a non-trivial target).
Goldens: `tools/goldens/controlnet_dump_reference.py`, diffusers 0.39.0 / torch 2.13.0,
CPU fp32, seed 20260804.

Import: **844 source tensors → 810 brain tensors**, two-way covered (the delta
is the 34 `BasicTransformerBlock`s' three host-side fusions/splits).
**1 251 014 160 parameters = 5.00 GB fp32.**

One Tesla P40 (`BRAIN_DEVICE=gpu1`, wgpu/Vulkan), `--release`, targeted
`cargo test --release -p brain-controlnet` (never a workspace
`--tests --examples` build).

| suite | tests | measured |
|---|---|---|
| `brain-controlnet` lib | 19, 0 failed | seam unit tests (ordering, by-name rejection), manifest counts, `cond` conversions, kernel-set prefix check |
| `tests/smoke.rs` | 6, 0 failed (1.6 s) | synthetic import round-trip + two-way rejection; tiny forward finite and **non-zero at every point**; tapped graph **bit-identical** to the pooled one; the seam against a real tiny UNet; **residual placement** (below) |
| `tests/parity.rs` | 2, 0 failed (42.7 s both legs) | see below |

**The consumer-side gate**
(`smoke.rs::a_down_residual_reaches_the_output_only_through_the_up_path`).
`scale = 0 is a no-op` and `scale = 1 moves the output` are jointly blind to
*where* a residual lands: diffusers adds a `down.k` residual to
`down_block_res_samples`, which only the up path reads, and at the **last** down
point adding it to the running hidden state as well is shape-legal — `down.{n-1}`
is exactly the tensor entering `mid_block`. That mistake type-checks, still
vanishes at `scale = 0`, still moves the output at `scale = 1`, and leaves
`tests/parity.rs` at 1e-11 because the bug is in the **consumer**, not in this
crate's residuals. The gate injects at one point at a time and requires the
`mid.resnet1` tap to be **bit-identical** (max |Δ| exactly 0.0) while the output
moves. Verified to fail — naming `down.3` — against a deliberately mutated
`Unet::new_controlled` that double-counts; reverted after.

This is the *only* thing gating the residual placement inside the UNet, and it
does it at toy dims on synthetic weights. A full-size diffusers
`StableDiffusionXLControlNetPipeline` UNet forward with these residuals supplied
is still **not** gated, and "SDXL + ControlNet reproduces diffusers end to end"
is not claimed.

**Parity, 32×32 latent (256×256 conditioning image), 1082 steps:**
**140 comparisons, 0 failed.** Every cosine 1.0000000000; worst
**1−cos 1.914e-11** (`time_proj`, host math), worst **rel_l2 6.187e-6**
(same stage). The ten residuals: `down.0` rel_l2 6.652e-7 → `down.8` 4.273e-6,
`mid` **4.924e-6** (max_abs 2.213e-4). The ten at `conditioning_scale = 0.75`,
compared against a *separate diffusers forward* at 0.75: worst rel_l2
**4.925e-6**.

**Parity, 24×16 latent (192×128 conditioning image), 1064 steps:**
**140 comparisons, 0 failed**, worst **1−cos 2.234e-11** and worst
**rel_l2 6.683e-6** (`down2.attn1.proj_out`); `mid` residual rel_l2 6.166e-6.

*Why a second, non-square leg exists:* at a square latent an H/W transposition
is invisible **everywhere**, and this crate's one genuinely new stage — the
conditioning embedder, which halves H and W three times on its own bookkeeping —
is exactly where such a bug would live. Every other gate in the imaging
workstream (`unet`, `vae`, `vqgan`, `restore`) is square, so this is the only
place in the UNet family where that class can fail. It cost one extra dumper run
and 17 s of test time.

Both a cosine gate (≥0.9999) **and** a `rel_l2` gate (1e-3) are asserted. Cosine
alone is scale-invariant, and a ControlNet has a whole family of whole-tensor
scale mistakes available to it (`conditioning_scale` applied twice, applied to
the conditioning embedding instead of the residual, a dropped zero-conv bias).

**Second backend — the full parity ladder was run there too.**
`BRAIN_DEVICE=cpu` (Cranelift JIT, 48 threads): lib 19 + smoke 5 + **both**
parity legs, **0 failed** (parity 40.2 s for the pair). This is the leg that
exercises the *materialised* `attn_*_bidir` self-attention path and
`attn_slab_words(.., with_up = false)`, since `backend-cpu` reports
`workgroup_reductions = false` and takes neither of the GPU's cooperative
routes — `flash_attn_bidir`, `flash_attn_bidir_split`, `gn_stats_wg`,
`matmul_reg2` and `matmul_reg3` all decline to JIT and are replaced.

| leg | steps (CPU / GPU) | comparisons | worst 1−cos | worst rel_l2 | `mid` residual rel_l2 |
|---|---|---|---|---|---|
| 32×32 | 1112 / 1082 | 140, 0 failed | 1.914e-11 | 6.187e-6 | 4.595e-6 |
| 24×16 | 1112 / 1064 | 140, 0 failed | 1.914e-11 | 6.187e-6 | 4.813e-6 |

The CPU step count is shape-independent while the GPU's is not: the lowered
`im2col_at + matmul` conv path chunks over spatial positions, and the CPU
backend does not take it. The worst stage on both CPU legs is `time_proj` —
**host** math, identical in both legs by construction — so the CPU low digits
are not a fingerprint of the device path; `backend-cpu`'s reduction order
depends on rayon splitting and must not be quoted as one.

**The shared `crates/unet` change did not regress its owner.**
`brain-unet` re-run after the hoist on the same P40:
`sdxl_unet_forward_matches_diffusers` **165 comparisons, 0 failed**, worst
`up1.attn1.proj_out` at cosine 0.9999999999, `out.sample` max_abs **1.705e-5**
/ rel_l2 **3.258e-6** — digit-for-digit the numbers in
`docs/models/unet/status.md`. Plus `brain-unet` 9 lib + 3 smoke on **both** the
P40 and `BRAIN_DEVICE=cpu`, 0 failed.

Workspace `cargo build` (lib+bin): **zero rustc warnings**
(`cargo build --message-format=short 2>&1 | grep -E '^[^ ]+\.rs:[0-9]+:[0-9]+: warning:'`
returns nothing). `cargo clippy --workspace --all-targets` **exits 0**;
**zero** warnings originate in `crates/controlnet` or `crates/unet`.
Absolute-path gate `grep -rnE '"/(data|home|tmp|opt|mnt|root)/' crates` empty.

## 5. The ZipDepth synergy — what it costs, and what was done

brain's own `crates/depth` produces depth maps, so a depth-conditioned
ControlNet needs no external preprocessor. `controlnet::cond::from_depth` is the
whole adapter (normalise to `[0, 1]`, replicate to three channels — what the
diffusers depth examples do to a DPT output before `prepare_image`), and it is
unit-tested including the flat-map case that would otherwise divide by zero.

It is deliberately **a function over an already-computed map, not a dependency
on `crates/depth`**. The cost is not the code; it is that **there is no
depth-conditioned SDXL ControlNet checkpoint on this machine**, so a
`depth::Predictor` → ControlNet path could not be parity-gated and would ship as
untested plumbing behind a `depth → vision → model` dependency edge. Wiring it
is a few lines *once a depth ControlNet checkpoint is fetched*; that fetch is
the actual prerequisite.

## 6. Shared files touched (flagged)

| file | change | risk |
|---|---|---|
| `crates/unet/src/model.rs` | `Rec` made public with `new`/`conditioning`/`down_path`/`mid_block` + public block methods; `attn_slab_words` extracted; `Unet::new_controlled` / `control_shapes` / `accepts_control` / `run_with_control` added (`Unet::new` delegates, signature unchanged) | the largest edit; parity re-run and identical |
| `crates/unet/src/import.rs` | `remap_manifest(who, raw, manifest)` extracted; `remap(raw, cfg)` delegates | small, additive |

No other crate, no root `Cargo.toml`, no `docs/imaging/plan.md`, no `AGENTS.md`.

## 7. Owed next

1. **`check_controlnet`.** No backward exists. The zero-convs and the
   conditioning embedder are the only trainable parts in the usual recipe
   (`Role::Frozen` on the copied blocks would mirror `check_sam2`), and the
   copied blocks' backward is the same one `check_unet` still owes — so the two
   should land together, not separately.
2. **The serving contract**, entirely: `controlnet::caps`,
   `resident_controlnet.rs`, `run_batch`, D-Bus, an example. Note the natural
   unit is *UNet + ControlNet together*, since the residuals are consumed one
   denoising step at a time.
3. **A fused device path.** Today `ControlNet::run` reads the residuals back to
   the host and `Unet::run_with_control` writes them again. Both graphs already
   live on one device with one kernel set, so the round trip is removable:
   `Unet::control_inputs` are device buffers and the ControlNet's scaled outputs
   are device buffers of the same length. Not done because nothing yet composes
   the two in a loop, and doing it before there is a sampler would be an
   optimization with no gate.
4. **Resolution.** Parity is at 32×32 and 24×16. SDXL's native 128×128 latent
   needs a 1024×1024 conditioning image through the embedder (16 ch at 1024²) —
   never run, no number claimed.
5. **`BlockKind`/`LevelKind`.** `config.rs` re-exports `unet::config::BlockKind`
   as `LevelKind` so a caller need not depend on `crates/unet`; nothing in the
   tree uses that alias yet.
6. `guess_mode` (the `torch.logspace(-1, 0, …)` per-point scale ramp) and
   `global_pool_conditions` are **not implemented**. Both are one-line additions
   over `Residuals`, and neither has a golden.

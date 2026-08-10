# PuLID-FLUX identity conditioning (`crates/pulid`) — status ledger

Chronological, measured-only. Numbers below were produced by the commands
quoted; nothing is estimated.

PuLID turns "keep this person's face" into a model operation: a face photo
becomes 32 ID tokens, and those tokens are cross-attended into the FLUX.1 image
stream at 20 points along the backbone. Weights: `pulid_flux_v0.9.1.safetensors`
(312 tensors, BF16, 1.14 GB → 562 M params → 2.25 GB fp32).

## What this port is, and what it deliberately is not

It is mostly **wiring three already-gated components together**:

| piece | where it lives | prior gate |
|---|---|---|
| ArcFace 512-d embedding | `crates/facenet` (insightface *antelopev2*, which is literally what `PuLIDPipeline.get_id_embedding` calls) | cosine 1.0000000 |
| EVA-CLIP-L/336 image tower + the 5 tapped hidden states | `crates/clip` (`EvaVision`, `EvaVisionConfig::PULID_TAPS = [3,7,11,15,19]`) | 0.99999999 |
| the FLUX.1 12 B MMDiT backbone | `crates/flux1` | reduced-depth fp32, worst 1−cos 1.5e-11 |

Genuinely new here: the `IDFormer` Perceiver resampler, the injected
`PerceiverAttentionCA`, and the injection schedule. **No kernel and no shared
block was added.**

## The three risky facts, verified against the reference (not assumed)

Sources: `PuLID/pulid/encoders_transformer.py`, `PuLID/pulid/pipeline_flux.py`,
`PuLID/flux/model.py`, `PuLID/flux/sampling.py` at v0.9.1.

1. **Where the ID attention is inserted, and at what interval.** After
   double-stream block `i` when `i % 2 == 0`, after single-stream block `i` when
   `i % 4 == 0`. At FLUX.1's 19 + 38 that is 10 + 10 = **20 sites**, matching the
   20 `PerceiverAttentionCA` modules in the checkpoint and
   `pipeline_flux.py`'s `num_ca` formula. `ca_idx` is **one sequential counter
   shared by both loops** (doubles take 0..10, singles 10..20) — so at a reduced
   depth the mapping is *not* "singles start at 10", and the port reproduces the
   counter rather than the full-depth numbers. `PulidConfig::schedule` + its unit
   tests.
2. **Is the ID embedding projected/normalised before injection?** *Projected,
   yes*: `IDFormer` ends with `latents[:, :32] @ proj_out` (1024 → 2048).
   *Normalised, no*: there is no norm on the encoder output. Each cross-attention
   module applies **its own** `norm1` to the ID tokens and `norm2` to the image
   tokens, so normalisation is per-site and part of the attention. (Two
   normalisations do happen upstream, inside the ID *condition*: the EVA-CLIP cls
   embedding is L2-normalised before the concat, the ArcFace embedding is not.)
3. **How is the ID contribution scaled and combined?**
   `img = img + id_weight * ca(id, img)` — **added to the image residual
   stream, never concatenated as tokens**, and the *image* rows are the attention
   QUERIES with the ID tokens as KEYS/VALUES (`pulid_ca[k](x=id, latents=img)`,
   whose `forward(self, x, latents)` puts `to_q` on `latents`). The **start-step**
   schedule is a *sampler* property, not a model one: `flux/sampling.py` passes
   `id=None` for steps below `start_step`, which in brain is the plain
   `Flux1Model::forward`. `crates/flux1` has no sampler loop, so nothing here
   could schedule against one.

## Import — two-way coverage

`pulid::import`: **312 source tensors → 172 encoder + 140 cross-attention**
parameters, validated in both directions (missing tensor → error by name, unused
source tensor → error, duplicate mapping → error, shape and length both checked;
no zero-fill path). Unit-tested against a synthetic checkpoint built by
inverting the remap, so a typo on either side fails in 40 ms rather than after a
1.1 GB load.

Only two tensors need surgery:

* `latents` `[1,32,1024]` → `[32,1024]` (drop the batch axis);
* `proj_out` `[1024,2048]` → **transposed** to `[2048,1024]`, because the
  reference applies it as a bare `latents @ proj_out` while every brain matmul
  is `x @ Wᵀ`.

`to_kv` stays **fused** `[2·inner, k]` on purpose: the reference's
`chunk(2, -1)` puts k at column 0 and v at column `inner`, which is byte-for-byte
the fused-KV layout `attn_scores_cross` / `attn_apply_cross` already bind
(`kv_stride = 2·inner`, `k_off = 0`, `v_off = inner`). And the reference scales q
and k by `dim_head**-0.25` each, so the product is the `1/sqrt(head_dim)` those
kernels apply themselves — no repacking and no scale fix-up anywhere.

## Kernel reuse (nothing added)

| reference op | brain |
|---|---|
| `nn.LayerNorm` | `layernorm` / `layernorm_rows` via **`model::block::ln_variant`**, keyed on the queried `DeviceCaps` |
| `nn.Linear` | `matmul` / `matmul_gemv` / `matmul_reg3` via **`model::block::gemm_variant`** — the same three-kernel fp32 tier and the same selection rule `crates/flux1` and `crates/flux2` use — then `bias_add` |
| `nn.LeakyReLU()` | `leaky_relu`, slope 0.01 |
| `nn.GELU()` | **`gelu_erf`**, not `gelu` — the reference constructs `nn.GELU()`, whose default is `approximate='none'`. `crates/flux1` next door uses the tanh form; picking the wrong one here is a silent ~1e-3 error, not a crash |
| Perceiver attention | `attn_{scores,softmax,apply}_cross` |
| residual add | `add2` |
| `img + id_weight · ca(...)` | `axpy` (`out += s·in`) |
| `cat(learned latents, id tokens)` | `region_copy` |

Concatenations are **row ranges of one buffer**, not copies: `cat(id_tokens,
mapping_i(vit))` is written by the two producing linears at row offsets 0 and 5,
and `cat(norm1(ctx), norm2(latents))` by the two LayerNorms at row offsets 0 and
582. Every offset is a whole number of 1024- or 3072-float rows, clearing the
64-float (256-byte) storage-binding alignment by construction.

## Goldens

`tools/goldens/pulid_dump_reference.py`, three files under `testdata/pulid/`:

* `idformer.safetensors` — the ID pipeline with **inputs taken from the fixtures
  brain already gates**: the ArcFace embedding from
  `face/antelopev2/e2e.safetensors:photo0_embedding` and the EVA-CLIP
  `cls_embed_l2norm` + `pulid_hidden{0..4}` from
  `clip/eva02_l336/image.safetensors`. So reproducing this golden means brain
  composed its own parity-gated towers into the ID embedding, not that it
  replayed an opaque blob.
* `ca.safetensors` — `PerceiverAttentionCA` modules 0 and 19 on a seeded image
  slab, with module 0's internals tapped.
* `flux_cond.safetensors` — one conditioned transformer evaluation.

Two self-validations run **inside** the dumper and are asserted, per the
playbook:

* the staged replay of `IDFormer` / `PerceiverAttentionCA` reproduces the
  module's own `forward` at **max abs 0.000e0** (so the tapped intermediates are
  the real ones);
* the reduced-depth FLUX.1 harness reproduces `crates/flux1`'s own
  `dit_small.safetensors:out` at **max abs 0.000e0** *before* anything is
  injected — i.e. the conditioned golden differs from the already-gated one by
  the injection and nothing else.

## Forward parity — measured

Tesla P40, native Vulkan backend, `--release`,
`BRAIN_PULID=pulid_flux_v0.9.1.safetensors`. Gate: cosine ≥ 0.999999 **and**
rel_l2 ≤ 1e-4 per stage (cosine alone is scale-invariant).

### 1. ID embedding pipeline — 29 comparisons, 0 failed

Worst cosine **1.0000000000** (worst `1−cos` = 8.60e-13, at `map3_out`).

| stage | cosine | 1−cos | max_abs | rel_l2 |
|---|---|---|---|---|
| `id_tokens` | 1.0000000000 | 7.95e-14 | 5.960e-7 | 3.807e-7 |
| `latents_in` | 1.0000000000 | 7.34e-14 | 5.960e-7 | 3.703e-7 |
| `map0_out` … `map4_out` | 1.0000000000 | 3.51e-13 … 8.60e-13 | ≤3.910e-5 | ≤1.196e-6 |
| `ctx0` | 1.0000000000 | 3.51e-13 | 3.815e-5 | 6.382e-7 |
| `layer0_attn` … `layer9_ff` (20) | 1.0000000000 | 1.46e-13 … 4.10e-13 | ≤3.357e-4 | ≤8.745e-7 |
| **`id_embedding`** | **1.0000000000** | 3.00e-15 | 1.465e-3 | 1.764e-7 |

`id_embedding`'s max_abs of 1.5e-3 is scale, not drift: rel_l2 is 1.8e-7.

### 2. Injected cross-attention — 8 comparisons, 0 failed

Worst cosine **1.0000000000** (worst `1−cos` = 8.35e-13).

| stage | cosine | 1−cos | max_abs | rel_l2 |
|---|---|---|---|---|
| `ca0.norm1_id` | 1.0000000000 | 1.67e-14 | 3.815e-6 | 1.163e-7 |
| `ca0.norm2_img` | 1.0000000000 | 1.69e-13 | 4.768e-6 | 2.787e-7 |
| `ca0.q` | 1.0000000000 | 7.47e-13 | 1.526e-5 | 1.085e-6 |
| `ca0.kv` | 1.0000000000 | 7.92e-14 | 2.861e-6 | 2.934e-7 |
| `ca0.ctx` | 1.0000000000 | 5.21e-13 | 4.947e-6 | 8.931e-7 |
| `ca0.out` | 1.0000000000 | 8.35e-13 | 3.576e-6 | 1.016e-6 |
| `ca19.out` | 1.0000000000 | 6.80e-13 | 1.335e-4 | 1.016e-6 |
| `ca0.out@id_weight=0.5` | 1.0000000000 | 5.52e-14 | 2.384e-7 | 3.927e-8 |

The last row is the `id_weight` dial: the contribution scales linearly and
nothing else moves.

### 3. One conditioned transformer evaluation — 10 comparisons, 0 failed

`BRAIN_FLUX1_TRANSFORMER=FLUX.1-Kontext-dev/transformer`, reduced depth
**2 double + 2 single** (the truncation `crates/flux1`'s own fp32 gate uses),
256 text + 256 image tokens, `id_weight = 1.0`, **2 injection sites**.

Worst cosine **1.0000000000** (worst `1−cos` = 1.44e-11, at `cond.sg1_txt`).

| stage | cosine | 1−cos | max_abs | rel_l2 |
|---|---|---|---|---|
| `uncond.out` (adapter absent) | 1.0000000000 | 4.19e-12 | 3.147e-5 | 3.004e-6 |
| `cond.db0_img` | 1.0000000000 | 1.07e-11 | 1.945e-4 | 4.651e-6 |
| `cond.db1_img` | 1.0000000000 | 2.61e-12 | 7.172e-4 | 2.350e-6 |
| `cond.sg0_img` | 1.0000000000 | 1.41e-11 | 1.556e-3 | 5.303e-6 |
| `cond.sg1_img` | 1.0000000000 | 6.51e-12 | 2.686e-2 | 3.671e-6 |
| `cond.db{0,1}_txt`, `cond.sg{0,1}_txt` | 1.0000000000 | 4.74e-12 … 1.44e-11 | ≤4.639e-3 | ≤5.467e-6 |
| **`cond.out`** | **1.0000000000** | 4.09e-12 | 2.670e-5 | 2.990e-6 |

**Non-vacuity, asserted in the test:** conditioned vs unconditioned prediction
differs by max_abs **0.6126** — the same number the Python dumper reports, from
a completely separate computation. A no-op adapter would fail this assertion, and
so would a schedule that fired zero sites (also asserted).

### Cross-backend — `BRAIN_DEVICE=cpu`, measured

`brain-wgsl-cpu` (Cranelift JIT, 48 threads), same fixtures, same gate:
**all three gates pass, 47 comparisons / 0 failed** — IDFormer 29, the injected
cross-attention 8, and `flux1_conditioned_parity` 10, every stage at cosine
**1.0000000000**, worst `1−cos` 1.45e-11 (`cond.sg1_txt`). The conditioned gate
reports the **identical** non-vacuity number as on the GPU, max_abs **0.6126**.
(The review pass measured the conditioned gate here; the first run of this
ledger had it SKIPPING because `BRAIN_FLUX1_TRANSFORMER` was unset.)

This run is not a formality: `DeviceCaps::workgroup_reductions` is **false** on
the CPU JIT, so `Emit::ln` takes the `layernorm` arm there and the
`layernorm_rows` arm on the P40 — the two backends execute different kernels for
every one of the 40+ LayerNorms in this port, and agreeing to 1e-13 is what
proves the *selection site*, not just one kernel. The same capability puts the
GEMM tier on its reference arm there (`gemm_variant`'s `Reference(matmul)`,
matching `flux1::Flux1Model::gemm_tier`), which `backend-cpu` routes to its AVX2
GEMM, so the two backends do not share a matmul implementation either.

## NOT gated, NOT claimed

* **End-to-end generation.** `crates/flux1` has no sampler loop and no VAE glue,
  so "generate a face" cannot be run at all, let alone gated. There is no image
  in this ledger and there is no `id_weight`/`start_step` sweep, because there is
  nothing to sweep.
* **Full-depth conditioning.** fp32 FLUX.1 is 47.6 GiB and does not fit one
  24 GiB card, which is why flux1's own fp32 gate is reduced depth too. The
  20-site full-depth schedule is gated as a *schedule* (`PulidConfig::schedule`),
  not as a forward. An int8 full-depth conditioned run is possible in principle
  (flux1 has the tier) and has **not** been run.
* **The PuLID image preprocessing.** The reference builds the EVA-CLIP input from
  a facexlib RetinaFace alignment plus a BiSeNet face parse (background
  whitened, face greyscaled); neither model exists in brain, so the EVA-CLIP half
  of `id_cond` is replayed from the `crates/clip` fixture rather than computed
  from a face photo. The **ArcFace half needs none of it** — PuLID calls
  insightface antelopev2, which *is* `crates/facenet`.
* **`id_cond` is not computed anywhere in this crate — either half.** To be
  precise about the bullet above: `crates/pulid` contains no image → `id_cond`
  path at all. `IdFormer::set_inputs` takes the 1280-d condition and the five
  577×1024 hidden states as host slices, and the parity test supplies them from
  the `face/antelopev2` and `clip/eva02_l336` goldens. The ArcFace half is
  *unblocked* (facenet already does detect + align + embed) but is **not wired**;
  the EVA half is additionally blocked on the missing face parse. Until that
  wiring lands, this crate does not depend on `brain-facenet` or `brain-clip`.
* **Multi-image identity.** The reference's `IDFormer.forward` reshapes by
  `num_id_token * num_duotu` when `x.ndim == 3`, i.e. several reference photos
  per identity. brain takes one 1-D `id_cond` (`num_duotu = 1`) and asserts it.
* Backward / gradcheck (`check_pulid` in `docs/imaging/plan.md` §4 is still
  open), INT8, and the **serving contract** — no capability manifest, no
  residency adapter, no `run_batch`, no D-Bus surface, no CLI. Same position as
  `clip` / `t5` / `flux1` / `unet`.

## Shared change made outside `crates/pulid` — FLAGGED

`crates/flux1` gained an **injection seam**, because there was no way to add
dispatches between backbone blocks from outside the crate:

* **new file** `crates/flux1/src/inject.rs` — `trait BlockInject` +
  `struct InjectSite { x, n_txt, n, d, n_pred }`;
* `crates/flux1/src/lib.rs` — `pub mod inject;` + one re-export line;
* `crates/flux1/src/model.rs` — `run()` takes a trailing
  `Option<&dyn BlockInject>`; two 3-line hook call sites (one per block loop);
  two new public entry points `forward_injected` / `forward_traced_injected`.
  Existing `forward` / `forward_traced` are unchanged in signature and pass
  `None`.

`crates/flux1`'s own suite was re-run after the seam landed:
**13 tests, 0 failed**, `dit_small` worst stage `sg1_txt` at cosine
**0.999999999985** and `dit_small_edit` at **0.999999999987** — unchanged, so
the seam is inert when no adapter is passed (and `flux1_conditioned_parity`'s
`uncond.out` row asserts the same thing from the PuLID side).

The seam is deliberately **not PuLID-shaped**: "run extra dispatches on the image
rows after block *i*" is also exactly what a FLUX ControlNet and an IP-Adapter
need, so `crates/controlnet`'s backbone-agnostic `ControlAdapter`
(`docs/imaging/plan.md` §2) should implement this trait for FLUX rather than add
a second mechanism. **If the controlnet track edited `crates/flux1/src/model.rs`
concurrently, these two changes must be merged by hand.**

One consequence worth naming: a `Step` is only meaningful to the `Gpu` handle
that created it, so the adapter and the backbone must share one handle *and* one
pipeline list. `pulid::joint_kernels()` is that list — `flux1::KERNELS` with
PuLID's extras appended, de-duplicated by name and **without moving any flux1
index**. De-duplication is load-bearing, not tidiness: `gpu_core::upgrade`
resolves its slow→fast redirects by the FIRST matching name, so a second
`("layernorm", …)` entry would be a pipeline that silently never upgrades. It is
returned from a `OnceLock` as a `'static` slice because kernel sets are
identified by slice **address** (`gpu_core::testgpu::dev`'s pool key,
`Gpu::new_like`) — a fresh `Vec` per call would build a fresh device per call.

## Adversarial review pass — what it changed

An independent re-run reproduced every number in the sections above (GPU and
CPU, real weights, nothing skipping) and found no correctness defect. Six
things were fixed; all measured on the same box (Tesla P40, native Vulkan,
`--release`).

1. **The GEMM tier was the training-shaped one.** `Emit::linear` dispatched
   through `model::block::pick_gemm` with `matmul`/`matmul_reg2`. `pick_gemm` is
   the *training-shaped* rule ("is this output big enough to fill a 128×128
   tile?"); the inference-shaped sibling every DiT in the repo uses is
   `model::block::gemm_variant`, and `AGENTS.md` names it as the one rule shared
   by `flux1`/`flux2`. The gap is not academic here: PuLID has two genuinely
   skinny-M linear chains — the `id_map` chain at **m = 1** and every injected
   `to_kv` at **m = 32** (the ID tokens) — the exact regime `matmul_gemv` exists
   for, and `pick_gemm` routes both to the naive one-thread-per-output kernel,
   which re-streams the whole weight once per output element. Everything with
   `m < 128` (the IDFormer's 37 latent rows: `to_q`, `to_out`, both FF linears,
   all 10 layers) also fell to naive. Now: `KERNELS` registers the same
   `matmul` / `matmul_gemv` / `matmul_reg3` trio `flux1` does, `Ki::tier` mirrors
   `Flux1Model::gemm_tier` (gated on the queried
   `DeviceCaps::workgroup_reductions`, so `backend-cpu` keeps the reference arm),
   and `Emit::linear` calls `gemm_variant`. **Measured, same weights, same box:**

   | | before | after |
   |---|---|---|
   | `IdFormer::forward` (one ID embedding) | 244.9 ms | **67.1 ms** (3.7×) |
   | one `PulidCa` injection, `n_img = 4096` (a 1024² latent), submit-only | 46.8 ms | **36.5 ms** (1.28×) |

   At the released 20-site schedule that is ~7 s of GPU time per 50-step
   generation, plus 0.18 s once for the ID embedding. Parity is unchanged: all
   47 comparisons still pass on both backends (numbers below).

   Registering `matmul_reg3` instead of `matmul_reg2` is also what keeps
   `joint_kernels()` from compiling a *second* register-tiled matmul alongside
   flux1's — the joint list now adds only the 7 kernels flux1 genuinely lacks.

2. **Three unused dependencies.** `brain-facenet`, `brain-clip` and
   `brain-imaging` were declared in `Cargo.toml` and referenced by nothing but
   doc comments — cargo links every dependency into every test binary and does
   not warn. Removed, with the reason (and the condition for putting them back)
   in the manifest. This is also the honest statement of the port's scope: the
   crate has **no image → `id_cond` path at all**, for either half. See the
   sharpened bullet in "NOT gated, NOT claimed".

3. **`InjectSite` could not express the noise span.** It carried `n_txt` and `n`
   only, so an adapter on the Kontext edit path could not tell the generated
   image rows from the appended reference-image tokens and would silently
   condition both. Added `n_pred` + `pred_rows()`. `PulidAdapter` keeps
   conditioning the whole image stream (which is what the reference does, and on
   a t2i run the two are identical) and now *says* so; a ControlNet or
   IP-Adapter that must not touch the reference tokens has the field.

4. **A third copy of the LayerNorm selection rule.** `Emit::ln` re-derived
   `DefaultSelector.select(Op::LayerNorm, …) == WorkgroupPerOutput → 64 threads
   per row`, which `model::block::ln_variant` already owns — as did
   `flux1::Flux1Model::ln_rows`, for the structural reason that `ln_variant` was
   private and `layernorm_fwd` binds whole buffers while both DiTs normalise a
   row range. `ln_variant` is now `pub` (it already returned
   `(index, threads)` rather than a `Step`, exactly so both binding shapes could
   share it) and **both** call sites use it. One rule, three callers, no copies.

5. **`flux1_conditioned_parity` gated the schedule only through brain's own
   restatement of it.** It read `small_double`/`small_single`/`id_weight` out of
   the golden's manifest and ignored the `ca_indices`, `double_interval`,
   `single_interval` and `num_ca` the Python dumper had recorded independently.
   It now asserts all four, plus that the shared `ca_idx` counter is sequential
   across the two loops. **The coverage limit this exposes is now written into
   the test:** at 2 + 2 the site list is `{double 0, single 0}` for *any*
   interval ≥ 2, so the injected forward cannot distinguish interval 4 from 3 or
   5 on the single stream — the interval values are gated by the manifest
   assertion and by `config::tests`, not by a forward. Re-dumping the golden at a
   deeper truncation (say 5 + 5 → doubles {0,2,4}, singles {0,4}) would close
   that, and now tightens the assertions automatically when it happens; it was
   not done here because `dump_flux_cond`'s self-validation against `dit_small`
   only holds at the depth `dit_small` itself was dumped from.

6. **Doc drift** from (1), (2) and (4), fixed in place.

Not fixed, reported only: `cargo build -p brain-pulid` surfaces 2 rustc warnings
from **`brain-t5`** (`K_AXPY` never used, `d_bias_blk` never read), reached via
`flux1 → t5`. `crates/t5/src/train.rs` was being edited by another track during
this review, so they were left alone.

## Follow-ups, in dependency order

1. **The sampler loop and VAE glue in `crates/flux1`** — the single blocker on
   any end-to-end claim. Until it exists, `id_weight` and `start_step` cannot be
   evaluated, and no identity-fidelity number (ArcFace cosine between the
   reference face and the generated one) can be produced.
2. **Face preprocessing** for the EVA-CLIP half: a face-parsing model (BiSeNet or
   a substitute) plus the greyscale/whiten transform, or a measured decision that
   feeding the plain aligned crop is good enough. Note `crates/facenet` already
   provides detection + 5-point alignment; what is missing is the parse.
3. **Full-depth conditioned run at int8**, reusing `flux1::Precision::Int8`, to
   get a number for the 20-site schedule rather than the 2-site one.
4. **`check_pulid`** (backward over the adapter with the backbone frozen) — the
   `docs/imaging/plan.md` §4 entry. **Read this before starting it:** the
   forwards here are *inference*-shaped and deliberately violate the SSA
   discipline a training-mode forward needs. `IdFormer` ping-pongs `lat_a`/
   `lat_b` and reuses `nkv`/`q`/`kv`/`fh`/`fg`/`fo` across all 10 layers, so
   layer *l*'s activations are gone by layer *l+1* (this is exactly why
   `read_tap` re-submits a prefix instead of reading after one forward), and
   `PulidCa::inject_steps` mutates the backbone's residual slab in place. A
   training-mode forward must allocate per-layer buffers; it is a second graph,
   not a flag on this one. Zero of PuLID's 312 parameters have any gradient
   coverage today — there is no `gradcheck::check_pulid` and no `pulid` entry in
   `crates/gradcheck`, so nothing is silently passing.
5. **Serving contract**: `pulid::caps`, a residency adapter, D-Bus, an example.
6. `docs/imaging/plan.md` and `AGENTS.md` still describe `crates/pulid` as an
   empty skeleton; both need an entry. Neither was edited here because both are
   shared with concurrent tracks.

## Reproduce

```bash
python tools/goldens/pulid_dump_reference.py \
    --pulid  /path/to/pulid_flux_v0.9.1.safetensors \
    --code   /path/to/PuLID \
    --testdata testdata \
    --transformer /path/to/FLUX.1-Kontext-dev/transformer

BRAIN_PULID=/path/to/pulid_flux_v0.9.1.safetensors \
BRAIN_FLUX1_TRANSFORMER=/path/to/FLUX.1-Kontext-dev/transformer \
cargo test --release -p brain-pulid -- --nocapture --test-threads=1
# (On the original dev box only: prefix CARGO_HOME=/data/resources/cargo-home
#  -- a machine-local quirk, not part of the command.)
```

Every test skips itself (never fails) when its fixture or its weights are absent.

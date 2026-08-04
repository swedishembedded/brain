# Imaging workstream — segmentation, identity-preserving editing, restoration

Plan of record for adding pixel-accurate segmentation, identity-preserving
conditioning, face restoration and instruction editing to brain, plus the
pipeline that composes them.

Resources (code, weights, papers, configs) live under
`/data/workspace/resources/{sam2,identity,face-restore,pipeline,flux1-kontext}`
— the same layout as the existing `resources/flux` workstream. Nothing in
`crates/**` may reference those paths (AGENTS.md "no absolute paths"); weights
resolve from `$BRAIN_TESTDATA` or a CLI flag.

---

## 1. Goal

Make *"change only X, keep the person the same"* an exact operation rather than
a prompt-engineering hope:

| Need | Model | Why it is the enabler |
|---|---|---|
| Pixel-accurate region | **SAM 2** | point/box-prompted masks with hair-edge accuracy; every "only X" edit becomes exact by construction |
| The right face | **PuLID** (+ ArcFace) | injects a face-recognition embedding into diffusion attention, so identity is a *model operation*, not a post-hoc patch |
| Clean teeth/eyes/skin | **CodeFormer** | fixes exactly the artefacts generation leaves, with an identity-fidelity dial |
| Edit without drift | **FLUX.1 Kontext** | trained on edit triplets to preserve identity across edits — the capability FLUX.2 Klein lacks |

Plus, per the follow-up scope decision: text-prompted segmentation, matting +
inpaint/removal, and an upscale tail.

---

## 2. The single most important architectural finding

**FLUX.1 is the keystone, and it is one port, not two.**

- `FLUX.1-Kontext-dev` is a FLUX.1-architecture MMDiT (12B: 19 double-stream +
  38 single-stream blocks, 3-axis RoPE θ=10000, per-block modulation, T5-XXL +
  CLIP-L conditioning, 16-channel VAE).
- `PuLID-FLUX v0.9.1` — the identity adapter we downloaded — targets **that same
  FLUX.1 backbone**.

So one `crates/flux1` unlocks items 2 *and* 4 of the request. It is also a
**cheap** port relative to its size, because `crates/flux2` already implements
the harder descendant: double-stream joint attention, single-stream parallel
blocks, modulation folding, interleaved multi-axis RoPE, flow-matching
scheduling, LoRA, INT8 and sharding all exist. FLUX.1 differs in four bounded
ways — per-block (not 3 global) modulation, 3 RoPE axes at θ=10000 (not 4 at
2000), T5+CLIP (not Qwen3) conditioning, and a 16-ch (not 128-ch) VAE latent.

### The second finding: the UNet family is cheap, so InstantID stays

InstantID is SDXL-based — it needs a `UNet2DConditionModel` *plus* a ControlNet,
and at the time this was written brain had **no UNet diffusion backbone** (only
DiT: `dit`, `zimage`, `flux2`). The initial read was that this made InstantID a
bad trade.

> Correction, recorded because the sentence above was stated too strongly:
> `crates/wm-diamond` **is** a UNet-shaped diffusion model (an EDM world model
> with a down/mid/up + skips + resnets + self-attention `DiamondUNet`). What was
> missing was a *text-conditioned image* UNet, and — the part that matters for
> reuse — `wm-diamond` records its graph **by hand** rather than through
> `vae::blocks::Builder`. Since `crates/unet` landed (phase 4b) brain has two
> independent UNet graph recorders; migrating `wm-diamond` onto the shared
> builder is an open follow-up, filed in `docs/models/unet/status.md`.

Measuring it says otherwise: **a UNet + ControlNet family needs zero new
kernels.**

| UNet component | What it needs | Status in brain |
|---|---|---|
| ResBlock | GroupNorm, SiLU, conv, timestep scale-shift | `gn_{stats,apply,dx,dgamma,dbeta,part,dsum}`, `silu`, `conv2d_gd`, `film_chan` — **all present** |
| Spatial transformer | self-attn + cross-attn + GEGLU FF | full attention family + `geglu_shift{,_da,_db}` — **present** |
| Up blocks | nearest upsample + conv | `upsample2`, `resize_nearest` — **present** |
| ControlNet | conv conditioning stack + zero-convs + residual add | plain convs and `add` — **present** |
| SDXL VAE | AutoencoderKL, 4 latent channels | `crates/vae` is already **config-driven** (`VaeConfig`); 4-ch is a config, not code |

So the work is *composition*, not new GPU math — the expensive part of a brain
port is exactly the part that is already done. What genuinely has to be built:

1. **`crates/unet`** — the `UNet2DConditionModel` graph (down/mid/up blocks,
   skip connections, timestep + added-conditioning embeddings).
2. **`crates/controlnet`** — see below; deliberately **backbone-agnostic**.
3. **`crates/clip`** — brain has no CLIP. SDXL needs CLIP-L **and**
   OpenCLIP-bigG/14; FLUX.1 needs CLIP-L; PuLID needs EVA-CLIP-L/336. One crate
   serves all three — another consolidation the workstream pays for once.
4. **Discrete schedulers in `crates/diffusion`** — today it is flow-matching
   only (`FlowMatchEulerScheduler`). SDXL needs DDIM / Euler-ancestral /
   DPM-Solver++ with ε- and v-prediction parameterisations.

**Revised recommendation: build the UNet family and keep InstantID.** PuLID-FLUX
is still worth doing (it rides the FLUX.1 backbone Kontext needs anyway), so the
two identity approaches become complementary rather than exclusive — and having
both lets us measure identity fidelity against each other on the same fixtures.

### ControlNet should be a seam, not an SDXL crate

The important design choice. A ControlNet is a trainable copy of the backbone's
early blocks whose zero-conv outputs are **added as residuals** at named
injection points. That structure is not UNet-specific — FLUX ControlNets inject
into double-stream blocks the same way.

So `crates/controlnet` defines a backbone-agnostic `ControlAdapter` trait over
*named injection points*, implemented by both `unet` and the FLUX DiTs, rather
than hardcoding SDXL's down-block list. This is the "make brain better in the
process" part: one control-conditioning mechanism for every current and future
diffusion backbone, and it composes with the existing sharding/INT8/residency
machinery instead of duplicating it.

Two synergies worth naming: brain's **ZipDepth already produces depth maps**, so
a depth-ControlNet is wired end-to-end inside brain with no external
preprocessor; and a UNet family also unlocks the whole SD/SDXL ecosystem
(community LoRAs, inpainting checkpoints, T2I-Adapter, IP-Adapter) on top of
brain's existing LoRA and INT8 paths.

### Two substitutions still worth making

- **Florence-2 over GroundingDINO** for text→box. GroundingDINO needs
  multi-scale **deformable attention** (a bilinear-gather attention kernel with
  no relative in brain, plus its backward). Florence-2 is a DaViT vision encoder
  + a BART-style encoder-decoder — and brain already has `crates/seq2seq`
  (bidirectional encoder + causal/cross-attention decoder, gradient-checked).
  Weights for both were downloaded; Florence-2 is the lower-risk path.
- **Defer LaMa.** Its Fast Fourier Convolution needs a full FFT kernel family
  (forward + backward) that nothing else in brain wants. Object removal can be
  served first by FLUX.1 inpainting through the mask SAM 2 produces. Revisit FFT
  only if LaMa-specific quality is needed.

---

## 3. Gap analysis

### 3.1 Kernels

brain has 329 WGSL kernels and the base is better than expected: `conv2d_gd` is
fully general (arbitrary K, stride, pad, dilation, **groups** — so 7×7 depthwise
ConvNeXt blocks and the 7×7/s4 patch embed are already covered), `vq_argmin`
exists for VQ codebook lookup (CodeFormer), `rope2d` for 2D RoPE, BatchNorm is
complete, and the splat scan/sort primitives are reusable.

Genuinely missing, in dependency order:

| # | Kernel | Needed by | Notes |
|---|---|---|---|
| 1 | `maxpool2d` + `_dx` | Hiera `q_pool`, SCRFD | **DONE.** Generalized `maxpool5` (added `stride` + explicit `Ho`/`Wo`); `maxpool5{,_dx}` deleted and SPPF migrated, so there is one max-pool, not two |
| 2 | `convtr2d` + `_dw` + `_dx` | SAM 2 mask decoder (~~VQGAN/CodeFormer decoder~~ — see note) | **DONE.** Built on the `convtr1d` shape template; same 12-word `Params` as `conv2d_gd` (square K, symmetric pad, bias-free). **Correction:** the VQGAN/CodeFormer generator does NOT use a transposed convolution — `vqgan_arch.py`'s `Upsample` is `F.interpolate(scale_factor=2, mode='nearest')` followed by `Conv2d(k3,s1,p1)`, which is the existing `upsample2` + conv path. `crates/vqgan` dispatches no `convtr2d` |
| 3 | `prelu` + `_bwd` + `_bwd_wg` | ArcFace IResNet | **DONE.** `leaky_relu` has a *fixed* slope; PReLU's is a learned per-channel parameter needing its own grad. Ships as a barrier-free reference (`prelu_bwd`, dispatch `C`) **plus** a cooperative twin (`prelu_bwd_wg`, dispatch `C*64`), the `gn_stats`/`gn_stats_wg` pairing. Selecting between them on the queried `DeviceCaps::workgroup_reductions` is a **correctness gate, not a perf tweak** — `backend-cpu` reports it false, and a barrier-only version returned `da` ALL ZEROS there with no error, i.e. a PReLU whose slopes never move |
| 4 | `grid_sample` + `_dx` + `_dgrid` | face alignment (similarity warp), ROI align | **DONE.** Bilinear gather, `padding_mode='zeros'`, both `align_corners`. The backward is split in two (different parallelization, different bindings) rather than one `_bwd`, matching `conv2d_gd_dx`/`_dw` |
| 5 | `resize_bicubic` + `_dx` | SAM 2 `pos_embed` interpolation | **DONE.** Joins `resize_bilinear`/`resize_nearest`, byte-identical `Params`. NOT the same function as `mirror::preprocess::resize_bicubic_torch` (that one is `antialias=True`); antialiased downsampling still has no kernel |

**Measure before adding two more.** Per `docs/kernel-checklist.md`, the rule is
to fix *selection*, not add copies:

- *Windowed attention* (Hiera, Swin): brain's `chunked_bidir_fwd` is already
  **span-based** — disjoint windows are exactly disjoint spans. Try composing it
  with `region_copy` first; only add `window_partition`/`window_reverse` if the
  profile says the gather dominates.
- *LayerNorm2d* (channels-first): compose `nchw_nlc` → `layernorm_rows` →
  `nlc_nchw` first. `layernorm_rows` is the coalesced cooperative kernel
  (2.3–9.1× on a P40); one-thread-per-row is the documented coalescing trap.

Every new kernel: `@workgroup_size(64)`, ≤8 storage buffers, fp32, no atomics/
subgroups/f16, then `make kernels-regen`.

#### 3.1.1 Measured — both decisions, phase 0

Both measurements were **run**, not asserted, and both are pinned by a test that
re-runs them. A methodology note that invalidated the first attempt at the
LayerNorm2d number and must be observed by anything timing this engine:
`WgpuBackend::submit` with an empty clear list only appends to `pending` — it
encodes and queues nothing. A timing loop of bare `submit`s measures host-side
bind-group construction and reports it as device bandwidth (the first run showed
**377 GB/s on a ~346 GB/s card**, and a permute cost that did not vary with size
— both signatures of a host cost). Every timed region must be bracketed by
`Gpu::poll_wait()`, which flushes the pending pass and blocks.

**LayerNorm2d — compose vs fuse.** `crates/vision/tests/imaging_blocks.rs::layernorm2d_composition_cost`,
Tesla P40 (Vulkan), release, synchronised, warm-ups drained:

| shape | bytes | total | 2 permutes | permute | norm |
|---|---|---|---|---|---|
| 1×96×64×64 | 1.5 MiB | 0.104 ms | 0.077 ms (74.5 %) | 81.3 GB/s | 0.027 ms |
| 1×896×32×32 | 3.5 MiB | 0.212 ms | 0.143 ms (67.4 %) | 102.8 GB/s | 0.069 ms |
| 1×448×64×64 | 7.0 MiB | 0.349 ms | 0.260 ms (74.6 %) | 112.8 GB/s | 0.088 ms |
| 1×224×128×128 | 14.0 MiB | 0.816 ms | 0.700 ms (85.9 %) | 83.9 GB/s | 0.115 ms |
| 1×112×256×256 | 28.0 MiB | 3.000 ms | 2.474 ms (82.5 %) | 47.5 GB/s | 0.526 ms |

**The measurement does not support the prediction it was asked to test.** The
two permutes are 67–86 % of the composition and run at **14–33 % of the roof**,
not at it. The mechanism explains why: `nchw_nlc` gathers
`x[(n*C+ch)*HW + l]` with `ch` varying fastest, so a warp's 32 loads land `H*W`
floats apart; `nlc_nchw` is its mirror. **Both permutes already pay the sector
amplification that was the reason to reject a fused kernel**, and it worsens with
`H*W` (47.5 GB/s at `HW=65536` vs 102.8 at `HW=1024`). The row-oriented
LayerNorm between them is the only coalesced stage.

*Decision:* composition is what shipped and is the right **first**
implementation — correct, no new kernel, one `*_rows` selection site. But a
fused `layernorm2d` would trade ~6 passes (4 of them strided) for ~2 strided
passes, so it is **not ruled out** and this measurement must not be cited as if
it ruled it out. Adding it is its own task: `docs/kernel-checklist.md`,
gradcheck, `make kernels-regen`.

**Windowed attention — gather vs a dedicated kernel.**
`crates/model/tests/vit_windowed.rs::measure_gather_vs_windowed_attention`,
same box, 10 iterations, `perm/blk` = permutation ÷ block (not ÷ their sum):

| config | perm | attn | q_pool | block | perm/blk |
|---|---|---|---|---|---|
| 64×64×256 / w8 | 0.158 ms | 4.687 ms | 0.234 ms | 44.094 ms | **0.36 %** |
| 128×128×112 / w8 | 0.153 ms | 16.116 ms | 0.253 ms | 39.718 ms | **0.39 %** |

*Decision:* the gather does **not** dominate — it is under half a percent of the
block. `embed` + `row_scatter` compose the partition and its exact inverse, so
**no `window_partition` / `window_reverse` kernel is added.** Revisit only if a
profile at a materially different shape moves this above a few percent.

### 3.2 Shared blocks (the reuse story)

New capability belongs in existing shared homes, not per-model copies:

- **`model::vit`** — **DONE.** `WindowPlan` (regular + Swin-shifted `axis_cuts`,
  verified against the roll algebra on both axes), windowed-span attention, and
  `q_pool`/`q_pool_bwd`, so Hiera, Swin (BiRefNet) and DaViT (Florence-2) all
  compose from it. Zero new kernels: the partition is `embed` + `row_scatter`
  (exact inverses) and `q_pool` is `nlc_nchw` → `maxpool2d` → `nchw_nlc`.
  Two alignment constraints are load-bearing and now enforced rather than
  assumed — `min_storage_buffer_offset_alignment` is 256 B, so per-span `probs`
  slabs are padded to 64 floats and the `q`/`kv` row offsets moved into the
  kernels' own `q_off`/`k_off`/`v_off` Params instead of being buffer slices.
  `attn_apply_cross` still has **no output-offset Param**, so `ctx` stays sliced
  and a span's `q0*C` must be 64-aligned (always true for `C % 64 == 0`);
  `WindowPlan::ctx_bindable(dim)` lets a model check up front. Adding `out_off`
  to those three kernels is a deliberate kernel task (their ABI is shared with
  `seq2seq` and `fastvlm`), not a phase-0 edit.
- **`vision::blocks`** — **DONE.** `ConvTranspose`, generic `MaxPool`,
  `LayerNorm2d` and the ConvNeXt `CXBlock` sit next to the existing spec-driven
  `Conv`/`SPPF`, each with its backward and a finite-difference test. `SPPF`
  composes the new `MaxPool` (one max-pool dispatch site in the crate).
  Still owed for finetuning: `CXBlock` does not model **DropPath** (identity at
  eval, so parity is unaffected).
- **`crates/sam2`** *(new)* — **FORWARD DONE.** Hiera trunk → FPN neck → prompt
  encoder → two-way mask decoder, image path only. Both released variants
  (`sam2.1_hiera_tiny`, `sam2.1_hiera_large`) are parity-gated stage by stage
  against a hooked `sam2.modeling.sam2_base.SAM2Base`; measured numbers in
  §"Phases 1–3b gate" below. No block or kernel is private to the crate: the
  trunk composes `model::vit`'s windowed spans + `q_pool`, the neck and decoder
  compose `vision::blocks`, and the ImageNet constants come from
  `imaging::{IMAGENET_MEAN, IMAGENET_STD}`. Genuinely model-local and justified:
  the host-side positional encoding (`hostpe`: 2D sinusoidal `sine`, the random
  Fourier `pe_encode`/`dense_pe`, `tile_chw`) — nothing else in the tree has a 2D
  sinusoidal PE — and the two-way decoder's cross trio, which needs `k` from
  `queries+pe` and `v` from `queries`, a split `vit::cross_q_fwd`'s single fused
  kv buffer cannot express.
- **`crates/facenet`** *(new)* — **FORWARD DONE.** ArcFace IResNet-100 + SCRFD
  detection + the 5-point similarity-transform alignment. This mirrors
  `crates/speaker` (ECAPA-TDNN) exactly: an embedding model consumed by a
  generative one. Measured numbers in §"Phases 1–3b gate" below; ledger:
  `docs/models/face/status.md`.

  This is the **first ONNX import in the repo** — the insightface `antelopev2`
  release ships ONNX and only ONNX. The protobuf reader was therefore put in
  `crates/onnx` (`onnx::read`, built on the existing `decode_model`), not in the
  model crate: a private decoder inside `facenet` would be a second reader of the
  same wire format that nothing compares against the first. Three hoists came out
  of the port rather than staying local — `vision::blocks::AvgPool` (the sibling
  of `MaxPool`; SCRFD's ResNet-D shortcut is `AveragePool(2,2)→conv1×1`, and
  spelling it as a strided 1×1 gives the right shape and a half-pixel shift in
  every feature), `vision::blocks::PReLU` (a *learned* per-channel slope, so a
  block with a parameter and a gradient, not an `Act` variant), and
  `model::hostmath::{cosine, l2_normalize}`.
- **`crates/vqgan`** *(new, small)* — **FORWARD DONE.** The VQ
  encoder/decoder/codebook shared by CodeFormer. Forward parity vs the
  `basicsr` reference is cosine 1.000000000 on every one of the 25+25 blocks at
  128², on the 512² end-to-end (synthetic + real face), and on the quantizer
  unit — with **zero** code-index disagreements — for **both** checkpoints
  (`codeformer.pth`, `vqgan_code1024.pth`). The CodeFormer transformer +
  fidelity dial are `crates/restore`; the serving contract is met by
  `vqgan::caps` + `resident_restore::VqganResident` + `examples/restore/`
  (2026-08-04). Backward/gradcheck is the remaining follow-up.

  Reuse was made real by **hoisting**, not copying: `crates/vae`'s private
  `Builder` is now the public `vae::blocks::Builder` (parameterised by
  `BlockNames`, so `conv_shortcut`/`to_q…` and `conv_out`/`q…` are the same
  block), and `AutoencoderKL` was migrated onto it in the same change.
  `vqgan` adds **no kernel and no block**: `vq_argmin` (via `wm_core::vq::Vq`)
  for the assignment and `embed` for the lookup. Ledger (measured numbers,
  findings, what the deferred work needs): `docs/models/vqgan/status.md`.
- **`crates/clip`** *(new)* — **FORWARD DONE.** One config-driven graph serving
  all three towers the workstream needs: CLIP-L and OpenCLIP-bigG/14 text (SDXL
  conditioning, and CLIP-L again for FLUX.1) plus the EVA-CLIP-L/336 image tower
  (PuLID). Measured numbers in §"Phases 1–3b gate" below. The EVA attention
  dispatches through `model::block::bidir_fwd` rather than a hand-written param
  list — the `docs/kernel-checklist.md` §B surface — with the backward slots set
  to `usize::MAX` sentinels so a forward-only tower panics loudly instead of
  running a silently-wrong kernel. One new kernel: `quick_gelu` (`x·σ(1.702x)`),
  which is genuinely absent from the tree and is **not** `silu` (`x·σ(x)`); the
  factor is the whole difference and mixing them cost cosine 0.504 before it was
  added. `vit::vit_block_fwd` deliberately does not fit (no `inner_ln` hook, an
  `fc1/act/fc2` MLP rather than SwiGLU, and it is non-SSA so the per-stage taps
  could not exist through it).

  The CLIP **BPE tokenizer is implemented** (2026-08-04) as
  `data::clip_bpe::ClipBpe`, next to the GPT-2 and Qwen BPEs. It reuses
  `data::bpe`'s merge loop verbatim and adds only what differs: the `</w>`
  word-end marker, CLIP's pre-tokenization (no leading-space rule, ONE digit per
  pre-token, whitespace dropped), lowercase + whitespace collapse, and the
  `<|startoftext|>`/`<|endoftext|>` frame at the fixed 77 context. Gated by
  `crates/data/tests/clip_tokenizer_parity.rs` at **exact id equality** vs HF's
  `CLIPTokenizer` over 32 tricky strings × both SDXL tokenizers, plus an
  8000-string randomized differential run against `transformers` (0 mismatches
  on either tower) that is how the two pre-tokenizer defects below were found —
  it is a review tool, not a committed test, since it needs `transformers`.
  Two things the pinned corpus now nails down, each having been wrong:
  (a) a greedy `[^\s\p{L}\p{N}]+` run does **not** yield to the `'s`/`'re`
  alternatives — Python's alternation priority applies where a match *starts*,
  so `%'s` is `["%'", "s"]`, and breaking out of the run was wrong on 351/3000
  fuzzed strings; (b) `\p{L}` is the general category `L*`, not
  `char::is_alphabetic` (which is the `Alphabetic` property, `L* | Nl |
  Other_Alphabetic`), so a roman numeral (`Nl`) must take the `[\p{N}]` branch.
  The residual `Other_Alphabetic` gap (1510 codepoints — combining marks, so
  Indic/Thai/Hebrew vowel signs) is measured and documented in the module, not
  silent.
  A finding worth carrying: the two SDXL tokenizers are **not** "same ids,
  different padding" — `tokenizer_2` registers its pad token `!` as an *added*
  token, so HF splits on it before the BPE and a literal `!` is id 0 there
  versus `!</w>` (256) in `tokenizer/`.
  *Still owed:* `crates/clip`'s parity tests still feed token ids from the
  goldens rather than driving the tokenizer, and no `encode_prompt`-shaped
  helper ties tokenizer → tower yet.
  Preprocessing (decode → bicubic 336 → normalize) is likewise absent;
  its goldens (`image_raw`/`image_mean`/`image_std`) are already dumped and are
  the gate for wiring `crates/imaging` to it.
- **`crates/imaging`** *(new)* — **DONE.** The image substrate that was ~60 sites
  across 24 crates: decode/encode, a device `Ctx` (resize/crop/pad/affine over
  the kernels), mask algebra (union, feather, dilate, composite, IoU), tiling for
  >1 MP work, colour conversion, and the letterbox. Consolidating this is a
  precondition for a *pipeline* rather than four models with private image code.

  **The rule the crate is held to: being the home means the old copy is gone,
  not shadowed.** A consolidation crate that only *adds* a definition raises the
  copy count. Each item therefore has no predecessor, or its predecessor now
  re-exports it. Post-migration census: host bilinear resize **6 → 1**
  (`imaging::host::resize_bilinear_hwc`, channel count a parameter), letterbox
  definitions **1**, host CHW/HWC/RGB8 converters **3, all in `imaging::pixels`**,
  P6 header writers outside `events`/`imaging` **0**, `IMAGENET_MEAN`/`STD`
  **1 pair**. `crates/capture` lost `convert.rs` and is V4L2-only again.
  `imaging` has **7 reverse dependencies** (`yolo`, `npu`, `mirror`, `depth`,
  `wm-display`, `cli`, plus itself in the workspace list) — not a crate with no
  consumers.

  Deliberately **not** migrated, each for a stated numeric reason, listed in
  `crates/imaging/src/lib.rs`: `data::{imageset,gen_detect}` (a real dependency
  cycle — `imaging → vision → model → data`), `mirror`'s PIL/torch bicubic
  variants (parity-gated, and not `resize_bicubic.wgsl`'s `a = −0.75`
  no-antialias function), `zimage::pipeline::{feather_mask,downsample_mask}`
  (different float summation order, no in-tree inpaint metric to gate the ramp),
  `zimage::caps::build_outpaint_canvas` (needs a `pad_mode` word on
  `pad2d.wgsl`), and `fastvlm::caps::pad_resize_chw` (pad-then-resize is a
  different function).

  Two open items carried forward, both recorded at their call sites:
  survey §6.2 — `brain depth calib` letterboxes while `depth::predict` does an
  aspect-preserving bilinear resize; the migration preserved this bit-for-bit so
  the INT8 scales did not move, and fixing it needs its own quantized-accuracy
  gate. And `events::ppm` stays the P6 definition (`imaging::codec` re-exports
  it) because moving it would give the wasm build a JPEG decoder it cannot use.

### 3.3 Training / finetuning (explicit requirement)

Per AGENTS.md every model is gradient-checked, and per the follow-up all of these
need training/finetuning. New `gradcheck` entry points:

| Entry | Covers | Loss | Status |
|---|---|---|---|
| `check_sam2` | Hiera + FPN + prompt encoder + two-way mask decoder | focal + dice on masks, MSE on the IoU head | **LANDED, decoder only** — trunk + neck `Role::Frozen`; 128 tensors |
| `check_arcface` | IResNet backbone + margin head | ArcFace additive-angular-margin CE | **LANDED** — 53 tensors |
| `check_vqgan` | VQ encoder/generator/codebook | straight-through VQ + codebook/commitment | **LANDED** — 87 tensors |
| `check_clip` | CLIP-L **text tower only** | — | **LANDED, partial** — bigG and the EVA image tower owe theirs |
| `check_t5` | the T5-XXL encoder backward (`t5::train::T5Trainer`) | MSE on the encoder output | **LANDED** — 17 tensors, + `_one_block` (10), `_tiled` (10), `_rel_bias_elementwise` (24 entries) |
| `check_codeformer` | CodeFormer's **code-prediction Transformer**, VQ autoencoder frozen | CE on code indices | **LANDED** — 34 tensors, + `_one_layer` (20) |
| `check_flux1` | FLUX.1 MMDiT | flow-matching MSE (mirror `check_flux2`) | **NOT DELIVERED.** Stated blocker: hoist `flux2::grad`'s scalar `Fp` primitives into `crates/dit` rather than take a `flux1 → flux2` dependency |
| `check_unet` | SDXL `UNet2DConditionModel` | — | **NOT DELIVERED** — `crates/unet` has no backward |
| `check_controlnet` | the ControlNet's down/mid copy + zero-convs | — | **NOT DELIVERED** — `crates/controlnet` has no backward at all |
| `check_pulid` | ID adapter layers only, backbone frozen | flow-matching MSE + ID cosine | **NOT DELIVERED** — `crates/pulid` has no training-mode forward; its `IdFormer` ping-pongs `lat_a`/`lat_b` and reuses `nkv`/`q`/`kv` across all 10 layers, so a training graph is a **second graph, not a flag** |

**The gate that gates the gate.** `directional_check` keeps the *best-agreeing*
of `n_dirs` random ±1 contractions, which is blind to a **partially** wrong
gradient — the exact signature of a folded or shared parameter accumulated over
only some of its contributors. Measured: deleting T5's cross-block `axpy` fold
leaves `rel_bias.weight` **33 % wrong** (‖Δg‖₂ 0.672 vs ‖g‖₂ 2.044) and every T5
directional check still passes on both backends. `gradcheck::elementwise_check`
(per-**entry** central differences) is the answer and is wired for
`rel_bias.weight`. **Every other folded/shared parameter in the repo is still
covered only by the blind check** — `restore`'s `position_emb` passes today
because its injected error happened to be large, not because anything
structurally catches it.

LoRA finetune paths mirror `flux2::{lora,finetune}`. SAM 2 finetuning follows
the upstream MOSE recipe (`sam2.1_hiera_b+_MOSE_finetune.yaml`, downloaded).

**The VQ straight-through estimator is the one genuinely new gradient form** —
`vq_argmin` is currently inference-only; the backward copies the decoder grad
past the quantiser and adds the codebook commitment term.

### 3.4 NPU

Each model needs a `crates/npu/src/<model>_topology.rs` + `_export.rs` built on
the shared `TopoBase` DSL. ConvTranspose, MaxPool, Resize and GridSample are all
standard ONNX operators, so nothing here needs a custom OpenVINO path. Cannot be
validated on this box (CPU + 2×P40 only) — export and graph-shape tests run
here; `make parity` NPU legs run on the Ultra box.

### 3.5 Serving contract

`capability::Media` already has `Image` and `Mask`, and D-Bus `Run` already
passes blobs by fd (memfd/dmabuf), so **no D-Bus surface change is needed for
single-shot segment/restore/edit calls**. What each model still owes (AGENTS.md,
`docs/serving-contract.md`): a `Provider`, a `resident_*.rs` adapter registered
in `resident::build_executor`, a real batched `run_batch`, and a runnable
`examples/<domain>/` client.

The **pipeline** is the one place the surface must grow: composing
segment → edit → restore → upscale as separate D-Bus round-trips would move
full-resolution images across the bus 4×. Proposal: express the pipeline as a
`capability::Action` itself (`imaging.pipeline`), taking a JSON graph of stages,
so it stays one `Run` call with fd in/out and is discoverable through
`brain caps` like everything else — no new method, no side channel.

---

## 4. Phasing

Each phase ends at a validation gate and is independently committable.

| Phase | Content | Gate |
|---|---|---|
| **0. Foundations** ✅ **DONE** | kernels 1–5 + `crates/imaging` + `vision::blocks` additions + `model::vit` windowed spans | `make kernels-regen`, `make test`, per-kernel parity vs a CPU reference — **met**, see below |
| **1. SAM 2** ◑ **FORWARD DONE** | Hiera trunk → FPN neck → prompt encoder → two-way mask decoder; image path only (no video memory bank) | parity vs dumped goldens — **met** (283/283, see below); the **serving contract is met** (`sam2::caps` `segment`, `resident_sam2.rs`, D-Bus `Run`, `examples/vision/`, `run_batch` grouped by image); `check_sam2` **landed for the mask decoder** (128 tensors, 0 failures, trunk+neck `Role::Frozen`) — **full-trunk training still owed** |
| **2. Face recognition** ◑ **FORWARD DONE** | `crates/facenet`: SCRFD + IResNet-100 + alignment | cosine ≥0.999 vs insightface goldens — **met** (1.0000000 on every tap, see below); the **serving contract is met** (`facenet::caps` `detect`/`embed`, `resident_facenet.rs`, D-Bus `Run`, `examples/vision/`; the insightface detector letterbox now lives in `caps.rs`), `check_arcface` **landed** |
| **3. CodeFormer** ◑ **FORWARD DONE** | `crates/vqgan` + `crates/restore` (code-prediction transformer, `Fuse_sft_block` CFT, identity-fidelity dial `w`) | VQ encoder/decoder/codebook parity **and** the full restorer — **met** (see the phase 3c/4 gate below); the **serving contract is met** for both crates (`vqgan::caps` `encode`/`decode`, `restore::caps` `restore_face`, `resident_restore.rs`, D-Bus `Run`, `examples/restore/`); `check_vqgan` **landed** for the VQ half; **`check_codeformer` — the code transformer + CFT — is still owed**, and `adain=True` — the reference CLI's actual path — is **not implemented** |
| **3b. Text encoders** ◑ **FORWARD DONE** | `crates/clip`: CLIP-L + OpenCLIP-bigG + EVA-CLIP-L/336 behind one config-driven graph | cosine ≥0.9999 vs HF per encoder — **met** (see below); the **BPE tokenizer is done** (`data::clip_bpe`, exact-id gate vs HF on both SDXL tokenizers) but is **not yet wired into `crates/clip`'s tests**; **image preprocessing is not implemented**; `check_clip` **landed for the CLIP-L text tower**; bigG and the EVA image tower still owe theirs |
| **4. FLUX.1** ◑ **TRANSFORMER FORWARD DONE** | `crates/flux1` reusing `dit`/`vae`/`model::block`; **`crates/t5`** (T5-XXL encoder) + phase-3b CLIP-L; Kontext edit path | forward cosine vs diffusers — **met for the transformer and for T5-XXL** (see the phase 3c/4 gate below). **fp32 parity is gated at reduced depth only** (47.6 GiB does not fit a 24 GiB card); full depth is gated at **int8**. No sampler loop, no VAE glue, no CLI, no `check_flux1` |
| **4b. UNet family** ◑ **FORWARD DONE** | `crates/unet` (SDXL `UNet2DConditionModel`) + discrete schedulers (DDIM/Euler/Euler-a/DPM++, ε and v-pred) in `crates/diffusion` | forward cosine vs diffusers SDXL — **met** (165 comparisons / 0 failed, `out.sample` cosine 1.0000000000, rel_l2 3.258e-6; schedulers 66 checks / 0 failed — see the phase-4b gate below); `check_unet` **still owed**, and so is the **whole serving contract** (no capability, no residency adapter, no `run_batch`, no D-Bus, no example) plus the sampler loop, the VAE/text-encoder glue and batch > 1 |
| **4c. ControlNet** ◑ **SEAM + RESIDUAL PARITY DONE** | `crates/controlnet`: backbone-agnostic `ControlAdapter` over named injection points; SDXL impl first, FLUX impl second; depth conditioning fed by brain's own ZipDepth | residual parity vs diffusers ControlNet — **met** (140 comparisons / 0 failed, worst 1−cos 1.914e-11, on **both** backends; see the phase-4c/5 gate below). The FLUX impl is **not** written (only asserted implementable), ZipDepth is **not** wired, and **`check_controlnet` does not exist** — there is no backward in the crate at all. No serving contract |
| **5. Identity** ◐ **PuLID FORWARD DONE; InstantID NOT STARTED** | PuLID adapter on the phase-4 FLUX.1 backbone **and** InstantID (phase-4b/4c SDXL + IP-Adapter-FaceID), both fed by phase-2 ArcFace embeddings | PuLID: IDFormer + the injected CA + the **conditioned FLUX.1 forward** parity-gated on both backends (47 comparisons, worst 1−cos 1.44e-11). **Not met and not started:** InstantID, `check_pulid`, `check_instantid`, and the head-to-head measurement. **`crates/pulid` has no image → `id_cond` path** — `id_cond` is a host slice supplied by the caller, so the phase-2 ArcFace and phase-3b EVA-CLIP towers are *composable* but **not composed** |
| **6. Pipeline** | `imaging.pipeline` action + matting (BiRefNet) + Florence-2 text→box + Real-ESRGAN tail | end-to-end "change only X" on a fixture set |

#### Phase 0 gate — what was actually run

Tesla P40 (wgpu/Vulkan) and `BRAIN_DEVICE=cpu`, release. Workspace `cargo build`
and `cargo build --release`: **zero rustc warnings** (the two `cargo:warning`
lines are build-script notes from `brain-vulkan`/`brain-npu` about an absent
glslc / OpenVINO and are not code warnings). Absolute-path gate
`grep -rnE '"/(data|home|tmp|opt|mnt|root)/' crates` empty.

| suite | result |
|---|---|
| `brain-imaging` | 41 lib + 3 `color` + 19 `device_ops` — 63 passed, 0 failed (device ops re-run on the P40) |
| `brain-vision` | 8 lib + 13 `imaging_blocks` — 21 passed, 0 failed |
| `brain-model` | 54 lib + 8 `vit_windowed` + 2 `vit_block_gradcheck` + 3 `vit_bwd_kernels` + 2 `chunked_attn` + rest of the integration set — 0 failed |
| `brain-gradcheck` | 14 lib; `maxpool2d_kernels` 10, `grid_sample_kernels` 12, `prelu_kernels` 10, `resize_bicubic_kernels` 9, `convtr2d_kernels` 8, `attn_bidir_fd` 3, `attn_cross_fd` 3 — 0 failed |
| migrated crates | `yolo` 105, `depth` 112, `npu` 47, `wm-display` 17, `capture` 4, `capability` 9, `mirror` 10, `backend-cpu` 21, `cli` 8 — 0 failed |
| vit consumers | `fastvlm` 31, `moondream` 34, `qwenvl` 38, `qwen-asr` 3 lib tests — 0 failed |

Known and carried into phase 1, none of them blocking: `ImagingKernelIds`
re-implements `vision::ids::ConvKernelIds`'s name→index resolver (11 of 16 names
overlap; the fix is a free `vision::ids::need()` both call — a tidiness defect,
both are pure lookups over the same array); `imaging` depends transitively on
`events` → `forecast` for a 40-line PPM parser; `imaging::Ctx` submits eagerly
and returns owned buffers, so phase 1's `resize_bicubic` inside SAM 2's
pos-embed path must dispatch directly rather than through `Ctx`; and
`fastvlm/src/encoder.rs` still hand-rolls the per-span cross-attention backward
trio that `model::vit` now owns.

#### Phases 1–3b gate — what was actually run

**Scope of this gate is goldens → import → FORWARD parity, nothing else.**
Backward/gradcheck (`check_sam2`, `check_arcface`, `check_codeformer`,
`check_clip`) and the serving contract (`Provider`, residency adapter, a real
`run_batch`, D-Bus, `examples/`) are **deliberately deferred** and are not
claimed by any number below.

Tesla P40 (wgpu/Vulkan), `--release`, run as four targeted
`cargo test --release -p <crate>` invocations (never a workspace
`--tests --examples` build). Workspace `cargo build` (lib+bin): **zero rustc
warnings** — verified with
`cargo build --message-format=short 2>&1 | grep -E '^[^ ]+\.rs:[0-9]+:[0-9]+: warning:'`
returning nothing; the two remaining `cargo:warning` lines are the
`brain-vulkan`/`brain-npu` build-script notes about an absent glslc / OpenVINO
and are not code warnings.

| suite | tests | measured |
|---|---|---|
| `brain-sam2` | 11 lib + 2 parity, 0 failed (25.4 s) | **283/283 stage comparisons, worst cosine 0.9999999999.** hiera-tiny: import 471 source → 317 image-path + 153 video-only skipped; encoder 31 stages, worst 1.0000000000 (`trunk_feat3`); 5 prompt cases × 22 stages, worst 1.0000000000. hiera-large: import 903 → 749 + 153 skipped; encoder 32 stages, worst 1.0000000000 (`lateral_level2`); 5 × 22 stages, worst **0.9999999999** (`point2_negpos/final_attn_out`) |
| `brain-facenet` | 16 lib + 7 parity, 0 failed (7.9 s) | ArcFace: every tap **cosine 1.0000000** — `blob`/`stem`/20 stage internals/`layer1..4`, `block00..48` (49 residual adds, worst max_abs 4.768e-7), `bn2` 1.147e-6, `fc` 1.460e-6, `embedding` 9.418e-6, `embedding_normed` 7.898e-7. E2E on 4 photos: all 1.0000000; **cosine matrix max \|Δ\| 2.384e-7**. Alignment: `M` ≤1.478e-5, `grid` ≤4.768e-7, `warp_gs` ≤1.159e-2 — vs **cv2** deliberately loose (cos 0.9999972, max_abs 0.5001: cv2's 5-bit fixed-point weights vs fp32 bilinear). SCRFD: 41 taps 1.0000000; decode+NMS on 4 photos, box ≤6.104e-5 px, kps ≤1.221e-4 px, scores exact to 6 digits (0.821795 / 0.833756 / 0.891094 / 0.817362) |
| `brain-vqgan` | 9 lib + 9 parity, 0 failed (12.1 s) | **cosine 1.000000000 everywhere; 0 code-index mismatches everywhere.** `stages_128` (25 enc + 25 gen + 11 sub-taps): codeformer worst `enc.21` 1−cos **1.84e-11** / relL2 6.262e-6; vqgan_code1024 worst `gen.19` 1−cos **1.08e-11** / relL2 4.619e-6; 0/16 indices each. `e2e_512_face`: codeformer worst `vq.min_dist` 1−cos **1.62e-10** / relL2 1.799e-5, `output` max\|Δ\| 1.955e-5; vqgan_code1024 1−cos **1.63e-10** / relL2 1.804e-5, `output` 1.687e-5; 0/256 each. `e2e_512_synth`: 1−cos **4.37e-11**, 0/256. Quantizer unit: 1−cos 4.25e-14 / 4.60e-14, 0 mismatches. `generate(z_q)` reproduces `decode` **bit-identically** (max\|Δ\| 0.0), and the pooled (`taps=false`) production graph is **bit-identical** to the tapped one on both checkpoints |
| `brain-clip` | 8 lib + 2 parity, 0 failed (21.8 s) | **148 stage checks, 0 failed.** CLIP-L text 32 stages, all cosine 1.00000000, worst relL2 `layer11_out` 1.541e-6 (max_abs 1.648e-3), `last_hidden_state` 1.723e-6, `pooled` 2.009e-6. OpenCLIP-bigG text 53 stages, all 1.00000000, `layer31_out` 3.088e-6, `last_hidden_state` 3.442e-6, `text_embeds` 1.984e-6. SDXL conditioning: `prompt_embeds` 1.638e-6, `pooled_prompt_embeds` 1.984e-6. EVA-CLIP-L/336 image 49 stages: worst **cosine 0.99999999** at `block22_out`/`block23_out`/`norm_out` (relL2 1.029e-4 / 1.027e-4 / 1.129e-4), collapsing to `cls_embed_l2norm` 1.00000000 / relL2 7.901e-6 through the head. B=2 batched replay 12 stages, both rows identical to the B=1 golden |
| shared homes touched | `brain-vision` 8+13, `brain-model` 60+22, `brain-onnx` 9+3, `brain-backend-cpu` 20+1, `brain-vae` 2+3 — 0 failed | the `vae::blocks` hoist did **not** regress its first user: **FLUX.2 VAE encode and decode parity still cosine 1.000000** on the P40, `pack` max_abs 0.0 |

**Not measured by this gate, and therefore not claimed:** the CPU-JIT backend
legs (`BRAIN_DEVICE=cpu`) were run by the porting agents but not re-run here; the
SAM 2 no-object path (`NO_OBJ_SCORE = -1024` / `no_obj_ptr`) has no golden and
has never executed; the antialiased 1024→256 mask-prompt downsample is not
implemented; and every fixture in this gate came from the local mirrors, so
`scripts/data/fetch-testdata.sh` provisioning was verified for the sam2 checkpoints
and the two antelopev2 `.onnx` only — the ~1 GB of stage goldens per model are
regenerated from the `tools/goldens/*_dump_reference.py` commands recorded in that
script, not mirrored.

#### Phase 3c/4 gate — what was actually run

**Scope of this gate is goldens → import → FORWARD parity, nothing else.**
Backward/gradcheck (`check_codeformer`, `check_flux1`, `check_t5`) and the
serving contract (`Provider`, residency adapter, `run_batch`, D-Bus,
`examples/`) are **deliberately deferred** and are not claimed by any number
below. At the time of this gate none of `crates/{t5,flux1,restore}` had a CLI
subcommand, a capability manifest or a D-Bus surface — **`crates/restore` has
since gained all three** (`restore::caps`, `resident_restore.rs`,
`examples/restore/`, 2026-08-04); `t5`/`flux1` still have none. `crates/flux1`
has **no sampler loop and no VAE glue**, so "FLUX.1 works" is *not* a claim this
gate supports — what it supports is that one transformer evaluation reproduces
diffusers.

One Tesla P40 (`BRAIN_DEVICE=gpu0`, wgpu/Vulkan), `--release`, run as targeted
`cargo test --release -p <crate>` invocations (never a workspace
`--tests --examples` build).

| suite | tests | measured |
|---|---|---|
| `brain-t5` | 4 lib + 1 parity + 2 smoke + 1 tiny_ref, 0 failed (parity 48.1 s) | T5-XXL from the Kontext `text_encoder_2`: import **219 source tensors → 171 parameters**, `relative_position_bucket` 16384 entries **exact**. **42 stages at B=2/T=128, 0 failed, worst cosine 0.9999999992** (`block23_out`, max_abs 3.271e0, relL2 4.048e-5 — that is 1.4e-5 of the stage's own peak \|x\| = 2.36e5, i.e. scale, not drift). `position_bias` and `embed` at max_abs **0.000e0**; `last_hidden_state` 0.9999999996 / 9.284e-4 / 2.831e-5; `last_hidden_state[content]` **1.0000000000** / 2.694e-5. Plus a **checkpoint-free** `tiny_ref` gate at deliberately distinct dims (`heads=2, d_kv=64, d_model=64`, so `heads ≠ d_kv` and `heads·d_kv ≠ d_model`) — 17 stages, cosine 1.0000000000 |
| `brain-flux1` | 8 lib + 2 dit_parity + 3 model_smoke, 0 failed (reduced-depth 49.4 s, full-depth 203.1 s) | **Two gates, because the fp32 model does not fit one card.** *Reduced depth (2 double + 2 single), fp32, real Kontext-dev weights* — import 1160 tensors, 1048 dropped as out-of-depth; t2i `out` cosine **0.999999999996** (1−cos 4.187e-12) max_abs 3e-5, edit `out` **0.999999999993** (6.633e-12) max_abs 8e-5; worst stage anywhere `sg1_txt` **0.999999999985** (1.511e-11). *Full depth (19+38), **int8*** — t2i `out` **0.998544477641** max_abs 0.40522, worst `db18_img` 0.994796217798; edit `out` **0.999137355865** max_abs 0.61119, worst `sg36_txt` 0.989315144936; `pre_final` 0.998047889316 / 0.997514888243. **The full-depth fp32 number is NOT measured and is not claimed** — 47.6 GiB of weights against a 24 GiB card |
| `brain-restore` | 4 parity, 0 failed (35.6 s) | CodeFormer `codeformer.pth`: **515 source tensors → 533 runtime tensors**, two-way covered. **0/256 predicted code indices differ from the reference** on both cases (face 201 distinct, synth 125 distinct), and `quant_feat` is **bit-exact** (max\|Δ\| 0.000e0) at every `w`. Encoder+transformer: face worst `logits_norm` 1−cos **2.99e-12** / relL2 2.421e-6; synth worst `ft.08` **3.79e-12** / 2.726e-6. Generator+CFT, worst 1−cos across `w = 0, .25, .5, .75, 1` — face **4.84e-12 / 5.45e-11 / 5.62e-11 / 5.73e-11 / 6.07e-11**, synth up to **7.38e-11** (relL2 1.207e-5), every stage cosine 1.000000000. The dial does something monotone and matches the torch-computed drift in `manifest.json`: face max\|out(w)−out(0)\| = **0 / 0.9444 / 1.4420 / 1.5193 / 1.6321**, synth **0 / 2.1283 / 2.8225 / 3.4448 / 3.7374**. The pooled (`taps=false`) production graph is **bit-identical** to the tapped one |
| shared homes touched | `brain-flux2` 26 (+4 ignored measurements), `brain-vqgan` 11+9, `brain-vae` 2+1, `brain-model` all green — 0 failed | the `model::block::gemm_variant` hoist did **not** regress its first users: flux2 `dit_parity`, `e2e_parity`, `int8_parity`, `host_forward_parity`, `import_real`, `model_grad` and the **bit-identical** `batch_parity` all still pass; vqgan 0 code-index mismatches unchanged; `vae` FLUX.2 encode/decode parity unchanged |

**Not measured by this gate, and therefore not claimed:**

* **FLUX.1 full-depth fp32.** Only reduced-depth fp32 and full-depth int8 ran.
* **T5-XXL at T=512**, the length the FLUX contract actually uses. Parity is
  gated at T=128/B=2 only. T=512 cannot run fp32 on one 24 GiB card, and its
  score-slab dispatch (33.5 M threads = 524 288 workgroups) exceeds the 65 535
  per-dimension limit, so it takes the kernels' **2D-grid path — which T=128
  never exercises**. That is the first thing to check when T=512 is attempted.
* **CodeFormer `adain=True`**, which is what `inference_codeformer.py` actually
  runs; what is gated here is the `adain=False` graph. It needs unbiased
  variance with eps *inside* the sqrt (no brain reduction does this today) plus
  a golden re-dump. Also unimplemented: face detection/alignment, batch > 1,
  and sizes ≠ 512².
* **The CPU-JIT leg** (`BRAIN_DEVICE=cpu`) for all three. `backend-cpu`'s
  reduction order depends on rayon splitting, so its low digits are not a
  fingerprint and must not be quoted as one.
* **`crates/vae`'s Z-Image encode/decode parity**, which **skipped** — there is
  no Z-Image VAE checkpoint on this box. Its FLUX.2 leg did run and is green.
* Anything about **speed**. No profile was taken and no speed claim is made.

**Phase 4 is the long pole and it is also disk- and bandwidth-bound** (34 GB at
~1.5 MB/s ≈ 6 h). Phases 0–3 need no Kontext weights, so they proceed in
parallel with the download.

#### Phase 4b gate — what was actually run

**Scope: goldens → import → FORWARD parity, plus the schedulers. Nothing
else.** `check_unet` and the **entire serving contract** (`Provider`, residency
adapter, `run_batch`, D-Bus, `examples/`) are deferred and are claimed by no
number here — unlike phases 1/2/3, `crates/unet` has **no** capability manifest,
**no** `resident_unet.rs` and **no** example. There is also no sampler loop and
no VAE/text-encoder glue, so "SDXL works" is *not* supported; what is supported
is that one UNet evaluation and four schedulers reproduce diffusers. Full
ledger: `docs/models/unet/status.md`.

One Tesla P40 (`BRAIN_DEVICE=gpu1`, wgpu/Vulkan), `--release`, targeted
`cargo test --release -p <crate>` (never a workspace `--tests --examples`).

| suite | tests | measured |
|---|---|---|
| `brain-unet` | 10 lib + 3 smoke + 1 parity (1 ignored), 0 failed (parity 40.7 s) | SDXL `unet/`: import **1680 source tensors → 1610 brain tensors**, two-way covered (the delta is the 70 `BasicTransformerBlock`s' three host-side fusions/splits); **2158 steps** at a 32×32 latent. **165 comparisons, 0 failed** (162 device taps + 2 host + `out.sample`), **worst cosine 0.9999999999** (`up1.attn1.proj_out`, max_abs 3.728e-4, rel_l2 1.539e-5); `out.sample` cosine **1.0000000000** / max_abs 1.705e-5 / rel_l2 **3.258e-6**. Both a cosine gate (≥0.9999) **and** a `rel_l2` gate (1e-3) are asserted — cosine alone is scale-invariant, so a dropped `output_scale_factor` or a doubled attention scale passes it |
| `brain-diffusion` | 7 lib + 66 scheduler checks + 5 other, 0 failed | DDIM / Euler / Euler-ancestral / DPM-Solver++(2M) × {ε, v-pred} × {4, 20} steps: the `timesteps` vector **exact** (0.000e0), the `sigmas` table incl. the terminal entry, `init_noise_sigma`, `scale_model_input` at every step, and the **full** `step()` trajectory. **Worst max_rel 7.510e-6** (`ddim.epsilon.20.traj`); the eight `init_noise_sigma` checks at ≤4.502e-7 |
| residency (`#[ignore]`d, not a parity gate) | 1 | **2 567 463 684 parameters = 10.27 GB fp32**; the production graph at a **128×128 latent** (SDXL's native size) is **2198 dispatches** at **4.06 s per forward** — the full-resolution fp32 UNet forward runs on one P40. The ~14 GB in the VRAM table below is UNet + both text encoders; the UNet alone is 10.27 GB |

Defects this phase found and fixed, each with the number that proved it, are in
`docs/models/unet/status.md`. Two are worth naming here because they are not
UNet-specific: `diffusion::Sigmas::init_noise_sigma` had its two
`timestep_spacing` branches **inverted**, so every SDXL sampling run would have
started from 11.028 instead of 11.074 (SDXL ships `leading`) and nothing gated
it; and `vae::blocks::Builder` uploaded weights with `storage_init`
(mapped-at-creation, no staging drain), which **OOM'd a P40 with ~20 GB free**
at 10.27 GB and is now the `storage()`+`write()`+`poll_wait()` pattern
`paramstore` and `zimage` already used.

**Not measured by this gate, and therefore not claimed:** the 128×128 latent
(the golden is dumped at 32×32; the graph is resolution-independent, so what
32×32 gates is the composition), batch > 1 and therefore CFG-as-one-forward,
INT8, the VAE/text encoders, anything about speed beyond the single 4.06 s
wall-clock, and `BRAIN_DEVICE=cpu` numbers (a run-to-run determinism finding on
that backend is contested — see the ledger).

#### Phase 4c/5 gate — what was actually run (integration pass)

**Scope: the control seam + ControlNet residual parity, PuLID forward parity,
and the first two backwards in the imaging stack (`check_t5`,
`check_codeformer`). Nothing else.** No backward exists in `crates/controlnet`
or `crates/pulid` at all; `check_flux1`, `check_unet`, `check_controlnet`,
`check_pulid` and `check_instantid` are **absent**, not failing. Neither new
crate has any part of the serving contract. Ledgers:
`docs/models/{controlnet,pulid}/status.md`.

Two Tesla P40s and `BRAIN_DEVICE=cpu`, `--release`, targeted
`cargo test --release -p <crate>` (never a workspace `--tests --examples`).
**All weight env vars were set for this run** (`BRAIN_CONTROLNET`,
`BRAIN_PULID`, `BRAIN_FLUX1_TRANSFORMER`, `BRAIN_T5_XXL`, `BRAIN_SDXL`,
`BRAIN_VQGAN_WEIGHTS`, `BRAIN_RESTORE_WEIGHTS`, `BRAIN_FLUX1_FULL=1`) and the
logs were grepped for `SKIP` — **nothing self-skipped** except where stated.

**Gradient checks — `crates/gradcheck/tests/imaging_models.rs`, 10 checks,
`(atol, rtol) = (4e-3, 8e-2)`, seed 1. `10 passed / 0 failed` on BOTH a P40 and
`BRAIN_DEVICE=cpu`.** Running both is a correctness requirement, not
thoroughness: a `var<workgroup>` reduction with no barrier-free sibling returns
**all zeros** on `backend-cpu` with no error. Both new trainers route their
norms through the shared selection seams (`block::rms_variant`,
`block::LayerNormIds` + `layernorm_fwd`/`ln_stats_fwd`/`layernorm_dx_bwd`), and
every backward reduction they dispatch (`layernorm_dgamma`/`dbeta`,
`bias_grad`, `attn_bwd_dbias`, `emb_bwd`) is a barrier-free gather.

| check | tensors | max abs err (P40 / cpu) |
|---|---|---|
| `check_sam2` (decoder only) | 128 | 2.19e-2 / 2.28e-2 |
| `check_arcface` | 53 | 3.83e-1 / 3.85e-1 |
| `check_vqgan` | 87 | 1.69e-4 / 1.66e-4 |
| `check_clip` (CLIP-L text only) | 28 | 1.14e-2 / 1.13e-2 |
| **`check_t5`** | 17 | 3.74e-3 / 1.87e-3 |
| **`check_t5_one_block`** | 10 | 4.34e-4 / 3.98e-4 |
| **`check_t5_tiled`** | 10 | 3.84e-1 / 2.93e-1 |
| **`check_t5_rel_bias_elementwise`** | 24 entries | 1.77e-4 / 2.55e-4 |
| **`check_codeformer`** | 34 | 5.51e-4 / 3.70e-4 |
| **`check_codeformer_one_layer`** | 20 | 2.84e-4 / 2.53e-4 |

*Read the absolute column, not a relative one.* The gate is
`|a − n| ≤ atol + rtol·max(|a|,|n|)`; `check_arcface` and `check_sam2` report
`max_rel` up to 0.6 / 0.95 on directional derivatives that are themselves ~0,
which is what the `atol` floor exists for. Quoting `max_rel` alone here would be
misleading in the opposite direction from usual.

**No regression in the pre-existing entries.** `cargo test --release -p
brain-gradcheck --lib` — which carries `check_gpt`, `check_qwen`(+`_lora`,
`qwen2`, `_mrope`, `vlm_splice`), `check_moe`, `check_glm`(+`_mtp`),
`check_pid`, `check_seq2seq`, `check_autoencoder`, `check_lfm`, `check_flux2`
alongside the imaging modules — is **31 passed / 0 failed on `gpu0`, on `gpu1`,
on `cpu`, and with `BRAIN_DEVICE` unset**.

| suite | device | result |
|---|---|---|
| `brain-gradcheck` `imaging_models` | P40 / cpu | **10 / 10**, 0 failed (11.5 s / 16.1 s) |
| `brain-gradcheck` `--lib` | gpu0, gpu1, cpu, default | **31 / 31**, 0 failed |
| `brain-controlnet` | P40 | 19 lib + **2 parity** + 6 smoke, 0 failed (parity 32.8 s — a real run, not a skip) |
| `brain-unet` (regression) | P40 | 9 lib + **1 parity** (1 `#[ignore]`d) + 3 smoke, 0 failed (parity 40.1 s) |
| `brain-restore` | P40 | 20 lib + **4 parity**, 0 failed (parity 20.5 s) |
| `brain-pulid` | P40 | 7 lib + **3 parity**, 0 failed (parity 52.1 s) |
| `brain-flux1` (regression) | P40 | 8 lib + **2 dit_parity** + 3 smoke, 0 failed — including the full-depth int8 leg with `BRAIN_FLUX1_FULL=1` (162.7 s) |
| `brain-t5` | P40 | 5 lib **passed**; `t5_xxl_encoder_stage_parity` **FAILED — `wgpu error: Out of Memory`**, see below |

**The one failure, and what it is not.** `brain-t5`'s XXL parity OOM'd on a P40
that had **4.5 GB of VRAM held by processes outside this session** (confirmed
with `nvidia-smi`; the owning PIDs are not visible in this namespace). The
T5-XXL fp32 encoder is ~19 GB against a 24 GB card, so it needs a card that is
essentially idle. This is an **environment result, not a code result** — but it
means **T5-XXL forward parity was NOT re-measured in this pass** and the
42-stage / 0.9999999992 figure in the phase-3c/4 gate above is carried over
from that gate, not reconfirmed here.

**Warnings gate.** `cargo build` (lib+bin) **exit 0 with 0 rustc code warnings**
(the two `cargo:warning` lines remain the `brain-vulkan`/`brain-npu`
build-script notes about an absent glslc / OpenVINO). `cargo clippy --workspace
--all-targets` **exit 0** — checked by exit code, not by grepping the text —
with **192** individual warnings, **zero of them in any file this workstream
touched**, and a per-crate distribution byte-identical to the run taken before
any of these changes. The workstream's stated "190" baseline is a stale figure
against this counting method (`^warning:` lines minus the per-crate summaries);
what is *measured* is that this integration contributed **0**.
`grep -rnE '"/(data|home|tmp|opt|mnt|root)/' crates` is empty.

**Legs that did NOT run in this pass, and are therefore unverified here:** the
shared-home regression suites `brain-model`, `brain-flux2`, `brain-vqgan`,
`brain-vae` and `brain-gpu-core`, and the `BRAIN_DEVICE=cpu` forward legs for
`controlnet`/`pulid`/`unet`/`t5`/`restore`. They were queued and cut for time,
not skipped because they were expected to fail; the cross-backend *gradient*
legs (which are the correctness gate that `backend-cpu` uniquely catches) did
all run. `model::block::ln_variant` going from private to `pub` cannot change
behaviour, and the 41 gradcheck tests exercise `model::block` heavily on both
backends — but that is an argument, not a measurement.

**Not measured by this gate, and therefore not claimed:** any backward for
`controlnet`, `pulid`, `unet` or `flux1`; the composed UNet+ControlNet at full
SDXL dims on real weights (the residual gate is the *producer*, and the
placement of a residual inside the backbone is gated separately and only at toy
dims by `smoke.rs::a_down_residual_reaches_the_output_only_through_the_up_path`);
a FLUX ControlNet (the seam is asserted backbone-agnostic by inspection and by
`Layout::Tokens` existing, **not** by a running FLUX implementation); PuLID's
single-interval schedule value (at the gated 2+2 depth the site list is
`{double 0, single 0}` for any interval ≥ 2, so the forward cannot distinguish
4 from 3 or 5 — written into the test); anything about speed; and anything about
image quality or identity fidelity for either model.

### VRAM budget (24–48 GB target)

brain's **kernel arithmetic** is fp32 (no f16/bf16 datatypes — that is what keeps
it portable to old GPUs and WebGPU), but that is not the same as fp32 storage:
brain has a full **INT8 path** — per-channel symmetric weights packed 4-per-`u32`
(`model::int8::quantize_weight`), DP4A GEMMs (`matmul_i8`, `matmul_i8_dyn`,
`matmul_i8_gemv`, ~4× the fp32 rate on Pascal), dynamic per-token activation
scales (`max_abs_row` → `quant_pack`), and int8 paged KV. Norms, RoPE and
attention stay fp32.

So INT8 is the **primary** residency mechanism here, not a fallback, and there is
direct precedent for exactly this shape of model: `zimage::int8` quantizes a DiT
and `qwen::q8` takes the Qwen3-4B encoder from ~16 GB fp32 to ~4.8 GB so it fits
one P40.

| Model | fp32 weights | INT8 linears | Fits |
|---|---|---|---|
| FLUX.1 Kontext DiT (12 B) | ~47.6 GB | **~12 GB** | one P40 |
| T5-XXL encoder (4.8 B) | ~19 GB | ~5 GB | one P40, and evictable after conditioning |
| SDXL UNet + both text encoders (3.5 B) | ~14 GB | ~4 GB | one P40 **without** INT8 |
| ControlNet (SDXL) | ~5 GB | ~1.4 GB | alongside the UNet |
| SAM 2 / ArcFace / CodeFormer / BiRefNet | <1 GB each | — | all resident together |

Layered on top, in order of preference: INT8 weights → `crates/residency`
tiering (T5-XXL is needed once per generation, so it is evicted after
conditioning) → tensor sharding across the two P40s (`model::shard`/`plan`) only
if the first two are not enough.

---

SDXL is the kinder starting point: UNet 2.6 B + both text encoders ≈ 3.5 B, so
~14 GB at fp32 — one P40 with no INT8 at all. That makes phases 4b/4c the
practical place to get the identity pipeline working end to end first, with
FLUX.1 Kontext as the higher-quality path behind INT8 (~12 GB, also one card).

#### Phase 0–3b training gate — what was actually run

Backward + gradcheck landed after the forward gate. Measured on **both** a Tesla
P40 and `BRAIN_DEVICE=cpu` (both matter — see below), all four passing at the
workspace tolerance `(atol, rtol) = (4e-3, 8e-2)`:

| check | scope | result |
|---|---|---|
| `check_sam2` | **decoder only**, Hiera trunk + FPN neck `Role::Frozen` (the common finetune mode) | **128 tensors, 0 failures** across seeds 1/2/3/7/11 |
| `check_arcface` | tiny IResNet under the real additive-angular-margin CE; folded conv, train-mode BN, PReLU per-channel slope | pass |
| `check_vqgan` | VQ straight-through + codebook/commitment, and every encoder/generator param through `vae::blocks::grad` | pass |
| `check_clip` | CLIP-L **text tower only**; bigG and the EVA image tower still owe theirs | pass |

`eps = 5e-4`, not the workspace default `5e-3`: a ±1 direction over `numel`
entries is an L2 step of `eps·√numel`, which at `5e-3` moves these tensors
comparably to their init scale, deep into the nonlinear regime. Same reason
`yolo/tests/p3_gradcheck.rs` drops to `5e-4`.

**Running both backends is a correctness requirement, not thoroughness.** A
`var<workgroup>` + `workgroupBarrier()` reduction with no barrier-free sibling
returns **all zeros** on `backend-cpu` with no error — a trainable parameter
whose gradient is silently dead and whose loss curve still looks plausible. A
GPU-only gate passes that completely. `vision::PReLU::backward` selects on the
queried `DeviceCaps::workgroup_reductions` for exactly this reason.

**What finite differences cannot see.** They gate the backward against whatever
forward is emitted, so a *mis-weighted objective* is self-consistent and passes.
`beta` sits on the VQ **codebook** term in `vqgan_arch.py:55`, not on the
commitment term that file's own line-29 comment claims — pinned by reading the
reference, not by the gate.

Still owed: full-trunk SAM 2 training, the CodeFormer transformer + dial `w`,
and bigG/EVA backwards. *(The serving contract, listed here as owed for all
four, has since landed for `sam2`/`facenet`/`vqgan`/`restore`; `clip` still owes
it — it needs `data::clip_bpe` wired in and EVA image preprocessing.)*

## 5. Open decisions

1. ~~InstantID dropped in favour of PuLID-FLUX~~ — **resolved: build both.**
   Measuring the UNet family showed it needs no new kernels (§2), so InstantID
   stays and brain gains a UNet diffusion backbone + a generic ControlNet seam.
2. **Florence-2 over GroundingDINO**, **LaMa deferred** (§2).
3. SAM 2 **video** (memory attention + memory encoder) is out of scope for
   phase 1; the image path is what image editing needs. The config and code are
   downloaded, so it is additive later.
4. Which SD backbone is the *default* for the editing pipeline — SDXL (fits one
   P40, whole-ecosystem compatibility) or FLUX.1 Kontext (better edit fidelity,
   needs INT8 + sharding). Deferred until phase 5 can measure both.

---

## 7. Where the workstream actually stands

Written at the end of the build-out. The wins are recorded above with their
measured numbers; this section is the **honest ledger of what is not done**,
because that is what the next person needs.

### Gated and working

| Crate | Forward | Backward | Serving contract |
|---|---|---|---|
| `imaging` | n/a (substrate) | n/a | n/a |
| `sam2` | ✅ 283/283, worst cosine 0.9999999999 | ✅ decoder only, trunk frozen | ✅ |
| `facenet` | ✅ every tap cosine 1.0000000 | ✅ `check_arcface` | ✅ |
| `vqgan` | ✅ cosine 1.000000000, 0 code-index mismatches | ✅ VQ straight-through | ✅ |
| `restore` | ✅ 0/256 indices differ, `quant_feat` bit-exact | ✅ `check_codeformer` | ✅ |
| `clip` | ✅ 148 stage checks; served path cosine 1.0 vs HF | ✅ text tower | ✅ |
| `t5` | ✅ 42 stages, worst 0.9999999992 | ✅ `check_t5` | ❌ |
| `flux1` | ◑ reduced-depth fp32 + full-depth int8 | ❌ | ❌ |
| `unet` | ✅ 165 comparisons, worst 0.9999999999 | ❌ | ❌ |
| `controlnet` | ✅ 140 comparisons, both backends | ❌ | ❌ |
| `pulid` | ✅ ID pipeline + injection | ❌ | ❌ |
| `instantid` | ❌ shapes + import only | ❌ | ❌ |
| `imgpipe` | ✅ bit-exactness contract | n/a | ✅ discoverable |

### Not done — the real list

1. **`crates/instantid` has no forward.** Shapes, import validation and the
   reference ladder are in; the Resampler and the decoupled attention are not.
   It should **reuse `crates/pulid`'s `Emit`**, which already records this exact
   Perceiver block — writing a second one is the duplication this workstream
   spent its whole length avoiding.
2. **`flux1` has no sampler loop and no VAE glue.** One transformer evaluation
   reproduces diffusers; "FLUX.1 generates an image" is not supported by
   anything. PuLID and Kontext inherit that limit.
3. **`flux1` full-depth fp32 is unmeasured** (47.6 GiB vs a 24 GiB card), and
   **T5 at T=512** — the length FLUX actually uses — is untested and takes a
   2D-grid dispatch path T=128 never exercises.
4. **CodeFormer `adain=True`** — what `inference_codeformer.py` actually runs —
   is not implemented; the gated graph is `adain=False`.
5. **No backward** for `unet`, `controlnet`, `flux1`, `pulid`, `instantid`; no
   serving contract for those or `t5`.
6. **The depth INT8 scale delta is unmeasured.** The calibration preprocessing
   was fixed, but no checkpoint exists on the dev box to quantify the change.
7. **BiRefNet matting and Real-ESRGAN upscale are downloaded but not ported** —
   `imgpipe` has hooks, not stages.
8. **190 clippy warnings**, ratcheted by `make clippy`. 69 are doc-list
   indentation needing per-site judgment.
9. **NPU topologies** for the new models are not written; nothing here has run
   on an NPU, and this box has none.
10. **`wm-diamond` still hand-records its UNet** rather than using
    `vae::blocks::Builder`, so brain has two UNet graph recorders.

### The three findings worth carrying forward

* **A gate that never runs is worse than no gate.** Three separate ones were
  silently green: `sam2`'s parity self-skipped on every machine but this one,
  `flux2`'s host/device parity had dims the device could not bind, and
  `cargo clippy` aborted before linting most of the workspace — twice.
  `make clippy` now checks the exit code, not the output.
* **Cosine is scale-invariant.** A dropped scale factor scores 1.0. The `unet`
  ladder gates `rel_l2` as well for exactly that reason.
* **Finite differences gate the backward against whatever forward is emitted.**
  A mis-weighted objective is self-consistent and passes — which is why the VQ
  `beta` placement is pinned by reading the reference, not by `check_vqgan`.

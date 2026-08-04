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
and brain today has **no UNet diffusion model at all** (only DiT: `dit`,
`zimage`, `flux2`). The initial read was that this made InstantID a bad trade.

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
  (`codeformer.pth`, `vqgan_code1024.pth`). Backward/gradcheck, the CodeFormer
  transformer + fidelity dial, and the serving contract are follow-ups.

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

  The CLIP **BPE tokenizer is not implemented** — it belongs in `crates/data`
  next to the GPT-2 and Qwen BPEs, and the parity tests feed token ids from the
  goldens. Preprocessing (decode → bicubic 336 → normalize) is likewise absent;
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

| Entry | Covers | Loss |
|---|---|---|
| `check_sam2` | Hiera + FPN + prompt encoder + two-way mask decoder | focal + dice on masks, MSE on the IoU head |
| `check_arcface` | IResNet backbone + margin head | ArcFace additive-angular-margin CE |
| `check_flux1` | FLUX.1 MMDiT | flow-matching MSE (mirror `check_flux2`) |
| `check_pulid` | ID adapter layers only, backbone frozen | flow-matching MSE + ID cosine |
| `check_codeformer` | VQ encoder/decoder + transformer | CE on code indices + straight-through VQ + L1/perceptual |

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
| **1. SAM 2** ◑ **FORWARD DONE** | Hiera trunk → FPN neck → prompt encoder → two-way mask decoder; image path only (no video memory bank) | parity vs dumped goldens — **met** (283/283, see below); `check_sam2`, capability + D-Bus + example **still owed** |
| **2. Face recognition** ◑ **FORWARD DONE** | `crates/facenet`: SCRFD + IResNet-100 + alignment | cosine ≥0.999 vs insightface goldens — **met** (1.0000000 on every tap, see below); `check_arcface` **still owed** |
| **3. CodeFormer** ◑ **VQGAN FORWARD DONE** | `crates/vqgan` + restorer, identity-fidelity dial `w` | VQ encoder/decoder/codebook parity — **met** (see below); the CodeFormer **transformer + dial `w` are not implemented**; `check_codeformer` **still owed** |
| **3b. Text encoders** ◑ **FORWARD DONE** | `crates/clip`: CLIP-L + OpenCLIP-bigG + EVA-CLIP-L/336 behind one config-driven graph | cosine ≥0.9999 vs HF per encoder — **met** (see below); the **BPE tokenizer and image preprocessing are not implemented**; `check_clip` **still owed** |
| **4. FLUX.1** | `crates/flux1` reusing `dit`/`vae`/`diffusion`; T5-XXL encoder + phase-3b CLIP-L; Kontext edit path | forward cosine vs diffusers; `check_flux1` |
| **4b. UNet family** | `crates/unet` (SDXL `UNet2DConditionModel`) + discrete schedulers (DDIM/Euler-a/DPM++, ε and v-pred) in `crates/diffusion` | forward cosine vs diffusers SDXL; `check_unet` |
| **4c. ControlNet** | `crates/controlnet`: backbone-agnostic `ControlAdapter` over named injection points; SDXL impl first, FLUX impl second; depth conditioning fed by brain's own ZipDepth | residual parity vs diffusers ControlNet; `check_controlnet` |
| **5. Identity** | PuLID adapter on the phase-4 FLUX.1 backbone **and** InstantID (phase-4b/4c SDXL + IP-Adapter-FaceID), both fed by phase-2 ArcFace embeddings | ID cosine vs reference for each; `check_pulid`, `check_instantid`; measure the two against each other on shared fixtures |
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

**Phase 4 is the long pole and it is also disk- and bandwidth-bound** (34 GB at
~1.5 MB/s ≈ 6 h). Phases 0–3 need no Kontext weights, so they proceed in
parallel with the download.

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

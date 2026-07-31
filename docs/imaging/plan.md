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
| 2 | `convtr2d` + `_dw` + `_dx` | SAM 2 mask decoder, VQGAN/CodeFormer decoder | **DONE.** Built on the `convtr1d` shape template; same 12-word `Params` as `conv2d_gd` (square K, symmetric pad, bias-free) |
| 3 | `prelu` + `_bwd` | ArcFace IResNet | `leaky_relu` has a *fixed* slope; PReLU's is a learned per-channel parameter needing its own grad |
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
  (2.3–9.1× on a P40); a naive fused `layernorm2d` would likely be *slower*, and
  one-thread-per-row is the documented coalescing trap.

Every new kernel: `@workgroup_size(64)`, ≤8 storage buffers, fp32, no atomics/
subgroups/f16, then `make kernels-regen`.

### 3.2 Shared blocks (the reuse story)

New capability belongs in existing shared homes, not per-model copies:

- **`model::vit`** — extend with windowed-span attention and `q_pool` so Hiera,
  Swin (BiRefNet) and DaViT (Florence-2) all compose from it. It already has
  QK-norm, 2D-RoPE tables, span-chunked attention and a full backward.
- **`vision::blocks`** — add `ConvTranspose`, generic `MaxPool`, `LayerNorm2d`
  and the ConvNeXt `CXBlock` next to the existing spec-driven `Conv`/`SPPF`.
- **`crates/facenet`** *(new)* — ArcFace IResNet-100 + SCRFD detection + the
  5-point similarity-transform alignment. This mirrors `crates/speaker`
  (ECAPA-TDNN) exactly: an embedding model consumed by a generative one. The
  request already noted the pattern is familiar; make it structurally identical.
- **`crates/vqgan`** *(new, small)* — the VQ encoder/decoder/codebook shared by
  CodeFormer, reusing `crates/vae`'s conv/attention blocks and `vq_argmin`.
- **`crates/imaging`** *(new)* — the image substrate currently scattered across
  `depth` (viz), `yolo` (letterbox), `capture` (YUYV) and `zimage`: decode/encode,
  resize/crop/pad/letterbox, mask algebra (union, feather, dilate, composite),
  tiling for >1 MP work, and colour conversion. Consolidating this is a
  precondition for a *pipeline* rather than four models with private image code.

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
| **0. Foundations** | kernels 1–5 + `crates/imaging` + `vision::blocks` additions + `model::vit` windowed spans | `make kernels-regen`, `make test`, per-kernel parity vs a CPU reference |
| **1. SAM 2** | Hiera trunk → FPN neck → prompt encoder → two-way mask decoder; image path only (no video memory bank) | parity vs dumped goldens; `check_sam2`; capability + D-Bus + example |
| **2. Face recognition** | `crates/facenet`: SCRFD + IResNet-100 + alignment | cosine ≥0.999 vs insightface goldens; `check_arcface` |
| **3. CodeFormer** | `crates/vqgan` + restorer, identity-fidelity dial `w` | parity goldens; `check_codeformer` |
| **3b. Text encoders** | `crates/clip`: CLIP-L + OpenCLIP-bigG + EVA-CLIP-L/336 behind one config-driven graph | cosine ≥0.9999 vs HF per encoder; `check_clip` |
| **4. FLUX.1** | `crates/flux1` reusing `dit`/`vae`/`diffusion`; T5-XXL encoder + phase-3b CLIP-L; Kontext edit path | forward cosine vs diffusers; `check_flux1` |
| **4b. UNet family** | `crates/unet` (SDXL `UNet2DConditionModel`) + discrete schedulers (DDIM/Euler-a/DPM++, ε and v-pred) in `crates/diffusion` | forward cosine vs diffusers SDXL; `check_unet` |
| **4c. ControlNet** | `crates/controlnet`: backbone-agnostic `ControlAdapter` over named injection points; SDXL impl first, FLUX impl second; depth conditioning fed by brain's own ZipDepth | residual parity vs diffusers ControlNet; `check_controlnet` |
| **5. Identity** | PuLID adapter on the phase-4 FLUX.1 backbone **and** InstantID (phase-4b/4c SDXL + IP-Adapter-FaceID), both fed by phase-2 ArcFace embeddings | ID cosine vs reference for each; `check_pulid`, `check_instantid`; measure the two against each other on shared fixtures |
| **6. Pipeline** | `imaging.pipeline` action + matting (BiRefNet) + Florence-2 text→box + Real-ESRGAN tail | end-to-end "change only X" on a fixture set |

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

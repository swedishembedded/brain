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

### Scope warning: InstantID is a much worse fit than PuLID

The request said "PuLID **or** InstantID". They are not interchangeable here:

- **InstantID is SDXL-based** — it needs a `UNet2DConditionModel` *plus* a
  ControlNet. brain has **no UNet diffusion model at all** (only DiT: `dit`,
  `zimage`, `flux2`). That is an entire new architecture family — ResBlock/
  attention U-Net, SDXL's dual text encoders, plus a ControlNet clone of the
  down-blocks — before any identity work starts.
- **PuLID-FLUX rides the FLUX.1 backbone we need anyway** for Kontext, and its
  ID-injection mechanism (extra cross-attention layers keyed by an ArcFace +
  EVA-CLIP embedding) is the *same* IP-Adapter-FaceID idea the request points at.

**Recommendation: implement PuLID-FLUX; do not port SDXL.** If InstantID's exact
weights are required later, the mechanism can be re-hosted on FLUX.1 rather than
bringing in SDXL. This is flagged as a decision, not taken unilaterally — it
drops one named model from the request in favour of the equivalent capability.

### Two more substitutions worth making

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
| 1 | `maxpool2d` + `_dx` | Hiera `q_pool`, SCRFD | **Generalize `maxpool5`** (add `stride`); do not add a copy — it is already parameterised on K and pad |
| 2 | `convtr2d` + `_dw` + `_dx` | SAM 2 mask decoder, VQGAN/CodeFormer decoder | `convtr1d` exists as the shape template |
| 3 | `prelu` + `_bwd` | ArcFace IResNet | `leaky_relu` has a *fixed* slope; PReLU's is a learned per-channel parameter needing its own grad |
| 4 | `grid_sample` + `_bwd` | face alignment (similarity warp), ROI align | bilinear gather |
| 5 | `resize_bicubic` + `_dx` | SAM 2 `pos_embed` interpolation | joins `resize_bilinear`/`resize_nearest` |

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
| **4. FLUX.1** | `crates/flux1` reusing `dit`/`vae`/`diffusion`; T5-XXL + CLIP-L encoders; Kontext edit path | forward cosine vs diffusers; `check_flux1` |
| **5. PuLID** | ID adapter on the phase-4 backbone, fed by phase-2 embeddings | ID cosine vs reference; `check_pulid` |
| **6. Pipeline** | `imaging.pipeline` action + matting (BiRefNet) + Florence-2 text→box + Real-ESRGAN tail | end-to-end "change only X" on a fixture set |

**Phase 4 is the long pole and it is also disk- and bandwidth-bound** (34 GB at
~1.5 MB/s ≈ 6 h). Phases 0–3 need no Kontext weights, so they proceed in
parallel with the download.

### VRAM budget (24–48 GB target)

FLUX.1 at bf16 is 23.8 GB of weights — brain is **fp32-only**, so a naive
resident copy is ~47.6 GB and does *not* fit one P40. Three existing mechanisms
apply, in order: INT8 weights (`model::int8`, `qwen::q8` pattern) → ~12 GB;
tensor/expert sharding across the two P40s (`model::shard`/`plan`); and
`crates/residency` tiering so the text encoder is evicted after conditioning is
computed (T5-XXL is only needed once per generation). SAM 2, ArcFace, CodeFormer
and BiRefNet are all <1 GB and can stay resident together.

---

## 5. Open decisions

1. **InstantID dropped in favour of PuLID-FLUX** (§2) — needs sign-off, since it
   removes a named model from the request.
2. **Florence-2 over GroundingDINO**, **LaMa deferred** (§2).
3. SAM 2 **video** (memory attention + memory encoder) is out of scope for
   phase 1; the image path is what image editing needs. The config and code are
   downloaded, so it is additive later.

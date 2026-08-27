# AGENTS.md - brain (edge-AI model training framework)

Routing guide for this repo. **brain** trains and evaluates **neural networks
from scratch on the CPU, GPU and NPU**, **pure Rust + raw WGSL kernels**, with
validated parity between all supported accelerator backends.

Brain provides both training and inference, with quantization, model residency,
and concurrent serving on top of one hand-written kernel engine.

It is a self-contained Cargo **workspace** of ~60 crates under `crates/` - no
Python in the build/test path; backprop correctness is gated by an in-repo
finite-difference gradient checker (`crates/gradcheck`), not a PyTorch oracle.

The engine is **architecture-agnostic**: the WGSL kernels (`crates/kernels`, see
[`docs/reference/kernels.md`](docs/reference/kernels.md) for the generated catalogue)
are reusable building blocks, not a fixed model. New architectures should be
composed from them, keeping the gradient-check discipline. If a new architecture
requires new kernels, they should be created only after checking whether an
equivalent kernel already exists, and the newly added kernel must be a proper
fast and scalable kernel - not a naive one.

---

## Models (today)

### Language / decoder LMs

1. **GPT decoder** (`crates/gpt2`) - dense nanogpt-parity baseline: token+learned
   positional embeddings, pre-LN, causal MHA, GELU MLP, untied `lm_head`, masked
   CE. `brain gpt2 {train,eval,infer}`.
2. **Qwen3 decoder** (`crates/qwen3`) - dense GQA + QK-norm + RoPE + SwiGLU;
   HF **and GGUF** import behind ONE load path (`import::Naming` sniffs the
   convention from the file's own tensor names, so `import::source` /
   `shard_source` serve both and a caller passes a path, not a format; the
   llama.cpp name map in `gguf_import.rs` is transcribed from llama.cpp's own
   `tensor_mapping.py`/`constants.py` at a named revision and gated
   **bit-for-bit** against the safetensors route, because the mistake worth
   catching - a swapped `k`/`v` - is shape-compatible on every GQA layer).
   LoRA finetune, INT8 (`q8.rs`), tensor/expert sharding, tool-call
   eval, and the **concurrent paged-KV serving engine** (`serve.rs`, see below).
   `brain qwen3 {import,infer,serve,export,precompile,train,finetune,toolcall}`,
   plus the generic `brain import` for a `qwen3` GGUF.
2b. **Qwen3.5-35B-A3B hybrid decoder** (`crates/qwen35moe`) - 40 layers, 3:1
   **Gated DeltaNet** (chunked linear-attention, `model::gdn`) : **GQA**
   (`full_attention_interval=4`), a 256-expert top-8 sigmoid-gated-shared-expert
   sparse MoE on every layer, a sigmoid attention-output gate + partial RoPE +
   M-RoPE on the GQA layers, and a spliced vision front-end reusing
   `crates/qwen3vl`'s ViT + PatchMerger as-is (`qwen35moe::vl::Qwen35Vl`, no
   DeepStack - this model has none). GGUF streaming import (`import_gguf`,
   never a whole-model fp32 disk intermediate - the real checkpoint is ~140 GB
   at fp32), INT8 (`q8.rs`), rank-8 LoRA on the 9 targetable GDN/GQA
   projections (never the MoE experts), and cross-GPU pipeline sharding
   (`model::shard::Shardable`) for real weights that exceed one card.
   Gradient-checked (`gradcheck::check_qwen35moe`, `check_qwen35moe_lora`).
   `brain qwen35moe {infer,export}`; the GGUF
   conversion runs through the generic `brain import` (see below).
   See `.agents/roadmap/qwen35moe.md`.
2c. **Qwen3.8-27B dense hybrid decoder** (`crates/qwen35`, upstream
   `Qwen/Qwen3.8-27B-FP8`) - the dense sibling of qwen35moe above: identical
   3:1 Gated DeltaNet : GQA mixer split and M-RoPE, but a plain dense SwiGLU
   MLP on every layer (no MoE), plus a single-layer multi-token-prediction
   (MTP) head sharing the token embedding/LM head, and the same spliced
   Qwen3-VL vision front-end (`qwen35::vl::Qwen35Vl`). Weights ship as
   DeepSeek-V3-style blockwise FP8, dequantized host-side at import (no GGUF
   importer for this arch). Rank/alpha LoRA on all 12 targetable GDN/GQA/MLP
   projections, full finetune, incremental single-sequence decode, paged
   HTTP/D-Bus/batched serving (`caps.rs`/`serve.rs`), and cross-GPU pipeline
   sharding (`model::shard::Shardable`). Gradient-checked
   (`gradcheck::check_qwen35`, `check_qwen35_lora`, `check_qwen35_mtp`).
   `brain qwen35 infer`. See `.agents/roadmap/qwen35.md`.
3. **Sparse MoE Transformer** (`crates/toymoe`) - RMSNorm/RoPE, top-k experts; with
   **federated/sharded** expert training (`crates/federated`).
4. **GLM-5.2 decoder** (`crates/glmdsa`) - `glm_moe_dsa`: **MLA** (low-rank q/kv with
   a decoupled nope/rope head split + interleaved RoPE), a **sigmoid `noaux_tc`
   MoE** (per-expert selection bias, shared expert, `first_k_dense_replace`
   dense→MoE schedule), untied `lm_head`, plus the DSA indexer and MTP.
   Gradient-checked (`gradcheck::check_glm`, `check_glm_mtp`).
   `brain glmdsa {train,finetune,infer,eval,import,export}`.
5. **PID event/effect Transformer** (`crates/toypid`) - LayerNorm, learned positions,
   biased linears; backs the WebGPU browser demo (`crates/web`).
6. **Seq2seq** (`crates/toyseq2seq`) - encoder-decoder Transformer (bidirectional
   encoder + causal/cross-attention decoder), gradient-checked.
6b. **LFM2.5-Encoder** (`crates/lfm2`) - LiquidAI's bidirectional hybrid
   short-conv/attention encoder (GQA + QK-norm + RoPE, gated depthwise conv,
   tied MLM head, 8k context); imported 1:1, parity-gated per stage; chunked
   long-context inference. `brain lfm2 {import,fill-mask,embed}`.
7. **Bottleneck autoencoder** (`crates/toyautoencoder`) - sequence → single
   compressed representation → MLP reconstruction, MSE head; gradient-checked.

### Vision & 3D

8. **YOLOv8-style detector** (`crates/yolov8`) - from-scratch anchor-free detector:
   CSP backbone → PAN-FPN neck → decoupled DFL head, assigner + BCE/CIoU/DFL loss,
   NMS box decode. Byte-compatible with canonical `yolov8n` for weight import.
   `brain yolov8 {train,eval,detect,fine-tune}`; Intel-NPU path via `brain npu …`.
9. **ZipDepth monocular depth** (`crates/zipdepth`) - the 6.1M pure-conv depth net
   (QARep/RepVGG blocks, SE/strip/global-context attention, convex upsampling),
   exact vs the reference PyTorch on the released checkpoints. Realtime demo
   (`brain zipdepth --image|--camera`, SDL views incl. autostereograms) on
   CPU/GPU/Vulkan and the Intel NPU (ONNX/OpenVINO, cosine 0.99998).
   *(Depth Anything 3 was planned as a second arch behind the same contract and
   is **dropped** - see `crates/zipdepth/src/lib.rs`.)*
9b. **Face recognition** (`crates/scrfd` + `crates/arcface`) - the insightface
    *antelopev2* stack, split into its two architectures:
    **SCRFD-10GF** detector (ResNet-D backbone + PAFPN + 3-stride anchor head) →
    **5-point similarity alignment** (Umeyama solve on the host, applied with the
    `grid_sample` kernel) → the **ArcFace IResNet-100** 512-d embedding (PReLU
    with learned per-channel slopes; BatchNorm folded into the convs by the
    release). Imported from **ONNX** - the first import in the repo to read the
    protobuf, via the new `onnx::read` - with two-way coverage (462 / 119
    tensors). Forward-parity-gated per stage at cosine 1.000000.
    See `.agents/roadmap/scrfd.md`. **Serving contract met**, as TWO models:
    `scrfd::caps` (`detect`, `BRAIN_SCRFD_DIR`) and `arcface::caps` (`embed`,
    `BRAIN_ARCFACE_DIR`), with `crates/cli/src/resident_{scrfd,arcface}.rs`,
    D-Bus `Run` and `examples/vision/`. The detector is self-sufficient; the
    embedder depends on it (its default `align=true` detects first), so with
    both resident SCRFD is loaded twice - accepted, it is 17 MB. *(`run_batch`
    is the serial default and says why: both released graphs are built for
    `Shape::new(1,3,side,side)`. Backward/gradcheck are listed in that ledger.)*
9c. **SAM 2.1 promptable segmentation** (`crates/sam2`) - the **image path** of
    Meta's SAM 2.1: Hiera trunk (windowed attention with a per-block window
    schedule + `q_pool` stride stages) → FPN neck → prompt encoder (points, boxes
    and a mask prompt, over a random-Fourier positional encoding) → the two-way
    mask decoder (4 mask tokens + IoU head + object-score token). Both released
    variants (`hiera_tiny`, `hiera_large`) imported with two-way coverage and
    **forward-parity-gated per stage at cosine ≥0.9999999999** over 283
    comparisons. Composes `model::vit`'s windowed spans and `vision::blocks`;
    adds no kernel. **Serving contract met**: `sam2::caps` (`segment`),
    `crates/cli/src/resident_sam2.rs` (`BRAIN_SAM2_WEIGHTS`), D-Bus `Run`,
    `examples/vision/`; `run_batch` groups a batch **by image**, so N prompts on
    one frame cost ONE Hiera trunk pass and N decoder passes. *(Forward only: the
    video memory bank and backward/gradcheck are deferred - see the Limits
    section of `docs/models/sam2.md`.)*
10. **WorldMirror-2 multi-view 3D reconstruction** (`crates/worldmirror2`) - the
    HY-World 2.0 1.26B feed-forward model: per-frame DINOv2 ViT-L/14 encoding,
    24 alternating frame/global attention levels (QK-norm + normalized 2D RoPE),
    DPT heads (depth/points/normals/gaussians) + iterative camera head; photos →
    a navigable 3DGS scene. Imported exactly, parity-gated per stage.
    `brain worldmirror2 {import,infer,demo,export-npu}`.
11. **3D Gaussian Splatting** (`crates/splat`) - from-scratch tiled 3DGS
    rasterizer (atomic-free/barrier-free WGSL: generic scan + radix sort →
    per-tile compositing) with forward AND backward (autograd-verified), Inria
    PLY IO, interactive WASD+mouse viewer, and `splat fit` scene optimization.
    `brain splat {info,render,view,fit}`.

### Image generation (diffusion)

12. **Z-Image** (`crates/s3dit`) - Tongyi text-to-image: the **S³-DiT**
    single-stream diffusion transformer over Qwen3-4B caption features + VAE
    latents. Forward/inference first; LoRA, INT8, sharding, and hand-written
    backward (`grad.rs`/`devgrad.rs`/`modelgrad.rs`) alongside. Assembled from
    the shared `dit` / `diffusion` / `vae` / `qwen3` crates.

12b. **FLUX.2 Klein** (`crates/flux2`) - BFL's 4B/9B MMDiT text-to-image +
    image-editing family: double-stream img/txt blocks (joint attention) +
    single-stream parallel blocks, **3 global modulation linears** folded into
    LayerNorm affine params, 4-axis interleaved RoPE (theta 2000), Qwen3
    multi-layer masked-pad text conditioning ([9,18,27] concat), FLUX.2 VAE
    (128-ch latent = 32ch × 2×2 unshuffle + eval-BatchNorm). Klein = 4-step
    distilled, no CFG; `base` variants = 50 steps + CFG, same tensors.
    Parity-gated per stage (forward cosine 1.000000 vs diffusers).
    `brain flux2 generate` (t2i + `--ref` editing + `--mask` **masked latent
    conditioning** - a spatial preservation dial where `--strength` is a global
    one: white regenerates, black tracks the source latent renoised to each
    step's own sigma, so preserved regions reach sigma 0 as the source exactly.
    An all-white mask is bit-for-bit the unmasked run; masks are authored, not
    inferred - `.agents/roadmap/flux2.md` records why the depth-based
    generators tried do not work). `brain flux2 finetune <dir> --out
    <adapter.brain>` trains a LoRA on a captioned-image folder (`brain label`
    writes one) through the same `flux2::finetune::run` the `lora_train`
    capability action drives. **Two trainers, one op sequence**, selected by
    `--trainer device|host` (device is the default and the choice is printed
    on every run): `host` is the FD-gradchecked reference (`grad.rs` +
    `modelgrad.rs`), `device` is its WGSL instantiation (`devgrad.rs` +
    `devtrain.rs`) with the frozen base resident on the card and only the
    low-rank factors differentiated - so no dense `dW` is ever formed and the
    base is only ever read. Gated against the reference at cosine 1.000000000
    and rel_l2 < 1e-6 on both block kinds and the whole model
    (`tests/dev_grad.rs`, `tests/device_train.rs`); attention runs as real
    GEMMs over `head_pack`ed head-major operands in BOTH directions, not the
    naive `attn_*_bidir` family. `--cards N` spreads the block stack over N
    GPUs (klein-9b's fp32 frozen base does not fit one 24 GiB card; klein-4b's
    does). See `.agents/roadmap/flux2.md` for the measured step cost and the
    per-kernel profile the next optimisation pass is measured from. The adapter must NOT be named `.safetensors`: `Pipeline::build_dit`
    uses that extension to recognise a third-party ai-toolkit/ComfyUI LoRA, so
    the CLI refuses it. 9B weights are NC-licensed - see
    `docs/models/flux2.md`.

12b-bis. **FLUX.1 / Kontext** (`crates/flux1`) - BFL's 12 B MMDiT: 19
    double-stream blocks (separate img/txt weights, joint attention over
    `cat(txt, img)`) then 38 single-stream parallel blocks, with **per-block**
    modulation (77 linears - so unlike FLUX.2 the modulation stays on the device
    as 77 `m = 1` GEMVs), biased linears throughout, GELU(tanh) MLPs, 3-axis
    interleaved RoPE (theta 10000), and the Kontext edit path (reference images
    VAE-encoded and appended, axis-0 id = 1). Imported diffusers **1160 → 780**
    BFL tensors, two-way covered. Adds **no kernel**. **Forward-parity-gated at
    reduced depth in fp32 (worst 1−cos 1.5e-11, the enforced floor) and at
    full depth in int8, where `out` cosine measured 0.9985 / 0.9991 on real
    image fixtures - that number is what a real run happened to produce, NOT
    the test's enforced floor, which is 0.95 (int8 is a lossy tier; the floor
    only needs to catch a broken port, not reproduce this specific
    measurement)** - the full-depth fp32 number does NOT fit a 24 GiB card
    and is not claimed. See `.agents/roadmap/flux1.md`.
    **The sampler loop, VAE and text-encoder glue exist**
    (`pipeline::Flux1::generate` - T5-XXL context + CLIP-L pooled
    conditioning, BFL's own linear `calculate_shift` schedule - NOT FLUX.2
    Klein's `empirical_mu`, different constants - a 16-channel VAE decode via
    its scalar affine, not FLUX.2's BatchNorm packing). **Serving contract
    met**: `flux1::caps` (`text2image`), `resident_flux1::Flux1Resident`
    (`BRAIN_FLUX1_DIR`), D-Bus `Run`, `examples/imagegen/flux1_generate.py`.
    *(Text-to-image only - no Kontext editing, img2img or LoRA yet; no batch >
    1; backward/gradcheck deferred. Unlike this cluster's other served
    models, the PIPELINE glue has no end-to-end fixture in this workspace to
    verify it against - see `crates/flux1/src/pipeline.rs`'s module docs for
    the honest scope of what is and is not checked.)*

12b-ter. **T5-XXL encoder** (`crates/t5encoder`) - the text encoder FLUX.1 conditions
    on: bidirectional encoder-only T5 (RMSNorm, no bias, **no** `1/√d_kv`
    attention scale, learned relative-position bucket bias shared by every
    layer, gated-GELU `wi_0`/`wi_1` FFN). Imported **219 → 171** tensors,
    two-way covered; `relative_position_bucket` exact over all 16384 entries.
    Forward-parity-gated per stage at **42/42, worst cosine 0.9999999992**
    (B=2, T=128), plus a **checkpoint-free** `tiny_ref` gate at deliberately
    distinct dims (`heads ≠ d_kv`, `heads·d_kv ≠ d_model`) at cosine 1.0000000000
    - because at XXL those three numbers are all equal and a swap would be
    invisible. See `.agents/roadmap/t5encoder.md`. **Backward exists**: a full
    hand-written T5 backward (`crates/t5encoder/src/train.rs`) gated by
    `gradcheck::check_t5{,_one_block,_tiled}`. **Serving contract met**:
    `t5encoder::caps` (`encode`, `variant` picks flux_xxl/wan_umt5, tokenized
    via `data::unigram`), `resident_t5encoder::T5encoderResident`
    (`BRAIN_T5ENCODER_DIR`), D-Bus `Run`, `examples/embedding/t5_embed.py`.
    *(**T=512 - the length FLUX.1 actually uses - is untested** at the model
    level, and the served path has no fixture to verify end to end in this
    workspace's checked-in test data - see `.agents/roadmap/t5encoder.md`.)*

12c. **VQGAN / CodeFormer VQ autoencoder** (`crates/vqgan`) - the VQ
    encoder/codebook/generator that CodeFormer's face restoration is built on:
    a 25-block encoder and a 25-block generator over `vae::blocks` (conv,
    GroupNorm, self-attention, ResNet, nearest-upsample), with `vq_argmin` for
    the codebook assignment and `embed` for the lookup. Adds **no kernel and no
    block**. Both released checkpoints (`codeformer.pth`, `vqgan_code1024.pth`)
    imported and forward-parity-gated at cosine 1.000000000 with **zero**
    code-index disagreements. See `.agents/roadmap/vqgan.md`. **Serving
    contract met**: `vqgan::caps` (`encode`/`decode` - the codes travel as a
    `Media::Bytes` blob), `resident_restore::VqganResident`
    (`BRAIN_VQGAN_WEIGHTS`), D-Bus `Run`, `examples/restore/`. **Training /
    backward done** (`crates/vqgan/src/train.rs`, gated by
    `gradcheck::check_vqgan`). *(`run_batch` is the serial default and says
    why.)*

12c-ter. **Real-ESRGAN super-resolution** (`crates/rrdbnet`) - the imaging
    pipeline's upscale tail: `RRDBNet`, a residual-in-residual dense block
    trunk mapping `[3,H,W]` to `[3,scale*H,scale*W]`. The reference
    discriminator is training-only and is not ported. No new kernels - the
    whole net is conv + LeakyReLU + channel concat + nearest-2x upsample,
    composed from `vae::blocks::Builder` (the same shared conv-block builder
    the VQGAN family uses) plus `leaky_relu`/`concat2`/`scale_add`. Feeds RGB
    in `[0,1]`, brain's own wire format, so import needs no affine, only the
    HWC-blob-to-CHW-model permutation. A real tiling path with the halo the
    released net needs is implemented; blending the tile overlap measured
    WORSE than cropping it, so cropping is what ships. **Serving contract
    met**: `rrdbnet::caps` (`upscale`, env-gated on `BRAIN_ESRGAN_WEIGHTS`),
    `resident_upscale::UpscaleResident`, D-Bus `Run`, and wired into
    `imgpipe` as the pipeline's `UPSCALE_MODEL` stage. *(No backward/
    gradcheck yet - forward-only, matching its siblings in this cluster.)*

12c-bis. **CodeFormer face restoration** (`crates/codeformer`) - what turns the
    VQ autoencoder above into a blind face restorer: the **code-prediction
    Transformer** (9 pre-LN `TransformerSALayer`s over the flattened encoder
    output, learned `position_emb` added to q/k but **not** v - which is why the
    fused `in_proj_weight` is split into `qk`/`v` at import - erf-GELU MLP, and
    a biasless 1024-way head whose argmax replaces the nearest-neighbour
    codebook search), the **controllable feature transformation**
    (`Fuse_sft_block`) at four resolutions, and the **identity-fidelity dial
    `w`** (`w = 0` maximum quality, `w = 1` maximum fidelity; a one-element
    device buffer read by `scale_add`, so changing it is a write, not a graph
    rebuild). Composes `vqgan::model::run_blocks` + `vae::blocks` and adds
    **no kernel and no block**. Two-way import coverage over all 515 checkpoint
    tensors; forward-parity-gated per stage at cosine 1.000000000 with **zero**
    code-index disagreements at every `w`. See `.agents/roadmap/codeformer.md`.
    **Serving contract met**: `codeformer::caps` (`restore_face`, `w` as a plain
    float param), `resident_restore::RestoreResident` (`BRAIN_CODEFORMER_WEIGHTS`),
    D-Bus `Run`, `examples/restore/`. *(Forward only: `adain=True` - the
    reference CLI's path - face detection/alignment, batch > 1, sizes ≠ 512² and
    backward/gradcheck are all deferred and listed in that ledger.)*

12d. **CLIP text + image towers** (`crates/clip`) - one config-driven graph for
    the three encoders the imaging workstream needs: **CLIP-L** and
    **OpenCLIP-bigG/14** text (SDXL conditioning; CLIP-L again for FLUX.1) and
    the **EVA-CLIP-L/336** image tower (PuLID), the latter with 2D RoPE on q/k
    excluding cls, subln attention and naive SwiGLU. Imported from the SDXL
    `text_encoder{,_2}` safetensors and the EVA `.pt`, two-way covered, and
    forward-parity-gated per stage vs `transformers`/`open_clip` (cross-checked
    against `diffusers.encode_prompt` for the SDXL conditioning pair).
    The **CLIP BPE tokenizer** now lives in `crates/data` next to the GPT-2/Qwen3
    BPEs (`data::clip_bpe::ClipBpe`) - it reuses `data::bpe`'s merge loop and adds
    only what differs (`</w>` word-end marker, CLIP's pre-tokenization, lowercase
    + whitespace collapse, the 77-token `<|startoftext|>`…`<|endoftext|>` frame),
    gated at **exact id equality** vs HF `CLIPTokenizer` on both SDXL tokenizers.
    **Serving contract met**: `clip::caps` (`embed_text` batched per-tower,
    `embed_image`), `resident_clip::ClipResident` (`BRAIN_CLIP_DIR`), D-Bus
    `Run`, `examples/embedding/clip_embed.py`. **Backward is gated**
    (`gradcheck::check_clip`, wired in
    `crates/gradcheck/tests/imaging_models.rs`; `check_clip_bigg` and
    `check_clip_tiled` cover the bigG tower and the tiled path).

12e. **SDXL UNet2DConditionModel** (`crates/sdxlunet`) - the first UNet *diffusion
    backbone* in the imaging stack (`crates/diamond` has a UNet-shaped world
    model, recorded by hand): `CrossAttnDownBlock2D`/`DownBlock2D` → mid → up
    with skip concats, ResBlocks whose timestep embedding is **added**
    (`resnet_time_scale_shift: "default"` → `add_chan_bcast`, *not* `film_chan`),
    `Transformer2DModel` spatial transformers (self-attn + cross-attn + GEGLU) on
    the two inner levels, and the SDXL added conditioning
    `concat([pooled_text, time_ids_sinusoids])`. **Two GroupNorm epsilons in one
    graph** (1e-5 in the resnets, 1e-6 inside every transformer), which is why
    `vae::blocks::Builder` gained `set_eps`. Imported diffusers **1680 → 1610**
    tensors, two-way covered; adds **no kernel and no block** - the conv half is
    `vae::blocks::Builder`, the transformer half is `model::block`.
    Forward-parity-gated per stage at **165 comparisons / 0 failed**, worst
    cosine 0.9999999999, `out.sample` cosine 1.0000000000 and rel_l2 3.258e-6
    (both cosine *and* rel_l2 are asserted - cosine alone is scale-invariant).
    **2 567 463 684 params = 10.27 GB fp32**, 2198 dispatches and 4.06 s per
    forward at SDXL's native 128×128 latent, so it fits one 24 GB P40.
    The **discrete schedulers** it needs - DDIM / Euler / Euler-ancestral /
    DPM-Solver++(2M) × {ε, v-pred} - live in `diffusion::discrete`, gated at
    **66 checks / 0 failed** (timesteps, sigmas, `init_noise_sigma`,
    `scale_model_input` and the full step trajectory). See
    `.agents/roadmap/sdxlunet.md`. **The sampler loop, VAE and text-encoder
    glue exist** (`pipeline::Sdxl::generate` - dual CLIP conditioning, a
    discrete Euler step, CFG, VAE decode - "SDXL works" end to end).
    **Serving contract met**: `sdxlunet::caps` (`text2image`),
    `resident_sdxl::SdxlResident` (`BRAIN_SDXL_DIR`), D-Bus `Run`,
    `examples/imagegen/sdxl_generate.py`. **Backward + gradcheck done**
    (`crates/sdxlunet/src/train.rs`, gated by `gradcheck::check_unet` and
    `check_unet_conditioning_elementwise`) - which required teaching the shared
    `vae::blocks` tape to RECORD the transformer half rather than
    `push_step` past it; see `.agents/rules/lessons.md` #55. *(No batching -
    every request is its own multi-step sample, see `resident_sdxl.rs`'s module
    docs for why.)*

12f. **ControlNet** (`crates/controlnet`) - phase 4c: a **backbone-agnostic
    control seam** plus the SDXL `ControlNetModel` that is its first producer.
    The deliverable is the seam: `ControlAdapter` declares a backbone's *named*
    `InjectionPoint`s (`Layout::Spatial{c,h,w}` for a UNet, `Layout::Tokens{t,d}`
    for a DiT stream) and `check_compatible`/`order_for` match producer to
    consumer **by name and element count** - because diffusers zips a bare tuple,
    and SDXL's four 320-ch and three 640-ch points make a permutation
    type-check, run, and produce a plausible image. Nothing in `adapter.rs`
    mentions convolutions, resolutions or SDXL, so a FLUX DiT implements it
    unchanged. The trainable copy **is** the UNet's blocks - recorded by
    `sdxlunet::model::Rec`, adding **no kernel** beyond `scale_chan`.
    Imported **844 → 810** tensors, **1 251 014 160 params = 5.00 GB fp32**.
    Residual-parity-gated vs a hooked diffusers `ControlNetModel` at
    **140 comparisons / 0 failed, worst 1−cos 1.914e-11, worst rel_l2 6.187e-6**
    on both a P40 and `BRAIN_DEVICE=cpu`. See `.agents/roadmap/controlnet.md`.
    **Serving contract met**: `controlnet::caps` (`text2image` - a sampler
    loop built on `Unet::new_controlled` + `Unet::run_with_control`, one
    `ControlNet::run` per step, reusing the same two CLIP towers/scheduler/VAE
    calls `sdxlunet::caps`'s does), `resident_controlnet::ControlnetResident`
    (`BRAIN_SDXL_DIR` + `BRAIN_CONTROLNET_DIR`), D-Bus `Run`,
    `examples/imagegen/controlnet_generate.py`. *(No backward, no
    `check_controlnet`, no INT8, no batch > 1 - every request is its own
    multi-step sample, same as plain SDXL. "InstantID works" is still NOT
    claimed - InstantID is a separate crate, `crates/instantid`, whose forward
    is not implemented at all.)*

12g. **PuLID-FLUX identity conditioning** (`crates/pulid`) - phase 5's FLUX half:
    the `IDFormer` Perceiver resampler (a face embedding → 32 ID tokens) and the
    injected `PerceiverAttentionCA`, cross-attended into the FLUX.1 image stream
    at **20 sites** (after double block `i` when `i%2==0`, after single block `i`
    when `i%4==0`, one **shared sequential** `ca_idx` counter across both loops).
    `img = img + id_weight · ca(id, img)` - added to the residual stream, never
    concatenated as tokens. Composes `clip::EvaVision`
    (`EvaVisionConfig::PULID_TAPS`) and `flux1::{Flux, inject, KERNELS}`; adds
    **no kernel and no shared block**, and contains no second face model and no
    second CLIP. 312 tensors → 562 M params. Parity-gated vs a hooked reference
    on both backends - IDFormer 29 taps, the CA unit 8, and the **conditioned
    FLUX.1 forward** 10, worst 1−cos **1.44e-11**. See `.agents/roadmap/pulid.md`.
    **The image → `id_cond` path exists** (`idcond::IdCond::from_image` /
    `idcond::compose` - ArcFace raw embedding ‖ L2-normalised EVA-CLIP CLS,
    the reference's asymmetric convention, gated by
    `idcond::tests::the_eva_half_is_normalised_and_the_arcface_half_is_not`).
    **Serving contract met**: `pulid::caps` (`text2image` - ArcFace +
    EVA-CLIP → `IdFormer` → `PulidAdapter` → `flux1::pipeline::Flux1::
    generate_injected`), `resident_pulid::PulidResident` (`BRAIN_FLUX1_DIR` +
    `BRAIN_PULID_DIR` + `BRAIN_ARCFACE_DIR` + `BRAIN_CLIP_DIR`), D-Bus `Run`,
    `examples/imagegen/pulid_generate.py`. *(Forward only: no backward, no
    `check_pulid`, no batch > 1. Only `dev` is validated against a PuLID
    reference. One real, documented gap: the served path resizes the face
    crop straight to EVA-CLIP-L/336 rather than reproducing the reference's
    RetinaFace+BiSeNet preprocessing (`crates/pulid/src/caps.rs`'s module
    docs). "PuLID works" end to end is not claimed - no reference dump of a
    full ID-conditioned generation exists in this workspace to check it
    against.)*

12h. **Wan2.1 / Wan2.2 text-to-video** (`crates/wan`) - the first VIDEO model:
    a DiT denoising a 3D `(frame, height, width)` latent volume under flow
    matching, SDXL's attention topology (self-attention over the volume, then a
    *separate* cross-attention into the text encoding - not FLUX/Z-Image's joint
    sequence), 3-axis RoPE, umT5-XXL conditioning, and a **causal 3D VAE** at a
    (4, 8, 8) stride. Modulation is token-independent and folds into LayerNorm
    affines. Adds **four kernels total** - `conv3d`/`conv3d_dx`/`conv3d_dw` for
    the VAE and `attn_keypad_mask` for umT5's key padding; the whole transformer
    is existing kernels at Wan's shapes. Parity-gated: VAE every boundary at
    cosine 1.000000, the real 1.3B DiT at 4,680 tokens at cosine 1.000000000
    (rel_l2 3.755e-6) against BOTH the official repo and diffusers, on Vulkan and
    the CPU JIT. **Serving contract met**: `wan::caps` (one `t2v` action),
    `resident_wan::WanResident`, D-Bus `Subscribe` + `Cancel`,
    `examples/videogen/`. **Training done, host only**: `grad.rs`/`modelgrad.rs`/
    `lora.rs`/`finetune.rs`, gated by `gradcheck::check_wan`
    (block FD 1.8e-9, model FD 1.7e-8, LoRA a bit-exact no-op at init).
    `brain wan t2v` is one command to a playable mp4, with auto-fetch.
    *(**No image-to-video**, no INT8, no `lora_train` action, no batched forward,
    and NOT optimized - see `.agents/roadmap/wan.md` for the published per-kernel
    profile that a later optimization pass is measured against.)*

12i. **LTX-2.5** (`crates/ltxv`, + `crates/gemma4` as its text encoder) - a
    **two-stream (video + audio)** diffusion transformer, and the widest port
    in the repo. Implemented and parity-gated: the causal 3D video VAE
    (`vae3d`), the 2D causal-conv **audio VAE** (`audio_vae`) and the
    BigVGAN/snakebeta **vocoder** (`vocoder`) all on real weights; the
    video-only DiT stream and the **audio DiT stream + bidirectional
    audio<->video cross-attention** (`block::LtxAvBlock`, per-block `[5,dim]`
    adaLN tables, a cross-modality-sigma gate, a shared time-only cross-modal
    RoPE space) at **tiny-config op-sequence parity only**
    (`tests/dit_parity.rs`); the convolution-free `DiffusionVideoDecoder` /
    3D neighborhood attention (`na_decoder`), not yet wired into the
    pipeline. Also int8, LoRA/finetune (video and A/V), sharding, and a
    long-form path. **Serving contract met**: `ltxv::caps`,
    `resident_ltxv::LtxvResident`, a `catalog.rs` entry.
    *(**The `t2v` pipeline is a SMOKE TEST**, and says so: real scheduler
    math, real CFG denoise loop and real VAE decode, but a **tiny
    random-weight DiT** and a **stub text context** - the real 22B checkpoint
    import and the real Gemma-4 text encoder are both open. Do not read
    "serving contract met" here as "LTX-2.5 generates video". See
    `pipeline.rs`'s module doc and `.agents/roadmap/ltxv.md`.)*
    **`crates/gemma4`** is LTX-2.5's text tower (the text-only forward path
    through a 12B unified text/vision/audio model - no vision or audio
    tower): tiny-dim op-sequence parity against goldens, plus a real
    686-tensor shape-level import. Real-weight per-stage parity at full
    scale is a tracked gap.

### Audio / speech

13. **Qwen3-TTS** (`crates/qwen3tts` + `crates/mimi` + `crates/ecapatdnn` +
    `crates/audio`) - Talker (multi-codebook dual-track Qwen3 decoder) + 5-layer
    MTP code predictor → 12 Hz Mimi-style neural codec (RVQ + transformer +
    SEANet conv-transpose decoder); ECAPA-TDNN speaker encoder for voice cloning.
    `brain qwen3tts {import,clone,synth,design,serve,sim,finetune}`.
13b. **ASR / speech-to-text** - two imported, parity-gated models served through the
    full stack (capability + residency + batched `run_batch` + D-Bus
    `StreamTranscribe` + `examples/asr/`):
    * **Nemotron 3.5 ASR Streaming 0.6B** (`crates/nemotronasr`) - FastConformer
      encoder (depthwise-sep causal subsampling, macaron FFs, Transformer-XL
      rel-pos attention, GLU conv module) + RNN-T transducer; the *streaming* model,
      true batched forward across concurrent windows. Fully trainable/gradchecked.
    * **Qwen3-ASR 1.7B** (`crates/qwen3asr`) - Whisper-style audio encoder + a spliced
      Qwen3-1.7B decoder (reuses `crates/qwen3`); offline, fixed audio window.
    Shared audio-in/text-out contract in `audio::asr_caps`. See
    `.agents/roadmap/asr.md`.
13c. **Qwen3-Omni-30B** (`crates/qwen3omnimoe`) - Thinker (dense-then-MoE Qwen3
    decoder, real M-RoPE incl. audio/image/video splice) + Talker
    (sigmoid-gated MoE) + 5-layer MTP code predictor → Code2Wav vocoder,
    composed end to end: text/speech/image/video in, text + real synthesized
    speech out, served over D-Bus/OpenAI/Anthropic (`brain caps`/`brain qwen3omnimoe ...`,
    `examples/omni/omni.py`). int8-native checkpoint import exists and has
    been run for real (70GB→36GB, 54,764 tensors, exact two-way name
    coverage). A layer-sharded int8 dual-GPU Thinker (`crates/qwen3omnimoe/src/
    int8_resident.rs`, `int8_thinker_resident.rs`) is built and validated on
    two real P40s against the REAL 30B checkpoint: real int8 expert AND
    non-expert weights resident across both cards, real cross-device residual
    handoff, KV-cached greedy decode, and the same chat request contract
    `brain/qwen3omnimoe` takes (`qwen3omnimoe::caps::chat_generate_spec`), so it serves
    `/v1/chat/completions` and `/v1/messages` rather than raw token ids only
    (`Executor::register_multi`, `crates/cli/src/resident_omni.rs::
    int8_thinker_multi_from_env`, env `BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT` +
    `BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR`). Measured: **2.3 s/token vs 57.6
    s/token** for the streaming bf16 path, identical output, 16.9/16.7 GiB on
    two 24 GB cards. Text only - multimodal input and `speak` still require
    `brain/qwen3omnimoe`; see `.agents/roadmap/qwen3omnimoe.md`.
    **Qwen3-VL** (`crates/qwen3vl`) is a separate served model, `brain/qwen3vl` - reuses `crates/qwen3`'s decoder
    (KV-cache decode path carries real M-RoPE + DeepStack support), image +
    text in, greedy text out, `brain caps`/`brain qwen3vl ...` (`crates/qwen3vl/src/
    caps.rs`). No residency adapter yet (not servable over D-Bus/HTTP);
    real-checkpoint coverage of the served path is a skip-if-absent smoke in
    `qwen3vl::caps::tests` (runs when `BRAIN_QWEN3VL_WEIGHTS` is set).
    Full ledger: `.agents/roadmap/qwen3omnimoe.md`.
13d. **FastVLM** (`crates/fastvlm`) - served VLM, `brain/fastvlm`: FastViTHD
    conv/attention vision tower + `mlp2x_gelu` projector spliced into a Qwen2
    decoder; one `caption` action (per-token Progress) over `brain fastvlm caption` and
    D-Bus (stateless resident). fp32/int8 decoder precision; training loop
    exists (`train_smoke.rs`) but has no CLI verb. **Moondream 3**
    (`crates/moondream3`) - SigLIP ViT with overlap multi-crop + a
    parallel-block sparse-MoE decoder, gradient-checked and import-covered.
    **Serving contract met**: `moondream3::caps` (one streaming `caption`
    action), `crates/cli/src/resident_moondream3.rs`, a `catalog.rs` entry,
    D-Bus `Subscribe`, `examples/vision/moondream3_caption.py`.
    **int8 is the default and is what makes it loadable at all**: the fp32
    build is 32.8 GiB of weights plus 10.3 GiB of per-block activation scratch
    (~43 GiB); `Precision::Int8` quantizes the 1280 expert tensors
    (`MoeFfn8` over `moe_linear_gated_i8`) and puts all 24 blocks on ONE shared
    `BlockScratch`, together ~8.8 GiB. Precision is part of the instance key,
    so the two are separately budgeted. *(Decode IS KV-cached
    (`generate_kv`: one masked batched prefill seeds every layer's cache, then
    `O(pos)` steps), gated token-for-token against the `O(T²)` recompute path;
    `run_batch` does REAL batching on the
    vision half (N requests' crops through one `SiglipEncoder::encode`; the
    decoder half is per-request and says why); region/point/detect heads recognized but not built; GPU-placeable,
    with the device plumbing gated by a tiny-config CPU-vs-card agreement test
    rather than by a real-weight run. No real-weight run exists in this workspace, so the composed
    path is gated by checkpoint-free tests through the production loader.)*
    Both documented on `docs/models/vlm.md`; full ledger
    `.agents/roadmap/vlm.md`.

13c-bis. **FastVLM-0.5B and Moondream 3** (`crates/fastvlm`, `crates/moondream3`) -
    the other two vision-language architectures alongside Qwen3-VL above; all
    three share one shape (vision encoder → connector/projector →
    autoregressive text decoder, image embeddings spliced into the decoder's
    stream) and one validation ladder, documented together on
    `docs/models/vlm.md`. **FastVLM** (Apple): FastViTHD hybrid
    conv/attention vision encoder + `mlp2x_gelu` projector in front of a Qwen2
    decoder (LLaVA-style splice, image token id `-200`). Real-weight
    validated end to end - decoder logits at mean|Δ|≈3e-6 vs `transformers`,
    the full in-brain pipeline (its own vision tower → projector → decoder)
    reproduces HF's caption token-for-token on a real image, and a 300-step
    finetune smoke test collapses loss from ~ln(vocab) to 0.01. **Serving
    contract met**: `fastvlm::caps` (stateless `FastVlmProvider`), `brain
    caps`/`brain fastvlm caption` - no residency adapter yet, matching Qwen3-VL's own
    state. **Moondream 3**: a vision encoder (SigLIP-style, overlap
    multi-crop) + a parallel-block decoder with MoE expert sharding.
    Decoder gradient-checked and import-covered (662 tensors); full-model
    real-weight parity does not fit this box's memory (a 28 GB checkpoint),
    so its decoder is instead streamed and parity-checked per-block - a real
    bug (a missing fused-qkv bias the checkpoint carries but the block
    didn't apply) was caught this way after gradchecks alone had missed it.
    Region/point/detect heads are recognized on import but not built.
    *(Not yet wired into the capability system - no `caps.rs` - so it is
    validated but not servable via `brain caps`/`brain moondream3 ...` yet.)*

13e. **DeepSeek-OCR** (`crates/deepseek2ocr`, over `crates/sam1` +
    `crates/clip` + `crates/deepseek2` + `crates/gguf`) - a document page in,
    text/markdown out: the **DeepEncoder** (SAM ViT-B at 1024² with decomposed
    relative-position bias → a 16x conv token compressor → CLIP-L/14 with its
    patch embed **bypassed** in favour of those tokens → concat
    `[clip_spatial, compressor_flat]` → one projector linear) spliced into the
    **DeepSeek-V2-family MoE decoder** (12 layers, 64 routed experts top-6 +
    2 shared fused, plain MHA - not MLA). Imported from the shipped
    `ggml-org/DeepSeek-OCR-GGUF` Q8_0 pair with two-way coverage over both
    files. Parity-gated per stage against a checkpoint-free golden dump at
    deliberately non-coincidental dims (SAM patch embed → decoder logits), and
    real-weight-gated at production shape; the SAM tower's own real-weight
    parity is what found the **wgpu 3-or-more-block corruption at 1024²** that
    pins this whole model to the CPU backend. The **real 273-row image block**
    is assembled (256 projector rows + 16 `image_newline` + 1 `view_separator`,
    the mmproj's two learned vectors) by `layout::RowGather` and sized from
    `prompt::build_prompt`'s own `n_rows`; the backward is the exact adjoint
    and reaches the input pixels. Adds **no kernel** beyond the six
    `attn_relpos_*` the SAM bias needed (gradient-checked,
    `gradcheck::check_deepseekocr_relpos{,_elementwise}`).
    **Serving contract met**: `deepseek2ocr::caps` (`generate`, streaming, real
    `prompt_tokens`/`completion_tokens`/`finish_reason`),
    `crates/cli/src/resident_deepseekocr.rs` (`BRAIN_DEEPSEEK_OCR_DIR`), one
    `catalog.rs` entry wiring `brain caps`/`brain deepseek2ocr ...`/D-Bus/OpenAI/Anthropic at
    once, `examples/vision/deepseek-ocr/`. The production checkpoint loader is
    `deepseek2ocr::import` - this crate's four real-weight test binaries are
    thin wrappers over it, so a served run and its own parity test cannot
    disagree about which tensors they loaded.
    *(**Split backend**: `caps::Session::load` builds the vision encoder
    (SAM+CLIP+glue) on `Gpu::new_wgpu` and the decoder on `Gpu::new_cpu` -
    `crates/sam1`'s wgpu corruption at 1024x1024/3+ blocks that used to force
    an all-CPU build is fixed and confirmed at real-weight scale. Because it
    then holds real bytes on TWO devices, it is the repo's second
    `MultiDeviceResidentModel` (after the int8 Omni Thinker) and the first one
    in `catalog.rs`: `estimate_multi` names `(Gpu(i), 6 GiB)` for the vision
    tower and `(Cpu, 16 GiB)` for the host side - a decomposition of the one
    measured 21.32 GiB all-CPU peak, so the halves sum to it rather than each
    claiming it - and `activate_multi` builds the tower on exactly the
    reserved card via scoped registry selection. It used to report a RAM-only
    `MemCost` (`vram == 0`), which left the tower's device bytes invisible to
    the budget. Never an env mutation from inside a
    server-lifetime resident. `run_batch` is the serial default and says why
    (per-image encoder pass, no decoder batch axis). Decode IS KV-cached
    (`DeepseekV2::generate_greedy_kv`, `O(1)` per token past the prompt).
    Still not done: EOS early-stop, sampling beyond greedy, INT8, a
    finetune-style CLI verb for the LoRA backward that already exists, and
    the Base/Gundam multi-tile layouts (`rows.rs` and
    `RowGather` already support them; `DeepseekV2::enable_mm_splice` takes ONE
    run). The composed image+decoder decode loop has **no multimodal oracle** -
    llama.cpp's debug callback segfaults inside this model's CLIP graph - so it
    is gated on completing, finite logits and causal self-consistency, and
    token-for-token agreement with the reference is NOT claimed. Full ledger:
    `.agents/roadmap/deepseek2ocr.md`.)*

13f. **CosyVoice 2/3** (`crates/cosyvoice` + `crates/s3tokenizer` +
    `crates/campplus`) - LLM-based streaming zero-shot voice cloning: a
    Qwen2.5-0.5B speech-token LM (hosted on `crates/qwen3`) + a causal
    flow-matching mel decoder (CosyVoice 2's UNet CFM estimator, CosyVoice
    3's 22-layer adaLN-zero DiT estimator) + an ISTFT/NSF HiFT vocoder,
    conditioned on a reference clip via S3Tokenizer (FSQ speech tokens) and
    CAM++ (a 192-d x-vector). One crate, one architecture id, both
    generations as a `variant` config, not two ids. Real-weight parity
    proven for every component of BOTH generations (cosine >= 0.9999998
    everywhere, most rungs at 1.0000000000); the speech-token LM is
    additionally gradient-checked (block FD 1.09e-9, model FD < 2e-6) and
    LoRA-capable. **Serving contract met for CosyVoice 2**: `cosyvoice::caps`
    (one `synth` action, streaming), `crates/cli/src/resident_cosyvoice.rs`
    (load-per-call, following `resident_minimaxmusic3.rs`), `brain
    caps`/`brain cosyvoice synth`/D-Bus/HTTP, `examples/tts/
    cosyvoice_synth.py`. **Both generations run**: `pipeline::Variant` selects
    the LM config, the flow decoder (UNet CFM vs the 22-layer adaLN-zero DiT)
    and the vocoder (non-causal vs causal HiFT), and everything the two share -
    CAM++, S3Tokenizer, the prompt mel, the truncation rule, the token budget -
    is written once. `variant="cosyvoice3"` needs the `BRAIN_COSYVOICE_*` dirs
    pointing at a CosyVoice 3 checkpoint; a wrong-generation checkpoint fails in
    the importer, never silently. Flow decoder and HiFT vocoder training (both
    generations) remain forward-only, and the CosyVoice 3 path has no
    real-weight end-to-end run recorded on this box. Full ledger:
    `.agents/roadmap/cosyvoice.md`.

13g. **MiniMax Music 3** (`crates/minimaxmusic3`) - the repo's music-generation
    model: lyrics + a structured caption in, a full song (up to 5 minutes,
    44.1 kHz stereo) out. Five chained components, only three of them new
    here: a **Global LLM** (a real Qwen3-8B, reused verbatim from
    `crates/qwen3` - `vocab=200000`, from the checkpoint's own
    `language_model/config.json`, NOT the smaller published Qwen3-8B preset)
    emitting one semantic RVQ code per 25 Hz frame under CFG → an **RVQ depth
    decoder** (4-layer causal transformer predicting the 7 residual codebooks
    per frame) → a **condition encoder** (softmax layer-mix, conv proj,
    resample to latent rate) → a **36-layer flow-matching DiT** denoising
    Flow-VAE latents in 200-frame chunks (100-frame hop, 172-latent overlap
    splice) → a **DAC-style SnakeBeta vocoder**. Ported from an unmerged
    `diffusers` PR - there is no official upstream inference code. INT8,
    LoRA and sharding on the DiT; a discriminator and training loops exist
    per component. **Serving contract met**: `minimaxmusic3::caps`
    (`generate`), `resident_minimaxmusic3::MinimaxMusic3Resident`, a
    `catalog.rs` entry, D-Bus, `examples/musicgen/`.
    *(Load-per-call, like `cosyvoice`. The real short end-to-end WAV gate is
    written but blocked on this box; joint generator+discriminator training
    and a multi-resolution discriminator are open. See
    `.agents/roadmap/minimaxmusic3.md`.)*

### Forecasting

14. **Chronos-2** (`crates/chronos2`) - encoder-only T5-style patch transformer,
    time+group attention, multi-patch quantile head. Imported exactly, parity-gated.
15. **Kronos** (`crates/kronos`) - BSQ tokenizer (OHLCV bar → hierarchical tokens)
    + autoregressive decoder with a dual head. Imported exactly, parity-gated.
16. **FinCast** (`crates/fincast`) - TimesFM-style patched decoder with a sparse
    top-2 MoE and a probabilistic-quantile head. Imported exactly, parity-gated.
    *(Reference is research/educational use only.)*
    All three sit behind the model-agnostic `forecast::ForecastModel` seam;
    `crates/fcbench` holds baselines + the rolling-origin backtester.
    `brain forecast {compare,serve,import,finetune}`.

### World models (playable, action-conditioned video)

17. **DIAMOND** (`crates/diamond`) - EDM diffusion world model (Atari-100k):
    pre-recorded UNet graph, torch `.pt` import, playable. Parity fixtures via
    `make wm-fixtures`.
18. **GenieRedux-G** (`crates/genieredux`) - CoinRun ST-transformer world model
    (QK-normalized biased attention, GEGLU FFN, PEG); tokenizer/MaskGIT dynamics
    in progress. `brain diamond {play,replay,bench,import,finetune,export}`
    (`diamond` is the one served world-model architecture today; SDL window
    via `crates/wm-display`).

> `crates/timeseries` and `crates/autodiff` are **placeholders** - declared in the
> workspace, implemented in a later phase.

---

## Serving & runtime stack

The recent workstream (P7.x) is concurrent LLM serving. Key pieces:

| Piece | Where | What |
|---|---|---|
| Paged KV foundation | `crates/model/src/paged.rs` | block allocator, `BlockTable` (+`truncate`) |
| Serving engine | `crates/qwen3/src/serve.rs` | shared block pools, batched **ragged paged decode**, batched + **chunked prefill**, **int8 paged KV - the serving DEFAULT** (3.88× smaller pool at Qwen3's `head_dim=128`; opt out with `--kv-fp32` / `BRAIN_QWEN_KV_INT8=0`), calibration opt-in (`--kv-calib` / `BRAIN_QWEN_KV_CALIB=1`), **speculative decoding**, on-device greedy/top-K sampling head, `Engine::load` from checkpoint; implements the generic `model::serve::PagedDecoder` seam. Served context length defaults to `BRAIN_QWEN_CTX=24576` (sized to what int8 KV buys; the fp32 opt-out is guarded and refuses over the iGPU's 8 GiB policy budget rather than OOMing - `crates/cli/src/resident_llm.rs`). |
| Scheduler | `crates/model/src/serve.rs` | `PagedDecoder`-generic continuous batching (multi-sequence concurrent admission/decode) + real (non-greedy) sampling; `qwen3::serve::Scheduler` is a type alias over `Engine` - the seam a future decoder LM adopts by implementing `PagedDecoder`, not by duplicating the scheduler |
| Shared Qwen chat serving | `crates/qwen3/src/chat.rs` | chat-template rendering, tool-call/stop-string streaming, cancellation - one implementation shared by `resident_llm.rs` (HTTP/D-Bus) and `qwen3::caps.rs` (`brain qwen3 infer`), so they cannot diverge |
| Residency | `crates/residency` | tiers model weights GPU/RAM/disk by a size/reload-cost-aware policy within a memory budget; schedules jobs (batch-by-model, queue-age-aware, parallel lanes); `crates/residency/src/admission.rs` is the shared edge-concurrency-ceiling/admit-deadline policy both HTTP and D-Bus read from |
| Capability interface | `crates/capability` | models advertise a `Manifest` of typed `ActionSpec`s; CLI (`brain caps` / `brain <arch> <verb>`) and the event API dispatch generically - adding a capability = implementing `Action`, no new subcommand or event variant |
| Transports | `crates/server` | one JSONL protocol over **stdio, TCP, and Unix socket**; thread-per-connection, bounded, panic-isolated |
| D-Bus surface | `crates/dbus` | exposes the same `residency::Executor`-backed resident models HTTP serves (`Run`/`Subscribe`, streaming `Progress::delta`/`Progress::event`, the same admission deadline + concurrency ceiling as HTTP) over `com.swedishembedded.Brain1`, passing images/streams via fd (memfd/mmap + dmabuf). Example client: `examples/dbus`. Also serves the stats snapshot (`StatsSnapshot` method + `StatsStream` signal) and `ResidentModels` - just the `StatsSnapshot.models` rows with `resident == true`, so a client doesn't have to pull and parse the whole tree to answer "what's warm" |
| Stats subsystem | `crates/stats` (`brain-stats`) | self-describing, hierarchical JSON `StatsSnapshot` (accelerators/models/executor/requests/connections + open `extra`), assembled from `StatsSource` contributors; `braintop` renders it |
| Event HFSM | `crates/runtime`, `crates/events`, `crates/hfsm` | `camera_frame`→`object_detected`, `user_text`→`brain_text_chunk` |
| Python client | `brain-py/` | drives the `brain` binary as an event-driven subprocess (not in the build/test path) |

Multi-GPU scaling lives in `crates/model`:
`{distributed,parallel,collective,netcollective,shard,plan,grid}.rs` - see
`docs/scaling/`.

### Stats & braintop

`crates/stats` (`brain-stats`) is the **data-driven contract** a live-monitoring
TUI (`braintop`, and `braintop --cli`) renders from. `StatsSnapshot` is a
hierarchical tree of typed sections - each a **collection keyed by `id`**
(`accelerators`, `models` with per-instance residency, `executor`, `requests`,
`connections`) - plus an open `extra: BTreeMap<String, Value>` at every level.

**To add a metric:** add a field to the relevant typed section in
`crates/stats/src/snapshot.rs` (or, for something with no typed home yet, emit
into an `extra` map). It flows through the JSON snapshot and braintop renders it
automatically - typed views for known sections, a generic tree view for `extra`;
no schema migration for the `extra` path. **Never hardcode a count** - N
accelerators / N models / N instances all render from the data (one GPU or eight,
zero models or fifty).

**How it's assembled:** a `StatsSource` contributes into a snapshot; an
`Assembler` walks all registered sources (no central switchboard). The live wiring
is `build::ExecutorSource`, which reads one residency `Executor` clone - its
counters (`Executor::stats`), its manifest catalog (`Executor::manifests`), and
its residency + budget report (`Executor::residency`, backed by
`ResidencyManager::report` via a dispatcher `Msg::Report` round-trip) - to fill
accelerators (one row per budgeted device), models (catalog joined with
placement), and the executor section. `requests`/`connections` are left to
dedicated sources (a `JobRegistry`-backed request source can be layered in without
touching `ExecutorSource`).

**Surface:** the D-Bus `Manager` (`crates/dbus/src/service.rs`) exposes a
`StatsSnapshot() -> String` method and emits a `StatsStream` signal carrying the
same JSON at ≥2 Hz (`service::STATS_INTERVAL`, 500 ms) from a background task;
braintop subscribes there instead of polling.

`crates/stats` is serde + assembly only - it pulls no GPU/model/engine code (just
`brain-residency` and `brain-capability`), so it stays light enough for any
front-end to depend on.

---

## Workspace layout (`crates/`)

### Engine core

| Crate | Responsibility |
|---|---|
| `kernels` | every WGSL kernel (the source of truth) as consts + `src()` |
| `gpu-core` | compute-device facade: selects and forwards to an eager `Backend` |
| `backend-api` | `Backend`/`GraphBackend` traits, neutral buffer/step handles, registry - a new backend depends only on this |
| `backend-wgpu` | wgpu (Vulkan/Metal/DX12/GL/WebGPU) eager backend - **the default** |
| `backend-cpu` | native CPU backend: WGSL → Cranelift JIT across cores, AVX2 fast paths |
| `backend-vulkan` | native Vulkan (ash + naga WGSL→SPIR-V) eager backend |
| `wgsl-cpu` | the CPU backend's compiler: WGSL → naga IR → Cranelift JIT |
| `vulkan` | **optional, non-default** `VK_KHR_cooperative_matrix` matmul path (excluded from `default-members`; build with `-p brain-vulkan` / cli feature `vulkan-coopmat`) |
| `paramstore` / `optim` | param/grad/Adam buffers; AdamW + global grad-norm clip |
| `arch` | **the canonical model-architecture registry** (`ARCHS`). brain used to have four drifting answers to "which architecture is this" (the CLI's subcommand names, `modelstore::plan`'s HF-class substring scan, the GGUF importer table, `ModelCard::family`); all four read this table now. The `[a-z0-9]+` naming rule - llama.cpp's `LLM_ARCH_*` vocabulary where one exists, the upstream paper/repo name otherwise - lives here. **Adding a model means adding its row here**, not inventing a name at a call site |
| `memauth` | **the process-wide memory authority.** `residency`'s per-device integer budgets cannot express two things: (1) on an integrated GPU/NPU, "device VRAM" and "system RAM" are the SAME physical bytes, so two independent pools double-count; (2) `MemAvailable`/cgroup `memory.max` move while the process runs, so a budget probed once at startup is blind. `Topology` declares which devices share a `PoolId`, `PoolProbe` is the injectable live view (real one reads `/proc/meminfo` + cgroup v2; every test uses `FixedProbe` and never touches the machine), `MemoryAuthority` answers "may I allocate N bytes on device D right now" as an RAII `Grant` |
| `weightset` | **within-instance** weight residency: a fixed-size window of device slots over a model's weight *groups* (a transformer block, not a tensor). Distinct from `residency::EvictionPolicy`, which scores the *past* for an unpredictable request stream - a denoise or decode loop visits its groups in an order known exactly in advance, so `ResidencyPlan` plans over the known future instead. Pure host-side bookkeeping, no device memory, fully unit-testable without a GPU |
| `shutdown` | process lifecycle for every serving surface: one shutdown source (`Shutdown`) and one readiness latch (`ready::Gate`). SIGINT/SIGTERM disposition is **process-wide** - if each surface calls `tokio::signal::ctrl_c()` independently, only the first registration receives it and which one is unspecified. `install_signals` owns the single registration on a dedicated thread, so it does not matter which surface's runtime is built first |
| `checkpoint` | `.safetensors` container + manifest/SHA-256 + expert-shard I/O (no fs on wasm) |
| `model` | architecture-agnostic `Model` abstraction, generic trainer, shared block builders (`block.rs`, `vit.rs`), paged KV, and the multi-GPU parallelism layer |
| `autodiff` | shared SSA forward-cache / reverse-mode scaffolding - **placeholder** |
| `imaging` | the image substrate: decode/encode, device-dispatched resize/pad/crop/layout, colour normalisation, **mask algebra** (threshold/dilate/erode/feather/invert/union/intersect/difference/composite) and tiling - one home for what was scattered across zipdepth, yolov8, worldmirror2, s3dit, capture and cli |
| `imgpipe` | the composed pipeline: a stage list executed as ONE capability call, dispatching its model stages back through `capability::Registry`. Pixels outside the mask come back **bit-identical** |
| `captioner` | the model-agnostic captioning seam (`Clip` in, text out) plus the resumable folder labeler behind `brain label`. No model code: the implementors live in the VLM crates and depend on this one. Designed for video (the unit is a clip, not a frame), image path built |
| `data` | char + GPT-2/Qwen3/**CLIP** BPE tokenizers, the shared deterministic PRNGs (`rng::Rng` for datasets, `rng::Lcg` for tests/fixtures), dataset generators, loaders (masking/alignment), normalization | The captioned-image dataset's `captions.yaml`/`captions.jsonl` are parsed by `serde_norway`/`serde_json` into the typed `imageset::CaptionFile`/`CaptionLine` schemas - never by a hand-rolled scanner, which is how `key: |` once yielded the literal prompt `"|"`.
| `eval` | perplexity + task exact-match (LM) and detection metrics (mAP@0.5/precision/recall) |
| `gradcheck` | finite-difference backprop correctness gate |
| `bench` | model-agnostic architecture-evaluation suite - *does it **learn**?* (see below) |
| `perf` | performance benchmarking suite - *how **fast**, at what cost, still correct?* (see below) |
| `trace` | the tracing/observability front end: the `--trace-<family>` registry, the 0-5 level scale, the ONE `tracing_subscriber` install (text/JSON, stdout/file). Library crates depend on `tracing` only, never on this |
| `cli` | the `brain` binary (aggregates everything) |
| `web` | wasm32/WebGPU PID demo (empty off wasm32) |

### Model crates

| Crate | Model |
|---|---|
| `gpt2` / `qwen3` / `qwen35moe` / `qwen35` / `toymoe` / `glmdsa` / `toypid` | decoder LMs (see Models) |
| `toyseq2seq` / `toyautoencoder` / `timeseries` | encoder-decoder / bottleneck AE / placeholder |
| `federated` | vertical expert split/assemble, hash-verified manifests, train-scope |
| `yolov8` / `vision` | detector; shared conv-net blocks (spec-driven `Conv` incl. fused/register-tiled eval paths, `BatchNorm`, `PReLU`, `MaxPool`/`AvgPool`, `SPPF`, bottlenecks, `fold_bn`) |
| `zipdepth` | ZipDepth: model/blocks/import/fuse, `Predictor`, viz/stereo/effects, INT8 calib |
| `worldmirror2` / `splat` | WorldMirror-2; 3DGS rasterizer + PLY IO + `fit` + viewer |
| `scrfd` / `arcface` / `sam2` / `clip` | SCRFD face detection; ArcFace identity embedding (+ the 5-point alignment and its trainer); SAM 2.1 promptable segmentation (image path); CLIP-L/OpenCLIP-bigG/EVA-CLIP text+image towers |
| `diffusion` / `dit` / `vae` / `s3dit` | flow-matching core; shared DiT blocks; AutoencoderKL; Z-Image |
| `flux1` / `flux2` / `t5encoder` | FLUX.1/Kontext 12B MMDiT + edit path; FLUX.2 Klein 4B/9B MMDiT; T5-XXL and umT5-XXL text-conditioning encoders |
| `wan` | Wan2.1/2.2 text-to-video: the 3D-latent DiT, the causal 3D VAE, the sampling pipeline, both importers, and the host trainer/LoRA |
| `ltxv` / `gemma4` | LTX-2.5 two-stream (video+audio) DiT, both VAEs, the vocoder, the NA diffusion decoder, int8/LoRA/shard/long-form - pipeline is a smoke test, see Models 12i; Gemma-4 text-only tower that conditions it |
| `sdxlunet` / `controlnet` / `pulid` / `instantid` | SDXL UNet backbone; the backbone-agnostic control seam + its SDXL producer; PuLID identity conditioning on FLUX.1; InstantID's IP-Adapter-FaceID shapes (**forward not implemented** - see `crates/instantid/src/lib.rs`) |
| `vqgan` / `codeformer` / `rrdbnet` | VQGAN/CodeFormer VQ autoencoder; CodeFormer face restoration; Real-ESRGAN super-resolution - the imaging pipeline's code/restore/upscale tail |
| `audio` / `mimi` / `ecapatdnn` / `qwen3tts` | wav/STFT/mel + 1D conv builders; Mimi codec; ECAPA-TDNN; Talker+MTP |
| `minimaxmusic3` | MiniMax Music 3 lyrics+caption → song: RVQ depth decoder, condition encoder, 36-layer flow-matching DiT, DAC-style vocoder (Global LLM is `crates/qwen3` verbatim) |
| `cosyvoice` / `s3tokenizer` / `campplus` | CosyVoice 2/3 zero-shot voice cloning; FSQ speech tokenizer; CAM++ x-vector |
| `atif` / `rl` | the training-from-trajectories pair: `atif` is a manual byte-for-byte mirror of sven's Agent Trajectory Interchange Format crate (re-sync by diffing against sven's `crates/atif`; brain adds no behaviour of its own), `rl` is `model::train::fit` lifted to reward-weighted batches over any `Model` that implements `enable_weighted_loss` (today only `qwen3`) |
| `qwen3asr` | Whisper-style + Nemotron 3.5 FastConformer streaming ASR |
| `qwen3omnimoe` / `qwen3vl` / `fastvlm` / `moondream3` | Qwen3-Omni-30B Thinker (multi-GPU resident); Qwen3-VL-4B; FastVLM-0.5B; Moondream 3 - see `docs/models/vlm.md` for the latter three |
| `deepseek2ocr` / `deepseek2` / `sam1` | DeepSeek-OCR: the composite (DeepEncoder + splice + decoder, `import`/`caps` incl. the served `generate`); its DeepSeek-V2-family MoE decoder; the SAM-1 ViT-B tower the DeepEncoder is built on |
| `forecast` / `fcbench` / `chronos2` / `kronos` / `fincast` | forecasting seam, backtester, three imported models |
| `wm-core` / `diamond` / `genieredux` / `wm-display` | world-model trait + fake model; DIAMOND; GenieRedux-G; SDL window |

### Deployment / IO

| Crate | Responsibility |
|---|---|
| `onnx` | pure-Rust ONNX graph model + serializer (export), plus the import side: a **reader** (`read`: initializers/nodes/attributes) and the coverage-checked topological **`walk`** both face crates import through; vendored `prost`, no `protoc` |
| `npu` | YOLOv8/ZipDepth → ONNX export + BN fold + brain-native INT8 PTQ + fake-quant simulator + OpenVINO **Intel NPU** runtime (`runtime-linking`) |
| `capture` | V4L2 webcam (hand-rolled ioctl FFI, YUYV→RGB, latest-frame slot) |
| `capability` / `residency` / `stats` / `server` / `dbus` / `runtime` / `events` / `hfsm` | the serving/runtime stack (table above) |

---

## Task → where to look

| Task | Where |
|---|---|
| Architecture & crate graph | `.agents/rules/architecture.md` |
| **Defects this repo has already paid for** (gates that lie, metrics that cannot see a bug, backend-specific silent-zero gradients) | **`.agents/rules/lessons.md`** - read before designing a gate |
| Testing strategy + gradient-check gate | `.agents/rules/testing.md` |
| **Porting a new model** (goldens → import → kernel contracts → parity ladder → training) | **`.agents/rules/porting.md`** - read BEFORE starting any port |
| Multi-GPU scaling (data / pipeline / tensor parallel) | `docs/scaling/*.md`; `crates/model/src/{distributed,parallel,collective,shard,plan,grid}.rs` |
| Performance: methodology (profiling, kernel selection, INT8, where numbers live) | `docs/performance/overview.md` |
| Performance: session-specific findings (what sped a given model up + why, with real numbers) | `.agents/roadmap/<model>.md` |
| **Performance benchmarking** (`brain perf`): design | `docs/performance/benchmarking.md`; `crates/perf`, `crates/cli/src/perf_cli.rs` |
| Perf regression gate (hard floors vs a committed baseline) | `brain perf gate`; `crates/perf/src/gate.rs` |
| Device capabilities (class/limits/numeric tiers, queried never assumed) | `backend_api::DeviceCaps`; filled per backend, `Gpu::caps()` |
| Canonical GPU registry / placement (`brain devices`, `Gpu::new_on`, `with_gpu`) | `docs/introduction/hardware.md`; `crates/gpu-core/src/devices.rs` |
| Kernel selection policy + autotuner (which variant runs, measured per device) | `backend_api::select` (`candidates`/`DefaultSelector`/`AutoTuner`), `gpu_core::tune`; `BRAIN_NO_AUTOTUNE=1` forces static |
| Roofline probe (compute/bandwidth ceiling used for "% of roof") | `gpu_core::roof`; bounded by `BRAIN_ROOF_BUDGET_S` (default 10s); off by default on the CPU device class, force-run there with `BRAIN_NO_ROOF=0` |
| GPU backend wait bound (Vulkan fence wait, wgpu `poll`) | `BRAIN_GPU_WAIT_S` (default 30s) - a wedged submit now panics with which call site timed out instead of hanging the process forever; see `.agents/rules/lessons.md` #38 |
| Kernel specialisation (one WGSL source, tunable constants) | `kernels::template` |
| Prompt-prefix cache (paged block reuse across requests) | `model::paged::PrefixCache`; adoption in `qwen3::serve::Engine::prefill` |
| Int8 serving weights + on-device decode window | `qwen3::serve` (`--weights-int8` / target suffix `:i8w`; `DECODE_WINDOW`) |
| Engine internals | `docs/engine/{overview,training,vulkan,web}.md` |
| **Profile a forward or a BACKWARD, per kernel kind** | `crates/sdxlunet/src/bin/unet_bench.rs` (forward, + a `gemm` mode that A/Bs kernels for correctness AND speed) and `crates/vqgan/src/bin/vqgan_bench.rs` (a full training step, both halves, + `gn`/`convbwd` A/B modes). Copy their shape; see `.agents/rules/kernels.md` §F.1 |
| **Add/adjust/dispatch a WGSL kernel** | **`.agents/rules/kernels.md`** - read BEFORE writing or dispatching one; then `crates/kernels/wgsl/*.wgsl` + **`make kernels-regen`** + **`make kernels-table`** |
| **Which kernels already exist** (before writing a new one) | the catalogue in **`docs/reference/kernels.md`** - every kernel with what it does, how, its structural optimisation level, and per-backend support |
| **Something is slow (model, kernel, training step)** | **`.agents/rules/kernels.md` §F** - the ORDERED loop that found the big wins (profile per kernel kind → check for an already-faster sibling → measure the branch your hardware skips → sweep for the crossover → fix it in the SELECTOR → mutation-verify → re-profile); then **§E** (measure-first rules + the killed hypotheses), `.agents/rules/porting.md` §10, case studies in `docs/performance/overview.md` |
| MoE toy task / honest eval methodology | `README.md` |
| Federated MoE pipeline (done vs remaining) | `docs/training/federated-experts.md`; `crates/federated/src/{shard,sha256}.rs` |
| GPT model / training / sampling | `crates/gpt2/src/{model,train,sample,init}.rs` |
| Qwen model / import / LoRA / INT8 / sharding | `crates/qwen3/src/{model,import,finetune,q8,shard,sample}.rs` |
| **Qwen concurrent serving (paged KV, continuous batching, spec decode)** | `crates/qwen3/src/serve.rs`, `crates/model/src/paged.rs`, `crates/cli/src/qwen_cli.rs` |
| Qwen3.5-35B-A3B model / import / LoRA / INT8 / sharding / vision splice | `crates/qwen35moe/src/{model,import,lora,q8,shard,vl}.rs`, `model::gdn` (shared Gated DeltaNet kernels), `.agents/roadmap/qwen35moe.md` |
| Qwen3.5-35B-A3B serving (`caps.rs`, resident, D-Bus/HTTP) | `crates/qwen35moe/src/{caps,serve}.rs`, `crates/cli/src/{qwen35moe_cli,resident_qwen35moe}.rs`, `examples/llm/` |
| Qwen3.8-27B dense model / import / LoRA / finetune / sharding / MTP / vision splice | `crates/qwen35/src/{model,import,finetune,shard,vl}.rs`, `model::gdn` (shared Gated DeltaNet kernels), `.agents/roadmap/qwen35.md` |
| Qwen3.8-27B serving (`caps.rs`, resident, D-Bus/HTTP) | `crates/qwen35/src/{caps,serve}.rs`, `crates/cli/src/{qwen35_cli,resident_qwen35}.rs` |
| Model residency / job scheduling | `crates/residency/src/{manager,scheduler,executor,budget,lru,place}.rs` |
| Capability manifests + generic dispatch (`brain caps` / `brain <arch> <verb>`) | `crates/capability/src/lib.rs`, `crates/cli/src/caps_cli.rs` |
| Deterministic weight-free mock `Provider` (synthetic image/mask/video/audio/text/bytes, for a `capability::Provider` consumer that must not load real weights) | `crates/capability-mock/src/lib.rs` |
| Served-model catalog (manifest + weight-free provider ctor per model, ~70 crates, in ONE list, no CLI dependency) | `crates/catalog/src/lib.rs`; the CLI-local residency-adapter extension over it lives in `crates/cli/src/catalog.rs` |
| **Captioning/labeling a dataset with any VLM** (the seam, not one model) | `crates/captioner/src/{lib,label}.rs` - `Captioner`/`Clip`/`Capabilities`; implementors in `crates/qwen3vl/src/captioner.rs` and `crates/fastvlm/src/captioner.rs`; verb in `crates/cli/src/label_cli.rs`; `docs/training/labeling.md` |
| JSONL transports (stdio / TCP / unix) | `crates/server/src/{transport,controller_session}.rs` |
| D-Bus control surface | `crates/dbus`, `examples/dbus` |
| **Stats snapshot / braintop contract** (add a metric, data-driven sections) | `crates/stats/src/{snapshot,source,build}.rs`; D-Bus `StatsSnapshot`/`StatsStream` in `crates/dbus/src/service.rs`; `Executor::residency` in `crates/residency/src/{executor,manager}.rs` |
| Event/HFSM controller (`brain serve --stdio`) | `crates/runtime/src/{lib,pump}.rs`, `crates/cli/src/run_cli.rs`, `crates/events/src/lib.rs` |
| GLM-5.2 (MLA + MoE + DSA indexer + MTP) | `docs/models/glmdsa.md`, `docs/models/glmdsa/npu.md`; `crates/glmdsa`, `crates/cli/src/glm_cli.rs` |
| LFM2.5-Encoder (bidir conv/attn hybrid, MLM, 8k) | `docs/models/lfm2/{readme,status}.md`; `crates/lfm2`, `crates/cli/src/lfm_cli.rs`; goldens via `tools/goldens/lfm2_dump_reference.py` |
| YOLO model / loss / inference | `crates/yolov8/src/{model,head,blocks,loss,assign,infer,nms,config}.rs`; `docs/models/yolov8/readme.md` |
| YOLO → Intel NPU (export/quantize/run/bench) | `crates/npu`, `crates/onnx`, `crates/cli/src/npu_cli.rs`, `docs/models/yolov8/npu.md` |
| ZipDepth: guide / ledger (incl. GPU perf root causes) | `docs/models/zipdepth/{readme,status}.md`; `crates/zipdepth/src/*`, `crates/cli/src/depth_cli.rs` |
| Face recognition (SCRFD + alignment + ArcFace) | `.agents/roadmap/scrfd.md`; `crates/scrfd/src/{config,import,model,detect}.rs` + `crates/arcface/src/{config,import,model,align,train}.rs`; goldens via `tools/goldens/{scrfd,arcface}_dump_reference.py` |
| **Read an ONNX file** (initializers, nodes, attributes) | `crates/onnx/src/read.rs` - the import front-end; `crates/onnx` is otherwise export-only |
| VQGAN / CodeFormer VQ autoencoder | `.agents/roadmap/vqgan.md`; `crates/vqgan/src/{config,import,model}.rs` over `crates/vae/src/blocks.rs`; goldens via `tools/goldens/codeformer_dump_reference.py` |
| FLUX.1 / Kontext (12 B MMDiT, per-block modulation, edit path) | `.agents/roadmap/flux1.md`; `crates/flux1/src/{config,import,model}.rs`; goldens via `tools/goldens/flux1_dump_reference.py` |
| T5-XXL encoder (FLUX.1 conditioning) | `.agents/roadmap/t5encoder.md`; `crates/t5encoder/src/{config,import,model,hostbias}.rs`; goldens via `tools/goldens/t5encoder_dump_reference.py` |
| **Which GEMM kernel a forward dispatches** (naive / skinny-M GEMV / 128×128 tiled), fp32 and int8 | `model::block::gemm_variant` - one rule, shared by flux1 and flux2; `block::pick_gemm` is the training-shaped sibling |
| CodeFormer face restoration (code Transformer + CFT + the `w` dial) | `.agents/roadmap/codeformer.md`; `crates/codeformer/src/{config,import,model}.rs` over `crates/vqgan` + `crates/vae`; goldens via `tools/goldens/codeformer_restore_dump_reference.py` |
| Real-ESRGAN super-resolution (the imaging pipeline's upscale tail) | `crates/rrdbnet/src/{config,import,model,caps}.rs` over `crates/vae/src/blocks.rs`; goldens via `tools/goldens/rrdbnet_dump_reference.py`; user-facing page `docs/models/rrdbnet.md` |
| SDXL UNet forward + the discrete samplers (DDIM/Euler/Euler-a/DPM++) | `.agents/roadmap/sdxlunet.md`; `crates/sdxlunet/src/{config,import,model,hostemb}.rs` over `crates/vae/src/blocks.rs`; `crates/diffusion/src/discrete.rs`; goldens via `tools/goldens/sdxlunet_dump_reference.py` |
| **Adding control conditioning to ANY diffusion backbone** (the named-injection-point seam, not an SDXL crate) | `crates/controlnet/src/adapter.rs` - `ControlAdapter`/`ControlSource`/`InjectionPoint`/`Residuals`; SDXL producer in `src/{config,import,model}.rs`; `.agents/roadmap/controlnet.md`; goldens via `tools/goldens/controlnet_dump_reference.py` |
| PuLID identity conditioning on FLUX.1 (IDFormer + the 20 cross-attention sites) | `.agents/roadmap/pulid.md`; `crates/pulid/src/{config,import,model,adapter}.rs`; the backbone seam is `crates/flux1/src/inject.rs`; goldens via `tools/goldens/pulid_dump_reference.py` |
| **Upload a host `&[f32]` to a device buffer** | `Gpu::write_f32` (`crates/gpu-core`) - the `&[f32]` half of `Gpu::write`/`read`; never re-derive `to_bits().collect()` at a call site |
| CLIP-L / OpenCLIP-bigG / EVA-CLIP text+image towers | `crates/clip/src/{config,import,model}.rs`; goldens via `tools/goldens/clip_dump_reference.py`; user-facing page `docs/models/clip.md` |
| ZipDepth → Intel NPU (fp32 ONNX, exact parity) | `npu::depth_topology`, `crates/zipdepth/src/fuse.rs` |
| SAM 2.1 promptable segmentation (image path) | `crates/sam2/src/{config,import,model,hostpe}.rs`; goldens via `tools/goldens/sam2_dump_reference.py`; user-facing page `docs/models/sam2.md` |
| DeepSeek-OCR (document image -> text/markdown) | `.agents/roadmap/deepseek2ocr.md`; `crates/deepseek2ocr/src/{config,encoder,layout,model,preprocess,prompt,rows,import,caps}.rs` over `crates/{sam1,clip,deepseek2,gguf}`; resident `crates/cli/src/resident_deepseekocr.rs`; goldens via `tools/goldens/deepseek_ocr_dump_reference.py`; user-facing page `docs/models/deepseek2ocr.md` |
| WorldMirror-2 (photos → 3DGS scene) | `docs/models/worldmirror2/{readme,status}.md`; `crates/worldmirror2`, `crates/cli/src/mirror_cli.rs` |
| 3D Gaussian Splatting rasterizer + viewer + fit | `docs/models/splat/{readme,status}.md`; `crates/splat`, `crates/cli/src/splat_cli.rs` |
| Shared ViT block builder (DINOv2/trunk/camera-head) | `crates/model/src/vit.rs` |
| Fused conv eval paths (act selector, register tiling, grouped) | `crates/vision/src/blocks.rs`, `crates/kernels/wgsl/conv_act*.wgsl`, `conv2d_gd_reg.wgsl`, `crates/backend-cpu/src/fast_conv.rs` |
| Detection metrics (mAP/precision/recall) | `crates/eval/src/detection.rs` |
| Synthetic detection dataset (RGB shapes + GT boxes) | `crates/data/src/gen_detect.rs` |
| Datasets & tokenizers | `crates/data/src/{prepare,gen_*,tokenizer,bpe,clip_bpe,qwen_tokenizer,loader,binio,rng}.rs` |
| TTS: guide / acceleration | `docs/models/qwen3tts/{readme,acceleration}.md`; `crates/{qwen3tts,mimi,ecapatdnn,audio}`, `crates/cli/src/{tts_cli,tts_serve}.rs` |
| **ASR (speech-to-text)**: status / serving / perf | `.agents/roadmap/asr.md`; `crates/{nemotronasr,qwen3asr}`, shared `audio::asr_caps`, `crates/cli/src/resident_asr.rs`, D-Bus `StreamTranscribe` (`crates/dbus`), `examples/asr/` |
| Forecasting models + backtester | `docs/models/{chronos2,kronos,fincast}/status.md`; `crates/{forecast,fcbench,chronos2,kronos,fincast}`, `crates/cli/src/forecast_cli.rs` |
| World models (playable) | `docs/models/world-models/{status,playbooks,fixtures}.md` + `specs/`; `crates/{wm-core,wm-display,diamond,genieredux}`, `crates/cli/src/wm_cli.rs` |
| Z-Image / diffusion stack | `docs/models/s3dit/{readme,status}.md`; `crates/{s3dit,dit,diffusion,vae}` |
| FLUX.2 Klein: guide / ledger | `docs/models/flux2/{readme,status}.md`; `crates/flux2`, `crates/cli/src/flux2_cli.rs`; goldens via `tools/goldens/flux2_dump_reference.py` |
| **Video generation (Wan)**: guide / roadmap + perf baseline | `docs/models/wan.md`, `.agents/roadmap/wan.md`; `crates/wan`, `crates/cli/src/{wan_cli,resident_wan}.rs`, `crates/wan/src/bin/wan_bench.rs`, `examples/videogen/`; goldens via `tools/goldens/wan_{dit,vae,t5,schedule}_dump_reference.py` |
| Finetuning guides | `docs/guides/finetune/{plan,datasets}.md` |
| **"change only X" end to end** (segment -> refine -> restore -> composite) | `crates/imgpipe` - the bit-exactness contract and why it holds is in its module docs |
| Image handling of ANY kind (resize/pad/crop/letterbox/masks/tiling/codecs) | `crates/imaging` - check here BEFORE writing a pixel loop; five copies of `chw_to_hwc` is what created it |
| Identity conditioning (ArcFace -> ID tokens -> diffusion attention) | `crates/pulid` (FLUX.1, wired), `crates/instantid` (SDXL, shapes only); `pulid::idcond` documents the raw-vs-normalised asymmetry that silently breaks it |
| Clippy gate (exit code + a warning ratchet) | `make clippy`, `scripts/gates/clippy-gate.sh` - clippy ABORTS on a denied lint and then reports nothing, so always check the exit code |
| CLI subcommands | `crates/cli/src/{main,args,*_cli}.rs` |
| **Tracing/observability** (`--trace-<family> <0-5>`, adding a family, instrumenting a crate) | `crates/trace` - the family registry is `crates/trace/src/registry.rs`; the CLI wiring is `install_tracing` in `crates/cli/src/main.rs` |
| **Quantize any checkpoint to a GGUF** (tier, per-tensor policy, streaming write, two-way coverage) | `crates/checkpoint/src/quantize.rs` (`Tier`/`Policy`/`plan`/`convert`), `checkpoint::quant::quantize_par`, `checkpoint::gguf_write::Writer`; CLI `brain quantize` in `crates/cli/src/quantize_cli.rs` |

---

## Essential commands

**Always build through the Makefile, never `cargo` directly:** `make build`
(debug), `make release` (optimized), `make test` (suite). They wrap cargo with
the project's expected flags/targets, and - critically - all three share the
same `./target` dir, so a `make build` after a `make release` (or vice versa)
reuses the other's downloaded/compiled dependency graph instead of a cold
rebuild. Interleaving raw `cargo build -p <crate>` calls (or worse, a
one-off `CARGO_HOME` override on just that call) does not add a second cache -
it just adds an extra, redundant compile pass against the same `./target`, and
if the `CARGO_HOME` differs from the shell's own default it can even pull a
second copy of the registry. Do not override `CARGO_HOME` per-command; if a
build fails with a registry/permission error under the shell's default
`CARGO_HOME`, fix that env var's value once (for the session/shell), not
per-invocation.

```bash
make build                           # debug build
make release && make test            # optimized build + full suite (MOE_SKIP_GPU_TESTS=1 to skip GPU;
                                     # tests run at TEST_THREADS=8 on the pooled test device - every
                                     # test binary shares one device via gpu_core::testgpu)
make gradcheck                       # backprop correctness gate
make parity                          # cross-backend parity: CPU == Vulkan == NPU (scripts/gates/parity-gate.sh)
make kernels-regen                   # regenerate the kernel const block after adding/removing a .wgsl
make kernels-table                   # regenerate README.md's kernel catalogue (gated by kernels-table/check)
make docs                            # docs bundle -> build/docs/brain-docs.{md,pdf} (needs pandoc + xelatex)

make data/<name>                     # calculator|reverser|wordcalc|timeseries|shakespeare_char|gpt|detect|tts
make train/gpt/<name>                # train GPT -> out/gpt-<name>.safetensors
make eval/gpt/<name>                 # perplexity + exact-match
make train/yolo | eval/yolo | detect/yolo
make depth/demo | depth/smoke | depth/camera | train/zipdepth
make mirror/import | mirror/infer | mirror/demo | splat/view
make wm/play | wm-fixtures           # world models (SDL window; fixtures need torch)
make forecast/compare | forecast/serve
make export/yolo-onnx | quantize/yolo | sim/yolo-int8 | run/yolo-npu | bench/yolo-npu
make federated-demo                  # MoE train -> split -> verify -> merge
make web/dev                         # WebGPU browser demo (crates/web)
make bench                           # architecture-evaluation suite (see below)
```

Direct binary - the model is selected by the command:

```bash
./target/release/brain <verb> <arch> [opts]      # or: brain <arch> <verb> [opts] - same command
# infra verbs: data devices npu federated bench perf forecast caps serve gradcheck flops
# archs with their own dedicated CLI module: gpt2 qwen3 qwen35moe qwen35 glmdsa lfm2 qwen3tts
#   yolov8 zipdepth flux2 worldmirror2 splat qwen3omnimoe diamond toypid toymoe
# every other arch (`brain caps` lists them all) is reached the same way, its
# verb being the exact capability action name, e.g. `brain scrfd detect`
# toymoe train | infer | eval        (the bare sparse-MoE toy model)
```

**GGUF import is generic.** `brain import FILE [--out PATH] [--id NAME]`
picks the importer from the file's own `general.architecture` via the registry
in `crates/cli/src/gguf_import.rs`; `--list` prints what's registered. Adding an
architecture means implementing `GgufArchitectureImporter` and adding one line
to that table - never a new per-model subcommand. The model-dir scan does NOT
convert on its own (fp32 dequant-on-load makes the output far larger than the
quantized source); it logs the exact command instead. That module's doc holds
the full reasoning.

**Quantization is generic too, and needs no registry at all.** `brain quantize
SRC --out PATH [--tier Q8_0] [--keep SUBSTR] [--plan]` is `import`'s export
sibling (`crates/checkpoint/src/quantize.rs`, driven by
`crates/cli/src/quantize_cli.rs`): it reads any `checkpoint::TensorSource`
with a manifest (a safetensors file, an HF-style directory, an existing GGUF)
and writes a quantized GGUF, streaming so peak host memory is one tensor
rather than the whole output. A tensor is quantized iff it is rank 2 with a
block-aligned fastest-varying dimension - both structural facts about the
block format, not heuristics. What a given architecture must NEVER quantize
regardless of shape (modulation tables, conditioning projections) is a
`--keep` substring list supplied by the caller, because no shape implies it;
`ltxv::int8::is_never_quantized` is the worked example that generalizes.
Every source tensor is accounted for in the output with a typed reason, which
`--plan` prints without writing anything.

**Device selection** - `--device` declares **which compute is schedulable**, not
"a backend". Omit it and brain uses *every* device present (all GPUs + CPU +
NPU), scheduling models across them. `BRAIN_DEVICE` does the same without a flag.
Parsing/resolution live in `crates/gpu-core/src/devices.rs`.

| value | schedulable compute |
|---|---|
| *(omitted)* | every device on the machine, together |
| `cpu` | CPU only, all cores |
| `gpu` | every GPU, nothing else |
| `npu` | NPU only |
| `vulkan` | every GPU, via the native-Vulkan backend |
| `gpu,cpu` | GPUs and CPU together (comma-separated = union) |
| `gpu0` / `gpu0,gpu1` | those physical cards only |
| `cpu21` | that one core |
| `cpu0-7` | cores 0..=7 (inclusive range) |
| `gpu1,cpu0-3` | one card plus four cores |

An indexed CPU selection **pins process affinity** (`sched_setaffinity`) and
sizes the rayon pool to match, so `cpu21` is genuinely one core. GPU indices are
**canonical**: the process-wide device registry (`gpu_core::devices`) enumerates
physical cards once with stable identity (PCI bus id → Vulkan deviceUUID →
vendor:device+ordinal) and orders them by PCI bus id, so `gpu0`/`gpu1` name the
same physical cards everywhere - `--device`, `Shard.gpu_index`,
`residency::Device::Gpu(i)` - and nvidia-smi order maps to them via PCI, not by
assumption. `brain devices` prints the table. Placement is explicit
(`Gpu::new_on`, scoped `devices::with_gpu`) - never env mutation;
`BRAIN_GPU_INDEX` remains user *input* only, parsed once at first registry use.
Out-of-range indices are errors, never silent clamps. See
`docs/introduction/hardware.md`.

This bounds where work **executes** - host RAM and disk stay available as
cache/spill tiers, so `--device gpu` still uses RAM for weight caching.

```bash
./target/release/brain gpt2 train data/calculator --device cpu --out out/gpt.safetensors
./target/release/brain perf run sweep --device gpu0 --target qwen-synth:12x768x12
BRAIN_DEVICE=cpu make test            # whole suite on CPU, no GPU needed
```

**Tracing** - `--trace-<family> <0-5>` is a GLOBAL option like `--device`,
valid on any subcommand, stripped from the args before dispatch. It is
`tracing` + `tracing-subscriber` used as intended: library crates emit through
the plain facade (`tracing::debug!`, `#[tracing::instrument]`) and their
`target` - the emitting Rust module path - is what labels each line with the
component it came from. `crates/trace` owns the ONE subscriber install and the
*family registry* that maps a short name onto the crates it covers.

| flag | meaning |
|---|---|
| `--trace-gpu N` / `--trace-ltxv N` | one registered family at level N |
| `--trace <family>=<level>` | the generic form; repeatable, needs no dedicated flag |
| `--trace-format text\|json` | how to render (default `text`) |
| `--trace-output -\|PATH` | where to write (default `-`, stdout) |
| `BRAIN_TRACE=ltxv=5,gpu=3` | the same levels without a flag; any flag overrides it |

Levels are `tracing`'s own five plus off: 0 off, 1 error, 2 warn, 3 info,
4 debug, 5 trace. **Adding a family is ONE entry in `brain_trace::FAMILIES`**
(short name -> the crate lib names it covers) plus the matching line in the
CLI help; a `crates/cli` test fails if those two ever disagree, and nothing
in the filter-construction code changes. Instrumenting a crate means adding
`tracing` (the facade only, never a subscriber) to its manifest and calling
the macros - no per-crate flag, no registration.

This is layered ON TOP of `BRAIN_PROFILE`/`gpu_core::profile::stage_time`,
which stays as it is: perf gates elsewhere parse its stage totals. An
instrumented function may therefore report the same stage timing through both
mechanisms; that overlap is deliberate, and consolidating them is a later
decision rather than a side effect of instrumenting a crate.

```bash
brain --trace-ltxv 5 --trace-format json --trace-output run.jsonl ltxv t2v --prompt "..."
brain --trace-gpu 4 devices          # device registry + adapter enumeration
```

Event/stdio controller - an HFSM (`crates/runtime`) reads JSONL events on stdin
and emits JSONL on stdout. `--gpt`/`--yolo` load real models (or `BRAIN_GPT2`/
`BRAIN_YOLOV8`); with neither, fake echo/detector models keep the loop usable:

```bash
printf '{"event":"user_text","text":"hi"}\n' | ./target/release/brain serve --stdio
```

---

## Benchmark suite (`crates/bench`)

`brain-bench` is a **model-agnostic** architecture-evaluation layer: each
benchmark owns its *dataset* and its *scoring*, the harness owns running it. Use
it to answer "does this architecture actually learn task X?" the same way across
tasks. See `crates/bench/README.md` for the full design.

**Run** (this box has two real Tesla P40s - `--device gpu0` selects one; a
GPU-less box still serves `--device gpu` through the llvmpipe software
rasteriser, and such runs must never be reported as GPU numbers):

```bash
BRAIN_DEVICE=cpu make bench          # every registered benchmark, one table
BRAIN_DEVICE=cpu make bench/mqar     # a single benchmark
./target/release/brain bench [--device cpu] [<name>] [--seed S]
```

**Registered benchmarks:** `mqar` (multi-query associative recall - the
reference), `toolcall` (map a user intent to one structured `TOOL_k args…` call,
scored exact-match on the assistant span only), the MAD family (`mad_recall`,
`mad_fuzzy_recall`, `mad_noisy_recall`, `mad_selective_copy`, `mad_memorize`,
`mad_compress`), and the formal-language / state-tracking probes `parity`,
`mod_add` (grokking), and `dyck` (Dyck-k brackets).

**Add a benchmark:** new module in `crates/bench/src/<name>.rs` implementing
`Benchmark` (`name`/`description`/`prepare`/`evaluate`/`threshold`) → register in
`bench::registry()` (and `registry_smoke()`) → add a learnability test in
`crates/bench/tests/` gated by `MOE_SKIP_GPU_TESTS` asserting a **measured**
threshold. The generic `make bench/%` rule picks it up with no further wiring.

### Evaluating a new architecture (turn-key harness)

The battery is architecture-agnostic via the `DecoderLm` seam, so the same
benchmarks score *any* architecture and results are directly comparable:

1. **Implement `DecoderLm`** (`train_decoder` + `load_scorer`, plus a `Scorer`).
2. **Add one line to `arch_registry()`** (`crates/bench/src/arch.rs`).
   Registered today: `gpt2`, `gpt2-small`, `gpt2-wide`, `toymoe`, `qwen3`, `glm`.
3. **Run + compare:**
   ```bash
   BRAIN_DEVICE=cpu make bench/eval ARCH=<name>   # -> results/<arch>-<seed>.json
   BRAIN_DEVICE=cpu make bench/compare            # leaderboard over results/*.json
   ```

**Capability axes** (`crates/bench/src/axes.rs`) group benchmarks into a profile -
`recall`, `copying`, `memory`, `state_tracking`, `compression`, `arithmetic`
(*informational*) - each scored as the mean of its benchmarks. `eval` writes a
JSON artifact (arch, size, params, commit, seed, per-benchmark + per-axis
results, gating pass-rate); `compare` diffs ≥2 side-by-side. `results/` is
git-ignored.

> Non-GPT caveat: `mad_compress` is a bottleneck autoencoder (MSE head), not a
> next-token decoder, so it ignores the supplied `DecoderLm` - its `compression`
> score does not yet reflect a candidate architecture.

### Predictive per-capability scaling + tuning advisor

`eval` says where an arch stands; **`scale`** predicts how each capability
improves as the model grows, and **`advise`** says what to tune.

- **`brain bench scale --arch <name>`** (`capscale.rs`): sweeps a SIZE grid
  (`L1xD32xH2 → L2xD64xH4 → L3xD96xH6`) and, per axis, trains+scores one
  representative benchmark at each size. Fits a saturating trend
  `score(N) ≈ ceil − A·N^(−β)`, records slope-per-doubling, β, R², predicted
  score@2x/@4x, and a verdict ∈ {improving, saturating, flat} →
  `results/scale-<arch>-<seed>.json`. The *shape* of the curve is the
  deliverable, not absolute scores.
- **Experts knob (future MoE sweeps):** the sweep dimension is a generic `Knob`
  enum; only `Knob::Size` is wired. Fill the `// TODO(experts)` branch in
  `capscale::grid_for` - the fit/advisor are dimension-agnostic.
- **`brain bench advise <eval.json> [<scale.json>]`** (`advisor.rs`): lever =
  headroom (1−score, gated axes) × size-slope; per-axis signal → action (rising
  slope → *increase size*; flat → *change the mechanism*; low `train_ce` + low
  eval → *more data/reg/steps*; ≈ceiling → *deprioritize*), each carrying
  score-per-Mparam. **`brain bench eval` prints the top-3 as a footer.**
  `make bench/scale ARCH=<name>` + `make bench/advise ARCH=<name>`.

`make bench/scaling` runs the multi-size scaling-law sweep (`L(N)=E+A·N^-alpha`);
`make bench/char` keeps the legacy GPT-on-char-datasets sweep.

---

## Performance benchmarking (`crates/perf`)

`brain perf` is the sibling of `brain bench` and answers a different question:
**how much correct work does brain deliver per unit of hardware, memory, energy
and time?** Full design in `docs/performance/benchmarking.md`.

```bash
brain perf list                       # scenarios + the standard workload matrix
make perf                             # latency + throughput + serve + sweep
make perf/sweep                       # one scenario
make perf/compare                     # leaderboard over results/perf-*.json
make perf/smoke                       # CI-sized run of everything
```

It is **model-agnostic by construction**: rather than counting tokens, it
measures *artifacts arriving over time* along `submit → admit → first → … →
done`. That specialises to TTFT/ITL/TPOT for a decoder and collapses to a single
latency for a one-shot model (detect, depth, a forecast). `capability::Action`'s
existing `Progress` callback supplies the timeline, so **any model implementing
`capability::Provider` is benchmarkable with no new benchmark code** - the reason
adopting that seam is worth doing per model.

Rules the harness enforces (each exists because violating it produces a
flattering-but-wrong number):

- warm-up requests never enter a statistic; failed/unfinished requests are never
  goodput and never leave the denominator;
- unmeasured fields serialise as `null`, never `0`, and an ungated run reports
  `correctness.passed: null` - never `true`;
- **goodput** (output meeting the SLO) is the comparison metric, not peak rate;
- `compare` refuses to rank across artifact units, excludes `valid: false` runs,
  and warns on every environment/workload axis that differs;
- a software rasteriser is labelled as one everywhere it appears.

**All 14 scenarios are implemented**: Tier 1 (`latency`, `throughput`, `serve`,
`sweep`, `startup`) plus Tier 2 - `mixed`, `overload`, `cancel`, `kvcache`,
`residency` ★, `placement`, `frontend`, `faults`, `soak` - the ones where brain
has something to measure that a single-model, single-GPU, HTTP-shaped harness
structurally cannot. Cross-cutting: `fidelity` (correctness gate) and `energy`.

Each scenario states what it *cannot* see: where a metric needs an engine
capability that does not exist (a pluggable admission policy, prefix caching, a
pipeline cache, a multi-rank harness), the field is `null` and the artifact
carries a `notes` string explaining why - read that field rather than assuming
a metric that isn't there was simply forgotten.

## Conventions & invariants

- **Write down what you learned, in the same change.** When you find a
  non-obvious defect - something that was silently wrong, or a gate that was
  green without running - add it to **`.agents/rules/lessons.md`** as part of the commit
  that fixes it, with the number that proved it. Not later, not in a follow-up:
  the reason is fresh exactly once, and every entry in that file is there because
  it cost someone a day.

  Where it goes:
  | what you learned | where it belongs |
  |---|---|
  | a cross-cutting defect class (a gate that lies, a metric that cannot see X) | `.agents/rules/lessons.md` |
  | a kernel-authoring or optimisation rule, incl. a killed hypothesis | `.agents/rules/kernels.md` |
  | a step in porting a model that was not obvious | `.agents/rules/porting.md` |
  | a measured number about ONE model | the test that asserts it (a parity/gradcheck test's own assertion is the durable record - a number nothing checks is a number that silently goes stale) |
  | work still outstanding on ONE model | `.agents/roadmap/<model>.md` |

  A commit message is not a home for a finding: nobody greps commit messages.
  If a lesson only exists in one, it will be relearned.

  **`docs/` is user-facing product documentation, not a workspace.** It is what
  gets published - it explains how to use brain, not how brain was built or
  what's left to do. Never write a status ledger, a plan, a measurement log, an
  audit result, or a "known gaps" section there. Internal findings, rules, and
  per-model roadmaps go in `.agents/` as the table above describes; a `docs/`
  page only changes when the public contract (a command, a flag, an env var, a
  model's supported capabilities) actually changes.

- **Zero compile warnings. Always.** A build that emits warnings is not done.
  Fix every warning the build reports - **including ones your change did not
  cause**. "Pre-existing" is not an exemption: warnings are only ever pre-existing
  because someone before you applied that exemption, and a noisy build is how a
  real defect hides in the scroll-back. Fix them **properly** - delete the dead
  code, use the unused binding, remove the stale `mut`, handle the ignored
  `Result`. Silencing with `#[allow(...)]`, `let _ =`, or an `_`-prefixed name is
  acceptable ONLY when the construct is genuinely intentional, and then it carries
  a comment saying why. Never suppress a warning to make a build quiet.

- **One implementation. Never re-implement anything that already exists in this
  workspace - no matter what it is.** This is the rule that most needs enforcing:
  before writing a function, search for it. `rmsnorm` once existed **seven**
  times (one WGSL kernel plus six host copies in `kronos`, `qwen3tts`, `chronos2`,
  `fincast`, `s3dit` and `mimi`), `rope` three times, `silu` four times. Every
  copy is a place the epsilon, the RoPE layout or the reduction order can drift
  from the kernel that is supposed to be authoritative, and nothing compares
  copies against each other.

  | need | where it belongs |
  |---|---|
  | math that runs on a device | a WGSL kernel in `crates/kernels/wgsl/`, dispatched via `gpu_core` |
  | math that genuinely runs on the host | **`model::hostmath`** - and nowhere else |
  | CPU-parallel execution (rayon) | `backend_cpu::par` only - the on-CPU scheduler's primitives; no other crate may depend on rayon |
  | deterministic filler in a test or fixture | **`data::rng::Lcg`** (`signed`/`unit`/`scaled` + the `vec*` forms). `data::rng::Rng` is SplitMix64 and defines the on-disk datasets - its stream must not move, so it is *not* the test PRNG. The copied `(s >> 33)/2^31 − 1.0` helper was one-sided (`[-1,0)`), so no test ever fed a positive value to an activation kernel; see `.agents/rules/testing.md` §0 |
  | shared model blocks | `model::block`, `model::vit` |
  | uploading a host `&[f32]` to a device buffer | **`Gpu::write_f32`** - the `&[f32]` sibling of `Gpu::write`/`Gpu::read`. It was missing, so this had congealed into two byte-identical private `fn write` helpers (`sdxlunet::model`, `controlnet::model`) and ~20 inline `to_bits().collect()` sites in 10 crates |
  | ONNX graph emission (DSL + shared norm/silu emitters) | `crates/npu/src/topo.rs` (`TopoBase`); model-specific graphs stay in `crates/npu/src/*_topology.rs` |

  Do **not** wrap a shared function in a local alias "for readability"
  (`fn silu(x) { hostmath::silu(x) }`). A local name is how a shared function
  becomes a private copy at the next edit. Call it directly.

  Two narrow exceptions, both of which must say why in a comment:
  1. a **gradcheck oracle** may re-derive the math independently (usually in
     `f64`) - an oracle that shares code with the thing it checks proves
     nothing (`s3dit::grad`);
  2. a **backend fast path** implements an op for its device and is validated
     against the WGSL reference (`backend-cpu::fast_ops`).

- **One GPU device per process.** Building a `Gpu` per model object deadlocks
  the driver under concurrency and a device leaked into process exit crashes
  it. Production code shares explicitly (`Gpu::share` for the same kernel set,
  `Gpu::new_like` for a different set on the same device, `share_or_new` when
  the backend may not support sharing); **test binaries use
  `gpu_core::testgpu::dev(KERNELS)`** - a weak pool whose device dies with its
  last in-process handle. Never write a per-crate fixture; that is how
  duplicate fixtures (and the crash) come back.

- **Host math does not run on the accelerator.** Anything in `model::hostmath`
  is invisible to `--device`: it will not use the GPU, Vulkan or the NPU
  whatever the user asked for, and a benchmark of such a path reports host
  numbers under a device label. Host math is for `m=1` decode steps, references
  and glue - never for a hot path. If it is hot, it needs a kernel.


- **WGSL is the source of truth.** Kernels live only in `crates/kernels/wgsl/`,
  embedded as consts; no kernel text is duplicated. After adding/removing a
  `.wgsl`, run **`make kernels-regen`** (`scripts/build/kernels-regen.sh`) to
  regenerate the const block + `ALL` registry in `crates/kernels/src/lib.rs`,
  **and `make kernels-table`** to regenerate README.md's kernel catalogue.

  **Keep the catalogue current - it is the list §F.3 tells you to check before
  writing a kernel, so a stale one causes the exact defect this repo pays for
  most (a fast sibling a later model never learned about).** Every row comes
  from a block the kernel DECLARES in its own header, so **edit the kernel, never
  the table**:

  ```wgsl
  // @what  Register-tiled matmul (out = x @ Wᵀ), ...
  // @how   register block per thread, 256-thread workgroup tile, 3 barriers
  // @opt   5      // 1-5, structural: see the README legend
  // @cpu   native-only   // yes | no | native | native-only
  // @gpu   yes-wg256     // yes | yes-wg256 | no
  // @npu   yes           // can crates/npu's ONNX DSL emit an equivalent op?
  // @quant none          // int8 | none
  ```

  A NEW kernel gets its block seeded by `scripts/build/seed-kernel-meta.py`
  (idempotent - it skips kernels that already have one); refine it by hand after.
  Changing a kernel's *structure* - adding a barrier, a register tile, a
  workgroup stage - changes its row, so update the block and regenerate after
  edits too, not only after adding or deleting a file.

  This is not optional bookkeeping. `make kernels-table/check` fails the build
  (via `test/full`) when a kernel is missing a field, when the table has drifted,
  **and when a declaration contradicts the code** - `@cpu` is cross-checked
  against the barrier count, `@gpu` against `@workgroup_size`, `@quant` against
  `dot4I8Packed`, `@opt 5` against the presence of a register block. A comment
  that claims a property the code lacks is not harmless: `dw_splitk_reduce`'s
  header asserted it compiled on `backend-cpu` while it did not, and that false
  claim is why a red `compile_all` read as noise for months instead of as the
  2D-grid correctness bug it was.
- **fp32 arithmetic only, core compute only** - single bind group, **≤8 storage
  buffers/kernel** (the WebGPU guarantee; the splat backward kernels bind 8),
  **no atomics, no subgroups, no f16** (the only mentions of those in the kernel
  tree are comments asserting their absence).
  *This is a rule about the arithmetic datatype, NOT about storage precision -
  do not read it as "brain is fp32-only".* brain has a full **INT8** path:
  per-channel symmetric weights packed 4-per-`u32` (`model::int8`), DP4A GEMMs
  (`matmul_i8`, `matmul_i8_dyn`, `matmul_i8_gemv`, ~4× the fp32 rate on Pascal),
  dynamic per-token activation scales (`max_abs_row` → `quant_pack`), and int8
  paged KV. Norms/RoPE/attention stay fp32. Quantizing is the FIRST tool for
  fitting a large model on a card (`s3dit::int8` for a DiT, `qwen3::q8` for an
  encoder: ~16 GB → ~4.8 GB), ahead of sharding. `@workgroup_size(64)` is the rule;
  the register-tiled matmuls (`matmul_reg*.wgsl`, `matmul_dw_reg.wgsl`,
  `matmul_dx_reg.wgsl`, `matmul_i8*.wgsl`) and `flash_attn_bidir_split.wgsl`
  use 256 - every one of them because a thread cooperates over a tile, and each
  must be gated on the device's **queried** `DeviceCaps::max_workgroup_size`
  (256 is the WebGPU floor, so a 64-thread fallback stays selectable).
  This is what keeps the engine portable to old GPUs and WebGPU.
- **Never put a large `var<function>` array behind a runtime loop bound.** WGSL
  function-scope arrays only become registers if the compiler can unroll every
  index; bound the loop by a `Params` field and the array lands in *local*
  memory (global-backed), and the kernel silently runs at memory bandwidth.
  This cost the FLUX.2 DiT 81 % of its forward - see
  `docs/performance/overview.md` for the pattern and the fix.
- **One thread per row is a COALESCING bug, at every row count.** A per-element
  norm/reduction kernel that gives thread *t* row *t* makes a warp's 32 loads
  `d` floats apart, so each 32-byte sector fetched serves ONE useful float - 8×
  read and write amplification that more rows do not fix (the loss is
  per-access efficiency, not thread count). The cooperative `*_rows` family
  (`rmsnorm_rows`, `softmax_rows`, and now `layernorm_rows` / `ln_stats_rows` /
  `layernorm_dx_rows`) walks one row with a 64-thread workgroup and is coalesced
  by construction: measured **19.4×** for QK-norm and **2.3–9.1×** for the
  LayerNorm family on a P40. `backend_api::select` (`Op::RmsNorm`,
  `Op::LayerNorm`) picks them wherever the queried
  `DeviceCaps::workgroup_reductions` holds, and `model::block`'s
  `layernorm_fwd` / `ln_stats_fwd` / `layernorm_dx_bwd` are the dispatch seam.
  Each carries exactly **one** top-level `workgroupBarrier()` - the CPU JIT
  splits a body at one barrier and no more, which is why they use a *shifted*
  single-pass mean/variance instead of the textbook two-pass.
- **Three backends, one build, one API.** `gpu-core` exposes a single
  `Gpu`/`DeviceBuffer`/`Step` surface; every model is written once against it.
  The accelerator is the *only* thing abstracted - there is no per-backend model
  code. `backend-wgpu` (default), `backend-cpu`, and `backend-vulkan` all compile
  into every native build and are selected at runtime (`--device` / `BRAIN_DEVICE`).
  The CPU backend reuses the **same WGSL** via the `wgsl-cpu` Cranelift JIT. On
  wasm only wgpu/WebGPU exists. `crates/vulkan` (coopmat) is excluded from
  `default-members`; `crates/web` is empty off wasm32.
- **The Intel NPU is NOT a `gpu-core` backend.** OpenVINO is a *whole-graph*
  compiler, so `--device npu` is a separate export→quantize→compile→run path
  (`crates/npu`), not a per-op backend. The default build stays free of OpenVINO
  at the source level; the runtime is loaded at run time (`runtime-linking`), so
  `make build`/`make test` stay green with no OpenVINO installed.
- **Weight-only quantization must be block/group-wise, never whole-channel-only.**
  A scale spanning an entire output channel (`inp` up to 3072 in these models)
  lets one outlier weight set the quantization step for every other weight
  sharing that channel -- GGUF's own block formats (`Q4_0`/`Q8_0`) use
  **32-element blocks** for exactly this reason, and K-quants (`Q4_K`/`Q5_K`)
  go further with per-role mixed precision. This was a *real, measured* defect,
  not a theoretical one: whole-channel INT8 on the Qwen3 KV-cache decode graph
  measured **cosine 0.994 / max_abs 11.2** against fp32 (same driver, same
  tokens) -- vs **1.000000 / 0.0002** at fp32. The fix
  (`crates/npu/src/topo.rs::linear_quant`, `QUANT_GROUP = 32`) dequantizes
  via `Cast`→`Reshape`→`Mul`→`Reshape` (per-`(group, out)` scales) rather than a
  single `DequantizeLinear(axis=1)` call, because ONNX's native block-quant
  needs opset 21's `block_size` attribute, whose OpenVINO-NPU op-coverage isn't
  established -- the group is applied by hand with ops already proven on that
  plugin. **Every NPU topology's quantized linear must go through this one
  shared emitter** (`crate::topo::linear_quant`) -- three files (`glm_topology`,
  `kronos_topology`, `fincast_topology`) had drifted into their own
  byte-identical, whole-channel-only copies before this was caught; a new
  per-model copy is the same mistake again. This applies equally to
  `model::int8::quantize_weight` (the GPU-serving INT8 path shared by
  `qwen3::q8`, `s3dit`, `flux1`/`flux2`) if/when it is audited for the same
  granularity.
- **Kernels follow `.agents/rules/kernels.md`** - before writing one, check for
  an existing fast sibling (`_rows`/`_wg`/`_reg*`/`_tiled`) and put the fix in
  *selection*, not a new copy: the single most expensive defect class here is a
  fast kernel a later model never learned about (`gn_stats`, fixed in 2025,
  re-cost 159× in `vae`). Before dispatching one, read its `Params` struct and
  copy a working call site - a mismatched param list is silently wrong, not a
  crash (`silu_mul` → cosine 0.504). Before optimizing, profile per kernel-kind
  and publish the table: every confident hypothesis on this engine has been
  wrong, and the profile has been right.
- **New model ports follow `.agents/rules/porting.md`** - reference goldens
  dumped FIRST (transformer I/O captured via forward hooks, replayed in the
  parity test), two-way import coverage, kernel Params read before dispatch,
  tiny-config smoke with step bisection, then the parity ladder
  (stage → forward → composed loop → real run). It encodes the exact failure
  modes already paid for; do not rediscover them.
- **Backprop is gated by `gradcheck`** (finite differences) - run it after any
  fwd/bwd math change. Entry points today: `check_gpt`, `check_qwen`,
  `check_qwen_lora`, `check_qwen35moe`, `check_qwen35moe_lora`, `check_qwen35`,
  `check_qwen35_lora`, `check_qwen35_mtp`, `check_moe`,
  `check_glm`, `check_glm_mtp`, `check_pid`,
  `check_seq2seq`, `check_autoencoder`, `check_lfm`, `check_flux2`,
  `check_wan` (+ `_conditioning`), `check_ltxv` (+ `_av`, `_conditioning`,
  `_av_conditioning`), `check_cosyvoice_lm` (+ `_block`), `check_qwen2`,
  `check_qwen_mrope`, `check_qwen3_weighted`, `check_vlm_splice`,
  `check_dit`, `check_vocoder`, `check_deepseekocr_relpos`
  (+ `_elementwise`), `check_matmul_bf16_weight`, `check_unet`
  (+ `_conditioning_elementwise`), plus the
  imaging workstream's `check_sam2` (+ `_on`), `check_arcface`, `check_vqgan`
  (+ `_lowered`), `check_clip` (+ `_bigg`, `_tiled`),
  `check_t5` (+ `_one_block`, `_tiled`, `_rel_bias_elementwise`)
  and `check_codeformer` (+ `_one_layer`). The authoritative list is
  `grep 'pub fn check_' crates/gradcheck/src/` - an entry point that is not
  wired into `crates/gradcheck/tests/` is not a gate. SSA-style forward (each stage
  writes a fresh buffer that doubles as the backprop activation cache) -
  preserve it when adding stages.

  **Full backward + a `gradcheck` entry point is the default expectation for
  every new model, not an opt-out.** Forward-only is the exception, and it
  requires the same explicit justification the models that already ship that
  way recorded when they did: `check_flux1`,
  `check_controlnet`, `check_pulid`, `check_instantid`, `check_chronos2`
  and `check_rrdbnet` are genuinely absent
  because those ports prioritized reaching a working forward pass on
  hardware-constrained checkpoints first, each documented in its own
  `.agents/roadmap/<model>.md` - that list is a record of what shipped
  under real constraints, not a template to reach for on a new port.
  `check_unet` used to head that list and is now CLOSED; the note that it
  would be "cheap because the forward composes existing adjoints" was half
  right (no kernel was needed) and half wrong in the dangerous direction - the
  transformer half was emitted with `Builder::push_step`, i.e. it was not
  differentiable at all rather than merely un-gated. `check_controlnet` is
  unblocked by it, since ControlNet's trainable copy IS those same recorded
  blocks. Do not
  cite "some models ship forward-only" as a reason to skip backward on a new
  model; if a genuine constraint forces that tradeoff, name it and record it
  the same way, in the same change.

  **`directional_check` alone does NOT catch a partially-wrong gradient.** It
  contracts a tensor onto one ±1 direction and keeps the *best-agreeing* of
  `n_dirs`; a *wholly* wrong gradient fails every direction, but a **folded or
  shared parameter** accumulated over only some of its contributors has a
  contraction that can be small, and best-of-`n` actively selects the direction
  where it is smallest. Measured: deleting T5's cross-block `axpy` fold leaves
  `rel_bias.weight` **33 % wrong** (‖Δg‖₂ 0.672 vs ‖g‖₂ 2.044, one entry
  sign-flipped) and **every** T5 directional check still passes on both
  backends. Use `gradcheck::elementwise_check` - per-**entry** central
  differences, `2·numel` forwards - for every folded/shared parameter.
- **Imported models are parity-gated, not gradient-guessed.** `worldmirror2`,
  `chronos2`, `kronos`, `fincast`, `zipdepth`, `diamond` are imported 1:1 from a
  reference checkpoint and verified stage-by-stage against dumped goldens
  (`scripts/parity-dump/`, `tools/goldens/*_dump_reference.py`). `make parity` is the
  cross-backend gate (CPU == Vulkan == NPU). Parity proves the *forward* pass
  matches the reference; it is not a substitute for `gradcheck` on a model
  that trains - see the "full backward is the default" note above.
- **Adding a capability ≠ adding a subcommand.** Implement `capability::Action`
  and list it in a `Provider`; `brain <arch> <verb>` and the event API pick it up.
- **Every new model ships the full serving contract - code is not "done" until it
  is served.** Adding a model means, in the same change:
  1. a **`capability::Provider`** (or a manifest via its `ResidentModel`) exposing
     its actions through the generalized interface - never a bespoke subcommand;
  2. a **residency adapter** (`crates/cli/src/resident_*.rs`, registered in
     `resident::build_executor`, env-gated) so it is **scheduled**, memory-budgeted,
     and swappable by the `Executor` like every other model;
  3. **true batching**: implement `Instance::run_batch` with a genuine batched
     forward wherever the architecture allows (see `resident_asr`/`resident.rs`
     yolov8) - never leave concurrent same-model work on the default serial loop
     without saying why;
  4. **D-Bus wiring + a runnable example.** The model's actions MUST be reachable
     over `crates/dbus` (`com.swedishembedded.Brain1`) and demonstrated by an
     example under `examples/<domain>/` with a README. If the model's shape fits the
     existing D-Bus surface (`Run`/`Subscribe`/`StreamTranscribe`/fd blobs), use it;
     if it does not, **extend or refactor the surface** (add a method, generalize a
     frame type) rather than bolting on a side channel - and update every existing
     client/example that the change touches. The full checklist lives in
     `.agents/rules/serving-contract.md` (linked from the Serving stack section); keep it and
     this bullet in sync.
  A model that trains and passes parity but cannot be discovered, scheduled, batched,
  and driven over D-Bus is **incomplete**.

  **Imaging/conditioning workstream status, so nobody has to infer it:** the
  contract is met for **`sam2`, `scrfd`, `arcface`, `vqgan`, `codeformer`,
  `clip`, `t5encoder`, `sdxlunet`, `controlnet`, `flux1` and `pulid`** -
  eleven models, each with a `caps` module, a `resident_*.rs` registered via
  `catalog.rs` (read generically by `build_executor`), and the existing
  D-Bus `Run`. `sam2`'s `run_batch` does real grouping (by image), `clip`'s
  and `t5encoder`'s batch rows into one forward at a shared context length;
  the rest - including `sdxlunet`, `controlnet`, `flux1` and `pulid`, each a
  full multi-step sample per call with no batch axis to fill - are the
  serial default and each says why in-file. All eleven now have a runnable
  `examples/` entry, under
  `examples/{vision,restore,embedding,imagegen}/`. `controlnet`'s `caps` is
  its own sampler loop (`sdxlunet::pipeline::Sdxl` has no seam for a per-step
  residual), built on `Unet::new_controlled` + `Unet::run_with_control`
  rather than composed on top of `pipeline::Sdxl` - see
  `crates/controlnet/src/caps.rs`'s module docs. `pulid::caps` composes FIVE
  models (ArcFace, EVA-CLIP, IDFormer, `PulidAdapter`, FLUX.1) and adds no
  numerics of its own - it drives `flux1::pipeline::Flux1::generate_injected`
  (a new method: `Flux1::generate` with every DiT step optionally routed
  through `forward_injected`, so `pulid` needs no dependency the other
  direction). `flux1` and `pulid` are the newest of the eleven and the only
  two with no end-to-end fixture in this workspace to verify their pipeline
  glue against - see each one's module docs' honest scope note.
- **Every served model is named `<vendor>/<repo>[-<QUANT>]`, matching its
  upstream URL exactly (case included) - never a bare short name.** `brain/`,
  `local/` and `test/` are reserved vendors for built-ins, hand-placed files,
  and test mocks respectively (`brain/mock`, `brain/yolov8`, `brain/s3dit`, …);
  everything else names a real HuggingFace repo (`Qwen/Qwen3-0.6B`,
  `Qwen/Qwen3-0.6B-Q4_K_M`). The grammar, the reserved-vendor list, and the
  legacy-name deprecation table live in **`crates/modelref`**
  (`ModelRef`/`Quant`, `modelref::alias`); the on-disk `<vendor>/<repo>/…`
  layout and the fetch resolution ladder live in **`crates/modelstore`**. A
  `capability::Manifest`'s `model` field is **always** the canonical id - never
  a legacy short name. Adding a model means adding its fully-qualified ref, not
  a convenient alias; if it already shipped under a short name, add a row to
  `modelref::alias::ROWS` instead of breaking existing callers, and consume it
  ONLY at the two dispatch seams that resolve a client-supplied name against the
  catalog (`apiserve::catalog::candidates` for HTTP, the D-Bus/`brain <arch>` model
  argument) - never bake it into a manifest or a test fixture. See
  `docs/using/models-and-weights.md`.
- **The API surface must stay WATERTIGHT - every change triggers a full security
  audit AND is covered by automated security tests.** Any change to `crates/apiserve`
  (the HTTP providers) OR `crates/dbus` (the D-Bus surface) - a route, handler, auth
  path, error shape, admission policy, or exposed method - is **not done** until you
  have BOTH:
  1. **Audited the WHOLE API** (not just the changed handler) against
     **`.agents/rules/api-security.md`** - authn/authz (key required on every route incl.
     the fallback, constant-time compare, no key in any log/response/error), input/DoS
     bounds (body-size 413, JSON depth, param/array bounds → 400), admission/backpressure
     (429/503, cancel-on-disconnect frees compute), SSRF/egress, error hygiene (no
     internal detail/paths/panic text in bodies), transport (localhost default). Run the
     pass with the **`security-review`** skill and fix every finding.
  2. **Encoded those requirements as automated tests** so a regression fails CI, not
     just a future audit: the socket-level **`tests/e2e/api_conformance.bats`** (via the
     `BRAIN_MOCK` model - auth matrix incl. no-enumeration, no-key-leak, 413, error
     hygiene, input-bound 400s, admission 429) AND the in-process
     **`crates/apiserve/tests/api.rs`**. Adding a route/field means adding its security
     assertions here.
  These surfaces are internet-reachable when bound, so **all request input is hostile**;
  never trust the client. Do not consider an API change complete without both.
- **API specs: at most two sources of truth.** brain's code (what it implements) and
  the **vendored upstream OpenAPI specs** (`crates/apiserve/tests/specs/`, a cached
  copy of what providers support) - validated against each other by the jsonschema
  conformance tests. There is **no** separate hand-maintained "brain spec." Refresh
  the vendored specs from upstream with the **`api-sync`** command (`.claude/commands/
  api-sync.md`), then adapt brain to any drift and re-green the conformance tests.
- **No absolute paths in source - anywhere.** Never hardcode a machine-specific
  absolute path (`/data/…`, `/home/…`, `/tmp/…`) in `crates/**`: not in code, not
  in a test `const`, not as a runtime default, not in a doc comment. Two homes for
  what used to be hardcoded:
  1. **Test / parity fixtures** live under the **gitignored `testdata/` tree** -
     inputs and goldens ONLY (audio/image/text fixtures, dumped-golden tensors);
     never a model checkpoint's `.git` directory, never runnable code or upstream
     docs/notebooks a test doesn't read. Resolved at runtime from
     `$BRAIN_TESTDATA` (default `<repo>/testdata`) via **`brain_testutil::testdata`**
     (`crates/testutil`, a dev-dependency - the one implementation; it used to be
     36 byte-identical copy-pasted helpers, one per crate). A test **skips
     itself** when its fixture is absent - through **`brain_testutil::skip`**
     (absent FIXTURE: a hard failure under `BRAIN_REQUIRE_FIXTURES=1`, which is
     what `make parity/strict` sets) or **`brain_testutil::skip_unavailable`**
     (absent HARDWARE - no discrete GPU, no NPU, no OpenVINO, no ffmpeg - which
     no flag may turn fatal, or the gate becomes one nobody can run). Never a
     bare `eprintln!` + `return`: cargo reports a skipped test as a PASS, so an
     unnamed skip is indistinguishable from a comparison that ran. A golden also
     records WHICH checkpoint produced it (`tools/goldens/golden_source.py`,
     enforced by `brain_testutil::golden::Source`), so a golden paired with the
     wrong tier is a named skip rather than a shape error deep in the importer.
     Populate the tree with
     **`make fetch/testdata`** (`scripts/data/fetch-testdata.sh`) - it hard-links from
     a local mirror, fetching only files not already present, organised as a tree
     (`testdata/<domain>/<model>/…`); there is currently no URL-download fallback
     (say so if you add one - don't leave the claim stale). The mirror location is
     an overridable script variable - the ONE permitted place a machine path may
     appear in `crates/**`'s fixture-resolution path.
  2. **In-repo artifacts** (`out/…` build outputs, `scratchpad/…`) are resolved
     **repo-relative** (`concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/…")`), never
     as an absolute literal.
  Runtime weight locations come from an **env var or CLI flag**, never a baked-in
  path. Grep gate (a string literal that *starts* an absolute machine path):
  `grep -rnE '"/(data|home|tmp|opt|mnt|root)/' crates --include='*.rs'` must stay
  empty. **`*.rs` only**, and `scripts/gates/check-no-machine-paths.sh` scopes
  itself the same way in both of its modes: the rule is about how brain's own
  source resolves a path, and `crates/**` also holds vendored third-party
  fixtures (`crates/apiserve/tests/specs/*.json` are upstream OpenAPI documents)
  whose example values are not ours to edit. (A `/data/`
  substring mid-string - a URL, or a torch-archive-internal `…/data/<key>` - is
  not a filesystem path and is fine.) `scripts/` and `tools/` get the equivalent
  check via `make check/scripts` (below) - they are not `crates/**`, but they are
  not exempt from the spirit of this rule either.
- **`scripts/` vs `tools/`.** `scripts/` is repo automation - invoked by a
  Makefile target or a bats test, nothing else. `tools/` is developer utilities a
  human runs by hand (golden dumpers, converters, benchmarks) - it needs
  `requirements.txt`, `crates/**` never does. **`make check/scripts`**
  (`scripts/gates/check-scripts.sh`, folded into `test/full`) is what keeps both from
  rotting the way they did before it existed: every `.sh` parses and every `.py`
  compiles; every tracked file is named **somewhere else** in the repo (a
  Makefile target, a bats test, a Rust doc comment citing it as a golden
  generator, a sibling script, a doc) or it is a true orphan and the gate fails;
  and no non-overridable absolute machine path outside a sanctioned
  `${VAR:-/path}` / `os.environ.get(V, "/path")` default. Adding a script means
  citing it from whatever actually uses it in the same change - an uncited
  script is indistinguishable from a dead one on the next `check/scripts` run.
- **Validate everything crossing into brain from outside, at the point of
  entry - structurally AND semantically.** Anything brain did not itself
  produce is untrusted: a bench-exported training JSONL, an imported HF
  `config.json`/`tokenizer_config.json`, a hand-written dataset, a checkpoint
  from an unfamiliar source. "Validate" means two different things and BOTH
  are required:
  1. **Structural** - the right fields, present, the right types. Parse into
     typed, `#[serde(deny_unknown_fields)]` structs with required fields as
     plain (non-`Option`) types (serde itself then fails loudly on anything
     missing or mistyped) - never permissive `serde_json::Value` indexing with
     `.unwrap_or(default)` fallbacks, which silently launders a missing or
     wrong-shaped field into a plausible-looking default. See
     `crates/checkpoint::st::ModelCard` for the field-typing convention and
     `crates/data::chat`'s `Wire*` structs for the full pattern on a real
     training-data pipeline.
  2. **Semantic** - the fields are individually well-typed but the DATA is
     still nonsense: a tool result whose `tool_call_id` names no tool call
     that was ever made, a tool response appearing before any assistant turn
     called a tool, a message sequence a codec/template cannot make sense of.
     Structural validation cannot catch this class at all - it is exactly the
     kind of defect that "only shows up when a particular field takes some
     particular value," the far more dangerous failure mode because it trains
     silently on garbage instead of refusing to run.
  Fail loudly and specifically (name the record/line/field), never coerce,
  never drop-and-continue. Validate ONCE, at the boundary where external data
  becomes an in-process value - not scattered re-checks downstream, and not
  deferred to a separate, optional command a caller has to remember to run
  (a validator nothing calls automatically is equivalent to no validator: see
  `.agents/rules/lessons.md` §1, "a gate that never runs is worse than no gate"). This
  generalizes the WATERTIGHT-API rule above (network input is hostile) to
  every other boundary: file input is exactly as hostile as network input,
  it just fails later and quieter.
- **Evaluate honestly.** Hold the input distribution fixed; separate the metric
  (perplexity) from the task (exact-match on held-out data); see `README.md` §3.
- **Gitignored:** `scratchpad/` (scratch weights, images, porting references),
  generated `data/`, `out/`, `build/`, `results/` (all `brain bench`/`brain perf`
  artifacts and ad-hoc script output, created on demand by its writers), and the
  world-model parity fixtures.

---

## Local Task Management Protocol

You are authorized to manage and execute tasks located in a local, gitignored
task-tracking folder outside version control. Each file in that folder
represents a distinct task using a Markdown + Frontmatter format. Each task can
contain items and checkbox lists. You may remove completed tasks from that
folder but only after ALL of the items have been completed.

### How to Pick Up a Task

When the user asks you to "pick up <some description> task" (or a specific task
ID/name):

1. Read the contents of the target file inside that folder. If no specific task
is named, look for the oldest file that matches the description.
2. Immediately modify that file's frontmatter to change `status: pending` to
`status: in_progress`.
3. Read the "Objective" and "In-Scope" definitions in that file. Do not wander
outside the defined scope.
4. Plan and execute the code changes required to complete the task.

### How to Complete a Task

Once each task milestone is fully built, tested, and verified:
1. Commit the changes to git as a series of self-contained, independent commits.
2. Move the task file into a `completed/` subfolder within that same
task-tracking folder (creating that subfolder the first time). No deletion, no
confirmation prompt - moving is not destructive, the file's history and content
are preserved, and the user can always delete it later themselves.
3. If the task was only partially completed, do not move it - update its
frontmatter/body to record what is done vs. remaining and leave it where it is.


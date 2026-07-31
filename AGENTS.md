# AGENTS.md — brain (edge-AI model training framework)

Routing guide for this repo. **brain** trains and evaluates **neural networks
from scratch on the CPU, GPU and NPU**, **pure Rust + raw WGSL kernels**, with
validated parity between all supported accelerator backends.

Brain provides both training and inference, with quantization, model residency,
and concurrent serving on top of one hand-written kernel engine.

It is a self-contained Cargo **workspace** of ~60 crates under `crates/` — no
Python in the build/test path; backprop correctness is gated by an in-repo
finite-difference gradient checker (`crates/gradcheck`), not a PyTorch oracle.

The engine is **architecture-agnostic**: the 320 WGSL kernels (`crates/kernels`)
are reusable building blocks, not a fixed model. New architectures are composed
from them, keeping the gradient-check discipline.

---

## Models (today)

### Language / decoder LMs

1. **GPT decoder** (`crates/gpt`) — dense nanogpt-parity baseline: token+learned
   positional embeddings, pre-LN, causal MHA, GELU MLP, untied `lm_head`, masked
   CE. `brain gpt {train,eval,gen}`.
2. **Qwen3 decoder** (`crates/qwen`) — dense GQA + QK-norm + RoPE + SwiGLU;
   HF import, LoRA finetune, INT8 (`q8.rs`), tensor/expert sharding, tool-call
   eval, and the **concurrent paged-KV serving engine** (`serve.rs`, see below).
   `brain qwen {import,infer,serve,export,precompile,train,finetune,toolcall}`.
3. **Sparse MoE Transformer** (`crates/moe`) — RMSNorm/RoPE, top-k experts; with
   **federated/sharded** expert training (`crates/federated`).
4. **GLM-5.2 decoder** (`crates/glm`) — `glm_moe_dsa`: **MLA** (low-rank q/kv with
   a decoupled nope/rope head split + interleaved RoPE), a **sigmoid `noaux_tc`
   MoE** (per-expert selection bias, shared expert, `first_k_dense_replace`
   dense→MoE schedule), untied `lm_head`, plus the DSA indexer and MTP.
   Gradient-checked (`gradcheck::check_glm`, `check_glm_mtp`).
   `brain glm {train,finetune,infer,eval,import,export}`.
5. **PID event/effect Transformer** (`crates/pid`) — LayerNorm, learned positions,
   biased linears; backs the WebGPU browser demo (`crates/web`).
6. **Seq2seq** (`crates/seq2seq`) — encoder-decoder Transformer (bidirectional
   encoder + causal/cross-attention decoder), gradient-checked.
6b. **LFM2.5-Encoder** (`crates/lfm`) — LiquidAI's bidirectional hybrid
   short-conv/attention encoder (GQA + QK-norm + RoPE, gated depthwise conv,
   tied MLM head, 8k context); imported 1:1, parity-gated per stage; chunked
   long-context inference. `brain lfm {import,fill-mask,embed}`, `brain do lfm …`.
7. **Bottleneck autoencoder** (`crates/autoencoder`) — sequence → single
   compressed representation → MLP reconstruction, MSE head; gradient-checked.

### Vision & 3D

8. **YOLOv8-style detector** (`crates/yolo`) — from-scratch anchor-free detector:
   CSP backbone → PAN-FPN neck → decoupled DFL head, assigner + BCE/CIoU/DFL loss,
   NMS box decode. Byte-compatible with canonical `yolov8n` for weight import.
   `brain yolo {train,eval,detect,fine-tune}`; Intel-NPU path via `brain npu …`.
9. **ZipDepth monocular depth** (`crates/depth`) — the 6.1M pure-conv depth net
   (QARep/RepVGG blocks, SE/strip/global-context attention, convex upsampling),
   exact vs the reference PyTorch on the released checkpoints. Realtime demo
   (`brain depth --image|--camera`, SDL views incl. autostereograms) on
   CPU/GPU/Vulkan and the Intel NPU (ONNX/OpenVINO, cosine 0.99998).
   *(Depth Anything 3 was planned as a second arch behind the same contract and
   is **dropped** — see `crates/depth/src/lib.rs`.)*
9b. **Face recognition** (`crates/facenet`) — the insightface *antelopev2* stack:
    **SCRFD-10GF** detector (ResNet-D backbone + PAFPN + 3-stride anchor head) →
    **5-point similarity alignment** (Umeyama solve on the host, applied with the
    `grid_sample` kernel) → the **ArcFace IResNet-100** 512-d embedding (PReLU
    with learned per-channel slopes; BatchNorm folded into the convs by the
    release). Imported from **ONNX** — the first import in the repo to read the
    protobuf, via the new `onnx::read` — with two-way coverage (462 / 119
    tensors). Forward-parity-gated per stage at cosine 1.000000.
    See `docs/models/face/status.md`. *(Forward only: backward/gradcheck and the
    serving contract are deferred and listed in that ledger.)*
9c. **SAM 2.1 promptable segmentation** (`crates/sam2`) — the **image path** of
    Meta's SAM 2.1: Hiera trunk (windowed attention with a per-block window
    schedule + `q_pool` stride stages) → FPN neck → prompt encoder (points, boxes
    and a mask prompt, over a random-Fourier positional encoding) → the two-way
    mask decoder (4 mask tokens + IoU head + object-score token). Both released
    variants (`hiera_tiny`, `hiera_large`) imported with two-way coverage and
    **forward-parity-gated per stage at cosine ≥0.9999999999** over 283
    comparisons. Composes `model::vit`'s windowed spans and `vision::blocks`;
    adds no kernel. *(Forward only: the video memory bank, backward/gradcheck and
    the serving contract are deferred — see `docs/imaging/plan.md` §4.)*
10. **WorldMirror-2 multi-view 3D reconstruction** (`crates/mirror`) — the
    HY-World 2.0 1.26B feed-forward model: per-frame DINOv2 ViT-L/14 encoding,
    24 alternating frame/global attention levels (QK-norm + normalized 2D RoPE),
    DPT heads (depth/points/normals/gaussians) + iterative camera head; photos →
    a navigable 3DGS scene. Imported exactly, parity-gated per stage.
    `brain mirror {import,infer,demo,export-npu}`.
11. **3D Gaussian Splatting** (`crates/splat`) — from-scratch tiled 3DGS
    rasterizer (atomic-free/barrier-free WGSL: generic scan + radix sort →
    per-tile compositing) with forward AND backward (autograd-verified), Inria
    PLY IO, interactive WASD+mouse viewer, and `splat fit` scene optimization.
    `brain splat {info,render,view,fit}`.

### Image generation (diffusion)

12. **Z-Image** (`crates/zimage`) — Tongyi text-to-image: the **S³-DiT**
    single-stream diffusion transformer over Qwen3-4B caption features + VAE
    latents. Forward/inference first; LoRA, INT8, sharding, and hand-written
    backward (`grad.rs`/`devgrad.rs`/`modelgrad.rs`) alongside. Assembled from
    the shared `dit` / `diffusion` / `vae` / `qwen` crates.

12b. **FLUX.2 Klein** (`crates/flux2`) — BFL's 4B/9B MMDiT text-to-image +
    image-editing family: double-stream img/txt blocks (joint attention) +
    single-stream parallel blocks, **3 global modulation linears** folded into
    LayerNorm affine params, 4-axis interleaved RoPE (theta 2000), Qwen3
    multi-layer masked-pad text conditioning ([9,18,27] concat), FLUX.2 VAE
    (128-ch latent = 32ch × 2×2 unshuffle + eval-BatchNorm). Klein = 4-step
    distilled, no CFG; `base` variants = 50 steps + CFG, same tensors.
    Parity-gated per stage (forward cosine 1.000000 vs diffusers).
    `brain flux2 generate` (t2i + `--ref` editing). 9B weights are
    NC-licensed — see `docs/models/flux2/readme.md`.

12c. **VQGAN / CodeFormer VQ autoencoder** (`crates/vqgan`) — the VQ
    encoder/codebook/generator that CodeFormer's face restoration is built on:
    a 25-block encoder and a 25-block generator over `vae::blocks` (conv,
    GroupNorm, self-attention, ResNet, nearest-upsample), with `vq_argmin` for
    the codebook assignment and `embed` for the lookup. Adds **no kernel and no
    block**. Both released checkpoints (`codeformer.pth`, `vqgan_code1024.pth`)
    imported and forward-parity-gated at cosine 1.000000000 with **zero**
    code-index disagreements. See `docs/models/vqgan/status.md`. *(The
    CodeFormer **transformer and the fidelity dial `w` are not implemented**;
    backward/gradcheck and the serving contract are deferred.)*

12d. **CLIP text + image towers** (`crates/clip`) — one config-driven graph for
    the three encoders the imaging workstream needs: **CLIP-L** and
    **OpenCLIP-bigG/14** text (SDXL conditioning; CLIP-L again for FLUX.1) and
    the **EVA-CLIP-L/336** image tower (PuLID), the latter with 2D RoPE on q/k
    excluding cls, subln attention and naive SwiGLU. Imported from the SDXL
    `text_encoder{,_2}` safetensors and the EVA `.pt`, two-way covered, and
    forward-parity-gated per stage vs `transformers`/`open_clip` (cross-checked
    against `diffusers.encode_prompt` for the SDXL conditioning pair).
    *(Forward only: the **CLIP BPE tokenizer** belongs in `crates/data` next to
    the GPT-2/Qwen BPEs and is **not implemented**, image preprocessing is not
    implemented, and backward/gradcheck + the serving contract are deferred —
    see `docs/imaging/plan.md` §4.)*

### Audio / speech

13. **Qwen3-TTS** (`crates/tts` + `crates/codec` + `crates/speaker` +
    `crates/audio`) — Talker (multi-codebook dual-track Qwen3 decoder) + 5-layer
    MTP code predictor → 12 Hz Mimi-style neural codec (RVQ + transformer +
    SEANet conv-transpose decoder); ECAPA-TDNN speaker encoder for voice cloning.
    `brain tts {import,clone,synth,design,serve,sim,finetune}`.
13b. **ASR / speech-to-text** — two imported, parity-gated models served through the
    full stack (capability + residency + batched `run_batch` + D-Bus
    `StreamTranscribe` + `examples/asr/`):
    * **Nemotron 3.5 ASR Streaming 0.6B** (`crates/nemotron`) — FastConformer
      encoder (depthwise-sep causal subsampling, macaron FFs, Transformer-XL
      rel-pos attention, GLU conv module) + RNN-T transducer; the *streaming* model,
      true batched forward across concurrent windows. Fully trainable/gradchecked.
    * **Qwen3-ASR 1.7B** (`crates/qwen-asr`) — Whisper-style audio encoder + a spliced
      Qwen3-1.7B decoder (reuses `crates/qwen`); offline, fixed audio window.
    Shared audio-in/text-out contract in `audio::asr_caps`. See
    `docs/models/asr/status.md`.

### Forecasting

14. **Chronos-2** (`crates/chronos2`) — encoder-only T5-style patch transformer,
    time+group attention, multi-patch quantile head. Imported exactly, parity-gated.
15. **Kronos** (`crates/kronos`) — BSQ tokenizer (OHLCV bar → hierarchical tokens)
    + autoregressive decoder with a dual head. Imported exactly, parity-gated.
16. **FinCast** (`crates/fincast`) — TimesFM-style patched decoder with a sparse
    top-2 MoE and a probabilistic-quantile head. Imported exactly, parity-gated.
    *(Reference is research/educational use only.)*
    All three sit behind the model-agnostic `forecast::ForecastModel` seam;
    `crates/fcbench` holds baselines + the rolling-origin backtester.
    `brain forecast {compare,serve,import,finetune}`.

### World models (playable, action-conditioned video)

17. **DIAMOND** (`crates/wm-diamond`) — EDM diffusion world model (Atari-100k):
    pre-recorded UNet graph, torch `.pt` import, playable. Parity fixtures via
    `make wm-fixtures`.
18. **GenieRedux-G** (`crates/wm-genie`) — CoinRun ST-transformer world model
    (QK-normalized biased attention, GEGLU FFN, PEG); tokenizer/MaskGIT dynamics
    in progress. `brain wm {play,bench}` (SDL window via `crates/wm-display`).

> `crates/timeseries` and `crates/autodiff` are **placeholders** — declared in the
> workspace, implemented in a later phase.

---

## Serving & runtime stack

The recent workstream (P7.x) is concurrent LLM serving. Key pieces:

| Piece | Where | What |
|---|---|---|
| Paged KV foundation | `crates/model/src/paged.rs` | block allocator, `BlockTable` (+`truncate`) |
| Serving engine | `crates/qwen/src/serve.rs` | shared block pools, batched **ragged paged decode**, batched + **chunked prefill**, **int8 paged KV** (~4× smaller pool), **speculative decoding**, `Engine::load` from checkpoint |
| Scheduler | `crates/qwen/src/serve.rs` | continuous batching (multi-sequence concurrent decode) + throughput benchmark |
| Residency | `crates/residency` | tiers model weights GPU/RAM/disk by LRU within a memory budget; schedules jobs (batch-by-model, queue-age-aware, parallel lanes) |
| Capability interface | `crates/capability` | models advertise a `Manifest` of typed `ActionSpec`s; CLI (`brain caps` / `brain do`) and the event API dispatch generically — adding a capability = implementing `Action`, no new subcommand or event variant |
| Transports | `crates/server` | one JSONL protocol over **stdio, TCP, and Unix socket**; thread-per-connection, bounded, panic-isolated |
| D-Bus surface | `crates/dbus` | exposes `capability::Registry` over `com.swedishembedded.Brain1`, passing images/streams via fd (memfd/mmap + dmabuf). Example client: `examples/dbus` |
| Event HFSM | `crates/runtime`, `crates/events`, `crates/hfsm` | `camera_frame`→`object_detected`, `user_text`→`brain_text_chunk` |
| Python client | `brain-py/` | drives the `brain` binary as an event-driven subprocess (not in the build/test path) |

Multi-GPU scaling lives in `crates/model`:
`{distributed,parallel,collective,netcollective,shard,plan,grid}.rs` — see
`docs/scaling/`.

---

## Workspace layout (`crates/`)

### Engine core

| Crate | Responsibility |
|---|---|
| `kernels` | all 320 WGSL kernels (the source of truth) as consts + `src()` |
| `gpu-core` | compute-device facade: selects and forwards to an eager `Backend` |
| `backend-api` | `Backend`/`GraphBackend` traits, neutral buffer/step handles, registry — a new backend depends only on this |
| `backend-wgpu` | wgpu (Vulkan/Metal/DX12/GL/WebGPU) eager backend — **the default** |
| `backend-cpu` | native CPU backend: WGSL → Cranelift JIT across cores, AVX2 fast paths |
| `backend-vulkan` | native Vulkan (ash + naga WGSL→SPIR-V) eager backend |
| `wgsl-cpu` | the CPU backend's compiler: WGSL → naga IR → Cranelift JIT |
| `vulkan` | **optional, non-default** `VK_KHR_cooperative_matrix` matmul path (excluded from `default-members`; build with `-p brain-vulkan` / cli feature `vulkan-coopmat`) |
| `paramstore` / `optim` | param/grad/Adam buffers; AdamW + global grad-norm clip |
| `checkpoint` | `.safetensors` container + manifest/SHA-256 + expert-shard I/O (no fs on wasm) |
| `model` | architecture-agnostic `Model` abstraction, generic trainer, shared block builders (`block.rs`, `vit.rs`), paged KV, and the multi-GPU parallelism layer |
| `autodiff` | shared SSA forward-cache / reverse-mode scaffolding — **placeholder** |
| `data` | char + GPT-2 **BPE** tokenizers, dataset generators, loaders (masking/alignment), normalization |
| `eval` | perplexity + task exact-match (LM) and detection metrics (mAP@0.5/precision/recall) |
| `gradcheck` | finite-difference backprop correctness gate |
| `bench` | model-agnostic architecture-evaluation suite — *does it **learn**?* (see below) |
| `perf` | performance benchmarking suite — *how **fast**, at what cost, still correct?* (see below) |
| `cli` | the `brain` binary (aggregates everything) |
| `web` | wasm32/WebGPU PID demo (empty off wasm32) |

### Model crates

| Crate | Model |
|---|---|
| `gpt` / `qwen` / `moe` / `glm` / `pid` | decoder LMs (see Models) |
| `seq2seq` / `autoencoder` / `timeseries` | encoder-decoder / bottleneck AE / placeholder |
| `federated` | vertical expert split/assemble, hash-verified manifests, train-scope |
| `yolo` / `vision` | detector; shared conv-net blocks (spec-driven `Conv` incl. fused/register-tiled eval paths, `BatchNorm`, `PReLU`, `MaxPool`/`AvgPool`, `SPPF`, bottlenecks, `fold_bn`) |
| `depth` | ZipDepth: model/blocks/import/fuse, `Predictor`, viz/stereo/effects, INT8 calib |
| `mirror` / `splat` | WorldMirror-2; 3DGS rasterizer + PLY IO + `fit` + viewer |
| `diffusion` / `dit` / `vae` / `zimage` | flow-matching core; shared DiT blocks; AutoencoderKL; Z-Image |
| `audio` / `codec` / `speaker` / `tts` | wav/STFT/mel + 1D conv builders; Mimi codec; ECAPA-TDNN; Talker+MTP |
| `forecast` / `fcbench` / `chronos2` / `kronos` / `fincast` | forecasting seam, backtester, three imported models |
| `wm-core` / `wm-diamond` / `wm-genie` / `wm-display` | world-model trait + fake model; DIAMOND; GenieRedux-G; SDL window |

### Deployment / IO

| Crate | Responsibility |
|---|---|
| `onnx` | pure-Rust ONNX graph model + serializer (export), plus an import-side **reader** (`read`: initializers/nodes/attributes) used by `facenet`; vendored `prost`, no `protoc` |
| `npu` | YOLO/depth → ONNX export + BN fold + brain-native INT8 PTQ + fake-quant simulator + OpenVINO **Intel NPU** runtime (`runtime-linking`) |
| `capture` | V4L2 webcam (hand-rolled ioctl FFI, YUYV→RGB, latest-frame slot) |
| `capability` / `residency` / `server` / `dbus` / `runtime` / `events` / `hfsm` | the serving/runtime stack (table above) |

---

## Task → where to look

| Task | Where |
|---|---|
| Architecture & crate graph | `docs/architecture.md` *(crate graph is stale — see Doc gaps)* |
| Testing strategy + gradient-check gate | `docs/testing.md` |
| **Porting a new model** (goldens → import → kernel contracts → parity ladder → training) | **`docs/porting-playbook.md`** — read BEFORE starting any port |
| Multi-GPU scaling (data / pipeline / tensor parallel) | `docs/scaling/*.md`; `crates/model/src/{distributed,parallel,collective,shard,plan,grid}.rs` |
| Performance: CPU/GPU inference optimizations (what sped things up + why) | `docs/performance/overview.md`, `docs/performance/p40.md` |
| **Performance benchmarking** (`brain perf`): design / ledger | `docs/performance/benchmarking.md`, `docs/performance/status.md`; `crates/perf`, `crates/cli/src/perf_cli.rs` |
| Perf regression gate (hard floors vs a committed baseline) | `brain perf gate`; `crates/perf/src/gate.rs` |
| Device capabilities (class/limits/numeric tiers, queried never assumed) | `backend_api::DeviceCaps`; filled per backend, `Gpu::caps()` |
| Canonical GPU registry / placement (`brain devices`, `Gpu::new_on`, `with_gpu`) | `docs/engine/devices.md`; `crates/gpu-core/src/devices.rs` |
| Kernel selection policy + autotuner (which variant runs, measured per device) | `backend_api::select` (`candidates`/`DefaultSelector`/`AutoTuner`), `gpu_core::tune`; `BRAIN_NO_AUTOTUNE=1` forces static |
| Kernel specialisation (one WGSL source, tunable constants) | `kernels::template` |
| Prompt-prefix cache (paged block reuse across requests) | `model::paged::PrefixCache`; adoption in `qwen::serve::Engine::prefill` |
| Int8 serving weights + on-device decode window | `qwen::serve` (`--weights-int8` / target suffix `:i8w`; `DECODE_WINDOW`) |
| Engine internals | `docs/engine/{overview,training,vulkan,web}.md` |
| **Add/adjust/dispatch a WGSL kernel** | **`docs/kernel-checklist.md`** — read BEFORE writing or dispatching one; then `crates/kernels/wgsl/*.wgsl` + **`make kernels-regen`** |
| **Something is slow (model, kernel, training step)** | **`docs/kernel-checklist.md` §E** (measure-first rules + the five killed hypotheses), `docs/porting-playbook.md` §10 (the performance ladder), case studies in `docs/performance/overview.md` |
| MoE toy task / honest eval methodology | `README.md` |
| Federated MoE pipeline (done vs remaining) | `docs/federated.md`; `crates/federated/src/{shard,sha256}.rs` |
| GPT model / training / sampling | `crates/gpt/src/{model,train,sample,init}.rs` |
| Qwen model / import / LoRA / INT8 / sharding | `crates/qwen/src/{model,import,finetune,q8,shard,sample}.rs` |
| **Qwen concurrent serving (paged KV, continuous batching, spec decode)** | `crates/qwen/src/serve.rs`, `crates/model/src/paged.rs`, `crates/cli/src/qwen_cli.rs` |
| Model residency / job scheduling | `crates/residency/src/{manager,scheduler,executor,budget,lru,place}.rs` |
| Capability manifests + generic dispatch (`brain caps` / `brain do`) | `crates/capability/src/lib.rs`, `crates/cli/src/caps_cli.rs` |
| JSONL transports (stdio / TCP / unix) | `crates/server/src/{transport,controller_session}.rs` |
| D-Bus control surface | `crates/dbus`, `examples/dbus` |
| Event/HFSM controller (`brain run`) | `crates/runtime/src/{lib,pump}.rs`, `crates/cli/src/run_cli.rs`, `crates/events/src/lib.rs` |
| GLM-5.2 (MLA + MoE + DSA indexer + MTP) | `docs/models/glm/readme.md`, `docs/models/glm/npu.md`; `crates/glm`, `crates/cli/src/glm_cli.rs` |
| LFM2.5-Encoder (bidir conv/attn hybrid, MLM, 8k) | `docs/models/lfm/{readme,status}.md`; `crates/lfm`, `crates/cli/src/lfm_cli.rs`; goldens via `tools/lfm_dump_reference.py` |
| YOLO model / loss / inference | `crates/yolo/src/{model,head,blocks,loss,assign,infer,nms,config}.rs`; `docs/models/yolo/readme.md` |
| YOLO → Intel NPU (export/quantize/run/bench) | `crates/npu`, `crates/onnx`, `crates/cli/src/npu_cli.rs`, `docs/models/yolo/npu.md` |
| ZipDepth: guide / ledger (incl. GPU perf root causes) | `docs/models/depth/{readme,status}.md`; `crates/depth/src/*`, `crates/cli/src/depth_cli.rs` |
| Face recognition (SCRFD + alignment + ArcFace) | `docs/models/face/status.md`; `crates/facenet/src/{config,import,model,align,detect}.rs`; goldens via `tools/arcface_dump_reference.py` |
| **Read an ONNX file** (initializers, nodes, attributes) | `crates/onnx/src/read.rs` — the import front-end; `crates/onnx` is otherwise export-only |
| VQGAN / CodeFormer VQ autoencoder | `docs/models/vqgan/status.md`; `crates/vqgan/src/{config,import,model}.rs` over `crates/vae/src/blocks.rs`; goldens via `tools/codeformer_dump_reference.py` |
| CLIP-L / OpenCLIP-bigG / EVA-CLIP text+image towers | `crates/clip/src/{config,import,model}.rs`; goldens via `tools/clip_dump_reference.py`; plan/status in `docs/imaging/plan.md` |
| ZipDepth → Intel NPU (fp32 ONNX, exact parity) | `npu::depth_topology`, `crates/depth/src/fuse.rs` |
| SAM 2.1 promptable segmentation (image path) | `crates/sam2/src/{config,import,model,hostpe}.rs`; goldens via `tools/sam2_dump_reference.py`; plan/status in `docs/imaging/plan.md` |
| WorldMirror-2 (photos → 3DGS scene) | `docs/models/mirror/{readme,status}.md`; `crates/mirror`, `crates/cli/src/mirror_cli.rs` |
| 3D Gaussian Splatting rasterizer + viewer + fit | `docs/models/splat/{readme,status}.md`; `crates/splat`, `crates/cli/src/splat_cli.rs` |
| Shared ViT block builder (DINOv2/trunk/camera-head) | `crates/model/src/vit.rs` |
| Fused conv eval paths (act selector, register tiling, grouped) | `crates/vision/src/blocks.rs`, `crates/kernels/wgsl/conv_act*.wgsl`, `conv2d_gd_reg.wgsl`, `crates/backend-cpu/src/fast_conv.rs` |
| Detection metrics (mAP/precision/recall) | `crates/eval/src/detection.rs` |
| Synthetic detection dataset (RGB shapes + GT boxes) | `crates/data/src/gen_detect.rs` |
| Datasets & tokenizers | `crates/data/src/{prepare,gen_*,tokenizer,bpe,loader,binio,rng}.rs` |
| TTS: guide / acceleration | `docs/models/tts/{readme,acceleration}.md`; `crates/{tts,codec,speaker,audio}`, `crates/cli/src/{tts_cli,tts_serve}.rs` |
| **ASR (speech-to-text)**: status / serving / perf | `docs/models/asr/status.md`; `crates/{nemotron,qwen-asr}`, shared `audio::asr_caps`, `crates/cli/src/resident_asr.rs`, D-Bus `StreamTranscribe` (`crates/dbus`), `examples/asr/` |
| Forecasting models + backtester | `docs/models/{chronos2,kronos,fincast}/status.md`; `crates/{forecast,fcbench,chronos2,kronos,fincast}`, `crates/cli/src/forecast_cli.rs` |
| World models (playable) | `docs/models/world-models/{status,playbooks,fixtures}.md` + `specs/`; `crates/wm-*`, `crates/cli/src/wm_cli.rs` |
| Z-Image / diffusion stack | `crates/{zimage,dit,diffusion,vae}` *(no docs/ entry yet)* |
| FLUX.2 Klein: guide / ledger | `docs/models/flux2/{readme,status}.md`; `crates/flux2`, `crates/cli/src/flux2_cli.rs`; goldens via `tools/flux2_dump_reference.py` |
| Finetuning guides | `docs/guides/finetune/{plan,datasets}.md` |
| CLI subcommands | `crates/cli/src/{main,args,*_cli}.rs` |

---

## Essential commands

**Always build through the Makefile, never `cargo` directly:** `make build`
(debug), `make release` (optimized), `make test` (suite). They wrap cargo with
the project's expected flags/targets.

```bash
make build                           # debug build
make release && make test            # optimized build + full suite (MOE_SKIP_GPU_TESTS=1 to skip GPU;
                                     # tests run at TEST_THREADS=8 on the pooled test device — every
                                     # test binary shares one device via gpu_core::testgpu)
make gradcheck                       # backprop correctness gate
make parity                          # cross-backend parity: CPU == Vulkan == NPU (scripts/parity-gate.sh)
make kernels-regen                   # regenerate the kernel const block after adding/removing a .wgsl
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

Direct binary — the model is selected by the command:

```bash
./target/release/brain <cmd> [opts]
# data gpt qwen glm tts wm yolo depth mirror splat npu federated bench forecast
# caps|capabilities  do  run|serve  pid  gradcheck
# train | eval | generate           (these three are the bare MoE model)
```

**Device selection** — `--device` declares **which compute is schedulable**, not
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
same physical cards everywhere — `--device`, `Shard.gpu_index`,
`residency::Device::Gpu(i)` — and nvidia-smi order maps to them via PCI, not by
assumption. `brain devices` prints the table. Placement is explicit
(`Gpu::new_on`, scoped `devices::with_gpu`) — never env mutation;
`BRAIN_GPU_INDEX` remains user *input* only, parsed once at first registry use.
Out-of-range indices are errors, never silent clamps. See
`docs/engine/devices.md`.

This bounds where work **executes** — host RAM and disk stay available as
cache/spill tiers, so `--device gpu` still uses RAM for weight caching.

```bash
./target/release/brain gpt train data/calculator --device cpu --out out/gpt.safetensors
./target/release/brain perf run sweep --device gpu0 --target qwen-synth:12x768x12
BRAIN_DEVICE=cpu make test            # whole suite on CPU, no GPU needed
```

Event/stdio controller — an HFSM (`crates/runtime`) reads JSONL events on stdin
and emits JSONL on stdout. `--gpt`/`--yolo` load real models (or `BRAIN_GPT`/
`BRAIN_YOLO`); with neither, fake echo/detector models keep the loop usable:

```bash
printf '{"event":"user_text","text":"hi"}\n' | ./target/release/brain run
```

---

## Benchmark suite (`crates/bench`)

`brain-bench` is a **model-agnostic** architecture-evaluation layer: each
benchmark owns its *dataset* and its *scoring*, the harness owns running it. Use
it to answer "does this architecture actually learn task X?" the same way across
tasks. See `crates/bench/README.md` for the full design.

**Run** (this box has two real Tesla P40s — `--device gpu0` selects one; a
GPU-less box still serves `--device gpu` through the llvmpipe software
rasteriser, and such runs must never be reported as GPU numbers):

```bash
BRAIN_DEVICE=cpu make bench          # every registered benchmark, one table
BRAIN_DEVICE=cpu make bench/mqar     # a single benchmark
./target/release/brain bench [--device cpu] [<name>] [--seed S]
```

**Registered benchmarks:** `mqar` (multi-query associative recall — the
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
   Registered today: `gpt`, `gpt-small`, `gpt-wide`, `moe`, `qwen`, `glm`.
3. **Run + compare:**
   ```bash
   BRAIN_DEVICE=cpu make bench/eval ARCH=<name>   # -> results/<arch>-<seed>.json
   BRAIN_DEVICE=cpu make bench/compare            # leaderboard over results/*.json
   ```

**Capability axes** (`crates/bench/src/axes.rs`) group benchmarks into a profile —
`recall`, `copying`, `memory`, `state_tracking`, `compression`, `arithmetic`
(*informational*) — each scored as the mean of its benchmarks. `eval` writes a
JSON artifact (arch, size, params, commit, seed, per-benchmark + per-axis
results, gating pass-rate); `compare` diffs ≥2 side-by-side. `results/` is
git-ignored except two committed examples.

> Non-GPT caveat: `mad_compress` is a bottleneck autoencoder (MSE head), not a
> next-token decoder, so it ignores the supplied `DecoderLm` — its `compression`
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
  `capscale::grid_for` — the fit/advisor are dimension-agnostic.
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
and time?** Full design (including the scenarios not yet built) in
`docs/performance/benchmarking.md`; what exists in `docs/performance/status.md`.

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
`capability::Provider` is benchmarkable with no new benchmark code** — the reason
adopting that seam is worth doing per model.

Rules the harness enforces (each exists because violating it produces a
flattering-but-wrong number):

- warm-up requests never enter a statistic; failed/unfinished requests are never
  goodput and never leave the denominator;
- unmeasured fields serialise as `null`, never `0`, and an ungated run reports
  `correctness.passed: null` — never `true`;
- **goodput** (output meeting the SLO) is the comparison metric, not peak rate;
- `compare` refuses to rank across artifact units, excludes `valid: false` runs,
  and warns on every environment/workload axis that differs;
- a software rasteriser is labelled as one everywhere it appears.

**All 14 scenarios are implemented**: Tier 1 (`latency`, `throughput`, `serve`,
`sweep`, `startup`) plus Tier 2 — `mixed`, `overload`, `cancel`, `kvcache`,
`residency` ★, `placement`, `frontend`, `faults`, `soak` — the ones where brain
has something to measure that a single-model, single-GPU, HTTP-shaped harness
structurally cannot. Cross-cutting: `fidelity` (correctness gate) and `energy`.

Each scenario states what it *cannot* see: where a metric needs an engine
capability that does not exist (a pluggable admission policy, prefix caching, a
pipeline cache, a multi-rank harness), the field is `null` and the artifact
carries a `notes` string explaining why. See `docs/performance/status.md` for the
per-scenario table and the findings so far.

## Conventions & invariants

- **Zero compile warnings. Always.** A build that emits warnings is not done.
  Fix every warning the build reports — **including ones your change did not
  cause**. "Pre-existing" is not an exemption: warnings are only ever pre-existing
  because someone before you applied that exemption, and a noisy build is how a
  real defect hides in the scroll-back. Fix them **properly** — delete the dead
  code, use the unused binding, remove the stale `mut`, handle the ignored
  `Result`. Silencing with `#[allow(...)]`, `let _ =`, or an `_`-prefixed name is
  acceptable ONLY when the construct is genuinely intentional, and then it carries
  a comment saying why. Never suppress a warning to make a build quiet.

- **One implementation. Never re-implement anything that already exists in this
  workspace — no matter what it is.** This is the rule that most needs enforcing:
  before writing a function, search for it. `rmsnorm` once existed **seven**
  times (one WGSL kernel plus six host copies in `kronos`, `tts`, `chronos2`,
  `fincast`, `zimage` and `codec`), `rope` three times, `silu` four times. Every
  copy is a place the epsilon, the RoPE layout or the reduction order can drift
  from the kernel that is supposed to be authoritative, and nothing compares
  copies against each other.

  | need | where it belongs |
  |---|---|
  | math that runs on a device | a WGSL kernel in `crates/kernels/wgsl/`, dispatched via `gpu_core` |
  | math that genuinely runs on the host | **`model::hostmath`** — and nowhere else |
  | CPU-parallel execution (rayon) | `backend_cpu::par` only — the on-CPU scheduler's primitives; no other crate may depend on rayon |
  | shared model blocks | `model::block`, `model::vit` |
  | ONNX graph emission (DSL + shared norm/silu emitters) | `crates/npu/src/topo.rs` (`TopoBase`); model-specific graphs stay in `crates/npu/src/*_topology.rs` |

  Do **not** wrap a shared function in a local alias "for readability"
  (`fn silu(x) { hostmath::silu(x) }`). A local name is how a shared function
  becomes a private copy at the next edit. Call it directly.

  Two narrow exceptions, both of which must say why in a comment:
  1. a **gradcheck oracle** may re-derive the math independently (usually in
     `f64`) — an oracle that shares code with the thing it checks proves
     nothing (`zimage::grad`);
  2. a **backend fast path** implements an op for its device and is validated
     against the WGSL reference (`backend-cpu::fast_ops`).

- **One GPU device per process.** Building a `Gpu` per model object deadlocks
  the driver under concurrency and a device leaked into process exit crashes
  it. Production code shares explicitly (`Gpu::share` for the same kernel set,
  `Gpu::new_like` for a different set on the same device, `share_or_new` when
  the backend may not support sharing); **test binaries use
  `gpu_core::testgpu::dev(KERNELS)`** — a weak pool whose device dies with its
  last in-process handle. Never write a per-crate fixture; that is how
  duplicate fixtures (and the crash) come back.

- **Host math does not run on the accelerator.** Anything in `model::hostmath`
  is invisible to `--device`: it will not use the GPU, Vulkan or the NPU
  whatever the user asked for, and a benchmark of such a path reports host
  numbers under a device label. Host math is for `m=1` decode steps, references
  and glue — never for a hot path. If it is hot, it needs a kernel.


- **WGSL is the source of truth.** Kernels live only in `crates/kernels/wgsl/`,
  embedded as consts; no kernel text is duplicated. After adding/removing a
  `.wgsl`, run **`make kernels-regen`** (`scripts/kernels-regen.sh`) to
  regenerate the const block + `ALL` registry in `crates/kernels/src/lib.rs`.
- **fp32 arithmetic only, core compute only** — single bind group, **≤8 storage
  buffers/kernel** (the WebGPU guarantee; the splat backward kernels bind 8),
  **no atomics, no subgroups, no f16** (the only mentions of those in the kernel
  tree are comments asserting their absence).
  *This is a rule about the arithmetic datatype, NOT about storage precision —
  do not read it as "brain is fp32-only".* brain has a full **INT8** path:
  per-channel symmetric weights packed 4-per-`u32` (`model::int8`), DP4A GEMMs
  (`matmul_i8`, `matmul_i8_dyn`, `matmul_i8_gemv`, ~4× the fp32 rate on Pascal),
  dynamic per-token activation scales (`max_abs_row` → `quant_pack`), and int8
  paged KV. Norms/RoPE/attention stay fp32. Quantizing is the FIRST tool for
  fitting a large model on a card (`zimage::int8` for a DiT, `qwen::q8` for an
  encoder: ~16 GB → ~4.8 GB), ahead of sharding. `@workgroup_size(64)` is the rule;
  the register-tiled matmuls (`matmul_reg*.wgsl`, `matmul_dw_reg.wgsl`,
  `matmul_dx_reg.wgsl`, `matmul_i8*.wgsl`) and `flash_attn_bidir_split.wgsl`
  use 256 — every one of them because a thread cooperates over a tile, and each
  must be gated on the device's **queried** `DeviceCaps::max_workgroup_size`
  (256 is the WebGPU floor, so a 64-thread fallback stays selectable).
  This is what keeps the engine portable to old GPUs and WebGPU.
- **Never put a large `var<function>` array behind a runtime loop bound.** WGSL
  function-scope arrays only become registers if the compiler can unroll every
  index; bound the loop by a `Params` field and the array lands in *local*
  memory (global-backed), and the kernel silently runs at memory bandwidth.
  This cost the FLUX.2 DiT 81 % of its forward — see
  `docs/performance/overview.md` for the pattern and the fix.
- **One thread per row is a COALESCING bug, at every row count.** A per-element
  norm/reduction kernel that gives thread *t* row *t* makes a warp's 32 loads
  `d` floats apart, so each 32-byte sector fetched serves ONE useful float — 8×
  read and write amplification that more rows do not fix (the loss is
  per-access efficiency, not thread count). The cooperative `*_rows` family
  (`rmsnorm_rows`, `softmax_rows`, and now `layernorm_rows` / `ln_stats_rows` /
  `layernorm_dx_rows`) walks one row with a 64-thread workgroup and is coalesced
  by construction: measured **19.4×** for QK-norm and **2.3–9.1×** for the
  LayerNorm family on a P40. `backend_api::select` (`Op::RmsNorm`,
  `Op::LayerNorm`) picks them wherever the queried
  `DeviceCaps::workgroup_reductions` holds, and `model::block`'s
  `layernorm_fwd` / `ln_stats_fwd` / `layernorm_dx_bwd` are the dispatch seam.
  Each carries exactly **one** top-level `workgroupBarrier()` — the CPU JIT
  splits a body at one barrier and no more, which is why they use a *shifted*
  single-pass mean/variance instead of the textbook two-pass.
- **Three backends, one build, one API.** `gpu-core` exposes a single
  `Gpu`/`DeviceBuffer`/`Step` surface; every model is written once against it.
  The accelerator is the *only* thing abstracted — there is no per-backend model
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
- **Kernels follow `docs/kernel-checklist.md`** — before writing one, check for
  an existing fast sibling (`_rows`/`_wg`/`_reg*`/`_tiled`) and put the fix in
  *selection*, not a new copy: the single most expensive defect class here is a
  fast kernel a later model never learned about (`gn_stats`, fixed in 2025,
  re-cost 159× in `vae`). Before dispatching one, read its `Params` struct and
  copy a working call site — a mismatched param list is silently wrong, not a
  crash (`silu_mul` → cosine 0.504). Before optimizing, profile per kernel-kind
  and publish the table: every confident hypothesis on this engine has been
  wrong, and the profile has been right.
- **New model ports follow `docs/porting-playbook.md`** — reference goldens
  dumped FIRST (transformer I/O captured via forward hooks, replayed in the
  parity test), two-way import coverage, kernel Params read before dispatch,
  tiny-config smoke with step bisection, then the parity ladder
  (stage → forward → composed loop → real run). It encodes the exact failure
  modes already paid for; do not rediscover them.
- **Backprop is gated by `gradcheck`** (finite differences) — run it after any
  fwd/bwd math change. Entry points today: `check_gpt`, `check_qwen`,
  `check_qwen_lora`, `check_moe`, `check_glm`, `check_glm_mtp`, `check_pid`,
  `check_seq2seq`, `check_autoencoder`. SSA-style forward (each stage writes a
  fresh buffer that doubles as the backprop activation cache) — preserve it when
  adding stages.
- **Imported models are parity-gated, not gradient-guessed.** `mirror`,
  `chronos2`, `kronos`, `fincast`, `depth`, `wm-diamond` are imported 1:1 from a
  reference checkpoint and verified stage-by-stage against dumped goldens
  (`scripts/parity-dump/`, `tools/*_dump_reference.py`). `make parity` is the
  cross-backend gate (CPU == Vulkan == NPU).
- **Adding a capability ≠ adding a subcommand.** Implement `capability::Action`
  and list it in a `Provider`; `brain do` and the event API pick it up.
- **Every new model ships the full serving contract — code is not "done" until it
  is served.** Adding a model means, in the same change:
  1. a **`capability::Provider`** (or a manifest via its `ResidentModel`) exposing
     its actions through the generalized interface — never a bespoke subcommand;
  2. a **residency adapter** (`crates/cli/src/resident_*.rs`, registered in
     `resident::build_executor`, env-gated) so it is **scheduled**, memory-budgeted,
     and swappable by the `Executor` like every other model;
  3. **true batching**: implement `Instance::run_batch` with a genuine batched
     forward wherever the architecture allows (see `resident_asr`/`resident.rs`
     yolo) — never leave concurrent same-model work on the default serial loop
     without saying why;
  4. **D-Bus wiring + a runnable example.** The model's actions MUST be reachable
     over `crates/dbus` (`com.swedishembedded.Brain1`) and demonstrated by an
     example under `examples/<domain>/` with a README. If the model's shape fits the
     existing D-Bus surface (`Run`/`Subscribe`/`StreamTranscribe`/fd blobs), use it;
     if it does not, **extend or refactor the surface** (add a method, generalize a
     frame type) rather than bolting on a side channel — and update every existing
     client/example that the change touches. The full checklist lives in
     `docs/serving-contract.md` (linked from the Serving stack section); keep it and
     this bullet in sync.
  A model that trains and passes parity but cannot be discovered, scheduled, batched,
  and driven over D-Bus is **incomplete**.
- **Every change to an API surface triggers a full API security audit.** Any change
  to `crates/apiserve` (the HTTP providers) OR `crates/dbus` (the D-Bus surface) —
  a route, handler, auth path, error shape, admission policy, or exposed method —
  requires auditing the **whole** API against **`docs/api-security-audit.md`** (authn/
  authz, input/DoS bounds, admission/backpressure, cancel-on-disconnect, SSRF/egress,
  error hygiene, transport), not just the changed handler. Run the pass with the
  `security-review` skill; fix findings before the change is done. These surfaces are
  internet-reachable when bound, so all request input is hostile.
- **API specs: at most two sources of truth.** brain's code (what it implements) and
  the **vendored upstream OpenAPI specs** (`crates/apiserve/tests/specs/`, a cached
  copy of what providers support) — validated against each other by the jsonschema
  conformance tests. There is **no** separate hand-maintained "brain spec." Refresh
  the vendored specs from upstream with the **`api-sync`** command (`.claude/commands/
  api-sync.md`), then adapt brain to any drift and re-green the conformance tests.
- **No absolute paths in source — anywhere.** Never hardcode a machine-specific
  absolute path (`/data/…`, `/home/…`, `/tmp/…`) in `crates/**`: not in code, not
  in a test `const`, not as a runtime default, not in a doc comment. Two homes for
  what used to be hardcoded:
  1. **Test / parity fixtures** live under the **gitignored `testdata/` tree**,
     resolved at runtime from `$BRAIN_TESTDATA` (default `<repo>/testdata`, via a
     `#[cfg(test)] fn testdata(rel)` helper per crate). A test **skips itself** when
     its fixture is absent. Populate the tree with **`make fetch/testdata`**
     (`scripts/fetch-testdata.sh`) — it hard-links from a local mirror or downloads
     from a URL, fetching only files not already present, organised as a tree
     (`testdata/<domain>/<model>/…`). The mirror location is an overridable script
     variable — the ONE permitted place a machine path may appear.
  2. **In-repo artifacts** (`out/…` build outputs, `scratchpad/…`) are resolved
     **repo-relative** (`concat!(env!("CARGO_MANIFEST_DIR"), "/../../out/…")`), never
     as an absolute literal.
  Runtime weight locations come from an **env var or CLI flag**, never a baked-in
  path. Grep gate (a string literal that *starts* an absolute machine path):
  `grep -rnE '"/(data|home|tmp|opt|mnt|root)/' crates` must stay empty. (A `/data/`
  substring mid-string — a URL, or a torch-archive-internal `…/data/<key>` — is
  not a filesystem path and is fine.)
- **Evaluate honestly.** Hold the input distribution fixed; separate the metric
  (perplexity) from the task (exact-match on held-out data); see `README.md` §3.
- **Gitignored:** `scratchpad/` (scratch weights, images, porting references),
  generated `data/`, `out/`, `build/`, `results/*.json`, and the world-model
  parity fixtures.

---

## Doc gaps (as of this revision)

Known-stale or missing, in rough priority order:

1. **No `docs/models/` entry** for: `qwen` (incl. the whole paged-KV serving
   workstream), `zimage`/`dit`/`diffusion`/`vae`, `gpt`, `moe`, `pid`, `seq2seq`.
2. **No `status.md` ledger** for `qwen`, `tts`, `glm`, `yolo`, `zimage` (the
   models that have one: `depth`, `mirror`, `splat`, `chronos2`, `kronos`,
   `fincast`, `world-models`, and now `perf`).
3. The serving/runtime stack (`capability`, `residency`, `server`, `dbus`) has
   rich crate-level rustdoc but **no prose doc** — the table above is currently
   the only map of it.

Fixed in this revision: the CLI `HELP` now documents `qwen`, `glm`, `tts`,
`depth`, `forecast`, `caps`/`do` and `perf`; `docs/architecture.md`'s crate graph
and its "≤4 storage buffers" invariant are corrected.

Not a gap, despite appearances: `federated-moe.md` at the repo root is the
3148-line *source design* essay; `docs/federated.md` is the shorter "what brain
implements" doc and cites it deliberately. Keep both.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end Z-Image **text-to-image**: the individually-validated components
//! (Qwen-4B encoder, the S³-DiT, the FLUX VAE, the flow-match scheduler) wired
//! into one `generate`, following the diffusers `ZImagePipeline.__call__` recipe:
//!
//!   1. chat-template + tokenize the prompt, Qwen-4B → `hidden_states[-2]` (the
//!      caption features the DiT conditions on);
//!   2. a seeded Gaussian latent `[16, 1, H/8, W/8]`;
//!   3. an 8-step flow-match Euler sampler over the DiT — each step the DiT
//!      predicts velocity at `t = (1000 - t_sched)/1000`, the scheduler advances
//!      `x += (σ_next − σ)·v`; dynamic-shifted sigmas (mu from the sequence length);
//!   4. VAE decode of `latent/scaling + shift` → RGB, `[-1,1] → [0,1]`.
//!
//! Heavy compute stays on the GPU: the encoder runs INFERENCE-ONLY (Frozen, ~16 GB
//! weights — no train buffers) then drops; the DiT samples in int8 (13 GB); the VAE
//! decodes on-device. Peak VRAM is one model at a time — all under a 24 GB P40.
//! Models load sequentially and drop before the next, so peak VRAM is one model.

use std::collections::HashMap;

use data::qwen_tokenizer::QwenBpe;
use data::Tokenizer;
use diffusion::{default_z_image_sigmas, FlowMatchConfig, FlowMatchEulerScheduler};
use qwen3::{Qwen, QwenConfig, Shard, IGNORE};
use vae::{VaeConfig, VaeDecoder, VaeEncoder};

use crate::import::import_comfy;
use crate::{ZImageConfig, ZImageDitI8, ZImageDitShard, ZImageDitWindowed};

/// Filesystem locations of the four Z-Image components (never hard-coded — from
/// the environment, mirroring the crate's tests).
/// Qwen `<|endoftext|>` — the pad id the masked caption-encoder path uses
/// (same id `flux2::pipeline::PAD_TOKEN` uses for the same tokenizer family).
const PAD_TOKEN: u32 = 151643;

#[derive(Clone)]
pub struct Paths {
    pub dit: String,
    pub vae: String,
    pub qwen: String,
    pub tokenizer: String,
}

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        let g = |k: &str| std::env::var(k).map_err(|_| format!("set {k} to the Z-Image {k} path"));
        Ok(Paths { dit: g("BRAIN_S3DIT_DIT")?, vae: g("BRAIN_S3DIT_VAE")?, qwen: g("BRAIN_S3DIT_QWEN")?, tokenizer: g("BRAIN_S3DIT_TOKENIZER")? })
    }
}

/// Read a component's tensors from `path`: a single safetensors file, or an
/// HF-style directory (single `model.safetensors` or a sharded
/// `model.safetensors.index.json` + `model-*.safetensors` set) --
/// `crates/flux2/src/pipeline.rs` already gives its text-encoder path this
/// leniency; every Z-Image component gets it too, so a directly-fetched HF
/// `Tongyi-MAI/Z-Image-Turbo` tree (each component its own subdirectory,
/// `transformer/`/`text_encoder/` sharded) loads with no repacking.
fn read_component_tensors(path: &str) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        checkpoint::safetensors::read_model_dir(p)
    } else {
        checkpoint::safetensors::read(path)
    }
}

/// The streaming sibling of [`read_component_tensors`]: open the same single
/// file or HF-style directory as a header-only mmap
/// (`checkpoint::weightio::WeightReader`) — no tensor bytes read. Used for the
/// encoder and the (no-adapter) DiT build, whose weights stream straight to
/// the device one at a time; [`read_component_tensors`] stays for the VAE
/// (168 MB, not the acute OOM) and the LoRA-adapter-folding path, which
/// mutates weights in place and therefore needs them all in host memory
/// regardless of how they were read.
fn open_component(path: &str) -> Result<checkpoint::weightio::WeightReader, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        checkpoint::weightio::WeightReader::open_hf_dir(p).map_err(|e| e.to_string())
    } else {
        checkpoint::weightio::WeightReader::open(path).map_err(|e| e.to_string())
    }
}

/// Generation options.
pub struct Opts {
    pub steps: u32,
    pub guidance: f32,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    /// High-fidelity DiT: `false` = int8 on one P40 (~13 GB, cosine 0.99, fast);
    /// `true` = full-precision fp32 sharded across both P40s (higher fidelity, no
    /// quantisation error, needs 2 GPUs).
    pub hifi: bool,
}

/// The sampler's starting state: `(latent, first scheduler step, inpaint mask
/// in latent space, the VAE-encoded init latent)`. The last two are `None` for
/// a plain text2image run, which starts from pure noise at step 0.
type LatentInit = (Vec<f32>, usize, Option<Vec<f32>>, Option<Vec<f32>>);

/// The denoiser backend chosen by [`Opts::hifi`]. All three expose the same
/// `forward(latent, cap, t)`; the sampler is identical either way.
enum DitEngine {
    I8(Box<ZImageDitI8>),
    Shard(Box<ZImageDitShard>),
    /// fp32 on a box with fewer than 2 GPUs: `ZImageDitShard` is
    /// unconditionally 2-GPU and fails outright there ("need 2 discrete
    /// GPUs, found 0"), so the single-GPU path streams the main layer stack
    /// through a weight window instead of sharding it. `dit_path`/`cfg` are
    /// kept so `forward` can hand the window a *fresh* streaming source
    /// every call — the same checkpoint `build_from_source` opened, since a
    /// rotating slot's miss must be able to re-read it an unbounded number
    /// of calls later.
    Windowed { dit: Box<ZImageDitWindowed>, dit_path: String, cfg: ZImageConfig },
}

/// The fp32 ("hifi") DiT needs 2 GPUs to shard (`ZImageDitShard`) OR a
/// weight window on 1 (`ZImageDitWindowed`) — a real machine-shape fact
/// (how many GPUs actually exist), never a size heuristic, mirroring
/// `pipeline::default_bulk_gpu`'s identical reasoning for the encoder.
pub fn hifi_needs_window(gpu_count: usize) -> bool {
    gpu_count < 2
}

/// How many of the main stack's blocks stay resident in the fp32 single-GPU
/// window. `BRAIN_S3DIT_WINDOW_BLOCKS` overrides; the default (2) keeps the
/// window's own device footprint small (~1.4 GB at Z-Image-Turbo's real
/// shape — dim 3840, hidden 10240) regardless of model size, at the cost of
/// reloading every other block once per denoise step.
pub fn window_blocks_from_env() -> u32 {
    std::env::var("BRAIN_S3DIT_WINDOW_BLOCKS").ok().and_then(|s| s.parse().ok()).filter(|&n| n > 0).unwrap_or(2)
}

/// The real fp32 ("hifi") cost the residency layer should budget against —
/// `(vram_bytes, ram_bytes)` — computed from the SAME machine-shape decision
/// (`hifi_needs_window`) and the SAME per-block shape `ZImageDitWindowed`
/// actually allocates, not a separate hardcoded guess: the number the
/// scheduler budgets and the number the code allocates must be the same
/// expression, or a claim like "fp32 needs 24 GB" silently outlives the
/// windowed engine that made it false on a one-GPU box.
///
/// `ram_bytes` is the CPU-resident fp32 Qwen-4B encoder's footprint (~16 GB,
/// measured — `pipeline.rs`'s own doc comments quote this figure for the
/// CPU-resident case throughout): today the hifi path always builds the
/// encoder on CPU regardless of GPU count (a separate, narrower gap from
/// the one `hifi_needs_window` closes — the int8 path's analogous default
/// was fixed by `default_bulk_gpu`, hifi's was not, in this session).
pub fn hifi_cost_bytes(gpu_count: usize) -> (u64, u64) {
    const ENCODER_RAM: u64 = 16 << 30;
    if !hifi_needs_window(gpu_count) {
        return (24 << 30, ENCODER_RAM); // unchanged: ZImageDitShard, 2 GPUs
    }
    (windowed_dit_vram_bytes(&ZImageConfig::turbo(), window_blocks_from_env()), ENCODER_RAM)
}

/// VRAM for `window` main-stack blocks plus the fully-resident refiners, at
/// `cfg`'s real block shape — the exact element counts
/// [`crate::block::BlockWeights::alloc`]/[`crate::block::NormBufs::alloc`]
/// allocate (`wq`/`wk`/`wv`/`wo`: `dim×dim`; `nq`/`nk`: `head_dim`;
/// `w1`/`w2`/`w3`: `hidden×dim`; plus `NormBufs`'s four `dim`-sized
/// buffers), all `f32`. `overhead` is a generous flat allowance for the
/// residual/cos-sin/Scratch buffers next to the weights at Z-Image-Turbo's
/// real shape (dim 3840) — not a byte-exact derivation of every kernel's
/// intermediate buffer.
fn windowed_dit_vram_bytes(cfg: &ZImageConfig, window: u32) -> u64 {
    let bd = cfg.block_dims();
    let (dim, hidden, head_dim) = (bd.dim as u64, bd.hidden as u64, bd.head_dim as u64);
    let per_block_elems = 4 * dim * dim + 2 * head_dim + 3 * hidden * dim + 4 * dim;
    let per_block_bytes = per_block_elems * 4;
    let refiner_blocks = 2 * cfg.n_refiner_layers as u64; // noise + context, fully resident
    let window = (window as u64).min(cfg.n_layers as u64);
    let overhead = 256u64 << 20;
    per_block_bytes * (refiner_blocks + window) + overhead
}

/// The dominant term of a plain-int8 [`crate::DitI8Cache`]'s host bytes at
/// `cfg`'s shape — the packed int8 weights (`wq`/`wk`/`wv`/`wo`/`w1`/`w2`/`w3`,
/// 1 byte/element once packed), leaving out the much smaller per-row f32
/// scales and norm arrays. Not bit-exact like [`windowed_dit_vram_bytes`]
/// (that one backs a hard device-fit decision; this one backs an
/// observability/estimate number) — real and order-of-magnitude correct is
/// the bar: `ResidentModel::estimate_at(Warm)` reporting `0` for an
/// instance that actually retained ~5 GB would be a worse lie than a
/// slightly-approximate few-GB figure.
pub fn int8_cache_bytes_estimate(cfg: &ZImageConfig) -> u64 {
    let bd = cfg.block_dims();
    let (dim, hidden) = (bd.dim as u64, bd.hidden as u64);
    let packed_per_block = 4 * dim * dim + 3 * hidden * dim;
    let blocks = cfg.n_layers as u64 + 2 * cfg.n_refiner_layers as u64;
    packed_per_block * blocks
}

impl DitEngine {
    fn build(hifi: bool, cfg: ZImageConfig, weights: crate::block::Tensors, lh: u32, lw: u32, cap_len: u32) -> DitEngine {
        // The adapter/LoRA-folding path (the only caller of `build`) has no
        // on-disk checkpoint to reopen — the weights it holds are an
        // in-memory map already mutated by the fold, never written back to
        // disk — so `Windowed` (which must reopen a real file on every
        // miss) is not reachable from here; `hifi` still gets `Shard`
        // unconditionally, the pre-existing (documented) 2-GPU requirement
        // for fp32 + a LoRA adapter together.
        if hifi {
            DitEngine::Shard(Box::new(ZImageDitShard::build(cfg, weights, 1, lh, lw, cap_len)))
        } else {
            DitEngine::I8(Box::new(ZImageDitI8::build(cfg, weights, 1, lh, lw, cap_len)))
        }
    }

    /// [`Self::build`] over a streaming `checkpoint::TensorSource` (a
    /// `crate::import::comfy_source` over a mmap'd `WeightReader`, in the
    /// production loader below) — never materializes the whole DiT
    /// checkpoint on the host. `dit_path` is only used by the `Windowed`
    /// case (hifi, fewer than 2 GPUs) to reopen the checkpoint fresh on
    /// every later `forward` call.
    fn build_from_source(hifi: bool, cfg: ZImageConfig, src: &dyn checkpoint::TensorSource, dit_path: &str, lh: u32, lw: u32, cap_len: u32) -> DitEngine {
        if !hifi {
            return DitEngine::I8(Box::new(ZImageDitI8::build_from_source(cfg, src, 1, lh, lw, cap_len)));
        }
        if hifi_needs_window(gpu_core::devices::schedulable_gpu_count()) {
            let window = window_blocks_from_env();
            let dit = ZImageDitWindowed::build_from_source(cfg.clone(), src, window, 1, lh, lw, cap_len, Some("gpu"));
            DitEngine::Windowed { dit: Box::new(dit), dit_path: dit_path.to_string(), cfg }
        } else {
            DitEngine::Shard(Box::new(ZImageDitShard::build_from_source(cfg, src, 1, lh, lw, cap_len)))
        }
    }
    fn forward(&self, latent: &[f32], cap: &[f32], t: f32) -> Vec<f32> {
        match self {
            DitEngine::I8(d) => d.forward(latent, cap, t),
            DitEngine::Shard(d) => d.forward(latent, cap, t),
            DitEngine::Windowed { dit, dit_path, cfg } => {
                let reader = open_component(dit_path).unwrap_or_else(|e| panic!("re-opening the DiT checkpoint for a windowed forward: {e}"));
                let src = crate::import::comfy_source(&reader, cfg);
                dit.forward(&src, latent, cap, t)
            }
        }
    }

    /// Observability for the churn claim: `0` for `I8`/`Shard` (nothing to
    /// stream, every block resident since build), the windowed engine's
    /// real reload count otherwise — the number that answers "is this
    /// actually reloading every block every step, or holding the pinned
    /// prefix as designed" without re-deriving it from timing.
    fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        match self {
            DitEngine::I8(_) | DitEngine::Shard(_) => Vec::new(),
            DitEngine::Windowed { dit, .. } => vec![("dit_window_reloads".to_string(), serde_json::json!(dit.main_reloads()))],
        }
    }
}

/// Build the `DitEngine` for `(hifi, adapter)` on `dit_gpu` — extracted out
/// of `build_adapted` verbatim so `rebuild_dit_from_i8_cache` (the promote
/// path) can share every OTHER piece of `build_adapted` (encoder, VAE) and
/// differ on only this one, swapping it for `ZImageDitI8::rebuild_from_cache`.
/// The DiT config for a checkpoint at `dit_dir` — the ONE seam every build
/// site in this file routes through (audit F44: three sites hardcoded
/// `ZImageConfig::turbo()` independently, so per-checkpoint support would
/// have to land three times). Today it still returns the shipped Turbo
/// config unconditionally: the released Z-Image-Turbo is the only known
/// checkpoint, and its `transformer/config.json` key map has not been
/// verified in-repo, so deriving field values here would be guesswork.
/// When a second checkpoint (or the verified key map) lands, teach THIS
/// function to read `<dit_dir>/config.json`; no build site needs touching.
fn dit_config(_dit_dir: &str) -> ZImageConfig {
    ZImageConfig::turbo()
}

fn build_dit_engine(paths: &Paths, hifi: bool, adapter: Option<&str>, dit_gpu: u32, lh: u32, lw: u32, cap_len: u32, progress: &mut dyn FnMut(&str)) -> Result<DitEngine, String> {
    progress(if hifi { "building DiT (fp32, 2×GPU)" } else { "building DiT (int8, GPU)" });
    let zcfg = dit_config(&paths.dit);
    if let Some(ap) = adapter {
        // LoRA folding mutates weights element-wise in place, so it needs an
        // owned, complete map regardless of how the checkpoint is read --
        // eager here is the adapter path's actual requirement, not a
        // leftover the loader fix missed. Still gets the retention fix:
        // `weights` drops at the end of this block, and the built engine
        // (ZImageDitI8/Shard) only ever keeps its 13-tensor HostWeights
        // (see crate::dev's doc).
        let mut weights = import_comfy(read_component_tensors(&paths.dit).map_err(|e| format!("read dit: {e}"))?, &zcfg);
        progress(&format!("folding LoRA adapter {ap}"));
        let tcfg = crate::finetune::train_cfg(&zcfg, lh, lw, cap_len);
        crate::finetune::load_adapter_folded(ap, &tcfg, &mut weights)?;
        gpu_core::devices::with_gpu(dit_gpu, || DitEngine::build(hifi, zcfg.clone(), weights, lh, lw, cap_len))
    } else {
        // Streaming: one tensor at a time, straight to the device (via
        // block::upload_named's chunk-bounded path). Peak host allocation
        // for the DiT's weights is then one tensor (~157 MB for
        // feed_forward.w1 once BF16 is converted), never the whole ~24 GB
        // model this replaces (import_comfy(read_component_tensors(..))
        // materializing the entire checkpoint up front).
        let dreader = open_component(&paths.dit).map_err(|e| format!("open dit: {e}"))?;
        let dsrc = crate::import::comfy_source(&dreader, &zcfg);
        gpu_core::devices::with_gpu(dit_gpu, || DitEngine::build_from_source(hifi, zcfg.clone(), &dsrc, &paths.dit, lh, lw, cap_len))
    }
}

/// A **resident** text-to-image pipeline: the Qwen-4B encoder (CPU), the DiT
/// (int8 on one P40, or fp32 sharded across both), and the VAE decoder are built
/// ONCE for a fixed output size and caption length, then reused across many
/// generations — no ~20 GB reload per image. Each model keeps its own device
/// handle from build time, so [`generate`](Self::generate) just runs forwards.
/// Captions are padded/truncated to `cap_len` so the built graphs stay valid for
/// any prompt (masked pad: `<|endoftext|>` pad tokens, excluded as attention
/// keys past the true content length — HF attention_mask semantics).
/// `BRAIN_S3DIT_ENCODER_RESIDENT=1`: opt back into a resident encoder even
/// when it shares the DiT's card (a box with room for both). Pure function
/// of the environment so [`encoder_on_demand`]'s decision stays testable.
fn resident_override() -> bool {
    std::env::var("BRAIN_S3DIT_ENCODER_RESIDENT").ok().as_deref() == Some("1")
}

/// Whether the int8 GPU encoder should be built on-demand (never resident
/// alongside the DiT) rather than resident for the pipeline's whole life.
/// True exactly when the encoder was asked to share the DiT's own card --
/// the only case where residency would make the two compete for one card's
/// budget at once (~9.5 GB encoder + ~13 GB DiT) -- and the caller hasn't
/// opted back into the old both-resident behaviour.
fn encoder_on_demand(bulk: u32, dit_gpu: u32, resident_override: bool) -> bool {
    bulk == dit_gpu && !resident_override
}

/// The encoder's GPU card, defaulted when the caller gave no explicit
/// `BRAIN_S3DIT_ENCODER_GPU`. On a box with more than one GPU, which card
/// should host the encoder is a real choice (a second, otherwise-idle card
/// vs. sharing the DiT's) that only the caller can make — `None` (CPU) stays
/// the conservative default there, unchanged. On a box with exactly one GPU
/// there is no second card to choose, so CPU-by-default would only be
/// trading a smaller, on-demand int8 encoder for a larger, permanently
/// resident fp32 one — defaulting to sharing the DiT's card is strictly
/// better and is what makes memory residency automatic rather than a flag
/// the caller has to know to set.
fn default_bulk_gpu(explicit: Option<u32>, dit_gpu: u32, gpu_count: usize) -> Option<u32> {
    explicit.or(if gpu_count == 1 { Some(dit_gpu) } else { None })
}

/// Where/how the Qwen-4B text encoder runs.
enum Encoder {
    /// Whole model on the CPU (default) — no VRAM cost, ~38 s/encode.
    Cpu(Box<Qwen>),
    /// Whole int8 encoder on one card. The 7 per-layer linears are DP4A int8
    /// (~4× smaller than fp32), so the whole Qwen3-4B encoder is ~9.5 GB resident
    /// and fits a single 24 GB card alongside nothing else — leaving the DiT its
    /// own card. Encode runs on-GPU (~1-2 s). The robust "superfast" path; the
    /// fp32 [`Encoder::Split`] does not fit two P40s (2× non-ReBAR overhead).
    Gpu8(Box<Qwen>),
    /// Split across two cards: `s0` (embedding + the first `cut` layers) on the
    /// mostly-empty card, `s1` (the remaining layers up to the penultimate) on the
    /// DiT's card. The fp32 encoder is ~23 GB resident — too big for one 24 GB
    /// card, but a thin tail fits alongside the 13 GB int8 DiT while the bulk sits
    /// on the spare card. Encode runs on-GPU (~1-2 s) with one small host-staged
    /// residual at the cut.
    Split { s0: Box<Qwen>, s1: Box<Qwen>, cap_len: u32 },
    /// The single-GPU case: the int8 encoder (~9.5 GB) and the int8 DiT (~13 GB)
    /// would together exceed a 24 GB card's usable budget (and comfortably
    /// exceed a unified-memory box's smaller one) if both stayed resident at
    /// once, so this variant is never resident between calls — `encode` builds
    /// it fresh from `qreader_path`, runs the one forward it needs, and drops it
    /// before returning, exactly mirroring the SDXL fix: build, use once,
    /// drop, THEN run the part that stays resident (here, the
    /// DiT sampling loop). Costs a rebuild (~1-2 s: open + int8 quantize +
    /// upload) every `generate()` call instead of once at `build()` time — the
    /// deliberate trade for never holding both models on the card together.
    /// `BRAIN_S3DIT_ENCODER_RESIDENT=1` opts back into [`Encoder::Gpu8`] on a
    /// box with enough VRAM to hold both (a 24 GB+ discrete card).
    OnDemand { qreader_path: String, qcfg: QwenConfig, cap_len: u32, gpu: u32 },
}

impl Encoder {
    /// Encode `tokens` (a right-padded `cap_len` sequence) with the pads
    /// EXCLUDED as attention keys past `content_len` — the HF
    /// `attention_mask` semantics (`Qwen::encode_padded`). An exact-length
    /// prompt (`content_len == tokens.len()`) takes the original unmasked
    /// path unchanged.
    fn encode(&self, tokens: &[u32], content_len: usize) -> Result<Vec<f32>, String> {
        let padded = content_len < tokens.len();
        match self {
            Encoder::Cpu(q) | Encoder::Gpu8(q) => Ok(if padded { q.encode_padded(tokens, content_len) } else { q.encode(tokens) }),
            Encoder::Split { s0, s1, cap_len } => {
                // Targets are unused (we read a hidden state, not a loss).
                let ign = vec![IGNORE; *cap_len as usize];
                if padded {
                    s0.arm_pad_kmask(tokens, content_len);
                    s1.arm_pad_kmask(tokens, content_len);
                }
                s0.set_batch(tokens, &ign);
                s0.run_forward(); // embed + layers 0..cut
                let boundary = s0.read_out_res(); // res[cut] (host)
                s1.write_in_res(&boundary); // res[cut] on the DiT card
                s1.run_forward(); // layers cut..n_layers-1
                let out = s1.read_out_res(); // res[n_layers-1] == penultimate hidden (== Qwen::encode)
                if padded {
                    s0.disarm_kmask();
                    s1.disarm_kmask();
                }
                Ok(out)
            }
            Encoder::OnDemand { qreader_path, qcfg, cap_len, gpu } => {
                let reader = open_component(qreader_path).map_err(|e| format!("open qwen: {e}"))?;
                let src = qwen3::import::hf_source(&reader, qcfg)?;
                let n = qcfg.n_layers as usize;
                let cap = gpu_core::devices::with_gpu(*gpu, || {
                    let e = Qwen::new_shard_i8(qcfg.clone(), 1, *cap_len, &src, Shard { start: 0, end: n - 1, embed: true, head: false, gpu_index: *gpu as usize });
                    if padded { e.encode_padded(tokens, content_len) } else { e.encode(tokens) }
                    // `e` drops here -- the encoder never outlives this call.
                })?;
                Ok(cap)
            }
        }
    }
}

pub struct HotPipeline {
    tok: QwenBpe,
    enc: Encoder,
    dit: DitEngine,
    vae: VaeDecoder,
    cap_len: u32,
    lh: u32,
    lw: u32,
    width: u32,
    height: u32,
    hifi: bool,
}

impl HotPipeline {
    pub fn cap_len(&self) -> u32 {
        self.cap_len
    }
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn hifi(&self) -> bool {
        self.hifi
    }

    /// Model-specific observability (`Instance::metrics`'s contract) — the
    /// weight-window's reload count when the DiT is streaming
    /// ([`DitEngine::Windowed`]), nothing otherwise.
    pub fn metrics(&self) -> Vec<(String, serde_json::Value)> {
        self.dit.metrics()
    }

    /// Build the resident models for `width×height`, `cap_len` caption tokens, and
    /// the chosen precision. This is the slow one-time step (~weights load + int8
    /// quantise / shard build); `generate` afterwards is fast. `progress(msg)`
    /// streams the build stages.
    pub fn build(paths: &Paths, width: u32, height: u32, cap_len: u32, hifi: bool, progress: impl FnMut(&str)) -> Result<HotPipeline, String> {
        Self::build_adapted(paths, width, height, cap_len, hifi, None, progress)
    }

    /// Like [`build`](Self::build), optionally folding a saved LoRA adapter into
    /// the DiT weights before the (int8/fp32) engine is built — so the resident
    /// pipeline generates adapter-conditioned images with no other change.
    pub fn build_adapted(paths: &Paths, width: u32, height: u32, cap_len: u32, hifi: bool, adapter: Option<&str>, mut progress: impl FnMut(&str)) -> Result<HotPipeline, String> {
        if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
            return Err("width/height must be multiples of 16".into());
        }
        let (lh, lw) = (height / 8, width / 8);

        progress("loading tokenizer");
        let tok = QwenBpe::from_file(&paths.tokenizer)?;

        // Where the Qwen-4B encoder runs. `BRAIN_S3DIT_ENCODER_GPU=<i>` (when NOT
        // hifi — hifi already uses both cards for the DiT) shards it across two
        // cards: the bulk (embedding + first ~¾ of the layers) on GPU `i` (the
        // otherwise-empty card) and the thin tail on the DiT's card. The whole fp32
        // encoder (~23 GB resident) does not fit one 24 GB card, but this split
        // does, and the encode then runs on-GPU (~1-2 s) instead of ~38 s on the
        // CPU. Unset ⇒ CPU. Card-agnostic: you choose the bulk-card index.
        let qcfg = QwenConfig::qwen3_4b();
        // Streaming: `qreader` mmaps the checkpoint's header only, and `qsrc`
        // resolves brain's parameter names against it with no tensor bytes
        // read yet -- `Qwen::new_shard{,_i8}` below pull one tensor at a time
        // straight to the device. This replaces the old
        // read_component_tensors -> brain_init_from_hf pair, which built two
        // full ~16 GB host copies of the Qwen3-4B encoder (the file read, then
        // the renamed map) before a single device byte was ever allocated.
        let qreader = open_component(&paths.qwen).map_err(|e| format!("open qwen: {e}"))?;
        let qsrc = qwen3::import::hf_source(&qreader, &qcfg)?;
        // User env input, parsed to canonical card indices at this edge; all
        // placement below goes through the device registry (explicit Shard
        // indices / scoped `with_gpu`), never env mutation.
        let enc_gpu_explicit: Option<u32> = std::env::var("BRAIN_S3DIT_ENCODER_GPU")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>().map_err(|_| format!("bad BRAIN_S3DIT_ENCODER_GPU {s:?}")))
            .transpose()?;
        // The DiT/VAE card: the ambient selection (`--device gpu<i>`, the
        // residency-assigned scope, or BRAIN_GPU_INDEX), canonical card 0 otherwise.
        let dit_gpu: u32 = gpu_core::devices::current_gpu().unwrap_or(0);
        // See default_bulk_gpu's doc: on a single-GPU box, defaulting to CPU
        // would only trade a smaller on-demand int8 encoder for a larger,
        // permanently resident fp32 one -- share the DiT's card instead.
        let enc_gpu = default_bulk_gpu(enc_gpu_explicit, dit_gpu, gpu_core::devices::schedulable_gpu_count());

        // For the 2-card encoder split we interleave the builds: bulk shard on the
        // empty GPU `bulk`, THEN the DiT on GPU `dit_gpu` (while that card is still
        // empty, so the DiT's transient upload-staging spike has headroom), THEN the
        // thin tail shard packed on top of the DiT. Building the tail last is what
        // makes GPU `dit_gpu` fit — the DiT's peak staging never overlaps the tail's
        // resident bytes. `split` carries the params needed to finish after the DiT.
        let mut split: Option<(Qwen, usize, usize, u32)> = None; // (s0, cut, n, di)
        let enc_cpu = match (enc_gpu, hifi) {
            (Some(bulk), false) if std::env::var("BRAIN_S3DIT_ENCODER_FP32SPLIT").ok().as_deref() == Some("1") => {
                let n = qcfg.n_layers as usize;
                // fp32 2-card split (opt-in; needs a large-binding / ReBAR card — it
                // does NOT fit two P40s). Cut point: layers 0..cut (+ embedding) on
                // the bulk card, cut..n-1 on the DiT's card. The fp32 encoder is
                // ~16 GB and each card's usable budget (weights × ~2 alloc overhead
                // on non-ReBAR Pascal) is < 24 GB, so the bulk must NOT exceed ~11 GB
                // (≈ embed + ⅔ of the layers). `BRAIN_S3DIT_ENCODER_CUT` overrides.
                let cut = std::env::var("BRAIN_S3DIT_ENCODER_CUT").ok().and_then(|s| s.parse().ok()).unwrap_or((n * 2) / 3).min(n - 1);
                progress(&format!("building Qwen-4B encoder bulk (fp32 split @{cut}: GPU {bulk} + GPU {dit_gpu})"));
                // Bulk shard: embedding + layers 0..cut on the (empty) bulk card.
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                let s0 = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qsrc, false, Shard { start: 0, end: cut, embed: true, head: false, gpu_index: bulk as usize });
                split = Some((s0, cut, n, dit_gpu));
                None // tail (and thus Encoder::Split) assembled after the DiT below
            }
            (Some(bulk), false) if encoder_on_demand(bulk, dit_gpu, resident_override()) => {
                // Same card as the DiT: the int8 encoder (~9.5 GB) and the
                // int8 DiT (~13 GB) together would exceed most single cards'
                // usable budget (and comfortably exceed a unified-memory
                // box's), so neither is built here -- `Encoder::OnDemand`
                // builds, encodes, and drops the encoder inside every
                // `generate()` call instead, so it and the DiT are never
                // resident at once. Opt back into the
                // old both-resident behaviour with
                // `BRAIN_S3DIT_ENCODER_RESIDENT=1` on a card with room for
                // both (a real 24 GB+ discrete GPU).
                progress(&format!("using on-demand Qwen-4B encoder (int8 DP4A, GPU {bulk} shared with the DiT)"));
                Some(Encoder::OnDemand { qreader_path: paths.qwen.clone(), qcfg: qcfg.clone(), cap_len, gpu: bulk })
            }
            (Some(bulk), false) => {
                // A genuinely separate card from the DiT's (or
                // BRAIN_S3DIT_ENCODER_RESIDENT=1 opting back in): whole
                // Qwen-4B in int8 (DP4A), built once, resident for the
                // pipeline's whole life. ~9.5 GB resident. `end: n-1` skips
                // the unused last layer (encode reads the penultimate hidden).
                let n = qcfg.n_layers as usize;
                progress(&format!("building Qwen-4B encoder (int8 DP4A, GPU {bulk})"));
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                let e = Qwen::new_shard_i8(qcfg.clone(), 1, cap_len, &qsrc, Shard { start: 0, end: n - 1, embed: true, head: false, gpu_index: bulk as usize });
                Some(Encoder::Gpu8(Box::new(e)))
            }
            _ => {
                progress("building Qwen-4B encoder (CPU/AVX2)");
                gpu_core::set_default_backend(gpu_core::Backend::Cpu);
                let e = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qsrc, false, Shard::whole(qcfg.n_layers as usize));
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                Some(Encoder::Cpu(Box::new(e)))
            }
        };
        gpu_core::set_default_backend(gpu_core::Backend::Wgpu);

        let dit = build_dit_engine(paths, hifi, adapter, dit_gpu, lh, lw, cap_len, &mut progress)?;

        // Now the DiT is resident and its staging is reclaimed — pack the thin
        // encoder tail on top of it (GPU `dit_gpu`), then assemble Encoder::Split.
        let enc = match (enc_cpu, split) {
            (Some(e), _) => e,
            (None, Some((s0, cut, n, di))) => {
                progress(&format!("building Qwen-4B encoder tail (layers {cut}..{})", n - 1));
                let s1 = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qsrc, false, Shard { start: cut, end: n - 1, embed: false, head: false, gpu_index: di as usize });
                Encoder::Split { s0: Box::new(s0), s1: Box::new(s1), cap_len }
            }
            (None, None) => unreachable!("encoder neither CPU nor split"),
        };
        drop(qsrc);
        drop(qreader);

        // VAE placement: with the int8 GPU encoder, the encoder card is idle during
        // decode (encode is step 1, decode is the last step — never concurrent), so
        // put the VAE there. That frees the DiT card of the VAE's multi-GB decode
        // activations, raising the max image size (GPU 0 = DiT only). The latent
        // crosses DiT→VAE through host memory already, so no cross-device GPU copy.
        // CPU/fp32-split encoders keep the VAE on the DiT card (unchanged) - UNLESS
        // that card is the 2-GPU fp32 `Shard` engine's: measured on a real 24 GB
        // P40 (`nvidia-smi` during the real Z-Image-Turbo checkpoint's `Shard`
        // build) each card lands within half a GB of its 24 GB ceiling from the
        // DiT shard's weights ALONE - the default wgpu backend's ~2.00x real-VRAM-
        // per-uploaded-BYTE cost on this non-ReBAR card (a property of wgpu's
        // Vulkan HAL, not the hardware; the fix is a different device backend,
        // which this code path does not take) applied to "half the ~33 GB fp32
        // checkpoint" already consumes essentially the whole 24 GB, independent of
        // how the `cut` between the two cards is chosen (shifting blocks from one
        // to the other does not create headroom - the SUM is already at the
        // ceiling). There is no room left on EITHER card for the VAE's own weight
        // upload (same per-byte cost), so it decodes on the CPU instead, exactly
        // the reasoning `hifi`'s Qwen-4B encoder already applies to itself above,
        // extended to the other GPU-resident piece of this same pipeline.
        let vae_on_cpu = matches!(dit, DitEngine::Shard(_));
        let vae_card = match &enc {
            Encoder::Gpu8(_) if !vae_on_cpu => enc_gpu.unwrap_or(dit_gpu),
            _ => dit_gpu,
        };
        progress(&format!("building VAE decoder ({})", if vae_on_cpu { "CPU".to_string() } else { format!("GPU {vae_card}") }));
        // `VaeDecoder::from_diffusers`'s `Some("cpu")`/`Some("gpu")` already pick
        // the device explicitly regardless of the ambient default backend, so no
        // `set_default_backend` toggle is needed here (unlike the encoder above,
        // whose `Qwen::new_shard` has no such explicit-device parameter).
        let vtensors = tensors_map(read_component_tensors(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
        let vae = if vae_on_cpu {
            VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, lh, lw, Some("cpu"))
        } else {
            gpu_core::devices::with_gpu(vae_card, || {
                VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, lh, lw, Some("gpu"))
            })?
        };

        Ok(HotPipeline { tok, enc, dit, vae, cap_len, lh, lw, width, height, hifi })
    }

    /// The plain-int8 (no adapter, no hifi) tokenizer+encoder+VAE assembly,
    /// shared by [`Self::build_adapted_with_cache`] and
    /// [`Self::build_from_dit_cache`] — the two demote/promote entry
    /// points — so neither re-derives the (already load-bearing, tested
    /// via `build_adapted`) encoder-selection logic independently.
    /// `build_dit` receives `(dit_gpu, lh, lw)` and must build+place the
    /// DiT on `dit_gpu` itself; this function does not touch it beyond
    /// that. Left OUT of `build_adapted` itself deliberately: that
    /// function's adapter/hifi generality needs its own, unmodified path,
    /// never sharing code with a promote-only feature.
    fn assemble_int8_pipeline(paths: &Paths, width: u32, height: u32, cap_len: u32, build_dit: impl FnOnce(u32, u32, u32) -> Result<DitEngine, String>, mut progress: impl FnMut(&str)) -> Result<HotPipeline, String> {
        if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
            return Err("width/height must be multiples of 16".into());
        }
        let (lh, lw) = (height / 8, width / 8);

        progress("loading tokenizer");
        let tok = QwenBpe::from_file(&paths.tokenizer)?;

        let qcfg = QwenConfig::qwen3_4b();
        let qreader = open_component(&paths.qwen).map_err(|e| format!("open qwen: {e}"))?;
        let qsrc = qwen3::import::hf_source(&qreader, &qcfg)?;
        let enc_gpu_explicit: Option<u32> = std::env::var("BRAIN_S3DIT_ENCODER_GPU")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>().map_err(|_| format!("bad BRAIN_S3DIT_ENCODER_GPU {s:?}")))
            .transpose()?;
        let dit_gpu: u32 = gpu_core::devices::current_gpu().unwrap_or(0);
        let enc_gpu = default_bulk_gpu(enc_gpu_explicit, dit_gpu, gpu_core::devices::schedulable_gpu_count());

        // Same three arms as build_adapted's !hifi cases (no fp32-split arm
        // -- that one only ever applies when hifi, which these entry points
        // never are).
        let enc = match enc_gpu {
            Some(bulk) if encoder_on_demand(bulk, dit_gpu, resident_override()) => {
                progress(&format!("using on-demand Qwen-4B encoder (int8 DP4A, GPU {bulk} shared with the DiT)"));
                Encoder::OnDemand { qreader_path: paths.qwen.clone(), qcfg: qcfg.clone(), cap_len, gpu: bulk }
            }
            Some(bulk) => {
                let n = qcfg.n_layers as usize;
                progress(&format!("building Qwen-4B encoder (int8 DP4A, GPU {bulk})"));
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                let e = Qwen::new_shard_i8(qcfg.clone(), 1, cap_len, &qsrc, Shard { start: 0, end: n - 1, embed: true, head: false, gpu_index: bulk as usize });
                Encoder::Gpu8(Box::new(e))
            }
            None => {
                progress("building Qwen-4B encoder (CPU/AVX2)");
                gpu_core::set_default_backend(gpu_core::Backend::Cpu);
                let e = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qsrc, false, Shard::whole(qcfg.n_layers as usize));
                gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
                Encoder::Cpu(Box::new(e))
            }
        };
        drop(qsrc);
        drop(qreader);
        gpu_core::set_default_backend(gpu_core::Backend::Wgpu);

        let dit = build_dit(dit_gpu, lh, lw)?;

        let vae_card = match &enc {
            Encoder::Gpu8(_) => enc_gpu.unwrap_or(dit_gpu),
            _ => dit_gpu,
        };
        progress(&format!("building VAE decoder (GPU {vae_card})"));
        gpu_core::set_default_backend(gpu_core::Backend::Wgpu);
        let vtensors = tensors_map(read_component_tensors(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
        let vae = gpu_core::devices::with_gpu(vae_card, || VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, lh, lw, Some("gpu")))?;

        Ok(HotPipeline { tok, enc, dit, vae, cap_len, lh, lw, width, height, hifi: false })
    }

    /// Like [`Self::build`] (plain int8, no adapter), but ALSO returns a
    /// [`crate::DitI8Cache`] snapshot of the DiT's already-quantized host
    /// weights — the demote-preparing build. Costs real, permanent host
    /// RAM for as long as the cache lives (see
    /// `ZImageDitI8::build_from_source_with_cache`'s doc); an explicit
    /// choice by the caller (`crates/cli/src/resident.rs`'s
    /// `ZImageInstance`), never the default [`Self::build`] path.
    pub fn build_adapted_with_cache(paths: &Paths, width: u32, height: u32, cap_len: u32, progress: impl FnMut(&str)) -> Result<(HotPipeline, crate::DitI8Cache), String> {
        let paths = paths.clone();
        let cache_slot: std::rc::Rc<std::cell::RefCell<Option<crate::DitI8Cache>>> = Default::default();
        let slot = cache_slot.clone();
        let dit_paths = paths.clone();
        let pipe = Self::assemble_int8_pipeline(
            &paths,
            width,
            height,
            cap_len,
            move |dit_gpu, lh, lw| {
                let zcfg = dit_config(&dit_paths.dit);
                let dreader = open_component(&dit_paths.dit).map_err(|e| format!("open dit: {e}"))?;
                let dsrc = crate::import::comfy_source(&dreader, &zcfg);
                let (dit, cache) = gpu_core::devices::with_gpu(dit_gpu, || crate::ZImageDitI8::build_from_source_with_cache(zcfg, &dsrc, 1, lh, lw, cap_len))?;
                *slot.borrow_mut() = Some(cache);
                Ok(DitEngine::I8(Box::new(dit)))
            },
            progress,
        )?;
        let cache = cache_slot.borrow_mut().take().expect("assemble_int8_pipeline only returns Ok after build_dit ran and set this");
        Ok((pipe, cache))
    }

    /// [`Self::build_adapted_with_cache`]'s promote sibling: rebuild the
    /// int8 pipeline WITHOUT touching the DiT checkpoint at all, using
    /// `cache` instead. `crates/cli/src/resident.rs`'s
    /// `ZImageInstance::promote` is the caller.
    pub fn build_from_dit_cache(paths: &Paths, width: u32, height: u32, cap_len: u32, cache: &crate::DitI8Cache, progress: impl FnMut(&str)) -> Result<HotPipeline, String> {
        Self::assemble_int8_pipeline(
            paths,
            width,
            height,
            cap_len,
            |dit_gpu, lh, lw| Ok(DitEngine::I8(Box::new(gpu_core::devices::with_gpu(dit_gpu, || crate::ZImageDitI8::rebuild_from_cache(cache, 1, lh, lw, cap_len))?))),
            progress,
        )
    }

    /// Tokenize `prompt`, pad/truncate to `cap_len`, and run encode → DiT sampling
    /// → VAE decode — all on the resident models. Fast (no weight loads).
    /// `cancel` is polled between sampling steps: a cancelled token aborts with
    /// `Err("cancelled")` (pass an unarmed `Default` token to run uninterrupted).
    pub fn generate(&self, prompt: &str, seed: u64, steps: u32, cancel: &capability::CancelToken, mut progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
        let steps = steps.max(1);
        let total = steps + 2;

        // 1. tokenize + pad/truncate to the built cap_len.
        progress(1, total, "encoding prompt (Qwen-4B, CPU)");
        let templated = self.tok.apply_chat_template(&[("user", prompt)], true);
        let mut tokens = self.tok.encode(&templated);
        let cl = self.cap_len as usize;
        if tokens.len() > cl {
            // Loud, not silent: the image is conditioned on a PREFIX of the
            // user's prompt (audit F18).
            let msg = format!("warning: prompt is {} tokens but this resident was built for cap_len {} -- conditioning on the first {} tokens only", tokens.len(), cl, cl);
            eprintln!("zimage: {msg}");
            progress(1, total, &msg);
            tokens.truncate(cl);
        }
        let content = tokens.len().min(cl);
        // Masked-pad, like flux2's encoder path: a dedicated PAD token,
        // excluded as an attention KEY past the true content length (HF
        // attention_mask semantics). The old scheme repeated the LAST token
        // with no mask, so the caption features -- all cap_len rows of which
        // the S3-DiT attends unmasked -- depended on how many copies of the
        // final token the encoder saw (the unsoundness class the LFM ledger
        // documents; audit F17).
        tokens.resize(cl, PAD_TOKEN);
        let cap = self.enc.encode(&tokens, content)?; // [cap_len · 2560]

        // 2. seeded latent + scheduler.
        let n = (16 * self.lh * self.lw) as usize;
        let mut lat = randn(n, seed);
        let seq_len = ((self.lh / 2) * (self.lw / 2)) as usize;
        let sigmas = dynamic_shift(&default_z_image_sigmas(steps as usize), calc_mu(seq_len));
        let mut sched = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1000, shift: 1.0, invert_sigmas: false });
        sched.set_timesteps(&sigmas);
        let ts = sched.timesteps().to_vec();
        let sig_full = sched.sigmas().to_vec();

        // 3. flow-match sampling on the resident DiT.
        for i in 0..steps as usize {
            if cancel.is_cancelled() {
                return Err("cancelled".into());
            }
            progress(2 + i as u32, total, if self.hifi { "sampling (fp32, 2×GPU)" } else { "sampling" });
            let t_dit = (1000.0 - ts[i]) / 1000.0;
            let v: Vec<f32> = self.dit.forward(&lat, &cap, t_dit).iter().map(|&x| -x).collect();
            let dt = sig_full[i + 1] - sig_full[i];
            for (x, &vv) in lat.iter_mut().zip(&v) {
                *x += dt * vv;
            }
        }

        // 4. VAE decode + postprocess.
        progress(total, total, "decoding (VAE)");
        let dec_in: Vec<f32> = lat.iter().map(|&x| x / VAE_SCALE + VAE_SHIFT).collect();
        let chw = self.vae.decode(&dec_in);
        let (h, w) = (self.height as usize, self.width as usize);
        let mut hwc = vec![0f32; h * w * 3];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    hwc[(y * w + x) * 3 + c] = (chw[(c * h + y) * w + x] * 0.5 + 0.5).clamp(0.0, 1.0);
                }
            }
        }
        Ok(Image { hwc, w, h })
    }
}

/// A generated image: interleaved-RGB HWC in `[0,1]`.
pub struct Image {
    pub hwc: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

/// An input image (+ optional mask) to condition generation on — the shared
/// substrate of image2image, inpaint and outpaint. The image is VAE-encoded to a
/// latent, partially re-noised (per `strength`), and denoised the rest of the way.
pub struct Init<'a> {
    /// Source image, HWC interleaved RGB in `[0,1]`, size `opts.width×opts.height`.
    pub image: &'a [f32],
    /// `0` = keep the input unchanged, `1` = ignore it (full re-generation). Sets
    /// how far back into the noise schedule sampling starts.
    pub strength: f32,
    /// Optional inpaint mask, `height·width` single-channel in `[0,1]` (`1` =
    /// regenerate, `0` = keep). When present, kept regions are re-anchored to the
    /// (noised) input at every step so only the masked area changes.
    pub mask: Option<&'a [f32]>,
    /// Feather radius in **latent cells** (VAE 8× downscale). `0` = a hard mask
    /// edge; larger blurs the mask boundary so the regenerated region blends
    /// smoothly into the kept pixels instead of showing a seam.
    pub feather: u32,
}

/// Deterministic standard-normal samples via xorshift64* + Box–Muller — a fixed
/// seed always yields the same latent (so generation is reproducible).
fn randn(n: usize, seed: u64) -> Vec<f32> {
    model::hostmath::randn(n, seed)
}

/// diffusers `calculate_shift` for Z-Image (base_seq 256, max 4096, shifts 0.5..1.15).
fn calc_mu(seq_len: usize) -> f32 {
    let m = (1.15 - 0.5) / (4096.0 - 256.0);
    0.5 + m * (seq_len as f32 - 256.0)
}

/// diffusers exponential time-shift: `σ' = e^mu / (e^mu + 1/σ − 1)`.
fn dynamic_shift(sigmas: &[f32], mu: f32) -> Vec<f32> {
    let e = mu.exp();
    sigmas.iter().map(|&s| e / (e + 1.0 / s - 1.0)).collect()
}

pub(crate) fn tensors_map(v: Vec<checkpoint::safetensors::StTensor>) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    v.into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

/// The FLUX-style 16-channel VAE config Z-Image ships (weights/vae/config.json).
pub(crate) fn zimage_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 16,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        norm_num_groups: 32,
        norm_eps: 1e-6,
        mid_block_add_attention: true,
        scaling_factor: 0.3611,
        shift_factor: 0.1159,
        use_quant_conv: false,
        use_post_quant_conv: false,
        patch_size: [1, 1],
        batch_norm_eps: 1e-4,
    }
}

/// VAE scale/shift → DiT latent space. Decode is `z/scale + shift`; encode is the
/// inverse `(z − shift)·scale`. (Z-Image VAE: scale 0.3611, shift 0.1159.)
const VAE_SCALE: f32 = 0.3611;
const VAE_SHIFT: f32 = 0.1159;

/// Area-average-pool a full-res mask `[h·w]` down to latent resolution `[lh·lw]`
/// (VAE 8× downscale), keeping soft values in `[0,1]`.
fn downsample_mask(mask: &[f32], w: usize, h: usize, lw: usize, lh: usize) -> Vec<f32> {
    let (sx, sy) = (w / lw, h / lh);
    let mut out = vec![0f32; lw * lh];
    for ly in 0..lh {
        for lx in 0..lw {
            let mut s = 0.0;
            for yy in 0..sy {
                for xx in 0..sx {
                    s += mask[(ly * sy + yy) * w + (lx * sx + xx)];
                }
            }
            out[ly * lw + lx] = s / (sx * sy) as f32;
        }
    }
    out
}

/// Separable box blur of a latent-resolution mask `[lh·lw]`, `radius` cells each
/// side, clamped at the borders — feathers a hard mask into a smooth ramp so the
/// inpaint/outpaint boundary blends. `radius = 0` returns the mask unchanged.
fn feather_mask(mask: &[f32], lw: usize, lh: usize, radius: usize) -> Vec<f32> {
    if radius == 0 {
        return mask.to_vec();
    }
    let win = (2 * radius + 1) as f32;
    // horizontal
    let mut h = vec![0f32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            let mut s = 0.0;
            for d in 0..=2 * radius {
                let xx = (x + d).saturating_sub(radius).min(lw - 1);
                s += mask[y * lw + xx];
            }
            h[y * lw + x] = s / win;
        }
    }
    // vertical
    let mut out = vec![0f32; lw * lh];
    for y in 0..lh {
        for x in 0..lw {
            let mut s = 0.0;
            for d in 0..=2 * radius {
                let yy = (y + d).saturating_sub(radius).min(lh - 1);
                s += h[yy * lw + x];
            }
            out[y * lw + x] = s / win;
        }
    }
    out
}

/// Generate an image from `prompt` (text-to-image). `progress(step, total, msg)`
/// streams updates.
pub fn generate(prompt: &str, opts: &Opts, paths: &Paths, progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
    generate_core(prompt, opts, paths, None, progress)
}

/// Image-to-image / inpaint / outpaint: regenerate `init.image` toward `prompt`.
/// A mask (`init.mask`) restricts changes to the masked region (inpaint/outpaint).
pub fn generate_img(prompt: &str, opts: &Opts, paths: &Paths, init: Init, progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
    generate_core(prompt, opts, paths, Some(init), progress)
}

/// Shared pipeline for all four actions. With `init = None` it starts from pure
/// noise (text2image); with an init image it VAE-encodes it, re-noises to the
/// `strength`-determined step, and (for inpaint) re-anchors the kept region each
/// step.
fn generate_core(prompt: &str, opts: &Opts, paths: &Paths, init: Option<Init>, mut progress: impl FnMut(u32, u32, &str)) -> Result<Image, String> {
    let total = opts.steps + 2; // encode + N sampling + decode
    if !opts.width.is_multiple_of(16) || !opts.height.is_multiple_of(16) {
        return Err("width/height must be multiples of 16".into());
    }
    let (lh, lw) = (opts.height / 8, opts.width / 8); // VAE downscale 8

    // 1. tokenize (chat template) --------------------------------------------
    let tok = QwenBpe::from_file(&paths.tokenizer)?;
    let templated = tok.apply_chat_template(&[("user", prompt)], true);
    let tokens = tok.encode(&templated);
    let cap_len = tokens.len() as u32;

    // 2. Qwen-4B encode → caption features (penultimate hidden). Dropped after. -
    //
    // The encoder is a SINGLE forward pass (not the heavy iterative compute), and
    // its ~16.8 GB of f32 weights plus Vulkan's upload staging bump a 24 GB P40's
    // ceiling. So we run just this one-shot on the CPU (AVX2+FMA matmul, a few
    // seconds) and keep the *heavy* work — the 8-step DiT and the VAE — on the GPU
    // (VRAM), which is where the repeated compute belongs. This is the intended
    // "CPU only as a fallback for the piece that doesn't fit" split.
    progress(1, total, "encoding prompt (Qwen-4B, CPU/AVX2)");
    let qcfg = QwenConfig::qwen3_4b();
    // Streaming: peak host allocation for the encoder is one tensor, not the
    // whole ~16 GB model (see `build_adapted`'s identical fix, above).
    let qreader = open_component(&paths.qwen).map_err(|e| format!("open qwen: {e}"))?;
    let qsrc = qwen3::import::hf_source(&qreader, &qcfg)?;
    let cap = {
        gpu_core::set_default_backend(gpu_core::Backend::Cpu); // encoder → CPU (AVX2)
        let enc = Qwen::new_shard(qcfg.clone(), 1, cap_len, &qsrc, false, qwen3::Shard::whole(qcfg.n_layers as usize));
        let c = enc.encode(&tokens); // [cap_len · 2560]
        gpu_core::set_default_backend(gpu_core::Backend::Wgpu); // heavy compute → GPU
        c
    };
    drop(qsrc);
    drop(qreader);

    // 3. scheduler (dynamic-shifted sigmas; brain applies shift=1 so we pre-shift).
    // `sig_full` is the N+1 sigmas (N shifted step sigmas + terminal 0).
    let seq_len = ((lh / 2) * (lw / 2)) as usize; // DiT patch 2
    let sigmas = dynamic_shift(&default_z_image_sigmas(opts.steps as usize), calc_mu(seq_len));
    let mut sched = FlowMatchEulerScheduler::new(FlowMatchConfig { num_train_timesteps: 1000, shift: 1.0, invert_sigmas: false });
    sched.set_timesteps(&sigmas);
    let ts = sched.timesteps().to_vec();
    let sig_full = sched.sigmas().to_vec();

    // 4. starting latent -----------------------------------------------------
    // Fixed seeded noise. text2image starts from it directly (σ≈1). An init image
    // is VAE-encoded to a latent `lat0`, then re-noised to the strength-chosen
    // step: `x = (1−σ)·lat0 + σ·noise` (flow-matching forward). A mask + `lat0`
    // are kept for per-step re-anchoring of the un-masked region (inpaint).
    let n = (16 * lh * lw) as usize;
    let noise = randn(n, opts.seed);
    let plane = (lh * lw) as usize;
    let (mut lat, start_step, mask_lat, lat0): LatentInit = match &init {
        None => (noise.clone(), 0, None, None),
        Some(init) => {
            progress(1, total, "encoding image (VAE)");
            let (h, w) = (opts.height as usize, opts.width as usize);
            if init.image.len() != 3 * h * w {
                return Err(format!("init image is {} floats, expected {} (HWC {w}×{h}×3)", init.image.len(), 3 * h * w));
            }
            // HWC [0,1] → CHW [-1,1] (VAE-native).
            let mut chw_in = vec![0f32; 3 * h * w];
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        chw_in[(c * h + y) * w + x] = init.image[(y * w + x) * 3 + c] * 2.0 - 1.0;
                    }
                }
            }
            let vtensors = tensors_map(read_component_tensors(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
            let mean = {
                let enc = VaeEncoder::from_diffusers(zimage_vae_config(), &vtensors, opts.height, opts.width, Some("gpu"));
                enc.encode_mean(&chw_in, lh, lw)
            };
            let lat0: Vec<f32> = mean.iter().map(|&z| (z - VAE_SHIFT) * VAE_SCALE).collect();

            let strength = init.strength.clamp(0.0, 1.0);
            let init_t = ((opts.steps as f32 * strength).round() as usize).min(opts.steps as usize);
            let start = (opts.steps as usize).saturating_sub(init_t);
            let sig = sig_full[start];

            let mask_lat = init.mask.map(|m| {
                let ds = downsample_mask(m, w, h, lw as usize, lh as usize);
                feather_mask(&ds, lw as usize, lh as usize, init.feather as usize)
            });

            // The masked (`mask=1`, "regenerate") region starts from PURE noise,
            // not the strength-blended `(1-sigma)*lat0 + sigma*noise` used for
            // image2image: that blend leaves a `(1-sigma)` sliver of the ORIGINAL
            // latent under the mask - small in magnitude (e.g. 8% at the default
            // strength 0.85) but STRUCTURED rather than random, which a
            // diffusion model latches onto far more readily than noise of the
            // same magnitude, pulling the "regenerated" region back toward the
            // original content instead of the prompt. `mask=1` means "ignore the
            // original here", so its starting point must not carry any of it.
            // The kept (`mask=0`) region still uses the strength blend, for
            // consistency with the per-step re-anchor below (which overwrites
            // every kept cell every iteration regardless).
            let lat_init: Vec<f32> = match &mask_lat {
                Some(mask) => (0..n)
                    .map(|idx| {
                        let m = mask[idx % plane];
                        let kept = (1.0 - sig) * lat0[idx] + sig * noise[idx];
                        m * noise[idx] + (1.0 - m) * kept
                    })
                    .collect(),
                None => lat0.iter().zip(&noise).map(|(&x0, &nz)| (1.0 - sig) * x0 + sig * nz).collect(),
            };
            (lat_init, start, mask_lat, Some(lat0))
        }
    };

    // 5. flow-match sampling over the DiT ------------------------------------
    // int8 on one P40 (default), or full-precision fp32 sharded across both P40s
    // when `hifi` — no quantisation error, at the cost of a second card.
    let zcfg = dit_config(&paths.dit);
    // Streaming: peak host allocation for the DiT is one tensor, not the
    // whole ~24 GB checkpoint (see `build_adapted`'s identical fix, above).
    let dreader = open_component(&paths.dit).map_err(|e| format!("open dit: {e}"))?;
    let dsrc = crate::import::comfy_source(&dreader, &zcfg);
    {
        let dit = DitEngine::build_from_source(opts.hifi, zcfg, &dsrc, &paths.dit, lh, lw, cap_len);
        for i in start_step..opts.steps as usize {
            progress(2 + i as u32, total, if opts.hifi { "sampling (fp32, 2×GPU)" } else { "sampling" });
            let t_dit = (1000.0 - ts[i]) / 1000.0;
            // The reference negates the DiT output before the Euler step
            // (`noise_pred = -noise_pred; scheduler.step(noise_pred, …)`): brain's
            // scheduler is the bare `x + (σ_next−σ)·v`, so we negate here to match.
            let v: Vec<f32> = dit.forward(&lat, &cap, t_dit).iter().map(|&x| -x).collect();
            let dt = sig_full[i + 1] - sig_full[i];
            for (x, &vv) in lat.iter_mut().zip(&v) {
                *x += dt * vv;
            }
            // Inpaint: re-anchor the KEPT region to the input latent noised to the
            // next step's σ, so only the masked region is freely regenerated.
            if let (Some(mask), Some(lat0)) = (&mask_lat, &lat0) {
                let snext = sig_full[i + 1];
                for c in 0..16 {
                    for (p, &mp) in mask.iter().enumerate() {
                        let idx = c * plane + p;
                        let keep = 1.0 - mp;
                        let orig = (1.0 - snext) * lat0[idx] + snext * noise[idx];
                        lat[idx] = mp * lat[idx] + keep * orig;
                    }
                }
            }
        }
    } // dit dropped → free VRAM before the VAE

    // 6. VAE decode ----------------------------------------------------------
    //
    // On the GPU (VRAM): the decoder graph is built over the *latent* dims
    // (`lh × lw`); it upsamples ×8 internally to the `height × wpx` image. Passing
    // the latent dims keeps every buffer small (well under the P40's 2 GiB binding
    // limit), so this runs on-device alongside the DiT.
    progress(total, total, "decoding (VAE)");
    let vtensors = tensors_map(read_component_tensors(&paths.vae).map_err(|e| format!("read vae: {e}"))?);
    let vae = VaeDecoder::from_diffusers(zimage_vae_config(), &vtensors, lh, lw, Some("gpu"));
    let dec_in: Vec<f32> = lat.iter().map(|&x| x / VAE_SCALE + VAE_SHIFT).collect();
    let chw = vae.decode(&dec_in); // [3 · H · W] in [-1, 1]

    // 7. postprocess: [-1,1] → [0,1], CHW → HWC ------------------------------
    let (h, w) = (opts.height as usize, opts.width as usize);
    let mut hwc = vec![0f32; h * w * 3];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                hwc[(y * w + x) * 3 + c] = (chw[(c * h + y) * w + x] * 0.5 + 0.5).clamp(0.0, 1.0);
            }
        }
    }
    Ok(Image { hwc, w, h })
}

#[cfg(test)]
mod component_tensor_tests {
    use super::*;

    #[test]
    fn reads_a_single_file_and_an_hf_style_directory_the_same_way() {
        let dir = std::env::temp_dir().join(format!("zimage-read-component-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Single-file path (unchanged behavior).
        let file = dir.join("dit.safetensors");
        checkpoint::st::save_safetensors(file.to_str().unwrap(), &[("w".to_string(), vec![2], vec![1.0, 2.0])], &serde_json::Value::Null, None).unwrap();
        let from_file = read_component_tensors(file.to_str().unwrap()).unwrap();
        assert_eq!(from_file.len(), 1);
        assert_eq!(from_file[0].name, "w");

        // Directory path: an HF-style component dir holding one `model.safetensors`
        // (no index) -- the shape a fetched Z-Image `vae/` role uses.
        let subdir = dir.join("component");
        std::fs::create_dir_all(&subdir).unwrap();
        checkpoint::st::save_safetensors(
            subdir.join("model.safetensors").to_str().unwrap(),
            &[("w".to_string(), vec![2], vec![3.0, 4.0])],
            &serde_json::Value::Null,
            None,
        )
        .unwrap();
        let from_dir = read_component_tensors(subdir.to_str().unwrap()).unwrap();
        assert_eq!(from_dir.len(), 1);
        assert_eq!(from_dir[0].data, vec![3.0, 4.0]);

        // The streaming sibling must read the identical bytes from both shapes.
        use checkpoint::TensorSource;
        let streamed_file = open_component(file.to_str().unwrap()).unwrap();
        let mut got = None;
        assert!(streamed_file.with_tensor("w", &mut |d| got = Some(d.to_vec())));
        assert_eq!(got.unwrap(), vec![1.0, 2.0]);

        let streamed_dir = open_component(subdir.to_str().unwrap()).unwrap();
        let mut got = None;
        assert!(streamed_dir.with_tensor("w", &mut |d| got = Some(d.to_vec())));
        assert_eq!(got.unwrap(), vec![3.0, 4.0]);

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod encoder_scheduling_tests {
    use super::*;

    /// The scheduling decision the OOM fix hinges on: on-demand exactly when
    /// the encoder shares the DiT's own card, never when it has a card of
    /// its own, and never when the caller explicitly opted back into the
    /// old both-resident behaviour.
    #[test]
    fn on_demand_exactly_when_sharing_the_dits_card() {
        assert!(encoder_on_demand(0, 0, false), "same card as the DiT -> on-demand");
        assert!(!encoder_on_demand(1, 0, false), "a separate card -> resident (no conflict to avoid)");
        assert!(!encoder_on_demand(0, 0, true), "BRAIN_S3DIT_ENCODER_RESIDENT=1 -> resident even on the same card");
        assert!(!encoder_on_demand(1, 0, true), "override + separate card -> still resident");
    }

    /// With no explicit `BRAIN_S3DIT_ENCODER_GPU`, the encoder must not
    /// default to a CPU-resident fp32 build (~16 GB, never dropped) on a box
    /// with exactly one GPU: there is no "separate bulk card" for CPU to be
    /// conserving VRAM against, so CPU-by-default is strictly worse than
    /// sharing the DiT's own card (smaller int8 footprint, on-demand, and
    /// dropped before the DiT builds) -- this is the "automatic regardless
    /// of machine shape" requirement, not a magic env var the caller must
    /// know to set. A box with more than one GPU keeps today's behaviour
    /// (CPU by default) unchanged: there a real bulk-card choice exists and
    /// picking one automatically would be guessing.
    #[test]
    fn default_bulk_gpu_shares_the_dits_card_when_theres_only_one_gpu() {
        assert_eq!(default_bulk_gpu(None, 0, 1), Some(0), "one GPU total -> share it with the DiT");
        assert_eq!(default_bulk_gpu(None, 2, 1), Some(2), "one GPU total -> that GPU is dit_gpu, share it");
        assert_eq!(default_bulk_gpu(None, 0, 2), None, "two GPUs -> ambiguous, keep the CPU default");
        assert_eq!(default_bulk_gpu(None, 0, 0), None, "no GPU probed -> nothing to default to");
        assert_eq!(default_bulk_gpu(Some(1), 0, 1), Some(1), "an explicit choice is never overridden");
    }

    /// The number the residency layer budgets against must be the number
    /// `ZImageDitWindowed` actually allocates: exact arithmetic over
    /// `ZImageConfig::turbo()`'s real shape (dim 3840, hidden 10240,
    /// head_dim 128), not a rough guess. `window=2` -> 2 main blocks + the
    /// 4 fully-resident refiner blocks (2 noise + 2 context) = 6 blocks'
    /// worth of weights + a flat 256 MiB overhead allowance.
    #[test]
    fn windowed_dit_vram_bytes_matches_the_real_turbo_shape_exactly() {
        let cfg = ZImageConfig::turbo();
        let per_block_bytes: u64 = (4 * 3840 * 3840 + 2 * 128 + 3 * 10240 * 3840 + 4 * 3840) * 4;
        let want = per_block_bytes * 6 + (256u64 << 20);
        assert_eq!(windowed_dit_vram_bytes(&cfg, 2), want);
        assert_eq!(want, 4_515_543_040, "sanity-check the hand-derived constant itself");
    }

    /// `window >= n_layers` must cost exactly what full residency costs:
    /// every block (refiners + all 30 main layers) resident, no smaller and
    /// no larger a number than `ZImageDit`'s own equivalent build allocates.
    #[test]
    fn windowed_dit_vram_bytes_with_a_full_window_covers_every_block() {
        let cfg = ZImageConfig::turbo();
        let per_block_bytes: u64 = (4 * 3840 * 3840 + 2 * 128 + 3 * 10240 * 3840 + 4 * 3840) * 4;
        let want = per_block_bytes * (30 + 4) + (256u64 << 20);
        assert_eq!(windowed_dit_vram_bytes(&cfg, 30), want);
        // An oversized window clamps to n_layers, not the requested value.
        assert_eq!(windowed_dit_vram_bytes(&cfg, 999), want);
    }

    /// `hifi_cost_bytes` picks the windowed estimate on a one-GPU box and
    /// the unchanged 2-GPU-shard estimate otherwise; the encoder's RAM
    /// figure is the same either way (the hifi path's encoder is
    /// CPU-resident regardless of GPU count today).
    #[test]
    fn hifi_cost_bytes_switches_on_gpu_count_and_keeps_the_shard_estimate_unchanged() {
        let (vram_1gpu, ram_1gpu) = hifi_cost_bytes(1);
        let (vram_2gpu, ram_2gpu) = hifi_cost_bytes(2);
        assert_eq!(vram_1gpu, windowed_dit_vram_bytes(&ZImageConfig::turbo(), window_blocks_from_env()));
        assert_eq!(vram_2gpu, 24 << 30);
        assert_eq!(ram_1gpu, ram_2gpu);
        assert!(vram_1gpu < vram_2gpu, "the windowed estimate must be dramatically smaller, or the whole fix is pointless");
    }

    /// A real, non-zero, roughly-5-GB-at-Turbo's-shape number — not a
    /// placeholder. `estimate_at(Warm)` reporting `0` for a cache that
    /// actually holds several GB would defeat the entire point of the
    /// figure (an OOM the budgeting layer thinks can't happen).
    #[test]
    fn int8_cache_bytes_estimate_is_a_real_multi_gigabyte_number_at_turbo_shape() {
        let bytes = int8_cache_bytes_estimate(&ZImageConfig::turbo());
        assert!(bytes > 4 << 30, "expected multiple GB, got {} bytes", bytes);
        assert!(bytes < 8 << 30, "expected the DOMINANT term only, not double-counted, got {} bytes", bytes);
    }
}

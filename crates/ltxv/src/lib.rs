// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LTX-2.5: a two-stream (video + audio) diffusion transformer.
//!
//! This crate implements the causal 3D video VAE (encoder + conv decoder,
//! see [`vae3d`]'s module doc), the **video-ONLY stream of the DiT**
//! ([`config`]/[`rope`]/[`block`]/[`dit`]): `BasicAVTransformerBlock`'s video
//! path (self-attention, text cross-attention with AdaLN modulation, the
//! FFN) and the full tiny-model forward around it (`patchify_proj`, the
//! per-token adaLN-single timestep conditioning, the output stage), and the
//! **end-to-end `t2v` pipeline + serving contract**
//! ([`pipeline`]/[`caps`]): real scheduler math, a real CFG
//! denoise loop, the real VAE decode, and the capability/CLI/residency
//! plumbing every model in this repo ships - composed as a SMOKE TEST (real
//! VAE + tiny random-weight DiT + a stub text context, no real 22B DiT or
//! Gemma-4 text encoder yet, see [`pipeline`]'s module doc for exactly what
//! that means and why). The real text encoder / embeddings-connector are NOT
//! implemented - tracked gaps on this port's own roadmap ledger. (The
//! convolution-free "diffusion decoder", `DiffusionVideoDecoder`/3D
//! neighborhood attention, IS implemented as its own module - see
//! [`na_decoder`] below - but the pipeline does not yet wire it in as an
//! alternative to the conv decoder [`vae3d`] uses.)
//!
//! Only the tiny-config op sequence is proven for the DiT itself
//! (`config::LtxDitConfig::tiny`, `crates/ltxv/tests/dit_parity.rs`) -
//! real-22B-checkpoint weight import is a separate, tracked gap
//! (`dit::load_tiny_weights` is a simple name-keyed loader for the golden's
//! OWN tiny weights only, not a real-checkpoint importer;
//! `dit::random_tiny_weights` is what [`pipeline::generate`] actually uses).
//!
//! This crate also implements the 2D causal-conv **audio VAE** ([`audio_vae`])
//! and the BigVGAN/snakebeta **base vocoder** ([`vocoder`], no bandwidth
//! extension), both real weights, real parity.
//!
//! The **audio DiT stream + bidirectional audio<->video
//! cross-attention** (`config::LtxAvDitConfig`/`block::LtxAvBlock`/
//! `dit::LtxAvDit`, `LTXModelType::AudioVideo`) is likewise implemented - own
//! self-/text-cross-attention/FFN/adaLN conditioning for the audio stream
//! (structurally identical to the video-only path, narrower dims), coupled
//! every block by the A2V/V2A cross-attention (own per-block `[5,dim]` adaLN
//! tables, a cross-modality-sigma-driven gate, a shared time-only cross-modal
//! RoPE space - see `block`'s and `rope`'s module docs). Tiny-config
//! op-sequence parity only, same bar and same real-checkpoint-import gap as
//! the video-only stream (`crates/ltxv/tests/dit_parity.rs`).
//!
//! **Training for the video-only DiT** ([`grad`]/[`modelgrad`]/
//! [`lora`]/[`finetune`]) is implemented as a from-scratch, float-type-generic
//! host reimplementation of [`LtxDit::forward`]'s exact op sequence (forward +
//! analytic backward, `f64` for the finite-difference gradcheck oracle,
//! `f32` for the trainer), a flow-matching training-target helper, LoRA in
//! the ComfyUI key layout, and a synthetic-data fine-tune loop -
//! `gradcheck::check_ltxv`/`check_ltxv_conditioning`, `crates/ltxv/tests/
//! {block_grad,host_forward_parity,lora_train,overfit}.rs`. The audio
//! stream's own training reference (`LtxAvDitConfig`/`LtxAvBlock`/
//! `LtxAvDit`) is covered separately - see below.
//!
//! This crate also implements the two real-weight **latent upscalers**
//! ([`upsampler`], spatial x2 / temporal x2 - small conv/resblock nets over
//! `vae::blocks3d::Builder3d`, no timestep/conditioning of any kind) and the
//! real-weight **duration-prediction head** ([`duration_head`], eager host
//! math - `Linear` + a 1-query `MultiheadAttention` pooler + a small MLP).
//! Both real parity, same bar as every other small real-weight component
//! this port has landed.
//!
//! The **NA diffusion decoder** ([`na_decoder`]) is also implemented - the
//! convolution-free `DiffusionVideoDecoder` (3D neighborhood-attention
//! blocks + AdaLN-Zero modulation), real weights, real parity on every tap
//! including the full `CombinedDiffusionNABlock` stack. General
//! overlapping-tile chunked decode, the `CHUNKED`/`BLACKWELL_DSL` block
//! variants, and multi-step Euler sampling remain a tracked gap (moot for
//! the real checkpoint, whose own config collapses sampling to one step -
//! see `na_decoder.rs`'s doc).
//!
//! **DFR (Diffusion Fidelity Rendering) geometry + a smoke-level
//! multi-stage pipeline** ([`dfr`]/[`pipeline::generate_dfr`]) round out this
//! crate - the real, weight-free tile-boundary/keyframe-segment-canvas/
//! generated-keyframe-slot-token-append math ([`dfr`], unit-tested, no
//! `#[ignore]` needed), and a pipeline that runs it end to end: half-res
//! base generation with appended keyframe slots, a REAL spatial x2 latent
//! upscale, a full-res detailing pass (re-noised from the upscaled result,
//! no IC-LoRA - one does not exist in this repo), and 0-2 REAL temporal x2
//! upsample rounds with tile-based stitching. Still the tiny random-weight
//! DiT and the stub text context the base pipeline uses - see [`pipeline`]'s
//! module doc for exactly which DFR mechanics are real here and which remain
//! a documented gap (the IC-LoRA spatial-detailing adapter, real 22B
//! quality, per-token/partial-strength anchor-keyframe carry-forward across
//! temporal rounds, and the NA decoder as an alternative decode path).
//!
//! **Training for the audio+video DiT** ([`av_grad`]/[`av_modelgrad`]/
//! [`av_lora`]/[`av_finetune`]) extends the video-only training reference
//! onto [`LtxAvDit`], closing the gap that reference's own doc named. Reuses
//! [`grad::self_attn_and_text_ca_fwd`]/`_bwd` and [`grad::mlp_fwd`]/`_bwd`
//! UNCHANGED for both streams' self-/text-cross-attention/FFN (the
//! video-only path was refactored, behaviour-preserving, to expose these as
//! the two composable phases `crate::block`'s own device path already
//! draws them as); [`av_grad`] adds only what is genuinely new - the
//! audio<->video cross-attention, whose Q/K/V project between
//! DIFFERENT-width streams and whose residual gate is a single row shared
//! by every token (driven by the OTHER modality's scalar sigma), not a
//! per-token gate. `to_gate_logits` (gated attention) and both embeddings
//! connectors stay out of scope, the same scope line
//! (`config::LtxAvDitConfig::tiny`, not `tiny_gated`). LoRA targets 28
//! leaves per block (both streams' attention+FFN, plus all four A<->V
//! cross-attention projections - [`av_lora`]'s own doc explains why those
//! are included, not merely mirrored for symmetry). [`av_finetune`] adds a
//! synthetic procedural dataset with exact ground truth (`data::gen_clips`,
//! already used by `wan`'s own LoRA gates) and proves a concept-only LoRA
//! measurably moves GENERATED OUTPUT toward the concept on a held-out
//! prompt - not merely that loss goes down (lesson #3) - plus a save/reload
//! round trip closed in a genuinely separate OS process (lesson #23).

pub mod audio;
pub mod audio_vae;
pub mod av_finetune;
pub mod av_stream;
pub mod av_grad;
pub mod av_lora;
pub mod av_modelgrad;
pub mod block;
pub mod caps;
pub mod clipmetric;
pub mod config;
pub mod devplan;
pub mod devres;
pub mod dfr;
pub mod dit;
pub mod duration_head;
pub mod finetune;
pub mod grad;
pub mod gguf_src;
pub mod import;
pub mod int8;
pub mod latentdump;
pub mod longform;
pub mod lora;
pub mod modelgrad;
pub mod na_decoder;
pub mod text_cache;
pub mod patchify;
pub mod pipeline;
pub mod rope;
pub mod shard;
pub mod upsampler;
pub mod vae3d;
pub mod vocoder;
pub mod weightcache;

pub use config::{LtxAudioDitConfig, LtxAvDitConfig, LtxDitConfig};
pub use devplan::{DevicePlan, Placement};
pub use dit::{load_tiny_weights, AvDitTaps, DitBatch, DitTaps, LtxAvDit, LtxDit};
pub use pipeline::{GenOpts, Paths};
pub use vae3d::{LtxVaeConfig, LtxVaeDecoder, LtxVaeEncoder};

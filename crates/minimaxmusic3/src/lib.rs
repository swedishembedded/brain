// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax Music 3: lyrics + a structured music-description caption in,
//! a full song (up to 5 minutes, 44.1 kHz stereo) out.
//! (`MiniMaxAI/MiniMax-Music3`, no official upstream inference code - ported
//! from an unmerged `diffusers` PR, commit `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`).
//!
//! Five chained components:
//!
//! ```text
//! lyrics + caption
//!        |
//!    [tokenize: data::qwen_tokenizer::QwenBpe]
//!        v
//! Global LLM (real Qwen3-8B, reused verbatim from `crates/qwen3`)
//!    autoregressive, CFG-guided: one semantic RVQ code per 25 Hz frame
//!        |
//!        +--> RVQ depth decoder (this crate: `depth_decoder`)
//!        |        4-layer causal transformer, autoregressively predicts the
//!        |        7 residual codebooks per frame from the LLM's hidden state
//!        v
//! per-frame hidden states (LLM + 7 depth steps, 8 total)
//!        |
//!    [condition encoder: this crate: `condition_encoder`]
//!        softmax layer-mix -> conv proj -> nearest-resample 25 Hz -> latent rate
//!        v
//! Flow-matching DiT (this crate: `dit`/`block`)
//!    36-layer, partial-RoPE self-attn + gated FFN, denoises Flow-VAE latents
//!    in 200-frame chunks (100-frame hop, 172-latent overlap splice)
//!        v
//! Vocoder (this crate: `vocoder`, DAC-style)
//!    SnakeBeta + weight-normalized conv/conv-transpose -> 44.1 kHz stereo wav
//! ```
//!
//! Reused as-is rather than reimplemented: `crates/qwen3` for the Global LLM
//! (a real Qwen3-8B architecture, `hidden=4096, layers=36, heads=32,
//! kv_heads=8, head_dim=128, vocab=200000` - confirmed against the real
//! checkpoint's own `config.json`, NOT the smaller published Qwen3-8B's
//! `vocab=151936` preset), `data::qwen_tokenizer::QwenBpe` for tokenization,
//! `audio::conv` (forward AND backward) for every conv/conv-transpose in the
//! vocoder, `model::block`'s `Bidir`/`rope2d_partial`/`LayerNorm`/`swiglu`/
//! `kv_expand` Step-builders for the DiT's bidirectional partial-RoPE
//! transformer blocks, and `diffusion::FlowMatchEulerScheduler` (with its new
//! `invert_sigmas` option - the reference scheduler is constructed with
//! `invert_sigmas=True, num_train_timesteps=1, shift=1.0`) for the DiT's
//! sampling loop.
//!
//! Status: condition encoder, vocoder, the RVQ depth decoder, and the DiT's
//! forward are all landed with real-weight parity cosine 1.0. The vocoder,
//! the depth decoder, AND the DiT (`dit_train::Trainer`, device-dispatched
//! through the same `model::block` builders `dit::forward` uses) are fully
//! trainable - full fine-tune (gradcheck on every named parameter, a real
//! overfit demonstration) and LoRA (exact no-op at zero-init, a directional
//! FD check on every adapter, and a training demonstration with the base
//! weights provably frozen) for all three, including the DiT
//! (`dit_lora`, fold-then-run through a fresh `dit_train::Trainer` per
//! step). The vocoder also has a from-scratch STFT-magnitude adversarial
//! discriminator (gradchecked end to end, including through the STFT, and
//! shown to learn). The DiT's int8/shard, the Global LLM's own wiring,
//! joint generator+discriminator training, and serving land
//! component-by-component, one crate module at a time.

pub mod condition_encoder;
pub mod config;
pub mod depth_decoder;
pub mod depth_lora;
pub mod discriminator;
pub mod dit;
pub mod dit_lora;
pub mod dit_train;
pub mod lora;
pub mod train;
pub mod vocoder;

pub use config::{ConditionEncoderConfig, DepthDecoderConfig, DitConfig, VocoderConfig};

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS 12 Hz neural audio codec (Mimi/Moshi-style): decode **and** encode.
//!
//! The ENCODE path ([`Codec::encode`], wav `->` codes `[T,16]`) mirrors the
//! HuggingFace `MimiModel` encoder: SEANet conv encoder -> encoder transformer
//! (LayerNorm + gelu MLP) -> frame-rate-match downsample -> split-RVQ
//! nearest-codebook encode (argmin + residual on the host). It is exact vs the
//! reference `tokenizer.encode` (100% code-match on the golden dump).
//!
//! Pipeline (decode): codes `[T,16]` -> SplitResidualVectorQuantizer dequant
//! (1 semantic + 15 acoustic codebooks, embedding gather + sum) -> pre-conv ->
//! 8-layer sliding-window GQA transformer (RoPE, RMSNorm, LayerScale) ->
//! 2 conv-transpose upsample stages -> SEANet decoder (SnakeBeta residual units
//! + causal transposed convs, upsample rates 8·5·4·3) -> 24 kHz waveform.
//!
//! Reuses `crates/audio` conv builders, `crates/model::block` transformer
//! builders, the shared WGSL engine, and `checkpoint::safetensors` for import.

pub mod config;
pub mod import;
pub mod model;
pub mod recon;

pub use config::CodecConfig;
pub use import::import;
pub use model::Codec;

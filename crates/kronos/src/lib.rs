// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Kronos (NeoQuasar, arXiv 2508.02739, MIT) reimplemented from scratch in
//! brain — a financial K-line/candlestick foundation model. Two stages:
//! 1. a **BSQ tokenizer** autoencoder mapping each OHLCV(+amount) bar to
//!    hierarchical discrete `(s1, s2)` tokens;
//! 2. an **autoregressive decoder** over those tokens with a dual head (predict
//!    `s1`, then `s2` conditioned on the sampled `s1` via a dependency layer).
//!
//! Native representation for the `ForecastModel` adapter = **samples** (AR
//! rollout of future bars). Imported exactly, parity-gated per stage vs the
//! reference (T0 layout → T1 encode → T2 decode → T3 block → T4 dual head → T5
//! generate). Full plan + resolved ambiguities in `docs/models/kronos/status.md`.
//!
//! This milestone: the two configs + their `param_list()`s in the reference's
//! own `state_dict` names, ready for the T0 layout gate.

pub mod config;
pub mod decoder;
pub mod forecaster;
pub mod generate;
pub mod import;
pub mod kvcache;
pub mod nn;
pub mod preprocess;
pub mod finetune;
pub mod tokenizer;
pub mod train;

pub use config::{KronosConfig, KronosTokenizerConfig, Param};
pub use decoder::KronosDecoder;
pub use forecaster::KronosForecaster;
pub use generate::{GenOpts, KronosModel};
pub use preprocess::{denormalize, indices_to_bipolar, normalize, quantized_to_indices, Norm};
pub use tokenizer::KronosTokenizer;

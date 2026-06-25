// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! brain's data layer: tokenizers, dataset generators, on-disk formats, and
//! batching — a self-contained pure-Rust port of nanogpt's data pipeline.
//!
//! Datasets (mirroring nanogpt) are produced by [`prepare`] into a directory:
//! - **char-level** (`shakespeare`, `calculator`, `reverser`, `wordcalc`):
//!   `train.bin`/`val.bin` (`u16`), `meta.json` (vocab), `input.txt`.
//! - **bpe** (`gpt`): `train.bin`/`val.bin` (GPT-2 `u16` ids), `input.txt`.
//! - **timeseries**: `train.f32`/`val.f32` (raw `f32`), `meta.json` (shape).
//!
//! Training reads a split back with [`loader::TokenDataset`] (token tasks) or
//! [`binio::read_f32_bin`] (time series).

pub mod binio;
pub mod bpe;
pub mod loader;
pub mod rng;
pub mod tokenizer;

// Dataset generators (one module per source), ported 1:1 from
// `scratchpad/reference/nanogpt/data_generators/*.py`.
pub mod gen_calculator;
pub mod gen_detect;
pub mod gen_reverser;
pub mod gen_timeseries;
pub mod gen_wordcalc;

pub mod prepare;

pub use prepare::{prepare, Dataset};
pub use tokenizer::{CharTokenizer, Tokenizer};

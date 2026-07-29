// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! NVIDIA Nemotron 3.5 ASR Streaming (0.6B): a FastConformer encoder + RNN-T
//! transducer. Built on the shared WGSL engine; parity-gated against the
//! HuggingFace reference. Kernels → encoder → RNN-T → greedy decode.

pub mod config;
pub mod encoder;
pub mod import;
pub mod kernels;
pub mod reference;

pub use config::NemotronConfig;

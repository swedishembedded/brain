// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! NVIDIA Nemotron 3.5 ASR Streaming (0.6B): a FastConformer encoder + RNN-T
//! transducer. Built on the shared WGSL engine; parity-gated against the
//! HuggingFace reference. Work in progress — kernels land first, then the
//! Conformer encoder, the LSTM prediction network + joint, and greedy decode.

pub mod kernels;

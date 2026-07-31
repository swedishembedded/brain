// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP-family text and image encoders: CLIP-L, OpenCLIP-bigG and EVA-CLIP
//! behind one config-driven graph.
//!
//! * [`config`] — the reference configurations and canonical tensor manifests.
//! * [`import`] — checkpoint import with two-way coverage validation.
//! * [`model`] — the forward graphs (`ClipText`, `EvaVision`).
//!
//! Scope today is **forward parity only**. The backward, the `capability`
//! Provider / residency adapter / D-Bus surface, and the CLIP BPE tokenizer
//! (which belongs in `crates/data` next to the GPT-2 and Qwen BPEs, not here)
//! are follow-up work.

pub mod config;
pub mod import;
pub mod model;
